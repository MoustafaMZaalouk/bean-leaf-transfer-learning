"""Dataset IO, transparent features, leakage audit and train-only model selection."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET = "AI-Lab-Makerere/beans"
DATA_REVISION = "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
MODEL = "microsoft/resnet-18"
MODEL_REVISION = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_SHA256 = {
    "train": "7f905a7323966a58e89b8e839ed656bb869fc82d16a3fadc7dce40972a5f8b19",
    "validation": "33f774593d8b31585457b70c224744e9409ffdee4e91a11822b1ebfe8242928f",
    "test": "534a6b0648f585d69b7ec0ad7a7540720d60c8db8106dc6d0508296316f6cb27",
}
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


class ImageSequence:
    """Keep compressed bytes in RAM; decode only the current image or batch."""

    def __init__(self, encoded):
        self.encoded = encoded

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [Image.open(io.BytesIO(item)).convert("RGB") for item in self.encoded[index]]
        return Image.open(io.BytesIO(self.encoded[index])).convert("RGB")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_data(cache: Path) -> tuple[dict, dict]:
    """Read pinned parquet files, recording digests and preserving official row order."""
    cache.mkdir(parents=True, exist_ok=True)
    result, manifest = {}, {}
    for split, expected in SPLITS.items():
        filename = f"{split}-00000-of-00001.parquet"
        path = cache / filename
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATA_REVISION}/data/{filename}"
        if not path.exists():
            temporary = path.with_suffix(".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        digest = sha256(path.read_bytes())
        if digest != EXPECTED_SHA256[split]:
            raise ValueError(
                f"Dataset SHA-256 mismatch for {split}; remove the corrupt cache file"
            )
        manifest[split] = {"url": url, "sha256": digest}
        rows = pq.read_table(path).to_pylist()
        if len(rows) != expected:
            raise ValueError(f"Unexpected row count for {split}: {len(rows)}")
        images, labels, ids = [], [], []
        for index, row in enumerate(rows):
            images.append(row["image"]["bytes"])
            label = int(row["labels"])
            if label not in range(len(CLASSES)):
                raise ValueError(f"Invalid class: {label}")
            labels.append(label)
            ids.append(f"{split}:{index:04d}")
        result[split] = {"images": ImageSequence(images), "y": np.array(labels), "ids": ids}
    validate_data(result)
    return result, manifest


def validate_data(data: dict) -> None:
    """Validate aligned records, valid labels and globally unique split-qualified IDs."""
    seen = set()
    for split, values in data.items():
        if set(values) != {"images", "y", "ids"}:
            raise ValueError(f"Invalid schema for {split}")
        if not (len(values["images"]) == len(values["y"]) == len(values["ids"])):
            raise ValueError(f"Unaligned records for {split}")
        if not set(np.asarray(values["y"]).tolist()) <= {0, 1, 2}:
            raise ValueError(f"Invalid labels for {split}")
        ids = values["ids"]
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Repeated IDs across or within splits")
        seen.update(ids)


def color_features(images: list[Image.Image]) -> np.ndarray:
    """30 features: RGB means, standard deviations, and eight-bin/channel histograms."""
    features = []
    for image in images:
        pixels = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        flat = pixels.reshape(-1, 3)
        hist = [np.histogram(flat[:, c], bins=8, range=(0, 1))[0] / len(flat) for c in range(3)]
        features.append(np.concatenate([flat.mean(0), flat.std(0), *hist]))
    return np.asarray(features, dtype=np.float64)


def pixel_hash(image: Image.Image) -> str:
    """Decoded RGB hash includes dimensions, so JPEG metadata does not affect equality."""
    rgb = image.convert("RGB")
    return sha256(str(rgb.size).encode() + rgb.tobytes())


def difference_hash(image: Image.Image) -> int:
    gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), "big")


def audit_duplicates(data: dict, threshold: int = 4) -> dict:
    """Find cross-split exact matches and heuristic dHash candidates, never prove independence."""
    records = []
    for split, values in data.items():
        for image, label, row_id in zip(values["images"], values["y"], values["ids"]):
            records.append((split, row_id, int(label), pixel_hash(image), difference_hash(image)))
    exact, near = [], []
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            if a[0] == b[0]:
                continue
            distance = (a[4] ^ b[4]).bit_count()
            pair = {"a": a[1], "b": b[1], "labels": [a[2], b[2]], "hamming": distance}
            if a[3] == b[3]:
                exact.append(pair)
            elif distance <= threshold:
                near.append(pair)
    return {
        "exact_cross_split": exact,
        "near_cross_split_candidates": near,
        "dhash_threshold": threshold,
    }


def pooled_features(model, processor, images, batch_size: int = 8) -> np.ndarray:
    """Eval mode plus inference_mode prevents both gradient and batch-statistic updates."""
    import torch

    model.eval()
    model.requires_grad_(False)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = processor(images=images[start : start + batch_size], return_tensors="pt")
            pooled = model(**batch).pooler_output.flatten(1)
            batches.append(pooled.cpu().numpy())
    return np.concatenate(batches)


def select_classifier(x_train, y_train, x_val, y_val, grid=C_GRID):
    """Fit scaler and classifier on training only; validation selects C by macro F1."""
    candidates = []
    best = None
    for c in grid:
        pipeline = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=3000, random_state=42)
        )
        pipeline.fit(x_train, y_train)
        score = f1_score(y_val, pipeline.predict(x_val), average="macro", zero_division=0)
        candidates.append({"C": c, "validation_macro_f1": float(score)})
        if best is None or score > best[0]:
            best = (score, c, pipeline)
    return best[2], {"selected_C": best[1], "candidates": candidates}


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "correct": int(np.sum(np.asarray(y_true) == np.asarray(y_pred))),
        "n": len(y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=4) + "\n")
