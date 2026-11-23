"""Run from repository root: python scripts/run_experiment.py."""

import argparse
import importlib.metadata
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, ResNetModel

from beanleaf.finetune import fine_tune, predict, gradcam

from beanleaf.core import (
    CLASSES,
    C_GRID,
    DATASET,
    DATA_REVISION,
    MODEL,
    MODEL_REVISION,
    audit_duplicates,
    color_features,
    load_data,
    metrics,
    pooled_features,
    select_classifier,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument(
        "--smoke", action="store_true", help="Six/class/split, one epoch, no headline claims"
    )
    parser.add_argument(
        "--features-only", action="store_true", help="Populate feature cache, then stop"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data, manifest = load_data(args.cache / "data")
    if args.smoke:
        for values in data.values():
            chosen = np.concatenate([np.flatnonzero(values["y"] == c)[:6] for c in range(3)])
            values["images"] = [values["images"][int(i)] for i in chosen]
            values["y"] = values["y"][chosen]
            values["ids"] = [values["ids"][int(i)] for i in chosen]
    print("Dataset loaded", flush=True)
    audit = audit_duplicates(data)
    write_json(args.output / "duplicate_audit.json", audit)
    if audit["exact_cross_split"]:
        raise RuntimeError(
            "Exact cross-split duplicates detected. Review before reporting scores."
        )
    processor = AutoImageProcessor.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = ResNetModel.from_pretrained(MODEL, revision=MODEL_REVISION)
    model.eval()
    model.requires_grad_(False)
    print("Frozen model loaded", flush=True)
    feature_sets = {"color": {}, "resnet18_frozen": {}}
    for split, values in data.items():
        cache_path = args.cache / f"features-{split}-{'smoke' if args.smoke else 'full'}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            feature_sets["color"][split] = cached["color"]
            feature_sets["resnet18_frozen"][split] = cached["resnet"]
        else:
            feature_sets["color"][split] = color_features(values["images"])
            feature_sets["resnet18_frozen"][split] = pooled_features(
                model, processor, values["images"]
            )
            np.savez(
                cache_path,
                color=feature_sets["color"][split],
                resnet=feature_sets["resnet18_frozen"][split],
            )
        print(f"Features extracted: {split}", flush=True)
    if args.features_only:
        print("Feature cache ready", flush=True)
        return
    results = {
        "smoke": args.smoke,
        "dataset": DATASET,
        "dataset_revision": DATA_REVISION,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "seed": 42,
        "C_grid": C_GRID,
        "classes": CLASSES,
        "train_counts": np.bincount(data["train"]["y"], minlength=3).tolist(),
        "validation_counts": np.bincount(data["validation"]["y"], minlength=3).tolist(),
        "test_counts": np.bincount(data["test"]["y"], minlength=3).tolist(),
        "models": {},
    }
    predictions = {"ids": data["test"]["ids"], "y_true": data["test"]["y"].tolist()}
    frozen_classifier = None
    for name, features in feature_sets.items():
        classifier, selection = select_classifier(
            features["train"],
            data["train"]["y"],
            features["validation"],
            data["validation"]["y"],
        )
        if name == "resnet18_frozen":
            frozen_classifier = classifier
        predicted = classifier.predict(features["test"])
        results["models"][name] = {
            "feature_count": features["train"].shape[1],
            **selection,
            "test": metrics(data["test"]["y"], predicted),
        }
        predictions[name] = {
            "predicted": predicted.tolist(),
            "probabilities": classifier.predict_proba(features["test"]).tolist(),
        }
    # Explain the frozen learned classifier before changing any backbone weights.
    cam_records = []
    scaler, logistic = frozen_classifier.steps[0][1], frozen_classifier.steps[1][1]

    def frozen_scores(pooled):
        scaled = (pooled - torch.tensor(scaler.mean_, dtype=pooled.dtype)) / torch.tensor(
            scaler.scale_, dtype=pooled.dtype
        )
        return scaled @ torch.tensor(logistic.coef_.T, dtype=pooled.dtype) + torch.tensor(
            logistic.intercept_, dtype=pooled.dtype
        )

    failures = np.flatnonzero(
        np.array(predictions["resnet18_frozen"]["predicted"]) != data["test"]["y"]
    )
    cam_indices = list(failures[:3])
    if len(cam_indices) < 3:
        cam_indices += [i for i in range(len(data["test"]["y"])) if i not in cam_indices][
            : 3 - len(cam_indices)
        ]
    for index in cam_indices:
        target = predictions["resnet18_frozen"]["predicted"][index]
        rgb, heat = gradcam(model, processor, data["test"]["images"][index], frozen_scores, target)
        cam_records.append((rgb, heat))
    np.savez_compressed(
        args.output / "gradcam_arrays.npz",
        rgb=np.array([x[0] for x in cam_records]),
        heat=np.array([x[1] for x in cam_records]),
        indices=cam_indices,
    )
    head, history = fine_tune(model, processor, data, epochs=1)
    probabilities = predict(model, head, processor, data["test"]["images"])
    predictions["resnet18_partial_finetune"] = {
        "predicted": probabilities.argmax(1).tolist(),
        "probabilities": probabilities.tolist(),
    }
    results["models"]["resnet18_partial_finetune"] = {
        "history": history,
        "selected_epoch": max(history, key=lambda x: x["validation_macro_f1"])["epoch"],
        "test": metrics(data["test"]["y"], probabilities.argmax(1)),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        + sum(p.numel() for p in head.parameters()),
    }
    majority = int(np.bincount(data["train"]["y"]).argmax())
    results["majority_baseline"] = metrics(
        data["test"]["y"], np.full(len(data["test"]["y"]), majority)
    )
    results["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    write_json(args.output / "metrics.json", results)
    write_json(args.output / "predictions.json", predictions)
    write_json(args.output / "data_manifest.json", manifest)
    write_json(
        args.output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_threads": args.threads,
            "packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "Pillow",
                    "matplotlib",
                    "scikit-learn",
                    "pyarrow",
                    "torch",
                    "transformers",
                    "safetensors",
                ]
            },
        },
    )
    # Images for standalone plot regeneration, not raw dataset redistribution.
    selected = np.flatnonzero(
        np.array(predictions["resnet18_frozen"]["predicted"]) != data["test"]["y"]
    )[:8]
    failure_dir = args.output / "failure_images"
    failure_dir.mkdir(exist_ok=True)
    for index in selected:
        data["test"]["images"][index].save(failure_dir / f"test-{index:04d}.png")
    if audit["near_cross_split_candidates"]:
        review = args.output / "duplicate_candidates"
        review.mkdir(exist_ok=True)
        for pair in audit["near_cross_split_candidates"]:
            for row_id in [pair["a"], pair["b"]]:
                split, index = row_id.split(":")
                data[split]["images"][int(index)].save(review / f"{split}-{index}.png")
    print(results, flush=True)


if __name__ == "__main__":
    main()
