"""Correctness checks independent of network access or pretrained downloads."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.metrics import f1_score
from transformers import ResNetConfig, ResNetModel

from beanleaf.core import (
    ImageSequence,
    audit_duplicates,
    color_features,
    metrics,
    pixel_hash,
    pooled_features,
    validate_data,
    select_classifier,
)
from beanleaf.finetune import augment

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts/evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_metrics_known_answer():
    result = metrics([0, 0, 1, 2], [0, 1, 1, 2])
    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert result["macro_f1"] == pytest.approx(7 / 9)


def test_color_features_known_red_image():
    vector = color_features([Image.new("RGB", (128, 128), (255, 0, 0))])[0]
    np.testing.assert_allclose(vector[:6], [1, 0, 0, 0, 0, 0])
    histograms = vector[6:].reshape(3, 8)
    np.testing.assert_allclose(histograms.sum(1), 1)
    assert histograms[0, 7] == histograms[1, 0] == histograms[2, 0] == 1


def test_decoded_duplicate_audit_detects_leakage():
    red = Image.new("RGB", (10, 10), "red")
    data = {
        "train": {"images": [red], "y": [0], "ids": ["train:0000"]},
        "test": {"images": [red.copy()], "y": [1], "ids": ["test:0000"]},
    }
    audit = audit_duplicates(data)
    assert audit["exact_cross_split"][0]["labels"] == [0, 1]
    assert pixel_hash(red) != pixel_hash(Image.new("RGB", (20, 5), "red"))


def test_scaler_does_not_fit_validation_and_selection_matches_scores():
    train = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    labels = np.array([0, 0, 1, 1, 2, 2])
    validation = np.array([[1000], [1001], [1002]], dtype=float)
    val_labels = np.array([0, 1, 2])
    model, record = select_classifier(train, labels, validation, val_labels)
    assert model.steps[0][1].mean_[0] == 2.5
    chosen = next(x for x in record["candidates"] if x["C"] == record["selected_C"])
    assert chosen["validation_macro_f1"] == f1_score(
        val_labels, model.predict(validation), average="macro"
    )
    assert chosen["validation_macro_f1"] == max(
        x["validation_macro_f1"] for x in record["candidates"]
    )


def tiny_model():
    return ResNetModel(
        ResNetConfig(
            embedding_size=4, hidden_sizes=[4, 8, 16, 32], depths=[1, 1, 1, 1], layer_type="basic"
        )
    )


def processor(images, return_tensors):
    return {
        "pixel_values": torch.stack(
            [
                torch.tensor(np.asarray(image.resize((32, 32))).copy()).permute(2, 0, 1).float()
                / 255
                for image in images
            ]
        )
    }


def test_frozen_extraction_preserves_every_parameter_and_bn_buffer():
    torch.set_num_threads(1)
    model = tiny_model()
    model.train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    images = [Image.new("RGB", (32, 32), "green"), Image.new("RGB", (32, 32), "red")]
    a = pooled_features(model, processor, images, batch_size=1)
    b = pooled_features(model, processor, images, batch_size=2)
    np.testing.assert_allclose(a, b, atol=1e-6)
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_wilson_and_per_class_known_answer():
    assert evaluate.wilson(0, 0) == [None, None]
    low, high = evaluate.wilson(5, 10)
    assert low == pytest.approx(0.236593, abs=1e-6)
    assert high == pytest.approx(0.763407, abs=1e-6)
    result = evaluate.per_class(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]))
    assert result["angular_leaf_spot"]["precision"] == 1
    assert result["angular_leaf_spot"]["recall"] == 0.5


def test_committed_predictions_match_all_recorded_metrics_and_schema():
    predictions = json.loads((ROOT / "reports/predictions.json").read_text())
    recorded = json.loads((ROOT / "reports/metrics.json").read_text())
    assert len(predictions["y_true"]) == 128
    assert set(predictions["y_true"]) == {0, 1, 2}
    assert len(set(predictions["ids"])) == 128
    assert all(x.startswith("test:") for x in predictions["ids"])
    for name, value in recorded["models"].items():
        assert metrics(predictions["y_true"], predictions[name]["predicted"]) == value["test"]
        probabilities = np.array(predictions[name]["probabilities"])
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        assert np.array_equal(probabilities.argmax(1), predictions[name]["predicted"])


def test_image_sequence_decodes_rgb_without_retaining_pixels():
    import io

    output = io.BytesIO()
    Image.new("L", (8, 8), 128).save(output, format="PNG")
    sequence = ImageSequence([output.getvalue()])
    assert sequence[0].mode == "RGB"
    assert sequence[0].getpixel((0, 0)) == (128, 128, 128)
    assert len(sequence[:1]) == 1


def test_augmentation_is_repeatable():
    import random

    image = Image.fromarray(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))
    a = augment(image, random.Random(42))
    b = augment(image, random.Random(42))
    np.testing.assert_array_equal(a, b)


def test_data_validation_rejects_overlap_bad_labels_and_schema():
    a = {"images": [None], "y": [0], "ids": ["shared"]}
    with pytest.raises(ValueError, match="Repeated IDs"):
        validate_data({"train": a, "test": a})
    with pytest.raises(ValueError, match="labels"):
        validate_data({"train": {**a, "y": [99]}})
    with pytest.raises(ValueError, match="schema"):
        validate_data({"train": {"y": [0]}})
    with pytest.raises(ValueError, match="Unaligned"):
        validate_data({"train": {**a, "ids": []}})


def test_partial_finetune_changes_only_allowed_backbone_stage():
    from beanleaf.finetune import fine_tune

    torch.manual_seed(42)
    model = tiny_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    images = [Image.new("RGB", (32, 32), color) for color in ["red", "green", "blue"]]
    data = {
        split: {"images": images, "y": np.array([0, 1, 2])} for split in ["train", "validation"]
    }
    fine_tune(model, processor, data, epochs=1)
    changed = []
    for key, value in model.state_dict().items():
        if not torch.equal(before[key], value):
            changed.append(key)
            assert key.startswith("encoder.stages.3.")
            assert "running_" not in key and "num_batches_tracked" not in key
    assert changed


def test_gradcam_zero_logit_gives_zero_heat_and_removes_hook():
    class FakeProcessor:
        image_std = [1, 1, 1]
        image_mean = [0, 0, 0]

        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.ones(1, 3, 224, 224)}

    from beanleaf.finetune import gradcam

    model = tiny_model()
    count = len(model.encoder.stages[-1]._forward_hooks)
    rgb, heat = gradcam(model, FakeProcessor(), None, lambda pooled: pooled[:, :3] * 0, 0)
    assert len(model.encoder.stages[-1]._forward_hooks) == count
    assert np.all(heat == 0)
    assert np.all(rgb == 1)


def test_data_download_checksum_is_enforced(tmp_path):
    from beanleaf.core import load_data

    (tmp_path / "train-00000-of-00001.parquet").write_bytes(b"corrupt download")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_data(tmp_path)
