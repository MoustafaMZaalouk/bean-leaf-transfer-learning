"""Limited-budget fine-tuning: final ResNet stage and a new three-class head only."""

import random

import numpy as np
import torch
from PIL import ImageEnhance, ImageOps


def augment(image, rng):
    """Train-only flips and mild brightness change, fixed Python RNG stream."""
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.5:
        image = ImageOps.flip(image)
    return ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))


def fine_tune(model, processor, data, epochs=1):
    model.requires_grad_(False)
    model.encoder.stages[-1].requires_grad_(True)
    head = torch.nn.Linear(model.config.hidden_sizes[-1], 3)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.stages[-1].parameters(), "lr": 0.0001},
            {"params": head.parameters(), "lr": 0.001},
        ],
        weight_decay=0.01,
    )
    rng = random.Random(42)
    order_rng = np.random.default_rng(42)
    history, best_score, best_state = [], -1, None
    for epoch in range(epochs):
        # BatchNorm statistics stay fixed, including in the trainable stage.
        model.eval()
        head.train()
        indices = order_rng.permutation(len(data["train"]["y"]))
        total_loss = 0.0
        for start in range(0, len(indices), 8):
            batch_ids = indices[start : start + 8]
            images = [augment(data["train"]["images"][int(i)], rng) for i in batch_ids]
            inputs = processor(images=images, return_tensors="pt")
            labels = torch.tensor(data["train"]["y"][batch_ids], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            scores = head(model(**inputs).pooler_output.flatten(1))
            loss = torch.nn.functional.cross_entropy(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_ids)
        probabilities = predict(model, head, processor, data["validation"]["images"])
        from sklearn.metrics import f1_score

        score = f1_score(data["validation"]["y"], probabilities.argmax(1), average="macro")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(indices),
                "validation_macro_f1": float(score),
            }
        )
        if score > best_score:
            best_score = score
            best_state = (
                {k: v.detach().clone() for k, v in model.state_dict().items()},
                {k: v.detach().clone() for k, v in head.state_dict().items()},
            )
        print(f"Fine-tune epoch {epoch + 1}: {history[-1]}", flush=True)
    model.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    return head, history


def predict(model, head, processor, images):
    model.eval()
    head.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            inputs = processor(images=images[start : start + 8], return_tensors="pt")
            logits = head(model(**inputs).pooler_output.flatten(1))
            outputs.append(logits.softmax(-1).numpy())
    return np.concatenate(outputs)


def gradcam(model, processor, image, score_function, target):
    """Grad-CAM from final spatial stage for the requested classifier's logit."""
    model.eval()
    activations = []
    handle = model.encoder.stages[-1].register_forward_hook(
        lambda module, inputs, output: activations.append(output)
    )
    try:
        inputs = processor(images=[image], return_tensors="pt")
        inputs["pixel_values"].requires_grad_(True)
        pooled = model(**inputs).pooler_output.flatten(1)
        activation = activations[0]
        gradients = torch.autograd.grad(score_function(pooled)[0, target], activation)[0]
        heat = (gradients.mean((2, 3), keepdim=True) * activation).sum(1).relu()
        heat = torch.nn.functional.interpolate(
            heat[:, None], size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0]
        heat = heat.detach().numpy()
        heat /= max(float(heat.max()), 1e-12)
        # The model processor center-crops, so overlay uses that exact image geometry.
        normalized = inputs["pixel_values"][0].detach().numpy().transpose(1, 2, 0)
        rgb = np.clip(normalized * np.array(processor.image_std) + processor.image_mean, 0, 1)
        return rgb, heat
    finally:
        handle.remove()
