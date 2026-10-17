from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from src.data.transforms import build_normalizer


class RawCachedECGDataset(torch.utils.data.Dataset):
    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root)
        self.classes = sorted([p.name for p in self.cache_root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.classes:
            for path in sorted((self.cache_root / cls_name).glob("*.pt")):
                self.samples.append((path, self.class_to_idx[cls_name]))
        if not self.samples:
            raise ValueError(f"No cached tensors found under {self.cache_root}")
        self.normalize = build_normalizer()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        item = torch.load(path, map_location="cpu", weights_only=False)
        raw = item["image"].float().clamp(0.0, 1.0)
        image = self.normalize(raw)
        return image, raw, label, str(item.get("source", path))


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, image: torch.Tensor, target_class: int | None = None) -> tuple[torch.Tensor, int, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        pred_class = int(logits.argmax(dim=1).item())
        target = pred_class if target_class is None else int(target_class)
        score = logits[:, target].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu()
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam, pred_class, torch.softmax(logits.detach(), dim=1).cpu().squeeze(0)


def select_case_indices(predictions: pd.DataFrame, per_class_correct: int = 2, error_cases: int = 6) -> list[int]:
    df = predictions.copy()
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    df["confidence"] = df[prob_cols].max(axis=1)
    df["correct"] = df["label"] == df["pred"]
    indices: list[int] = []
    for _, group in df[df["correct"]].sort_values("confidence", ascending=False).groupby("label"):
        indices.extend(group.head(per_class_correct).index.tolist())
    errors = df[~df["correct"]].sort_values("confidence", ascending=False).head(error_cases)
    indices.extend(errors.index.tolist())
    return list(dict.fromkeys(int(i) for i in indices))


def save_gradcam_overlay(raw: torch.Tensor, cam: torch.Tensor, output_path: str | Path, alpha: float = 0.42) -> None:
    raw_np = raw.permute(1, 2, 0).numpy()
    cam_np = cam.numpy()
    heatmap = plt.get_cmap("jet")(cam_np)[..., :3]
    overlay = (1 - alpha) * raw_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    image = Image.fromarray((overlay * 255).astype(np.uint8))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def gradcam_target_layer(model: torch.nn.Module, layer_name: str = "spatial.cnn.layer4"):
    module = model
    for part in layer_name.split("."):
        module = getattr(module, part)
    if isinstance(module, torch.nn.Sequential):
        module = module[-1]
    return module
