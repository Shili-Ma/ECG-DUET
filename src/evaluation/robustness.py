from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from src.data.transforms import build_normalizer
from src.models.ecg_model import ECGThreeClassModel


IMAGE_SUFFIX = ".pt"


class PerturbedCachedECGDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        perturbation: str = "clean",
        severity: float | int | str = 0,
        seed: int = 42,
    ):
        self.cache_root = Path(cache_root)
        self.classes = sorted([p.name for p in self.cache_root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.classes:
            for path in sorted((self.cache_root / cls_name).glob(f"*{IMAGE_SUFFIX}")):
                self.samples.append((path, self.class_to_idx[cls_name]))
        if not self.samples:
            raise ValueError(f"No cached tensors found under {self.cache_root}")
        self.perturbation = perturbation
        self.severity = severity
        self.seed = seed
        self.normalize = build_normalizer()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        item = torch.load(path, map_location="cpu", weights_only=False)
        image = item["image"].float().clamp(0.0, 1.0)
        image = apply_perturbation(image, self.perturbation, self.severity, self.seed + index)
        image = self.normalize(image.clamp(0.0, 1.0))
        return image, label


def apply_perturbation(image: torch.Tensor, perturbation: str, severity, seed: int = 42) -> torch.Tensor:
    name = str(perturbation).lower()
    if name in {"clean", "none", "identity"}:
        return image
    if name == "brightness":
        return image * float(severity)
    if name == "contrast":
        mean = image.mean(dim=(1, 2), keepdim=True)
        return (image - mean) * float(severity) + mean
    if name == "rotation":
        return TF.rotate(
            image,
            angle=float(severity),
            interpolation=InterpolationMode.BILINEAR,
            fill=[1.0, 1.0, 1.0],
        )
    if name == "translate":
        _, h, w = image.shape
        pixels = int(round(float(severity) * w))
        return TF.affine(
            image,
            angle=0.0,
            translate=[pixels, 0],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=[1.0, 1.0, 1.0],
        )
    if name == "noise":
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype) * float(severity)
        return image + noise
    if name == "blur":
        radius = float(severity)
        kernel = max(3, int(radius * 4) * 2 + 1)
        return TF.gaussian_blur(image, kernel_size=[kernel, kernel], sigma=[radius, radius])
    if name == "jpeg":
        pil = TF.to_pil_image(image)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return TF.to_tensor(compressed.convert("RGB"))
    if name == "downsample":
        scale = float(severity)
        _, h, w = image.shape
        small_h = max(1, int(round(h * scale)))
        small_w = max(1, int(round(w * scale)))
        x = image.unsqueeze(0)
        x = F.interpolate(x, size=(small_h, small_w), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x.squeeze(0)
    raise ValueError(f"Unknown perturbation: {perturbation}")


def default_perturbation_grid() -> list[dict[str, float | int | str]]:
    return [
        {"perturbation": "clean", "severity": 0, "level": 0, "level_name": "clean"},
        {"perturbation": "brightness", "severity": 0.85, "level": 1, "level_name": "mild darkening"},
        {"perturbation": "brightness", "severity": 1.15, "level": 1, "level_name": "mild brightening"},
        {"perturbation": "brightness", "severity": 0.70, "level": 2, "level_name": "moderate darkening"},
        {"perturbation": "brightness", "severity": 1.30, "level": 2, "level_name": "moderate brightening"},
        {"perturbation": "brightness", "severity": 0.55, "level": 3, "level_name": "severe darkening"},
        {"perturbation": "brightness", "severity": 1.45, "level": 3, "level_name": "severe brightening"},
        {"perturbation": "contrast", "severity": 0.85, "level": 1, "level_name": "mild reduction"},
        {"perturbation": "contrast", "severity": 1.15, "level": 1, "level_name": "mild increase"},
        {"perturbation": "contrast", "severity": 0.70, "level": 2, "level_name": "moderate reduction"},
        {"perturbation": "contrast", "severity": 1.30, "level": 2, "level_name": "moderate increase"},
        {"perturbation": "contrast", "severity": 0.55, "level": 3, "level_name": "severe reduction"},
        {"perturbation": "contrast", "severity": 1.45, "level": 3, "level_name": "severe increase"},
        {"perturbation": "rotation", "severity": -3, "level": 1, "level_name": "mild negative rotation"},
        {"perturbation": "rotation", "severity": 3, "level": 1, "level_name": "mild positive rotation"},
        {"perturbation": "rotation", "severity": -5, "level": 2, "level_name": "moderate negative rotation"},
        {"perturbation": "rotation", "severity": 5, "level": 2, "level_name": "moderate positive rotation"},
        {"perturbation": "rotation", "severity": -8, "level": 3, "level_name": "severe negative rotation"},
        {"perturbation": "rotation", "severity": 8, "level": 3, "level_name": "severe positive rotation"},
        {"perturbation": "translate", "severity": -0.02, "level": 1, "level_name": "mild left shift"},
        {"perturbation": "translate", "severity": 0.02, "level": 1, "level_name": "mild right shift"},
        {"perturbation": "translate", "severity": -0.05, "level": 2, "level_name": "moderate left shift"},
        {"perturbation": "translate", "severity": 0.05, "level": 2, "level_name": "moderate right shift"},
        {"perturbation": "translate", "severity": -0.08, "level": 3, "level_name": "severe left shift"},
        {"perturbation": "translate", "severity": 0.08, "level": 3, "level_name": "severe right shift"},
        {"perturbation": "noise", "severity": 0.01, "level": 1, "level_name": "mild noise"},
        {"perturbation": "noise", "severity": 0.03, "level": 2, "level_name": "moderate noise"},
        {"perturbation": "noise", "severity": 0.05, "level": 3, "level_name": "severe noise"},
        {"perturbation": "blur", "severity": 1, "level": 1, "level_name": "mild blur"},
        {"perturbation": "blur", "severity": 2, "level": 2, "level_name": "moderate blur"},
        {"perturbation": "blur", "severity": 3, "level": 3, "level_name": "severe blur"},
        {"perturbation": "jpeg", "severity": 90, "level": 1, "level_name": "mild compression"},
        {"perturbation": "jpeg", "severity": 70, "level": 2, "level_name": "moderate compression"},
        {"perturbation": "jpeg", "severity": 50, "level": 3, "level_name": "severe compression"},
        {"perturbation": "downsample", "severity": 0.75, "level": 1, "level_name": "mild downsampling"},
        {"perturbation": "downsample", "severity": 0.50, "level": 2, "level_name": "moderate downsampling"},
        {"perturbation": "downsample", "severity": 0.25, "level": 3, "level_name": "severe downsampling"},
    ]


def parse_perturbation_grid(path: str | None = None) -> list[dict[str, float | int | str]]:
    if path is None:
        return default_perturbation_grid()
    df = pd.read_csv(path)
    if "perturbation" not in df.columns or "severity" not in df.columns:
        raise ValueError("Perturbation CSV must contain columns: perturbation,severity")
    if "level" not in df.columns:
        df["level"] = pd.NA
    if "level_name" not in df.columns:
        df["level_name"] = ""
    return [
        {
            "perturbation": str(row["perturbation"]),
            "severity": row["severity"],
            "level": row["level"],
            "level_name": row["level_name"],
        }
        for _, row in df.iterrows()
    ]


def load_main_model(results_dir: str | Path = "results", device: torch.device | None = None):
    results_dir = Path(results_dir)
    summary_path = results_dir / "main" / "metrics_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing main summary: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    checkpoint_path = Path(summary["best_checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", summary.get("args", {}))
    class_to_idx = checkpoint.get("class_to_idx", summary.get("class_to_idx"))
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    model = ECGThreeClassModel(
        img_size=int(args.get("seq_len", 256)),
        t_embed=128,
        t_layers=2,
        s_idcnn=128,
        fusion_out=256,
        num_classes=len(class_names),
        pretrained_backbone=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, class_names, summary, device


def evaluate_loader_for_robustness(model, loader, device, class_names: list[str]) -> tuple[pd.DataFrame, dict]:
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            all_labels.append(labels.cpu())
            all_preds.append(preds)
            all_probs.append(probs)
    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()
    probs = torch.cat(all_probs).numpy()
    rows = {
        "label": labels,
        "pred": preds,
        "label_name": [class_names[i] for i in labels],
        "pred_name": [class_names[i] for i in preds],
    }
    for idx, name in enumerate(class_names):
        rows[f"prob_{name}"] = probs[:, idx]
    pred_df = pd.DataFrame(rows)
    summary = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }
    for idx, name in enumerate(class_names):
        summary[f"f1_{name}"] = float(f1_score(labels, preds, labels=[idx], average=None, zero_division=0)[0])
    return pred_df, summary


def robustness_auc(summary_df: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    rows = []
    for perturbation, group in summary_df[summary_df["perturbation"] != "clean"].groupby("perturbation"):
        values = group[metric].dropna().astype(float)
        if values.empty:
            continue
        rows.append(
            {
                "perturbation": perturbation,
                f"{metric}_mean": float(values.mean()),
                f"{metric}_min": float(values.min()),
                f"{metric}_robustness_auc": float(np.trapezoid(values.to_numpy()) / max(1, len(values) - 1)),
            }
        )
    return pd.DataFrame(rows)
