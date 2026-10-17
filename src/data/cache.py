from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from tqdm import tqdm

from .transforms import cache_tensor_from_image


def list_imagefolder(root: Path) -> Tuple[List[str], List[Tuple[Path, int]]]:
    classes = sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not classes:
        raise ValueError(f"No class folders found under {root}")
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    samples: List[Tuple[Path, int]] = []
    for cls_name in classes:
        cls_dir = root / cls_name
        for path in sorted(cls_dir.iterdir()):
            if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                samples.append((path, class_to_idx[cls_name]))
    if not samples:
        raise ValueError(f"No images found under {root}")
    return classes, samples


def build_tensor_cache(
    source_root: Path,
    cache_root: Path,
    target_width: int,
    target_height: int,
    rebuild: bool = False,
) -> Dict[str, int]:
    classes, samples = list_imagefolder(source_root)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    expected = len(samples)
    existing = list(cache_root.rglob("*.pt")) if cache_root.exists() else []
    if not rebuild and len(existing) == expected:
        return class_to_idx

    cache_root.mkdir(parents=True, exist_ok=True)
    if rebuild or len(existing) != expected:
        for path in existing:
            path.unlink()

    for cls_name in classes:
        (cache_root / cls_name).mkdir(parents=True, exist_ok=True)

    for src, label in tqdm(samples, desc=f"Build cache {source_root}"):
        cls_name = classes[label]
        dst = cache_root / cls_name / f"{src.stem}.pt"
        if dst.exists() and not rebuild:
            continue
        with Image.open(src) as img:
            tensor = cache_tensor_from_image(img, target_width, target_height)
        torch.save({"image": tensor, "label": label, "source": str(src)}, dst)
    return class_to_idx
