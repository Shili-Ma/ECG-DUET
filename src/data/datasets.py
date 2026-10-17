from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .transforms import build_normalizer, build_train_augmentation


class CachedECGDataset(Dataset):
    def __init__(self, cache_root: str, train: bool = False):
        self.cache_root = Path(cache_root)
        self.classes = sorted([p.name for p in self.cache_root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            for path in sorted((self.cache_root / cls_name).glob("*.pt")):
                self.samples.append((path, self.class_to_idx[cls_name]))
        if not self.samples:
            raise ValueError(f"No cached tensors found under {self.cache_root}")

        self.train = train
        self.normalize = build_normalizer()
        self.augment = build_train_augmentation()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        item = torch.load(path, map_location="cpu", weights_only=False)
        image = item["image"].float()
        if self.train:
            image = self.augment(image)
        image = self.normalize(image)
        return image, label

    def labels(self) -> List[int]:
        return [label for _, label in self.samples]

