from typing import List

import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    weights_per_class = 1.0 / counts.clamp_min(1.0)
    weights = weights_per_class[torch.tensor(labels)]
    return WeightedRandomSampler(weights=weights, num_samples=len(labels), replacement=True)


def class_weights_from_labels(labels: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights.to(device)

