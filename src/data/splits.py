from typing import List, Tuple

import torch


def split_indices(n_total: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    return indices[:n_train], indices[n_train:]

