from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF


def resize_with_white_padding(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    img = img.convert("RGB")
    orig_w, orig_h = img.size
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (target_width, target_height), color=(255, 255, 255))
    paste_x = (target_width - new_w) // 2
    paste_y = (target_height - new_h) // 2
    padded.paste(img_resized, (paste_x, paste_y))
    return padded


def cache_tensor_from_image(img: Image.Image, target_width: int, target_height: int) -> torch.Tensor:
    padded = resize_with_white_padding(img, target_width, target_height)
    return TF.to_tensor(padded)


def build_train_augmentation():
    return T.Compose([
        T.RandomApply([T.ColorJitter(brightness=0.08, contrast=0.08)], p=0.35),
        T.RandomAffine(degrees=2, translate=(0.01, 0.02), fill=1.0),
    ])


def build_normalizer():
    return T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

