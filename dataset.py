"""
Dataset + augmentation pipeline.

Fixes the baseline's two data bugs:
  1. Correct 10-class label map INCLUDING Flowers (600); no phantom background.
  2. Masks are read as 16-bit and resized with NEAREST (baseline used bilinear,
     which corrupts integer labels).

Augmentation is the domain-generalization core: heavy photometric distortion plus
weather/occlusion effects so the source-domain model survives the unseen test domain.
"""

import os
import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    VALUE_MAP, IGNORE_INDEX, IMAGE_SUBDIR, MASK_SUBDIR, split_dir,
    IMAGENET_MEAN, IMAGENET_STD,
)

cv2.setNumThreads(0)  # avoid oversubscription with DataLoader workers

# Precompute a lookup table: raw 16-bit id -> train index (vectorized remap).
_MAX_ID = max(VALUE_MAP) + 1
_LUT = np.full(_MAX_ID, IGNORE_INDEX, dtype=np.uint8)
for _raw, _idx in VALUE_MAP.items():
    _LUT[_raw] = _idx


def convert_mask(raw: np.ndarray) -> np.ndarray:
    """Map a raw 16-bit mask to a uint8 class-index mask (unmapped -> IGNORE_INDEX)."""
    raw = raw.astype(np.int64)
    out = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
    in_range = raw < _MAX_ID
    out[in_range] = _LUT[raw[in_range]]
    return out


def _try(make):
    """Build an albumentations transform, returning None if the installed version
    doesn't support the given signature (keeps us robust across versions)."""
    try:
        return make()
    except (TypeError, ValueError):
        return None


def build_transforms(img_h: int, img_w: int, train: bool):
    norm = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]

    if not train:
        return A.Compose([A.Resize(img_h, img_w, interpolation=cv2.INTER_LINEAR)] + norm)

    # geometric (mask handled with nearest internally by albumentations)
    rrc = _try(lambda: A.RandomResizedCrop(size=(img_h, img_w), scale=(0.5, 1.0),
                                           ratio=(1.4, 2.1), p=1.0))
    if rrc is None:  # older albumentations API
        rrc = _try(lambda: A.RandomResizedCrop(height=img_h, width=img_w, scale=(0.5, 1.0),
                                               ratio=(1.4, 2.1), p=1.0))
    geometric = [A.HorizontalFlip(p=0.5)]
    geometric.append(rrc if rrc is not None else A.Resize(img_h, img_w))

    # photometric / weather / occlusion (image only) — simulate the domain shift
    photometric = [
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25,
                             val_shift_limit=15, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)),
                 A.MotionBlur(blur_limit=7),
                 A.GaussNoise()], p=0.3),
    ]
    for risky in (
        lambda: A.RandomShadow(p=0.3),
        lambda: A.RandomFog(p=0.15),
        lambda: A.RandomSunFlare(src_radius=120, p=0.1),
        lambda: A.CoarseDropout(p=0.3),  # occlusion -> helps thin/rare classes (Logs)
    ):
        t = _try(risky)
        if t is not None:
            photometric.append(t)

    return A.Compose(geometric + photometric + norm)


class MaskDataset(Dataset):
    """Loads (image, mask, id) triples for a split ('train'/'val'/'test')."""

    def __init__(self, split: str, img_h: int, img_w: int, augment: bool = False):
        base = split_dir(split)
        self.image_dir = os.path.join(base, IMAGE_SUBDIR)
        self.mask_dir = os.path.join(base, MASK_SUBDIR)
        self.ids = sorted(os.listdir(self.image_dir))
        self.has_masks = os.path.isdir(self.mask_dir) and len(os.listdir(self.mask_dir)) > 0
        self.tf = build_transforms(img_h, img_w, train=augment)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        data_id = self.ids[idx]
        img = cv2.imread(os.path.join(self.image_dir, data_id), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.has_masks:
            raw = cv2.imread(os.path.join(self.mask_dir, data_id), cv2.IMREAD_UNCHANGED)
            mask = convert_mask(raw)
        else:
            mask = np.full(img.shape[:2], IGNORE_INDEX, dtype=np.uint8)

        out = self.tf(image=img, mask=mask)
        image = out["image"]                       # float tensor (3, H, W)
        mask = out["mask"].long()                  # long tensor (H, W)
        return image, mask, data_id


def count_class_pixels(split: str = "train") -> np.ndarray:
    """Total pixel count per class over a split (for class-balanced loss weights)."""
    from config import NUM_CLASSES
    base = split_dir(split)
    mask_dir = os.path.join(base, MASK_SUBDIR)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for path in glob.glob(os.path.join(mask_dir, "*.png")):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        idx = convert_mask(raw)
        binc = np.bincount(idx[idx != IGNORE_INDEX].reshape(-1), minlength=NUM_CLASSES)
        counts += binc[:NUM_CLASSES]
    return counts
