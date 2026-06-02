"""
Dataset + augmentation pipeline.

Correct 10-class label map (incl. Flowers 600), 16-bit mask reading, NEAREST mask resize.

Augmentation is the domain-generalization core. It is split into:
  - geometric (flip + wide-scale RandomResizedCrop) applied jointly to image+mask. The wide
    scale range (0.25-1.0) creates zoomed-in crops so classes that are tiny in the train
    domain but large in the test domain (e.g. Rocks: 1.2% -> 18% of pixels) are seen big.
  - photometric / weather / occlusion (image only) — strong color randomization to bridge
    the desert env-A -> env-B appearance gap.

`two_view=True` returns two independent photometric views of the same geometric crop, for the
DG consistency loss in train.py.
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


def build_geometric(img_h: int, img_w: int, train: bool):
    if not train:
        return A.Compose([A.Resize(img_h, img_w, interpolation=cv2.INTER_LINEAR)])
    rrc = _try(lambda: A.RandomResizedCrop(size=(img_h, img_w), scale=(0.25, 1.0),
                                           ratio=(1.4, 2.1), p=1.0))
    if rrc is None:  # older albumentations API
        rrc = _try(lambda: A.RandomResizedCrop(height=img_h, width=img_w, scale=(0.25, 1.0),
                                               ratio=(1.4, 2.1), p=1.0))
    return A.Compose([A.HorizontalFlip(p=0.5),
                      rrc if rrc is not None else A.Resize(img_h, img_w)])


def build_photometric(train: bool):
    """Image-only appearance randomization (returns None when not training)."""
    if not train:
        return None
    aug = [
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=35, val_shift_limit=20, p=0.6),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
    ]
    for risky in (
        lambda: A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.4),
        lambda: A.CLAHE(clip_limit=4.0, p=0.2),
        lambda: A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=7),
                         A.GaussNoise()], p=0.3),
        lambda: A.RandomShadow(p=0.3),
        lambda: A.RandomFog(p=0.15),
        lambda: A.RandomSunFlare(src_radius=120, p=0.1),
        lambda: A.ToGray(p=0.05),
        lambda: A.ChannelShuffle(p=0.05),
        lambda: A.Solarize(p=0.05),
        lambda: A.CoarseDropout(p=0.3),     # occlusion -> robustness for thin classes
    ):
        t = _try(risky)
        if t is not None:
            aug.append(t)
    return A.Compose(aug)


def build_post():
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


class MaskDataset(Dataset):
    """Loads (image, mask, id) — or (image1, image2, mask, id) when two_view — for a split."""

    def __init__(self, split: str, img_h: int, img_w: int, augment: bool = False,
                 two_view: bool = False):
        base = split_dir(split)
        self.image_dir = os.path.join(base, IMAGE_SUBDIR)
        self.mask_dir = os.path.join(base, MASK_SUBDIR)
        self.ids = sorted(os.listdir(self.image_dir))
        self.has_masks = os.path.isdir(self.mask_dir) and len(os.listdir(self.mask_dir)) > 0
        self.geo = build_geometric(img_h, img_w, augment)
        self.photo = build_photometric(augment)
        self.post = build_post()
        self.two_view = two_view and augment

    def __len__(self):
        return len(self.ids)

    def _view(self, img_g):
        x = self.photo(image=img_g)["image"] if self.photo is not None else img_g
        return self.post(image=x)["image"]

    def __getitem__(self, idx):
        data_id = self.ids[idx]
        img = cv2.imread(os.path.join(self.image_dir, data_id), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.has_masks:
            raw = cv2.imread(os.path.join(self.mask_dir, data_id), cv2.IMREAD_UNCHANGED)
            mask = convert_mask(raw)
        else:
            mask = np.full(img.shape[:2], IGNORE_INDEX, dtype=np.uint8)

        g = self.geo(image=img, mask=mask)
        img_g, mask_g = g["image"], g["mask"]
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_g)).long()

        if self.two_view:
            return self._view(img_g), self._view(img_g), mask_t, data_id
        return self._view(img_g), mask_t, data_id


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
