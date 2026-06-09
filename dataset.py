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
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    VALUE_MAP, IGNORE_INDEX, IMAGE_SUBDIR, MASK_SUBDIR, split_dir,
    IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES,
)

SKY_INDEX = CLASS_NAMES.index("Sky")   # never paste rocks over the sky

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


def pasta_perturb(img: np.ndarray, alpha: float = 3.0, k: float = 2.0,
                  beta: float = 0.25) -> np.ndarray:
    """PASTA (Proportional Amplitude Spectrum Training Augmentation, arXiv:2212.00979).

    Perturbs the Fourier *amplitude* spectrum (low-level style/texture) with frequency-
    dependent multiplicative jitter while leaving the *phase* (semantics/layout) untouched.
    Higher spatial frequencies get stronger perturbation, simulating the appearance shift
    between the synthetic source and the unseen target environment.

    img: HxWx3 uint8 RGB. Returns a uint8 RGB image of the same shape.
    """
    h, w = img.shape[:2]
    # centered (DC at middle, post-fftshift) normalized radial frequency, max ~0.5
    m = (np.arange(h) - h / 2.0)[:, None]
    n = (np.arange(w) - w / 2.0)[None, :]
    freq = np.sqrt(m * m + n * n) / np.sqrt(h * h + w * w)
    sigma = (2.0 * alpha * np.sqrt(freq)) ** k + beta            # (H, W)

    x = img.astype(np.float32)
    out = np.empty_like(x)
    for c in range(img.shape[2]):
        f = np.fft.fftshift(np.fft.fft2(x[..., c]))
        amp, phase = np.abs(f), np.angle(f)
        eps = 1.0 + sigma * np.random.randn(h, w)
        np.clip(eps, 0.0, None, out=eps)                        # no negative amplitude
        f2 = (amp * eps) * np.exp(1j * phase)
        out[..., c] = np.fft.ifft2(np.fft.ifftshift(f2)).real
    return np.clip(out, 0, 255).astype(np.uint8)


class MaskDataset(Dataset):
    """Loads (image, mask, id) — or (image1, image2, mask, id) when two_view — for a split."""

    def __init__(self, split: str, img_h: int, img_w: int, augment: bool = False,
                 two_view: bool = False, pasta: bool = False, pasta_alpha: float = 3.0,
                 pasta_k: float = 2.0, pasta_beta: float = 0.25, pasta_p: float = 0.5,
                 copy_paste: bool = False, copy_paste_p: float = 0.5,
                 copy_paste_classes=(7,), copy_paste_scale=(1.0, 2.5),
                 copy_paste_min_pixels: int = 1500):
        base = split_dir(split)
        self.image_dir = os.path.join(base, IMAGE_SUBDIR)
        self.mask_dir = os.path.join(base, MASK_SUBDIR)
        self.ids = sorted(os.listdir(self.image_dir))
        self.has_masks = os.path.isdir(self.mask_dir) and len(os.listdir(self.mask_dir)) > 0
        self.geo = build_geometric(img_h, img_w, augment)
        self.photo = build_photometric(augment)
        self.post = build_post()
        self.two_view = two_view and augment
        # PASTA (image-only); applied per view so the two consistency views get independent
        # amplitude perturbations of the same crop -> stronger style-invariance signal.
        self.pasta = pasta and augment
        self.pasta_alpha, self.pasta_k, self.pasta_beta, self.pasta_p = (
            pasta_alpha, pasta_k, pasta_beta, pasta_p)
        # Copy-paste (train-only): build a per-class index of source images from THIS split.
        self.copy_paste = copy_paste and augment and self.has_masks
        self.copy_paste_p = copy_paste_p
        self.copy_paste_classes = tuple(copy_paste_classes)
        self.copy_paste_scale = tuple(copy_paste_scale)
        self._cp_ids = {}
        if self.copy_paste:
            for cls in self.copy_paste_classes:
                self._cp_ids[cls] = build_class_index(split, cls, copy_paste_min_pixels)
            kept = {c: len(v) for c, v in self._cp_ids.items()}
            print(f"Copy-paste source index (min {copy_paste_min_pixels}px): {kept}")

    def __len__(self):
        return len(self.ids)

    def _view(self, img_g):
        x = self.photo(image=img_g)["image"] if self.photo is not None else img_g
        if self.pasta and np.random.rand() < self.pasta_p:
            x = pasta_perturb(x, self.pasta_alpha, self.pasta_k, self.pasta_beta)
        return self.post(image=x)["image"]

    def _paste_class(self, img_g, mask_g):
        """Paste 1-3 scaled-up source regions to manufacture larger, more frequent rock
        fields (env-B-like) than the small scattered rocks of the train domain."""
        for _ in range(np.random.randint(1, 4)):
            img_g, mask_g = self._paste_once(img_g, mask_g)
        return img_g, mask_g

    def _paste_once(self, img_g, mask_g):
        """Composite a scaled-up region of a target class from a random source image onto
        the current crop (image + label mask). Returns (img, mask), unchanged on failure."""
        h, w = mask_g.shape[:2]
        cls = self.copy_paste_classes[np.random.randint(len(self.copy_paste_classes))]
        ids = self._cp_ids.get(cls, [])
        if not ids:
            return img_g, mask_g
        sid = ids[np.random.randint(len(ids))]
        s_img = cv2.imread(os.path.join(self.image_dir, sid), cv2.IMREAD_COLOR)
        s_raw = cv2.imread(os.path.join(self.mask_dir, sid), cv2.IMREAD_UNCHANGED)
        if s_img is None or s_raw is None:
            return img_g, mask_g
        s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
        s_img = cv2.resize(s_img, (w, h), interpolation=cv2.INTER_LINEAR)
        s_mask = cv2.resize(convert_mask(s_raw), (w, h), interpolation=cv2.INTER_NEAREST)

        region = s_mask == cls
        if region.sum() < 100:
            return img_g, mask_g
        ys, xs = np.where(region)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop_img = s_img[y0:y1, x0:x1]
        crop_reg = region[y0:y1, x0:x1]

        scale = np.random.uniform(*self.copy_paste_scale)            # enlarge -> big rock fields
        nh = max(1, min(int(crop_img.shape[0] * scale), h))
        nw = max(1, min(int(crop_img.shape[1] * scale), w))
        crop_img = cv2.resize(crop_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        crop_reg = cv2.resize(crop_reg.astype(np.uint8), (nw, nh),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        if not crop_reg.any():
            return img_g, mask_g

        # bias placement toward the lower (ground) part of the frame
        lo = min(int(0.30 * h), h - nh)
        oy = np.random.randint(lo, h - nh + 1)
        ox = np.random.randint(0, w - nw + 1)
        img_g, mask_g = img_g.copy(), mask_g.copy()
        sub_img = img_g[oy:oy + nh, ox:ox + nw]
        sub_mask = mask_g[oy:oy + nh, ox:ox + nw]
        # never paste over Sky -> keeps composites physically plausible (rocks on ground)
        allowed = crop_reg & (sub_mask != SKY_INDEX)
        sub_img[allowed] = crop_img[allowed]
        sub_mask[allowed] = cls
        return img_g, mask_g

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
        if self.copy_paste and np.random.rand() < self.copy_paste_p:
            img_g, mask_g = self._paste_class(img_g, mask_g)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_g)).long()

        if self.two_view:
            return self._view(img_g), self._view(img_g), mask_t, data_id
        return self._view(img_g), mask_t, data_id


def build_class_index(split: str, class_idx: int, min_pixels: int = 1500,
                      cache_path: str = None) -> list:
    """Return the list of image ids in `split` whose mask contains at least `min_pixels`
    pixels of `class_idx`. Result is cached to JSON (per class) like the class-weight cache."""
    from config import ROOT
    cache_path = cache_path or os.path.join(ROOT, f"class_index_c{class_idx}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            blob = json.load(f)
        if blob.get("min_pixels") == min_pixels:
            return blob["ids"]
    base = split_dir(split)
    mask_dir = os.path.join(base, MASK_SUBDIR)
    ids = []
    for path in sorted(glob.glob(os.path.join(mask_dir, "*"))):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        if int((convert_mask(raw) == class_idx).sum()) >= min_pixels:
            ids.append(os.path.basename(path))
    with open(cache_path, "w") as f:
        json.dump({"ids": ids, "class_idx": class_idx, "min_pixels": min_pixels}, f)
    return ids


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
