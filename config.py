"""
Central configuration: dataset paths, class definitions, color palette, and default
hyperparameters. Everything tunable lives here so train.py / test.py stay thin.
"""

import os
from dataclasses import dataclass, asdict, field
from typing import List

# ---------------------------------------------------------------------------
# Paths (relative to this file's directory = repo root)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "Offroad_Segmentation_Training_Dataset", "train")
VAL_DIR = os.path.join(ROOT, "Offroad_Segmentation_Training_Dataset", "val")
TEST_DIR = os.path.join(ROOT, "Offroad_Segmentation_testImages")

# Each split dir contains these two subfolders (same filenames in both):
IMAGE_SUBDIR = "Color_Images"
MASK_SUBDIR = "Segmentation"


def split_dir(split: str) -> str:
    return {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}[split]


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# Raw 16-bit mask ID -> contiguous train index. 10 real classes, NO background.
# (The provided baseline incorrectly dropped Flowers/600 and added an unused bg.)
VALUE_MAP = {
    100: 0,     # Trees
    200: 1,     # Lush Bushes
    300: 2,     # Dry Grass
    500: 3,     # Dry Bushes
    550: 4,     # Ground Clutter
    600: 5,     # Flowers
    700: 6,     # Logs
    800: 7,     # Rocks
    7100: 8,    # Landscape (catch-all ground)
    10000: 9,   # Sky
}
CLASS_NAMES: List[str] = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
    "Flowers", "Logs", "Rocks", "Landscape", "Sky",
]
NUM_CLASSES = len(CLASS_NAMES)          # 10
IGNORE_INDEX = 255                       # any unmapped pixel -> ignored in loss/metrics

# Distinct RGB colors for visualization (index-aligned with CLASS_NAMES).
COLOR_PALETTE = [
    (34, 139, 34),     # Trees        - forest green
    (0, 255, 0),       # Lush Bushes  - lime
    (210, 180, 140),   # Dry Grass    - tan
    (139, 90, 43),     # Dry Bushes   - brown
    (128, 128, 0),     # Ground Clutter - olive
    (255, 0, 255),     # Flowers      - magenta
    (139, 69, 19),     # Logs         - saddle brown
    (128, 128, 128),   # Rocks        - gray
    (200, 162, 200),   # Landscape    - lavender
    (135, 206, 235),   # Sky          - sky blue
]

# ImageNet normalization (DINOv2 expects this).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Backbone variants: name -> (HuggingFace model id, patch_size). Hidden dim, depth, and
# number of register tokens are read from the model config at build time, so adding a new
# ViT backbone here is enough.
BACKBONES = {
    # DINOv2 (open access)
    "dinov2-small":     ("facebook/dinov2-small", 14),
    "dinov2-base":      ("facebook/dinov2-base", 14),
    "dinov2-large":     ("facebook/dinov2-large", 14),
    # DINOv2 with registers (open; register tokens -> cleaner dense maps)
    "dinov2-reg-base":  ("facebook/dinov2-with-registers-base", 14),
    "dinov2-reg-large": ("facebook/dinov2-with-registers-large", 14),
    # DINOv3 (GATED: accept license on HF + `huggingface-cli login`)
    "dinov3-base":      ("facebook/dinov3-vitb16-pretrain-lvd1689m", 16),
    "dinov3-large":     ("facebook/dinov3-vitl16-pretrain-lvd1689m", 16),
    # backward-compat aliases for earlier checkpoints (config saved backbone='base', etc.)
    "small": ("facebook/dinov2-small", 14),
    "base":  ("facebook/dinov2-base", 14),
    "large": ("facebook/dinov2-large", 14),
}


def patch_size_of(backbone: str) -> int:
    return BACKBONES[backbone][1]


@dataclass
class Config:
    # data / model
    backbone: str = "dinov3-large"
    img_h: int = 576            # 36 * 16 (16:9)
    img_w: int = 1024           # 64 * 16
    num_classes: int = NUM_CLASSES

    # Rein adapters
    use_rein: bool = True
    rein_tokens: int = 100      # learnable tokens per adapter

    # optimization
    epochs: int = 50
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    num_workers: int = 4

    # loss
    ce_weight: float = 1.0
    dice_weight: float = 1.0
    lovasz_weight: float = 1.0          # Lovasz-Softmax: directly optimizes IoU
    use_class_weights: bool = True

    # domain-generalization consistency regularization (two photometric views)
    consistency: bool = False
    consistency_weight: float = 1.0

    # PASTA: Fourier amplitude-spectrum augmentation (bridges the env-A->B style/texture gap).
    # Perturbs the amplitude spectrum (style) while preserving phase (semantics).
    pasta: bool = False
    pasta_alpha: float = 3.0     # perturbation strength
    pasta_k: float = 2.0         # frequency exponent (higher freq -> stronger perturbation)
    pasta_beta: float = 0.25     # baseline jitter applied to all frequencies
    pasta_p: float = 0.5         # per-image probability

    # Copy-paste augmentation: paste scaled-up Rocks regions to manufacture the large rock
    # fields that dominate the unseen test env (env B) but barely exist in train (env A).
    copy_paste: bool = False
    copy_paste_p: float = 0.5
    copy_paste_classes: tuple = (7,)         # Rocks (train index 7)
    copy_paste_scale: tuple = (1.5, 3.0)     # random up-scale of the pasted region
    copy_paste_min_pixels: int = 1500        # min class pixels for a source image to qualify

    # EMA / regularization
    use_ema: bool = True
    ema_decay: float = 0.999

    # misc
    seed: int = 0
    amp: bool = True            # bf16 mixed precision
    early_stop_patience: int = 12
    out_dir: str = os.path.join(ROOT, "runs")

    def validate(self):
        assert self.backbone in BACKBONES, f"backbone must be one of {list(BACKBONES)}"
        p = patch_size_of(self.backbone)
        assert self.img_h % p == 0 and self.img_w % p == 0, \
            f"img_h/img_w must be multiples of {p} for backbone '{self.backbone}'"
        return self

    def to_dict(self):
        return asdict(self)


def get_default_config() -> Config:
    return Config()
