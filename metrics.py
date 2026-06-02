"""
Evaluation metrics for semantic segmentation.

Uses a running confusion matrix accumulated over the whole dataset (the statistically
correct way to compute mIoU — averaging per-image IoU, as the baseline did, is biased
when small/absent classes appear). Also holds the color palette + colorization helper.
"""

import numpy as np
import torch

from config import NUM_CLASSES, IGNORE_INDEX, CLASS_NAMES, COLOR_PALETTE

# (N, 3) uint8 palette, index-aligned with CLASS_NAMES.
PALETTE = np.array(COLOR_PALETTE, dtype=np.uint8)


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Map a (H, W) class-index mask to an (H, W, 3) RGB image."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[mask == c] = PALETTE[c]
    return out


class ConfusionMatrix:
    """Accumulates an (C, C) confusion matrix over batches: rows = GT, cols = pred."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """pred, target: torch tensors or np arrays of class indices, any shape."""
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        # ignore predictions/targets outside [0, num_classes)
        k = (target >= 0) & (target < self.num_classes)
        pred, target = pred[k], target[k]
        idx = self.num_classes * target.astype(np.int64) + pred.astype(np.int64)
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += binc.reshape(self.num_classes, self.num_classes)

    # --- derived metrics ---------------------------------------------------
    def iou_per_class(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp
        denom = tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denom > 0, tp / denom, np.nan)  # absent classes -> NaN
        return iou

    def dice_per_class(self) -> np.ndarray:
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = np.where(denom > 0, 2 * tp / denom, np.nan)
        return dice

    def mean_iou(self) -> float:
        return float(np.nanmean(self.iou_per_class()))

    def mean_dice(self) -> float:
        return float(np.nanmean(self.dice_per_class()))

    def pixel_accuracy(self) -> float:
        total = self.mat.sum()
        return float(np.diag(self.mat).sum() / total) if total > 0 else 0.0

    def summary_str(self) -> str:
        iou = self.iou_per_class()
        dice = self.dice_per_class()
        lines = ["EVALUATION RESULTS", "=" * 50,
                 f"Mean IoU:        {self.mean_iou():.4f}",
                 f"Mean Dice:       {self.mean_dice():.4f}",
                 f"Pixel Accuracy:  {self.pixel_accuracy():.4f}",
                 "=" * 50, "", "Per-Class IoU / Dice:", "-" * 50]
        for i, name in enumerate(CLASS_NAMES):
            iou_s = f"{iou[i]:.4f}" if not np.isnan(iou[i]) else "  N/A "
            dice_s = f"{dice[i]:.4f}" if not np.isnan(dice[i]) else "  N/A "
            lines.append(f"  {name:<16} IoU {iou_s}   Dice {dice_s}")
        return "\n".join(lines)
