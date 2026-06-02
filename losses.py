"""
Loss = weighted CrossEntropy + Dice.

CrossEntropy with median-frequency class weights handles the heavy class imbalance
(Logs / Flowers / Ground Clutter are rare); Dice directly optimizes region overlap
(correlates with the IoU metric we're scored on). Both honor ignore_index.
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, IGNORE_INDEX, ROOT


def class_weights_from_counts(counts: np.ndarray, mode: str = "median_freq",
                              clip: float = 6.0) -> np.ndarray:
    """Class weights for segmentation from per-class PIXEL counts.

    'median_freq' (median-frequency balancing, Eigen & Fergus 2015): w_c = median(freq)/freq_c.
    This is robust at the billions-of-pixels scale of dense labels, where the effective-number
    method (Cui et al.) saturates and collapses to uniform weights. Absent classes get weight 0;
    present-class weights are normalized to mean 1 and clipped to avoid extremes.
    """
    counts = counts.astype(np.float64)
    total = counts.sum()
    present = counts > 0
    weights = np.zeros_like(counts)
    if not present.any() or total == 0:
        return weights.astype(np.float32)
    freq = counts / total
    if mode == "inv_sqrt":
        weights[present] = 1.0 / np.sqrt(freq[present])
    else:  # median_freq
        med = np.median(freq[present])
        weights[present] = med / freq[present]
    weights[present] /= weights[present].mean()
    if clip:
        weights[present] = np.clip(weights[present], 1.0 / clip, clip)
    return weights.astype(np.float32)


def get_class_weights(cache_path: str = None, **kw) -> torch.Tensor:
    """Class weights from the training masks. The expensive per-class PIXEL counts are
    cached; weights are (re)computed cheaply from them on each call (so changing the
    weighting scheme needs no cache invalidation)."""
    cache_path = cache_path or os.path.join(ROOT, "class_weights.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            counts = np.array(json.load(f)["counts"], dtype=np.float64)
    else:
        from dataset import count_class_pixels
        counts = count_class_pixels("train").astype(np.float64)
        with open(cache_path, "w") as f:
            json.dump({"counts": counts.tolist()}, f, indent=2)
    weights = class_weights_from_counts(counts, **kw)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Lovasz-Softmax (Berman et al., 2018) — a smooth surrogate that directly optimizes
# the IoU (Jaccard) index. Strong for rare / thin classes where CE+Dice plateau.
# ---------------------------------------------------------------------------
def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probs, target, ignore_index: int = IGNORE_INDEX):
    """probs: (B,C,H,W) softmax probabilities; target: (B,H,W). Averages over the
    classes present in the batch ('present' variant)."""
    b, c, h, w = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, c)        # (P, C)
    target = target.reshape(-1)                             # (P,)
    valid = target != ignore_index
    probs, target = probs[valid], target[valid]
    if probs.numel() == 0:
        return probs.sum() * 0.0
    losses = []
    for cls in range(c):
        fg = (target == cls).float()
        if fg.sum() == 0:                                   # class absent in this batch
            continue
        errors = (fg - probs[:, cls]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX,
                 smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        valid = (target != self.ignore_index).unsqueeze(1).float()       # (B,1,H,W)
        target_c = target.clamp(0, self.num_classes - 1)
        onehot = F.one_hot(target_c, self.num_classes).permute(0, 3, 1, 2).float()
        probs, onehot = probs * valid, onehot * valid
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        card = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """ce_w * CE(class_weights) + dice_w * Dice + lovasz_w * Lovasz-Softmax."""

    def __init__(self, class_weights=None, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 lovasz_weight: float = 1.0, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_w, self.dice_w, self.lovasz_w = ce_weight, dice_weight, lovasz_weight
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        loss = self.ce_w * self.ce(logits, target)
        if self.dice_w:
            loss = loss + self.dice_w * self.dice(logits, target)
        if self.lovasz_w:
            probs = F.softmax(logits.float(), dim=1)
            loss = loss + self.lovasz_w * lovasz_softmax(probs, target, self.ignore_index)
        return loss
