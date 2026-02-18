"""
Optimized Segmentation Training Script — Hackathon Edition v3
Improvements over v2:
  1. Class-weighted Focal Loss  — kills dying-class problem (biggest win)
  2. ASPP decoder (DeepLabV3+)  — multi-scale receptive field in head
  3. Top-3 checkpoint saving    — enables ensemble at test time
  4. Dynamic class weights      — computed from actual dataset pixel counts
  + All previous: DINOv2-base frozen→unfreeze, AMP, AdamW+OneCycleLR,
    Dice loss, joint augmentations, gradient accumulation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import random
import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# ============================================================================
# CONFIG — edit here only
# ============================================================================
BACKBONE_SIZE           = "base"    # small=384d | base=768d | large=1024d
BATCH_SIZE              = 4
ACCUM_STEPS             = 2         # effective batch = BATCH_SIZE * ACCUM_STEPS
LR                      = 3e-4
N_EPOCHS                = 35
IMG_H                   = int(((540 // 2) // 14) * 14)  # 476
IMG_W                   = int(((960 // 2) // 14) * 14)  # 476
HEAD_CHANNELS           = 256
NUM_WORKERS             = 4
PIN_MEMORY              = True
BACKBONE_UNFREEZE_EPOCH = 10        # unfreeze last DINOv2 block here
BACKBONE_LR             = LR * 0.08 # 1.5e-5 — conservative for pretrained weights
TOP_K_CHECKPOINTS       = 3         # save top-k by val mIoU (for ensemble)
# Focal loss
FOCAL_GAMMA             = 2.0       # focusing parameter  (2.0 is standard)
FOCAL_ALPHA             = 0.25      # base alpha (overridden per-class by freq weights)
DICE_WEIGHT             = 0.25       # combined loss = (1-DICE_WEIGHT)*focal + DICE_WEIGHT*dice

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
VAL_DIR    = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset', 'val')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'train_stats')
CKPT_DIR   = os.path.join(SCRIPT_DIR, 'checkpoints')
BEST_MODEL = os.path.join(SCRIPT_DIR, 'best_segmentation_head.pth')
LAST_MODEL = os.path.join(SCRIPT_DIR, 'last_segmentation_head.pth')

# ============================================================================
# CLASS MAPPING — 11 classes including Flowers (600)
# ============================================================================
VALUE_MAP = {
    0:     0,   # Background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}
N_CLASSES = len(VALUE_MAP)  # 11

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

_LUT = np.zeros(10001, dtype=np.uint8)
for raw, cls in VALUE_MAP.items():
    if raw <= 10000:
        _LUT[raw] = cls


def convert_mask_fast(mask_pil):
    arr = np.array(mask_pil, dtype=np.uint16)
    return Image.fromarray(_LUT[np.clip(arr, 0, 10000)].astype(np.uint8))


# ============================================================================
# CLASS WEIGHT COMPUTATION — scans train masks once, caches result
# ============================================================================
def compute_class_weights(train_dir, n_classes, cache_path=None):
    """
    Counts pixel frequency of each class across all training masks.
    Returns inverse-frequency weights as a torch.FloatTensor.
    Rare classes (0 pixels) get maximum weight = sum_of_others / n_classes.
    """
    if cache_path and os.path.exists(cache_path):
        weights = torch.load(cache_path)
        print(f"  Loaded class weights from cache: {cache_path}")
        return weights

    masks_dir = os.path.join(train_dir, 'Segmentation')
    counts    = np.zeros(n_classes, dtype=np.float64)

    mask_files = sorted(os.listdir(masks_dir))
    print(f"  Scanning {len(mask_files)} masks to compute class frequencies...")
    for fname in tqdm(mask_files, desc="  Counting pixels", leave=False):
        mask = Image.open(os.path.join(masks_dir, fname))
        arr  = _LUT[np.clip(np.array(mask, dtype=np.uint16), 0, 10000)]
        for c in range(n_classes):
            counts[c] += (arr == c).sum()

    total = counts.sum()
    # Inverse-frequency: weight_c = total / (n_classes * count_c)
    # Clip so empty classes get weight = max(present_weights) instead of inf
    weights = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = total / (n_classes * counts[present])
    if not present.all():
        # Give absent classes the highest weight among present
        weights[~present] = weights[present].max() if present.any() else 1.0

    # Normalize so mean weight = 1 (keeps LR scale stable)
    weights = weights / weights.mean()
    weights_t = torch.tensor(weights, dtype=torch.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        torch.save(weights_t, cache_path)

    print("  Class weights computed:")
    for name, w, c in zip(CLASS_NAMES, weights, counts):
        c_str = f"{int(c):>12,}" if c > 0 else "           0"
        print(f"    {name:<20}: pixels={c_str}  weight={w:.4f}")

    return weights_t


# ============================================================================
# AUGMENTATION
# ============================================================================
class JointAugment:
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img  = TF.hflip(img);  mask = TF.hflip(mask)
        if random.random() < 0.2:
            img  = TF.vflip(img);  mask = TF.vflip(mask)
        if random.random() < 0.4:
            angle = random.uniform(-20, 20)
            img  = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST,  fill=0)
        if random.random() < 0.6:
            img = transforms.ColorJitter(brightness=0.35, contrast=0.35,
                                         saturation=0.3, hue=0.08)(img)
        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        if random.random() < 0.4:
            scale = random.uniform(0.7, 1.0)
            ch = int(img.height * scale);  cw = int(img.width * scale)
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(ch, cw))
            img  = TF.resized_crop(img,  i, j, h, w, (IMG_H, IMG_W), antialias=True)
            mask = TF.resized_crop(mask, i, j, h, w, (IMG_H, IMG_W),
                                   interpolation=TF.InterpolationMode.NEAREST)

        # Ensure final output is exactly IMG_H x IMG_W — augmentation paths
        # may leave the image at original size, which breaks batching.
        if img.size != (IMG_W, IMG_H):
            img  = transforms.Resize((IMG_H, IMG_W))(img)
            mask = transforms.Resize((IMG_H, IMG_W),
                                     interpolation=transforms.InterpolationMode.NEAREST)(mask)

        return img, mask


# ============================================================================
# DATASET
# ============================================================================
_MASK_RESIZE = transforms.Resize((IMG_H, IMG_W), interpolation=transforms.InterpolationMode.NEAREST)
_TO_TENSOR   = transforms.ToTensor()
_NORMALIZE   = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class MaskDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.aug       = JointAugment() if augment else None

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        name = self.data_ids[idx]
        img  = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, name))
        mask = convert_mask_fast(mask)

        if self.aug:
            img, mask = self.aug(img, mask)
        else:
            img  = transforms.Resize((IMG_H, IMG_W))(img)
            mask = _MASK_RESIZE(mask)

        img_t  = _NORMALIZE(_TO_TENSOR(img))
        mask_t = (_TO_TENSOR(mask) * 255).squeeze(0).long()
        return img_t, mask_t


# ============================================================================
# MODEL — ASPP (DeepLabV3+) style head
# ============================================================================
class ConvBnGelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1, groups=1):
        super().__init__()
        pad = dilation * (k // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    def forward(self, x): return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = ConvBnGelu(ch, ch, k=7, groups=ch)
        self.pw = ConvBnGelu(ch, ch, k=1)
    def forward(self, x): return x + self.pw(self.dw(x))


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Rates tuned for our small spatial grid (~34×34 tokens):
    [1, 2, 4, 6] covers local→global context well at this resolution.
    """
    def __init__(self, in_ch, out_ch, rates=(2, 4, 6)):
        super().__init__()
        self.b0  = ConvBnGelu(in_ch, out_ch, k=1)                      # 1×1
        self.b1  = ConvBnGelu(in_ch, out_ch, k=3, dilation=rates[0])   # dilated
        self.b2  = ConvBnGelu(in_ch, out_ch, k=3, dilation=rates[1])   # dilated
        self.b3  = ConvBnGelu(in_ch, out_ch, k=3, dilation=rates[2])   # dilated
        # Global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        # Fuse 5 branches
        self.fuse = ConvBnGelu(out_ch * 5, out_ch, k=1)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        h, w  = x.shape[-2:]
        gap   = F.interpolate(self.gap(x), size=(h, w), mode='bilinear', align_corners=False)
        out   = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), gap], dim=1)
        return self.drop(self.fuse(out))


class SegmentationHead(nn.Module):
    """
    DeepLabV3+-style head:
      patch tokens → project → ASPP → upsample ×2 → ResBlock
                                     → upsample ×2 → RefineBlock → classify
    The 4× upsample happens in two stages so the model can learn at each scale.
    """
    def __init__(self, in_channels, out_channels, token_h, token_w, ch=HEAD_CHANNELS):
        super().__init__()
        self.H, self.W = token_h, token_w

        # 1×1 projection from embedding dim to ch
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU()
        )

        # ASPP module
        self.aspp = ASPP(ch, ch, rates=(2, 4, 6))

        # Upsample ×2 + refine
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 2, stride=2, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU()
        )
        self.res1 = ResBlock(ch)

        # Upsample ×2 + refine
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.GELU()
        )
        self.refine = nn.Sequential(
            ConvBnGelu(ch // 2, ch // 2, k=3),
            ConvBnGelu(ch // 2, ch // 2, k=3),
        )

        self.classifier = nn.Conv2d(ch // 2, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)       # [B, ch, H, W]
        x = self.aspp(x)       # [B, ch, H, W]  — multi-scale context
        x = self.up1(x)        # [B, ch, 2H, 2W]
        x = self.res1(x)
        x = self.up2(x)        # [B, ch//2, 4H, 4W]
        x = self.refine(x)
        return self.classifier(x)


# ============================================================================
# LOSS — Focal + Dice (class-weighted)
# ============================================================================
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with per-class alpha weighting.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Computed via CE-from-log-softmax so it's numerically stable.
    """
    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.05):
        super().__init__()
        self.gamma  = gamma
        self.smooth = label_smoothing
        # Register class_weights as a buffer so it moves with .to(device)
        if class_weights is not None:
            self.register_buffer('alpha', class_weights)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        # logits: [B, C, H, W]  targets: [B, H, W] long
        B, C, H, W = logits.shape
        # Log-softmax for numerical stability
        log_p = F.log_softmax(logits, dim=1)                    # [B, C, H, W]
        # Gather log prob of ground-truth class
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H, W]

        pt = log_pt.exp()                                        # [B, H, W]

        # Focal weight
        focal_w = (1.0 - pt) ** self.gamma                      # [B, H, W]

        # Per-class alpha weight (broadcast over spatial dims)
        if self.alpha is not None:
            at = self.alpha[targets]                             # [B, H, W]
            focal_w = at * focal_w

        # Label smoothing: mix hard target with uniform distribution
        if self.smooth > 0:
            # Smooth version of -log_pt: blend with -mean(log_p)
            smooth_loss = -log_p.mean(dim=1)                    # [B, H, W]
            ce_loss     = -(1 - self.smooth) * log_pt - self.smooth * smooth_loss
        else:
            ce_loss = -log_pt

        loss = (focal_w * ce_loss).mean()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes, class_weights=None, smooth=1e-6):
        super().__init__()
        self.n = n_classes
        self.s = smooth
        # Normalised class weights for Dice (so rare classes count more)
        if class_weights is not None:
            self.register_buffer('cw', class_weights / class_weights.sum())
        else:
            self.cw = None

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        loss  = 0.0
        cnt   = 0
        for c in range(self.n):
            p = probs[:, c].reshape(-1)
            t = (targets.reshape(-1) == c).float()
            inter = (p * t).sum()
            denom = p.sum() + t.sum()
            if denom > 0:
                d = 1 - (2 * inter + self.s) / (denom + self.s)
                w = self.cw[c].item() * self.n if self.cw is not None else 1.0
                loss += w * d
                cnt  += 1
        return loss / max(cnt, 1)


class CombinedLoss(nn.Module):
    def __init__(self, n_classes, class_weights=None,
                 focal_gamma=2.0, dice_weight=0.4):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, class_weights=class_weights,
                               label_smoothing=0.05)
        self.dice  = DiceLoss(n_classes, class_weights=class_weights)
        self.dw    = dice_weight

    def forward(self, logits, targets):
        return (1 - self.dw) * self.focal(logits, targets) + \
               self.dw       * self.dice(logits,  targets)


# ============================================================================
# METRICS
# ============================================================================
def batch_iou(logits, targets, n_classes):
    pred = torch.argmax(logits, 1).view(-1).cpu().numpy()
    tgt  = targets.view(-1).cpu().numpy()
    ious = []
    for c in range(n_classes):
        p = pred == c;  t = tgt == c
        u = (p | t).sum()
        ious.append((p & t).sum() / u if u > 0 else float('nan'))
    return np.array(ious)


@torch.no_grad()
def evaluate(head, backbone, loader, device, n_classes, backbone_unfrozen):
    head.eval()
    all_iou = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            if backbone_unfrozen:
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            else:
                with torch.no_grad():
                    feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = head(feats)
            logits = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
        all_iou.append(batch_iou(logits.float(), labels, n_classes))
    head.train()
    class_iou = np.nanmean(all_iou, axis=0)
    return float(np.nanmean(class_iou)), class_iou


# ============================================================================
# BACKBONE UNFREEZING
# ============================================================================
def unfreeze_last_dino_block(backbone, optimizer, backbone_lr):
    last_block = backbone.blocks[-1]
    for p in last_block.parameters():
        p.requires_grad_(True)

    backbone.train()
    for blk in backbone.blocks[:-1]:
        blk.eval()
        for p in blk.parameters():
            p.requires_grad_(False)
    for name, module in backbone.named_children():
        if name != 'blocks':
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    # We don't add a new param-group here because the optimizer was created
    # with the backbone last-block group already. Just return its param count.
    return sum(p.numel() for p in last_block.parameters())


# ============================================================================
# TOP-K CHECKPOINT MANAGER
# ============================================================================
class TopKCheckpoints:
    """
    Maintains a min-heap of the top-k checkpoints by val mIoU.
    When a new checkpoint is better than the worst saved, replaces it.
    """
    def __init__(self, k, ckpt_dir):
        self.k        = k
        self.ckpt_dir = ckpt_dir
        self.heap     = []   # (miou, filepath) — min-heap
        os.makedirs(ckpt_dir, exist_ok=True)

    def update(self, val_miou, epoch, head, backbone, backbone_unfrozen,
               meta: dict):
        """Save checkpoint if it belongs in top-k. Returns True if saved."""
        fpath = os.path.join(self.ckpt_dir, f'ckpt_ep{epoch:03d}_iou{val_miou:.4f}.pth')

        payload = {
            'epoch': epoch,
            'state_dict': head.state_dict(),
            'backbone_state_dict': (backbone.blocks[-1].state_dict()
                                    if backbone_unfrozen else None),
            'backbone_unfrozen': backbone_unfrozen,
            'val_iou': val_miou,
            **meta
        }

        if len(self.heap) < self.k:
            torch.save(payload, fpath)
            heapq.heappush(self.heap, (val_miou, fpath))
            return True
        elif val_miou > self.heap[0][0]:   # better than worst in heap
            _, old_path = heapq.heappop(self.heap)
            if os.path.exists(old_path):
                os.remove(old_path)
            torch.save(payload, fpath)
            heapq.heappush(self.heap, (val_miou, fpath))
            return True
        return False

    def best_path(self):
        """Return path of the highest-IoU checkpoint."""
        if not self.heap:
            return None
        return max(self.heap, key=lambda x: x[0])[1]

    def all_paths(self):
        """Return all saved checkpoint paths, best-first."""
        return [p for _, p in sorted(self.heap, key=lambda x: -x[0])]


# ============================================================================
# PLOTTING
# ============================================================================
def save_plots(history, best_class_iou, best_iou, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'],   label='val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['train_iou'], label='train')
    axes[1].plot(history['val_iou'],   label='val')
    axes[1].axhline(0.80, color='red', linestyle='--', alpha=0.5, label='Target 0.80')
    axes[1].set_title('mIoU'); axes[1].legend(); axes[1].grid(True)

    valid = [v if not np.isnan(v) else 0 for v in best_class_iou]
    colors = ['#e74c3c' if v < 0.5 else '#2ecc71' for v in valid]
    bars = axes[2].bar(range(N_CLASSES), valid, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='mAP@0.5 threshold')
    axes[2].set_xticks(range(N_CLASSES))
    axes[2].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7)
    axes[2].set_title(f'Per-Class IoU (Best mIoU={best_iou:.4f})')
    axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,   exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    trainset = MaskDataset(TRAIN_DIR, augment=True)
    valset   = MaskDataset(VAL_DIR,   augment=False)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print(f"Train: {len(trainset)} | Val: {len(valset)} | Classes: {N_CLASSES}")

    # ── Class weights (computed from pixel frequencies) ────────────────────
    print("\nComputing class weights...")
    weight_cache = os.path.join(OUTPUT_DIR, 'class_weights.pt')
    class_weights = compute_class_weights(TRAIN_DIR, N_CLASSES, cache_path=weight_cache)
    class_weights = class_weights.to(device)

    # ── Backbone ──────────────────────────────────────────────────────────────
    print(f"\nLoading DINOv2-{BACKBONE_SIZE}...")
    archs    = {"small": "vits14", "base": "vitb14_reg",
                "large": "vitl14_reg", "giant": "vitg14_reg"}
    backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{archs[BACKBONE_SIZE]}")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    print("Backbone loaded & fully frozen.")

    # Probe dims — use a non-augmented (resized) sample to avoid patch-size assertion
    with torch.no_grad():
        s_img, _ = valset[0]
        feat = backbone.forward_features(s_img.unsqueeze(0).to(device))["x_norm_patchtokens"]
    n_embed = feat.shape[2]
    tok_h, tok_w = IMG_H // 14, IMG_W // 14
    print(f"Embed: {n_embed} | Tokens: {tok_h}×{tok_w} | Head out: {tok_h*4}×{tok_w*4}")

    # ── Head (ASPP decoder) ───────────────────────────────────────────────────
    head = SegmentationHead(n_embed, N_CLASSES, tok_h, tok_w).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Head params: {n_params/1e6:.2f}M  |  Classes: {CLASS_NAMES}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────────────
    criterion = CombinedLoss(N_CLASSES, class_weights=class_weights,
                             focal_gamma=FOCAL_GAMMA, dice_weight=DICE_WEIGHT)
    # Include the backbone's last block in the optimizer from the start as a
    # separate param-group. They are initially frozen (requires_grad=False)
    # so optimizer won't update them until we unfreeze later — this avoids
    # adding a param-group after scheduler creation which breaks OneCycleLR.
    last_block_params = list(backbone.blocks[-1].parameters())
    optimizer = optim.AdamW([
        {'params': list(head.parameters()), 'lr': LR, 'weight_decay': 1e-4},
        {'params': last_block_params,        'lr': BACKBONE_LR, 'weight_decay': 1e-4}
    ])
    steps_per_epoch = max(1, len(train_loader) // ACCUM_STEPS)
    # Match `max_lr` to optimizer param-groups: main head + backbone last block
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR, BACKBONE_LR],
        steps_per_epoch=steps_per_epoch,
        epochs=N_EPOCHS,
        pct_start=0.1, div_factor=10, final_div_factor=1000
    )
    scaler = GradScaler()

    topk   = TopKCheckpoints(TOP_K_CHECKPOINTS, CKPT_DIR)
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    best_iou       = 0.0
    best_class_iou = np.zeros(N_CLASSES)
    backbone_unfrozen = False

    ckpt_meta = dict(n_embed=n_embed, token_h=tok_h, token_w=tok_w,
                     n_classes=N_CLASSES, backbone_size=BACKBONE_SIZE,
                     class_names=CLASS_NAMES, img_h=IMG_H, img_w=IMG_W)

    print(f"\nStarting {N_EPOCHS} epochs | EffBatch={BATCH_SIZE*ACCUM_STEPS} | AMP=ON")
    print(f"Focal γ={FOCAL_GAMMA} | Dice weight={DICE_WEIGHT} | Top-{TOP_K_CHECKPOINTS} checkpoints")
    print(f"Backbone unfreeze at epoch {BACKBONE_UNFREEZE_EPOCH} (bb_lr={BACKBONE_LR:.2e})")
    print("=" * 70)

    for epoch in range(N_EPOCHS):
        # ── Unfreeze last DINOv2 block ─────────────────────────────────────
        if not backbone_unfrozen and (epoch + 1) >= BACKBONE_UNFREEZE_EPOCH:
            n_new = unfreeze_last_dino_block(backbone, optimizer, BACKBONE_LR)
            backbone_unfrozen = True
            print(f"\n  ★ Epoch {epoch+1}: Unfroze last DINOv2 block "
                  f"({n_new/1e6:.2f}M params, bb_lr={BACKBONE_LR:.2e})")

        head.train()
        if backbone_unfrozen:
            backbone.blocks[-1].train()

        t_losses, t_ious = [], []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{N_EPOCHS} [Train]", leave=False)
        for step, (imgs, labels) in enumerate(pbar):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                if backbone_unfrozen:
                    feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                else:
                    with torch.no_grad():
                        feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = head(feats)
                logits = F.interpolate(logits, size=imgs.shape[2:],
                                       mode='bilinear', align_corners=False)
                loss   = criterion(logits, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                all_trainable = list(head.parameters())
                if backbone_unfrozen:
                    all_trainable += [p for p in backbone.blocks[-1].parameters()
                                      if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            t_losses.append(loss.item() * ACCUM_STEPS)
            t_ious.append(float(np.nanmean(
                batch_iou(logits.float().detach(), labels, N_CLASSES))))
            pbar.set_postfix(loss=f"{t_losses[-1]:.4f}", iou=f"{t_ious[-1]:.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # ── Validation loss ───────────────────────────────────────────────
        head.eval()
        v_losses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast():
                    feats  = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    logits = head(feats)
                    logits = F.interpolate(logits, size=imgs.shape[2:],
                                           mode='bilinear', align_corners=False)
                    v_losses.append(criterion(logits.float(), labels).item())
        head.train()

        val_miou, class_iou = evaluate(head, backbone, val_loader, device,
                                       N_CLASSES, backbone_unfrozen)
        train_loss = float(np.mean(t_losses))
        train_miou = float(np.mean(t_ious))
        val_loss   = float(np.mean(v_losses))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_miou)
        history['val_iou'].append(val_miou)

        bb_lr_str = (f" bb_lr={optimizer.param_groups[-1]['lr']:.2e}"
                     if backbone_unfrozen else " backbone=frozen")
        print(f"\nEp {epoch+1:02d} | loss {train_loss:.4f}/{val_loss:.4f} | "
              f"mIoU {train_miou:.4f}/{val_miou:.4f} | "
              f"head_lr={optimizer.param_groups[0]['lr']:.2e}{bb_lr_str}")

        for cname, ciou in zip(CLASS_NAMES, class_iou):
            flag  = " ← ZERO" if (not np.isnan(ciou) and ciou < 0.01) else ""
            iou_s = f"{ciou:.4f}" if not np.isnan(ciou) else "  N/A"
            print(f"  {cname:<20} : {iou_s}{flag}")

        # ── Save checkpoints ──────────────────────────────────────────────
        ckpt_meta['val_iou']   = val_miou
        ckpt_meta['class_iou'] = class_iou.tolist()
        saved = topk.update(val_miou, epoch + 1, head, backbone,
                            backbone_unfrozen, ckpt_meta)
        if saved:
            print(f"  ✓ Top-{TOP_K_CHECKPOINTS} checkpoint saved")

        if val_miou > best_iou:
            best_iou       = val_miou
            best_class_iou = class_iou.copy()
            # Copy best to canonical path for easy access
            torch.save({
                'epoch': epoch + 1,
                'state_dict': head.state_dict(),
                'backbone_state_dict': (backbone.blocks[-1].state_dict()
                                        if backbone_unfrozen else None),
                'backbone_unfrozen': backbone_unfrozen,
                **ckpt_meta
            }, BEST_MODEL)
            print(f"  ★ BEST model updated  mIoU={best_iou:.4f}")

    # ── Save last checkpoint ──────────────────────────────────────────────────
    torch.save({'state_dict': head.state_dict(), **ckpt_meta}, LAST_MODEL)

    # ── Plots ─────────────────────────────────────────────────────────────────
    save_plots(history, best_class_iou, best_iou, OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Training complete!  Best Val mIoU : {best_iou:.4f}")
    print(f"Best model   : {BEST_MODEL}")
    print(f"Last model   : {LAST_MODEL}")
    print(f"Top-{TOP_K_CHECKPOINTS} ckpts : {topk.all_paths()}")
    print(f"Plots        : {OUTPUT_DIR}/training_curves.png")
    print("\nTop-k checkpoints for ensemble (pass to test script with --ensemble):")
    for p in topk.all_paths():
        print(f"  {p}")


if __name__ == "__main__":
    main()