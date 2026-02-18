"""
Optimized Segmentation Test/Evaluation Script — Hackathon Edition v3
New features:
  1. Test Time Augmentation (TTA) — hflip + multi-scale, averages logits
  2. Checkpoint Ensemble         — average logits across top-k checkpoints
  3. Full metrics: mIoU, mAP@0.5, Dice, Pixel Accuracy (per-class + mean)
  4. Auto-loads config from checkpoint (no manual settings needed)
  5. Includes Flowers (600) — 11 classes

Usage:
  # Single model, no TTA:
  python test_segmentation.py --model_path best_segmentation_head.pth

  # Single model + TTA:
  python test_segmentation.py --model_path best_segmentation_head.pth --tta

  # Ensemble of top-3 + TTA (best possible score):
  python test_segmentation.py \
      --ensemble checkpoints/ckpt_ep020_iou0.82.pth \
                 checkpoints/ckpt_ep019_iou0.81.pth \
                 checkpoints/ckpt_ep018_iou0.80.pth \
      --tta --has_labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import glob
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast

# ============================================================================
# CLASS MAPPING — 11 classes
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
N_CLASSES_DEFAULT = len(VALUE_MAP)
CLASS_NAMES_DEFAULT = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]
COLOR_PALETTE = np.array([
    [0,   0,   0  ],   # Background
    [34,  139, 34 ],   # Trees
    [0,   200, 0  ],   # Lush Bushes
    [210, 180, 140],   # Dry Grass
    [139, 90,  43 ],   # Dry Bushes
    [128, 128, 0  ],   # Ground Clutter
    [255, 200, 0  ],   # Flowers
    [139, 69,  19 ],   # Logs
    [128, 128, 128],   # Rocks
    [160, 82,  45 ],   # Landscape
    [135, 206, 235],   # Sky
], dtype=np.uint8)

_LUT = np.zeros(10001, dtype=np.uint8)
for raw, cls in VALUE_MAP.items():
    if raw <= 10000:
        _LUT[raw] = cls


def convert_mask_fast(mask_pil):
    arr = np.array(mask_pil, dtype=np.uint16)
    return Image.fromarray(_LUT[np.clip(arr, 0, 10000)].astype(np.uint8))


def mask_to_color(mask_np, n_classes, palette):
    h, w  = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        color[mask_np == c] = palette[c]
    return color


# ============================================================================
# DATASET
# ============================================================================
class SegDataset(Dataset):
    def __init__(self, data_dir, img_h, img_w, has_labels=True):
        self.image_dir  = os.path.join(data_dir, 'Color_Images')
        self.masks_dir  = os.path.join(data_dir, 'Segmentation')
        self.has_labels = (has_labels and os.path.isdir(self.masks_dir))
        self.img_h, self.img_w = img_h, img_w
        self.data_ids   = sorted(os.listdir(self.image_dir))
        self.img_tf = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_tf = transforms.Resize(
            (img_h, img_w), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        name  = self.data_ids[idx]
        img   = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        img_t = self.img_tf(img)
        if self.has_labels:
            mask   = Image.open(os.path.join(self.masks_dir, name))
            mask   = convert_mask_fast(mask)
            mask   = self.mask_tf(mask)
            mask_t = (transforms.ToTensor()(mask) * 255).squeeze(0).long()
        else:
            mask_t = torch.full((self.img_h, self.img_w), -1, dtype=torch.long)
        return img_t, mask_t, name


# ============================================================================
# MODEL — must exactly match training architecture
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
    def __init__(self, in_ch, out_ch, rates=(2, 4, 6)):
        super().__init__()
        self.b0  = ConvBnGelu(in_ch, out_ch, k=1)
        self.b1  = ConvBnGelu(in_ch, out_ch, k=3, dilation=rates[0])
        self.b2  = ConvBnGelu(in_ch, out_ch, k=3, dilation=rates[1])
        self.b3  = ConvBnGelu(in_ch, out_ch, k=3, dilation=rates[2])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        self.fuse = ConvBnGelu(out_ch * 5, out_ch, k=1)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        h, w = x.shape[-2:]
        gap  = F.interpolate(self.gap(x), size=(h, w), mode='bilinear', align_corners=False)
        out  = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), gap], dim=1)
        return self.drop(self.fuse(out))


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, token_h, token_w, ch=256):
        super().__init__()
        self.H, self.W = token_h, token_w
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU()
        )
        self.aspp = ASPP(ch, ch, rates=(2, 4, 6))
        self.up1  = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 2, stride=2, bias=False),
            nn.BatchNorm2d(ch), nn.GELU()
        )
        self.res1   = ResBlock(ch)
        self.up2    = nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False),
            nn.BatchNorm2d(ch // 2), nn.GELU()
        )
        self.refine = nn.Sequential(
            ConvBnGelu(ch // 2, ch // 2, k=3),
            ConvBnGelu(ch // 2, ch // 2, k=3),
        )
        self.classifier = nn.Conv2d(ch // 2, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        x = self.aspp(x)
        x = self.up1(x);  x = self.res1(x)
        x = self.up2(x);  x = self.refine(x)
        return self.classifier(x)


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(ckpt_path, device):
    """Load backbone + head from a checkpoint. Returns (backbone, head, meta)."""
    ckpt = torch.load(ckpt_path, map_location=device)

    backbone_size = ckpt.get('backbone_size', 'base')
    n_embed       = ckpt.get('n_embed', 768)
    tok_h         = ckpt.get('token_h', 34)
    tok_w         = ckpt.get('token_w', 34)
    n_classes     = ckpt.get('n_classes', N_CLASSES_DEFAULT)
    img_h         = ckpt.get('img_h', int(((540 // 2) // 14) * 14))
    img_w         = ckpt.get('img_w', int(((960 // 2) // 14) * 14))
    class_names   = ckpt.get('class_names', CLASS_NAMES_DEFAULT[:n_classes])

    archs    = {"small": "vits14", "base": "vitb14_reg",
                "large": "vitl14_reg", "giant": "vitg14_reg"}
    backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{archs[backbone_size]}")

    # Restore fine-tuned last block if it was unfrozen during training
    if ckpt.get('backbone_unfrozen') and ckpt.get('backbone_state_dict') is not None:
        backbone.blocks[-1].load_state_dict(ckpt['backbone_state_dict'])
        print(f"  Restored fine-tuned last DINOv2 block from checkpoint")

    backbone.eval().to(device)

    head = SegmentationHead(n_embed, n_classes, tok_h, tok_w)
    state = ckpt.get('state_dict', ckpt)
    head.load_state_dict(state)
    head.eval().to(device)

    meta = dict(n_embed=n_embed, tok_h=tok_h, tok_w=tok_w, n_classes=n_classes,
                img_h=img_h, img_w=img_w, class_names=class_names,
                backbone_size=backbone_size)
    return backbone, head, meta


# ============================================================================
# INFERENCE — single forward pass, returns logits [B, C, H, W]
# ============================================================================
@torch.no_grad()
def _forward(backbone, head, imgs, out_h, out_w):
    with autocast():
        feats  = backbone.forward_features(imgs)["x_norm_patchtokens"]
        logits = head(feats)
        logits = F.interpolate(logits, size=(out_h, out_w),
                               mode='bilinear', align_corners=False)
    return logits.float()


# ============================================================================
# TEST TIME AUGMENTATION (TTA)
# ============================================================================
TTA_SCALES = [0.75, 1.0, 1.25]   # relative to training resolution

def tta_predict(backbone, head, imgs, img_h, img_w, device):
    """
    TTA strategy:
      - Original scale + hflip
      - 0.75× scale + hflip
      - 1.25× scale + hflip
    All 6 predictions are averaged in probability space.
    Returns averaged logits [B, C, img_h, img_w].
    """
    B = imgs.shape[0]
    accum = torch.zeros(B, head.classifier.out_channels, img_h, img_w,
                        device=device, dtype=torch.float32)
    count = 0

    for scale in TTA_SCALES:
        # Rescale to nearest multiple of 14 (DINOv2 patch size)
        sh = int(round(img_h * scale / 14) * 14)
        sw = int(round(img_w * scale / 14) * 14)
        sh, sw = max(sh, 14), max(sw, 14)

        imgs_s = F.interpolate(imgs, size=(sh, sw), mode='bilinear', align_corners=False)

        # Original orientation
        logits = _forward(backbone, head, imgs_s, img_h, img_w)
        accum += F.softmax(logits, dim=1)
        count += 1

        # Horizontal flip
        imgs_f  = torch.flip(imgs_s, dims=[3])
        logits_f = _forward(backbone, head, imgs_f, img_h, img_w)
        # Flip logits back so spatial positions align
        accum += F.softmax(torch.flip(logits_f, dims=[3]), dim=1)
        count += 1

    # Return log of averaged probabilities (consistent with argmax)
    return torch.log(accum / count + 1e-8)


# ============================================================================
# ENSEMBLE — average probabilities across multiple models
# ============================================================================
def ensemble_predict(models, imgs, img_h, img_w, device, use_tta=False):
    """
    Average softmax probabilities across all (backbone, head) pairs.
    If use_tta=True, each model also applies TTA.
    Returns averaged logits [B, C, img_h, img_w].
    """
    B         = imgs.shape[0]
    n_classes = models[0][1].classifier.out_channels
    accum     = torch.zeros(B, n_classes, img_h, img_w, device=device)

    for backbone, head in models:
        if use_tta:
            logits = tta_predict(backbone, head, imgs, img_h, img_w, device)
        else:
            logits = _forward(backbone, head, imgs, img_h, img_w)
        accum += F.softmax(logits, dim=1)

    return torch.log(accum / len(models) + 1e-8)


# ============================================================================
# METRICS
# ============================================================================
def compute_iou_per_class(pred_np, tgt_np, n_classes):
    ious = []
    for c in range(n_classes):
        p = pred_np == c;  t = tgt_np == c
        u = (p | t).sum()
        ious.append((p & t).sum() / u if u > 0 else float('nan'))
    return np.array(ious)


def compute_map_at_05(pred_np, tgt_np, n_classes):
    ious = compute_iou_per_class(pred_np, tgt_np, n_classes)
    aps  = [1.0 if iou >= 0.5 else 0.0 for iou in ious if not np.isnan(iou)]
    return (float(np.mean(aps)) if aps else float('nan')), ious


def compute_dice_per_class(pred_np, tgt_np, n_classes, smooth=1e-6):
    dice = []
    for c in range(n_classes):
        p = (pred_np == c).astype(float)
        t = (tgt_np  == c).astype(float)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice.append((2 * inter + smooth) / (denom + smooth) if denom > 0 else float('nan'))
    return np.array(dice)


# ============================================================================
# MAIN
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Segmentation evaluation — v3 (TTA + Ensemble)')
    # Single model OR ensemble
    parser.add_argument('--model_path', default=os.path.join(script_dir, 'best_segmentation_head.pth'),
                        help='Single model checkpoint (ignored if --ensemble is set)')
    parser.add_argument('--ensemble', nargs='+', default=None,
                        help='Two or more checkpoint paths to ensemble. '
                             'Example: --ensemble ckpt1.pth ckpt2.pth ckpt3.pth')
    parser.add_argument('--tta',        action='store_true', default=False,
                        help='Enable Test Time Augmentation (flip + multi-scale)')
    parser.add_argument('--data_dir',   default=os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages'))
    parser.add_argument('--output_dir', default=os.path.join(script_dir, 'predictions'))
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Lower if OOM (TTA/ensemble is memory-heavy)')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--has_labels', action='store_true', default=False,
                        help='Set if data_dir has Segmentation/ ground-truth folder')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, 'masks')
    color_dir  = os.path.join(args.output_dir, 'masks_color')
    comp_dir   = os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, color_dir, comp_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Determine checkpoint paths ────────────────────────────────────────────
    ckpt_paths = args.ensemble if args.ensemble else [args.model_path]
    print(f"\nMode : {'ENSEMBLE (' + str(len(ckpt_paths)) + ' models)' if len(ckpt_paths) > 1 else 'Single model'}"
          f" | TTA : {'ON' if args.tta else 'OFF'}")

    # ── Load models ───────────────────────────────────────────────────────────
    models   = []
    meta_ref = None   # use first checkpoint's meta as reference
    for i, cpath in enumerate(ckpt_paths):
        print(f"\nLoading checkpoint {i+1}/{len(ckpt_paths)}: {cpath}")
        backbone, head, meta = load_model(cpath, device)
        models.append((backbone, head))
        if meta_ref is None:
            meta_ref = meta
        print(f"  backbone={meta['backbone_size']}  n_classes={meta['n_classes']}  "
              f"img={meta['img_h']}×{meta['img_w']}")

    n_classes   = meta_ref['n_classes']
    img_h       = meta_ref['img_h']
    img_w       = meta_ref['img_w']
    class_names = meta_ref['class_names']
    palette     = COLOR_PALETTE[:n_classes]

    print(f"\nClasses ({n_classes}): {class_names}")
    if args.tta:
        n_aug = len(TTA_SCALES) * 2
        print(f"TTA: {len(TTA_SCALES)} scales × 2 (flip) = {n_aug} predictions per model")
        print(f"Total predictions per image: {n_aug * len(models)}")
        print("Tip: lower --batch_size if you get OOM with TTA enabled.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nLoading data from: {args.data_dir}  (has_labels={args.has_labels})")
    dataset = SegDataset(args.data_dir, img_h, img_w, has_labels=args.has_labels)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f"Samples: {len(dataset)}")

    # ── Inference loop ────────────────────────────────────────────────────────
    all_iou_batches  = []
    all_dice_batches = []
    all_pixel_acc    = []
    all_map05        = []
    sample_count     = 0

    print(f"\nRunning inference...")
    pbar = tqdm(loader, unit='batch')

    with torch.no_grad():
        for imgs, labels, names in pbar:
            imgs = imgs.to(device, non_blocking=True)

            # Predict with ensemble and/or TTA
            if len(models) > 1:
                logits = ensemble_predict(models, imgs, img_h, img_w, device, use_tta=args.tta)
            elif args.tta:
                logits = tta_predict(models[0][0], models[0][1], imgs, img_h, img_w, device)
            else:
                logits = _forward(models[0][0], models[0][1], imgs, img_h, img_w)

            preds = torch.argmax(logits, dim=1)  # [B, H, W]

            for i in range(imgs.shape[0]):
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                name    = names[i]
                base    = os.path.splitext(name)[0]

                # Save raw mask (class IDs)
                Image.fromarray(pred_np).save(os.path.join(masks_dir, f'{base}_pred.png'))

                # Save colorized mask
                pred_color = mask_to_color(pred_np, n_classes, palette)
                cv2.imwrite(os.path.join(color_dir, f'{base}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Metrics (only when ground truth exists)
                if args.has_labels and labels[i].min() >= 0:
                    tgt_np         = labels[i].cpu().numpy()
                    map05, iou_cls = compute_map_at_05(pred_np, tgt_np, n_classes)
                    all_map05.append(map05)
                    all_iou_batches.append(iou_cls)
                    all_dice_batches.append(compute_dice_per_class(pred_np, tgt_np, n_classes))
                    all_pixel_acc.append(float((pred_np == tgt_np).sum() / tgt_np.size))

                # Visual comparison
                if sample_count < args.num_samples:
                    img_np = imgs[i].cpu().numpy()
                    img_np = np.moveaxis(img_np, 0, -1)
                    img_np = img_np * np.array([0.229, 0.224, 0.225]) + \
                             np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                    if args.has_labels and labels[i].min() >= 0:
                        gt_color = mask_to_color(labels[i].cpu().numpy().astype(np.uint8),
                                                 n_classes, palette)
                        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                        axes[0].imshow(img_np);     axes[0].set_title('Input');       axes[0].axis('off')
                        axes[1].imshow(gt_color);   axes[1].set_title('Ground Truth'); axes[1].axis('off')
                        axes[2].imshow(pred_color); axes[2].set_title('Prediction');  axes[2].axis('off')
                    else:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        axes[0].imshow(img_np);     axes[0].set_title('Input');      axes[0].axis('off')
                        axes[1].imshow(pred_color); axes[1].set_title('Prediction'); axes[1].axis('off')

                    mode_tag = []
                    if len(models) > 1: mode_tag.append(f'Ensemble×{len(models)}')
                    if args.tta:        mode_tag.append('TTA')
                    plt.suptitle(f"{name}  [{', '.join(mode_tag) or 'single'}]", fontsize=9)
                    plt.tight_layout()
                    plt.savefig(os.path.join(comp_dir, f'sample_{sample_count:04d}.png'),
                                dpi=120, bbox_inches='tight')
                    plt.close()

                sample_count += 1

            if all_iou_batches:
                pbar.set_postfix(mIoU=f"{float(np.nanmean(all_iou_batches)):.3f}",
                                 mAP05=f"{float(np.nanmean(all_map05)):.3f}")

    # ── Aggregate & Print ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("EVALUATION RESULTS")
    print(f"{'='*65}")

    if all_iou_batches:
        class_iou  = np.nanmean(all_iou_batches,  axis=0)
        class_dice = np.nanmean(all_dice_batches, axis=0)
        mean_iou   = float(np.nanmean(class_iou))
        mean_dice  = float(np.nanmean(class_dice))
        mean_pacc  = float(np.mean(all_pixel_acc))
        mean_map05 = float(np.nanmean(all_map05))

        tta_str = " + TTA" if args.tta else ""
        ens_str = f" + Ensemble×{len(models)}" if len(models) > 1 else ""
        print(f"  Mode            : Single{ens_str}{tta_str}")
        print(f"  Mean IoU        : {mean_iou:.4f}")
        print(f"  Mean Dice       : {mean_dice:.4f}")
        print(f"  Pixel Accuracy  : {mean_pacc:.4f}")
        print(f"  mAP@0.5         : {mean_map05:.4f}   ← IoU≥0.5 counts as TP per class")
        print(f"\n  Per-Class Results:")
        print(f"  {'Class':<22} {'IoU':>8} {'Dice':>8} {'mAP@0.5':>10}")
        print(f"  {'-'*52}")
        for cname, ciou, cdice in zip(class_names, class_iou, class_dice):
            iou_s  = f"{ciou:.4f}"  if not np.isnan(ciou)  else "  N/A"
            dice_s = f"{cdice:.4f}" if not np.isnan(cdice) else "  N/A"
            ap_s   = "PASS" if (not np.isnan(ciou) and ciou >= 0.5) else "FAIL"
            zero   = " ← ZERO" if (not np.isnan(ciou) and ciou < 0.01) else ""
            print(f"  {cname:<22} {iou_s:>8} {dice_s:>8} {ap_s:>10}{zero}")

        # Save report
        report_path = os.path.join(args.output_dir, 'evaluation_metrics.txt')
        with open(report_path, 'w') as f:
            f.write("SEGMENTATION EVALUATION REPORT\n")
            f.write("=" * 65 + "\n")
            f.write(f"Checkpoints : {ckpt_paths}\n")
            f.write(f"Data        : {args.data_dir}\n")
            f.write(f"Mode        : Single{ens_str}{tta_str}\n")
            f.write(f"N Classes   : {n_classes}\n")
            f.write(f"N Samples   : {sample_count}\n\n")
            f.write(f"Mean IoU        : {mean_iou:.4f}\n")
            f.write(f"Mean Dice       : {mean_dice:.4f}\n")
            f.write(f"Pixel Accuracy  : {mean_pacc:.4f}\n")
            f.write(f"mAP@0.5         : {mean_map05:.4f}\n\n")
            f.write(f"{'Class':<22} {'IoU':>8} {'Dice':>8} {'mAP@0.5':>10}\n")
            f.write("-" * 52 + "\n")
            for cname, ciou, cdice in zip(class_names, class_iou, class_dice):
                iou_s  = f"{ciou:.4f}"  if not np.isnan(ciou)  else "   N/A"
                dice_s = f"{cdice:.4f}" if not np.isnan(cdice) else "   N/A"
                ap_s   = "    PASS" if (not np.isnan(ciou) and ciou >= 0.5) else "    FAIL"
                f.write(f"{cname:<22} {iou_s:>8} {dice_s:>8} {ap_s:>10}\n")
        print(f"\n  Report : {report_path}")

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        valid  = [v if not np.isnan(v) else 0 for v in class_iou]
        colors = ['#e74c3c' if v < 0.5 else '#2ecc71' for v in valid]
        ax.bar(range(n_classes), valid, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(mean_iou, color='blue', linestyle='--', label=f'mIoU={mean_iou:.4f}')
        ax.axhline(0.5, color='orange', linestyle=':', label='mAP@0.5 threshold')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('IoU')
        ax.set_title(f'Per-Class IoU — mIoU={mean_iou:.4f}  mAP@0.5={mean_map05:.4f}'
                     f'\n{ens_str.strip()}{tta_str}')
        ax.set_ylim(0, 1); ax.legend(); ax.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'per_class_metrics.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Chart  : {args.output_dir}/per_class_metrics.png")

    else:
        print("  No ground-truth labels found → metrics skipped.")
        print("  Re-run with --has_labels if ground truth is available.")

    print(f"\n{'='*65}")
    print(f"Images processed : {sample_count}")
    print(f"Outputs in       : {args.output_dir}/")


if __name__ == "__main__":
    main()