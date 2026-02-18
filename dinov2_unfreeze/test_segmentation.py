"""
Optimized Segmentation Test/Evaluation Script — Hackathon Edition
- Loads best checkpoint (auto-detects config)
- Computes: mIoU, per-class IoU, mAP@0.5, Dice, Pixel Accuracy
- Saves: colored masks, comparison images, metrics report
- Includes Flowers class (600) — 11 classes total
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast

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
    600:   6,   # Flowers  ← FIXED
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}
N_CLASSES_DEFAULT = len(VALUE_MAP)  # 11

CLASS_NAMES_DEFAULT = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

COLOR_PALETTE = np.array([
    [0,   0,   0  ],   # Background  - black
    [34,  139, 34 ],   # Trees       - forest green
    [0,   200, 0  ],   # Lush Bushes - lime
    [210, 180, 140],   # Dry Grass   - tan
    [139, 90,  43 ],   # Dry Bushes  - brown
    [128, 128, 0  ],   # Ground Clutter - olive
    [255, 200, 0  ],   # Flowers     - yellow
    [139, 69,  19 ],   # Logs        - saddle brown
    [128, 128, 128],   # Rocks       - gray
    [160, 82,  45 ],   # Landscape   - sienna
    [135, 206, 235],   # Sky         - sky blue
], dtype=np.uint8)

# Fast LUT
_LUT = np.zeros(10001, dtype=np.uint8)
for raw, cls in VALUE_MAP.items():
    if raw <= 10000:
        _LUT[raw] = cls


def convert_mask_fast(mask_pil):
    arr = np.array(mask_pil, dtype=np.uint16)
    arr = np.clip(arr, 0, 10000)
    return Image.fromarray(_LUT[arr].astype(np.uint8))


def mask_to_color(mask_np, n_classes, palette):
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        color[mask_np == c] = palette[c]
    return color


# ============================================================================
# DATASET — handles both labeled (val) and unlabeled (testImages) sets
# ============================================================================
class SegDataset(Dataset):
    """Works with or without Segmentation ground-truth folder."""

    def __init__(self, data_dir, img_h, img_w, has_labels=True):
        self.image_dir  = os.path.join(data_dir, 'Color_Images')
        self.masks_dir  = os.path.join(data_dir, 'Segmentation') if has_labels else None
        self.has_labels = has_labels and (self.masks_dir is not None) and os.path.isdir(self.masks_dir or '')
        self.img_h, self.img_w = img_h, img_w
        self.data_ids   = sorted(os.listdir(self.image_dir))

        self.img_tf = transforms.Compose([
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_tf = transforms.Resize((img_h, img_w), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        name = self.data_ids[idx]
        img  = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        img_t = self.img_tf(img)

        if self.has_labels:
            mask = Image.open(os.path.join(self.masks_dir, name))
            mask = convert_mask_fast(mask)
            mask = self.mask_tf(mask)
            mask_t = (transforms.ToTensor()(mask) * 255).squeeze(0).long()
        else:
            mask_t = torch.zeros(self.img_h, self.img_w, dtype=torch.long) - 1  # sentinel

        return img_t, mask_t, name


# ============================================================================
# MODEL — must match training architecture
# ============================================================================
class ConvBnGelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, groups=1, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2, groups=groups, bias=bias),
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


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, token_h, token_w, ch=256):
        super().__init__()
        self.H, self.W = token_h, token_w

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU()
        )
        self.res1 = ResBlock(ch)
        self.res2 = ResBlock(ch)
        self.up1  = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 2, stride=2, bias=False),
            nn.BatchNorm2d(ch), nn.GELU()
        )
        self.res3 = ResBlock(ch)
        self.up2  = nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False),
            nn.BatchNorm2d(ch // 2), nn.GELU()
        )
        self.res4 = nn.Sequential(
            ConvBnGelu(ch // 2, ch // 2, k=3),
            ConvBnGelu(ch // 2, ch // 2, k=3),
        )
        self.classifier = nn.Conv2d(ch // 2, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        x = self.res1(x); x = self.res2(x)
        x = self.up1(x);  x = self.res3(x)
        x = self.up2(x);  x = self.res4(x)
        return self.classifier(x)


# ============================================================================
# METRICS
# ============================================================================
def compute_iou_per_class(pred_np, tgt_np, n_classes):
    ious = []
    for c in range(n_classes):
        p = pred_np == c; t = tgt_np == c
        u = (p | t).sum()
        ious.append((p & t).sum() / u if u > 0 else float('nan'))
    return np.array(ious)


def compute_map_at_05(pred_np, tgt_np, n_classes):
    """
    mAP@0.5 for segmentation:
    For each class, treat IoU >= 0.5 as a True Positive (AP = 1), else 0.
    Returns mean over classes present in ground truth.
    """
    ious = compute_iou_per_class(pred_np, tgt_np, n_classes)
    aps  = []
    for iou in ious:
        if np.isnan(iou):
            continue  # class not present
        aps.append(1.0 if iou >= 0.5 else 0.0)
    return float(np.mean(aps)) if aps else float('nan'), ious


def compute_dice_per_class(pred_np, tgt_np, n_classes, smooth=1e-6):
    dice = []
    for c in range(n_classes):
        p = (pred_np == c).astype(float)
        t = (tgt_np  == c).astype(float)
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice.append((2 * inter + smooth) / (denom + smooth) if denom > 0 else float('nan'))
    return np.array(dice)


def pixel_accuracy(pred_np, tgt_np):
    return float((pred_np == tgt_np).sum() / tgt_np.size)


# ============================================================================
# MAIN
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Segmentation evaluation — hackathon edition')
    parser.add_argument('--model_path',  default=os.path.join(script_dir, 'best_segmentation_head.pth'))
    parser.add_argument('--data_dir',    default=os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages'))
    parser.add_argument('--output_dir',  default=os.path.join(script_dir, 'predictions'))
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of visual comparison saves')
    parser.add_argument('--has_labels',  action='store_true', default=False,
                        help='Set if test dir has Segmentation/ ground truth')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir  = os.path.join(args.output_dir, 'masks')
    color_dir  = os.path.join(args.output_dir, 'masks_color')
    comp_dir   = os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, color_dir, comp_dir]:
        os.makedirs(d, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)

    # Read config from checkpoint (graceful fallback to defaults)
    backbone_size = ckpt.get('backbone_size', 'base')
    n_embed       = ckpt.get('n_embed', 768)
    tok_h         = ckpt.get('token_h', 34)
    tok_w         = ckpt.get('token_w', 34)
    n_classes     = ckpt.get('n_classes', N_CLASSES_DEFAULT)
    img_h         = ckpt.get('img_h', int(((540 // 2) // 14) * 14))
    img_w         = ckpt.get('img_w', int(((960 // 2) // 14) * 14))
    class_names   = ckpt.get('class_names', CLASS_NAMES_DEFAULT[:n_classes])

    print(f"Config → backbone={backbone_size}, n_classes={n_classes}, "
          f"img={img_h}×{img_w}, tokens={tok_h}×{tok_w}")
    print(f"Classes: {class_names}")

    # Backbone
    archs = {"small": "vits14", "base": "vitb14_reg",
             "large": "vitl14_reg", "giant": "vitg14_reg"}
    print(f"Loading DINOv2-{backbone_size}...")
    backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{archs[backbone_size]}")
    backbone.eval().to(device)

    # Head
    head = SegmentationHead(n_embed, n_classes, tok_h, tok_w)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    head.load_state_dict(state)
    head.eval().to(device)
    print("Model loaded.")

    # Dataset
    print(f"Loading data from {args.data_dir}  (has_labels={args.has_labels})")
    dataset = SegDataset(args.data_dir, img_h, img_w, has_labels=args.has_labels)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f"Samples: {len(dataset)}")

    # Evaluation loop
    all_iou_batches  = []
    all_dice_batches = []
    all_pixel_acc    = []
    all_map05        = []
    sample_count     = 0

    print(f"\nRunning inference on {len(dataset)} images...")
    pbar = tqdm(loader, unit='batch')

    with torch.no_grad():
        for imgs, labels, names in pbar:
            imgs = imgs.to(device, non_blocking=True)

            with autocast():
                feats  = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = head(feats)
                logits = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)

            preds = torch.argmax(logits, dim=1)  # [B, H, W]

            for i in range(imgs.shape[0]):
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                name    = names[i]
                base    = os.path.splitext(name)[0]

                # Save raw mask
                Image.fromarray(pred_np).save(os.path.join(masks_dir, f'{base}_pred.png'))

                # Save colored mask
                pred_color = mask_to_color(pred_np, n_classes, COLOR_PALETTE[:n_classes])
                cv2.imwrite(
                    os.path.join(color_dir, f'{base}_pred_color.png'),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
                )

                # Metrics (only if labels available)
                if args.has_labels and labels[i].min() >= 0:
                    tgt_np = labels[i].cpu().numpy()
                    miou_img, iou_cls = compute_map_at_05(pred_np, tgt_np, n_classes)
                    all_map05.append(miou_img)
                    all_iou_batches.append(iou_cls)
                    all_dice_batches.append(compute_dice_per_class(pred_np, tgt_np, n_classes))
                    all_pixel_acc.append(pixel_accuracy(pred_np, tgt_np))

                # Save visual comparison
                if sample_count < args.num_samples:
                    # Denormalize image
                    img_np = imgs[i].cpu().numpy()
                    img_np = np.moveaxis(img_np, 0, -1)
                    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                    if args.has_labels and labels[i].min() >= 0:
                        gt_np    = labels[i].cpu().numpy().astype(np.uint8)
                        gt_color = mask_to_color(gt_np, n_classes, COLOR_PALETTE[:n_classes])
                        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                        axes[0].imshow(img_np);     axes[0].set_title('Input');      axes[0].axis('off')
                        axes[1].imshow(gt_color);   axes[1].set_title('Ground Truth'); axes[1].axis('off')
                        axes[2].imshow(pred_color); axes[2].set_title('Prediction'); axes[2].axis('off')
                    else:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        axes[0].imshow(img_np);     axes[0].set_title('Input');      axes[0].axis('off')
                        axes[1].imshow(pred_color); axes[1].set_title('Prediction'); axes[1].axis('off')

                    plt.suptitle(name, fontsize=9)
                    plt.tight_layout()
                    plt.savefig(os.path.join(comp_dir, f'sample_{sample_count:04d}.png'),
                                dpi=120, bbox_inches='tight')
                    plt.close()

                sample_count += 1

            if all_iou_batches:
                pbar.set_postfix(mIoU=f"{float(np.nanmean(all_iou_batches)):.3f}",
                                 mAP05=f"{float(np.nanmean(all_map05)):.3f}")

    # ---- Aggregate & Print ----
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

        print(f"  Mean IoU         : {mean_iou:.4f}")
        print(f"  Mean Dice        : {mean_dice:.4f}")
        print(f"  Pixel Accuracy   : {mean_pacc:.4f}")
        print(f"  mAP@0.5          : {mean_map05:.4f}   ← IoU≥0.5 counts as TP per class")
        print(f"\n  Per-Class IoU:")
        for cname, ciou, cdice in zip(class_names, class_iou, class_dice):
            iou_s  = f"{ciou:.4f}"  if not np.isnan(ciou)  else "  N/A "
            dice_s = f"{cdice:.4f}" if not np.isnan(cdice) else "  N/A "
            ap_s   = "PASS" if (not np.isnan(ciou) and ciou >= 0.5) else "FAIL"
            print(f"    {cname:<20}: IoU={iou_s}  Dice={dice_s}  mAP@0.5={ap_s}")

        # Save metrics to file
        report_path = os.path.join(args.output_dir, 'evaluation_metrics.txt')
        with open(report_path, 'w') as f:
            f.write("SEGMENTATION EVALUATION REPORT\n")
            f.write("=" * 65 + "\n")
            f.write(f"Model      : {args.model_path}\n")
            f.write(f"Data       : {args.data_dir}\n")
            f.write(f"N Classes  : {n_classes}\n")
            f.write(f"N Samples  : {sample_count}\n\n")
            f.write(f"Mean IoU         : {mean_iou:.4f}\n")
            f.write(f"Mean Dice        : {mean_dice:.4f}\n")
            f.write(f"Pixel Accuracy   : {mean_pacc:.4f}\n")
            f.write(f"mAP@0.5          : {mean_map05:.4f}\n\n")
            f.write("Per-Class Results:\n")
            f.write(f"{'Class':<22} {'IoU':>8} {'Dice':>8} {'mAP@0.5':>10}\n")
            f.write("-" * 55 + "\n")
            for cname, ciou, cdice in zip(class_names, class_iou, class_dice):
                iou_s  = f"{ciou:.4f}"  if not np.isnan(ciou)  else "   N/A"
                dice_s = f"{cdice:.4f}" if not np.isnan(cdice) else "   N/A"
                ap_s   = "    PASS" if (not np.isnan(ciou) and ciou >= 0.5) else "    FAIL"
                f.write(f"{cname:<22} {iou_s:>8} {dice_s:>8} {ap_s:>10}\n")
        print(f"\n  Report saved: {report_path}")

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        valid = [v if not np.isnan(v) else 0 for v in class_iou]
        colors = [COLOR_PALETTE[i] / 255 for i in range(min(n_classes, len(COLOR_PALETTE)))]
        ax.bar(range(n_classes), valid, color=colors, edgecolor='black')
        ax.axhline(y=mean_iou, color='red', linestyle='--', label=f'Mean={mean_iou:.4f}')
        ax.axhline(y=0.5, color='orange', linestyle=':', label='mAP@0.5 threshold')
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('IoU')
        ax.set_title(f'Per-Class IoU  (mIoU={mean_iou:.4f}  mAP@0.5={mean_map05:.4f})')
        ax.set_ylim(0, 1); ax.legend(); ax.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Chart saved : {args.output_dir}/per_class_metrics.png")

    else:
        print("  No ground-truth labels → skipping metric computation.")
        print("  Predictions saved to masks/ and masks_color/")
        print("  Re-run with --has_labels flag if ground truth is available.")

    print(f"\n{'='*65}")
    print(f"Total images processed : {sample_count}")
    print(f"Outputs in             : {args.output_dir}/")
    print(f"  masks/          — raw class-ID PNGs")
    print(f"  masks_color/    — RGB visualizations")
    print(f"  comparisons/    — side-by-side samples ({args.num_samples})")


if __name__ == "__main__":
    main()