"""
Optimized Segmentation Training Script — Hackathon Edition
- Fixes: Flowers class (600) added → 11 classes total
- DINOv2-base backbone (768d, higher capacity)
- Deep FPN-style head (256 channels, residual blocks)
- AdamW + OneCycleLR scheduler
- Mixed precision (AMP) + gradient accumulation
- Combined CE + Dice loss with label smoothing
- Joint augmentations
- Best checkpoint saving by val mIoU
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# ============================================================================
# CONFIG
# ============================================================================
BACKBONE_SIZE = "base"          # small=384d | base=768d | large=1024d
BATCH_SIZE    = 4               # per GPU step
ACCUM_STEPS   = 2               # effective batch = BATCH_SIZE * ACCUM_STEPS
LR            = 3e-4
N_EPOCHS      = 20
IMG_H         = int(((540 // 2) // 14) * 14)   # 476
IMG_W         = int(((960 // 2) // 14) * 14)   # 476
HEAD_CHANNELS = 256
NUM_WORKERS   = 4
PIN_MEMORY    = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
VAL_DIR    = os.path.join(SCRIPT_DIR, '..', 'Offroad_Segmentation_Training_Dataset', 'val')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'train_stats')
BEST_MODEL = os.path.join(SCRIPT_DIR, 'best_segmentation_head.pth')
LAST_MODEL = os.path.join(SCRIPT_DIR, 'last_segmentation_head.pth')

# ============================================================================
# CLASS MAPPING — 11 classes including Flowers (600) — FIXED
# ============================================================================
VALUE_MAP = {
    0:     0,   # Background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers  ← was MISSING in original script
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

# Precomputed LUT for O(1) mask conversion
_LUT = np.zeros(10001, dtype=np.uint8)
for raw, cls in VALUE_MAP.items():
    if raw <= 10000:
        _LUT[raw] = cls


def convert_mask_fast(mask_pil):
    arr = np.array(mask_pil, dtype=np.uint16)
    arr = np.clip(arr, 0, 10000)
    return Image.fromarray(_LUT[arr].astype(np.uint8))


# ============================================================================
# AUGMENTATION
# ============================================================================
class JointAugment:
    def __call__(self, img, mask):
        # Horizontal flip
        if random.random() < 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # Vertical flip
        if random.random() < 0.2:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        # Random rotation ±20°
        if random.random() < 0.4:
            angle = random.uniform(-20, 20)
            img  = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

        # Color jitter (image only)
        if random.random() < 0.6:
            img = transforms.ColorJitter(brightness=0.35, contrast=0.35,
                                         saturation=0.3, hue=0.08)(img)

        # Random grayscale (image only)
        if random.random() < 0.1:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        # Random resized crop
        if random.random() < 0.4:
            scale = random.uniform(0.7, 1.0)
            ch = int(img.height * scale)
            cw = int(img.width  * scale)
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(ch, cw))
            img  = TF.resized_crop(img,  i, j, h, w, (IMG_H, IMG_W), antialias=True)
            mask = TF.resized_crop(mask, i, j, h, w, (IMG_H, IMG_W),
                                   interpolation=TF.InterpolationMode.NEAREST)

        # Ensure final size matches training resolution (guards when other
        # augmentations don't change spatial dims). Always resize image and
        # nearest-resize the mask to avoid interpolation artifacts.
        img = transforms.Resize((IMG_H, IMG_W))(img)
        mask = transforms.Resize((IMG_H, IMG_W), interpolation=transforms.InterpolationMode.NEAREST)(mask)

        return img, mask


# ============================================================================
# DATASET
# ============================================================================
IMG_NORM = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
MASK_RESIZE = transforms.Resize((IMG_H, IMG_W), interpolation=transforms.InterpolationMode.NEAREST)


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
            mask = MASK_RESIZE(mask)

        img_t  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])(img)
        mask_t = (transforms.ToTensor()(mask) * 255).squeeze(0).long()
        return img_t, mask_t


# ============================================================================
# MODEL — Deep segmentation head with residual blocks
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
    def __init__(self, in_channels, out_channels, token_h, token_w, ch=HEAD_CHANNELS):
        super().__init__()
        self.H, self.W = token_h, token_w

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU()
        )
        self.res1 = ResBlock(ch)
        self.res2 = ResBlock(ch)

        # Upsample ×2
        self.up1  = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 2, stride=2, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU()
        )
        self.res3 = ResBlock(ch)

        # Upsample ×2 again
        self.up2  = nn.Sequential(
            nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.GELU()
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
        x = self.res1(x)
        x = self.res2(x)
        x = self.up1(x)
        x = self.res3(x)
        x = self.up2(x)
        x = self.res4(x)
        return self.classifier(x)


# ============================================================================
# LOSS
# ============================================================================
class DiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1e-6):
        super().__init__()
        self.n = n_classes
        self.s = smooth

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
                loss += 1 - (2 * inter + self.s) / (denom + self.s)
                cnt  += 1
        return loss / max(cnt, 1)


class CombinedLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.dice = DiceLoss(n_classes)

    def forward(self, logits, targets):
        return 0.6 * self.ce(logits, targets) + 0.4 * self.dice(logits, targets)


# ============================================================================
# METRICS
# ============================================================================
def batch_iou(logits, targets, n_classes):
    pred = torch.argmax(logits, 1).view(-1).cpu().numpy()
    tgt  = targets.view(-1).cpu().numpy()
    ious = []
    for c in range(n_classes):
        p = pred == c; t = tgt == c
        u = (p | t).sum()
        ious.append((p & t).sum() / u if u > 0 else float('nan'))
    return np.array(ious)


@torch.no_grad()
def evaluate(head, backbone, loader, device, n_classes):
    head.eval()
    all_iou = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast():
            feats  = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = head(feats)
            logits = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
        all_iou.append(batch_iou(logits.float(), labels, n_classes))
    head.train()
    class_iou = np.nanmean(all_iou, axis=0)
    return float(np.nanmean(class_iou)), class_iou


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

    # Data
    trainset = MaskDataset(TRAIN_DIR, augment=True)
    valset   = MaskDataset(VAL_DIR,   augment=False)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True)
    val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    print(f"Train: {len(trainset)} | Val: {len(valset)} | Classes: {N_CLASSES}")
    print(f"Classes: {CLASS_NAMES}")

    # Backbone (frozen)
    print(f"\nLoading DINOv2-{BACKBONE_SIZE}...")
    archs = {"small": "vits14", "base": "vitb14_reg",
             "large": "vitl14_reg", "giant": "vitg14_reg"}
    backbone = torch.hub.load("facebookresearch/dinov2", f"dinov2_{archs[BACKBONE_SIZE]}")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    print("Backbone loaded & frozen.")

    # Probe dims
    with torch.no_grad():
        s_img, _ = trainset[0]
        feat = backbone.forward_features(s_img.unsqueeze(0).to(device))["x_norm_patchtokens"]
    n_embed = feat.shape[2]
    tok_h, tok_w = IMG_H // 14, IMG_W // 14
    print(f"Embed: {n_embed} | Tokens: {tok_h}×{tok_w} | Head upsamples to {tok_h*4}×{tok_w*4}")

    # Head
    head = SegmentationHead(n_embed, N_CLASSES, tok_h, tok_w).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Head params: {n_params/1e6:.2f}M")

    # Optim
    criterion = CombinedLoss(N_CLASSES)
    optimizer = optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_loader) // ACCUM_STEPS)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=steps_per_epoch,
        epochs=N_EPOCHS,
        pct_start=0.1, div_factor=10, final_div_factor=1000
    )
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    best_iou = 0.0
    best_class_iou = np.zeros(N_CLASSES)

    print(f"\nStarting {N_EPOCHS} epochs | EffBatch={BATCH_SIZE*ACCUM_STEPS} | AMP=ON")
    print("=" * 70)

    for epoch in range(N_EPOCHS):
        head.train()
        t_losses, t_ious = [], []
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{N_EPOCHS} [Train]", leave=False)
        for step, (imgs, labels) in enumerate(pbar):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                with torch.no_grad():
                    feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = head(feats)
                logits = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                loss   = criterion(logits, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            t_losses.append(loss.item() * ACCUM_STEPS)
            ious = batch_iou(logits.float().detach(), labels, N_CLASSES)
            t_ious.append(float(np.nanmean(ious)))
            pbar.set_postfix(loss=f"{t_losses[-1]:.4f}", iou=f"{t_ious[-1]:.3f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # Val loss
        head.eval()
        v_losses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast():
                    feats  = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    logits = head(feats)
                    logits = F.interpolate(logits, size=imgs.shape[2:], mode='bilinear', align_corners=False)
                    v_losses.append(criterion(logits.float(), labels).item())
        head.train()

        val_miou, class_iou = evaluate(head, backbone, val_loader, device, N_CLASSES)

        train_loss = float(np.mean(t_losses))
        train_miou = float(np.mean(t_ious))
        val_loss   = float(np.mean(v_losses))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_miou)
        history['val_iou'].append(val_miou)

        print(f"\nEp {epoch+1:02d} | loss {train_loss:.4f}/{val_loss:.4f} | "
              f"mIoU {train_miou:.4f}/{val_miou:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

        # Per-class breakdown
        for cname, ciou in zip(CLASS_NAMES, class_iou):
            flag = " ←" if np.isnan(ciou) else ""
            iou_s = f"{ciou:.4f}" if not np.isnan(ciou) else "N/A "
            print(f"  {cname:<20} : {iou_s}{flag}")

        # Save best
        if val_miou > best_iou:
            best_iou = val_miou
            best_class_iou = class_iou.copy()
            torch.save({
                'epoch': epoch + 1,
                'state_dict': head.state_dict(),
                'val_iou': val_miou,
                'class_iou': class_iou.tolist(),
                'n_embed': n_embed,
                'token_h': tok_h,
                'token_w': tok_w,
                'n_classes': N_CLASSES,
                'backbone_size': BACKBONE_SIZE,
                'class_names': CLASS_NAMES,
                'img_h': IMG_H,
                'img_w': IMG_W,
            }, BEST_MODEL)
            print(f"  ★ BEST saved  mIoU={best_iou:.4f}")

    # Save last
    torch.save({
        'state_dict': head.state_dict(),
        'n_embed': n_embed, 'token_h': tok_h, 'token_w': tok_w,
        'n_classes': N_CLASSES, 'backbone_size': BACKBONE_SIZE,
        'class_names': CLASS_NAMES, 'img_h': IMG_H, 'img_w': IMG_W,
    }, LAST_MODEL)

    # Plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].plot(history['train_loss'], label='train'); axes[0].plot(history['val_loss'], label='val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history['train_iou'], label='train'); axes[1].plot(history['val_iou'], label='val')
    axes[1].set_title('mIoU'); axes[1].legend(); axes[1].grid(True)
    valid_iou = [v if not np.isnan(v) else 0 for v in best_class_iou]
    axes[2].bar(range(N_CLASSES), valid_iou)
    axes[2].set_xticks(range(N_CLASSES))
    axes[2].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
    axes[2].set_title(f'Per-Class IoU (Best Val mIoU={best_iou:.4f})')
    axes[2].grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*70}")
    print(f"Done! Best Val mIoU: {best_iou:.4f}")
    print(f"Best model : {BEST_MODEL}")
    print(f"Last model : {LAST_MODEL}")
    print(f"Plots      : {OUTPUT_DIR}/training_curves.png")


if __name__ == "__main__":
    main()