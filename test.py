"""
Evaluate / run inference for the offroad segmentation model.

    python test.py --weights runs/<ts>/best.pth --split val
    python test.py --weights runs/<ts>/best.pth --split test --tta
    python test.py --weights runs/<ts>/best.pth --split test --speed

Writes per-class IoU, confusion matrix, colorized masks, comparison + failure images.
"""

import os
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, NUM_CLASSES, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, PATCH_SIZE
from dataset import MaskDataset
from model import build_model
from metrics import ConfusionMatrix, mask_to_color


def denormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def round14(x):
    return max(PATCH_SIZE, int(round(x / PATCH_SIZE)) * PATCH_SIZE)


@torch.no_grad()
def predict_prob(model, imgs, device, amp_dtype, tta=False):
    """Return per-pixel class probabilities (B, C, H, W)."""
    H, W = imgs.shape[-2:]

    def fwd(x):
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            return F.softmax(model(x).float(), dim=1)

    prob = fwd(imgs)
    if tta:
        prob = prob + torch.flip(fwd(torch.flip(imgs, dims=[3])), dims=[3])
        n = 2
        for s in (0.75, 1.25):
            sh, sw = round14(H * s), round14(W * s)
            scaled = F.interpolate(imgs, size=(sh, sw), mode="bilinear", align_corners=False)
            p = fwd(scaled)
            prob = prob + F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
            n += 1
        prob = prob / n
    return prob


def per_image_miou(pred, gt):
    cm = ConfusionMatrix()
    cm.update(pred, gt)
    return cm.mean_iou()


def save_confusion(cm: ConfusionMatrix, path):
    mat = cm.mat.astype(np.float64)
    norm = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right"); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (row-normalized)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def save_per_class_bar(cm: ConfusionMatrix, path):
    iou = cm.iou_per_class()
    vals = np.nan_to_num(iou)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(NUM_CLASSES), vals,
           color=[np.array(c) / 255 for c in mask_to_color(np.arange(NUM_CLASSES))],
           edgecolor="black")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_ylabel("IoU"); ax.set_ylim(0, 1)
    ax.axhline(cm.mean_iou(), color="red", ls="--", label=f"mIoU {cm.mean_iou():.3f}")
    ax.set_title("Per-Class IoU"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def save_triptych(img, gt_color, pred_color, path, title):
    n = 3 if gt_color is not None else 2
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 5))
    ax[0].imshow(img); ax[0].set_title("Input"); ax[0].axis("off")
    k = 1
    if gt_color is not None:
        ax[k].imshow(gt_color); ax[k].set_title("Ground Truth"); ax[k].axis("off"); k += 1
    ax[k].imshow(pred_color); ax[k].set_title("Prediction"); ax[k].axis("off")
    plt.suptitle(title); plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--num-vis", type=int, default=8, help="comparison + failure images to save")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) \
        else (torch.float16 if device.type == "cuda" else None)

    ckpt = torch.load(args.weights, map_location=device)
    cfg = Config(**ckpt["config"]).validate()
    print(f"Loaded {args.weights} (backbone={cfg.backbone}, {cfg.img_h}x{cfg.img_w})")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = args.output_dir or os.path.join("predictions", args.split)
    masks_dir = os.path.join(out_dir, "masks")
    color_dir = os.path.join(out_dir, "masks_color")
    comp_dir = os.path.join(out_dir, "comparisons")
    fail_dir = os.path.join(out_dir, "failure_cases")
    for d in (masks_dir, color_dir, comp_dir, fail_dir):
        os.makedirs(d, exist_ok=True)

    dataset = MaskDataset(args.split, cfg.img_h, cfg.img_w, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"{args.split}: {len(dataset)} images | has_masks={dataset.has_masks} | TTA={args.tta}")

    cm = ConfusionMatrix()
    worst = []          # list of (miou, id, img, gt_color_or_None, pred_color)
    n_vis = 0
    t_total, n_imgs = 0.0, 0

    for imgs, masks, ids in tqdm(loader, desc="infer"):
        imgs = imgs.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        prob = predict_prob(model, imgs, device, amp_dtype, tta=args.tta)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total += time.time() - t0
        n_imgs += imgs.shape[0]

        preds = prob.argmax(1)                       # (B, H, W)
        if dataset.has_masks:
            cm.update(preds, masks.to(device))

        for b in range(imgs.shape[0]):
            data_id = ids[b]
            base = os.path.splitext(data_id)[0]
            pred = preds[b].cpu().numpy().astype(np.uint8)
            pred_color = mask_to_color(pred)
            cv2.imwrite(os.path.join(masks_dir, f"{base}.png"), pred)
            cv2.imwrite(os.path.join(color_dir, f"{base}.png"),
                        cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

            img_rgb = denormalize(imgs[b])
            gt_color = None
            if dataset.has_masks:
                gt = masks[b].numpy().astype(np.uint8)
                gt_color = mask_to_color(gt)
                im = per_image_miou(pred, gt)
                worst.append((im, data_id, img_rgb, gt_color, pred_color))
                worst = sorted(worst, key=lambda x: x[0])[:args.num_vis]

            if n_vis < args.num_vis:
                save_triptych(img_rgb, gt_color, pred_color,
                              os.path.join(comp_dir, f"{base}.png"), f"{data_id}")
                n_vis += 1

    # metrics
    if dataset.has_masks:
        summary = cm.summary_str()
        print("\n" + summary)
        with open(os.path.join(out_dir, "evaluation_metrics.txt"), "w") as f:
            f.write(summary + "\n")
            if args.speed:
                f.write(f"\nInference: {1000*t_total/max(1,n_imgs):.2f} ms/image "
                        f"(batch {args.batch_size}, TTA={args.tta})\n")
        save_per_class_bar(cm, os.path.join(out_dir, "per_class_iou.png"))
        save_confusion(cm, os.path.join(out_dir, "confusion_matrix.png"))
        for rank, (im, data_id, img_rgb, gt_color, pred_color) in enumerate(worst):
            save_triptych(img_rgb, gt_color, pred_color,
                          os.path.join(fail_dir, f"{rank:02d}_iou{im:.3f}_{os.path.splitext(data_id)[0]}.png"),
                          f"{data_id}  (image mIoU={im:.3f})")
    else:
        print("\nNo ground-truth masks for this split; saved predictions only.")

    if args.speed:
        print(f"Inference speed: {1000*t_total/max(1,n_imgs):.2f} ms/image "
              f"(batch {args.batch_size}, TTA={args.tta})")
    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
