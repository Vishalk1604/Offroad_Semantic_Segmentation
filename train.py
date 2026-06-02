"""
Train DINOv2 (frozen) + Rein adapters + DPT head for offroad semantic segmentation.

Trains ONLY on train/ + val/. Test images are never touched here.

Example:
    python train.py --backbone base --epochs 50 --batch-size 4 --img-h 378 --img-w 672
"""

import os
import json
import math
import time
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_default_config, Config
from dataset import MaskDataset
from model import build_model
from losses import CombinedLoss, get_class_weights
from metrics import ConfusionMatrix


# ---------------------------------------------------------------------------
# EMA over trainable parameters only (keeps the frozen backbone un-duplicated)
# ---------------------------------------------------------------------------
class ParamEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def copy_to(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, amp_dtype):
    model.eval()
    cm = ConfusionMatrix()
    losses = []
    for imgs, masks, _ in tqdm(loader, desc="val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(imgs)
            loss = loss_fn(logits, masks)
        losses.append(loss.item())
        cm.update(logits.argmax(1), masks)
    return float(np.mean(losses)), cm


def save_curves(history, out_dir):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax[0, 0].plot(ep, history["train_loss"], label="train")
    ax[0, 0].plot(ep, history["val_loss"], label="val")
    ax[0, 0].set_title("Loss"); ax[0, 0].set_xlabel("epoch"); ax[0, 0].legend(); ax[0, 0].grid(True)
    ax[0, 1].plot(ep, history["val_miou"], color="tab:green")
    ax[0, 1].set_title("Validation mIoU"); ax[0, 1].set_xlabel("epoch"); ax[0, 1].grid(True)
    ax[1, 0].plot(ep, history["val_dice"], color="tab:orange")
    ax[1, 0].set_title("Validation Dice"); ax[1, 0].set_xlabel("epoch"); ax[1, 0].grid(True)
    ax[1, 1].plot(ep, history["val_acc"], color="tab:purple")
    ax[1, 1].set_title("Validation Pixel Accuracy"); ax[1, 1].set_xlabel("epoch"); ax[1, 1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130)
    plt.close()


def parse_args(cfg: Config):
    p = argparse.ArgumentParser(description="Train offroad segmentation model")
    p.add_argument("--backbone", default=cfg.backbone, choices=["small", "base", "large"])
    p.add_argument("--img-h", type=int, default=cfg.img_h)
    p.add_argument("--img-w", type=int, default=cfg.img_w)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--grad-accum", type=int, default=cfg.grad_accum)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    p.add_argument("--warmup-epochs", type=int, default=cfg.warmup_epochs)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--no-rein", action="store_true")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=cfg.out_dir)
    return p.parse_args()


def main():
    cfg = get_default_config()
    args = parse_args(cfg)
    cfg = Config(
        backbone=args.backbone, img_h=args.img_h, img_w=args.img_w, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr,
        weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers, use_rein=not args.no_rein, use_ema=not args.no_ema,
        use_class_weights=not args.no_class_weights, amp=not args.no_amp, seed=args.seed,
        out_dir=args.out_dir,
    ).validate()

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if cfg.amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Device: {device} | AMP: {amp_dtype}")

    run_dir = os.path.join(cfg.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # data
    train_set = MaskDataset("train", cfg.img_h, cfg.img_w, augment=True)
    val_set = MaskDataset("val", cfg.img_h, cfg.img_w, augment=False)
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # model
    model = build_model(cfg).to(device)

    # loss
    class_weights = None
    if cfg.use_class_weights:
        class_weights = get_class_weights().to(device)
        print("Class weights:", np.round(class_weights.cpu().numpy(), 3))
    loss_fn = CombinedLoss(class_weights=class_weights,
                           ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight)

    # optimizer + cosine schedule with warmup (per optimizer-step)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ema = ParamEMA(model, cfg.ema_decay) if cfg.use_ema else None
    start_epoch = 0
    history = {k: [] for k in ["train_loss", "val_loss", "val_miou", "val_dice", "val_acc"]}
    best_miou = -1.0
    epochs_no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    print("\nStarting training...\n" + "=" * 70)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running = []
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for i, (imgs, masks, _) in enumerate(pbar):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model(imgs)
                loss = loss_fn(logits, masks) / cfg.grad_accum
            loss.backward()
            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema:
                    ema.update(model)
            running.append(loss.item() * cfg.grad_accum)
            pbar.set_postfix(loss=f"{np.mean(running[-50:]):.3f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # validate with EMA weights
        backup = ema.copy_to(model) if ema else None
        val_loss, cm = evaluate(model, val_loader, device, loss_fn, amp_dtype)
        if ema:
            ema.restore(model, backup)

        miou, dice, acc = cm.mean_iou(), cm.mean_dice(), cm.pixel_accuracy()
        history["train_loss"].append(float(np.mean(running)))
        history["val_loss"].append(val_loss)
        history["val_miou"].append(miou)
        history["val_dice"].append(dice)
        history["val_acc"].append(acc)
        print(f"Epoch {epoch+1}: train_loss={np.mean(running):.4f} val_loss={val_loss:.4f} "
              f"val_mIoU={miou:.4f} val_Dice={dice:.4f} val_acc={acc:.4f}")

        # checkpoint best (save EMA weights into the full state_dict)
        if miou > best_miou:
            best_miou = miou
            epochs_no_improve = 0
            backup = ema.copy_to(model) if ema else None
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                        "val_miou": miou, "epoch": epoch + 1},
                       os.path.join(run_dir, "best.pth"))
            if ema:
                ema.restore(model, backup)
            with open(os.path.join(run_dir, "per_class_iou_best.txt"), "w") as f:
                f.write(cm.summary_str())
            print(f"  ** new best mIoU {miou:.4f} -> saved best.pth")
        else:
            epochs_no_improve += 1

        save_curves(history, run_dir)
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
                   os.path.join(run_dir, "last.pth"))

        if epochs_no_improve >= cfg.early_stop_patience:
            print(f"Early stopping (no val mIoU improvement for {cfg.early_stop_patience} epochs).")
            break

    # final history dump
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(f"Best val mIoU: {best_miou:.4f}\n\n")
        f.write("epoch\ttrain_loss\tval_loss\tval_mIoU\tval_Dice\tval_acc\n")
        for i in range(len(history["train_loss"])):
            f.write(f"{i+1}\t{history['train_loss'][i]:.4f}\t{history['val_loss'][i]:.4f}\t"
                    f"{history['val_miou'][i]:.4f}\t{history['val_dice'][i]:.4f}\t"
                    f"{history['val_acc'][i]:.4f}\n")
    print(f"\nDone. Best val mIoU: {best_miou:.4f}\nOutputs in: {run_dir}")


if __name__ == "__main__":
    main()
