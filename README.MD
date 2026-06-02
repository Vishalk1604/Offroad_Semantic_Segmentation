# Offroad Semantic Scene Segmentation

Semantic segmentation of off‑road desert scenes for UGV autonomy, on Duality AI's **Falcon**
synthetic dataset. The core challenge: **train on one synthetic desert environment, generalize to an
unseen one** — a domain‑generalization problem.

**Architecture:** frozen **DINOv2 ViT‑B/14** foundation backbone + **Rein** parameter‑efficient
adapters + **DPT** multi‑scale decoder. Built for an **RTX 5060 8 GB (Blackwell)** GPU.
See [`plan.md`](plan.md) for full methodology and [`problem.pdf`](problem.pdf) for the challenge brief.

---

## Results summary

| Setup | Val mIoU (env A) | **Test mIoU (unseen env B)** |
|---|---|---|
| Baseline — frozen DINOv2 ViT‑S + ConvNeXt head, CE loss, 10 ep | 0.2968 | 0.2232 |
| **Advanced — DINOv2‑B + Rein + DPT, CE+Dice, aug, EMA, 50 ep** | **0.6507** | **0.2997** |

Full result artifacts committed under [`initial_results/`](initial_results/) and
[`advanced_results/`](advanced_results/).

---

## Classes

10 classes (raw 16‑bit mask IDs → train index):

| ID | Class | Train idx |
|----|-------|-----------|
| 100 | Trees | 0 |
| 200 | Lush Bushes | 1 |
| 300 | Dry Grass | 2 |
| 500 | Dry Bushes | 3 |
| 550 | Ground Clutter | 4 |
| 600 | **Flowers** | 5 |
| 700 | Logs | 6 |
| 800 | Rocks | 7 |
| 7100 | Landscape | 8 |
| 10000 | Sky | 9 |

> **Fix vs. the original starter scripts:** the provided `value_map` omitted **Flowers (600)**,
> silently folding it into background. Both scripts are corrected (`600 → 5`).

---

## Setup (RTX 5060 / Blackwell)

> ⚠️ The RTX 5060 is **Blackwell (sm_120)** and needs **CUDA 12.8+**. The starter's
> `ENV_SETUP/install_packages.bat` pins CUDA 11.8 and **will not run** on this GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# PyTorch nightly with CUDA 12.8 — must be installed first
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# all other deps
pip install -r requirements.txt
# verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

DINOv2‑B weights (~330 MB) download automatically from HuggingFace on first run.

---

## Advanced pipeline

### Train
```powershell
python train.py --backbone base --epochs 50 --batch-size 8 --img-h 378 --img-w 672 --seed 0
```
Outputs in `runs/<timestamp>/`: `best.pth` (best val mIoU, EMA weights), `last.pth`,
`config.json`, `training_curves.png`, `metrics.txt`.

### Evaluate (unseen test set)
```powershell
python test.py --weights runs\<timestamp>\best.pth --split test --tta
```
Outputs in `predictions/test/`: `evaluation_metrics.txt`, `per_class_iou.png`,
`confusion_matrix.png`, `comparisons/`, `failure_cases/`.

### Useful flags
| Flag | Effect |
|------|--------|
| `--backbone small\|base\|large` | Backbone size (small = faster, large = more accurate) |
| `--batch-size N` | Adjust to VRAM; batch 8 uses ~3 GB |
| `--img-h H --img-w W` | Must be multiples of 14; higher = more accurate but slower |
| `--no-ema` | Disable EMA |
| `--num-workers 0` | Fix DataLoader issues on Windows |
| `--tta` (test.py) | Test‑time augmentation (h‑flip + multi‑scale) |
| `--speed` (test.py) | Report ms/image |

---

## Baseline pipeline (original starter scripts)

```powershell
# Train
python Offroad_Segmentation_Scripts/train_segmentation.py
# Evaluate
python Offroad_Segmentation_Scripts/test_segmentation.py
```
Outputs: `train_stats/` (curves + metrics), `predictions/` (masks + comparisons).

---

## Initial Baseline Results

Frozen DINOv2 ViT‑S/14 + small ConvNeXt head, plain CE loss, 10 epochs. No augmentation.

| Metric | Validation (env A) | Test (unseen env B) |
|---|---|---|
| **Mean IoU** | **0.2968** | **0.2232** |
| Dice | 0.4890 | — |
| Pixel accuracy | 0.7041 | — |

**Per‑class test IoU:**

| Sky | Landscape | Rocks | Dry Grass | Dry Bushes | Ground Clutter | Trees | Logs | Lush Bushes | Flowers |
|----:|----------:|------:|----------:|-----------:|---------------:|------:|-----:|------------:|--------:|
| 0.919 | 0.499 | 0.220 | 0.150 | 0.062 | 0.061 | 0.058 | 0.032 | 0.010 | 0.000 |

Rare classes (Flowers, Logs, Lush Bushes) collapse to near‑zero — motivating class‑balanced loss
and domain‑generalization augmentation in the advanced pipeline.

---

## Advanced Results

DINOv2‑B (frozen) + Rein adapters + DPT decoder. Weighted CE + Dice loss, photometric /
weather / occlusion augmentation, EMA, bf16 AMP, 50 epochs.

| Metric | Validation (env A) | Test (unseen env B, +TTA) |
|---|---|---|
| **Mean IoU** | **0.6507** | **0.2997** |
| Dice | 0.7751 | 0.3720 |
| Pixel accuracy | 0.8726 | 0.6799 |

**Per‑class test IoU:**

| Sky | Landscape | Dry Grass | Trees | Dry Bushes | Rocks | Lush Bushes | Ground Clutter | Logs | Flowers |
|----:|----------:|----------:|------:|-----------:|------:|------------:|---------------:|-----:|--------:|
| 0.975 | 0.676 | 0.466 | 0.411 | 0.388 | 0.080 | 0.000 | 0.000 | 0.000 | 0.000 |

Val mIoU improved from **0.297 → 0.651** (+119%). Test mIoU improved from **0.223 → 0.300** (+34%).
Common‑area classes (Sky, Landscape, Dry Grass, Trees) improved dramatically; rare/thin classes
(Logs, Flowers, Ground Clutter, Lush Bushes) remain the hard target for further improvement.

Full artifacts: [`advanced_results/`](advanced_results/) — training curves, per‑class IoU bar chart,
confusion matrix, 8 sample comparisons, 8 worst‑case failure images, and run config.

---

## Data

- `Offroad_Segmentation_Training_Dataset/` — `train/` (2857) + `val/` (317) image+mask pairs.
- `Offroad_Segmentation_testImages/` — 1002 images in the **unseen** environment (used only for evaluation).
- Images 960×540 RGB, masks 960×540 16‑bit grayscale. **Not committed to git** (download from Falcon).
