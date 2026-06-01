# Offroad Semantic Scene Segmentation — Baseline

Semantic segmentation of off‑road desert scenes for UGV autonomy, on Duality AI's **Falcon**
synthetic dataset (train on one desert environment, evaluate on an unseen one).

This is the **baseline**: the provided starter scripts (frozen **DINOv2 ViT‑S/14** backbone + a small
ConvNeXt‑style segmentation head), run on an **RTX 5060 (8 GB, Blackwell)**. See [`problem.pdf`](problem.pdf)
for the full challenge brief. Improvements are tracked in subsequent commits.

## Classes

10 classes (raw 16‑bit mask IDs → train index): Trees (100), Lush Bushes (200), Dry Grass (300),
Dry Bushes (500), Ground Clutter (550), **Flowers (600)**, Logs (700), Rocks (800), Landscape (7100),
Sky (10000).

> **One fix vs. the original starter:** the provided `value_map` omitted **Flowers (600)**, silently
> folding it into the background. We added it (`600 → 10`) in `train_segmentation.py` /
> `test_segmentation.py` so all 10 real classes are learned and scored. Everything else is unchanged.

## Setup (RTX 5060 / Blackwell)

> ⚠️ The RTX 5060 is **Blackwell (sm_120)** and needs **CUDA 12.8+**. The starter's
> `ENV_SETUP/install_packages.bat` pins **CUDA 11.8 and will not run** on this GPU. Use the steps below.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install numpy pillow opencv-python matplotlib tqdm
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The DINOv2 backbone downloads automatically via `torch.hub` on first run (internet required once).

## Run

```powershell
# Train (saves segmentation_head.pth + train_stats/ next to the script)
python Offroad_Segmentation_Scripts/train_segmentation.py

# Evaluate on the unseen test environment (has ground-truth masks) -> predictions/ + mean IoU
python Offroad_Segmentation_Scripts/test_segmentation.py
```

## Outputs

- `Offroad_Segmentation_Scripts/train_stats/` — loss / IoU / Dice / pixel‑accuracy curves and
  `evaluation_metrics.txt` (per‑epoch history).
- `Offroad_Segmentation_Scripts/segmentation_head.pth` — trained head weights.
- `predictions/` — raw + colorized prediction masks, side‑by‑side comparisons, `evaluation_metrics.txt`
  (**mean IoU + per‑class IoU** on the test set), and `per_class_metrics.png`.

**Interpreting:** `mean IoU` is the headline metric the challenge scores. Per‑class IoU shows which
classes lag — expect rare ones (Logs, Flowers, Ground Clutter) to score lowest. These baseline numbers
are the reference that later improvement commits are measured against.

## Data

- `Offroad_Segmentation_Training_Dataset/` — `train/` (2857) and `val/` (317) image+mask pairs.
- `Offroad_Segmentation_testImages/` — 1002 unseen images (+ masks) for evaluation.
- Images are 960×540 RGB; masks are 960×540 16‑bit grayscale. Datasets are downloaded from Falcon and
  are **not** committed to git.
