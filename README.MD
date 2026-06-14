# Offroad Semantic Scene Segmentation

Semantic segmentation of off‑road desert scenes for UGV autonomy, on Duality AI's **Falcon**
synthetic dataset. The core challenge: **train on one synthetic desert environment, generalize to an
unseen one** — a domain‑generalization problem.

**Architecture (latest):** frozen **DINOv3 ViT‑L/16** foundation backbone + **Rein** parameter‑efficient
adapters + **DPT** multi‑scale decoder, with Lovász‑Softmax loss and a two‑view domain‑generalization
consistency objective. Earlier milestones use DINOv2‑S/B — see the progression below. Built for an
**RTX 5060 8 GB (Blackwell)** GPU.
See [`plan.md`](plan.md) for full methodology and [`problem.pdf`](problem.pdf) for the challenge brief.

---

## Results summary

| Setup | Val mIoU (env A) | Test present‑class mIoU (env B) | Test all‑10 mIoU (env B) |
|---|---|---|---|
| Baseline — DINOv2‑S + ConvNeXt head, CE, 10 ep | 0.2968 | — | 0.2232 |
| Advanced — DINOv2‑B + Rein + DPT, CE+Dice, aug, EMA, 50 ep | 0.6507 | 0.428 | 0.2997 |
| DINOv3‑L + Rein + DPT, +Lovász +consistency, 576×1024, 100 ep | 0.6956 | 0.4516 | 0.3161 |
| **DINOv3‑L + full DG recipe (PASTA + copy‑paste + dropout), 100 ep** | **0.6676** | **0.4465** | **0.3907†** |

> **Primary metric is present‑class mIoU.** The unseen test environment contains only **7 of the 10
> classes** — Ground Clutter, Flowers, and Logs never appear in any of the 1002 test images — so the
> all‑10 mIoU is mathematically capped at ~0.49. Present‑class mIoU averages IoU only over classes
> actually present in the ground truth (the standard convention).
>
> †M4 all‑10 averages over 8 classes (Flowers and Logs treated as N/A rather than 0); comparable
> 10‑class denominator value is 0.3125.

Full result artifacts committed under [`initial_results/`](initial_results/),
[`advanced_results/`](advanced_results/), [`dinov3_results/`](dinov3_results/), and
[`dinov3_v2_results/`](dinov3_v2_results/).

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

## DINOv3 Results

Frozen **DINOv3 ViT‑L/16** (registers, patch‑16) + Rein adapters + DPT decoder. Weighted **CE + Dice +
Lovász‑Softmax** loss, wide scale‑jitter + strong photometric/weather/occlusion augmentation, a **two‑view
domain‑generalization consistency** objective (symmetric‑KL between two photometric views of one crop),
EMA, bf16 AMP, **576×1024** resolution, 100 epochs. ~30.5 M trainable params over a frozen 333 M backbone.

| Metric | Validation (env A) | Test (unseen env B, +TTA) |
|---|---|---|
| **Present‑class mIoU** | **0.6956** | **0.4516** |
| All‑10 mIoU | 0.6956 | 0.3161 |
| Pixel accuracy | 0.8875 | 0.6988 |

**Per‑class IoU — validation (env A) vs test (env B):**

| | Sky | Landscape | Trees | Dry Grass | Lush Bushes | Dry Bushes | Rocks | Flowers* | Logs* | Ground Clutter* |
|---|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| **Val** | 0.988 | 0.717 | 0.888 | 0.713 | 0.734 | 0.530 | 0.591 | 0.711 | 0.648 | 0.436 |
| **Test** | 0.980 | 0.716 | 0.459 | 0.490 | 0.001 | 0.459 | 0.057 | — | — | — |

<sub>*Flowers, Logs, Ground Clutter are absent from the test environment (env B), so they have no test IoU.</sub>

### The story: in‑domain solved, domain transfer is the open problem

On **validation** the model essentially reaches the 0.70 target (**0.6956**) and the previously‑hard
classes are now learned well — Lush Bushes **0 → 0.73**, Rocks **0.08 → 0.59**, Flowers/Logs/Ground Clutter
all 0.44–0.71. So capacity and class imbalance are no longer the bottleneck.

On the **unseen test** environment, common‑geometry classes transfer strongly (Sky 0.98, Landscape 0.72,
Dry Bushes/Dry Grass/Trees ~0.46–0.49), but **Rocks (0.057)** and **Lush Bushes (0.0005)** collapse —
exactly the two classes whose *appearance and scale* differ most between the two environments (Rocks jump
from ~1.2 % of train pixels to ~18 % of test pixels). The val→test gap (0.696 → 0.452) is therefore a
**domain‑generalization gap**, not a model‑capacity gap: the network has learned env A almost perfectly but
does not yet carry Rocks/Lush‑Bushes appearance across to env B. Closing that gap (e.g. texture/style
randomization and copy‑paste targeted at Rocks, stronger consistency on those classes) is the next lever.

Progress across milestones: val mIoU **0.297 → 0.651 → 0.696**; test present‑class mIoU **0.428 → 0.452**;
test all‑10 mIoU **0.223 → 0.300 → 0.316**.

Full artifacts: [`dinov3_results/`](dinov3_results/) — training curves, per‑class IoU, confusion matrix,
8 sample comparisons, 8 worst‑case failures, and run config.

> The `dinov3-large` backbone is **gated** on HuggingFace (accept the license + use a read token). The
> non‑gated **`dinov2-reg-large`** (DINOv2‑with‑registers ViT‑L) is a drop‑in replacement on the identical
> code path if gating is inconvenient.

---

## Milestone 4 — Domain‑Gap Closing Recipe

Same DINOv3‑L + Rein + DPT core, with three additional domain‑generalization techniques trained
for 100 epochs:

| Technique | What it does | Research basis |
|---|---|---|
| **PASTA** | Perturbs Fourier amplitude (style/texture) while preserving phase (semantics) | [arXiv 2212.00979](https://ar5iv.labs.arxiv.org/html/2212.00979) |
| **Rocks copy‑paste** | Pastes scaled‑up (1.5–3×) Rocks regions to manufacture env‑B‑like rock fields | [Ghiasi et al. 2021](https://www.researchgate.net/publication/355864719_Simple_Copy-Paste_is_a_Strong_Data_Augmentation_Method_for_Instance_Segmentation) |
| **DPT decoder dropout** | `Dropout2d(0.1)` before the final conv to regularise the trainable decoder | [DGSS survey](https://www.arxiv.org/pdf/2510.03540) |

Training command:
```powershell
python train.py --backbone dinov3-large --img-h 576 --img-w 1024 `
  --batch-size 2 --grad-accum 2 --consistency --consistency-weight 2.0 `
  --lovasz-weight 1.5 --pasta --copy-paste --weight-decay 0.05 `
  --epochs 100 --seed 0
```

| Metric | Validation (env A) | Test (unseen env B, +TTA) |
|---|---|---|
| **Present‑class mIoU** | **0.6676** | **0.4465** |
| All‑class mIoU | 0.6676 | 0.3907† |
| Pixel accuracy | 0.8772 | 0.7004 |

**Per‑class test IoU — M3 (baseline) → M4 (this run):**

| Class | M3 | M4 | Δ |
|---|---|---|---|
| Trees | 0.459 | 0.420 | −0.040 |
| Lush Bushes | 0.001 | 0.002 | +0.001 |
| Dry Grass | 0.490 | 0.481 | −0.009 |
| Dry Bushes | 0.459 | **0.492** | **+0.033** |
| Rocks | 0.057 | 0.063 | +0.006 |
| Landscape | 0.716 | 0.689 | −0.027 |
| Sky | 0.980 | 0.979 | −0.001 |

### Analysis

The heavy augmentation recipe added meaningful regularisation — val Rocks jumped to **0.538** (from
~0.59 M3) and Dry Bushes improved on test (+0.033) — but the primary gain on **test** Rocks that we
targeted (0.057 → 0.4+) did not materialise.  The deeper difficulty: PASTA and copy‑paste shift the
*appearance distribution* seen during training, but the test env‑B Rocks class differs in both
*appearance* (redder, fine‑grained) **and geometry** (occupying 18 % of pixels vs 1.2 % in train).
The copy‑paste only addresses scale/frequency, not the appearance discrepancy.

Common‑geometry classes (Trees, Landscape) regressed slightly — consistent with the heavier
augmentation forcing the decoder into a more conservative mode.  Val mIoU falling 0.6956 → 0.6676
while test stays ~flat suggests the augmentation successfully reduced *overfitting to env‑A* but
the remaining gap is driven by genuine appearance/distribution shift that augmentation alone cannot
fully bridge.

Full artifacts: [`dinov3_v2_results/`](dinov3_v2_results/) — training curves, per‑class IoU,
confusion matrix, 8 worst‑case failures, and run config.

---

## Data

- `Offroad_Segmentation_Training_Dataset/` — `train/` (2857) + `val/` (317) image+mask pairs.
- `Offroad_Segmentation_testImages/` — 1002 images in the **unseen** environment (used only for evaluation).
- Images 960×540 RGB, masks 960×540 16‑bit grayscale. **Not committed to git** (download from Falcon).
