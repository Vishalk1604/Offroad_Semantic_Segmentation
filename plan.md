# Project Plan — Offroad Semantic Scene Segmentation

> **Generalizing synthetic desert perception to unseen terrain with a frozen vision
> foundation model.**
> DINOv2 (frozen) + Rein parameter‑efficient adapters + a DPT decoder, trained on
> Duality AI's Falcon digital‑twin data and evaluated on a **novel** desert environment.

---

## 1. Problem & Why It Matters

Unmanned Ground Vehicles (UGVs) navigating off‑road need **per‑pixel scene understanding** to tell
drivable ground from obstacles (rocks, logs, bushes) in real time. Real labelled off‑road data is
scarce and expensive, so Duality AI generates it synthetically from their **Falcon** digital‑twin
simulator. The catch — and the entire point of this challenge — is the **domain gap**: we train on
one set of synthetic desert environments and must perform on a **different, unseen** desert
environment. This is a **domain‑generalization (DG)** problem, not a vanilla segmentation task.

**Scoring:** IoU = 80 pts, report clarity = 20 pts. **Constraint:** < 50 ms / image inference,
single **8 GB RTX 5060 (Blackwell)** GPU.

## 2. Dataset

| Split | Pairs | Source environment | Use |
|-------|-------|--------------------|-----|
| `train` | 2857 | desert env A | training |
| `val`   | 317  | desert env A | model selection |
| `test`  | 1002 | **novel** desert env B | held‑out generalization eval (never trained on) |

- **Images:** 960×540 RGB. **Masks:** 960×540, **16‑bit grayscale** (raw class IDs).
- **10 classes, no background** — every pixel is a real class; `Landscape` is the catch‑all ground.

| ID (raw) | 100 | 200 | 300 | 500 | 550 | 600 | 700 | 800 | 7100 | 10000 |
|----------|-----|-----|-----|-----|-----|-----|-----|-----|------|-------|
| **Class** | Trees | Lush Bushes | Dry Grass | Dry Bushes | Ground Clutter | Flowers | Logs | Rocks | Landscape | Sky |
| **Train idx** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

> **Correctness note:** the provided baseline `value_map` **omits Flowers (600)** and invents a
> background class that no pixel uses. We fix the label space to the 10 classes above. Rare classes
> (**Ground Clutter, Flowers, Logs**) are heavily under‑represented and partly absent from the test
> set — the canonical off‑road **class‑imbalance** problem, which our loss design targets directly.

## 3. Approach — Foundation Model for Domain Generalization

The literature is clear that **frozen self‑supervised vision foundation models generalize across
domains far better than supervised backbones**, and that **parameter‑efficient adapters (Rein)** on
top of a frozen DINOv2 are state‑of‑the‑art for synthetic→real / cross‑environment segmentation
(Rein, *Stronger, Fewer & Superior*, CVPR'24; Rein++ '25). We build exactly that:

```
            RGB image (resized to a multiple of 14)
                         │
            ┌────────────▼─────────────┐
            │   DINOv2 ViT-B/14         │   ← FROZEN foundation backbone
            │   (self-supervised)       │     (robust, domain-general features)
            └─────┬───────┬──────┬──────┘
        layer 3  │   6   │  9   │ 12   (tapped multi-scale tokens)
            ┌─────▼───────▼──────▼──────┐
            │   Rein adapters           │   ← TRAINABLE (parameter-efficient)
            │   (per-layer token refine)│     adapts frozen features to our domain
            └────────────┬──────────────┘
            ┌────────────▼──────────────┐
            │   DPT decoder             │   ← TRAINABLE
            │   (reassemble 4 scales →  │     dense, multi-scale fusion
            │    RefineNet fusion → up) │
            └────────────┬──────────────┘
                         ▼
            10-class logits @ full resolution → argmax mask
```

**Why this wins points:** the backbone stays frozen (cheap, fits 8 GB, never overfits the source
domain), while a few million trainable adapter + decoder parameters specialize it — keeping the
generalization power that the 80‑pt IoU on the *unseen* environment depends on.

- **Backbone:** DINOv2 ViT‑B/14 (frozen). Fallbacks: ViT‑S/14 (faster, guarantees < 50 ms),
  ViT‑L/14 (stretch, gradient checkpointing).
- **Adapters:** Rein‑style learnable token refinement on tapped layers (LoRA via `peft` as a drop‑in
  fallback). Only adapters + decoder are trainable.
- **Decoder:** DPT multi‑scale reassemble + fusion — far stronger than the baseline's tiny head.

## 4. Training Recipe

| Ingredient | Choice | Why |
|-----------|--------|-----|
| Loss | weighted **CE + Dice** (`ignore_index=255`) | class imbalance + directly optimizes overlap/IoU |
| Class weights | effective‑number‑of‑samples from train pixel freq | up‑weights rare Logs/Flowers/Clutter |
| Optimizer | AdamW, cosine LR + warmup, lr ≈ 6e‑5–1e‑4 | stable training of adapters + decoder |
| Precision | **bf16** AMP | Blackwell‑native, halves memory |
| **EMA** | exponential moving average of weights | free mIoU boost (GOOSE'25 winner) |
| **Augmentation** | photometric distortion, shadow/fog/sun‑flare, blur/noise, flip, scale‑crop, **CoarseDropout** | the DG core — simulates the unseen environment; occlusion helps Logs |
| Mask resize | **nearest** | preserves integer labels (baseline used bilinear — a bug) |
| Schedule | ~40–60 epochs, early‑stop on val mIoU | |

## 5. Evaluation Protocol

- **Primary metric:** mean IoU (confusion‑matrix based, accumulated over the dataset) + **per‑class IoU**.
- **Secondary:** Dice, pixel accuracy, **confusion matrix**, inference ms/image.
- **Generalization:** report val (env A) **and** held‑out test (env B) mIoU — the gap is the DG story.
- **Failure‑case analysis:** lowest‑IoU images surfaced with side‑by‑side input / GT / prediction
  overlays, focused on rare classes.

## 6. Expected Results

| Configuration | val mIoU (env A) | Notes |
|---------------|------------------|-------|
| Provided baseline (frozen ViT‑S + tiny head, CE, 10 ep, no aug) | ~0.30–0.45 | reproduced as reference |
| **Ours** (DINOv2‑B + Rein + DPT + balanced loss + aug + EMA) | **~0.65–0.80 (target)** | filled after training |

_Results table, training curves, per‑class IoU bars, and confusion matrix are inserted here after the
runs complete._

## 7. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Domain gap to unseen env B | frozen DINOv2 + Rein adapters + heavy photometric augmentation |
| Severe class imbalance (Logs/Flowers/Clutter) | effective‑number class weights + Dice loss |
| 8 GB VRAM (Blackwell) | frozen backbone, bf16, modest resolution (multiple of 14), grad‑accum |
| RTX 5060 / sm_120 toolchain | PyTorch **nightly cu128** (the cu118 baseline installer fails) |
| Baseline label/resize bugs | corrected 10‑class map (incl. Flowers) + nearest‑neighbor mask resize |

## 8. Future Work

- **Rein++ self‑training** on the unlabeled test environment (test‑time adaptation, no labels).
- **Multi‑scale / sliding‑window** inference and **test‑time augmentation** for extra mIoU.
- **Ensemble** with an efficient SegFormer‑B3 for the final submission.
- **INT8 / TensorRT** export to push inference well under the 50 ms budget.

## 9. References

- Rein — *Stronger, Fewer & Superior: Harnessing Vision Foundation Models for Domain Generalized
  Semantic Segmentation* (CVPR'24): <https://arxiv.org/html/2312.04265>
- *Rein++* (2025): <https://arxiv.org/html/2508.01667v1>
- *SegFormer* (NeurIPS'21): <https://arxiv.org/pdf/2105.15203>
- GOOSE 2025 off‑road challenge (photometric distortion + EMA): <https://arxiv.org/pdf/2505.11769>
- BRAVO'24 robustness winner (vision foundation models for segmentation): <https://arxiv.org/pdf/2409.17208>
- Cross‑dataset evaluation of off‑road segmentation: <https://www.sciencedirect.com/science/article/pii/S0957417426015691>
- DINOv2 (Oquab et al., 2023): <https://arxiv.org/abs/2304.07193> · DPT (Ranftl et al., 2021): <https://arxiv.org/abs/2103.13413>
