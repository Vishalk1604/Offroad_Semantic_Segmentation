# Off-Road Semantic Segmentation — Reproduction Guide

This repository contains our improved semantic segmentation pipeline developed on top of the provided hackathon baseline.
We introduce class-weight balancing, decoder architecture upgrades, and backbone experimentation to significantly improve IoU performance over the original model.

This guide explains how to reproduce our results.

---

## 📦 Setup

1. Clone this repository
2. Install dependencies (same environment as baseline)
3. Locate the original competition `scripts/` directory

---

## 🔁 Reproducing Our Main Results

### Step 1 — Replace Default Scripts

Copy **all files from this repository** into the original `scripts/` folder and replace the default files.

> The default configuration already corresponds to our final training phase which includes:

* Class weighting
* Decoder upgrade
* Tuned loss balancing

---

### Step 2 — Training Configuration

Inside the training script you can adjust:

```python
N_EPOCHS
BACKBONE_LR
DICE_WEIGHT
```

We achieved two main results using:

| Run | N_Epoch | Backbone_LR | Dice_weight | mAP@0.5  |
| --- | ------- | ----------- | ----------- | -------- |
| 1   | 25      | LR*0.05     | 0.4         | 0.1569   |
| 2   | 35      | LR*0.08     | 0.25        | 0.1668   |

*(Replace placeholders with actual values/results)*

---

### Step 3 — Train & Test

Train:

```bash
python train_segmentation.py
```

Test:

```bash
python test_segmentation.py
```

---

## 🧠 Reproducing DINOv2 Backbone Results

To replicate experiments using DINOv2:

### Step 1 — Choose Mode

Select **one**:

* Frozen Backbone
* Unfrozen Backbone

---

### Step 2 — Replace Scripts

Copy contents of one folder into `scripts/` and overwrite files:

```
dinov2_frozen/
```

or

```
dinov2_unfreeze/
```

---

### Step 3 — Train (10 Epochs)

```bash
python train_segmentation.py
```

---

### Results

| Mode     | Epochs | Result Val IoU  |
| -------- | ------ | --------------- |
| Frozen   | 20     | 0.4762          |
| Unfrozen | 20     | 0.4781          |


---

## 📊 Performance Comparison

| Model             | IoU             |
| ----------------- | --------------- |
| Baseline Provided | **0.29**        |
| Our Model — Run 1 | 0.3749          |
| Our Model — Run 2 | 0.4083          |
| DINOv2 Frozen     | 0.4762          |
| DINOv2 Unfrozen   | 0.4781          |

---

## 🧩 Summary

Key improvements over baseline:

* Class-weighted loss to address imbalance
* Enhanced decoder architecture
* Backbone learning-rate tuning
* Dice loss weighting optimization
* Optional DINOv2 backbone integration

These changes provide significant IoU improvement while keeping the pipeline reproducible and configurable.

---

## 👨‍💻 Notes

* Results may vary slightly due to randomness in training
* Ensure dataset paths match expected structure
* CUDA GPU recommended

---

## 📜 License

Add license if required.
