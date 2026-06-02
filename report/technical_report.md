# Partial Cross Entropy Loss for Point-Supervised Remote Sensing Segmentation
### Technical Report

> **How to use this document.** The Method section is complete. The Experiment
> sections give the purpose, hypothesis, and exact process; the **Results**
> tables/figures are marked `[fill in]` and are produced automatically by
> `notebooks/experiments.ipynb` (it saves `report/fig_*.png` and prints every
> number you need). Run the notebook top-to-bottom, then paste the numbers in.

---

## 1. Method

### 1.1 Problem
Semantic segmentation classifies every pixel of an image. Standard supervised
training needs a **dense mask** (every pixel labeled), which is prohibitively
expensive for remote-sensing imagery. In practice annotators provide only a few
**point labels** per image ("this pixel is a building", "this is water"), leaving
the vast majority of pixels unlabeled. This is **weakly / point-supervised**
segmentation: we must train a dense pixel classifier from sparse labels.

### 1.2 Partial Cross Entropy / Partial Focal loss
A standard loss averages a per-pixel term over **all** pixels — impossible when
most pixels have no label. The **partial** loss restricts the average to the
labeled pixels using a binary mask `MASK_labeled` (1 where a label exists):

```
partial_loss = sum( per_pixel_loss * MASK_labeled ) / sum( MASK_labeled )
```

- **Partial Cross Entropy (pCE):** `per_pixel_loss = CE(pred, GT)`. With a fully
  labeled mask this is identical to ordinary cross entropy (verified numerically
  in the notebook, §3a), so it is a strict generalization.
- **Partial Focal CE (pfCE)** — the assessment's target formula:

  ```
  pfCE = sum( FocalLoss(pred, GT) * MASK_labeled ) / sum( MASK_labeled )
  FocalLoss = -alpha * (1 - p_t)^gamma * log(p_t)
  ```

  where `p_t` is the predicted probability of the true class. The factor
  `(1 - p_t)^gamma` down-weights easy pixels and focuses learning on hard / rare
  classes; `alpha` is an optional balancing scalar. `gamma=0, alpha<=0` recovers
  pCE.

Implementation: [`src/losses.py`](../src/losses.py). Unlabeled pixels carry an
`ignore_index` (255); the mask is derived from the sparse label automatically, so
no-data and unlabeled pixels never contribute to the gradient. An all-unlabeled
batch is guarded by clamping the denominator.

### 1.3 Simulating point labels
The dataset ships dense masks, so we **simulate** the point-annotation scenario:
for each image and each class present in its mask, we randomly sample up to *N*
pixels of that class and keep only those as labels; all other pixels become
`ignore_index`. Sampling is seeded per image for full reproducibility.
Implementation: [`src/point_sampling.py`](../src/point_sampling.py). The
**evaluation always uses the original dense masks**, so metrics honestly reflect
how well the sparse supervision generalizes to every pixel.

### 1.4 Network and transfer learning
Model: **U-Net** with a **ResNet-34 encoder pretrained on ImageNet** (via
`segmentation_models_pytorch`), matching the "transfer learning baseline"
described in the project brief. Pretrained features are critical under sparse
supervision: with only a handful of labeled pixels, the model relies heavily on
priors learned from large-scale data. Implementation: [`src/model.py`](../src/model.py).

### 1.5 Dataset
**LoveDA** (Land-cOVER Domain Adaptive semantic segmentation): high-resolution
aerial tiles over urban and rural scenes, 7 classes — background, building, road,
water, barren, forest, agriculture (raw value 0 = no-data, ignored). We use a
capped subset (300 train / 100 val tiles) at 512×512 to keep a full experiment
sweep tractable on a free Colab GPU. Loader: [`src/dataset.py`](../src/dataset.py).

### 1.6 Training & metrics
AdamW, cosine LR schedule, mixed precision, 15 epochs per run. Training uses
**only the point labels** through the partial loss; after each epoch we evaluate
on the **full** masks. Metrics (confusion-matrix based,
[`src/metrics.py`](../src/metrics.py)): per-class IoU, **mean IoU (mIoU)**,
macro-F1, overall accuracy. The best-mIoU checkpoint per run is saved.

---

## 2. Experiment 1 — Effect of annotation density

**Purpose.** Quantify how the number of labeled points per class affects
segmentation quality — i.e., how much annotation effort is actually needed.

**Hypothesis.** mIoU increases with more points but with **diminishing returns**;
because of transfer learning, even very sparse supervision (a few points/class)
should recover a large fraction of the fully-supervised performance.

**Process.** Hold everything fixed (U-Net/ResNet-34, same train/val split, 15
epochs, seed=42, partial **focal** loss, γ=2, α=0.25). Vary only
`points_per_class ∈ {1, 5, 10, 20, 50}`. For each value: simulate points → train
→ record best val mIoU. Express each result as a percentage of the
fully-supervised upper bound (§4).

**Results.** `[fill in from notebook]`

| points/class | val mIoU | macro-F1 | overall acc | % of full-sup mIoU |
|--------------|----------|----------|-------------|--------------------|
| 1            | `…`      | `…`      | `…`         | `…`                |
| 5            | `…`      | `…`      | `…`         | `…`                |
| 10           | `…`      | `…`      | `…`         | `…`                |
| 20           | `…`      | `…`      | `…`         | `…`                |
| 50           | `…`      | `…`      | `…`         | `…`                |

Figure: `report/fig_density.png` (mIoU vs points/class).

**Discussion.** `[Describe where the curve flattens — the “good enough” density —
and how close sparse supervision gets to full supervision.]`

---

## 3. Experiment 2 — Partial CE vs Partial Focal loss

**Purpose.** Determine whether the focal variant improves over plain partial CE
under sparse point supervision.

**Hypothesis.** Focal loss down-weights easy/abundant pixels and emphasizes hard
ones, so it should **improve IoU on rarer classes** (e.g., road, water, barren)
and modestly raise mIoU at a fixed annotation budget.

**Process.** Fix `points_per_class=10` and all other settings; train once with
`loss='ce'` and once with `loss='focal'` (γ=2, α=0.25). Compare overall mIoU and
**per-class IoU**.

**Results.** `[fill in from notebook]`

| loss          | mIoU | macro-F1 | overall acc |
|---------------|------|----------|-------------|
| partial CE    | `…`  | `…`      | `…`         |
| partial focal | `…`  | `…`      | `…`         |

Figure: `report/fig_loss_perclass.png` (per-class IoU, CE vs focal).

**Discussion.** `[State whether focal helped, and specifically on which (rare)
classes; note any class where CE was better.]`

---

## 4. Fully-supervised upper bound (reference)

**Purpose.** Provide an upper bound for interpreting the point-supervised
results. **Process.** Identical pipeline, but train on the **full** masks
(`supervision='full'`). **Result:** mIoU = `[fill in]`. All point-supervised
numbers above are reported as a percentage of this value.

---

## 5. Conclusions

`[Summarize, e.g.:]`
- With ImageNet transfer learning, ~`[N]` points per class already reaches
  `[X]%` of full-mask mIoU — point supervision is highly label-efficient.
- Returns diminish beyond ~`[M]` points/class.
- Partial focal `[did / did not]` outperform partial CE overall, with the
  clearest gains on `[classes]`.

**Limitations / future work.** Single backbone and dataset subset; points sampled
uniformly (real clicks are biased toward object centers); no semi-supervised use
of unlabeled pixels, test-time augmentation, or ensembling — all listed in the
project brief as natural next steps to push performance further.

---

## 6. Reproducibility

- Code: `src/` (modular) + `notebooks/experiments.ipynb` (orchestration).
- Deterministic: fixed seeds for sampling (`point_seed`) and training (`seed`).
- One command per run, e.g.:
  ```bash
  python -m src.train --points-per-class 10 --loss focal --epochs 15
  python -m src.evaluate --checkpoint runs/exp/best.pt
  ```
- Each run writes `runs/<name>/summary.json` (config + full metric history) and
  `best.pt`, so every number in this report is traceable to a file.
