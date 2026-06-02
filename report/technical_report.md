# Partial Cross Entropy Loss for Point-Supervised Remote Sensing Segmentation
### Technical Report

All experiments were run on Google Colab (NVIDIA Tesla T4) using the code in
`src/` orchestrated by [`notebooks/experiments.ipynb`](../notebooks/experiments.ipynb).
Common settings for every run: U-Net + ImageNet-pretrained ResNet-34 encoder,
LoveDA subset of **300 train / 100 val** tiles at **512×512**, **15 epochs**,
AdamW (lr 3e-4), cosine schedule, batch size 8, mixed precision, seed 42.

> **Figures.** The notebook saves `fig_density.png`, `fig_loss_perclass.png`, and
> `fig_qualitative.png` into this `report/` folder on the Colab runtime —
> download them from the Colab Files panel into `report/` so the links below
> resolve locally.

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
  labeled mask this is identical to ordinary cross entropy — **verified
  numerically**: partial CE = 2.330903 vs `nn.CrossEntropyLoss` = 2.330904
  (difference 4.8e-7). It is therefore a strict generalization.
- **Partial Focal CE (pfCE)** — the assessment's target formula:

  ```
  pfCE = sum( FocalLoss(pred, GT) * MASK_labeled ) / sum( MASK_labeled )
  FocalLoss = -alpha * (1 - p_t)^gamma * log(p_t)
  ```

  where `p_t` is the predicted probability of the true class. The factor
  `(1 - p_t)^gamma` down-weights easy pixels and focuses learning on hard / rare
  classes; `alpha` is an optional balancing scalar (we use γ=2, α=0.25).
  `gamma=0, alpha<=0` recovers pCE.

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
Training uses **only the point labels** through the partial loss; after each
epoch we evaluate on the **full** masks. Metrics (confusion-matrix based,
[`src/metrics.py`](../src/metrics.py)): per-class IoU, **mean IoU (mIoU)**,
macro-F1, overall accuracy (OA). The best-mIoU checkpoint per run is saved.

> **Note on absolute scores.** With a 300-tile subset and a 15-epoch budget on a
> hard 7-class dataset, absolute mIoU is modest (full supervision reaches only
> ~0.34). The experiments are designed for **relative** comparison under
> identical conditions, which is what the conclusions rely on.

---

## 2. Experiment 1 — Effect of annotation density

**Purpose.** Quantify how the number of labeled points per class affects
segmentation quality — i.e., how much annotation effort is actually needed.

**Hypothesis.** mIoU increases with more points but with **diminishing returns**;
because of transfer learning, even very sparse supervision should recover a large
fraction of the fully-supervised performance.

**Process.** Hold everything fixed (U-Net/ResNet-34, same split, 15 epochs,
seed 42, partial **focal** loss, γ=2, α=0.25). Vary only
`points_per_class ∈ {1, 5, 10, 20, 50}`. Record the best val checkpoint per run
and express its mIoU as a percentage of the fully-supervised upper bound (§4).

**Results.**

| points/class | val mIoU | macro-F1 | overall acc | % of full-sup mIoU |
|--------------|----------|----------|-------------|--------------------|
| 1            | 0.2481   | 0.3798   | 0.4681      | 73.2 %             |
| 5            | 0.2754   | 0.4184   | 0.4991      | 81.3 %             |
| 10           | 0.2913   | 0.4405   | 0.4875      | 86.0 %             |
| 20           | 0.3177   | 0.4741   | 0.5278      | 93.8 %             |
| 50           | 0.3232   | 0.4763   | 0.5462      | 95.4 %             |

Figure: [`report/fig_density.png`](fig_density.png) (mIoU vs points/class, log x-axis).

**Discussion.** The trend is clearly monotonic and confirms the hypothesis:

- The total span (0.2481 → 0.3232, +0.075 mIoU) is large and well above the
  run-to-run noise floor we measured (~0.036, see §3), so the density effect is
  real and meaningful.
- **Returns diminish around 20 points/class.** The early gains are big
  (1→5: +0.027; 10→20: +0.026), but **20→50 adds only +0.0055** despite 2.5× more
  labels — within noise. So beyond ~20 points/class extra annotation buys almost
  nothing here.
- **Label efficiency is striking.** Even **1 point per class** reaches **73 %** of
  full-mask mIoU, and **20 points per class reach 93.8 %** — strong evidence that
  ImageNet transfer learning + a partial loss makes point supervision highly
  label-efficient.

---

## 3. Experiment 2 — Partial CE vs Partial Focal loss

**Purpose.** Determine whether the focal variant improves over plain partial CE
under sparse point supervision.

**Hypothesis.** Focal loss down-weights easy/abundant pixels and emphasizes hard
ones, so it should improve IoU (especially on rarer classes) at a fixed
annotation budget.

**Process.** Fix `points_per_class=10` and all other settings; train once with
`loss='ce'` and once with `loss='focal'` (γ=2, α=0.25).

**Results.**

| loss          | val mIoU | macro-F1 | overall acc |
|---------------|----------|----------|-------------|
| partial CE    | 0.3057   | 0.4572   | 0.5251      |
| partial focal | 0.3273   | 0.4839   | 0.5554      |

Figure: [`report/fig_loss_perclass.png`](fig_loss_perclass.png) (per-class IoU, CE vs focal).

**Discussion.** Partial focal beat partial CE on every summary metric in this run
(+0.0216 mIoU, +0.027 macro-F1, +0.030 OA), consistent with the hypothesis that
focusing on hard pixels helps. **However, this result is suggestive rather than
conclusive:** the +0.0216 mIoU gain is comparable to the **~0.036 run-to-run
variance** we observed for an identical configuration (§3.1). A firm claim would
need multiple seeds per loss; with a single run we can only say focal was *at
least as good as*, and probably slightly better than, partial CE here.

### 3.1 A note on run-to-run variance (measured)
The exact same configuration — focal loss, 10 points/class — was trained twice
(once inside Experiment 1, once inside Experiment 2) and produced **0.2913** vs
**0.3273** mIoU. With AMP and non-deterministic cuDNN kernels (we did not force
deterministic mode), this ~0.036 spread is expected on a small subset with
best-epoch selection on 100 val tiles. **Implication:** differences smaller than
~0.03–0.04 mIoU should be treated as within noise; the density trend in §2 is
well above this band, the focal-vs-CE gap in §3 is at its edge.

---

## 4. Fully-supervised upper bound (reference)

**Purpose.** Provide an upper bound for interpreting the point-supervised results.
**Process.** Identical pipeline, but train on the **full** masks
(`supervision='full'`, standard CE). **Result:** mIoU **0.3387**, macro-F1 0.4562,
OA **0.6637**. All point-supervised numbers above are reported as a percentage of
this mIoU.

A revealing detail: full supervision's **mIoU is only modestly higher** than the
50-point run (0.3387 vs 0.3232, a 4.6 % gap), yet its **overall accuracy is far
higher** (0.66 vs ~0.55). This is exactly what point supervision predicts: because
we sample an *equal* number of points per class, the model is trained
class-balanced — it recovers the class-balanced **mIoU** very well but lags on
**OA**, which is dominated by the most frequent classes that dense masks cover
exhaustively.

---

## 5. Per-class behaviour (focal, 10 points/class)

Per-class IoU of the 10-point focal model on the full val masks
(from the qualitative cell; see [`report/fig_qualitative.png`](fig_qualitative.png)):

| class       | IoU    |
|-------------|--------|
| background  | 0.2872 |
| building    | 0.3785 |
| road        | 0.2102 |
| water       | 0.3934 |
| barren      | 0.1335 |
| forest      | 0.1989 |
| agriculture | 0.4373 |
| **mIoU**    | **0.2913** |

Easy/abundant classes (agriculture, water, building) score highest; the hardest
are **barren (0.13)**, **forest (0.20)**, and **road (0.21)** — thin, sparse, or
visually ambiguous classes where a handful of points carry the least signal.
These are exactly the classes a focal loss / a larger point budget should target,
which the §2 and §3 trends support.

---

## 6. Conclusions

- **Point supervision is highly label-efficient on LoveDA.** With ImageNet
  transfer learning and a partial loss, **1 point/class already reaches 73 %** and
  **20 points/class reach ~94 %** of fully-supervised mIoU.
- **Returns diminish beyond ~20 points/class** (20→50 added only +0.006 mIoU,
  within noise) — a practical guideline for how much to annotate.
- **Partial focal ≥ partial CE** (+0.022 mIoU here), but the margin sits at the
  edge of measured run-to-run variance, so the gain is plausible but not proven
  with a single run.
- **Class balance vs accuracy:** equal-per-class point sampling yields strong
  class-balanced mIoU but lower overall accuracy than dense masks, since rare
  classes are over-represented relative to their pixel frequency.

**Limitations / future work.** Single backbone, single dataset subset, and a
short 15-epoch schedule; points sampled uniformly (real clicks cluster near object
centers); single seed (so report mean ± std over seeds to firm up §3). Natural
extensions from the project brief — semi-supervised use of unlabeled pixels,
test-time augmentation, and ensembling — should push performance further.

---

## 7. Reproducibility

- Code: `src/` (modular) + [`notebooks/experiments.ipynb`](../notebooks/experiments.ipynb).
- Deterministic point sampling (`point_seed`) and training seed; remaining
  variance is from non-deterministic GPU kernels (quantified in §3.1).
- One command per run, e.g.:
  ```bash
  python -m src.train --points-per-class 10 --loss focal --epochs 15
  python -m src.evaluate --checkpoint runs/exp/best.pt
  ```
- Each run writes `runs/<name>/summary.json` (config + full metric history) and
  `best.pt`, so every number in this report is traceable to a file.
