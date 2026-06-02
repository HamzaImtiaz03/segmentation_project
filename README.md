# Partial Cross Entropy Loss for Point-Supervised Remote Sensing Segmentation

Weakly-supervised semantic segmentation of remote-sensing imagery using only a
handful of **simulated point labels** per image, trained with a **Partial Cross
Entropy / Partial Focal loss**.

This repository is the solution to a technical assessment with three tasks:

1. **Implement partial cross entropy loss** — `src/losses.py`
2. **Apply it to real data** — find a remote-sensing segmentation dataset
   (here: [LoveDA](https://github.com/Junjue-Wang/LoveDA)), randomly sample
   simulated point labels (`src/point_sampling.py`), and add the loss to a real
   segmentation network (`src/model.py`, `src/train.py`).
3. **Design experiments + write a report** — `notebooks/experiments.ipynb`,
   `report/technical_report.md`.

## The idea in one paragraph

Full pixel masks are expensive to annotate. In practice annotators only click a
few **points** ("this is a building", "this is water") and leave the rest of the
image unlabeled. A normal loss compares the prediction against *every* pixel —
we don't have those labels. **Partial Cross Entropy** computes the loss *only*
on the labeled points (using a binary mask) and ignores everything else:

```
pfCE = sum( FocalLoss(pred, GT) * MASK_labeled ) / sum( MASK_labeled )
```

We then validate against the *full* ground-truth masks to measure how well point
supervision generalizes.

## Layout

```
src/
  config.py          # paths, class names, hyperparameters
  dataset.py         # LoveDA dataset + albumentations transforms
  point_sampling.py  # simulate point labels from full masks  (Task 2)
  losses.py          # PartialCrossEntropyLoss, PartialFocalLoss  (Task 1)
  model.py           # U-Net / DeepLabV3+ w/ ImageNet-pretrained encoder
  metrics.py         # confusion-matrix based mIoU / IoU / F1 / OA
  train.py           # training loop (point labels) + full-mask validation
  evaluate.py        # final eval + qualitative visualizations
notebooks/
  experiments.ipynb  # Colab: install, download data, run all experiments, plots
report/
  technical_report.md
```

## Quick start (Google Colab)

Open `notebooks/experiments.ipynb` in Colab (GPU runtime) and run top to bottom.
It installs dependencies, downloads a LoveDA subset, runs the density and
loss-type experiments, and produces the plots used in the report.

## Quick start (local)

```bash
pip install -r requirements.txt
# Point src/config.py DATA_ROOT at a downloaded LoveDA folder, then:
python -m src.train --points-per-class 10 --loss focal --epochs 20
python -m src.evaluate --checkpoint runs/best.pt
```

## Dataset

[LoveDA](https://github.com/Junjue-Wang/LoveDA) (Land-cOVER Domain Adaptive
semantic segmentation). 7 classes: background, building, road, water, barren,
forest, agriculture. Pixel value `0` = no-data and is ignored (`ignore_index=255`
after remapping). Openly downloadable from Zenodo / HuggingFace / Google Drive.
