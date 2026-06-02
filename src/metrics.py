"""Segmentation metrics via an accumulated confusion matrix.

Accumulate predictions vs full ground-truth masks over the whole validation set,
then derive per-class IoU, mean IoU (mIoU), macro-F1, and overall accuracy (OA).
Pixels with ``ignore_index`` (no-data) are excluded.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class ConfusionMatrix:
    """Rows = ground truth, columns = prediction."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target) -> None:
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        # bincount of (gt * C + pred) gives the flattened confusion matrix.
        k = self.num_classes
        idx = target.astype(np.int64) * k + pred.astype(np.int64)
        self.mat += np.bincount(idx, minlength=k * k).reshape(k, k)

    def compute(self) -> Dict[str, object]:
        m = self.mat.astype(np.float64)
        tp = np.diag(m)
        fp = m.sum(axis=0) - tp
        fn = m.sum(axis=1) - tp

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / (tp + fp + fn)              # nan where a class is absent
            f1 = 2 * tp / (2 * tp + fp + fn)
        overall_acc = tp.sum() / m.sum() if m.sum() > 0 else 0.0

        return {
            "per_class_iou": iou,
            "miou": float(np.nanmean(iou)),
            "macro_f1": float(np.nanmean(f1)),
            "overall_acc": float(overall_acc),
        }

    def reset(self) -> None:
        self.mat[:] = 0
