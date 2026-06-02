"""Task 1 -- Partial Cross Entropy and Partial Focal losses.

In point-supervised segmentation we only have labels for a tiny set of pixels
(the simulated "clicks"). A standard loss averages over *every* pixel, but we
have no target for the unlabeled ones. The **partial** losses fix this by
multiplying the per-pixel loss with a binary ``labeled_mask`` (1 where a label
exists, 0 elsewhere) and normalizing by the number of labeled pixels:

    partial_loss = sum( per_pixel_loss * MASK_labeled ) / sum( MASK_labeled )

For the focal variant this reproduces the formula in the assessment:

    pfCE = sum( FocalLoss(pred, GT) * MASK_labeled ) / sum( MASK_labeled )

Conventions
-----------
* ``logits``        : (B, C, H, W) raw network outputs.
* ``target``        : (B, H, W) long tensor of class ids; unlabeled pixels carry
                      ``ignore_index`` (default 255).
* ``labeled_mask``  : optional (B, H, W) {0,1}. If omitted it is derived as
                      ``target != ignore_index`` -- so passing a sparse target
                      is enough on its own.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_mask(
    target: torch.Tensor,
    labeled_mask: Optional[torch.Tensor],
    ignore_index: int,
) -> torch.Tensor:
    """Return a boolean mask of pixels that contribute to the loss."""
    valid = target != ignore_index
    if labeled_mask is None:
        return valid
    return labeled_mask.bool() & valid


def _safe_target(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace ignored entries with a dummy class (0) so gather/CE never index
    ``ignore_index``. The dummy contributions are zeroed out by ``mask`` later."""
    safe = target.clone()
    safe[~mask] = 0
    return safe


class PartialCrossEntropyLoss(nn.Module):
    """Cross entropy computed only over labeled pixels.

    With a fully-labeled mask this is numerically equivalent to
    ``nn.CrossEntropyLoss(ignore_index=...)`` -- verified in the notebook.
    """

    def __init__(self, ignore_index: int = 255, class_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.ignore_index = ignore_index
        # registered as buffer so it follows .to(device)
        self.register_buffer(
            "class_weight", class_weight if class_weight is not None else torch.empty(0)
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = _resolve_mask(target, labeled_mask, self.ignore_index)
        safe = _safe_target(target, mask)

        weight = self.class_weight if self.class_weight.numel() > 0 else None
        per_pixel = F.cross_entropy(logits, safe, weight=weight, reduction="none")  # (B,H,W)

        per_pixel = per_pixel * mask
        denom = mask.sum().clamp(min=1)  # guard against an all-unlabeled batch
        return per_pixel.sum() / denom


class PartialFocalLoss(nn.Module):
    """Focal loss computed only over labeled pixels (the assessment's pfCE).

    Focal loss = -alpha * (1 - p_t)^gamma * log(p_t), where p_t is the predicted
    probability of the true class. ``gamma`` down-weights easy pixels; ``alpha``
    is an optional scalar balancing factor (set <= 0 to disable). With gamma=0
    and alpha<=0 this reduces to partial cross entropy.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = _resolve_mask(target, labeled_mask, self.ignore_index)
        safe = _safe_target(target, mask)

        log_p = F.log_softmax(logits, dim=1)                      # (B,C,H,W)
        log_pt = log_p.gather(1, safe.unsqueeze(1)).squeeze(1)    # (B,H,W) log prob of true class
        pt = log_pt.exp()

        focal = -((1.0 - pt) ** self.gamma) * log_pt             # focal CE per pixel
        if self.alpha > 0:
            focal = self.alpha * focal

        focal = focal * mask
        denom = mask.sum().clamp(min=1)
        return focal.sum() / denom


def build_loss(name: str, ignore_index: int = 255, **kwargs) -> nn.Module:
    """Factory used by train.py / the notebook to pick a loss by name."""
    name = name.lower()
    if name in ("ce", "partial_ce", "pce"):
        return PartialCrossEntropyLoss(ignore_index=ignore_index)
    if name in ("focal", "partial_focal", "pfce"):
        return PartialFocalLoss(
            gamma=kwargs.get("focal_gamma", 2.0),
            alpha=kwargs.get("focal_alpha", 0.25),
            ignore_index=ignore_index,
        )
    raise ValueError(f"Unknown loss '{name}'. Use 'ce' or 'focal'.")
