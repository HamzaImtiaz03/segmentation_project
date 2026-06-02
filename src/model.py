"""Segmentation model factory -- pretrained (ImageNet) encoder for transfer learning.

We use ``segmentation_models_pytorch`` so we don't hand-build a backbone. The
ImageNet-pretrained encoder is the "transfer learning baseline" the assessment
document refers to: starting from features learned on millions of natural images
makes sparse point supervision far more effective.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp

from .config import NUM_CLASSES, Config


def build_model(cfg: Config):
    """Create a U-Net or DeepLabV3+ with the configured pretrained encoder."""
    arch = cfg.arch.lower()
    common = dict(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,  # "imagenet" or None
        in_channels=3,
        classes=NUM_CLASSES,
    )
    if arch == "unet":
        return smp.Unet(**common)
    if arch in ("deeplabv3plus", "deeplab", "dlv3p"):
        return smp.DeepLabV3Plus(**common)
    raise ValueError(f"Unknown arch '{cfg.arch}'. Use 'unet' or 'deeplabv3plus'.")
