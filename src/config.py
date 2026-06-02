"""Central configuration: dataset metadata, paths, and default hyperparameters.

Everything that an experiment might want to vary lives in :class:`Config`, so the
training/evaluation code stays free of magic numbers and the notebook can build
configs programmatically for the experiment sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# LoveDA dataset metadata
# ---------------------------------------------------------------------------
# LoveDA masks use pixel values 0..7 where 0 is "no-data" (must be ignored) and
# 1..7 are the real classes. We remap to 0..6 for the network and send no-data
# to IGNORE_INDEX. See dataset.py for the remapping.
IGNORE_INDEX = 255

# Real classes after remapping (LoveDA value - 1).
CLASS_NAMES: List[str] = [
    "background",   # LoveDA 1
    "building",     # LoveDA 2
    "road",         # LoveDA 3
    "water",        # LoveDA 4
    "barren",       # LoveDA 5
    "forest",       # LoveDA 6
    "agriculture",  # LoveDA 7
]
NUM_CLASSES: int = len(CLASS_NAMES)

# RGB colors for visualizing predictions (one per class, order matches above).
CLASS_COLORS: List[Tuple[int, int, int]] = [
    (255, 255, 255),  # background - white
    (255, 0, 0),      # building   - red
    (255, 255, 0),    # road       - yellow
    (0, 0, 255),      # water      - blue
    (159, 129, 183),  # barren     - purple
    (0, 255, 0),      # forest     - green
    (255, 195, 128),  # agriculture- orange
]


@dataclass
class Config:
    """All knobs for one training run."""

    # --- data ---
    data_root: str = "data/LoveDA"      # folder containing images_png/ + masks_png/
    image_size: int = 512               # square crop/resize fed to the network
    subset_train: int = 400             # cap #train tiles (Colab time budget); 0 = all
    subset_val: int = 100               # cap #val tiles; 0 = all
    num_workers: int = 2

    # --- supervision mode ---
    supervision: str = "points"         # "points" (sparse) | "full" (upper-bound baseline)

    # --- point-label simulation (Task 2) ---
    points_per_class: int = 10          # simulated clicks per present class per image
    point_seed: int = 1234              # deterministic sampling for reproducibility

    # --- model (transfer learning) ---
    arch: str = "unet"                  # "unet" | "deeplabv3plus"
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"   # None -> train backbone from scratch

    # --- loss (Task 1) ---
    loss: str = "focal"                 # "ce" | "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25           # set <=0 to disable alpha weighting

    # --- optimization ---
    epochs: int = 20
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True                    # mixed precision on GPU

    # --- misc ---
    seed: int = 42
    out_dir: str = "runs/exp"
    device: str = "cuda"                # falls back to cpu in train.py if no GPU

    # ImageNet normalization (encoder was pretrained with these stats).
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    tags: List[str] = field(default_factory=list)  # free-form labels for logging
