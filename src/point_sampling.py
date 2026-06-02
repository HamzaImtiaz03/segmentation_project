"""Task 2 (part 1) -- Simulate point labels from full segmentation masks.

The assessment provides only fully-labeled datasets, so we *simulate* the
realistic "a few clicks per image" annotation by randomly sampling a small
number of pixels per class from each ground-truth mask. The network then trains
on these sparse points only; the full mask is reserved for honest evaluation.

For an image we produce two arrays of the same (H, W) shape:

* ``sparse_label`` -- ``ignore_index`` everywhere except at the sampled points,
  which hold the true class id. This is the target fed to the partial loss.
* ``labeled_mask`` -- {0, 1}; 1 at the sampled points. Handy for visualization
  and for passing explicitly to the loss (the loss can also derive it from the
  sparse label alone).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def make_rng(seed: int, index: int = 0) -> np.random.Generator:
    """Deterministic per-image RNG so every run samples identical points.

    Combining the global ``seed`` with the sample ``index`` gives each image its
    own reproducible stream without all images sharing the same points.
    """
    return np.random.default_rng(np.random.SeedSequence([seed, index]))


def sample_point_labels(
    full_mask: np.ndarray,
    n_points_per_class: int,
    num_classes: int,
    ignore_index: int = 255,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly pick up to ``n_points_per_class`` labeled pixels per present class.

    Parameters
    ----------
    full_mask : (H, W) int array with class ids in ``[0, num_classes)`` and
        possibly ``ignore_index`` for no-data pixels (which are never sampled).
    n_points_per_class : number of clicks to simulate per class that appears in
        this image. Classes with fewer pixels than this are fully sampled.
    num_classes : total class count.
    ignore_index : value marking unlabeled pixels in the output.
    rng : optional numpy Generator (use :func:`make_rng` for reproducibility).

    Returns
    -------
    (sparse_label, labeled_mask) -- both (H, W); sparse_label dtype int64,
    labeled_mask dtype uint8.
    """
    if rng is None:
        rng = np.random.default_rng()

    sparse_label = np.full(full_mask.shape, ignore_index, dtype=np.int64)

    for c in range(num_classes):
        ys, xs = np.where(full_mask == c)
        n_available = len(ys)
        if n_available == 0:
            continue
        k = min(n_points_per_class, n_available)
        chosen = rng.choice(n_available, size=k, replace=False)
        sparse_label[ys[chosen], xs[chosen]] = c

    labeled_mask = (sparse_label != ignore_index).astype(np.uint8)
    return sparse_label, labeled_mask
