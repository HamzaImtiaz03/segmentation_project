"""LoveDA dataset loader producing (image, full_mask, sparse_label, labeled_mask).

LoveDA layout after download::

    LoveDA/
      Train/{Rural,Urban}/images_png/*.png
      Train/{Rural,Urban}/masks_png/*.png
      Val/{Rural,Urban}/images_png/*.png
      Val/{Rural,Urban}/masks_png/*.png

Mask pixel values are 0..7 where **0 is no-data** (ignored) and 1..7 are the
real classes. We remap value ``v -> v-1`` and send no-data to ``IGNORE_INDEX``,
so the network sees class ids 0..6.

The dataset applies an albumentations spatial transform to image+mask *together*,
then simulates point labels on the transformed full mask (so points stay
aligned), and finally converts everything to tensors.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import IGNORE_INDEX, NUM_CLASSES, Config
from .point_sampling import make_rng, sample_point_labels


def discover_pairs(data_root: str, split: str) -> List[Tuple[str, str]]:
    """Find (image_path, mask_path) pairs for ``split`` in {"train", "val"}."""
    split_dir = {"train": "Train", "val": "Val"}.get(split.lower(), split)
    root = Path(data_root)
    pairs: List[Tuple[str, str]] = []
    for images_dir in root.rglob("images_png"):
        # keep only dirs belonging to the requested split (e.g. .../Train/Urban/images_png)
        if split_dir.lower() not in str(images_dir).lower():
            continue
        masks_dir = images_dir.parent / "masks_png"
        for img_path in sorted(images_dir.glob("*.png")):
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path)))
    return pairs


def remap_mask(raw: np.ndarray) -> np.ndarray:
    """LoveDA raw values 0..7 -> class ids 0..6, no-data (0) -> IGNORE_INDEX."""
    out = raw.astype(np.int64) - 1
    out[raw == 0] = IGNORE_INDEX
    return out


def build_transforms(cfg: Config, train: bool):
    import albumentations as A  # imported lazily so the module loads without it

    size = cfg.image_size
    if train:
        return A.Compose([
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=cfg.mean, std=cfg.std),
        ])
    return A.Compose([
        A.Resize(size, size),  # masks use nearest-neighbour automatically
        A.Normalize(mean=cfg.mean, std=cfg.std),
    ])


class LoveDADataset(Dataset):
    """Returns a dict of tensors for one tile.

    Keys: ``image`` (3,H,W) float, ``full_mask`` (H,W) long, ``sparse_label``
    (H,W) long, ``labeled_mask`` (H,W) uint8.
    """

    def __init__(self, pairs: List[Tuple[str, str]], cfg: Config, train: bool):
        self.pairs = pairs
        self.cfg = cfg
        self.train = train
        self.transform = build_transforms(cfg, train)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        full_mask = remap_mask(raw_mask)

        augmented = self.transform(image=image, mask=full_mask)
        image = augmented["image"]                 # (H,W,3) normalized float32
        full_mask = augmented["mask"].astype(np.int64)

        # Simulate the point annotation on the (possibly cropped) mask.
        rng = make_rng(self.cfg.point_seed, idx)
        sparse_label, labeled_mask = sample_point_labels(
            full_mask,
            n_points_per_class=self.cfg.points_per_class,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            rng=rng,
        )

        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
            "full_mask": torch.from_numpy(full_mask).long(),
            "sparse_label": torch.from_numpy(sparse_label).long(),
            "labeled_mask": torch.from_numpy(labeled_mask),
        }


def build_datasets(cfg: Config) -> Tuple[LoveDADataset, LoveDADataset]:
    """Build train/val datasets, applying the optional subset caps from cfg."""
    train_pairs = discover_pairs(cfg.data_root, "train")
    val_pairs = discover_pairs(cfg.data_root, "val")
    if not train_pairs:
        raise FileNotFoundError(
            f"No LoveDA training pairs found under {cfg.data_root!r}. "
            "Check the path / that masks_png exists alongside images_png."
        )

    if cfg.subset_train > 0:
        train_pairs = train_pairs[: cfg.subset_train]
    if cfg.subset_val > 0:
        val_pairs = val_pairs[: cfg.subset_val]

    return (
        LoveDADataset(train_pairs, cfg, train=True),
        LoveDADataset(val_pairs, cfg, train=False),
    )
