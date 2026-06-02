"""Final evaluation + qualitative visualization helpers.

Functions here are used both from the CLI (compute metrics for a checkpoint) and
from the notebook (render image / GT / sampled-points / prediction grids).
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import torch

from .config import CLASS_COLORS, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, Config
from .dataset import build_datasets
from .metrics import ConfusionMatrix
from .model import build_model


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map a (H,W) class-id mask to an (H,W,3) RGB image; ignore -> black."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in enumerate(CLASS_COLORS):
        rgb[mask == cls] = color
    return rgb


def denormalize(image_tensor: torch.Tensor, cfg: Config) -> np.ndarray:
    """Undo ImageNet normalization for display -> (H,W,3) uint8."""
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(cfg.std) + np.array(cfg.mean)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


@torch.no_grad()
def compute_metrics(model, loader, device):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES, ignore_index=IGNORE_INDEX)
    for batch in loader:
        preds = model(batch["image"].to(device)).argmax(dim=1)
        cm.update(preds, batch["full_mask"])
    return cm.compute()


def metrics_table(metrics) -> str:
    """Pretty per-class IoU + summary table as a string."""
    lines = [f"{'class':<14}{'IoU':>8}"]
    for name, iou in zip(CLASS_NAMES, metrics["per_class_iou"]):
        val = "  n/a" if np.isnan(iou) else f"{iou:.4f}"
        lines.append(f"{name:<14}{val:>8}")
    lines.append("-" * 22)
    lines.append(f"{'mIoU':<14}{metrics['miou']:>8.4f}")
    lines.append(f"{'macro-F1':<14}{metrics['macro_f1']:>8.4f}")
    lines.append(f"{'overall acc':<14}{metrics['overall_acc']:>8.4f}")
    return "\n".join(lines)


@torch.no_grad()
def visualize_predictions(model, dataset, device, indices: Optional[List[int]] = None,
                          max_samples: int = 4):
    """Plot image | full GT | sampled points | prediction for a few samples.

    Returns the matplotlib Figure so the notebook can save it for the report.
    """
    import matplotlib.pyplot as plt

    model.eval()
    if indices is None:
        indices = list(range(min(max_samples, len(dataset))))

    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["image", "full GT", "sampled points", "prediction"]

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        pred = model(image).argmax(dim=1)[0].cpu().numpy()

        full = sample["full_mask"].numpy()
        sparse = sample["sparse_label"].numpy()
        # show points enlarged-ish by overlaying colored points on a gray canvas
        points_vis = np.full((*sparse.shape, 3), 40, dtype=np.uint8)
        labeled = sparse != IGNORE_INDEX
        points_vis[labeled] = colorize(np.where(labeled, sparse, 0))[labeled]

        panels = [denormalize(sample["image"], dataset.cfg), colorize(full),
                  points_vis, colorize(pred)]
        for col, (panel, title) in enumerate(zip(panels, titles)):
            axes[row, col].imshow(panel)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    fig.tight_layout()
    return fig


def load_checkpoint(path: str, device) -> tuple:
    """Rebuild the model+config from a saved checkpoint and load weights."""
    ckpt = torch.load(path, map_location=device)
    cfg = Config(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def _main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on the LoveDA val set.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None, help="override data_root in the checkpoint")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(a.checkpoint, device)
    if a.data_root:
        cfg.data_root = a.data_root
    _, val_ds = build_datasets(cfg)
    from torch.utils.data import DataLoader
    loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    print(metrics_table(compute_metrics(model, loader, device)))


if __name__ == "__main__":
    _main()
