"""Training loop -- trains on point labels, validates on full masks.

The model only ever sees the sparse point labels during training (via the
partial loss). After every epoch we evaluate against the *complete* ground-truth
masks, which is the honest measure of how well point supervision generalizes to
all pixels.

Usable two ways:
* programmatically -- ``history = train_model(cfg)`` (used by the notebook), and
* from the CLI -- ``python -m src.train --points-per-class 10 --loss focal``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import IGNORE_INDEX, NUM_CLASSES, Config
from .dataset import build_datasets
from .losses import build_loss
from .metrics import ConfusionMatrix
from .model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA not available -> falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def evaluate_model(model, loader, device) -> Dict[str, object]:
    """Run validation against full masks and return the metrics dict."""
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES, ignore_index=IGNORE_INDEX)
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        cm.update(preds, batch["full_mask"])
    return cm.compute()


def train_model(cfg: Config) -> Dict[str, object]:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    criterion = build_loss(
        cfg.loss, ignore_index=IGNORE_INDEX,
        focal_gamma=cfg.focal_gamma, focal_alpha=cfg.focal_alpha,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    use_full = cfg.supervision.lower() == "full"
    history = {"train_loss": [], "val_miou": [], "val_f1": [], "val_oa": []}
    best = {"miou": -1.0, "epoch": -1, "metrics": None}

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            if use_full:
                target = batch["full_mask"].to(device)
                mask = None  # loss derives the mask from non-ignore pixels
            else:
                target = batch["sparse_label"].to(device)
                mask = batch["labeled_mask"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, target, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        train_loss = running / max(len(train_loader), 1)
        metrics = evaluate_model(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_miou"].append(metrics["miou"])
        history["val_f1"].append(metrics["macro_f1"])
        history["val_oa"].append(metrics["overall_acc"])
        print(
            f"epoch {epoch+1:02d} | loss {train_loss:.4f} | "
            f"mIoU {metrics['miou']:.4f} | F1 {metrics['macro_f1']:.4f} | "
            f"OA {metrics['overall_acc']:.4f}"
        )

        if metrics["miou"] > best["miou"]:
            best = {"miou": metrics["miou"], "epoch": epoch, "metrics": metrics}
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg), "metrics": metrics},
                os.path.join(cfg.out_dir, "best.pt"),
            )

    # Persist the run summary for the report / notebook.
    summary = {
        "config": asdict(cfg),
        "history": {k: [float(x) for x in v] for k, v in history.items()},
        "best_miou": best["miou"],
        "best_epoch": best["epoch"],
        "best_per_class_iou": [float(x) for x in best["metrics"]["per_class_iou"]],
        "best_metrics": {k: (float(v) if not isinstance(v, np.ndarray) else None)
                          for k, v in best["metrics"].items()},
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train point-supervised segmentation.")
    p.add_argument("--data-root", default=Config.data_root)
    p.add_argument("--supervision", default=Config.supervision, choices=["points", "full"])
    p.add_argument("--points-per-class", type=int, default=Config.points_per_class)
    p.add_argument("--loss", default=Config.loss, choices=["ce", "focal"])
    p.add_argument("--arch", default=Config.arch)
    p.add_argument("--encoder", default=Config.encoder)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--image-size", type=int, default=Config.image_size)
    p.add_argument("--out-dir", default=Config.out_dir)
    a = p.parse_args()
    return Config(
        data_root=a.data_root, supervision=a.supervision,
        points_per_class=a.points_per_class, loss=a.loss, arch=a.arch,
        encoder=a.encoder, epochs=a.epochs, batch_size=a.batch_size,
        image_size=a.image_size, out_dir=a.out_dir,
    )


if __name__ == "__main__":
    train_model(_parse_args())
