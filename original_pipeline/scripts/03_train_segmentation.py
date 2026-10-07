"""Hasta-safe manifest ile segmentasyon eğitimi; test kümesine eğitim boyunca dokunmaz."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import BoMBRDataset
from src.io_utils import configure_torch, load_config, resolve_project_path, set_global_seed, write_json
from src.metrics import segmentation_metrics
from src.models import CompoundSegmentationLoss, create_segmentation_model


def evaluate(model, loader, criterion, device, amp_enabled: bool) -> dict:
    model.eval()
    losses, predictions, targets = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image)
                loss = criterion(logits, mask)
            losses.append(float(loss.item()))
            predictions.append(logits.argmax(dim=1).cpu().numpy().astype(np.uint8))
            targets.append(mask.cpu().numpy().astype(np.uint8))
    metric = segmentation_metrics(np.concatenate(predictions), np.concatenate(targets))
    metric["loss"] = float(np.mean(losses))
    return metric


def main(config_path: str, model_override: str | None) -> None:
    cfg = load_config(config_path)
    set_global_seed(int(cfg["project"]["seed"]))
    device = configure_torch()
    seg_cfg = cfg["segmentation"]
    model_name = model_override or seg_cfg["model"]

    manifest_path = resolve_project_path("splits/patient_safe_manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError("Split manifest bulunamadı. Önce 02_create_splits.py çalıştırın.")
    manifest = pd.read_csv(manifest_path)
    train_df = manifest.loc[manifest.subset == "train"].copy()
    val_df = manifest.loc[manifest.subset == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Train veya validation alt kümesi boş.")

    train_set = BoMBRDataset(train_df, int(cfg["data"]["image_size"]), train=True, include_mask=True)
    val_set = BoMBRDataset(val_df, int(cfg["data"]["image_size"]), train=False, include_mask=True)
    loader_args = dict(
        batch_size=int(seg_cfg["batch_size"]),
        num_workers=int(seg_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(seg_cfg["num_workers"]) > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = create_segmentation_model(model_name, cfg["data"]["num_classes"], bool(seg_cfg["pretrained"]))
    model = model.to(device)
    criterion = CompoundSegmentationLoss(seg_cfg["class_weights"], seg_cfg["loss_weights"]).to(device)
    optimizer = AdamW(model.parameters(), lr=float(seg_cfg["learning_rate"]), weight_decay=float(seg_cfg["weight_decay"]))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=float(seg_cfg["learning_rate"]) * 0.02)
    amp_enabled = bool(seg_cfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    output = resolve_project_path(cfg["project"]["output_dir"])
    checkpoint_dir = output / "checkpoints"
    metric_dir = output / "metrics"
    log_dir = output / "logs"
    for directory in (checkpoint_dir, metric_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    with (output / "config_used_segmentation.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)

    history: list[dict] = []
    best_dice, best_epoch, no_improvement = -np.inf, 0, 0
    best_state = None
    started = time.time()

    print(f"Train: {len(train_set)} | Validation: {len(val_set)} | Model: {model_name}")
    for epoch in range(1, int(seg_cfg["epochs"]) + 1):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{seg_cfg['epochs']}")
        for step, batch in enumerate(progress, start=1):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step((epoch - 1) + step / max(len(train_loader), 1))
            train_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{np.mean(train_losses):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        val = evaluate(model, val_loader, criterion, device, amp_enabled)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val["loss"],
            "val_mean_dice": val["mean_dice"],
            "val_mean_iou": val["mean_iou"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(" | ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()))

        if val["mean_dice"] > best_dice:
            best_dice, best_epoch, no_improvement = val["mean_dice"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_name": model_name, "epoch": epoch, "config": cfg,
                "state_dict": best_state, "validation": val,
            }, checkpoint_dir / f"segmentation_{model_name}_best.pt")
        else:
            no_improvement += 1
        if no_improvement >= int(seg_cfg["patience"]):
            print(f"Early stopping: {seg_cfg['patience']} epoch boyunca mDice iyileşmedi.")
            break

    if best_state is None:
        raise RuntimeError("Eğitim sonunda checkpoint oluşmadı.")
    write_json({
        "model_name": model_name, "best_epoch": int(best_epoch), "best_val_mean_dice": float(best_dice),
        "training_seconds": float(time.time() - started), "history": history,
    }, metric_dir / f"segmentation_training_{model_name}.json")
    pd.DataFrame(history).to_csv(metric_dir / f"segmentation_history_{model_name}.csv", index=False)
    print(f"\nTamamlandı. En iyi validation mDice={best_dice:.4f}, epoch={best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, choices=["unet_resnet34", "unetpp_resnet34", "fpn_resnet34"])
    args = parser.parse_args()
    main(args.config, args.model)
