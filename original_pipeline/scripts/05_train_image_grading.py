"""ResNet50 görüntü-only MF-grade sınıflandırıcısını train/val üzerinde eğitir."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import BoMBRDataset
from src.io_utils import configure_torch, load_config, resolve_project_path, set_global_seed, write_json
from src.metrics import classification_metrics
from src.models import ImageGradeClassifier


def evaluate(model, loader, criterion, device, amp_enabled: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    losses, predictions, labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Grade validation", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["grade"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, target)
            losses.append(float(loss.item()))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(target.cpu().tolist())
    metrics = classification_metrics(np.asarray(labels), np.asarray(predictions))
    metrics["loss"] = float(np.mean(losses))
    return metrics, np.asarray(labels), np.asarray(predictions)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_global_seed(int(cfg["project"]["seed"]))
    device = configure_torch()
    grading = cfg["grading"]
    manifest = pd.read_csv(resolve_project_path("splits/patient_safe_manifest.csv"))
    train_df = manifest.loc[manifest.subset == "train"].copy()
    val_df = manifest.loc[manifest.subset == "val"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation verisi bulunamadı.")

    train_set = BoMBRDataset(train_df, int(grading["image_size"]), train=True, include_mask=False)
    val_set = BoMBRDataset(val_df, int(grading["image_size"]), train=False, include_mask=False)
    sample_weights = train_df.mf_grade.map(
        {grade: 1.0 / count for grade, count in train_df.mf_grade.value_counts().items()}
    ).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader_args = dict(
        batch_size=int(grading["batch_size"]), num_workers=int(grading["num_workers"]),
        pin_memory=device.type == "cuda", persistent_workers=int(grading["num_workers"]) > 0,
    )
    train_loader = DataLoader(train_set, sampler=sampler, shuffle=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = ImageGradeClassifier(num_classes=4, pretrained=True).to(device)
    present = np.sort(train_df.mf_grade.unique())
    class_weights_np = compute_class_weight("balanced", classes=present, y=train_df.mf_grade.to_numpy())
    class_weights = np.ones(4, dtype=np.float32)
    class_weights[present] = class_weights_np
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = AdamW(model.parameters(), lr=float(grading["learning_rate"]), weight_decay=float(grading["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(grading["cnn_epochs"]), eta_min=float(grading["learning_rate"]) * 0.05)
    amp_enabled = bool(grading["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    out = resolve_project_path(cfg["project"]["output_dir"])
    checkpoint_dir, metric_dir = out / "checkpoints", out / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    history, best_state = [], None
    best_qwk, best_epoch, stale = -np.inf, 0, 0
    start = time.time()

    print(f"Image grade classifier | train={len(train_set)} | val={len(val_set)}")
    for epoch in range(1, int(grading["cnn_epochs"]) + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}/{grading['cnn_epochs']}"):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["grade"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        scheduler.step()
        val, _, _ = evaluate(model, val_loader, criterion, device, amp_enabled)
        row = {
            "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val["loss"],
            "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
            "val_qwk": val["quadratic_weighted_kappa"], "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if val["quadratic_weighted_kappa"] > best_qwk:
            best_qwk, best_epoch, stale = val["quadratic_weighted_kappa"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"epoch": epoch, "config": cfg, "state_dict": best_state, "validation": val}, checkpoint_dir / "image_grade_resnet50_best.pt")
        else:
            stale += 1
        if stale >= int(grading["patience"]):
            print(f"Early stopping: {grading['patience']} epoch QWK iyileşmedi.")
            break

    write_json({"best_epoch": int(best_epoch), "best_val_qwk": float(best_qwk), "seconds": float(time.time() - start), "history": history}, metric_dir / "image_grade_training.json")
    pd.DataFrame(history).to_csv(metric_dir / "image_grade_history.csv", index=False)
    print(f"\nTamamlandı. En iyi validation QWK={best_qwk:.4f}, epoch={best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
