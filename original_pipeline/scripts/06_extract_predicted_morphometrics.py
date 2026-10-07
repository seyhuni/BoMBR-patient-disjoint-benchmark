"""En iyi segmentasyon checkpoint'inden tahmin maskeleri ve morfometrik feature CSV'si üretir.

Bu betik ground-truth maskeden feature çıkarmaz; modelin öngördüğü maske üzerinden çalışır.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import BoMBRDataset, extract_morphometrics
from src.io_utils import configure_torch, load_config, resolve_project_path
from src.models import create_segmentation_model


def main(config_path: str, subset: str, model_override: str | None) -> None:
    cfg = load_config(config_path)
    device = configure_torch()
    name = model_override or cfg["segmentation"]["model"]
    out = resolve_project_path(cfg["project"]["output_dir"])
    checkpoint_path = out / "checkpoints" / f"segmentation_{name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Segmentasyon checkpoint bulunamadı: {checkpoint_path}")

    frame = pd.read_csv(resolve_project_path("splits/patient_safe_manifest.csv"))
    if subset == "development":
        frame = frame.loc[frame.subset.isin(["train", "val"])].copy()
    else:
        frame = frame.loc[frame.subset == subset].copy()
    if frame.empty:
        raise ValueError(f"{subset} için görüntü bulunamadı.")

    dataset = BoMBRDataset(frame, int(cfg["data"]["image_size"]), train=False, include_mask=False)
    loader = DataLoader(dataset, batch_size=int(cfg["segmentation"]["batch_size"]), shuffle=False, num_workers=int(cfg["segmentation"]["num_workers"]), pin_memory=device.type == "cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_segmentation_model(name, cfg["data"]["num_classes"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    records = []
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicted morphometrics ({subset})"):
            image = batch["image"].to(device, non_blocking=True)
            predictions = model(image).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for prediction in predictions:
                row = frame.iloc[cursor]
                record = extract_morphometrics(prediction)
                record.update({
                    "filename": row.filename, "patient": row.patient, "mf_grade": int(row.mf_grade),
                    "hemorrhage": int(row.hemorrhage), "subset": row.subset,
                })
                records.append(record)
                cursor += 1

    path = out / "metrics" / f"predicted_morphometrics_{subset}.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"{len(records)} görüntü için tahmin-maskesi morfometrisi kaydedildi: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subset", choices=["train", "val", "test", "development"], default="development")
    parser.add_argument("--model", default=None, choices=["unet_resnet34", "unetpp_resnet34", "fpn_resnet34"])
    args = parser.parse_args()
    main(args.config, args.subset, args.model)
