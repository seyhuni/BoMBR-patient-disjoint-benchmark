"""Seçilen checkpoint'i yalnızca test alt kümesinde değerlendirir ve overlay çıktıları üretir."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import BoMBRDataset
from src.io_utils import configure_torch, load_config, resolve_project_path, write_json
from src.metrics import segmentation_metrics
from src.models import create_segmentation_model

PALETTE_BGR = np.array([[40, 40, 160], [100, 190, 230], [235, 235, 235], [35, 35, 180]], dtype=np.uint8)


def main(config_path: str, model_name: str | None) -> None:
    cfg = load_config(config_path)
    device = configure_torch()
    name = model_name or cfg["segmentation"]["model"]
    output = resolve_project_path(cfg["project"]["output_dir"])
    checkpoint_path = output / "checkpoints" / f"segmentation_{name}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")

    manifest = pd.read_csv(resolve_project_path("splits/patient_safe_manifest.csv"))
    test_df = manifest.loc[manifest.subset == "test"].copy()
    if test_df.empty:
        raise ValueError("Test alt kümesi boş.")
    dataset = BoMBRDataset(test_df, int(cfg["data"]["image_size"]), train=False, include_mask=True)
    loader = DataLoader(dataset, batch_size=int(cfg["segmentation"]["batch_size"]), shuffle=False, num_workers=int(cfg["segmentation"]["num_workers"]), pin_memory=device.type == "cuda")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_segmentation_model(name, cfg["data"]["num_classes"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    preds, targets = [], []
    per_image = []
    overlay_dir = output / "predictions" / f"segmentation_{name}"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    image_cursor = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test segmentasyon"):
            image = batch["image"].to(device, non_blocking=True)
            logits = model(image)
            prediction = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            target = batch["mask"].numpy().astype(np.uint8)
            preds.append(prediction)
            targets.append(target)
            for b in range(len(prediction)):
                image_metric = segmentation_metrics(prediction[b:b+1], target[b:b+1])
                image_metric["filename"] = batch["filename"][b]
                image_metric["patient"] = batch["patient"][b]
                per_image.append(image_metric)
                if image_cursor < 12:
                    _save_overlay(dataset.frame.iloc[image_cursor], prediction[b], target[b], overlay_dir)
                image_cursor += 1

    overall = segmentation_metrics(np.concatenate(preds), np.concatenate(targets))
    overall.update({
        "model_name": name,
        "checkpoint": str(checkpoint_path),
        "n_test_images": int(len(test_df)),
        "evaluation_subset": "held_out_test",
    })
    write_json(overall, output / "metrics" / f"segmentation_test_{name}.json")
    pd.json_normalize(per_image, sep="_").to_csv(output / "metrics" / f"segmentation_per_image_{name}.csv", index=False)
    print("=" * 72)
    print(f"TEST mDice={overall['mean_dice']:.4f} | mIoU={overall['mean_iou']:.4f} | Pixel Acc={overall['pixel_accuracy']:.4f}")
    for index, class_name in enumerate(cfg["data"]["class_names"]):
        values = overall["per_class"][str(index)]
        print(f"{class_name:<10}: Dice={values['dice']:.4f} | IoU={values['iou']:.4f} | Recall={values['recall']:.4f}")


def _save_overlay(row: pd.Series, prediction: np.ndarray, target: np.ndarray, out_dir: Path) -> None:
    original = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
    original = cv2.resize(original, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred_color = PALETTE_BGR[prediction]
    truth_color = PALETTE_BGR[target]
    overlay_pred = cv2.addWeighted(original, 0.58, pred_color, 0.42, 0)
    overlay_truth = cv2.addWeighted(original, 0.58, truth_color, 0.42, 0)
    canvas = np.hstack([original, overlay_truth, overlay_pred])
    cv2.putText(canvas, "Original", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    w = prediction.shape[1]
    cv2.putText(canvas, "Ground truth", (w + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "Prediction", (2*w + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out_dir / f"overlay_{row.filename}"), canvas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, choices=["unet_resnet34", "unetpp_resnet34", "fpn_resnet34"])
    args = parser.parse_args()
    main(args.config, args.model)
