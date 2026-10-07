"""Segmentasyon mimarisini yalnızca validation mDice temelinde seçer; test metriklerini okumaz."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_config, resolve_project_path, write_json


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    metric_dir = resolve_project_path(cfg["project"]["output_dir"]) / "metrics"
    candidates = []
    for name in ["unet_resnet34", "unetpp_resnet34", "fpn_resnet34"]:
        path = metric_dir / f"segmentation_training_{name}.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as h:
                result = json.load(h)
            candidates.append({
                "model": name,
                "best_epoch": result["best_epoch"],
                "best_val_mean_dice": result["best_val_mean_dice"],
            })
    if not candidates:
        raise FileNotFoundError("Segmentasyon eğitim sonuçları bulunamadı. Önce 03_train_segmentation.py çalıştırın.")
    selected = max(candidates, key=lambda x: x["best_val_mean_dice"])
    write_json({"selection_metric": "validation mean Dice", "candidates": candidates, "selected": selected}, metric_dir / "selected_segmentation_model.json")
    print("Validation-only model selection")
    for c in sorted(candidates, key=lambda x: x["best_val_mean_dice"], reverse=True):
        print(f"{c['model']:<18} mDice={c['best_val_mean_dice']:.4f} at epoch {c['best_epoch']}")
    print(f"\nSELECTED={selected['model']}")
    print("Bu noktadan sonra SADECE seçili model için 04_evaluate_segmentation.py ile test değerlendirmesi yapın.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
