"""Hemoraji alt gruplarında grade ve segmentasyon performansını raporlar.

Önemli: Bir alt grupta yeterli örnek yoksa betik etki büyüklüğü/p-değeri üretmez;
'not_estimable' raporlar. Bu davranış bilimsel olarak zorunludur.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_config, resolve_project_path, write_json
from src.metrics import classification_metrics


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    return float((np.sum(x[:, None] > y[None, :]) - np.sum(x[:, None] < y[None, :])) / (len(x) * len(y)))


def main(config_path: str, model_name: str | None) -> None:
    cfg = load_config(config_path)
    name = model_name or cfg["segmentation"]["model"]
    out = resolve_project_path(cfg["project"]["output_dir"])
    manifest = pd.read_csv(resolve_project_path("splits/patient_safe_manifest.csv"))
    test_metadata = manifest.loc[manifest.subset == "test", ["filename", "hemorrhage"]]
    grade_pred = pd.read_csv(out / "predictions" / "grade_test_predictions.csv")
    seg_per_image = pd.read_csv(out / "metrics" / f"segmentation_per_image_{name}.csv")

    grade_data = grade_pred.merge(test_metadata, on="filename", validate="one_to_one")
    seg_data = seg_per_image.merge(test_metadata, on="filename", validate="one_to_one")
    groups = {str(k): int(v) for k, v in test_metadata.hemorrhage.value_counts().sort_index().items()}
    result: dict = {"test_hemorrhage_counts": groups}

    # Grade performance per subgroup only where class support makes metrics meaningful.
    grade_subgroups = {}
    for status, label in [(0, "non_hemorrhagic"), (1, "hemorrhagic")]:
        part = grade_data.loc[grade_data.hemorrhage == status]
        if len(part) >= 5 and part.true_grade.nunique() >= 2:
            grade_subgroups[label] = classification_metrics(part.true_grade.to_numpy(), part.hybrid_svm_prediction.to_numpy())
            grade_subgroups[label]["n_images"] = int(len(part))
        else:
            grade_subgroups[label] = {"status": "not_estimable", "n_images": int(len(part)), "reason": "insufficient sample size or grade variation"}
    result["hybrid_grade_by_hemorrhage"] = grade_subgroups

    metric_column = "per_class_3_dice"
    if metric_column not in seg_data.columns:
        raise ValueError(f"Segmentasyon metrik sütunu bulunamadı: {metric_column}")
    non_hemo = seg_data.loc[seg_data.hemorrhage == 0, metric_column].dropna().to_numpy()
    hemo = seg_data.loc[seg_data.hemorrhage == 1, metric_column].dropna().to_numpy()
    if len(non_hemo) >= 5 and len(hemo) >= 5:
        stat, pvalue = mannwhitneyu(non_hemo, hemo, alternative="two-sided")
        result["reticulin_dice_by_hemorrhage"] = {
            "status": "estimable", "non_hemorrhagic_n": int(len(non_hemo)), "hemorrhagic_n": int(len(hemo)),
            "non_hemorrhagic_median": float(np.median(non_hemo)), "hemorrhagic_median": float(np.median(hemo)),
            "mann_whitney_u": float(stat), "p_value": float(pvalue), "cliffs_delta": cliffs_delta(hemo, non_hemo),
        }
    else:
        result["reticulin_dice_by_hemorrhage"] = {
            "status": "not_estimable", "non_hemorrhagic_n": int(len(non_hemo)), "hemorrhagic_n": int(len(hemo)),
            "reason": "both groups require at least 5 images",
        }
    write_json(result, out / "metrics" / "hemorrhage_robustness_test.json")
    print("Hemoraji analizi kaydedildi:", out / "metrics" / "hemorrhage_robustness_test.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    main(args.config, args.model)
