"""BoMBR veri yapısı, maskeler ve metadata için ön-eğitim kalite kontrolü."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_config, project_root, read_metadata, resolve_project_path, write_json


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    data_root = resolve_project_path(cfg["data"]["root"])
    df = read_metadata(data_root, cfg["data"]["metadata_file"])
    output_metrics = resolve_project_path(cfg["project"]["output_dir"]) / "metrics"

    missing_images: list[str] = []
    missing_masks: list[str] = []
    unreadable_images: list[str] = []
    unreadable_masks: list[str] = []
    shape_mismatch: list[dict] = []
    invalid_mask_values: dict[str, list[int]] = {}
    image_shapes: Counter = Counter()
    class_pixel_counts = np.zeros(cfg["data"]["num_classes"], dtype=np.int64)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Veri doğrulama"):
        image_path = Path(row.image_path)
        mask_path = Path(row.mask_path_resolved)
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            unreadable_images.append(str(image_path))
            continue
        if mask is None:
            unreadable_masks.append(str(mask_path))
            continue

        # İndeksli maskelerin 3 kanal olarak yazılmış olma ihtimali desteklenir.
        if mask.ndim == 3:
            if np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(mask[..., 1], mask[..., 2]):
                mask = mask[..., 0]
            else:
                invalid_mask_values[str(mask_path)] = [-999]
                continue

        image_shapes[str(image.shape[:2])] += 1
        if image.shape[:2] != mask.shape[:2]:
            shape_mismatch.append({
                "image": str(image_path), "mask": str(mask_path),
                "image_shape": list(image.shape[:2]), "mask_shape": list(mask.shape[:2]),
            })

        values, counts = np.unique(mask, return_counts=True)
        invalid = values[~np.isin(values, np.arange(cfg["data"]["num_classes"]))]
        if len(invalid):
            invalid_mask_values[str(mask_path)] = [int(x) for x in invalid]
        else:
            class_pixel_counts += np.bincount(mask.ravel(), minlength=cfg["data"]["num_classes"])

    duplicate_files = df["filename"].duplicated(keep=False)
    same_patient_across_original_splits = (
        df.groupby("patient")["split"].nunique().loc[lambda s: s > 1].index.tolist()
    )

    report = {
        "n_images_metadata": int(len(df)),
        "n_patients": int(df["patient"].nunique()),
        "grade_counts": {str(k): int(v) for k, v in df["mf_grade"].value_counts().sort_index().items()},
        "original_split_counts": {str(k): int(v) for k, v in df["split"].value_counts().items()},
        "hemorrhage_counts": {str(k): int(v) for k, v in df["hemorrhage"].value_counts().sort_index().items()},
        "image_shapes": dict(image_shapes),
        "class_pixel_proportions": {
            name: float(class_pixel_counts[i] / max(1, class_pixel_counts.sum()))
            for i, name in enumerate(cfg["data"]["class_names"])
        },
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "unreadable_images": unreadable_images,
        "unreadable_masks": unreadable_masks,
        "image_mask_shape_mismatch": shape_mismatch,
        "invalid_mask_values": invalid_mask_values,
        "duplicate_filename_rows": df.loc[duplicate_files, "filename"].tolist(),
        "patients_in_both_original_train_and_test": same_patient_across_original_splits,
    }
    write_json(report, output_metrics / "dataset_validation.json")

    fatal = any([
        missing_images, missing_masks, unreadable_images, unreadable_masks,
        shape_mismatch, invalid_mask_values, report["duplicate_filename_rows"],
    ])
    print("\n" + "=" * 72)
    print("VERI DOGRULAMA OZETI")
    print("=" * 72)
    print(f"Görüntü: {len(df)} | Hasta: {df['patient'].nunique()}")
    print(f"Orijinal split: {report['original_split_counts']}")
    print(f"Grade dağılımı: {report['grade_counts']}")
    print(f"Hemoraji: {report['hemorrhage_counts']}")
    print(f"Mask class oranları: {report['class_pixel_proportions']}")
    if same_patient_across_original_splits:
        print("[UYARI] Hasta çakışması saptandı. Orijinal train/test split kullanılmayacak;")
        print("         02_create_splits.py hasta-bazlı yeni split oluşturacak.")
    if fatal:
        print("[HATA] Eğitimden önce düzeltilmesi gereken veri sorunu var.")
        print(f"Ayrıntı: {output_metrics / 'dataset_validation.json'}")
        raise SystemExit(1)
    print("[OK] Dosya ve maske kontrolleri başarılı.")
    print(f"Ayrıntılı rapor: {output_metrics / 'dataset_validation.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
