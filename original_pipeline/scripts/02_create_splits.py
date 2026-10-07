"""Hasta bazlı, sızıntısız train/validation/test manifesti üretir."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_config, project_root, read_metadata, resolve_project_path, write_json


def distribution_distance(reference: pd.DataFrame, candidate: pd.DataFrame) -> float:
    """Grade ve hemoraji dağılımlarındaki mutlak farkların toplamı."""
    grade_ref = reference["mf_grade"].value_counts(normalize=True).reindex(range(4), fill_value=0)
    grade_cand = candidate["mf_grade"].value_counts(normalize=True).reindex(range(4), fill_value=0)
    hem_ref = reference["hemorrhage"].value_counts(normalize=True).reindex([0, 1], fill_value=0)
    hem_cand = candidate["hemorrhage"].value_counts(normalize=True).reindex([0, 1], fill_value=0)
    return float(np.abs(grade_ref - grade_cand).sum() + 0.35 * np.abs(hem_ref - hem_cand).sum())


def candidate_group_holdout(
    df: pd.DataFrame, test_size: float, n_candidates: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Çoklu GroupShuffleSplit adayları arasında en dengeli split'i seçer."""
    groups = df["patient"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=n_candidates, test_size=test_size, random_state=seed)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for train_idx, holdout_idx in splitter.split(df, groups=groups):
        train_df = df.iloc[train_idx]
        holdout_df = df.iloc[holdout_idx]
        if train_df["mf_grade"].nunique() < 4 or holdout_df["mf_grade"].nunique() < 3:
            continue
        actual_size = len(holdout_df) / len(df)
        score = 4.0 * abs(actual_size - test_size) + distribution_distance(df, holdout_df)
        if best is None or score < best[2]:
            best = (train_idx, holdout_idx, score)
    if best is None:
        raise RuntimeError(
            "Tüm grade sınıflarını yeterince temsil eden group split üretilemedi. "
            "Hasta/grade dağılımını kontrol edin veya test_size değerini azaltın."
        )
    tr, ho, score = best
    return tr, ho, {"balance_score": float(score), "actual_holdout_fraction": float(len(ho) / len(df))}


def patient_sets_disjoint(manifest: pd.DataFrame) -> bool:
    patients = {subset: set(g["patient"]) for subset, g in manifest.groupby("subset")}
    return (
        not (patients["train"] & patients["val"])
        and not (patients["train"] & patients["test"])
        and not (patients["val"] & patients["test"])
    )


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    data_root = resolve_project_path(cfg["data"]["root"])
    df = read_metadata(data_root, cfg["data"]["metadata_file"])
    split_cfg = cfg["split"]

    # Orijinal split ancak hasta kümeleri ayrık ise korunur.
    original_patient_split_counts = df.groupby("patient")["split"].nunique()
    originals_disjoint = bool((original_patient_split_counts <= 1).all())
    use_original_test = split_cfg["strategy"] == "original_if_disjoint" and originals_disjoint

    if use_original_test:
        train_pool = df.loc[df["split"] == "train"].copy()
        test_df = df.loc[df["split"] == "test"].copy()
        train_indices, val_indices, val_info = candidate_group_holdout(
            train_pool,
            test_size=float(split_cfg["val_size_within_train"]),
            n_candidates=int(split_cfg["candidate_splits"]),
            seed=int(split_cfg["random_state"]),
        )
        train_df = train_pool.iloc[train_indices].copy()
        val_df = train_pool.iloc[val_indices].copy()
        test_info = {"strategy": "original_disjoint_test", "actual_test_fraction": float(len(test_df) / len(df))}
    else:
        train_val_idx, test_idx, test_info = candidate_group_holdout(
            df,
            test_size=float(split_cfg["test_size"]),
            n_candidates=int(split_cfg["candidate_splits"]),
            seed=int(split_cfg["random_state"]),
        )
        train_val_df = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_indices, val_indices, val_info = candidate_group_holdout(
            train_val_df,
            test_size=float(split_cfg["val_size_within_train"]),
            n_candidates=int(split_cfg["candidate_splits"]),
            seed=int(split_cfg["random_state"]) + 1,
        )
        train_df = train_val_df.iloc[train_indices].copy()
        val_df = train_val_df.iloc[val_indices].copy()
        test_info["strategy"] = "recomputed_group_holdout"

    train_df["subset"] = "train"
    val_df["subset"] = "val"
    test_df["subset"] = "test"
    manifest = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    if len(manifest) != len(df) or not patient_sets_disjoint(manifest):
        raise RuntimeError("Split doğrulaması başarısız: görüntü kaybı veya hasta sızıntısı var.")

    split_dir = resolve_project_path("splits")
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "patient_safe_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    per_subset = {}
    for subset, part in manifest.groupby("subset"):
        per_subset[subset] = {
            "n_images": int(len(part)),
            "n_patients": int(part["patient"].nunique()),
            "patients": sorted(part["patient"].unique().tolist()),
            "grade_counts": {str(k): int(v) for k, v in part["mf_grade"].value_counts().sort_index().items()},
            "hemorrhage_counts": {str(k): int(v) for k, v in part["hemorrhage"].value_counts().sort_index().items()},
        }
    report = {
        "source_records": int(len(df)),
        "original_splits_patient_disjoint": originals_disjoint,
        "test_selection": test_info,
        "validation_selection": val_info,
        "patient_sets_disjoint": patient_sets_disjoint(manifest),
        "subsets": per_subset,
        "random_state": int(split_cfg["random_state"]),
    }
    write_json(report, split_dir / "split_provenance.json")

    print("=" * 72)
    print("HASTA-BAZLI, SIZINTISIZ SPLIT OLUSTURULDU")
    print("=" * 72)
    for subset in ["train", "val", "test"]:
        stats = per_subset[subset]
        print(f"{subset:>5}: {stats['n_images']:3d} görüntü | {stats['n_patients']:2d} hasta | grade={stats['grade_counts']}")
    print(f"\nManifest: {manifest_path}")
    print(f"Provenance: {split_dir / 'split_provenance.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
