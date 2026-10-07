"""Hybrid Feature Fusion: eğitilmiş CNN embedding + predicted-mask morphometrics + RBF-SVM.

Test kümesi, hiperparametre seçimi ve PCA fit işlemlerinde kullanılmaz; yalnızca final raporlama içindir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import BoMBRDataset
from src.io_utils import configure_torch, load_config, resolve_project_path, write_json
from src.metrics import classification_metrics
from src.models import ImageGradeClassifier

MORPH_COLUMNS = [
    "pct_bone", "pct_fat", "pct_cell", "pct_reticulin", "ret_component_count",
    "ret_mean_component_area", "ret_median_elongation", "ret_elongation_sd",
    "ret_mean_perimeter", "ret_to_cell_ratio", "ret_lacunarity_proxy", "ret_fractal_proxy",
]


def extract_cnn_embeddings(model, frame: pd.DataFrame, image_size: int, batch_size: int, workers: int, device) -> tuple[np.ndarray, np.ndarray]:
    dataset = BoMBRDataset(frame, image_size=image_size, train=False, include_mask=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model.eval()
    embeddings, predictions = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="CNN embedding çıkarımı"):
            image = batch["image"].to(device, non_blocking=True)
            features = model.extract_features(image)
            logits = model.classifier(features)
            embeddings.append(features.cpu().numpy())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return np.vstack(embeddings), np.asarray(predictions)


def morph_matrix(path: Path, reference: pd.DataFrame) -> np.ndarray:
    features = pd.read_csv(path)
    merged = reference[["filename"]].merge(features, on="filename", how="left", validate="one_to_one")
    if merged[MORPH_COLUMNS].isna().any().any():
        missing = merged.loc[merged[MORPH_COLUMNS].isna().any(axis=1), "filename"].tolist()
        raise ValueError(f"Morfometrik feature eksik: {missing[:5]}")
    return merged[MORPH_COLUMNS].to_numpy(dtype=np.float32)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    device = configure_torch()
    out = resolve_project_path(cfg["project"]["output_dir"])
    manifest = pd.read_csv(resolve_project_path("splits/patient_safe_manifest.csv"))
    development = manifest.loc[manifest.subset.isin(["train", "val"])].copy().reset_index(drop=True)
    test = manifest.loc[manifest.subset == "test"].copy().reset_index(drop=True)
    if development.empty or test.empty:
        raise ValueError("Development veya test alt kümesi boş.")

    morph_dev_path = out / "metrics" / "predicted_morphometrics_development.csv"
    morph_test_path = out / "metrics" / "predicted_morphometrics_test.csv"
    if not morph_dev_path.exists() or not morph_test_path.exists():
        raise FileNotFoundError("Önce 06_extract_predicted_morphometrics.py ile development ve test feature'larını üretin.")

    checkpoint_path = out / "checkpoints" / "image_grade_resnet50_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("image_grade_resnet50_best.pt bulunamadı. Önce 05_train_image_grading.py çalıştırın.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cnn = ImageGradeClassifier(num_classes=4, pretrained=False).to(device)
    cnn.load_state_dict(checkpoint["state_dict"])

    image_size = int(cfg["grading"]["image_size"])
    batch_size = int(cfg["grading"]["batch_size"])
    workers = int(cfg["grading"]["num_workers"])
    dev_cnn, dev_cnn_pred = extract_cnn_embeddings(cnn, development, image_size, batch_size, workers, device)
    test_cnn, test_cnn_pred = extract_cnn_embeddings(cnn, test, image_size, batch_size, workers, device)
    dev_morph = morph_matrix(morph_dev_path, development)
    test_morph = morph_matrix(morph_test_path, test)
    y_dev, y_test = development.mf_grade.to_numpy(), test.mf_grade.to_numpy()

    # Klasik, açıklanabilir morfometri baseline'ı.
    baseline = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, random_state=cfg["project"]["seed"])),
    ])
    baseline.fit(dev_morph, y_dev)
    baseline_pred = baseline.predict(test_morph)

    # Fused features: 2048-dim CNN image representation + 12-dim predicted morphology.
    x_dev = np.hstack([dev_cnn, dev_morph])
    x_test = np.hstack([test_cnn, test_morph])
    n_groups = development.patient.nunique()
    folds = min(int(cfg["grading"]["fusion"]["cv_folds"]), n_groups)
    if folds < 2:
        raise ValueError("SVM nested CV için en az iki development hastası gerekir.")
    fusion_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=float(cfg["grading"]["fusion"]["pca_components"]), svd_solver="full", random_state=cfg["project"]["seed"])),
        ("svm", SVC(kernel="rbf", class_weight="balanced")),
    ])
    grid = {
        "svm__C": cfg["grading"]["fusion"]["svm_c_grid"],
        "svm__gamma": cfg["grading"]["fusion"]["svm_gamma_grid"],
    }
    search = GridSearchCV(
        fusion_pipeline, grid, scoring="f1_macro", cv=GroupKFold(n_splits=folds),
        n_jobs=-1, refit=True, return_train_score=True,
    )
    search.fit(x_dev, y_dev, groups=development.patient.to_numpy())
    fusion_pred = search.predict(x_test)

    result = {
        "protocol": {
            "development_samples": int(len(development)), "test_samples": int(len(test)),
            "development_patients": int(development.patient.nunique()), "test_patients": int(test.patient.nunique()),
            "feature_concatenation": "2048-dimensional ResNet50 embedding + 12 predicted-mask morphometric features",
            "pca_variance_target": float(cfg["grading"]["fusion"]["pca_components"]),
            "svm_search": {"cv": f"GroupKFold({folds})", "best_params": search.best_params_, "best_cv_macro_f1": float(search.best_score_)},
        },
        "morphometrics_only_hgb_test": classification_metrics(y_test, baseline_pred),
        "image_only_cnn_test": classification_metrics(y_test, test_cnn_pred),
        "hybrid_feature_fusion_svm_test": classification_metrics(y_test, fusion_pred),
    }
    write_json(result, out / "metrics" / "hybrid_feature_fusion_test.json")
    pd.DataFrame({
        "filename": test.filename, "patient": test.patient, "true_grade": y_test,
        "morph_hgb_prediction": baseline_pred, "cnn_prediction": test_cnn_pred, "hybrid_svm_prediction": fusion_pred,
    }).to_csv(out / "predictions" / "grade_test_predictions.csv", index=False)
    joblib.dump({"baseline": baseline, "hybrid_grid_search": search, "morph_columns": MORPH_COLUMNS}, out / "checkpoints" / "hybrid_feature_fusion.joblib")

    print("=" * 72)
    print("HELD-OUT TEST | GRADE PREDICTION")
    print("=" * 72)
    for name, values in result.items():
        if name == "protocol":
            continue
        print(f"{name}: Acc={values['accuracy']:.4f}, BalAcc={values['balanced_accuracy']:.4f}, Macro-F1={values['macro_f1']:.4f}, QWK={values['quadratic_weighted_kappa']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
