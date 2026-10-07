"""Gerçek test sonuçlarından yayın figürleri ve sonuç özeti üretir.

Renk: kırmızı-beyaz; metin içi font: Times New Roman. Hiçbir metrik simüle edilmez.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_config, resolve_project_path

CLASS_NAMES = ["MF-0", "MF-1", "MF-2", "MF-3"]
RED = "#8B0000"
LIGHT_RED = "#E6B8B7"
NEAR_WHITE = "#FCFCFC"
CMAP = LinearSegmentedColormap.from_list("white_red", [NEAR_WHITE, "#F4D6D4", RED])


def style() -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 9,
        "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "figure.dpi": 120,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as h:
        return json.load(h)


def save(fig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.65)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def segmentation_figure(seg: dict, destination: Path) -> None:
    class_labels = ["Bone", "Fat", "Cell", "Reticulin"]
    dice = [seg["per_class"][str(i)]["dice"] for i in range(4)]
    iou = [seg["per_class"][str(i)]["iou"] for i in range(4)]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(x - 0.18, dice, 0.36, color=RED, label="Dice")
    ax.bar(x + 0.18, iou, 0.36, color=LIGHT_RED, edgecolor=RED, linewidth=0.45, label="IoU")
    ax.set_xticks(x, class_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Held-out test segmentation performance (mDice={seg['mean_dice']:.3f}; mIoU={seg['mean_iou']:.3f})")
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()), ha="center", va="bottom", fontsize=6)
    save(fig, destination)


def grade_confusion_figure(results: dict, destination: Path) -> None:
    models = [
        ("Morphometrics\n(HGB)", results["morphometrics_only_hgb_test"]),
        ("Image-only\n(ResNet50)", results["image_only_cnn_test"]),
        ("Hybrid fusion\n(SVM)", results["hybrid_feature_fusion_svm_test"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.65))
    for ax, (title, metric) in zip(axes, models):
        cm = np.asarray(metric["confusion_matrix"], dtype=float)
        normalised = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) != 0)
        im = ax.imshow(normalised, cmap=CMAP, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                value = normalised[i, j]
                ax.text(j, i, f"{value:.0%}\n(n={int(cm[i,j])})", ha="center", va="center", fontsize=5.5, color="white" if value > 0.58 else "black")
        ax.set_title(f"{title}\nAcc={metric['accuracy']:.3f}, QWK={metric['quadratic_weighted_kappa']:.3f}", fontsize=7.2)
        ax.set_xticks(range(4), CLASS_NAMES, rotation=0)
        ax.set_yticks(range(4), CLASS_NAMES)
        ax.set_xlabel("Predicted", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("Reference", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.024, pad=0.025, label="Row-normalised proportion")
    save(fig, destination)


def grade_performance_figure(results: dict, destination: Path) -> None:
    rows = [
        ("Morphometrics\n(HGB)", results["morphometrics_only_hgb_test"]),
        ("Image-only\n(ResNet50)", results["image_only_cnn_test"]),
        ("Hybrid fusion\n(SVM)", results["hybrid_feature_fusion_svm_test"]),
    ]
    labels = [x[0] for x in rows]
    accuracy = [x[1]["accuracy"] for x in rows]
    macro_f1 = [x[1]["macro_f1"] for x in rows]
    qwk = [x[1]["quadratic_weighted_kappa"] for x in rows]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    for offset, vals, color, label in [(-0.24, accuracy, RED, "Accuracy"), (0, macro_f1, LIGHT_RED, "Macro-F1"), (0.24, qwk, "#F7E5E3", "QWK")]:
        bars = ax.bar(x + offset, vals, 0.22, color=color, edgecolor=RED, linewidth=0.4, label=label)
        for b, value in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, value+0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out test score")
    ax.set_ylim(0, 1.07)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    save(fig, destination)


def training_curves(seg_history: pd.DataFrame, grade_history: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    axes[0].plot(seg_history.epoch, seg_history.train_loss, color=RED, linewidth=1.2, label="Train loss")
    axes[0].plot(seg_history.epoch, seg_history.val_loss, color=LIGHT_RED, linewidth=1.2, label="Validation loss")
    axes[0].set_title("Segmentation training")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False); axes[0].grid(alpha=0.2, linewidth=0.5); axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].plot(grade_history.epoch, grade_history.val_qwk, color=RED, linewidth=1.2, label="Validation QWK")
    axes[1].plot(grade_history.epoch, grade_history.val_macro_f1, color=LIGHT_RED, linewidth=1.2, label="Validation macro-F1")
    axes[1].set_title("Image-grade training")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_ylim(0, 1.03)
    axes[1].legend(frameon=False); axes[1].grid(alpha=0.2, linewidth=0.5); axes[1].spines[["top", "right"]].set_visible(False)
    save(fig, destination)


def write_summary(seg: dict, grade: dict, hemorrhage: dict, split_provenance: dict, destination: Path) -> None:
    hybrid = grade["hybrid_feature_fusion_svm_test"]
    hgb = grade["morphometrics_only_hgb_test"]
    cnn = grade["image_only_cnn_test"]
    hemo_status = hemorrhage.get("reticulin_dice_by_hemorrhage", {})
    text = f"""# Reproducible Experiment Summary

## Data partition and leakage control

The locked manifest included **{split_provenance['subsets']['train']['n_images']}** training images, **{split_provenance['subsets']['val']['n_images']}** validation images, and **{split_provenance['subsets']['test']['n_images']}** held-out test images. Patient sets were pairwise disjoint: **{split_provenance['patient_sets_disjoint']}**. The test set was not used for early stopping, model selection, PCA fitting, or SVM hyperparameter optimisation.

## Segmentation

The final held-out test evaluation produced a mean Dice score of **{seg['mean_dice']:.3f}**, mean IoU of **{seg['mean_iou']:.3f}**, and pixel accuracy of **{seg['pixel_accuracy']:.3f}**. Reticulin-specific Dice was **{seg['per_class']['3']['dice']:.3f}**.

## Grade prediction

| Model | Accuracy | Balanced accuracy | Macro-F1 | QWK |
|---|---:|---:|---:|---:|
| Morphometrics-only HGB | {hgb['accuracy']:.3f} | {hgb['balanced_accuracy']:.3f} | {hgb['macro_f1']:.3f} | {hgb['quadratic_weighted_kappa']:.3f} |
| Image-only ResNet50 | {cnn['accuracy']:.3f} | {cnn['balanced_accuracy']:.3f} | {cnn['macro_f1']:.3f} | {cnn['quadratic_weighted_kappa']:.3f} |
| Hybrid Feature Fusion (ResNet50 + predicted morphology + SVM) | {hybrid['accuracy']:.3f} | {hybrid['balanced_accuracy']:.3f} | {hybrid['macro_f1']:.3f} | {hybrid['quadratic_weighted_kappa']:.3f} |

## Hemorrhage robustness

```json
{json.dumps(hemo_status, ensure_ascii=False, indent=2)}
```

> This automatically generated report contains only the outcomes produced by the current run. Do not replace missing or non-estimable results with assumed values.
"""
    destination.write_text(text, encoding="utf-8")


def main(config_path: str, model_name: str | None) -> None:
    style()
    cfg = load_config(config_path)
    name = model_name or cfg["segmentation"]["model"]
    out = resolve_project_path(cfg["project"]["output_dir"])
    metrics, figures = out / "metrics", out / "figures"
    seg = read_json(metrics / f"segmentation_test_{name}.json")
    grade = read_json(metrics / "hybrid_feature_fusion_test.json")
    hemorrhage = read_json(metrics / "hemorrhage_robustness_test.json")
    provenance = read_json(resolve_project_path("splits/split_provenance.json"))
    segmentation_figure(seg, figures / "figure_1_segmentation_per_class.png")
    grade_confusion_figure(grade, figures / "figure_2_grade_confusion_matrices.png")
    grade_performance_figure(grade, figures / "figure_3_grade_model_comparison.png")
    training_curves(
        pd.read_csv(metrics / f"segmentation_history_{name}.csv"),
        pd.read_csv(metrics / "image_grade_history.csv"),
        figures / "figure_4_training_curves.png",
    )
    write_summary(seg, grade, hemorrhage, provenance, out / "metrics" / "experiment_summary.md")
    print(f"Figürler: {figures}")
    print(f"Özet: {metrics / 'experiment_summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    main(args.config, args.model)
