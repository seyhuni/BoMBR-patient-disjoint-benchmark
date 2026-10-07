"""Gerçek deneylerde kullanılan segmentasyon ve grade metrikleri."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)


def segmentation_metrics(predictions: np.ndarray, targets: np.ndarray, n_classes: int = 4) -> dict:
    """Tüm test pikselleri üzerinden sınıf-bazlı Dice, IoU, precision ve recall."""
    class_rows = {}
    ious, dices = [], []
    for class_id in range(n_classes):
        pred = predictions == class_id
        target = targets == class_id
        tp = int(np.logical_and(pred, target).sum())
        fp = int(np.logical_and(pred, ~target).sum())
        fn = int(np.logical_and(~pred, target).sum())
        tn = int(np.logical_and(~pred, ~target).sum())
        iou = tp / max(tp + fp + fn, 1)
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        class_rows[str(class_id)] = {
            "iou": float(iou), "dice": float(dice), "precision": float(precision),
            "recall": float(recall), "specificity": float(specificity),
            "support_pixels": int(target.sum()),
        }
        ious.append(iou)
        dices.append(dice)
    return {
        "pixel_accuracy": float((predictions == targets).mean()),
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "per_class": class_rows,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> dict:
    """Ordinal grade tahmini için accuracy, balanced accuracy, macro-F1 ve QWK."""
    labels = list(range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        str(label): {
            "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
