"""평가 지표 계산."""
import numpy as np
from sklearn.metrics import (
    f1_score, balanced_accuracy_score,
    precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray | None = None,
                    label_names: list[str] | None = None) -> dict:
    labels = sorted(set(y_true))
    label_names = label_names or [str(l) for l in labels]

    metrics: dict = {
        "macro_f1":      f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc":  balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":  recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    # per-class F1
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    for name, score in zip(label_names, per_class):
        metrics[f"f1_{name}"] = score

    # ROC-AUC
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
        except Exception:
            metrics["roc_auc"] = float("nan")

    # 혼동행렬 (flat)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i, true_name in enumerate(label_names):
        for j, pred_name in enumerate(label_names):
            metrics[f"cm_{true_name}_pred_{pred_name}"] = int(cm[i, j])

    return metrics
