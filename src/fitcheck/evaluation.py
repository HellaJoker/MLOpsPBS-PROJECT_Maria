"""One definition of the evaluation metrics logged to MLflow.

Every run (baseline, optimized, hold-out comparison) logs metrics through
`classification_metrics`, so the metric *names* are identical across runs and the
MLflow "Compare" view can line them up side by side. What differs between runs is
recorded as params/tags (`model_variant`, `eval_set`), never baked into the metric name.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, recall_score)

from fitcheck.features import CLASS_LABELS

LABELS = list(CLASS_LABELS)  # [0, 1, 2]


def classification_metrics(y_true, y_pred, proba=None, near_boundary=None):
    """Standard metric dict for the fit classifier (same keys for every model).

    y_true, y_pred : class ids (0 Too Small, 1 Good Fit, 2 Too Large)
    proba          : optional predict_proba output -> adds `log_loss`
    near_boundary  : optional boolean mask of hard cases -> adds `accuracy_near_boundary`
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall_too_small": recalls[0],
        "recall_good_fit": recalls[1],
        "recall_too_large": recalls[2],
    }
    if proba is not None:
        # lower = better-calibrated confidence
        metrics["log_loss"] = log_loss(y_true, proba, labels=LABELS)
    if near_boundary is not None:
        mask = np.asarray(near_boundary, dtype=bool)
        metrics["accuracy_near_boundary"] = accuracy_score(y_true[mask], y_pred[mask])
    return {k: float(v) for k, v in metrics.items()}
