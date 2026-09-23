"""Binary detection metrics. Positive class is synthetic; higher score means synthetic."""

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def select_eer_threshold(labels: list[int], scores: list[float]) -> dict[str, float]:
    """Choose a deployment threshold from validation data only."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0 or len(y) != len(s) or not np.isin(y, [0, 1]).all():
        raise ValueError("Valid nonempty binary labels and matching scores required")
    if len(np.unique(y)) < 2 or len(np.unique(s)) < 2 or not np.isfinite(s).all():
        raise ValueError("Threshold selection requires two classes and nonconstant finite scores")
    fpr, tpr, thresholds = roc_curve(y, s, drop_intermediate=False)
    index = int(np.argmin(np.abs(fpr - (1.0 - tpr))))
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        finite = np.flatnonzero(np.isfinite(thresholds))
        if not len(finite):
            raise ValueError("No finite validation threshold")
        index = int(finite[0])
        threshold = float(thresholds[index])
    return {
        "threshold": threshold,
        "validation_fpr": float(fpr[index]),
        "validation_tpr": float(tpr[index]),
        "validation_eer_gap": float(abs(fpr[index] - (1.0 - tpr[index]))),
    }


def detection_metrics(labels: list[int], scores: list[float]) -> dict[str, Any]:
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("Nonempty one-dimensional labels and scores of equal length required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Labels must be 0/1 and scores must be finite")
    counts = {"n": len(y), "real": int((y == 0).sum()), "synthetic": int((y == 1).sum())}
    if len(np.unique(y)) < 2:
        return {
            **counts,
            "status": "undefined_single_class",
            "roc_auc": None,
            "eer": None,
            "fpr_at_tpr95": None,
        }
    unique_scores = np.unique(s)
    if len(unique_scores) < 2:
        return {
            **counts,
            "status": "degenerate_constant_scores",
            "unique_scores": 1,
            "score_min": float(unique_scores[0]),
            "score_max": float(unique_scores[0]),
            "roc_auc": None,
            "eer": None,
            "fpr_at_tpr95": None,
        }
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    difference = fpr + tpr - 1
    right = int(np.flatnonzero(difference >= 0)[0])
    if difference[right] == 0:
        eer = float(fpr[right])
    else:
        left = right - 1
        fraction = -difference[left] / (difference[right] - difference[left])
        eer = float(fpr[left] + fraction * (fpr[right] - fpr[left]))
    # First point attaining target recall; interpolate only the crossing segment.
    target = 0.95
    right = int(np.flatnonzero(tpr >= target)[0])
    left = right - 1
    fraction = (target - tpr[left]) / (tpr[right] - tpr[left])
    fpr95 = float(fpr[left] + fraction * (fpr[right] - fpr[left]))
    return {
        **counts,
        "status": "ok",
        "unique_scores": int(len(unique_scores)),
        "score_min": float(s.min()),
        "score_max": float(s.max()),
        "roc_auc": float(roc_auc_score(y, s)),
        "eer": eer,
        "fpr_at_tpr95": fpr95,
    }
