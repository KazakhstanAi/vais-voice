"""Binary detection metrics. Positive class is synthetic; higher score means synthetic."""

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


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
        "roc_auc": float(roc_auc_score(y, s)),
        "eer": eer,
        "fpr_at_tpr95": fpr95,
    }
