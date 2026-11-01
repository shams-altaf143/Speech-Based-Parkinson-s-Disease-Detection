from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score)

from .discovery import CLASS_NAMES


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def summarise(y: np.ndarray, p_pd: np.ndarray, thr: float = 0.5) -> dict:
    """Metrics over (aggregated) predictions. p_pd = P(PD)."""
    pred = (p_pd >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "auc": safe_auc(y, p_pd),
        "confusion_matrix": cm.tolist(),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),   # PD accuracy
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),   # HC accuracy
        "pd_confidence": float(p_pd[y == 1].mean()) if (y == 1).any() else float("nan"),
        "hc_conf_error": float(p_pd[y == 0].mean()) if (y == 0).any() else float("nan"),
        "report": classification_report(y, pred, labels=[0, 1], target_names=list(CLASS_NAMES),
                                        output_dict=True, zero_division=0),
    }
