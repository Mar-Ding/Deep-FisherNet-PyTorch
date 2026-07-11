from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor


def _voc2007_11_point_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score, kind="mergesort")
    positives = y_true[order] > 0
    true_positives = np.cumsum(positives, dtype=np.float64)
    false_positives = np.cumsum(~positives, dtype=np.float64)
    recall = true_positives / max(1, int(positives.sum()))
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    ap = 0.0
    for threshold in np.linspace(0.0, 1.0, 11):
        eligible = precision[recall >= threshold]
        ap += float(eligible.max()) if eligible.size else 0.0
    return ap / 11.0


def average_precision_per_class(
    targets: Tensor,
    scores: Tensor,
    mode: str = "continuous",
) -> np.ndarray:
    """Compute per-class AP, ignoring targets marked -1 (VOC difficult)."""
    if mode not in {"continuous", "voc2007"}:
        raise ValueError("mode must be 'continuous' or 'voc2007'")
    y_true = targets.detach().cpu().numpy()
    y_score = scores.detach().cpu().numpy()
    aps: list[float] = []
    for class_idx in range(y_true.shape[1]):
        valid = y_true[:, class_idx] >= 0
        class_targets = y_true[valid, class_idx]
        class_scores = y_score[valid, class_idx]
        if class_targets.size == 0 or class_targets.max() == 0:
            aps.append(float("nan"))
        elif mode == "voc2007":
            aps.append(_voc2007_11_point_ap(class_targets, class_scores))
        else:
            aps.append(float(average_precision_score(class_targets, class_scores)))
    return np.asarray(aps, dtype=np.float64)


def mean_average_precision(
    targets: Tensor,
    scores: Tensor,
    mode: str = "continuous",
) -> tuple[float, np.ndarray]:
    aps = average_precision_per_class(targets, scores, mode=mode)
    return float(np.nanmean(aps)), aps
