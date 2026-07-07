from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor


def average_precision_per_class(targets: Tensor, scores: Tensor) -> np.ndarray:
    """Compute VOC-style one-vs-rest AP for every class."""
    y_true = targets.detach().cpu().numpy()
    y_score = scores.detach().cpu().numpy()
    aps: list[float] = []
    for class_idx in range(y_true.shape[1]):
        if y_true[:, class_idx].max() == 0:
            aps.append(float("nan"))
        else:
            aps.append(float(average_precision_score(y_true[:, class_idx], y_score[:, class_idx])))
    return np.asarray(aps, dtype=np.float64)


def mean_average_precision(targets: Tensor, scores: Tensor) -> tuple[float, np.ndarray]:
    aps = average_precision_per_class(targets, scores)
    return float(np.nanmean(aps)), aps
