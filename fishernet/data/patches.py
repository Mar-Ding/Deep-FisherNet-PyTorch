from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


def make_dense_patch_boxes(
    height: int,
    width: int,
    patch_sizes: Iterable[int] = (64, 96, 128, 192),
    stride: int = 32,
    max_patches: int | None = 160,
) -> Tensor:
    """Create dense square patch boxes in image coordinates."""
    boxes: list[list[float]] = []
    for size in patch_sizes:
        if size <= 0:
            raise ValueError("patch sizes must be positive")
        patch_h = min(size, height)
        patch_w = min(size, width)
        y_starts = _starts(height, patch_h, stride)
        x_starts = _starts(width, patch_w, stride)
        for y1 in y_starts:
            for x1 in x_starts:
                boxes.append([float(x1), float(y1), float(x1 + patch_w), float(y1 + patch_h)])

    if not boxes:
        boxes = [[0.0, 0.0, float(width), float(height)]]

    out = torch.tensor(boxes, dtype=torch.float32)
    if max_patches is not None and out.shape[0] > max_patches:
        indices = torch.linspace(0, out.shape[0] - 1, steps=max_patches).round().long()
        out = out[indices]
    return out


def _starts(length: int, patch: int, stride: int) -> list[int]:
    if patch >= length:
        return [0]
    starts = list(range(0, max(length - patch + 1, 1), stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts
