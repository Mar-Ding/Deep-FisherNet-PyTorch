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
    coordinate_mode: str = "half_open",
) -> Tensor:
    """Create dense square patch boxes in image coordinates."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if coordinate_mode not in {"half_open", "caffe"}:
        raise ValueError("coordinate_mode must be 'half_open' or 'caffe'")

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
                if coordinate_mode == "caffe":
                    x2 = x1 + patch_w - 1
                    y2 = y1 + patch_h - 1
                else:
                    x2 = x1 + patch_w
                    y2 = y1 + patch_h
                boxes.append([float(x1), float(y1), float(x2), float(y2)])

    if not boxes:
        if coordinate_mode == "caffe":
            boxes = [[0.0, 0.0, float(width - 1), float(height - 1)]]
        else:
            boxes = [[0.0, 0.0, float(width), float(height)]]

    out = torch.tensor(boxes, dtype=torch.float32)
    if max_patches is not None and out.shape[0] > max_patches:
        indices = torch.linspace(0, out.shape[0] - 1, steps=max_patches).round().long()
        out = out[indices]
    return out


def transform_patch_boxes(
    boxes: Tensor,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    horizontal_flip: bool = False,
    coordinate_mode: str = "half_open",
) -> Tensor:
    """Map boxes from an original image into its resized/flipped view."""
    if coordinate_mode not in {"half_open", "caffe"}:
        raise ValueError("coordinate_mode must be 'half_open' or 'caffe'")
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("source and target dimensions must be positive")

    mapped = boxes.to(dtype=torch.float32).clone()
    if mapped.numel() == 0:
        return mapped

    if horizontal_flip:
        x1 = mapped[:, 0].clone()
        x2 = mapped[:, 2].clone()
        if coordinate_mode == "caffe":
            mapped[:, 0] = source_w - 1 - x2
            mapped[:, 2] = source_w - 1 - x1
        else:
            mapped[:, 0] = source_w - x2
            mapped[:, 2] = source_w - x1

    mapped[:, [0, 2]] *= target_w / float(source_w)
    mapped[:, [1, 3]] *= target_h / float(source_h)
    if coordinate_mode == "caffe":
        mapped[:, [0, 2]].clamp_(0.0, float(target_w - 1))
        mapped[:, [1, 3]].clamp_(0.0, float(target_h - 1))
    else:
        mapped[:, [0, 2]].clamp_(0.0, float(target_w))
        mapped[:, [1, 3]].clamp_(0.0, float(target_h))
    return mapped


def _starts(length: int, patch: int, stride: int) -> list[int]:
    if patch >= length:
        return [0]
    starts = list(range(0, max(length - patch + 1, 1), stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts
