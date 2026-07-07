from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as TF

from .patches import make_dense_patch_boxes


PASCAL_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(PASCAL_CLASSES)}


class VOCClassification(Dataset):
    """VOCDetection wrapper that returns multi-label targets and dense patch boxes."""

    def __init__(
        self,
        root: str | Path,
        year: str = "2007",
        image_set: str = "trainval",
        image_size: int = 448,
        train_scales: tuple[int, ...] | None = None,
        hflip_prob: float = 0.0,
        patch_sizes: tuple[int, ...] = (96, 128, 192, 256),
        patch_stride: int = 64,
        max_patches: int = 160,
        download: bool = False,
    ) -> None:
        self.dataset = VOCDetection(
            root=str(root),
            year=year,
            image_set=image_set,
            download=download,
        )
        self.image_size = image_size
        self.train_scales = train_scales
        self.hflip_prob = hflip_prob
        self.patch_sizes = patch_sizes
        self.patch_stride = patch_stride
        self.max_patches = max_patches

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, target = self.dataset[index]
        image = image.convert("RGB")
        image_size = random.choice(self.train_scales) if self.train_scales else self.image_size
        image = TF.resize(image, [image_size, image_size], interpolation=Image.BILINEAR)
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            image = TF.hflip(image)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(
            tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        labels = self._multi_hot(target)
        boxes = make_dense_patch_boxes(
            height=image_size,
            width=image_size,
            patch_sizes=self.patch_sizes,
            stride=self.patch_stride,
            max_patches=self.max_patches,
        )
        return {
            "image": tensor,
            "labels": labels,
            "boxes": boxes,
            "image_id": target["annotation"]["filename"],
        }

    @staticmethod
    def _multi_hot(target: dict[str, Any]) -> Tensor:
        labels = torch.zeros(len(PASCAL_CLASSES), dtype=torch.float32)
        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        for obj in objects:
            class_name = obj["name"]
            if class_name in CLASS_TO_INDEX:
                labels[CLASS_TO_INDEX[class_name]] = 1.0
        return labels


def collate_voc_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([sample["image"] for sample in batch], dim=0),
        "labels": torch.stack([sample["labels"] for sample in batch], dim=0),
        "boxes": [sample["boxes"] for sample in batch],
        "image_ids": [sample["image_id"] for sample in batch],
    }
