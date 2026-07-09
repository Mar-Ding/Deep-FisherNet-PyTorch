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


def resize_image(
    image: Image.Image,
    image_size: int,
    mode: str = "square",
) -> Image.Image:
    """Resize an image using either the legacy square path or paper-style longest side."""
    image = image.convert("RGB")
    if mode == "square":
        return TF.resize(image, [image_size, image_size], interpolation=Image.BILINEAR)
    if mode == "longest":
        width, height = image.size
        scale = image_size / float(max(height, width))
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))
        return TF.resize(image, [new_h, new_w], interpolation=Image.BILINEAR)
    raise ValueError(f"unknown resize mode: {mode}")


def tensorise_and_normalise(image: Image.Image) -> Tensor:
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return tensor


def load_and_preprocess(image: Image.Image, image_size: int, resize_mode: str = "square") -> Tensor:
    """Resize, tensorise, and normalise a single image."""
    return tensorise_and_normalise(resize_image(image, image_size, resize_mode))


def load_and_preprocess_with_size(
    image: Image.Image,
    image_size: int,
    resize_mode: str = "square",
) -> tuple[Tensor, int, int]:
    """Return preprocessed tensor plus actual resized height and width."""
    resized = resize_image(image, image_size, resize_mode)
    width, height = resized.size
    return tensorise_and_normalise(resized), height, width


class VOCClassification(Dataset):
    """VOCDetection wrapper that returns multi-label targets and dense patch boxes.

    Supports:
      - ``train_scales``: randomly pick a scale per sample for multi-scale training.
      - ``hflip_prob``: horizontal flip augmentation.
      - ``test_scales``: if set (eval mode), images are returned at the ``image_size``
        but ``test_scales`` is stored in the sample for downstream multi-scale processing.
    """

    def __init__(
        self,
        root: str | Path,
        year: str = "2007",
        image_set: str = "trainval",
        image_size: int = 448,
        train_scales: tuple[int, ...] | None = None,
        test_scales: tuple[int, ...] | None = None,
        hflip_prob: float = 0.0,
        patch_sizes: tuple[int, ...] = (96, 128, 192, 256),
        patch_stride: int = 64,
        max_patches: int = 160,
        resize_mode: str = "square",
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
        self.test_scales = test_scales
        self.hflip_prob = hflip_prob
        self.patch_sizes = patch_sizes
        self.patch_stride = patch_stride
        self.max_patches = max_patches
        self.resize_mode = resize_mode

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, target = self.dataset[index]
        image = image.convert("RGB")

        # Training mode: multi-scale augmentation
        if self.train_scales is not None:
            image_size = random.choice(self.train_scales)
            image = resize_image(image, image_size, self.resize_mode)
            if self.hflip_prob > 0 and random.random() < self.hflip_prob:
                image = TF.hflip(image)
        else:
            image_size = self.image_size
            image = resize_image(image, image_size, self.resize_mode)

        width, height = image.size
        tensor = tensorise_and_normalise(image)

        labels = self._multi_hot(target)
        boxes = make_dense_patch_boxes(
            height=height,
            width=width,
            patch_sizes=self.patch_sizes,
            stride=self.patch_stride,
            max_patches=self.max_patches,
        )
        return {
            "image": tensor,
            "labels": labels,
            "boxes": boxes,
            "image_id": target["annotation"]["filename"],
            "image_hw": (height, width),
            "test_scales": self.test_scales,  # None for training, tuple for eval
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
