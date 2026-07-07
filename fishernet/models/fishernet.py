from typing import List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import AlexNet_Weights, ResNet101_Weights, alexnet, resnet101
from torchvision.ops import roi_align

from .fisher_layer import FisherLayer


class AlexNetFisherNet(nn.Module):
    """AlexNet patch feature extractor plus differentiable Fisher aggregation."""

    def __init__(
        self,
        num_classes: int = 20,
        patch_dim: int = 256,
        num_components: int = 32,
        pretrained: bool = True,
        learn_priors: bool = False,
        roi_output_size: int = 6,
    ) -> None:
        super().__init__()
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        base = alexnet(weights=weights)
        self.features = nn.Sequential(*list(base.features.children())[:-1])
        self.roi_output_size = roi_output_size

        classifier_layers = list(base.classifier.children())[:-1]
        self.patch_mlp = nn.Sequential(
            *classifier_layers,
            nn.Linear(4096, patch_dim),
            nn.ReLU(inplace=True),
        )
        self.fisher = FisherLayer(
            feature_dim=patch_dim,
            num_components=num_components,
            learn_priors=learn_priors,
        )
        self.classifier = nn.Linear(self.fisher.output_dim, num_classes)

    def forward(
        self,
        images: Tensor,
        boxes: List[Tensor],
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        if len(boxes) != images.shape[0]:
            raise ValueError("boxes must contain one tensor per image")

        batch_features, mask = self.extract_patch_features(images, boxes)
        fisher_features = self.fisher(batch_features, mask=mask)
        logits = self.classifier(fisher_features)
        if return_features:
            return logits, fisher_features
        return logits

    def extract_patch_features(self, images: Tensor, boxes: List[Tensor]) -> tuple[Tensor, Tensor]:
        """Return normalized local descriptors packed as ``[B, M, D]`` plus a mask."""
        conv = self.features(images)
        rois = self._make_feature_rois(boxes, images.shape[-2:], conv.shape[-2:], images.device)
        if rois.numel() == 0:
            raise ValueError("at least one patch box is required")

        pooled = roi_align(
            conv,
            rois,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )
        patch_features = self.patch_mlp(torch.flatten(pooled, 1))
        patch_features = F.normalize(patch_features, p=2, dim=1)

        counts = [b.shape[0] for b in boxes]
        max_count = max(counts)
        batch_features = patch_features.new_zeros(images.shape[0], max_count, patch_features.shape[-1])
        mask = torch.zeros(images.shape[0], max_count, dtype=torch.bool, device=images.device)
        start = 0
        for batch_idx, count in enumerate(counts):
            end = start + count
            batch_features[batch_idx, :count] = patch_features[start:end]
            mask[batch_idx, :count] = True
            start = end

        return batch_features, mask

    @staticmethod
    def _make_feature_rois(
        boxes: List[Tensor],
        image_hw: tuple[int, int],
        feature_hw: tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        image_h, image_w = image_hw
        feature_h, feature_w = feature_hw
        scale_x = feature_w / float(image_w)
        scale_y = feature_h / float(image_h)
        rois = []
        for batch_idx, image_boxes in enumerate(boxes):
            if image_boxes.numel() == 0:
                continue
            scaled = image_boxes.to(device=device, dtype=torch.float32).clone()
            scaled[:, [0, 2]] *= scale_x
            scaled[:, [1, 3]] *= scale_y
            batch_column = torch.full((scaled.shape[0], 1), batch_idx, device=device)
            rois.append(torch.cat([batch_column, scaled], dim=1))
        if not rois:
            return torch.empty(0, 5, dtype=torch.float32, device=device)
        return torch.cat(rois, dim=0)


class ResNetFisherNet(nn.Module):
    """ResNet-101 patch feature extractor plus differentiable Fisher aggregation."""

    def __init__(
        self,
        num_classes: int = 20,
        patch_dim: int = 128,
        num_components: int = 64,
        pretrained: bool = True,
        learn_priors: bool = False,
        roi_output_size: int = 1,
    ) -> None:
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet101(weights=weights)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.roi_output_size = roi_output_size
        in_features = 2048 * roi_output_size * roi_output_size
        self.patch_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, patch_dim),
            nn.ReLU(inplace=True),
        )
        self.fisher = FisherLayer(
            feature_dim=patch_dim,
            num_components=num_components,
            learn_priors=learn_priors,
        )
        self.classifier = nn.Linear(self.fisher.output_dim, num_classes)

    def forward(
        self,
        images: Tensor,
        boxes: List[Tensor],
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        batch_features, mask = self.extract_patch_features(images, boxes)
        fisher_features = self.fisher(batch_features, mask=mask)
        logits = self.classifier(fisher_features)
        if return_features:
            return logits, fisher_features
        return logits

    def extract_patch_features(self, images: Tensor, boxes: List[Tensor]) -> tuple[Tensor, Tensor]:
        conv = self.features(images)
        rois = AlexNetFisherNet._make_feature_rois(boxes, images.shape[-2:], conv.shape[-2:], images.device)
        if rois.numel() == 0:
            raise ValueError("at least one patch box is required")
        pooled = roi_align(
            conv,
            rois,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )
        patch_features = self.patch_mlp(pooled)
        patch_features = F.normalize(patch_features, p=2, dim=1)

        counts = [b.shape[0] for b in boxes]
        max_count = max(counts)
        batch_features = patch_features.new_zeros(images.shape[0], max_count, patch_features.shape[-1])
        mask = torch.zeros(images.shape[0], max_count, dtype=torch.bool, device=images.device)
        start = 0
        for batch_idx, count in enumerate(counts):
            end = start + count
            batch_features[batch_idx, :count] = patch_features[start:end]
            mask[batch_idx, :count] = True
            start = end
        return batch_features, mask


def build_fishernet(
    backbone: str,
    num_classes: int = 20,
    patch_dim: int = 256,
    num_components: int = 32,
    pretrained: bool = True,
    roi_output_size: int | None = None,
) -> nn.Module:
    if backbone == "alexnet":
        return AlexNetFisherNet(
            num_classes=num_classes,
            patch_dim=patch_dim,
            num_components=num_components,
            pretrained=pretrained,
            roi_output_size=6 if roi_output_size is None else roi_output_size,
        )
    if backbone == "resnet101":
        return ResNetFisherNet(
            num_classes=num_classes,
            patch_dim=patch_dim,
            num_components=num_components,
            pretrained=pretrained,
            roi_output_size=1 if roi_output_size is None else roi_output_size,
        )
    raise ValueError(f"unknown backbone: {backbone}")
