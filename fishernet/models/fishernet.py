import math
from typing import List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import AlexNet_Weights, ResNet101_Weights, VGG16_Weights, alexnet, resnet101, vgg16
from torchvision.ops import roi_align

from .fisher_layer import FisherLayer


class CaffeL2Normalize(torch.autograd.Function):
    """Official Caffe L2Normalize: same forward, Caffe GPU backward."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        norm = (x.square().sum(dim=1, keepdim=True) + 1e-12).sqrt()
        y = x / norm
        ctx.save_for_backward(y, norm)
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor) -> tuple[Tensor]:
        y, norm = ctx.saved_tensors
        return (grad_y * (1.0 - y.square()) / norm,)


def _caffe_resnet_pool1() -> nn.MaxPool2d:
    # Caffe ResNet pool1 omits padding and relies on ceil output sizing.
    return nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)


def _make_resnet101_caffe_v1(pretrained: bool) -> nn.Module:
    weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
    base = resnet101(weights=weights)
    for layer_name in ("layer2", "layer3", "layer4"):
        block = getattr(base, layer_name)[0]
        block.conv1.stride = (2, 2)
        block.conv2.stride = (1, 1)
    return base


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
        fisher_kwargs: Optional[dict] = None,
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
            **(fisher_kwargs or {}),
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


class VGG16FisherNet(nn.Module):
    """VGG-16 patch feature extractor plus differentiable Fisher aggregation.

    Matches the paper's VGG16-based FisherNet:
      - Removes pool5 (last MaxPool) so conv feature map is at 1/16 scale.
      - Uses pre-trained fc6, fc7, then a new 256-dim reduction layer.
      - Fisher Layer: K=32, D=256 → FV dim = 16384.
    """

    def __init__(
        self,
        num_classes: int = 20,
        patch_dim: int = 256,
        num_components: int = 32,
        pretrained: bool = True,
        learn_priors: bool = False,
        roi_output_size: int = 7,
        fisher_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        base = vgg16(weights=weights)
        # Remove last MaxPool2d (pool5) so spatial stride is 1/16 instead of 1/32
        self.features = base.features[:-1]
        self.roi_output_size = roi_output_size

        # fc6, fc7, then dimension-reduction head
        self.patch_mlp = nn.Sequential(
            base.classifier[0],   # Linear(25088, 4096)
            base.classifier[1],   # ReLU
            base.classifier[2],   # Dropout(0.5)
            base.classifier[3],   # Linear(4096, 4096)
            base.classifier[4],   # ReLU
            base.classifier[5],   # Dropout(0.5)
            nn.Linear(4096, patch_dim),
            nn.ReLU(inplace=True),
        )
        self.fisher = FisherLayer(
            feature_dim=patch_dim,
            num_components=num_components,
            learn_priors=learn_priors,
            **(fisher_kwargs or {}),
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
        rois = AlexNetFisherNet._make_feature_rois(
            boxes, images.shape[-2:], conv.shape[-2:], images.device
        )
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
        batch_size = images.shape[0]
        batch_features = patch_features.new_zeros(batch_size, max_count, patch_features.shape[-1])
        mask = torch.zeros(batch_size, max_count, dtype=torch.bool, device=images.device)
        start = 0
        for batch_idx, count in enumerate(counts):
            end = start + count
            batch_features[batch_idx, :count] = patch_features[start:end]
            mask[batch_idx, :count] = True
            start = end
        return batch_features, mask


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
        fisher_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        base = _make_resnet101_caffe_v1(pretrained)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            _caffe_resnet_pool1(),
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
            **(fisher_kwargs or {}),
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
        rois = AlexNetFisherNet._make_feature_rois(
            boxes, images.shape[-2:], conv.shape[-2:], images.device
        )
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


class ResNet101SpatialFisherNet(nn.Module):
    """Official-prototxt-style ResNet-101 FisherNet.

    The official Res-101 model applies Fisher encoding over spatial positions
    of the res5c feature map, after a 1x1 PCA-like projection to 128 channels.
    It does not use dense ROI patch boxes.
    """

    def __init__(
        self,
        num_classes: int = 20,
        patch_dim: int = 128,
        num_components: int = 64,
        pretrained: bool = True,
        learn_priors: bool = True,
        freeze_bn: bool = False,
        pca_l2_caffe_backward: bool = False,
        fisher_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.freeze_bn = freeze_bn
        self.pca_l2_caffe_backward = pca_l2_caffe_backward
        base = _make_resnet101_caffe_v1(pretrained)
        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            _caffe_resnet_pool1(),
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.pca = nn.Conv2d(2048, patch_dim, kernel_size=1, bias=True)
        self.fisher = FisherLayer(
            feature_dim=patch_dim,
            num_components=num_components,
            learn_priors=learn_priors,
            **(fisher_kwargs or {}),
        )
        self.classifier = nn.Linear(self.fisher.output_dim, num_classes)

    def train(self, mode: bool = True) -> "ResNet101SpatialFisherNet":
        super().train(mode)
        if self.freeze_bn:
            self._freeze_bn()
        return self

    def _freeze_bn(self) -> None:
        for m in self.features.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(
        self,
        images: Tensor,
        boxes: Optional[List[Tensor]] = None,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        del boxes
        descriptors, mask = self.extract_patch_features(images, None)
        fisher_features = self.fisher(descriptors, mask=mask)
        logits = self.classifier(fisher_features)
        if return_features:
            return logits, fisher_features
        return logits

    def extract_patch_features(
        self,
        images: Tensor,
        boxes: Optional[List[Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        del boxes
        conv = self.features(images)
        projected = self.pca(conv)
        if self.pca_l2_caffe_backward:
            projected = CaffeL2Normalize.apply(projected)
        else:
            projected = F.normalize(projected, p=2, dim=1)
        descriptors = projected.flatten(2).transpose(1, 2).contiguous()
        mask = torch.ones(
            descriptors.shape[:2],
            dtype=torch.bool,
            device=descriptors.device,
        )
        return descriptors, mask


def build_fishernet(
    backbone: str,
    num_classes: int = 20,
    patch_dim: int = 256,
    num_components: int = 32,
    pretrained: bool = True,
    roi_output_size: int | None = None,
    learn_priors: bool = False,
    freeze_bn: bool = False,
    fisher_parameterization: str = "legacy",
    fisher_include_log_det: bool = False,
    fisher_scale_by_prior: bool = True,
    fisher_pooling: str = "mean",
    fisher_power_norm: bool = True,
    fisher_l2_norm: bool = True,
    fisher_second_order_scale: float = 1.0 / math.sqrt(2.0),
    fisher_caffe_backward_compat: bool = False,
    fisher_assignment_sigma_scale: float = 1.0,
    fisher_assignment_temperature: float = 1.0,
    pca_l2_caffe_backward: bool = False,
) -> nn.Module:
    fisher_kwargs = {
        "parameterization": fisher_parameterization,
        "include_log_det": fisher_include_log_det,
        "scale_by_prior": fisher_scale_by_prior,
        "pooling": fisher_pooling,
        "power_norm": fisher_power_norm,
        "l2_norm": fisher_l2_norm,
        "second_order_scale": fisher_second_order_scale,
        "caffe_backward_compat": fisher_caffe_backward_compat,
        "assignment_sigma_scale": fisher_assignment_sigma_scale,
        "assignment_temperature": fisher_assignment_temperature,
    }
    if backbone == "alexnet":
        return AlexNetFisherNet(
            num_classes=num_classes,
            patch_dim=patch_dim,
            num_components=num_components,
            pretrained=pretrained,
            learn_priors=learn_priors,
            roi_output_size=6 if roi_output_size is None else roi_output_size,
            fisher_kwargs=fisher_kwargs,
        )
    if backbone == "vgg16":
        return VGG16FisherNet(
            num_classes=num_classes,
            patch_dim=patch_dim,
            num_components=num_components,
            pretrained=pretrained,
            learn_priors=learn_priors,
            roi_output_size=7 if roi_output_size is None else roi_output_size,
            fisher_kwargs=fisher_kwargs,
        )
    if backbone == "resnet101":
        return ResNetFisherNet(
            num_classes=num_classes,
            patch_dim=patch_dim,
            num_components=num_components,
            pretrained=pretrained,
            learn_priors=learn_priors,
            roi_output_size=1 if roi_output_size is None else roi_output_size,
            fisher_kwargs=fisher_kwargs,
        )
    if backbone == "resnet101-spatial":
        return ResNet101SpatialFisherNet(
            num_classes=num_classes,
            patch_dim=patch_dim,
            num_components=num_components,
            pretrained=pretrained,
            learn_priors=learn_priors,
            freeze_bn=freeze_bn,
            pca_l2_caffe_backward=pca_l2_caffe_backward,
            fisher_kwargs=fisher_kwargs,
        )
    raise ValueError(f"unknown backbone: {backbone}")
