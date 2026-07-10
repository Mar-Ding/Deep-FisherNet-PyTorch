import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CaffeGpuEltwiseProduct(torch.autograd.Function):
    """Official Caffe Fisher EltwiseProduct with GPU backward quirks."""

    @staticmethod
    def forward(ctx, descriptors: Tensor, weight: Tensor, bias: Tensor) -> tuple[Tensor, Tensor]:
        y = weight[None, None, :, :] * descriptors[:, :, None, :] + bias[None, None, :, :]
        sigma = weight.abs().clamp_min(1e-7)
        ctx.save_for_backward(descriptors, weight)
        return y, sigma

    @staticmethod
    def backward(ctx, grad_y: Tensor, grad_sigma: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        descriptors, weight = ctx.saved_tensors
        grad_descriptors = (grad_y * weight[None, None, :, :]).sum(dim=2)

        normal_weight = (grad_y * descriptors[:, :, None, :]).sum(dim=(0, 1))
        normal_bias = grad_y.sum(dim=(0, 1))

        # The official CUDA kernel stores the EltwiseProduct parameter diff
        # as if [component, dim] were flattened with the axes swapped.
        grad_weight = normal_weight.transpose(0, 1).contiguous().reshape_as(weight)
        grad_weight = grad_weight + grad_sigma.reshape_as(weight) * weight.sign()
        grad_bias = normal_bias.transpose(0, 1).contiguous().reshape_as(weight)
        return grad_descriptors, grad_weight, grad_bias


class FisherLayer(nn.Module):
    """Differentiable Fisher Vector layer from Deep FisherNet.

    Input shape is ``[B, M, D]`` where ``M`` is the number of local patch
    descriptors and ``D`` is the patch feature dimension. The output is a
    fixed-length image descriptor with shape ``[B, 2 * K * D]``.
    """

    def __init__(
        self,
        feature_dim: int,
        num_components: int = 32,
        learn_priors: bool = False,
        eps: float = 1e-6,
        power_norm: bool = True,
        l2_norm: bool = True,
        parameterization: str = "legacy",
        include_log_det: bool = False,
        scale_by_prior: bool = True,
        pooling: str = "mean",
        second_order_scale: float = 1.0 / math.sqrt(2.0),
        caffe_backward_compat: bool = False,
        assignment_sigma_scale: float = 1.0,
        assignment_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_components = num_components
        self.learn_priors = learn_priors
        self.eps = eps
        self.power_norm = power_norm
        self.l2_norm = l2_norm
        self.parameterization = parameterization
        self.include_log_det = include_log_det
        self.scale_by_prior = scale_by_prior
        self.pooling = pooling
        self.second_order_scale = second_order_scale
        self.caffe_backward_compat = caffe_backward_compat
        if assignment_sigma_scale <= 0 or assignment_temperature <= 0:
            raise ValueError("assignment_sigma_scale and assignment_temperature must be positive")
        self.assignment_sigma_scale = float(assignment_sigma_scale)
        self.assignment_temperature = float(assignment_temperature)
        if parameterization not in {"legacy", "caffe"}:
            raise ValueError("parameterization must be 'legacy' or 'caffe'")
        if pooling not in {"mean", "sum"}:
            raise ValueError("pooling must be 'mean' or 'sum'")

        self.weight = nn.Parameter(torch.empty(num_components, feature_dim))
        self.bias = nn.Parameter(torch.empty(num_components, feature_dim))
        if learn_priors:
            self.prior_logits = nn.Parameter(torch.zeros(num_components))
        else:
            self.register_parameter("prior_logits", None)
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return 2 * self.num_components * self.feature_dim

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=1.0, std=0.02)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    @torch.no_grad()
    def initialize_from_gmm(
        self,
        means: Tensor,
        sigmas: Tensor,
        priors: Optional[Tensor] = None,
    ) -> None:
        """Initialize Fisher parameters from a fitted diagonal GMM."""
        if means.shape != (self.num_components, self.feature_dim):
            raise ValueError(f"means must have shape {(self.num_components, self.feature_dim)}")
        if sigmas.shape != (self.num_components, self.feature_dim):
            raise ValueError(f"sigmas must have shape {(self.num_components, self.feature_dim)}")
        inv_sigma = 1.0 / sigmas.clamp_min(self.eps)
        self.weight.copy_(inv_sigma)
        if self.parameterization == "caffe":
            self.bias.copy_(-means * inv_sigma)
        else:
            self.bias.copy_(-means)
        if priors is not None and self.prior_logits is not None:
            self.prior_logits.copy_(priors.clamp_min(self.eps).log())

    def forward(self, descriptors: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if descriptors.ndim != 3:
            raise ValueError("descriptors must have shape [B, M, D]")
        if descriptors.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {descriptors.shape[-1]}")

        sigma = None
        if self.parameterization == "caffe" and self.caffe_backward_compat:
            y, sigma = CaffeGpuEltwiseProduct.apply(descriptors, self.weight, self.bias)
        elif self.parameterization == "caffe":
            # Matches the official Caffe EltwiseProduct layer: y = w * x + b.
            y = (
                self.weight[None, None, :, :] * descriptors[:, :, None, :]
                + self.bias[None, None, :, :]
            )
        else:
            # Legacy PyTorch port: y = w * (x + b).
            y = self.weight[None, None, :, :] * (
                descriptors[:, :, None, :] + self.bias[None, None, :, :]
            )
        y = y / self.assignment_sigma_scale
        dist = y.square().sum(dim=-1)
        logits = -0.5 * dist / self.assignment_temperature
        if self.include_log_det:
            sigma_for_log_det = self.weight.abs().clamp_min(self.eps) if sigma is None else sigma
            log_det = sigma_for_log_det.log().sum(dim=-1) - self.feature_dim * math.log(self.assignment_sigma_scale)
            logits = logits + log_det[None, None, :]
        if self.prior_logits is not None:
            logits = logits + F.log_softmax(self.prior_logits, dim=0)[None, None, :]
        gamma = F.softmax(logits, dim=-1)

        if mask is not None:
            valid = mask.to(dtype=descriptors.dtype).unsqueeze(-1)
            gamma = gamma * valid
            denom = valid.sum(dim=1).clamp_min(1.0)
        else:
            denom = torch.full(
                (descriptors.shape[0], 1),
                descriptors.shape[1],
                dtype=descriptors.dtype,
                device=descriptors.device,
            )

        if self.prior_logits is not None and self.scale_by_prior:
            priors = F.softmax(self.prior_logits, dim=0).clamp_min(self.eps)
            prior_scale = priors.rsqrt()[None, None, :, None]
        else:
            prior_scale = 1.0

        first_order = gamma[..., None] * y * prior_scale
        second_order = gamma[..., None] * ((y.square() - 1.0) * self.second_order_scale) * prior_scale

        first_order = first_order.sum(dim=1)
        second_order = second_order.sum(dim=1)
        if self.pooling == "mean":
            first_order = first_order / denom[:, None, :]
            second_order = second_order / denom[:, None, :]
        fv = torch.cat([first_order.flatten(1), second_order.flatten(1)], dim=1)

        if self.power_norm:
            fv = fv.sign() * fv.abs().clamp_min(self.eps).sqrt()
        if self.l2_norm:
            fv = F.normalize(fv, p=2, dim=1, eps=self.eps)
        return fv
