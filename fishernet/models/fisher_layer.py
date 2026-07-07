import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


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
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_components = num_components
        self.learn_priors = learn_priors
        self.eps = eps
        self.power_norm = power_norm
        self.l2_norm = l2_norm

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
        """Initialize ``w=1/sigma`` and ``b=-mu`` from a fitted diagonal GMM."""
        if means.shape != (self.num_components, self.feature_dim):
            raise ValueError(f"means must have shape {(self.num_components, self.feature_dim)}")
        if sigmas.shape != (self.num_components, self.feature_dim):
            raise ValueError(f"sigmas must have shape {(self.num_components, self.feature_dim)}")
        self.weight.copy_(1.0 / sigmas.clamp_min(self.eps))
        self.bias.copy_(-means)
        if priors is not None and self.prior_logits is not None:
            self.prior_logits.copy_(priors.clamp_min(self.eps).log())

    def forward(self, descriptors: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if descriptors.ndim != 3:
            raise ValueError("descriptors must have shape [B, M, D]")
        if descriptors.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {descriptors.shape[-1]}")

        # y[b, m, k, d] = w[k, d] * (x[b, m, d] + b[k, d])
        y = self.weight[None, None, :, :] * (
            descriptors[:, :, None, :] + self.bias[None, None, :, :]
        )
        dist = y.square().sum(dim=-1)
        logits = -0.5 * dist
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

        if self.prior_logits is not None:
            priors = F.softmax(self.prior_logits, dim=0).clamp_min(self.eps)
            prior_scale = priors.rsqrt()[None, None, :, None]
        else:
            prior_scale = 1.0

        first_order = gamma[..., None] * y * prior_scale
        second_order = gamma[..., None] * ((y.square() - 1.0) / math.sqrt(2.0)) * prior_scale

        first_order = first_order.sum(dim=1) / denom[:, None, :]
        second_order = second_order.sum(dim=1) / denom[:, None, :]
        fv = torch.cat([first_order.flatten(1), second_order.flatten(1)], dim=1)

        if self.power_norm:
            fv = fv.sign() * fv.abs().clamp_min(self.eps).sqrt()
        if self.l2_norm:
            fv = F.normalize(fv, p=2, dim=1, eps=self.eps)
        return fv
