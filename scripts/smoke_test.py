from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.models import AlexNetFisherNet, FisherLayer


def main() -> None:
    torch.manual_seed(7)

    layer = FisherLayer(feature_dim=16, num_components=4, learn_priors=True)
    descriptors = torch.randn(2, 12, 16)
    mask = torch.ones(2, 12, dtype=torch.bool)
    fv = layer(descriptors, mask=mask)
    assert fv.shape == (2, 128)

    model = AlexNetFisherNet(
        num_classes=20,
        patch_dim=32,
        num_components=4,
        pretrained=False,
        learn_priors=True,
    )
    images = torch.randn(2, 3, 224, 224)
    boxes = [
        torch.tensor([[0, 0, 112, 112], [56, 56, 168, 168], [112, 112, 224, 224]], dtype=torch.float32),
        torch.tensor([[0, 0, 224, 224], [32, 32, 160, 160]], dtype=torch.float32),
    ]
    labels = torch.randint(0, 2, (2, 20)).float()
    logits = model(images, boxes)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    print({"fisher_shape": tuple(fv.shape), "logits_shape": tuple(logits.shape), "loss": float(loss)})


if __name__ == "__main__":
    main()
