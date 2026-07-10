from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.configs import PRESETS, apply_preset
from fishernet.models import build_fishernet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one FisherNet train step.")
    parser.add_argument("--preset", choices=PRESETS.keys())
    parser.add_argument("--backbone", choices=("alexnet", "resnet101"), default="alexnet")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-patches", type=int, default=160)
    parser.add_argument("--patch-dim", type=int, default=256)
    parser.add_argument("--num-components", type=int, default=32)
    parser.add_argument("--roi-output-size", type=int)
    parser.add_argument("--fisher-second-order-scale", type=float, default=2.0**-0.5)
    parser.add_argument("--fisher-caffe-backward-compat", action="store_true")
    parser.add_argument("--pca-l2-caffe-backward", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    return apply_preset(args)


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    model = build_fishernet(
        backbone=args.backbone,
        patch_dim=args.patch_dim,
        num_components=args.num_components,
        pretrained=False,
        roi_output_size=args.roi_output_size,
        fisher_second_order_scale=getattr(args, "fisher_second_order_scale", 2.0**-0.5),
        fisher_caffe_backward_compat=getattr(args, "fisher_caffe_backward_compat", False),
        pca_l2_caffe_backward=getattr(args, "pca_l2_caffe_backward", False),
    ).to(args.device)
    model.train()
    images = torch.randn(1, 3, args.image_size, args.image_size, device=args.device)
    patch = torch.tensor([0.0, 0.0, 128.0, 128.0], device=args.device)
    boxes = [patch.repeat(args.max_patches, 1)]
    labels = torch.randint(0, 2, (1, 20), device=args.device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    times = []
    for _ in range(args.steps):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, boxes)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    warm = times[2:] if len(times) > 2 else times
    max_mem_gb = torch.cuda.max_memory_allocated() / 1024**3 if args.device.startswith("cuda") else 0.0
    print(
        {
            "device": args.device,
            "avg_step_seconds_after_warmup": sum(warm) / len(warm),
            "max_memory_gb": max_mem_gb,
            "steps": times,
        }
    )


if __name__ == "__main__":
    main()
