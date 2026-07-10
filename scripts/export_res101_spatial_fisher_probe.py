"""
Export ResNet-101 spatial FisherNet intermediate tensors for Caffe alignment.

This probe mirrors the official Caffe blob layout where possible:
  res5c_pca_l2: [B, D, H, W]
  fisher1:      [B, K*D, H, W]
  fisher_gamma: [B, K, H, W]
  fisher_sum:   [B, 2*K*D]

Example:
  python scripts/export_res101_spatial_fisher_probe.py \
    --checkpoint outputs/c4/c4_res101_caffev1_frozenbn_4ep/best.pt \
    --image-path data/VOCdevkit/VOC2007/JPEGImages/000001.jpg \
    --out-dir debug_align/pytorch_c4_res101_fisher_000001 \
    --image-size 448 --resize-mode longest --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.configs import PRESETS, apply_preset
from fishernet.data.voc import PASCAL_CLASSES, load_and_preprocess_with_size
from fishernet.models import build_fishernet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export C4/Res101-spatial PCA and Fisher blobs for layer alignment."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--init-pt",
        type=Path,
        help="Optional PCA/GMM init payload from init_res101_spatial_params.py.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESETS.keys(), default="official-res101-spatial-frozenbn")
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--input-npy", type=Path, help="Optional preprocessed NCHW tensor to feed directly.")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--resize-mode", choices=("square", "longest"), default="longest")
    parser.add_argument("--backbone", choices=("resnet101-spatial",), default="resnet101-spatial")
    parser.add_argument("--patch-dim", type=int, default=128)
    parser.add_argument("--num-components", type=int, default=64)
    parser.add_argument("--learn-priors", action="store_true")
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument("--fisher-parameterization", choices=("legacy", "caffe"), default="caffe")
    parser.add_argument("--fisher-include-log-det", action="store_true")
    parser.add_argument("--fisher-scale-by-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fisher-pooling", choices=("mean", "sum"), default="sum")
    parser.add_argument("--fisher-second-order-scale", type=float, default=2.0**-0.5)
    parser.add_argument("--fisher-caffe-backward-compat", action="store_true")
    parser.add_argument("--pca-l2-caffe-backward", action="store_true")
    parser.add_argument(
        "--prior-logits-mode",
        choices=("log", "raw"),
        default="log",
        help=(
            "How to map init priors into learnable prior logits for diagnostics. "
            "'log' makes softmax(log(priors)) equal priors; 'raw' mirrors Caffe "
            "when probability values are loaded directly into fisher_weight."
        ),
    )
    parser.add_argument(
        "--second-order-scale",
        type=float,
        default=None,
        help="Override second-order Fisher scale; official Caffe prototxt uses 0.7071.",
    )
    parser.add_argument("--no-fisher-power-norm", action="store_true")
    parser.add_argument("--no-fisher-l2-norm", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained backbone.")
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    args = apply_preset(args)
    if args.image_path is None and args.input_npy is None:
        parser.error("one of --image-path or --input-npy is required")
    if (args.checkpoint is None) == (args.init_pt is None):
        parser.error("exactly one of --checkpoint or --init-pt is required")
    return args


def _torch_load(path: Path, device: str) -> dict[str, Any]:
    # Some returned Linux checkpoints contain pathlib.PosixPath in args.
    if sys.platform.startswith("win"):
        import pathlib

        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]
    return torch.load(path, map_location=device)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def save_blob(out_dir: Path, name: str, tensor: torch.Tensor) -> None:
    np.save(out_dir / f"{name}.npy", _to_numpy(tensor))


def make_input(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.input_npy is not None:
        arr = np.load(args.input_npy).astype("float32", copy=False)
        if arr.ndim != 4:
            raise ValueError(f"--input-npy must have NCHW shape, got {arr.shape}")
        tensor = torch.from_numpy(arr)
        return tensor, {
            "source": "input_npy",
            "input_npy": str(args.input_npy),
            "shape": list(arr.shape),
        }

    image = Image.open(args.image_path).convert("RGB")
    tensor, height, width = load_and_preprocess_with_size(image, args.image_size, args.resize_mode)
    return tensor.unsqueeze(0), {
        "source": "image_path",
        "image_path": str(args.image_path),
        "image_size": args.image_size,
        "resize_mode": args.resize_mode,
        "resized_hw": [height, width],
    }


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    model = build_fishernet(
        backbone=args.backbone,
        num_classes=len(PASCAL_CLASSES),
        patch_dim=args.patch_dim,
        num_components=args.num_components,
        pretrained=args.pretrained,
        learn_priors=args.learn_priors,
        freeze_bn=getattr(args, "freeze_bn", False),
        fisher_parameterization=args.fisher_parameterization,
        fisher_include_log_det=args.fisher_include_log_det,
        fisher_scale_by_prior=args.fisher_scale_by_prior,
        fisher_pooling=args.fisher_pooling,
        fisher_power_norm=not args.no_fisher_power_norm,
        fisher_l2_norm=not args.no_fisher_l2_norm,
        fisher_second_order_scale=getattr(args, "fisher_second_order_scale", 2.0**-0.5),
        fisher_caffe_backward_compat=getattr(args, "fisher_caffe_backward_compat", False),
        pca_l2_caffe_backward=getattr(args, "pca_l2_caffe_backward", False),
    )
    if args.checkpoint is not None:
        ckpt = _torch_load(args.checkpoint, args.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
    else:
        init = _torch_load(args.init_pt, args.device)
        model.pca.weight.data.copy_(init["pca_weight"].to(args.device))
        model.pca.bias.data.copy_(init["pca_bias"].to(args.device))
        model.fisher.initialize_from_gmm(
            init["means"].to(args.device),
            init["sigmas"].to(args.device),
            init.get("priors", None).to(args.device) if init.get("priors", None) is not None else None,
        )
        if (
            args.prior_logits_mode == "raw"
            and model.fisher.prior_logits is not None
            and init.get("priors", None) is not None
        ):
            model.fisher.prior_logits.data.copy_(init["priors"].to(args.device))
    return model.to(args.device).eval()


def export_fisher_debug(
    model: torch.nn.Module,
    image: torch.Tensor,
    out_dir: Path,
    second_order_scale: float | None = None,
) -> dict[str, Any]:
    fisher = model.fisher
    conv = model.features(image)
    pca = model.pca(conv)
    descriptors = F.normalize(pca.flatten(2).transpose(1, 2).contiguous(), p=2, dim=-1)
    batch, dim, height, width = pca.shape
    num_components = fisher.num_components

    if fisher.parameterization == "caffe":
        y = fisher.weight[None, None, :, :] * descriptors[:, :, None, :] + fisher.bias[None, None, :, :]
    else:
        y = fisher.weight[None, None, :, :] * (descriptors[:, :, None, :] + fisher.bias[None, None, :, :])

    dist = y.square().sum(dim=-1)
    logits = -0.5 * dist
    if fisher.include_log_det:
        log_det = fisher.weight.abs().clamp_min(fisher.eps).log().sum(dim=-1)
        logits = logits + log_det[None, None, :]
    if fisher.prior_logits is not None:
        priors = F.softmax(fisher.prior_logits, dim=0).clamp_min(fisher.eps)
        logits = logits + priors.log()[None, None, :]
    else:
        priors = None

    gamma = F.softmax(logits, dim=-1)
    if priors is not None and fisher.scale_by_prior:
        prior_scale = priors.rsqrt()[None, None, :, None]
    else:
        prior_scale = 1.0

    first_per_desc = gamma[..., None] * y * prior_scale
    second_scale = (1.0 / math.sqrt(2.0)) if second_order_scale is None else second_order_scale
    second_base = (y.square() - 1.0) * second_scale
    second_per_desc = gamma[..., None] * second_base * prior_scale
    first_sum = first_per_desc.sum(dim=1)
    second_sum = second_per_desc.sum(dim=1)
    if fisher.pooling == "mean":
        denom = torch.tensor(float(height * width), dtype=first_sum.dtype, device=first_sum.device)
        first_sum = first_sum / denom
        second_sum = second_sum / denom

    fv_pre_norm = torch.cat([first_sum.flatten(1), second_sum.flatten(1)], dim=1)
    fv_power = fv_pre_norm.sign() * fv_pre_norm.abs().clamp_min(fisher.eps).sqrt()
    fv = fv_power
    if fisher.l2_norm:
        fv = F.normalize(fv, p=2, dim=1, eps=fisher.eps)
    logits_cls = model.classifier(fisher(descriptors))

    def desc_to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).reshape(batch, x.shape[-1], height, width)

    def kd_to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).reshape(batch, num_components * dim, height, width)

    save_blob(out_dir, "input_tensor", image)
    save_blob(out_dir, "pytorch_res5c", conv)
    save_blob(out_dir, "pytorch_layer4", conv)
    save_blob(out_dir, "pytorch_pca", pca)
    save_blob(out_dir, "pytorch_pca_l2", desc_to_nchw(descriptors))
    save_blob(out_dir, "pytorch_descriptors_bmd", descriptors)
    save_blob(out_dir, "pytorch_fisher1", kd_to_nchw(y))
    save_blob(out_dir, "pytorch_fisher2", kd_to_nchw(y.square()))
    save_blob(out_dir, "pytorch_fisher3", dist.transpose(1, 2).reshape(batch, num_components, height, width))
    save_blob(out_dir, "pytorch_fisher4_new_logits", logits.transpose(1, 2).reshape(batch, num_components, height, width))
    save_blob(out_dir, "pytorch_fisher_gamma", gamma.transpose(1, 2).reshape(batch, num_components, height, width))
    save_blob(out_dir, "pytorch_fisher6", kd_to_nchw(first_per_desc))
    save_blob(out_dir, "pytorch_fisher7", kd_to_nchw(second_per_desc))
    save_blob(out_dir, "pytorch_fisher_first_sum", first_sum)
    save_blob(out_dir, "pytorch_fisher_second_sum", second_sum)
    save_blob(out_dir, "pytorch_fisher_sum", fv_pre_norm)
    save_blob(out_dir, "pytorch_fv_power", fv_power)
    save_blob(out_dir, "pytorch_fv", fv)
    save_blob(out_dir, "pytorch_class_logits", logits_cls)
    if priors is not None:
        save_blob(out_dir, "pytorch_fisher_priors", priors)

    return {
        "input_shape": list(image.shape),
        "res5c_shape": list(conv.shape),
        "pca_shape": list(pca.shape),
        "descriptor_shape_bmd": list(descriptors.shape),
        "num_components": num_components,
        "feature_dim": dim,
        "fisher_output_dim": fisher.output_dim,
        "fisher": {
            "parameterization": fisher.parameterization,
            "include_log_det": fisher.include_log_det,
            "scale_by_prior": fisher.scale_by_prior,
            "pooling": fisher.pooling,
            "power_norm": fisher.power_norm,
            "l2_norm": fisher.l2_norm,
            "has_priors": fisher.prior_logits is not None,
            "second_order_scale": second_scale,
        },
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args)
    image, input_meta = make_input(args)
    image = image.to(args.device)

    with torch.no_grad():
        debug_meta = export_fisher_debug(model, image, args.out_dir, args.second_order_scale)

    metadata = {
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "init_pt": str(args.init_pt) if args.init_pt is not None else None,
        "preset": args.preset,
        "input": input_meta,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")},
        "debug": debug_meta,
    }
    with (args.out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Exported Res101 spatial Fisher probe to {args.out_dir}")
    for name, shape in {
        "input_tensor": list(image.shape),
        "pytorch_res5c": debug_meta["res5c_shape"],
        "pytorch_pca": debug_meta["pca_shape"],
        "pytorch_fisher_sum": [1, debug_meta["fisher_output_dim"]],
    }.items():
        print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
