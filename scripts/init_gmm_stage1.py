from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.configs import PRESETS, apply_preset
from fishernet.data.voc import PASCAL_CLASSES, VOCClassification, collate_voc_batch
from fishernet.models import build_fishernet, load_stage1_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a diagonal GMM for Fisher layer init.")
    parser.add_argument("--preset", choices=PRESETS.keys())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/fishernet_voc2007/fisher_gmm.pt"))
    parser.add_argument("--year", default="2007")
    parser.add_argument("--image-set", default="trainval")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--backbone", choices=("alexnet", "vgg16", "resnet101"), default="alexnet")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--train-scales", type=int, nargs="+")
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[64, 96, 128, 160, 192, 224, 256])
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--patch-dim", type=int, default=256)
    parser.add_argument("--num-components", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=160)
    parser.add_argument("--roi-output-size", type=int)
    parser.add_argument("--resize-mode", choices=("square", "longest"), default="square")
    parser.add_argument("--patch-coordinate-frame", choices=("resized", "original"), default="resized")
    parser.add_argument("--patch-coordinate-mode", choices=("half_open", "caffe"), default="half_open")
    parser.add_argument("--patch-pooling", choices=("roi_align", "roi_pool"), default="roi_align")
    parser.add_argument("--patch-l2-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paper-new-layer-init", action="store_true")
    parser.add_argument("--fisher-second-order-scale", type=float, default=2.0**-0.5)
    parser.add_argument("--fisher-caffe-backward-compat", action="store_true")
    parser.add_argument("--pca-l2-caffe-backward", action="store_true")
    parser.add_argument("--max-descriptors", type=int, default=50000)
    parser.add_argument(
        "--descriptors-per-image",
        type=int,
        help="Random cap per image so the GMM sample covers more of trainval.",
    )
    parser.add_argument("--gmm-n-init", type=int, default=1)
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Load backbone weights from a Stage 1 checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    return apply_preset(args)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset = VOCClassification(
        root=args.data_root,
        year=args.year,
        image_set=args.image_set,
        image_size=args.image_size,
        train_scales=tuple(args.train_scales) if args.train_scales else None,
        hflip_prob=args.hflip_prob,
        patch_sizes=tuple(args.patch_sizes),
        patch_stride=args.patch_stride,
        max_patches=args.max_patches,
        resize_mode=args.resize_mode,
        patch_coordinate_frame=args.patch_coordinate_frame,
        patch_coordinate_mode=args.patch_coordinate_mode,
        download=args.download,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_voc_batch,
    )
    model = build_fishernet(
        backbone=args.backbone,
        num_classes=len(PASCAL_CLASSES),
        patch_dim=args.patch_dim,
        num_components=args.num_components,
        pretrained=args.pretrained,
        roi_output_size=args.roi_output_size,
        fisher_second_order_scale=getattr(args, "fisher_second_order_scale", 2.0**-0.5),
        fisher_caffe_backward_compat=getattr(args, "fisher_caffe_backward_compat", False),
        pca_l2_caffe_backward=getattr(args, "pca_l2_caffe_backward", False),
        patch_pooling=args.patch_pooling,
        patch_l2_norm=args.patch_l2_norm,
        paper_new_layer_init=args.paper_new_layer_init,
    ).to(args.device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        sd = ckpt.get("model", ckpt)
        loaded = load_stage1_weights(model, sd)
        transferred_heads = [pair for pair in loaded if pair[0].startswith("classifier.")]
        print(
            f"  Loaded {len(loaded)} Stage 1 tensors "
            f"({len(transferred_heads)}/6 fc6/fc7/reduction tensors)"
        )
        if args.backbone == "vgg16" and len(transferred_heads) != 6:
            raise RuntimeError(
                "Stage 1 checkpoint must provide fc6, fc7, and the 256-d reduction layer."
            )
    model.eval()

    chunks = []
    total = 0
    source_images = 0
    for batch in tqdm(loader, desc="collect descriptors"):
        images = batch["images"].to(args.device)
        boxes = [b.to(args.device) for b in batch["boxes"]]
        features, mask = model.extract_patch_features(images, boxes)
        for image_index in range(features.shape[0]):
            descriptors = features[image_index][mask[image_index]]
            if (
                args.descriptors_per_image is not None
                and descriptors.shape[0] > args.descriptors_per_image
            ):
                indices = torch.randperm(descriptors.shape[0], device=descriptors.device)[
                    : args.descriptors_per_image
                ]
                descriptors = descriptors[indices]
            chunks.append(descriptors.detach().cpu())
            total += descriptors.shape[0]
            source_images += 1
            if total >= args.max_descriptors:
                break
        if total >= args.max_descriptors:
            break

    descriptors_np = torch.cat(chunks, dim=0)[: args.max_descriptors].numpy()
    print(
        f"Fitting GMM from {descriptors_np.shape[0]} descriptors "
        f"sampled across {source_images} images"
    )
    gmm = GaussianMixture(
        n_components=args.num_components,
        covariance_type="diag",
        max_iter=args.gmm_max_iter,
        n_init=args.gmm_n_init,
        reg_covar=args.gmm_reg_covar,
        random_state=args.seed,
        verbose=1,
    )
    gmm.fit(descriptors_np)

    means = torch.from_numpy(gmm.means_.astype(np.float32))
    sigmas = torch.from_numpy(np.sqrt(gmm.covariances_).astype(np.float32)).clamp_min(1e-4)
    priors = torch.from_numpy(gmm.weights_.astype(np.float32)).clamp_min(1e-6)
    effective_components = float(torch.exp(-(priors * priors.log()).sum()))
    print(
        f"GMM converged={gmm.converged_} iterations={gmm.n_iter_} "
        f"effective_components={effective_components:.2f}/{args.num_components} "
        f"sigma_min={float(sigmas.min()):.6g} sigma_max={float(sigmas.max()):.6g}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "means": means,
            "sigmas": sigmas,
            "priors": priors,
            "source_images": source_images,
            "descriptor_count": int(descriptors_np.shape[0]),
            "converged": bool(gmm.converged_),
            "n_iter": int(gmm.n_iter_),
            "lower_bound": float(gmm.lower_bound_),
            "args": vars(args),
        },
        args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
