from __future__ import annotations

import argparse
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
from fishernet.models import build_fishernet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a diagonal GMM for Fisher layer init.")
    parser.add_argument("--preset", choices=PRESETS.keys())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/fishernet_voc2007/fisher_gmm.pt"))
    parser.add_argument("--year", default="2007")
    parser.add_argument("--image-set", default="trainval")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--backbone", choices=("alexnet", "resnet101"), default="alexnet")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--train-scales", type=int, nargs="+")
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[64, 96, 128, 160, 192, 224, 256])
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--patch-dim", type=int, default=256)
    parser.add_argument("--num-components", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=160)
    parser.add_argument("--roi-output-size", type=int)
    parser.add_argument("--max-descriptors", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    return apply_preset(args)


@torch.no_grad()
def main() -> None:
    args = parse_args()
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
    ).to(args.device)
    model.eval()

    chunks = []
    total = 0
    for batch in tqdm(loader, desc="collect descriptors"):
        images = batch["images"].to(args.device)
        boxes = [b.to(args.device) for b in batch["boxes"]]
        features, mask = model.extract_patch_features(images, boxes)
        descriptors = features[mask].detach().cpu()
        chunks.append(descriptors)
        total += descriptors.shape[0]
        if total >= args.max_descriptors:
            break

    descriptors_np = torch.cat(chunks, dim=0)[: args.max_descriptors].numpy()
    gmm = GaussianMixture(
        n_components=args.num_components,
        covariance_type="diag",
        max_iter=100,
        random_state=7,
        verbose=1,
    )
    gmm.fit(descriptors_np)

    means = torch.from_numpy(gmm.means_.astype(np.float32))
    sigmas = torch.from_numpy(np.sqrt(gmm.covariances_).astype(np.float32)).clamp_min(1e-4)
    priors = torch.from_numpy(gmm.weights_.astype(np.float32)).clamp_min(1e-6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"means": means, "sigmas": sigmas, "priors": priors, "args": vars(args)}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
