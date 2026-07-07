from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.configs import PRESETS, apply_preset
from fishernet.data.voc import PASCAL_CLASSES, VOCClassification, collate_voc_batch
from fishernet.models import build_fishernet
from fishernet.utils import mean_average_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch Deep FisherNet on VOC.")
    parser.add_argument("--preset", choices=PRESETS.keys())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--year", default="2007")
    parser.add_argument("--image-set", default="test")
    parser.add_argument("--backbone", choices=("alexnet", "resnet101"), default="alexnet")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[64, 96, 128, 160, 192, 224, 256])
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=800)
    parser.add_argument("--patch-dim", type=int, default=256)
    parser.add_argument("--num-components", type=int, default=32)
    parser.add_argument("--roi-output-size", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
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
        patch_sizes=tuple(args.patch_sizes),
        patch_stride=args.patch_stride,
        max_patches=args.max_patches,
        download=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_voc_batch,
    )
    model = build_fishernet(
        backbone=args.backbone,
        num_classes=len(PASCAL_CLASSES),
        patch_dim=args.patch_dim,
        num_components=args.num_components,
        pretrained=False,
        roi_output_size=args.roi_output_size,
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_labels = []
    all_scores = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="eval"), start=1):
        if args.max_batches is not None and batch_idx > args.max_batches:
            break
        images = batch["images"].to(args.device)
        labels = batch["labels"]
        boxes = [b.to(args.device) for b in batch["boxes"]]
        logits = model(images, boxes)
        all_labels.append(labels)
        all_scores.append(torch.sigmoid(logits).cpu())

    targets = torch.cat(all_labels, dim=0)
    scores = torch.cat(all_scores, dim=0)
    mean_ap, aps = mean_average_precision(targets, scores)
    print(f"mAP: {mean_ap:.4f}")
    for name, ap in zip(PASCAL_CLASSES, aps):
        print(f"{name:12s} {ap:.4f}")


if __name__ == "__main__":
    main()
