from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as TF
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.configs import PRESETS, apply_preset
from fishernet.data.patches import make_dense_patch_boxes
from fishernet.data.voc import (
    PASCAL_CLASSES,
    VOCClassification,
    collate_voc_batch,
    load_and_preprocess,
)
from fishernet.models import build_fishernet
from fishernet.utils import mean_average_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch Deep FisherNet on VOC.")
    parser.add_argument("--preset", choices=PRESETS.keys())
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--year", default="2007")
    parser.add_argument("--image-set", default="test")
    parser.add_argument("--backbone", choices=("alexnet", "vgg16", "resnet101"), default="alexnet")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--test-scales", type=int, nargs="+", default=None,
                        help="Multi-scale test sizes. Paper used: 480 576 688 864 1200. "
                             "Default: single-scale at image-size.")
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[64, 96, 128, 160, 192, 224, 256])
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=800)
    parser.add_argument("--patch-dim", type=int, default=256)
    parser.add_argument("--num-components", type=int, default=32)
    parser.add_argument("--roi-output-size", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--fit-svm", action="store_true",
                        help="Extract FV features from train set, fit linear SVM, then evaluate.")
    parser.add_argument("--train-image-set", default="trainval",
                        help="Dataset split to use for SVM training.")
    parser.add_argument("--svm-C", type=float, default=1.0,
                        help="Linear SVM C parameter (paper: C=1).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    return apply_preset(args)


def apply_test_scales(
    model: torch.nn.Module,
    image: Image.Image,
    boxes: torch.Tensor,
    scales: list[int],
    device: str,
    use_flip: bool = True,
) -> torch.Tensor:
    """Run one image through the model at multiple scales + optional flip, return averaged logits."""
    all_logits = []
    for scale in scales:
        tensor = load_and_preprocess(image, scale)
        tensor = tensor.unsqueeze(0).to(device)
        box_batch = [boxes.to(device)]
        logits = model(tensor, box_batch)
        all_logits.append(logits)

        if use_flip:
            flipped = TF.hflip(image)
            tensor_f = load_and_preprocess(flipped, scale)
            tensor_f = tensor_f.unsqueeze(0).to(device)
            logits_f = model(tensor_f, box_batch)
            all_logits.append(logits_f)

    return torch.stack(all_logits, dim=0).mean(dim=0)


@torch.no_grad()
def extract_fv_multi_scale(
    model: torch.nn.Module,
    raw_dataset: VOCDetection,
    boxes: torch.Tensor,
    scales: list[int],
    max_samples: int | None,
    device: str,
    use_flip: bool = True,
    disable_progress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract FV features for multi-scale evaluation, returns (fv, labels).

    Processes each image at all scales+flips, averages FV per image.
    """
    model.eval()
    model.fisher.power_norm = False
    model.fisher.l2_norm = False
    all_fv = []
    all_labels = []
    n = len(raw_dataset)
    if max_samples is not None:
        n = min(n, max_samples)

    for idx in tqdm(range(n), desc="extract FV", disable=disable_progress):
        image, target = raw_dataset[idx]
        image = image.convert("RGB")
        labels = torch.zeros(len(PASCAL_CLASSES), dtype=torch.float32)
        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]
        for obj in objects:
            name = obj["name"]
            if name in PASCAL_CLASSES:
                labels[PASCAL_CLASSES.index(name)] = 1.0

        view_fv = []
        for scale in scales:
            tensor = load_and_preprocess(image, scale).unsqueeze(0).to(device)
            box_batch = [boxes.to(device)]
            _, fv = model(tensor, box_batch, return_features=True)
            view_fv.append(fv)

            if use_flip:
                flipped = TF.hflip(image)
                tensor_f = load_and_preprocess(flipped, scale).unsqueeze(0).to(device)
                _, fv_f = model(tensor_f, box_batch, return_features=True)
                view_fv.append(fv_f)

        fv_avg = torch.stack(view_fv, dim=0).mean(dim=0)
        all_fv.append(fv_avg.cpu())
        all_labels.append(labels)

    return torch.cat(all_fv, dim=0), torch.stack(all_labels, dim=0)


torch.cuda.empty_cache()

def main() -> None:
    args = parse_args()
    multi_scale = args.test_scales is not None and len(args.test_scales) > 0

    # Build model
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
    model.fisher.power_norm = False
    model.fisher.l2_norm = False

    # Shared patch boxes (fixed per image size)
    boxes = make_dense_patch_boxes(
        height=args.image_size,
        width=args.image_size,
        patch_sizes=tuple(args.patch_sizes),
        stride=args.patch_stride,
        max_patches=args.max_patches,
    )

    if args.fit_svm:
        # --- SVM path ---
        try:
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("ERROR: --fit-svm requires scikit-learn.")
            sys.exit(1)

        print("Extracting train FV features for SVM...")
        raw_train = VOCDetection(
            root=str(args.data_root), year=args.year,
            image_set=args.train_image_set, download=False,
        )
        train_fv, train_labels = extract_fv_multi_scale(
            model, raw_train, boxes, args.test_scales if multi_scale else [args.image_size],
            max_samples=args.max_samples, device=args.device,
            use_flip=False,  # paper: no flip for SVM training features
        )
        print(f"Train FV: {train_fv.shape}")
        # Apply power-norm + L2 (same as FisherLayer forward)
        train_fv = train_fv.sign() * train_fv.abs().sqrt()
        train_fv = F.normalize(train_fv, p=2, dim=1, eps=1e-6).numpy()
        train_labels_np = train_labels.numpy()

        scaler = StandardScaler()
        train_fv = scaler.fit_transform(train_fv)

        svm_models = []
        for c in range(train_labels_np.shape[1]):
            y_binary = train_labels_np[:, c]
            if y_binary.max() == 0:
                svm_models.append(None)
                continue
            svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000, tol=1e-3, random_state=7)
            svm.fit(train_fv, y_binary)
            svm_models.append(svm)

        print("Extracting test FV features...")
        raw_test = VOCDetection(
            root=str(args.data_root), year=args.year,
            image_set=args.image_set, download=False,
        )
        test_fv, test_labels = extract_fv_multi_scale(
            model, raw_test, boxes, args.test_scales if multi_scale else [args.image_size],
            max_samples=args.max_samples, device=args.device,
            use_flip=True,   # paper: +horizontal flip for SVM test
        )
        test_fv = test_fv.sign() * test_fv.abs().sqrt()
        test_fv = F.normalize(test_fv, p=2, dim=1, eps=1e-6).numpy()
        test_fv = scaler.transform(test_fv)
        test_labels_np = test_labels.numpy()

        all_scores = []
        for c, svm in enumerate(svm_models):
            if svm is None:
                all_scores.append(torch.zeros(test_fv.shape[0]))
            else:
                scores = svm.decision_function(test_fv)
                all_scores.append(torch.from_numpy(scores))

        scores_tensor = torch.stack(all_scores, dim=1).float()
        mean_ap, aps = mean_average_precision(test_labels, scores_tensor)
        print(f"\n=== SVM mAP: {mean_ap:.4f} ===")

    else:
        # --- FC classifier path ---
        if multi_scale:
            print(f"Multi-scale evaluation with scales={args.test_scales}")
            raw_eval = VOCDetection(
                root=str(args.data_root), year=args.year,
                image_set=args.image_set, download=False,
            )
            n = len(raw_eval)
            if args.max_samples is not None:
                n = min(n, args.max_samples)
            all_labels = []
            all_scores = []
            for idx in tqdm(range(n), desc="eval multi-scale"):
                image, target = raw_eval[idx]
                image = image.convert("RGB")
                labels = torch.zeros(len(PASCAL_CLASSES), dtype=torch.float32)
                objects = target["annotation"].get("object", [])
                if isinstance(objects, dict):
                    objects = [objects]
                for obj in objects:
                    name = obj["name"]
                    if name in PASCAL_CLASSES:
                        labels[PASCAL_CLASSES.index(name)] = 1.0

                logits = apply_test_scales(
                    model, image, boxes, args.test_scales, args.device, use_flip=True,
                )
                all_labels.append(labels)
                all_scores.append(torch.sigmoid(logits).cpu())

            targets = torch.stack(all_labels, dim=0)
            scores = torch.cat(all_scores, dim=0)
        else:
            # Single-scale: use the dataset loader
            eval_dataset = VOCClassification(
                root=args.data_root,
                year=args.year,
                image_set=args.image_set,
                image_size=args.image_size,
                patch_sizes=tuple(args.patch_sizes),
                patch_stride=args.patch_stride,
                max_patches=args.max_patches,
                download=False,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_voc_batch,
            )
            all_labels = []
            all_scores = []
            for batch_idx, batch in enumerate(tqdm(eval_loader, desc="eval"), start=1):
                if args.max_samples is not None and batch_idx > args.max_samples:
                    break
                images = batch["images"].to(args.device)
                labels = batch["labels"]
                boxes_b = [b.to(args.device) for b in batch["boxes"]]
                logits = model(images, boxes_b)
                all_labels.append(labels)
                all_scores.append(torch.sigmoid(logits).cpu())

            targets = torch.cat(all_labels, dim=0)
            scores = torch.cat(all_scores, dim=0)

        mean_ap, aps = mean_average_precision(targets, scores)
        print(f"\n=== mAP: {mean_ap:.4f} ===")

    for name, ap in zip(PASCAL_CLASSES, aps):
        print(f"  {name:12s} {ap:.4f}")


if __name__ == "__main__":
    main()
