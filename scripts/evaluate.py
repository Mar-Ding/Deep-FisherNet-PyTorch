from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
    load_and_preprocess_with_size,
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
    parser.add_argument("--backbone", choices=("alexnet", "vgg16", "resnet101", "resnet101-spatial"), default="alexnet")
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
    parser.add_argument("--resize-mode", choices=("square", "longest"), default="square")
    parser.add_argument("--learn-priors", action="store_true")
    parser.add_argument("--fisher-parameterization", choices=("legacy", "caffe"), default="legacy")
    parser.add_argument("--fisher-include-log-det", action="store_true")
    parser.add_argument("--fisher-scale-by-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fisher-pooling", choices=("mean", "sum"), default="mean")
    parser.add_argument("--fisher-second-order-scale", type=float, default=2.0**-0.5)
    parser.add_argument("--fisher-caffe-backward-compat", action="store_true")
    parser.add_argument("--fisher-assignment-sigma-scale", type=float, default=1.0)
    parser.add_argument("--fisher-assignment-temperature", type=float, default=1.0)
    parser.add_argument("--pca-l2-caffe-backward", action="store_true")
    parser.add_argument("--no-fisher-power-norm", action="store_true")
    parser.add_argument("--no-fisher-l2-norm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--fit-svm", action="store_true",
                        help="Extract FV features from train set, fit linear SVM, then evaluate.")
    parser.add_argument("--train-image-set", default="trainval",
                        help="Dataset split to use for SVM training.")
    parser.add_argument("--svm-C", type=float, default=1.0,
                        help="Linear SVM C parameter (paper: C=1).")
    parser.add_argument("--svm-solver", choices=("sgd", "liblinear"), default="sgd",
                        help="SVM solver for --fit-svm. liblinear is slower but a useful accuracy check.")
    parser.add_argument("--svm-max-iter", type=int, default=1000)
    parser.add_argument("--svm-tol", type=float, default=1e-3)
    parser.add_argument("--feature-cache-dir", type=Path,
                        help="Cache extracted FV tensors so SVM C/solver sweeps do not repeat GPU forward.")
    parser.add_argument("--no-standardize", action="store_true",
                        help="Skip StandardScaler before SVM. Useful for ablation only.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    return apply_preset(args)


@torch.inference_mode()
def apply_test_scales(
    model: torch.nn.Module,
    image: Image.Image,
    scales: list[int],
    device: str,
    patch_sizes: tuple[int, ...],
    patch_stride: int,
    max_patches: int | None,
    resize_mode: str,
    use_flip: bool = True,
) -> torch.Tensor:
    """Run one image through the model at multiple scales + optional flip, return averaged logits."""
    all_logits = []
    for scale in scales:
        tensor, height, width = load_and_preprocess_with_size(image, scale, resize_mode)
        tensor = tensor.unsqueeze(0).to(device)
        boxes = make_dense_patch_boxes(height, width, patch_sizes, patch_stride, max_patches)
        box_batch = [boxes.to(device)]
        logits = model(tensor, box_batch)
        all_logits.append(logits)

        if use_flip:
            flipped = TF.hflip(image)
            tensor_f, height_f, width_f = load_and_preprocess_with_size(flipped, scale, resize_mode)
            tensor_f = tensor_f.unsqueeze(0).to(device)
            boxes_f = make_dense_patch_boxes(height_f, width_f, patch_sizes, patch_stride, max_patches)
            box_batch_f = [boxes_f.to(device)]
            logits_f = model(tensor_f, box_batch_f)
            all_logits.append(logits_f)

    return torch.stack(all_logits, dim=0).mean(dim=0)


@torch.no_grad()
def extract_fv_multi_scale(
    model: torch.nn.Module,
    raw_dataset: VOCDetection,
    scales: list[int],
    max_samples: int | None,
    device: str,
    patch_sizes: tuple[int, ...],
    patch_stride: int,
    max_patches: int | None,
    resize_mode: str,
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
            tensor, height, width = load_and_preprocess_with_size(image, scale, resize_mode)
            tensor = tensor.unsqueeze(0).to(device)
            boxes = make_dense_patch_boxes(height, width, patch_sizes, patch_stride, max_patches)
            box_batch = [boxes.to(device)]
            _, fv = model(tensor, box_batch, return_features=True)
            view_fv.append(fv)

            if use_flip:
                flipped = TF.hflip(image)
                tensor_f, height_f, width_f = load_and_preprocess_with_size(flipped, scale, resize_mode)
                tensor_f = tensor_f.unsqueeze(0).to(device)
                boxes_f = make_dense_patch_boxes(height_f, width_f, patch_sizes, patch_stride, max_patches)
                _, fv_f = model(tensor_f, [boxes_f.to(device)], return_features=True)
                view_fv.append(fv_f)

        fv_avg = torch.stack(view_fv, dim=0).mean(dim=0)
        all_fv.append(fv_avg.cpu())
        all_labels.append(labels)

    return torch.cat(all_fv, dim=0), torch.stack(all_labels, dim=0)


def cache_name(split: str, args: argparse.Namespace, scales: list[int], use_flip: bool) -> str:
    scale_tag = "-".join(str(s) for s in scales)
    max_tag = "full" if args.max_samples is None else str(args.max_samples)
    flip_tag = "flip" if use_flip else "noflip"
    return f"{split}_{args.image_set if split == 'test' else args.train_image_set}_{scale_tag}_{flip_tag}_{max_tag}.pt"


def load_or_extract_fv(
    split: str,
    model: torch.nn.Module,
    raw_dataset: VOCDetection,
    scales: list[int],
    use_flip: bool,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = None
    if args.feature_cache_dir is not None:
        args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = args.feature_cache_dir / cache_name(split, args, scales, use_flip)
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            print(f"Loaded cached {split} FV: {cache_path}")
            return payload["fv"], payload["labels"]

    fv, labels = extract_fv_multi_scale(
        model, raw_dataset, scales,
        max_samples=args.max_samples, device=args.device,
        patch_sizes=tuple(args.patch_sizes), patch_stride=args.patch_stride,
        max_patches=args.max_patches, resize_mode=args.resize_mode,
        use_flip=use_flip,
    )
    if cache_path is not None:
        torch.save(
            {
                "fv": fv.cpu(),
                "labels": labels.cpu(),
                "meta": {
                    "checkpoint": str(args.checkpoint),
                    "preset": args.preset,
                    "scales": scales,
                    "use_flip": use_flip,
                    "max_samples": args.max_samples,
                    "resize_mode": args.resize_mode,
                    "patch_sizes": list(args.patch_sizes),
                    "patch_stride": args.patch_stride,
                    "max_patches": args.max_patches,
                },
            },
            cache_path,
        )
        print(f"Saved cached {split} FV: {cache_path}")
    return fv, labels


def normalize_fv_for_svm(fv: torch.Tensor) -> np.ndarray:
    fv = fv.sign() * fv.abs().sqrt()
    fv = F.normalize(fv, p=2, dim=1, eps=1e-6)
    return fv.numpy()


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
        learn_priors=args.learn_priors,
        fisher_parameterization=args.fisher_parameterization,
        fisher_include_log_det=args.fisher_include_log_det,
        fisher_scale_by_prior=args.fisher_scale_by_prior,
        fisher_pooling=args.fisher_pooling,
        fisher_power_norm=not args.no_fisher_power_norm,
        fisher_l2_norm=not args.no_fisher_l2_norm,
        fisher_second_order_scale=getattr(args, "fisher_second_order_scale", 2.0**-0.5),
        fisher_caffe_backward_compat=getattr(args, "fisher_caffe_backward_compat", False),
        fisher_assignment_sigma_scale=args.fisher_assignment_sigma_scale,
        fisher_assignment_temperature=args.fisher_assignment_temperature,
        pca_l2_caffe_backward=getattr(args, "pca_l2_caffe_backward", False),
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.fisher.power_norm = False
    model.fisher.l2_norm = False

    if args.fit_svm:
        # --- SVM path ---
        try:
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import LinearSVC
        except ImportError:
            print("ERROR: --fit-svm requires scikit-learn.")
            sys.exit(1)

        scales = args.test_scales if multi_scale else [args.image_size]
        print("Extracting train FV features for SVM...")
        raw_train = VOCDetection(
            root=str(args.data_root), year=args.year,
            image_set=args.train_image_set, download=False,
        )
        train_fv, train_labels = load_or_extract_fv(
            "train", model, raw_train, scales, False, args
        )
        print(f"Train FV: {train_fv.shape}")
        # Apply power-norm + L2 (same as FisherLayer forward)
        train_fv = normalize_fv_for_svm(train_fv)
        train_labels_np = train_labels.numpy()

        scaler = None
        if not args.no_standardize:
            scaler = StandardScaler()
            train_fv = scaler.fit_transform(train_fv)

        svm_models = []
        for c in range(train_labels_np.shape[1]):
            y_binary = train_labels_np[:, c]
            if y_binary.max() == 0:
                svm_models.append(None)
                continue
            if args.svm_solver == "sgd":
                alpha = 1.0 / (args.svm_C * train_fv.shape[0])
                svm = SGDClassifier(
                    loss="hinge",
                    penalty="l2",
                    alpha=alpha,
                    max_iter=args.svm_max_iter,
                    tol=args.svm_tol,
                    random_state=7,
                )
            else:
                svm = LinearSVC(
                    C=args.svm_C,
                    max_iter=args.svm_max_iter,
                    tol=args.svm_tol,
                    dual="auto",
                    random_state=7,
                )
            svm.fit(train_fv, y_binary)
            svm_models.append(svm)

        print("Extracting test FV features...")
        raw_test = VOCDetection(
            root=str(args.data_root), year=args.year,
            image_set=args.image_set, download=False,
        )
        test_fv, test_labels = load_or_extract_fv(
            "test", model, raw_test, scales, True, args
        )
        test_fv = normalize_fv_for_svm(test_fv)
        if scaler is not None:
            test_fv = scaler.transform(test_fv)

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
        print(
            f"SVM solver={args.svm_solver} C={args.svm_C} "
            f"standardize={not args.no_standardize} max_iter={args.svm_max_iter} tol={args.svm_tol}"
        )

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
                    model, image, args.test_scales, args.device,
                    patch_sizes=tuple(args.patch_sizes), patch_stride=args.patch_stride,
                    max_patches=args.max_patches, resize_mode=args.resize_mode, use_flip=True,
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
                resize_mode=args.resize_mode,
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
            with torch.inference_mode():
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
