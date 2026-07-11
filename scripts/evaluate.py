from __future__ import annotations

import argparse
import hashlib
import json
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
from fishernet.data.patches import make_dense_patch_boxes, transform_patch_boxes
from fishernet.data.voc import (
    PASCAL_CLASSES,
    VOCClassification,
    VOCClassificationLabelStore,
    collate_voc_batch,
    resize_image,
    tensorise_and_normalise,
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
    parser.add_argument("--patch-coordinate-frame", choices=("resized", "original"), default="resized")
    parser.add_argument("--patch-coordinate-mode", choices=("half_open", "caffe"), default="half_open")
    parser.add_argument("--patch-pooling", choices=("roi_align", "roi_pool"), default="roi_align")
    parser.add_argument("--patch-l2-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paper-new-layer-init", action="store_true")
    parser.add_argument("--label-source", choices=("xml", "classification"), default="xml")
    parser.add_argument("--ap-mode", choices=("continuous", "voc2007"), default="continuous")
    parser.add_argument("--test-flip", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--svm-fit-intercept", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--svm-normalize-per-view",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply signed-square-root and L2 to each scale before averaging, as in the paper.",
    )
    parser.add_argument("--feature-cache-dir", type=Path,
                        help="Cache extracted FV tensors so SVM C/solver sweeps do not repeat GPU forward.")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--no-standardize", action="store_true",
                        help="Skip StandardScaler before SVM. Useful for ablation only.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    return apply_preset(args)


def prepare_image_view(
    image: Image.Image,
    scale: int,
    resize_mode: str,
    patch_sizes: tuple[int, ...],
    patch_stride: int,
    max_patches: int | None,
    patch_coordinate_frame: str,
    patch_coordinate_mode: str,
    horizontal_flip: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare one image view while keeping paper patch coordinates scale-invariant."""
    image = image.convert("RGB")
    original_width, original_height = image.size
    boxes = None
    if patch_coordinate_frame == "original":
        boxes = make_dense_patch_boxes(
            original_height,
            original_width,
            patch_sizes,
            patch_stride,
            max_patches,
            coordinate_mode=patch_coordinate_mode,
        )

    resized = resize_image(image, scale, resize_mode)
    if horizontal_flip:
        resized = TF.hflip(resized)
    width, height = resized.size
    if boxes is None:
        boxes = make_dense_patch_boxes(
            height,
            width,
            patch_sizes,
            patch_stride,
            max_patches,
            coordinate_mode=patch_coordinate_mode,
        )
    else:
        boxes = transform_patch_boxes(
            boxes,
            source_hw=(original_height, original_width),
            target_hw=(height, width),
            horizontal_flip=horizontal_flip,
            coordinate_mode=patch_coordinate_mode,
        )
    return tensorise_and_normalise(resized), boxes


def labels_for_target(
    target: dict,
    label_store: VOCClassificationLabelStore | None,
) -> torch.Tensor:
    image_id = target["annotation"]["filename"]
    if label_store is not None:
        return label_store[image_id]
    return VOCClassification._multi_hot(target)


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
    patch_coordinate_frame: str,
    patch_coordinate_mode: str,
    use_flip: bool = True,
) -> torch.Tensor:
    """Run one image through the model at multiple scales + optional flip, return averaged logits."""
    all_logits = []
    for scale in scales:
        tensor, boxes = prepare_image_view(
            image, scale, resize_mode, patch_sizes, patch_stride, max_patches,
            patch_coordinate_frame, patch_coordinate_mode,
        )
        tensor = tensor.unsqueeze(0).to(device)
        box_batch = [boxes.to(device)]
        logits = model(tensor, box_batch)
        all_logits.append(logits)

        if use_flip:
            tensor_f, boxes_f = prepare_image_view(
                image, scale, resize_mode, patch_sizes, patch_stride, max_patches,
                patch_coordinate_frame, patch_coordinate_mode, horizontal_flip=True,
            )
            tensor_f = tensor_f.unsqueeze(0).to(device)
            box_batch_f = [boxes_f.to(device)]
            logits_f = model(tensor_f, box_batch_f)
            all_logits.append(logits_f)

    return torch.stack(all_logits, dim=0).mean(dim=0)


@torch.no_grad()
def extract_fv_multi_scale(
    model: torch.nn.Module,
    raw_dataset: VOCDetection,
    label_store: VOCClassificationLabelStore | None,
    scales: list[int],
    max_samples: int | None,
    device: str,
    patch_sizes: tuple[int, ...],
    patch_stride: int,
    max_patches: int | None,
    resize_mode: str,
    patch_coordinate_frame: str,
    patch_coordinate_mode: str,
    use_flip: bool = True,
    normalize_per_view: bool = False,
    disable_progress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract FV features for multi-scale evaluation, returns (fv, labels).

    Processes each image at all scales+flips, then averages FV per image. The
    paper path normalizes each scale's FV before this mean.
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
        labels = labels_for_target(target, label_store)

        view_fv = []
        for scale in scales:
            tensor, boxes = prepare_image_view(
                image, scale, resize_mode, patch_sizes, patch_stride, max_patches,
                patch_coordinate_frame, patch_coordinate_mode,
            )
            tensor = tensor.unsqueeze(0).to(device)
            box_batch = [boxes.to(device)]
            _, fv = model(tensor, box_batch, return_features=True)
            view_fv.append(normalize_fv_tensor(fv) if normalize_per_view else fv)

            if use_flip:
                tensor_f, boxes_f = prepare_image_view(
                    image, scale, resize_mode, patch_sizes, patch_stride, max_patches,
                    patch_coordinate_frame, patch_coordinate_mode, horizontal_flip=True,
                )
                tensor_f = tensor_f.unsqueeze(0).to(device)
                _, fv_f = model(tensor_f, [boxes_f.to(device)], return_features=True)
                view_fv.append(normalize_fv_tensor(fv_f) if normalize_per_view else fv_f)

        fv_avg = torch.stack(view_fv, dim=0).mean(dim=0)
        all_fv.append(fv_avg.cpu())
        all_labels.append(labels)

    return torch.cat(all_fv, dim=0), torch.stack(all_labels, dim=0)


def cache_name(split: str, args: argparse.Namespace, scales: list[int], use_flip: bool) -> str:
    scale_tag = "-".join(str(s) for s in scales)
    max_tag = "full" if args.max_samples is None else str(args.max_samples)
    flip_tag = "flip" if use_flip else "noflip"
    checkpoint = args.checkpoint.resolve()
    cache_identity = {
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "preset": args.preset,
        "backbone": args.backbone,
        "split": split,
        "scales": scales,
        "use_flip": use_flip,
        "resize_mode": args.resize_mode,
        "patch_sizes": list(args.patch_sizes),
        "patch_stride": args.patch_stride,
        "max_patches": args.max_patches,
        "patch_coordinate_frame": args.patch_coordinate_frame,
        "patch_coordinate_mode": args.patch_coordinate_mode,
        "patch_pooling": args.patch_pooling,
        "patch_l2_norm": args.patch_l2_norm,
        "label_source": args.label_source,
        "fisher_parameterization": args.fisher_parameterization,
        "fisher_include_log_det": args.fisher_include_log_det,
        "fisher_scale_by_prior": args.fisher_scale_by_prior,
        "fisher_pooling": args.fisher_pooling,
        "fisher_second_order_scale": args.fisher_second_order_scale,
        "fisher_assignment_sigma_scale": args.fisher_assignment_sigma_scale,
        "fisher_assignment_temperature": args.fisher_assignment_temperature,
        "svm_normalize_per_view": args.svm_normalize_per_view,
    }
    digest = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    image_set = args.image_set if split == "test" else args.train_image_set
    return f"{split}_{image_set}_{scale_tag}_{flip_tag}_{max_tag}_{digest}.pt"


def load_or_extract_fv(
    split: str,
    model: torch.nn.Module,
    raw_dataset: VOCDetection,
    label_store: VOCClassificationLabelStore | None,
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
        model, raw_dataset, label_store, scales,
        max_samples=args.max_samples, device=args.device,
        patch_sizes=tuple(args.patch_sizes), patch_stride=args.patch_stride,
        max_patches=args.max_patches, resize_mode=args.resize_mode,
        patch_coordinate_frame=args.patch_coordinate_frame,
        patch_coordinate_mode=args.patch_coordinate_mode,
        use_flip=use_flip,
        normalize_per_view=args.svm_normalize_per_view,
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
                    "patch_coordinate_frame": args.patch_coordinate_frame,
                    "patch_coordinate_mode": args.patch_coordinate_mode,
                    "patch_pooling": args.patch_pooling,
                    "patch_l2_norm": args.patch_l2_norm,
                    "label_source": args.label_source,
                    "svm_normalize_per_view": args.svm_normalize_per_view,
                },
            },
            cache_path,
        )
        print(f"Saved cached {split} FV: {cache_path}")
    return fv, labels


def normalize_fv_tensor(fv: torch.Tensor) -> torch.Tensor:
    fv = fv.sign() * fv.abs().sqrt()
    return F.normalize(fv, p=2, dim=1, eps=1e-6)


def normalize_fv_for_svm(fv: torch.Tensor) -> np.ndarray:
    return normalize_fv_tensor(fv).numpy()


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
        patch_pooling=args.patch_pooling,
        patch_l2_norm=args.patch_l2_norm,
        paper_new_layer_init=args.paper_new_layer_init,
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
        train_label_store = (
            VOCClassificationLabelStore(args.data_root, args.year, args.train_image_set)
            if args.label_source == "classification"
            else None
        )
        train_fv, train_labels = load_or_extract_fv(
            "train", model, raw_train, train_label_store, scales, False, args
        )
        print(f"Train FV: {train_fv.shape}")
        if args.svm_normalize_per_view:
            train_fv = train_fv.numpy()
        else:
            train_fv = normalize_fv_for_svm(train_fv)
        train_labels_np = train_labels.numpy()

        scaler = None
        if not args.no_standardize:
            scaler = StandardScaler()
            train_fv = scaler.fit_transform(train_fv)

        svm_models = []
        for c in range(train_labels_np.shape[1]):
            valid = train_labels_np[:, c] >= 0
            y_binary = train_labels_np[valid, c]
            class_features = train_fv[valid]
            if y_binary.size == 0 or np.unique(y_binary).size < 2:
                svm_models.append(None)
                continue
            if args.svm_solver == "sgd":
                alpha = 1.0 / (args.svm_C * class_features.shape[0])
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
                    fit_intercept=args.svm_fit_intercept,
                    random_state=7,
                )
            svm.fit(class_features, y_binary)
            svm_models.append(svm)

        print("Extracting test FV features...")
        raw_test = VOCDetection(
            root=str(args.data_root), year=args.year,
            image_set=args.image_set, download=False,
        )
        test_label_store = (
            VOCClassificationLabelStore(args.data_root, args.year, args.image_set)
            if args.label_source == "classification"
            else None
        )
        test_fv, test_labels = load_or_extract_fv(
            "test", model, raw_test, test_label_store, scales, args.test_flip, args
        )
        if args.svm_normalize_per_view:
            test_fv = test_fv.numpy()
        else:
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
        mean_ap, aps = mean_average_precision(test_labels, scores_tensor, mode=args.ap_mode)
        print(f"\n=== SVM mAP: {mean_ap:.4f} ===")
        print(
            f"SVM solver={args.svm_solver} C={args.svm_C} "
            f"standardize={not args.no_standardize} test_flip={args.test_flip} "
            f"intercept={args.svm_fit_intercept} AP={args.ap_mode} "
            f"normalize_per_view={args.svm_normalize_per_view} "
            f"max_iter={args.svm_max_iter} tol={args.svm_tol}"
        )

    else:
        # --- FC classifier path ---
        if multi_scale:
            print(f"Multi-scale evaluation with scales={args.test_scales}")
            raw_eval = VOCDetection(
                root=str(args.data_root), year=args.year,
                image_set=args.image_set, download=False,
            )
            eval_label_store = (
                VOCClassificationLabelStore(args.data_root, args.year, args.image_set)
                if args.label_source == "classification"
                else None
            )
            n = len(raw_eval)
            if args.max_samples is not None:
                n = min(n, args.max_samples)
            all_labels = []
            all_scores = []
            for idx in tqdm(range(n), desc="eval multi-scale"):
                image, target = raw_eval[idx]
                image = image.convert("RGB")
                labels = labels_for_target(target, eval_label_store)

                logits = apply_test_scales(
                    model, image, args.test_scales, args.device,
                    patch_sizes=tuple(args.patch_sizes), patch_stride=args.patch_stride,
                    max_patches=args.max_patches, resize_mode=args.resize_mode,
                    patch_coordinate_frame=args.patch_coordinate_frame,
                    patch_coordinate_mode=args.patch_coordinate_mode,
                    use_flip=args.test_flip,
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
                patch_coordinate_frame=args.patch_coordinate_frame,
                patch_coordinate_mode=args.patch_coordinate_mode,
                label_source=args.label_source,
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

        mean_ap, aps = mean_average_precision(targets, scores, mode=args.ap_mode)
        print(f"\n=== mAP: {mean_ap:.4f} ===")
        print(f"AP={args.ap_mode} test_flip={args.test_flip}")

    for name, ap in zip(PASCAL_CLASSES, aps):
        print(f"  {name:12s} {ap:.4f}")
    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mAP": mean_ap,
            "AP": {name: float(ap) for name, ap in zip(PASCAL_CLASSES, aps)},
            "ap_mode": args.ap_mode,
            "fit_svm": args.fit_svm,
            "svm_C": args.svm_C if args.fit_svm else None,
            "svm_solver": args.svm_solver if args.fit_svm else None,
            "svm_fit_intercept": args.svm_fit_intercept if args.fit_svm else None,
            "svm_normalize_per_view": args.svm_normalize_per_view if args.fit_svm else None,
            "standardize": not args.no_standardize if args.fit_svm else None,
            "test_scales": args.test_scales,
            "test_flip": args.test_flip,
            "checkpoint": str(args.checkpoint),
            "preset": args.preset,
        }
        args.metrics_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
