from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.configs import PRESETS, apply_preset
from fishernet.data.voc import PASCAL_CLASSES, VOCClassification, collate_voc_batch
from fishernet.models import build_fishernet, load_stage1_weights
from fishernet.utils import mean_average_precision


def atomic_torch_save(payload: object, path: Path) -> None:
    """Write a checkpoint without exposing a partially written destination."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Deep FisherNet on VOC. Use --stage1 for whole-image CNN finetune "
                    "(paper Stage 1), or omit for FisherNet end-to-end (paper Stage 2)."
    )
    parser.add_argument("--preset", choices=list(PRESETS.keys()))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fishernet_voc2007"))
    parser.add_argument("--year", default="2007")
    parser.add_argument("--train-set", default="trainval")
    parser.add_argument("--val-set", default="test")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--backbone", choices=("alexnet", "vgg16", "resnet101", "resnet101-spatial"), default="alexnet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="sgd")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-lr", type=float, default=1e-1)
    parser.add_argument("--classifier-bias-lr", type=float)
    parser.add_argument("--fisher-lr", type=float, default=1e-4)
    parser.add_argument("--fisher-bias-lr", type=float)
    parser.add_argument("--prior-lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--caffe-param-rules",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Caffe-style 2x bias learning rates and zero decay for ordinary biases.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr-step-ratio", type=float, default=0.6)
    parser.add_argument("--lr-step-iterations", type=int, nargs="*")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--disable-epoch-lr-scheduler", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--val-every-epochs", type=int, default=1)
    parser.add_argument(
        "--save-every-epochs", type=int,
        help="Also retain epoch_N.pt checkpoints at this interval.",
    )
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--train-scales", type=int, nargs="+")
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[64, 96, 128, 160, 192, 224, 256])
    parser.add_argument("--patch-stride", type=int, default=32)
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
    parser.add_argument(
        "--loss-normalization",
        choices=("elements", "batch"),
        default="elements",
        help=(
            "BCE reduction. 'batch' matches standard Caffe SigmoidCrossEntropyLoss; "
            "'elements' matches the released MulticlassSigmoidCrossEntropy backward pass."
        ),
    )
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
    parser.add_argument("--max-patches", type=int, default=800)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--fisher-init", type=Path)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-pca", action="store_true")
    parser.add_argument("--freeze-fisher", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage1", action="store_true",
                        help="Stage 1: whole-image CNN finetune (no Fisher layer).")
    parser.add_argument("--stage1-weights", type=Path,
                        help="Path to Stage 1 checkpoint. Loads features.* into FisherNet.")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--log-every-steps", type=int, default=20,
        help="Print step loss and gradient norm at this optimizer-step interval.",
    )
    parser.add_argument(
        "--finite-check-every-steps", type=int, default=20,
        help="Check all trainable parameters for NaN/Inf at this optimizer-step interval.",
    )
    parser.add_argument(
        "--gradient-clip-norm", type=float,
        help="Optional gradient clipping. Non-finite gradients always stop training.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    return apply_preset(args)


# ---------------------------------------------------------------------------
# Stage 1: whole-image CNN classifier
# ---------------------------------------------------------------------------

class WholeImageCNN(nn.Module):
    """Plain CNN classifier for Stage 1 finetuning (no patches, no Fisher layer)."""

    def __init__(
        self,
        backbone: str = "vgg16",
        num_classes: int = 20,
        pretrained: bool = True,
        patch_dim: int = 256,
        paper_new_layer_init: bool = False,
    ) -> None:
        super().__init__()
        if backbone == "vgg16":
            weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            base = vgg16(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            self.classifier = nn.Sequential(
                nn.Flatten(),
                base.classifier[0],
                base.classifier[1],
                base.classifier[2],
                base.classifier[3],
                base.classifier[4],
                base.classifier[5],
                nn.Linear(4096, patch_dim),
                nn.ReLU(inplace=True),
                nn.Linear(patch_dim, num_classes),
            )
            if paper_new_layer_init:
                nn.init.normal_(self.classifier[7].weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.classifier[7].bias)
                nn.init.normal_(self.classifier[9].weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.classifier[9].bias)
        elif backbone == "alexnet":
            from torchvision.models import AlexNet_Weights, alexnet
            weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
            base = alexnet(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            self.classifier = nn.Sequential(
                *list(base.classifier.children())[:-1],
                nn.Linear(4096, num_classes),
            )
        else:
            raise ValueError(f"Stage 1 not supported for backbone={backbone}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.avgpool(x)
        return self.classifier(x)


def multilabel_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    normalization: str = "elements",
) -> torch.Tensor:
    """Binary cross entropy that ignores VOC difficult labels marked -1.

    Standard Caffe SigmoidCrossEntropyLoss divides only by batch size, while the
    released MulticlassSigmoidCrossEntropy backward pass also divides by the
    channel count. Both modes are kept so each training path can match its source.
    """
    if normalization not in {"elements", "batch"}:
        raise ValueError("normalization must be 'elements' or 'batch'")
    valid = labels >= 0
    targets = labels.clamp(0.0, 1.0)
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denominator = (
        valid.sum().clamp_min(1)
        if normalization == "elements"
        else logits.new_tensor(max(1, logits.shape[0]))
    )
    return (losses * valid).sum() / denominator


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, str]:
    iteration_milestones = getattr(args, "lr_step_iterations", None)
    if iteration_milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(iteration_milestones), gamma=args.lr_gamma
        )
        return scheduler, "iteration"
    if getattr(args, "disable_epoch_lr_scheduler", False):
        return None, "none"
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.epochs * args.lr_step_ratio))],
        gamma=args.lr_gamma,
    )
    return scheduler, "epoch"


def split_weight_and_bias_params(
    named_parameters: list[tuple[str, nn.Parameter]],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    weights: list[nn.Parameter] = []
    biases: list[nn.Parameter] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if name.endswith(".bias"):
            biases.append(parameter)
        else:
            weights.append(parameter)
    return weights, biases


def append_caffe_param_groups(
    groups: list[dict[str, object]],
    named_parameters: list[tuple[str, nn.Parameter]],
    weight_lr: float,
    bias_lr: float,
) -> None:
    weights, biases = split_weight_and_bias_params(named_parameters)
    if weights:
        groups.append({"params": weights, "lr": weight_lr})
    if biases:
        groups.append({"params": biases, "lr": bias_lr, "weight_decay": 0.0})


def checked_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    gradient_clip_norm: float | None,
    finite_check_every_steps: int,
) -> float:
    """Reject a non-finite update before it can silently corrupt a checkpoint."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    max_norm = float("inf") if gradient_clip_norm is None else gradient_clip_norm
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm=max_norm, error_if_nonfinite=True
    )
    optimizer.step()

    check_every = max(1, finite_check_every_steps)
    if (global_step + 1) % check_every == 0:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and not torch.isfinite(parameter).all():
                raise RuntimeError(
                    f"Non-finite parameter after optimizer step {global_step + 1}: {name}"
                )
    return float(grad_norm.detach())


def train_stage1(args: argparse.Namespace) -> None:
    """Stage 1: whole-image CNN finetune (paper: 9k iters, batch=32, lr=0.01/0.001)."""
    print("=== Stage 1: Whole-image CNN finetune ===")
    print(
        f"  backbone={args.backbone}, epochs={args.epochs}, batch={args.batch_size}, "
        f"max_steps={args.max_steps}"
    )
    print(f"  backbone_lr={args.backbone_lr}, classifier_lr={args.classifier_lr}")

    train_dataset = VOCClassification(
        root=args.data_root, year=args.year, image_set=args.train_set,
        image_size=args.image_size, resize_mode=args.resize_mode,
        hflip_prob=args.hflip_prob, patch_sizes=(args.image_size,), max_patches=1,
        label_source=args.label_source, download=args.download,
    )
    val_dataset = VOCClassification(
        root=args.data_root, year=args.year, image_set=args.val_set,
        image_size=args.image_size, resize_mode=args.resize_mode,
        patch_sizes=(args.image_size,), max_patches=1,
        label_source=args.label_source, download=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_voc_batch,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_voc_batch,
        pin_memory=args.device.startswith("cuda"),
    )

    model = WholeImageCNN(
        backbone=args.backbone,
        num_classes=len(PASCAL_CLASSES),
        pretrained=args.pretrained,
        patch_dim=args.patch_dim,
        paper_new_layer_init=args.paper_new_layer_init,
    ).to(args.device)
    start_epoch = 1
    best_map = -1.0
    global_step = 0
    resume_ckpt = None
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(resume_ckpt["model"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_map = float(resume_ckpt.get("best_mAP", resume_ckpt.get("val_mAP", -1.0)))
        global_step = int(resume_ckpt.get("global_step", 0))

    existing_named_params = []
    new_named_params = []
    for name, param in model.named_parameters():
        if name.startswith(("classifier.7.", "classifier.9.")):
            new_named_params.append((name, param))
        else:
            existing_named_params.append((name, param))

    if args.caffe_param_rules:
        param_groups: list[dict[str, object]] = []
        append_caffe_param_groups(
            param_groups,
            existing_named_params,
            weight_lr=args.backbone_lr,
            bias_lr=args.backbone_lr * 2.0,
        )
        append_caffe_param_groups(
            param_groups,
            new_named_params,
            weight_lr=args.classifier_lr,
            bias_lr=(
                args.classifier_bias_lr
                if args.classifier_bias_lr is not None
                else args.classifier_lr * 2.0
            ),
        )
    else:
        param_groups = [
            {"params": [param for _, param in existing_named_params], "lr": args.backbone_lr},
            {"params": [param for _, param in new_named_params], "lr": args.classifier_lr},
        ]

    optimizer = torch.optim.SGD(
        param_groups,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler, scheduler_unit = build_lr_scheduler(optimizer, args)
    if resume_ckpt is not None:
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if scheduler is not None and "scheduler" in resume_ckpt and resume_ckpt["scheduler"]:
            scheduler.load_state_dict(resume_ckpt["scheduler"])

    last_eval_map = best_map
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        reached_step_limit = False
        num_batches = len(train_loader)
        if args.max_train_batches is not None:
            num_batches = min(num_batches, args.max_train_batches)
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"stage1 train {epoch}/{args.epochs}", disable=args.no_progress),
            start=1,
        ):
            if batch_idx > num_batches:
                break
            if args.max_steps is not None and global_step >= args.max_steps:
                reached_step_limit = True
                break
            images = batch["images"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            logits = model(images)
            loss = multilabel_bce(logits, labels, args.loss_normalization)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if scheduler is not None and scheduler_unit == "iteration":
                scheduler.step()
            running_loss += float(loss.detach()) * images.shape[0]
            seen += images.shape[0]
            if args.max_steps is not None and global_step >= args.max_steps:
                reached_step_limit = True
                break
        if scheduler is not None and scheduler_unit == "epoch":
            scheduler.step()

        train_loss = running_loss / max(1, seen)
        should_validate = (
            reached_step_limit
            or epoch == args.epochs
            or epoch % max(1, args.val_every_epochs) == 0
        )
        if should_validate:
            last_eval_map = evaluate_classifier(
                model, val_loader, args.device,
                disable_progress=args.no_progress, max_batches=args.max_val_batches,
                ap_mode=args.ap_mode,
            )
            print(
                f"stage1 epoch={epoch} step={global_step} train_loss={train_loss:.4f} "
                f"val_mAP={last_eval_map:.4f}"
            )
        else:
            print(f"stage1 epoch={epoch} step={global_step} train_loss={train_loss:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "val_mAP": last_eval_map,
            "best_mAP": max(best_map, last_eval_map),
            "args": vars(args),
        }
        atomic_torch_save(checkpoint, args.output_dir / "last.pt")
        if (
            args.save_every_epochs is not None
            and epoch % max(1, args.save_every_epochs) == 0
        ):
            atomic_torch_save(checkpoint, args.output_dir / f"epoch_{epoch}.pt")
        if should_validate and last_eval_map > best_map:
            best_map = last_eval_map
            atomic_torch_save(model.state_dict(), args.output_dir / "best_stage1_features.pt")
            atomic_torch_save(checkpoint, args.output_dir / "best.pt")
        if reached_step_limit:
            break


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module, loader: DataLoader, device: str,
    disable_progress: bool = False, max_batches: int | None = None,
    ap_mode: str = "continuous",
) -> float:
    model.eval()
    all_labels, all_scores = [], []
    for batch_idx, batch in enumerate(tqdm(loader, desc="val", disable=disable_progress), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        images = batch["images"].to(device)
        labels = batch["labels"]
        logits = model(images)
        all_labels.append(labels)
        all_scores.append(torch.sigmoid(logits).cpu())
    targets = torch.cat(all_labels, dim=0)
    scores = torch.cat(all_scores, dim=0)
    mean_ap, _ = mean_average_precision(targets, scores, mode=ap_mode)
    return mean_ap


# ---------------------------------------------------------------------------
# Stage 2: FisherNet end-to-end training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 mode ---
    if args.stage1:
        train_stage1(args)
        return

    # --- Stage 2: FisherNet end-to-end ---
    train_dataset = VOCClassification(
        root=args.data_root, year=args.year, image_set=args.train_set,
        image_size=args.image_size,
        train_scales=tuple(args.train_scales) if args.train_scales else None,
        hflip_prob=args.hflip_prob,
        patch_sizes=tuple(args.patch_sizes), patch_stride=args.patch_stride,
        max_patches=args.max_patches, resize_mode=args.resize_mode,
        patch_coordinate_frame=args.patch_coordinate_frame,
        patch_coordinate_mode=args.patch_coordinate_mode,
        label_source=args.label_source, download=args.download,
    )
    val_dataset = VOCClassification(
        root=args.data_root, year=args.year, image_set=args.val_set,
        image_size=args.image_size,
        patch_sizes=tuple(args.patch_sizes), patch_stride=args.patch_stride,
        max_patches=args.max_patches, resize_mode=args.resize_mode,
        patch_coordinate_frame=args.patch_coordinate_frame,
        patch_coordinate_mode=args.patch_coordinate_mode,
        label_source=args.label_source, download=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_voc_batch,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_voc_batch,
        pin_memory=args.device.startswith("cuda"),
    )

    model = build_fishernet(
        backbone=args.backbone, num_classes=len(PASCAL_CLASSES),
        patch_dim=args.patch_dim, num_components=args.num_components,
        pretrained=args.pretrained, roi_output_size=args.roi_output_size,
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
        fisher_assignment_sigma_scale=args.fisher_assignment_sigma_scale,
        fisher_assignment_temperature=args.fisher_assignment_temperature,
        pca_l2_caffe_backward=getattr(args, "pca_l2_caffe_backward", False),
        patch_pooling=args.patch_pooling,
        patch_l2_norm=args.patch_l2_norm,
        paper_new_layer_init=args.paper_new_layer_init,
    ).to(args.device)

    # Load Stage 1 backbone weights if provided
    if args.stage1_weights is not None:
        print(f"Loading Stage 1 backbone weights from {args.stage1_weights}")
        stage1_ckpt = torch.load(args.stage1_weights, map_location=args.device)
        sd = stage1_ckpt if isinstance(stage1_ckpt, dict) and "model" not in stage1_ckpt \
            else stage1_ckpt.get("model", stage1_ckpt)
        loaded = load_stage1_weights(model, sd)
        transferred_heads = [pair for pair in loaded if pair[0].startswith("classifier.")]
        print(
            f"  Loaded {len(loaded)} Stage 1 tensors "
            f"({len(transferred_heads)}/6 fc6/fc7/reduction tensors)"
        )
        if args.backbone == "vgg16" and len(transferred_heads) != 6:
            raise RuntimeError(
                "Stage 1 checkpoint is incompatible: fc6, fc7, and the 256-d reduction "
                "must all transfer before paper Stage 2 training."
            )

    if args.fisher_init is not None:
        init = torch.load(args.fisher_init, map_location=args.device)
        if hasattr(model, "pca") and "pca_weight" in init:
            model.pca.weight.data.copy_(init["pca_weight"].to(args.device))
            if "pca_bias" in init and model.pca.bias is not None:
                model.pca.bias.data.copy_(init["pca_bias"].to(args.device))
            print(f"Loaded PCA projection from {args.fisher_init}")
        model.fisher.initialize_from_gmm(init["means"], init["sigmas"], init.get("priors"))
        print(f"Loaded Fisher GMM init from {args.fisher_init}")

    optimizer = build_optimizer(model, args)
    scheduler, scheduler_unit = build_lr_scheduler(optimizer, args)
    start_epoch = 1
    best_map = -1.0
    global_step = 0
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(resume_ckpt["model"])
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if scheduler is not None and "scheduler" in resume_ckpt and resume_ckpt["scheduler"]:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_map = float(resume_ckpt.get("best_mAP", resume_ckpt.get("val_mAP", -1.0)))
        global_step = int(resume_ckpt.get("global_step", 0))

    last_eval_map = best_map
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        accumulated_batches = 0
        reached_step_limit = False
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(train_loader)
        if args.max_train_batches is not None:
            num_batches = min(num_batches, args.max_train_batches)
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", disable=args.no_progress),
            start=1,
        ):
            if batch_idx > num_batches:
                break
            if args.max_steps is not None and global_step >= args.max_steps:
                reached_step_limit = True
                break
            images = batch["images"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            boxes = [b.to(args.device) for b in batch["boxes"]]
            logits = model(images, boxes)
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits at epoch={epoch} batch={batch_idx} step={global_step}"
                )
            loss = multilabel_bce(logits, labels, args.loss_normalization) / args.grad_accum_steps
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch} batch={batch_idx} step={global_step}"
                )
            loss.backward()
            running_loss += float(loss.detach()) * args.grad_accum_steps * images.shape[0]
            seen_samples += images.shape[0]
            accumulated_batches += 1
            if batch_idx % args.grad_accum_steps == 0 or batch_idx == num_batches:
                grad_norm = checked_optimizer_step(
                    model, optimizer, global_step, args.gradient_clip_norm,
                    args.finite_check_every_steps,
                )
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                global_step += 1
                if global_step == 1 or global_step % max(1, args.log_every_steps) == 0:
                    step_loss = float(loss.detach()) * args.grad_accum_steps
                    print(
                        f"train_progress epoch={epoch} step={global_step} "
                        f"step_loss={step_loss:.6f} grad_norm={grad_norm:.6f}",
                        flush=True,
                    )
                if scheduler is not None and scheduler_unit == "iteration":
                    scheduler.step()
                if args.max_steps is not None and global_step >= args.max_steps:
                    reached_step_limit = True
                    break
        if accumulated_batches:
            grad_norm = checked_optimizer_step(
                model, optimizer, global_step, args.gradient_clip_norm,
                args.finite_check_every_steps,
            )
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step == 1 or global_step % max(1, args.log_every_steps) == 0:
                step_loss = float(loss.detach()) * args.grad_accum_steps
                print(
                    f"train_progress epoch={epoch} step={global_step} "
                    f"step_loss={step_loss:.6f} grad_norm={grad_norm:.6f}",
                    flush=True,
                )
            if scheduler is not None and scheduler_unit == "iteration":
                scheduler.step()
            if args.max_steps is not None and global_step >= args.max_steps:
                reached_step_limit = True
        if scheduler is not None and scheduler_unit == "epoch":
            scheduler.step()

        train_loss = running_loss / max(1, seen_samples)
        should_validate = (
            reached_step_limit
            or epoch == args.epochs
            or epoch % max(1, args.val_every_epochs) == 0
        )
        if should_validate:
            last_eval_map = evaluate(
                model, val_loader, args.device,
                disable_progress=args.no_progress, max_batches=args.max_val_batches,
                ap_mode=args.ap_mode,
            )
            print(
                f"epoch={epoch} step={global_step} train_loss={train_loss:.4f} "
                f"val_mAP={last_eval_map:.4f}"
            )
        else:
            print(f"epoch={epoch} step={global_step} train_loss={train_loss:.4f}")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "val_mAP": last_eval_map,
            "best_mAP": max(best_map, last_eval_map),
            "args": vars(args),
        }
        atomic_torch_save(checkpoint, args.output_dir / "last.pt")
        if (
            args.save_every_epochs is not None
            and epoch % max(1, args.save_every_epochs) == 0
        ):
            atomic_torch_save(checkpoint, args.output_dir / f"epoch_{epoch}.pt")
        if should_validate and last_eval_map > best_map:
            best_map = last_eval_map
            atomic_torch_save(checkpoint, args.output_dir / "best.pt")
        if reached_step_limit:
            break


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if getattr(args, "freeze_backbone", False):
        for param in model.features.parameters():
            param.requires_grad_(False)
    feature_named_params = [
        (f"features.{name}", param)
        for name, param in model.features.named_parameters()
        if param.requires_grad
    ]
    if hasattr(model, "patch_mlp"):
        feature_named_params += [
            (f"patch_mlp.{name}", param)
            for name, param in model.patch_mlp.named_parameters()
            if param.requires_grad
        ]
    if hasattr(model, "pca"):
        if getattr(args, "freeze_pca", False):
            for param in model.pca.parameters():
                param.requires_grad_(False)
        feature_named_params += [
            (f"pca.{name}", param)
            for name, param in model.pca.named_parameters()
            if param.requires_grad
        ]
    if getattr(args, "freeze_fisher", False):
        for param in model.fisher.parameters():
            param.requires_grad_(False)
    fisher_bias_lr = args.fisher_lr if args.fisher_bias_lr is None else args.fisher_bias_lr
    classifier_bias_lr = args.classifier_lr if args.classifier_bias_lr is None else args.classifier_bias_lr
    param_groups = []
    if args.caffe_param_rules:
        append_caffe_param_groups(
            param_groups,
            feature_named_params,
            weight_lr=args.backbone_lr,
            bias_lr=args.backbone_lr * 2.0,
        )
    elif feature_named_params:
        param_groups.append({
            "params": [param for _, param in feature_named_params],
            "lr": args.backbone_lr,
        })
    if model.fisher.weight.requires_grad:
        param_groups.append({"params": [model.fisher.weight], "lr": args.fisher_lr})
    if model.fisher.bias.requires_grad:
        param_groups.append({"params": [model.fisher.bias], "lr": fisher_bias_lr})
    if model.classifier.weight.requires_grad:
        param_groups.append({"params": [model.classifier.weight], "lr": args.classifier_lr})
    if model.classifier.bias.requires_grad:
        param_groups.append({"params": [model.classifier.bias], "lr": classifier_bias_lr, "weight_decay": 0.0})
    if model.fisher.prior_logits is not None and model.fisher.prior_logits.requires_grad:
        prior_lr = args.fisher_lr if args.prior_lr is None else args.prior_lr
        param_groups.append({"params": [model.fisher.prior_logits], "lr": prior_lr})
    if not param_groups:
        raise ValueError("No trainable parameters left after applying freeze options.")
    if args.optimizer == "sgd":
        return torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)
    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: str,
    disable_progress: bool = False, max_batches: int | None = None,
    ap_mode: str = "continuous",
) -> float:
    model.eval()
    all_labels, all_scores = [], []
    for batch_idx, batch in enumerate(tqdm(loader, desc="val", disable=disable_progress), start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        images = batch["images"].to(device)
        labels = batch["labels"]
        boxes = [b.to(device) for b in batch["boxes"]]
        logits = model(images, boxes)
        all_labels.append(labels)
        all_scores.append(torch.sigmoid(logits).cpu())
    targets = torch.cat(all_labels, dim=0)
    scores = torch.cat(all_scores, dim=0)
    mean_ap, _ = mean_average_precision(targets, scores, mode=ap_mode)
    return mean_ap


if __name__ == "__main__":
    main()
