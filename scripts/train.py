from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.configs import PRESETS, apply_preset
from fishernet.data.voc import PASCAL_CLASSES, VOCClassification, collate_voc_batch
from fishernet.models import build_fishernet
from fishernet.utils import mean_average_precision


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
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr-step-ratio", type=float, default=0.6)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--train-scales", type=int, nargs="+")
    parser.add_argument("--hflip-prob", type=float, default=0.0)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[64, 96, 128, 160, 192, 224, 256])
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--patch-dim", type=int, default=256)
    parser.add_argument("--num-components", type=int, default=32)
    parser.add_argument("--roi-output-size", type=int)
    parser.add_argument("--resize-mode", choices=("square", "longest"), default="square")
    parser.add_argument("--learn-priors", action="store_true")
    parser.add_argument("--fisher-parameterization", choices=("legacy", "caffe"), default="legacy")
    parser.add_argument("--fisher-include-log-det", action="store_true")
    parser.add_argument("--fisher-scale-by-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fisher-pooling", choices=("mean", "sum"), default="mean")
    parser.add_argument("--no-fisher-power-norm", action="store_true")
    parser.add_argument("--no-fisher-l2-norm", action="store_true")
    parser.add_argument("--max-patches", type=int, default=800)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--fisher-init", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage1", action="store_true",
                        help="Stage 1: whole-image CNN finetune (no Fisher layer).")
    parser.add_argument("--stage1-weights", type=Path,
                        help="Path to Stage 1 checkpoint. Loads features.* into FisherNet.")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    return apply_preset(args)


# ---------------------------------------------------------------------------
# Stage 1: whole-image CNN classifier
# ---------------------------------------------------------------------------

class WholeImageCNN(nn.Module):
    """Plain CNN classifier for Stage 1 finetuning (no patches, no Fisher layer)."""

    def __init__(self, backbone: str = "vgg16", num_classes: int = 20, pretrained: bool = True) -> None:
        super().__init__()
        if backbone == "vgg16":
            weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            base = vgg16(weights=weights)
            self.features = base.features   # includes pool5 for whole-image use
            in_features = base.classifier[0].in_features  # 25088
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes),
            )
        elif backbone == "alexnet":
            from torchvision.models import AlexNet_Weights, alexnet
            weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
            base = alexnet(weights=weights)
            self.features = base.features
            self.classifier = nn.Sequential(
                *list(base.classifier.children())[:-1],
                nn.Linear(4096, num_classes),
            )
        else:
            raise ValueError(f"Stage 1 not supported for backbone={backbone}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        return self.classifier(x)


def train_stage1(args: argparse.Namespace) -> None:
    """Stage 1: whole-image CNN finetune (paper: 9k iters, batch=32, lr=0.01/0.001)."""
    print("=== Stage 1: Whole-image CNN finetune ===")
    print(f"  backbone={args.backbone}, epochs={args.epochs}, batch={args.batch_size}")
    print(f"  backbone_lr={args.backbone_lr}, classifier_lr={args.classifier_lr}")

    train_dataset = VOCClassification(
        root=args.data_root, year=args.year, image_set=args.train_set,
        image_size=args.image_size, resize_mode=args.resize_mode, download=args.download,
    )
    val_dataset = VOCClassification(
        root=args.data_root, year=args.year, image_set=args.val_set,
        image_size=args.image_size, resize_mode=args.resize_mode, download=False,
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

    model = WholeImageCNN(backbone=args.backbone, num_classes=len(PASCAL_CLASSES),
                          pretrained=args.pretrained).to(args.device)
    start_epoch = 1
    best_map = -1.0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_map = float(ckpt.get("val_mAP", -1.0))

    # Paper Stage 1: lr=0.01 for most layers, lr=0.001 for the last fc layer
    backbone_params = []
    last_fc_params = []
    for name, param in model.named_parameters():
        if "classifier.6" in name:  # last fc layer (index 6 in the Sequential)
            last_fc_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.SGD([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": last_fc_params, "lr": args.classifier_lr},
    ], momentum=args.momentum, weight_decay=args.weight_decay)

    # Paper: lr drops at 60% of iterations
    milestone = max(1, int(args.epochs * args.lr_step_ratio))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[milestone], gamma=args.lr_gamma,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        num_batches = len(train_loader)
        if args.max_train_batches is not None:
            num_batches = min(num_batches, args.max_train_batches)
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"stage1 train {epoch}/{args.epochs}", disable=args.no_progress),
            start=1,
        ):
            if batch_idx > num_batches:
                break
            images = batch["images"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach()) * images.shape[0]
            seen += images.shape[0]
        scheduler.step()

        train_loss = running_loss / max(1, seen)
        mean_ap = evaluate_classifier(
            model, val_loader, args.device,
            disable_progress=args.no_progress, max_batches=args.max_val_batches,
        )
        print(f"stage1 epoch={epoch} train_loss={train_loss:.4f} val_mAP={mean_ap:.4f}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_mAP": mean_ap,
            "args": vars(args),
        }, args.output_dir / "last.pt")
        if mean_ap > best_map:
            best_map = mean_ap
            torch.save(model.state_dict(), args.output_dir / "best_stage1_features.pt")
            torch.save({
                "model": model.state_dict(),
                "val_mAP": mean_ap,
            }, args.output_dir / "best.pt")


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module, loader: DataLoader, device: str,
    disable_progress: bool = False, max_batches: int | None = None,
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
    mean_ap, _ = mean_average_precision(targets, scores)
    return mean_ap


# ---------------------------------------------------------------------------
# Stage 2: FisherNet end-to-end training
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
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
        max_patches=args.max_patches, resize_mode=args.resize_mode, download=args.download,
    )
    val_dataset = VOCClassification(
        root=args.data_root, year=args.year, image_set=args.val_set,
        image_size=args.image_size,
        patch_sizes=tuple(args.patch_sizes), patch_stride=args.patch_stride,
        max_patches=args.max_patches, resize_mode=args.resize_mode, download=False,
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
        fisher_parameterization=args.fisher_parameterization,
        fisher_include_log_det=args.fisher_include_log_det,
        fisher_scale_by_prior=args.fisher_scale_by_prior,
        fisher_pooling=args.fisher_pooling,
        fisher_power_norm=not args.no_fisher_power_norm,
        fisher_l2_norm=not args.no_fisher_l2_norm,
    ).to(args.device)

    # Load Stage 1 backbone weights if provided
    if args.stage1_weights is not None:
        print(f"Loading Stage 1 backbone weights from {args.stage1_weights}")
        stage1_ckpt = torch.load(args.stage1_weights, map_location=args.device)
        sd = stage1_ckpt if isinstance(stage1_ckpt, dict) and "model" not in stage1_ckpt \
            else stage1_ckpt.get("model", stage1_ckpt)
        # Load only features.* keys (conv layers), skip classifier keys
        model_sd = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k.startswith("features.") and k in model_sd and v.shape == model_sd[k].shape:
                model_sd[k] = v
                loaded += 1
        model.load_state_dict(model_sd)
        print(f"  Loaded {loaded} feature layers from Stage 1 checkpoint")

    if args.fisher_init is not None:
        init = torch.load(args.fisher_init, map_location=args.device)
        model.fisher.initialize_from_gmm(init["means"], init["sigmas"], init.get("priors"))

    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.epochs * args.lr_step_ratio))],
        gamma=args.lr_gamma,
    )
    start_epoch = 1
    best_map = -1.0
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(resume_ckpt["model"])
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_map = float(resume_ckpt.get("val_mAP", -1.0))

    for epoch in range(start_epoch, start_epoch + args.epochs - 1 + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0
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
            images = batch["images"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)
            boxes = [b.to(args.device) for b in batch["boxes"]]
            logits = model(images, boxes)
            loss = F.binary_cross_entropy_with_logits(logits, labels) / args.grad_accum_steps
            loss.backward()
            running_loss += float(loss.detach()) * args.grad_accum_steps * images.shape[0]
            seen_samples += images.shape[0]
            if batch_idx % args.grad_accum_steps == 0 or batch_idx == num_batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        train_loss = running_loss / max(1, seen_samples)
        mean_ap = evaluate(
            model, val_loader, args.device,
            disable_progress=args.no_progress, max_batches=args.max_val_batches,
        )
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_mAP={mean_ap:.4f}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_mAP": mean_ap,
            "args": vars(args),
        }, args.output_dir / "last.pt")
        if mean_ap > best_map:
            best_map = mean_ap
            best_checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val_mAP": mean_ap,
                "args": vars(args),
            }
            torch.save(best_checkpoint, args.output_dir / "best.pt")


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    feature_params = list(model.features.parameters())
    if hasattr(model, "patch_mlp"):
        feature_params += list(model.patch_mlp.parameters())
    if hasattr(model, "pca"):
        feature_params += list(model.pca.parameters())
    fisher_bias_lr = args.fisher_lr if args.fisher_bias_lr is None else args.fisher_bias_lr
    classifier_bias_lr = args.classifier_lr if args.classifier_bias_lr is None else args.classifier_bias_lr
    param_groups = [
        {"params": feature_params, "lr": args.backbone_lr},
        {"params": [model.fisher.weight], "lr": args.fisher_lr},
        {"params": [model.fisher.bias], "lr": fisher_bias_lr},
        {"params": [model.classifier.weight], "lr": args.classifier_lr},
        {"params": [model.classifier.bias], "lr": classifier_bias_lr, "weight_decay": 0.0},
    ]
    if model.fisher.prior_logits is not None:
        param_groups.append({"params": [model.fisher.prior_logits], "lr": args.fisher_lr})
    if args.optimizer == "sgd":
        return torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)
    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: str,
    disable_progress: bool = False, max_batches: int | None = None,
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
    mean_ap, _ = mean_average_precision(targets, scores)
    return mean_ap


if __name__ == "__main__":
    main()
