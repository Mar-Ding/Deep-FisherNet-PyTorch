from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.configs import PRESETS, apply_preset
from fishernet.data.voc import PASCAL_CLASSES, VOCClassification, collate_voc_batch
from fishernet.models import build_fishernet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the missing ResNet101-spatial PCA and diagonal GMM parameters, "
            "then save both a PyTorch init .pt and a Caffe-field-compatible .mat."
        )
    )
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="official-res101-spatial-frozenbn")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/c3_res101_spatial_pca_gmm/init.pt"))
    parser.add_argument("--mat-output", type=Path)
    parser.add_argument("--year", default="2007")
    parser.add_argument("--train-set", default="trainval")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--backbone", choices=("resnet101-spatial",), default="resnet101-spatial")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--train-scales", type=int, nargs="+")
    parser.add_argument("--resize-mode", choices=("square", "longest"), default="square")
    parser.add_argument("--patch-dim", type=int, default=128)
    parser.add_argument("--num-components", type=int, default=64)
    parser.add_argument("--learn-priors", action="store_true")
    parser.add_argument("--fisher-parameterization", choices=("legacy", "caffe"), default="legacy")
    parser.add_argument("--fisher-include-log-det", action="store_true")
    parser.add_argument("--fisher-scale-by-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fisher-pooling", choices=("mean", "sum"), default="mean")
    parser.add_argument("--fisher-second-order-scale", type=float, default=2.0**-0.5)
    parser.add_argument("--fisher-caffe-backward-compat", action="store_true")
    parser.add_argument("--pca-l2-caffe-backward", action="store_true")
    parser.add_argument("--no-fisher-power-norm", action="store_true")
    parser.add_argument("--no-fisher-l2-norm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-raw-descriptors", type=int, default=50000)
    parser.add_argument("--max-gmm-descriptors", type=int, default=50000)
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--use-hflip", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args._explicit_args = {token for token in sys.argv[1:] if token.startswith("--")}
    return apply_preset(args)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_batch_norm(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def build_loader(args: argparse.Namespace) -> DataLoader:
    if args.batch_size != 1 and args.train_scales:
        raise ValueError("Use --batch-size 1 when train_scales are enabled; image sizes vary per sample.")
    dataset = VOCClassification(
        root=args.data_root,
        year=args.year,
        image_set=args.train_set,
        image_size=args.image_size,
        train_scales=tuple(args.train_scales) if args.train_scales else None,
        hflip_prob=0.5 if args.use_hflip else 0.0,
        patch_sizes=(64,),
        patch_stride=64,
        max_patches=1,
        resize_mode=args.resize_mode,
        download=args.download,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_voc_batch,
        pin_memory=str(args.device).startswith("cuda"),
        generator=generator,
    )


def sample_rows(tensor: torch.Tensor, max_rows: int) -> torch.Tensor:
    if tensor.shape[0] <= max_rows:
        return tensor
    index = torch.randperm(tensor.shape[0], device=tensor.device)[:max_rows]
    return tensor[index]


@torch.no_grad()
def collect_raw_res5c(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_descriptors: int,
) -> np.ndarray:
    model.eval()
    chunks: list[torch.Tensor] = []
    collected = 0
    for batch in tqdm(loader, desc="collect raw res5c"):
        remaining = max_descriptors - collected
        if remaining <= 0:
            break
        images = batch["images"].to(device, non_blocking=True)
        conv = model.features(images)
        desc = conv.permute(0, 2, 3, 1).reshape(-1, conv.shape[1])
        desc = sample_rows(desc, remaining)
        chunks.append(desc.detach().cpu())
        collected += desc.shape[0]
    if not chunks:
        raise RuntimeError("No raw descriptors collected.")
    return torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)


@torch.no_grad()
def collect_projected_descriptors(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_descriptors: int,
) -> np.ndarray:
    model.eval()
    chunks: list[torch.Tensor] = []
    collected = 0
    for batch in tqdm(loader, desc="collect PCA descriptors"):
        remaining = max_descriptors - collected
        if remaining <= 0:
            break
        images = batch["images"].to(device, non_blocking=True)
        descriptors, mask = model.extract_patch_features(images, None)
        desc = descriptors[mask]
        desc = sample_rows(desc, remaining)
        chunks.append(desc.detach().cpu())
        collected += desc.shape[0]
    if not chunks:
        raise RuntimeError("No projected descriptors collected.")
    return torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)


def fit_pca(raw: np.ndarray, patch_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, PCA]:
    pca = PCA(n_components=patch_dim, svd_solver="randomized", whiten=False, random_state=seed)
    pca.fit(raw)
    weight = torch.from_numpy(pca.components_.astype(np.float32)).view(patch_dim, raw.shape[1], 1, 1)
    bias_np = (-pca.mean_.dot(pca.components_.T)).astype(np.float32)
    bias = torch.from_numpy(bias_np)
    return weight, bias, pca


def fit_gmm(projected: np.ndarray, num_components: int, seed: int, max_iter: int) -> dict[str, torch.Tensor]:
    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type="diag",
        max_iter=max_iter,
        reg_covar=1e-6,
        random_state=seed,
        verbose=1,
    )
    gmm.fit(projected)
    means = torch.from_numpy(gmm.means_.astype(np.float32))
    sigmas = torch.from_numpy(np.sqrt(gmm.covariances_).astype(np.float32))
    priors = torch.from_numpy(gmm.weights_.astype(np.float32))
    return {"means": means, "sigmas": sigmas, "priors": priors}


def save_caffe_mat(path: Path, pca_weight: torch.Tensor, pca_bias: torch.Tensor, gmm: dict[str, torch.Tensor]) -> None:
    means = gmm["means"].numpy()
    sigmas = gmm["sigmas"].numpy()
    priors = gmm["priors"].numpy()
    weights = (1.0 / np.maximum(sigmas, 1e-6)).astype(np.float32)
    bias = (-means / np.maximum(sigmas, 1e-6)).astype(np.float32)
    pca_components = pca_weight.view(pca_weight.shape[0], pca_weight.shape[1]).numpy()
    savemat(
        path,
        {
            "weights": weights.reshape(-1, 1),
            "bias": bias.reshape(-1, 1),
            "priors_new": priors.reshape(-1, 1),
            "pca_w": pca_components.T.astype(np.float32),
            "pca_b": pca_bias.numpy().reshape(1, -1).astype(np.float32),
        },
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mat_output = args.mat_output or args.output.with_suffix(".mat")
    mat_output.parent.mkdir(parents=True, exist_ok=True)

    if args.backbone != "resnet101-spatial":
        raise ValueError("This initializer is only for backbone=resnet101-spatial.")

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
    ).to(args.device)
    freeze_batch_norm(model)

    loader = build_loader(args)
    print(f"Collecting up to {args.max_raw_descriptors} raw res5c descriptors...")
    raw = collect_raw_res5c(model, loader, args.device, args.max_raw_descriptors)
    print(f"Raw descriptor matrix: {raw.shape}")

    print(f"Fitting PCA 2048 -> {args.patch_dim}...")
    pca_weight, pca_bias, pca = fit_pca(raw, args.patch_dim, args.seed)
    model.pca.weight.data.copy_(pca_weight.to(args.device))
    model.pca.bias.data.copy_(pca_bias.to(args.device))

    loader = build_loader(args)
    print(f"Collecting up to {args.max_gmm_descriptors} projected descriptors...")
    projected = collect_projected_descriptors(model, loader, args.device, args.max_gmm_descriptors)
    projected = F.normalize(torch.from_numpy(projected), p=2, dim=1, eps=1e-6).numpy()
    print(f"Projected descriptor matrix: {projected.shape}")

    print(f"Fitting diagonal GMM with K={args.num_components}...")
    gmm = fit_gmm(projected, args.num_components, args.seed, args.gmm_max_iter)

    payload = {
        **gmm,
        "pca_weight": pca_weight,
        "pca_bias": pca_bias,
        "pca_mean": torch.from_numpy(pca.mean_.astype(np.float32)),
        "pca_explained_variance": torch.from_numpy(pca.explained_variance_.astype(np.float32)),
        "args": vars(args),
    }
    torch.save(payload, args.output)
    save_caffe_mat(mat_output, pca_weight, pca_bias, gmm)

    print(f"Saved PyTorch init: {args.output}")
    print(f"Saved Caffe-compatible mat: {mat_output}")


if __name__ == "__main__":
    main()
