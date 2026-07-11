from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fishernet.configs import PRESETS
from fishernet.data.voc import PASCAL_CLASSES, VOCClassification, collate_voc_batch
from fishernet.models import build_fishernet, load_stage1_weights


EXPECTED_FINAL_CONFIG = {
    "backbone": "vgg16",
    "batch_size": 2,
    "max_steps": 40000,
    "backbone_lr": 1e-4,
    "fisher_lr": 1e-1,
    "fisher_bias_lr": 2e-1,
    "classifier_lr": 1e-3,
    "classifier_bias_lr": 2e-3,
    "caffe_param_rules": True,
    "patch_coordinate_frame": "original",
    "patch_coordinate_mode": "caffe",
    "patch_pooling": "roi_pool",
    "patch_l2_norm": False,
    "fisher_parameterization": "caffe",
    "fisher_caffe_backward_compat": True,
    "learn_priors": False,
    "fisher_include_log_det": False,
    "fisher_scale_by_prior": False,
    "fisher_pooling": "mean",
    "no_fisher_power_norm": True,
    "no_fisher_l2_norm": True,
    # The released Caffe MulticlassSigmoidCrossEntropy backward pass divides by
    # batch size and channel count, equivalent to PyTorch's element mean.
    "loss_normalization": "elements",
    "test_flip": False,
    "svm_C": 1.0,
    "svm_solver": "liblinear",
    "svm_normalize_per_view": True,
    "no_standardize": True,
    "ap_mode": "voc2007",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the exact VGG16 paper reproduction path.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--gmm", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--max-descriptors", type=int, default=10000)
    parser.add_argument("--scale", type=int, default=688)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def audit_static_config() -> dict[str, object]:
    stage1 = PRESETS["stage1-vgg16-paper-final"]
    final = PRESETS["vgg16-paper-final"]
    mismatches = {
        key: {"expected": expected, "actual": final.get(key)}
        for key, expected in EXPECTED_FINAL_CONFIG.items()
        if final.get(key) != expected
    }
    stage1_ok = (
        stage1.get("max_steps") == 9000
        and stage1.get("lr_step_iterations") == [3000, 6000]
        and stage1.get("backbone_lr") == 1e-3
        and stage1.get("classifier_lr") == 1e-2
        and stage1.get("classifier_bias_lr") == 2e-2
        and stage1.get("caffe_param_rules") is True
        and stage1.get("loss_normalization") == "batch"
    )
    return {
        "stage1_config_ok": stage1_ok,
        "stage2_config_ok": not mismatches,
        "stage2_mismatches": mismatches,
    }


@torch.inference_mode()
def audit_artifacts(args: argparse.Namespace) -> dict[str, object]:
    if args.data_root is None or args.stage1_checkpoint is None or args.gmm is None:
        raise ValueError("--data-root, --stage1-checkpoint, and --gmm are required together")

    config = PRESETS["vgg16-paper-final"]
    model = build_fishernet(
        backbone="vgg16",
        num_classes=len(PASCAL_CLASSES),
        patch_dim=config["patch_dim"],
        num_components=config["num_components"],
        pretrained=False,
        roi_output_size=config["roi_output_size"],
        learn_priors=False,
        fisher_parameterization="legacy",
        fisher_include_log_det=False,
        fisher_scale_by_prior=False,
        fisher_pooling="mean",
        fisher_power_norm=False,
        fisher_l2_norm=False,
        fisher_second_order_scale=config["fisher_second_order_scale"],
        patch_pooling="roi_pool",
        patch_l2_norm=False,
        paper_new_layer_init=True,
    ).to(args.device)

    stage1 = torch.load(args.stage1_checkpoint, map_location=args.device)
    stage1_state = stage1.get("model", stage1)
    loaded = load_stage1_weights(model, stage1_state)
    transferred_heads = [pair for pair in loaded if pair[0].startswith("classifier.")]

    gmm = torch.load(args.gmm, map_location=args.device)
    means = gmm["means"].to(args.device)
    sigmas = gmm["sigmas"].to(args.device)
    priors = gmm.get("priors")
    model.fisher.initialize_from_gmm(means, sigmas, priors)
    model.eval()

    dataset = VOCClassification(
        root=args.data_root,
        year="2007",
        image_set="trainval",
        image_size=args.scale,
        patch_sizes=tuple(config["patch_sizes"]),
        patch_stride=config["patch_stride"],
        max_patches=config["max_patches"],
        resize_mode="longest",
        patch_coordinate_frame="original",
        patch_coordinate_mode="caffe",
        label_source="classification",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_voc_batch,
    )

    descriptor_chunks = []
    patch_counts = []
    fv_finite = True
    for image_index, batch in enumerate(tqdm(loader, desc="paper audit")):
        if image_index >= args.max_images:
            break
        images = batch["images"].to(args.device)
        boxes = [box.to(args.device) for box in batch["boxes"]]
        features, mask = model.extract_patch_features(images, boxes)
        valid_features = features[mask]
        descriptor_chunks.append(valid_features)
        patch_counts.append(int(mask.sum()))
        fv = model.fisher(features, mask)
        fv_finite = fv_finite and bool(torch.isfinite(fv).all())
        if sum(chunk.shape[0] for chunk in descriptor_chunks) >= args.max_descriptors:
            break

    descriptors = torch.cat(descriptor_chunks, dim=0)[: args.max_descriptors]
    standardized = (descriptors[:, None, :] - means[None, :, :]) / sigmas[None, :, :]
    gamma = torch.softmax(-0.5 * standardized.square().sum(dim=-1), dim=-1)
    usage = gamma.mean(dim=0)
    usage_effective = torch.exp(-(usage * usage.clamp_min(1e-12).log()).sum())
    assignment_entropy = -(gamma * gamma.clamp_min(1e-12).log()).sum(dim=1)

    if priors is not None:
        priors_cpu = priors.float().clamp_min(1e-12)
        prior_effective = float(torch.exp(-(priors_cpu * priors_cpu.log()).sum()))
    else:
        prior_effective = math.nan

    return {
        "stage1_global_step": int(stage1.get("global_step", -1)),
        "stage1_val_mAP": float(stage1.get("val_mAP", -1.0)),
        "stage1_loaded_tensors": len(loaded),
        "stage1_head_tensors": len(transferred_heads),
        "gmm_shape_ok": list(means.shape) == [32, 256] and list(sigmas.shape) == [32, 256],
        "gmm_sigma_min": float(sigmas.min()),
        "gmm_sigma_max": float(sigmas.max()),
        "gmm_prior_effective_components": prior_effective,
        "gmm_source_images": int(gmm.get("source_images", -1)),
        "gmm_descriptor_count": int(gmm.get("descriptor_count", -1)),
        "gmm_converged": bool(gmm.get("converged", False)),
        "gmm_iterations": int(gmm.get("n_iter", -1)),
        "descriptor_count": int(descriptors.shape[0]),
        "descriptor_std": float(descriptors.std()),
        "descriptor_norm_mean": float(descriptors.norm(dim=1).mean()),
        "patch_count_min": min(patch_counts),
        "patch_count_max": max(patch_counts),
        "gamma_usage_effective_components": float(usage_effective),
        "gamma_active_components_1e-3": int((usage > 1e-3).sum()),
        "gamma_max_mean": float(gamma.max(dim=1).values.mean()),
        "gamma_entropy_mean": float(assignment_entropy.mean()),
        "fv_finite": fv_finite,
    }


def main() -> None:
    args = parse_args()
    report = audit_static_config()
    supplied_artifact_args = [args.data_root, args.stage1_checkpoint, args.gmm]
    if any(value is not None for value in supplied_artifact_args):
        report.update(audit_artifacts(args))

    failures = []
    if not report["stage1_config_ok"] or not report["stage2_config_ok"]:
        failures.append("static configuration mismatch")
    if "stage1_head_tensors" in report:
        if report["stage1_head_tensors"] != 6:
            failures.append("incomplete Stage 1 head transfer")
        if report["stage1_global_step"] != 9000:
            failures.append("Stage 1 did not finish 9000 updates")
        if not report["gmm_shape_ok"] or report["gmm_sigma_min"] <= 0:
            failures.append("invalid GMM parameters")
        if not report["gmm_converged"]:
            failures.append("GMM did not converge")
        if report["descriptor_std"] <= 1e-6 or not report["fv_finite"]:
            failures.append("degenerate descriptor/FV values")
        if 0 <= report["gmm_source_images"] < 500:
            failures.append("GMM descriptors cover fewer than 500 trainval images")
        if report["gamma_active_components_1e-3"] < 4:
            failures.append("Fisher assignment uses fewer than four components")

    report["passed"] = not failures
    report["failures"] = failures
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.strict and failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
