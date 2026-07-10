"""
Compare Caffe and PyTorch intermediate blobs for the ResNet-101 alignment probe.

Example:
  python scripts/compare_alignment_probe.py \
    --pytorch-dir debug_align/pytorch_res101_caffe_v1_000001 \
    --caffe-dir debug_align/caffe_res101_000001_scale480 \
    --out debug_align/caffe_pytorch_backbone_compare.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64, copy=False)
    b_f = b.ravel().astype(np.float64, copy=False)
    if a_f.size != b_f.size:
        return -2.0
    norm = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if norm < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / norm)


def relative_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.ravel().astype(np.float64, copy=False)
    cand = candidate.ravel().astype(np.float64, copy=False)
    if ref.size != cand.size:
        return -1.0
    den = np.sum(ref * ref)
    if den < 1e-12:
        return 0.0
    diff = cand - ref
    return float(np.sum(diff * diff) / den)


def compare_arrays(caffe_arr: np.ndarray, pytorch_arr: np.ndarray) -> dict[str, float]:
    caffe64 = caffe_arr.astype(np.float64, copy=False)
    pytorch64 = pytorch_arr.astype(np.float64, copy=False)
    diff = pytorch64 - caffe64
    return {
        "cosine": cosine_sim(caffe64, pytorch64),
        "relative_mse": relative_mse(caffe64, pytorch64),
        "rel_l2": float(np.linalg.norm(diff.ravel()) / (np.linalg.norm(caffe64.ravel()) + 1e-12)),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Caffe vs PyTorch alignment blobs.")
    parser.add_argument("--caffe-dir", type=Path, required=True)
    parser.add_argument("--pytorch-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tol", type=float, default=0.95)
    args = parser.parse_args()

    layer_pairs = [
        ("data", "input_tensor", "input"),
        ("conv1", "relu", "conv1_relu"),
        ("pool1", "maxpool", "pool1"),
        ("res2c", "layer1", "res2c/layer1"),
        ("res3b3", "layer2", "res3b3/layer2"),
        ("res4b22", "layer3", "res4b22/layer3"),
        ("res5c", "layer4", "res5c/layer4"),
        ("res5c_pca", "pca", "res5c_pca"),
        ("res5c_pca_l2", "pca_l2", "res5c_pca_l2"),
        ("fisher1", "fisher1", "fisher1/y"),
        ("fisher2", "fisher2", "fisher2/y_squared"),
        ("fisher3", "fisher3", "fisher3/dist_sum"),
        ("fisher4_new", "fisher4_new_logits", "fisher4_new/logits"),
        ("fisher_gamma", "fisher_gamma", "fisher_gamma"),
        ("fisher6", "fisher6", "fisher6/first_order"),
        ("fisher7", "fisher7", "fisher7/second_order"),
        ("fisher_sum", "fisher_sum", "fisher_sum"),
    ]

    print(
        f"{'Layer':<18} {'Caffe shape':<18} {'PyTorch shape':<18} "
        f"{'CosSim':<12} {'RelL2':<12} {'MeanAbs':<12} {'Status'}"
    )
    print("-" * 105)

    rows: list[dict[str, object]] = []
    comparable_statuses: list[bool] = []
    for caffe_name, pytorch_name, label in layer_pairs:
        caffe_path = args.caffe_dir / f"caffe_{caffe_name}.npy"
        if pytorch_name == "input_tensor":
            pytorch_path = args.pytorch_dir / "input_tensor.npy"
        else:
            pytorch_path = args.pytorch_dir / f"pytorch_{pytorch_name}.npy"
        row: dict[str, object] = {
            "layer": label,
            "caffe": caffe_name,
            "pytorch": pytorch_name,
        }

        if not caffe_path.exists():
            row["status"] = "SKIP_NO_CAFFE"
            rows.append(row)
            print(f"{label:<18} {'<missing>':<18} {'-':<18} {'-':<12} {'-':<12} {'-':<12} {row['status']}")
            continue

        caffe_arr = np.load(caffe_path)
        row["caffe_shape"] = "x".join(str(s) for s in caffe_arr.shape)

        if not pytorch_path.exists():
            row["status"] = "CAFFE_ONLY"
            rows.append(row)
            print(
                f"{label:<18} {row['caffe_shape']:<18} {'-':<18} "
                f"{'-':<12} {'-':<12} {'-':<12} {row['status']}"
            )
            continue

        pytorch_arr = np.load(pytorch_path)
        row["pytorch_shape"] = "x".join(str(s) for s in pytorch_arr.shape)

        if caffe_arr.shape != pytorch_arr.shape:
            row["status"] = "SHAPE_MISMATCH"
            rows.append(row)
            print(
                f"{label:<18} {row['caffe_shape']:<18} {row['pytorch_shape']:<18} "
                f"{'-':<12} {'-':<12} {'-':<12} {row['status']}"
            )
            continue

        metrics = compare_arrays(caffe_arr, pytorch_arr)
        status = "PASS" if metrics["cosine"] >= args.tol else f"LOW_{metrics['cosine']:.4f}"
        row.update(metrics)
        row["status"] = status
        rows.append(row)
        comparable_statuses.append(status == "PASS")
        print(
            f"{label:<18} {row['caffe_shape']:<18} {row['pytorch_shape']:<18} "
            f"{metrics['cosine']:<12.9f} {metrics['rel_l2']:<12.6g} "
            f"{metrics['mean_abs']:<12.6g} {status}"
        )

    print("-" * 105)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.out}")

    if comparable_statuses:
        passed = sum(comparable_statuses)
        total = len(comparable_statuses)
        print(f"\nAlignment: {passed}/{total} comparable layers passed (cos >= {args.tol})")
        if passed == total:
            print("PASS: comparable layers are aligned. Fisher/PCA comparison can proceed if those blobs exist.")
        else:
            print("FAIL: inspect the first low layer.")
    else:
        print("No comparable layers found. Check blob names and paths.")


if __name__ == "__main__":
    main()
