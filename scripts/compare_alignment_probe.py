"""
compare_alignment_probe.py — Compare Caffe vs PyTorch intermediate features.

After running:
  1. export_alignment_probe.py  (locally) → pytorch_*.npy
  2. export_caffe_blobs.py      (on AutoDL) → caffe_*.npy

This script loads both, computes per-blob cosine similarity and relative
MSE, and prints a summary table.

Usage:
  python scripts/compare_alignment_probe.py \
      --caffe-dir debug_align/caffe_res101_000001_scale480 \
      --pytorch-dir debug_align/pytorch_b1_000001_scale480
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (flattened)."""
    a_f = a.ravel()
    b_f = b.ravel()
    if a_f.size != b_f.size:
        return -2.0  # size mismatch
    dot = np.dot(a_f, b_f)
    norm = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if norm < 1e-12:
        return 0.0
    return float(dot / norm)


def relative_mse(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b||^2 / ||a||^2"""
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    if a_f.size != b_f.size:
        return -1.0
    num = np.sum((a_f - b_f) ** 2)
    den = np.sum(a_f ** 2)
    if den < 1e-12:
        return 0.0
    return float(num / den)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Caffe vs PyTorch alignment blobs.")
    parser.add_argument("--caffe-dir", type=str, required=True)
    parser.add_argument("--pytorch-dir", type=str, required=True)
    parser.add_argument("--tol", type=float, default=0.95,
                        help="Cosine sim threshold for PASS (default: 0.95)")
    args = parser.parse_args()

    caffe_dir = Path(args.caffe_dir)
    pytorch_dir = Path(args.pytorch_dir)

    # Layer mapping: Caffe blob name → PyTorch blob name
    # Key mapping — adjust as needed based on actual naming
    layer_pairs = [
        # (caffe_name, pytorch_name, description)
        ("conv1", "conv1", "conv1 (7×7)"),
        ("pool1", "maxpool", "pool1 (3×3 max)"),
        ("res2a_relu", None, None),       # Caffe-specific, skip for now
        ("res3c_relu", "layer1", "layer1 (res3c)"),
        ("res4f_relu", "layer2", "layer2 (res4f)"),
        ("res5c_relu", "layer3", "layer3 (res5c) — backbone final"),
        ("res5c_pca", None, None),
        ("res5c_pca_l2", None, None),
        ("fisher_sum", None, None),
    ]

    print(f"{'Layer':<22} {'Caffe shape':<20} {'PyTorch shape':<20} {'CosSim':<10} {'RelMSE':<10} {'Status'}")
    print("-" * 90)

    statuses = []
    for caffe_name, pytorch_name, desc in layer_pairs:
        caffe_path = caffe_dir / f"caffe_{caffe_name}.npy"
        if pytorch_name is not None:
            pytorch_path = pytorch_dir / f"pytorch_{pytorch_name}.npy"
        else:
            pytorch_path = None

        if not caffe_path.exists():
            print(f"{caffe_name:<22} { '<missing>':<20} {'—':<20} {'—':<10} {'—':<10} {'SKIP (no caffe)'}")
            continue

        c_arr = np.load(str(caffe_path))

        if pytorch_path is not None and pytorch_path.exists():
            p_arr = np.load(str(pytorch_path))

            # Try to match shapes — may need transpose/reshape
            if c_arr.shape != p_arr.shape:
                # Caffe is (N, C, H, W), PyTorch is same — might be close but not identical
                sim = -3.0  # shape mismatch
                rmse = -3.0
                status = f"SHAPE ({c_arr.shape} vs {p_arr.shape})"
            else:
                sim = cosine_sim(c_arr, p_arr)
                rmse = relative_mse(c_arr, p_arr)
                status = "PASS" if sim >= args.tol else f"LOW ({sim:.4f})"
                statuses.append(sim >= args.tol)

            c_str = "×".join(str(s) for s in c_arr.shape)
            p_str = "×".join(str(s) for s in p_arr.shape)
            print(f"{caffe_name:<22} {c_str:<20} {p_str:<20} {sim:<10.4f} {rmse:<10.6f} {status}")
        else:
            # Caffe-only blob (e.g. Fisher layers), print info
            c_str = "×".join(str(s) for s in c_arr.shape)
            label = desc or caffe_name
            print(f"{caffe_name:<22} {c_str:<20} {'—':<20} {'—':<10} {'—':<10} {'CAFFE_ONLY'}")

    print("-" * 90)
    if statuses:
        n_pass = sum(statuses)
        n_total = len(statuses)
        print(f"\nBackbone alignment: {n_pass}/{n_total} layers passed (cos ≥ {args.tol})")
        if n_pass == n_total:
            print("✓ Backbone is aligned! Fisher layer comparison can proceed.")
        else:
            print("✗ Backbone misalignment detected. Investigate preprocessing.")
    else:
        print("No alignable layers found. Check blob names and paths.")


if __name__ == "__main__":
    main()
