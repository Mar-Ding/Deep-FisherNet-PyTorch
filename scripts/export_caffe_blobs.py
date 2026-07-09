"""
export_caffe_blobs.py — Forward through Caffe FisherNet and dump all blobs for alignment.

Two modes:
  A. Full FisherNet prototxt  — requires a caffemodel with ALL weights (incl. Fisher/PCA)
  B. Backbone-only prototxt   — requires only standard ResNet-101 caffemodel
     (auto-fallback if mode A fails)

Usage:
  python scripts/export_caffe_blobs.py \
      --caffe-python /path/to/caffe/python \
      --prototxt models/Res-101/test_image.prototxt \
      --caffemodel /path/to/model.caffemodel \
      --input-npy debug_align/pytorch_b1_000001_scale480/input_tensor.npy \
      --out-dir debug_align/caffe_res101_000001_scale480 \
      --gpu 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def try_load_net(caffe, prototxt: str, caffemodel: str, phase: int) -> tuple:
    """Try loading a caffemodel; return (net, True) or (None, False)."""
    try:
        net = caffe.Net(prototxt, caffemodel, phase)
        return net, True
    except Exception as e:
        return None, False


def export_blobs(net, out_dir: Path, prefix: str = "caffe") -> list[str]:
    """Dump all blob outputs as .npy files. Return list of saved blob names."""
    saved = []
    for blob_name, blob in net.blobs.items():
        arr = blob.data.copy()
        save_path = out_dir / f"{prefix}_{blob_name}.npy"
        np.save(str(save_path), arr)
        saved.append(blob_name)
        print(f"  blob {blob_name}: {arr.shape} [{arr.min():.4f}, {arr.max():.4f}] → {save_path.name}")
    return saved


def export_params(net, out_dir: Path, param_names: list[str], prefix: str = "caffe") -> None:
    """Dump selected parameter blobs."""
    for param_name in param_names:
        if param_name not in net.params:
            continue
        for i, p in enumerate(net.params[param_name]):
            arr = p.data.copy()
            save_path = out_dir / f"{prefix}_param_{param_name}_{i}.npy"
            np.save(str(save_path), arr)
            print(f"  param {param_name}[{i}]: {arr.shape} → {save_path.name}")


def main() -> None:
    args = parse_args()

    # Add Caffe Python path
    sys.path.insert(0, args.caffe_python)

    try:
        import caffe
        print(f"[caffe] OK: {caffe.__file__}")
    except ImportError as e:
        print(f"[caffe] FAIL: {e}")
        sys.exit(1)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    print(f"[caffe] GPU mode, device={args.gpu}")

    # Load input tensor
    input_tensor = np.load(args.input_npy)
    print(f"[caffe] Loaded input: {input_tensor.shape} dtype={input_tensor.dtype}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Try mode A: full FisherNet prototxt ===
    print(f"[caffe] Attempt 1: full FisherNet prototxt ({args.prototxt})")
    net, ok = try_load_net(caffe, args.prototxt, args.caffemodel, caffe.TEST)

    if not ok:
        # === Fallback: backbone-only prototxt ===
        backbone_prototxt = str(
            Path(args.prototxt).parent / "test_backbone.prototxt"
        )
        print(f"[caffe] Full prototxt failed. Attempt 2: backbone-only ({backbone_prototxt})")
        net, ok = try_load_net(caffe, backbone_prototxt, args.caffemodel, caffe.TEST)

    if not ok:
        print(f"[caffe] ERROR: Could not load caffemodel with either prototxt.")
        print(f"[caffe]   caffemodel: {args.caffemodel}")
        print(f"[caffe]   The model might be incompatible (different architecture).")
        sys.exit(1)

    print(f"[caffe] Loaded net. Blobs: {list(net.blobs.keys())}")

    # Set input
    net.blobs["data"].reshape(*input_tensor.shape)
    net.blobs["data"].data[...] = input_tensor

    # Forward
    output = net.forward()
    print(f"[caffe] Forward done.")

    # Dump all blobs
    saved = export_blobs(net, out_dir, prefix="caffe")
    print(f"[caffe] Exported {len(saved)} blobs.")

    # Dump Fisher/PCA params if they exist
    param_targets = ["res5c_pca", "fisher_weight", "fisher1_w", "fisher1_b"]
    export_params(net, out_dir, param_targets, prefix="caffe")

    # Save blob manifest
    with open(str(out_dir / "blob_names.txt"), "w") as f:
        for name in net.blobs.keys():
            shape = net.blobs[name].data.shape
            f.write(f"{name} {'×'.join(str(s) for s in shape)}\n")

    # Save metadata
    is_fisher = any("fisher" in b or "pca" in b for b in saved)
    with open(str(out_dir / "export_info.txt"), "w") as f:
        f.write(f"caffemodel: {args.caffemodel}\n")
        f.write(f"prototxt: {args.prototxt}\n")
        f.write(f"has_fisher_layers: {is_fisher}\n")
        f.write(f"blobs_exported: {len(saved)}\n")

    print(f"[caffe] Done. All blobs → {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Caffe blobs for alignment.")
    parser.add_argument("--caffe-python", type=str, required=True)
    parser.add_argument("--prototxt", type=str, required=True)
    parser.add_argument("--caffemodel", type=str, required=True)
    parser.add_argument("--input-npy", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
