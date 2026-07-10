"""
export_alignment_probe.py — prepare a single-image probe for Caffe <> PyTorch alignment.

Steps:
  1. Load VOC 2007 image 000001.jpg
  2. Caffe-style preprocessing (BGR, resize 224×224, mean subtraction)
  3. Save input_tensor.npy  (for Caffe forward)
  4. Run through PyTorch ResNet-101 backbone → save res5c features
  5. Save metadata (image shape, preprocessing params)

Usage:
  python scripts/export_alignment_probe.py \
      --image-path data/VOCdevkit/VOC2007/JPEGImages/000001.jpg \
      --out-dir debug_align/pytorch_b1_000001_scale480
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights


def caffe_preprocess(image: Image.Image, size: int = 224) -> np.ndarray:
    """Resize → centre-crop (if needed) → BGR → mean subtraction → CHW float32.

    This is the standard Caffe ImageNet preprocessing used by the FisherNet
    ResNet-101 test_image.prototxt (input: 1×3×224×224).
    """
    # Resize shorter side to `size`, then centre-crop `size × size`
    w, h = image.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - size) // 2
    top = (new_h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    # Numpy: HWC uint8 RGB
    img = np.asarray(image, dtype=np.float32)

    # RGB → BGR
    img = img[:, :, ::-1]

    # Mean subtraction (Caffe BGR order)
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img -= mean

    # HWC → CHW
    img = img.transpose(2, 0, 1)

    # Add batch dim → 1×3×224×224
    img = img[np.newaxis, ...]

    # Force contiguous layout (negative strides from transpose cause issues)
    img = np.ascontiguousarray(img)

    return img


@torch.no_grad()
def export_pytorch_resnet_backbone(
    input_tensor: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run the preprocessed input through a PyTorch ResNet-101 and dump layer outputs.

    We use a ResNet-101 that matches the Caffe ResNet-101 layer names / structure
    as closely as possible with torchvision's resnet101.

    Returns a dict of blob_name → numpy array.
    """
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).to(device)
    for layer_name in ("layer2", "layer3", "layer4"):
        block = getattr(model, layer_name)[0]
        block.conv1.stride = (2, 2)
        block.conv2.stride = (1, 1)
    model.eval()

    # Register forward hooks to capture intermediate features.
    # Map torchvision module paths → Caffe-style blob names.
    hook_targets: dict[str, str] = {
        "conv1": "conv1",           # conv1 (7×7, stride 2)
        "bn1": "bn_conv1",
        "relu": "conv1_relu",
        "maxpool": "pool1",
        "layer1.2.relu": "res3c_relu",  # last relu of layer1 = res3c (in Caffe ResNet naming)
        "layer2.3.relu": "res4f_relu",
        "layer3.22.relu": "res5c_relu",  # last relu of layer3 = res5c
        "layer4.2.relu": "res5c_relu",  # no - actually layer4 is res6. Let's track res5c properly.
    }

    # More reliable: use a simple sequential approach
    blobs: dict[str, np.ndarray] = {}
    tensor = torch.from_numpy(input_tensor).to(device)

    def _hook(name: str):
        def _fn(module, inp, out):
            blobs[name] = out.detach().cpu().numpy()
        return _fn

    # Register hooks
    handles = []
    # Manually trace through ResNet-101
    modules = {
        "conv1": model.conv1,
        "bn1": model.bn1,
        "relu": model.relu,
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
    }
    for caffe_name, mod in modules.items():
        handles.append(mod.register_forward_hook(_hook(caffe_name)))

    # Forward
    x = model.conv1(tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    blobs["maxpool"] = x.detach().cpu().numpy()
    x = model.layer1(x)
    blobs["layer1"] = x.detach().cpu().numpy()
    x = model.layer2(x)
    blobs["layer2"] = x.detach().cpu().numpy()
    x = model.layer3(x)
    blobs["layer3"] = x.detach().cpu().numpy()
    x = model.layer4(x)
    blobs["layer4"] = x.detach().cpu().numpy()

    for h in handles:
        h.remove()

    # Now let's find the Caffe-equivalent res5c layer.
    # In Caffe ResNet-101: res5c = the last bottleneck block of conv5_x (layer3 in torchvision)
    # Torchvision layer3 has 23 bottleneck blocks.
    # The last block's 3rd conv output = res5c (Caffe naming: res5c = last element-wise sum before relu)
    # But actually we want the output of layer3 (after all 23 blocks)
    # Actually, let's just use the relu output AFTER the last block of layer3

    # Get the exact last relu output from layer3
    # In torchvision, each bottleneck's output goes through relu.
    # layer3[-1].relu gives us the final activation.
    # But the torchvision bottleneck structure is: conv1 → bn1 → relu → conv2 → bn2 → relu → conv3 → bn3 → (downsample) → eltwise → relu
    # The final output of layer3[22] (after the final relu) = res5c_relu in Caffe.
    # We already captured "layer3" from the hook, but let's extract res5c_relu specifically.

    blobs["res5c_relu"] = blobs["layer4"]

    return blobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export alignment probe for Caffe ↔ PyTorch.")
    parser.add_argument("--image-path", type=str,
                        default="data/VOCdevkit/VOC2007/JPEGImages/000001.jpg")
    parser.add_argument("--out-dir", type=str,
                        default="debug_align/pytorch_b1_000001_scale480")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[probe] Using device: {device}")

    # 1. Load image
    image = Image.open(args.image_path).convert("RGB")
    print(f"[probe] Loaded {args.image_path}: {image.size}")

    # 2. Caffe-style preprocessing
    input_tensor = caffe_preprocess(image, size=args.size)
    print(f"[probe] Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"[probe] Range: [{input_tensor.min():.2f}, {input_tensor.max():.2f}]")

    # Save input_tensor.npy  (CHW, float32, batch=1)
    np.save(str(out_dir / "input_tensor.npy"), input_tensor)
    print(f"[probe] Saved {out_dir / 'input_tensor.npy'}")

    # 3. Save metadata
    metadata = {
        "source_image": str(args.image_path),
        "original_size": image.size,
        "preprocess": "caffe_resize224_center_crop224_bgr_mean[103.939,116.779,123.68]",
        "tensor_shape": list(input_tensor.shape),
        "tensor_dtype": str(input_tensor.dtype),
        "tensor_min": float(input_tensor.min()),
        "tensor_max": float(input_tensor.max()),
    }
    with open(str(out_dir / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[probe] Saved {out_dir / 'metadata.json'}")

    # 4. PyTorch backbone export
    print("[probe] Running PyTorch ResNet-101 forward...")
    blobs = export_pytorch_resnet_backbone(input_tensor, device)
    for name, arr in blobs.items():
        save_path = out_dir / f"pytorch_{name}.npy"
        np.save(str(save_path), arr)
        print(f"  -> {save_path.name}: {arr.shape} [{arr.min():.4f}, {arr.max():.4f}]")

    print("[probe] Done.")


if __name__ == "__main__":
    main()
