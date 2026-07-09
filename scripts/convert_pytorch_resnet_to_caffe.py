"""
convert_pytorch_resnet_to_caffe.py

Convert PyTorch ResNet-101 pretrained weights to Caffe .caffemodel format.
This creates a caffemodel compatible with the FisherNet test_backbone.prototxt.

Usage in the Caffe build environment:
  python scripts/convert_pytorch_resnet_to_caffe.py \
      --caffe-python external/official-fishernet/caffe-fishernet-conv/python \
      --prototxt external/official-fishernet/models/Res-101/test_backbone.prototxt \
      --out external/official-fishernet/models/Res-101/ResNet-101-model.caffemodel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision.models as models


def bottleneck_caffe_names(layer_idx: int, block_idx: int) -> dict:
    """
    Return Caffe layer names for a bottleneck block.
    
    layer_idx: 0=layer1(conv2_x), 1=layer2(conv3_x), 2=layer3(conv4_x), 3=layer4(conv5_x)
    block_idx: 0, 1, 2, ... within each layer
    
    Returns a dict: pytorch_key -> (caffe_conv_name, caffe_bn_name, caffe_scale_name)
    """
    # Caffe prefix: res2, res3, res4, res5
    stem = f"res{2 + layer_idx}"
    # Block letter: a, b, c, ...
    letter = chr(97 + block_idx)
    
    names = {}
    
    # Main branch
    names["conv1"] = f"{stem}_{letter}_branch2a"
    names["conv2"] = f"{stem}_{letter}_branch2b"
    names["conv3"] = f"{stem}_{letter}_branch2c"
    
    # BatchNorm and Scale for each conv
    for conv_name, caffe_conv in names.items():
        caffe_bn = caffe_conv.replace(f"{stem}_{letter}_branch", f"bn{2+layer_idx}_{letter}_branch")
        caffe_scale = caffe_conv.replace(f"{stem}_{letter}_branch", f"scale{2+layer_idx}_{letter}_branch")
        names[f"bn_{conv_name}"] = caffe_bn
        names[f"scale_{conv_name}"] = caffe_scale
    
    # Downsample (identity shortcut)
    names["downsample_conv"] = f"{stem}_{letter}_branch1"
    names["downsample_bn"] = f"bn{2+layer_idx}_{letter}_branch1"
    names["downsample_scale"] = f"scale{2+layer_idx}_{letter}_branch1"
    
    return names


def convert_weights(pt_model: torch.nn.Module, caffe_net) -> int:
    """Transfer PyTorch weights into Caffe net weights. Returns count of layers set."""
    count = 0
    
    # === conv1 ===
    set_param(caffe_net, "conv1", 0, pt_model.conv1.weight)
    count += 1
    
    # === bn_conv1 (BatchNorm) + scale_conv1 ===
    set_bn_scale(caffe_net, "bn_conv1", "scale_conv1",
                 pt_model.bn1.running_mean, pt_model.bn1.running_var,
                 pt_model.bn1.weight, pt_model.bn1.bias)
    count += 3  # BN+Scale counts as 3 param blobs
    
    # === Layers: layer1 → res2, layer2 → res3, layer3 → res4, layer4 → res5 ===
    layers = [pt_model.layer1, pt_model.layer2, pt_model.layer3, pt_model.layer4]
    
    for layer_idx, layer in enumerate(layers):
        for block_idx, block in enumerate(layer):
            caffe_names = bottleneck_caffe_names(layer_idx, block_idx)
            
            # Main branch convs
            for conv_idx, conv_name in enumerate(["conv1", "conv2", "conv3"]):
                pt_conv = getattr(block, conv_name)
                caffe_conv = caffe_names[conv_name]
                set_param(caffe_net, caffe_conv, 0, pt_conv.weight)
                # Bias is included in Scale layer (Caffe Conv doesn't have bias when followed by BN+Scale)
                count += 1
                
                # BN + Scale
                pt_bn = getattr(block, f"bn{conv_idx+1}")
                caffe_bn = caffe_names[f"bn_{conv_name}"]
                caffe_scale = caffe_names[f"scale_{conv_name}"]
                set_bn_scale(caffe_net, caffe_bn, caffe_scale,
                             pt_bn.running_mean, pt_bn.running_var,
                             pt_bn.weight, pt_bn.bias)
                count += 3
            
            # Downsample (if exists)
            if block.downsample is not None:
                pt_down_conv = block.downsample[0]
                pt_down_bn = block.downsample[1]
                
                caffe_conv = caffe_names["downsample_conv"]
                caffe_bn = caffe_names["downsample_bn"]
                caffe_scale = caffe_names["downsample_scale"]
                
                set_param(caffe_net, caffe_conv, 0, pt_down_conv.weight)
                set_bn_scale(caffe_net, caffe_bn, caffe_scale,
                             pt_down_bn.running_mean, pt_down_bn.running_var,
                             pt_down_bn.weight, pt_down_bn.bias)
                count += 4
    
    return count


def set_param(net, layer_name: str, blob_idx: int, pt_tensor: torch.Tensor) -> None:
    """Set a single Caffe param blob from a PyTorch tensor."""
    if layer_name not in net.params:
        print(f"  WARNING: layer '{layer_name}' not in net.params, skipping")
        return
    if blob_idx >= len(net.params[layer_name]):
        print(f"  WARNING: layer '{layer_name}' has no blob[{blob_idx}], skipping")
        return
    
    arr = pt_tensor.detach().cpu().numpy()
    caffe_shape = net.params[layer_name][blob_idx].data.shape
    
    if arr.shape != caffe_shape:
        print(f"  WARNING: shape mismatch for {layer_name}[{blob_idx}]: "
              f"PyTorch {arr.shape} vs Caffe {caffe_shape}. Reshaping.")
        arr = arr.reshape(caffe_shape)
    
    net.params[layer_name][blob_idx].data[...] = arr


def set_bn_scale(net, bn_name: str, scale_name: str,
                 running_mean, running_var, gamma, beta) -> None:
    """
    Set Caffe BatchNorm (3 blobs) + Scale (2 blobs) from PyTorch BN params.
    
    Caffe BatchNorm blobs:
      [0] = running_mean
      [1] = running_var
      [2] = scale_factor (usually 1.0)
    
    Caffe Scale blobs:
      [0] = gamma (weight)
      [1] = beta (bias)
    """
    # BatchNorm
    if bn_name in net.params:
        mean_arr = running_mean.detach().cpu().numpy()
        var_arr = running_var.detach().cpu().numpy()
        net.params[bn_name][0].data[...] = mean_arr
        net.params[bn_name][1].data[...] = var_arr
        net.params[bn_name][2].data[...] = 1.0
    
    # Scale
    if scale_name in net.params:
        gamma_arr = gamma.detach().cpu().numpy()
        beta_arr = beta.detach().cpu().numpy()
        net.params[scale_name][0].data[...] = gamma_arr
        net.params[scale_name][1].data[...] = beta_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch ResNet-101 to Caffe caffemodel.")
    parser.add_argument("--caffe-python", type=str, required=True)
    parser.add_argument("--prototxt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    # Add Caffe Python
    sys.path.insert(0, args.caffe_python)
    import caffe
    
    # Load PyTorch model
    print("[convert] Loading PyTorch ResNet-101 pretrained...")
    pt_model = models.resnet101(pretrained=True)
    pt_model.eval()
    print(f"[convert] PyTorch model loaded: {sum(p.numel() for p in pt_model.parameters()):,} params")
    
    # Create Caffe net from prototxt (random init)
    print(f"[convert] Creating Caffe net from {args.prototxt}")
    caffe.set_mode_cpu()
    net = caffe.Net(str(args.prototxt), caffe.TEST)
    print(f"[convert] Caffe net created. Layers with params: {list(net.params.keys())}")
    
    # Transfer weights
    print("[convert] Transferring weights...")
    count = convert_weights(pt_model, net)
    print(f"[convert] Transferred {count} param groups.")
    
    # Save
    out_path = str(args.out)
    net.save(out_path)
    import os
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[convert] Saved: {out_path} ({size_mb:.1f} MB)")
    print("[convert] Done.")


if __name__ == "__main__":
    main()
