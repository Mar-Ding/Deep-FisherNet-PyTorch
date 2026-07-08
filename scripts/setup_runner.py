#!/usr/bin/env python3
"""Setup: download VOC2007, init GMM, run probe."""
import os, subprocess, sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Deep-FisherNet-PyTorch")
os.chdir(str(ROOT))

def run(cmd, desc):
    print(f"\n=== {desc} ===")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[-500:])
    if r.returncode != 0:
        print(f"FAILED (exit={r.returncode})")
        sys.exit(r.returncode)
    return r

# 1. Download VOC2007
print("=== Downloading VOC2007 ===")
from torchvision.datasets import VOCDetection
ds = VOCDetection(root="data", year="2007", image_set="trainval", download=True)
print(f"trainval: {len(ds)} images")
ds = VOCDetection(root="data", year="2007", image_set="test", download=True)
print(f"test: {len(ds)} images")

# 2. GMM init for ResNet101
run(
    "python scripts/init_gmm.py --preset official-res101-like --data-root data "
    "--output outputs/official_res101_like/fisher_gmm.pt --pretrained --num-workers 4",
    "GMM init: ResNet101 official-like"
)

# 3. GMM init for VGG16
run(
    "python scripts/init_gmm.py --preset vgg16-paper-like --data-root data "
    "--output outputs/vgg16-paper-like/fisher_gmm.pt --pretrained --num-workers 4",
    "GMM init: VGG16 paper-like"
)

# 4. Probe: ResNet101 (1 epoch, 8 batches)
run(
    "python scripts/train.py --preset official-res101-like --data-root data "
    "--output-dir outputs/res101_probe "
    "--fisher-init outputs/official_res101_like/fisher_gmm.pt "
    "--pretrained --epochs 1 --max-train-batches 8 --max-val-batches 4 "
    "--num-workers 4 --no-progress",
    "Probe: ResNet101 (1 epoch, 8 batches)"
)

# 5. Probe: VGG16 (1 epoch, 8 batches)
run(
    "python scripts/train.py --preset vgg16-paper-like --data-root data "
    "--output-dir outputs/vgg16_probe "
    "--fisher-init outputs/vgg16-paper-like/fisher_gmm.pt "
    "--pretrained --epochs 1 --max-train-batches 8 --max-val-batches 4 "
    "--num-workers 4 --no-progress",
    "Probe: VGG16 (1 epoch, 8 batches)"
)

print("\n✅ Setup complete! Ready for full training.")
