#!/usr/bin/env bash
# Deep FisherNet — Remote GPU setup script
# Run this on the rented GPU machine (AutoDL / Featurize / etc.)
# Usage: bash scripts/setup_remote.sh [--vgg16] [--res101]
# Default: prepare everything for both presets

set -euo pipefail

REPO_URL="https://github.com/Mar-Ding/Deep-FisherNet-PyTorch.git"
PROJECT_DIR="Deep-FisherNet-PyTorch"
VOC_YEAR="2007"

echo "============================================"
echo "  Deep FisherNet — Remote GPU Setup"
echo "============================================"
echo ""

# ── 1. Clone repo ──────────────────────────────────
if [ ! -d "$PROJECT_DIR" ]; then
    echo "[1/6] Cloning repository..."
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
else
    echo "[1/6] Repository already exists, updating..."
    cd "$PROJECT_DIR"
    git pull
fi

# ── 2. Check GPU / PyTorch ─────────────────────────
echo "[2/6] Checking CUDA & PyTorch..."
python3 -c "
import torch, sys
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('  WARNING: CUDA not available!')
    sys.exit(1)
"

# ── 3. Install dependencies ─────────────────────────
echo "[3/6] Installing Python dependencies..."
pip install -r requirements.txt

# ── 4. Download VOC 2007 ────────────────────────────
echo "[4/6] Setting up PASCAL VOC 2007..."
python3 -c "
from pathlib import Path
from torchvision.datasets import VOCDetection
data_root = Path('data')
print(f'  Downloading to {data_root.resolve()}...')
ds = VOCDetection(root=str(data_root), year='2007', image_set='trainval', download=True)
print(f'  trainval: {len(ds)} images')
ds = VOCDetection(root=str(data_root), year='2007', image_set='test', download=True)
print(f'  test:     {len(ds)} images')
"

# ── 5. GMM Initialization ───────────────────────────
do_vgg16=false
do_res101=false
for arg in "$@"; do
    case $arg in
        --vgg16) do_vgg16=true ;;
        --res101) do_res101=true ;;
        --all) do_vgg16=true; do_res101=true ;;
    esac
done
if [ "$#" -eq 0 ]; then
    do_vgg16=true
    do_res101=true
fi

if [ "$do_vgg16" = true ]; then
    echo "[5a/6] GMM initialization for VGG16 (paper-like)..."
    python3 scripts/init_gmm.py \
        --preset vgg16-paper-like \
        --data-root data \
        --output outputs/vgg16-paper-like/fisher_gmm.pt \
        --pretrained --num-workers 4
fi

if [ "$do_res101" = true ]; then
    echo "[5b/6] GMM initialization for ResNet101 (official-like)..."
    python3 scripts/init_gmm.py \
        --preset official-res101-like \
        --data-root data \
        --output outputs/official_res101_like/fisher_gmm.pt \
        --pretrained --num-workers 4
fi

# ── 6. Quick probe (1 epoch, limited batches) ──────
echo "[6/6] Running quick probe (1 epoch, 8 batches)..."
if [ "$do_vgg16" = true ]; then
    echo "  VGG16 probe..."
    python3 scripts/train.py \
        --preset vgg16-paper-like \
        --data-root data \
        --output-dir outputs/vgg16_probe \
        --fisher-init outputs/vgg16-paper-like/fisher_gmm.pt \
        --pretrained --epochs 1 \
        --max-train-batches 8 --max-val-batches 4 \
        --num-workers 4 --no-progress
fi

if [ "$do_res101" = true ]; then
    echo "  ResNet101 probe..."
    python3 scripts/train.py \
        --preset official-res101-like \
        --data-root data \
        --output-dir outputs/res101_probe \
        --fisher-init outputs/official_res101_like/fisher_gmm.pt \
        --pretrained --epochs 1 \
        --max-train-batches 8 --max-val-batches 4 \
        --num-workers 4 --no-progress
fi

echo ""
echo "============================================"
echo "  Setup complete! Ready for training."
echo "============================================"
echo ""
echo "Quick links:"
echo "  VGG16 full train:   bash scripts/run_vgg16_full.sh"
echo "  ResNet101 train:    bash scripts/run_res101_full.sh"
echo "  Monitor:            watch -n 2 nvidia-smi"
echo ""
