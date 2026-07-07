# Deep FisherNet for Object Classification

PyTorch reproduction of **Deep FisherNet for Object Classification** (NIPS 2016) on PASCAL VOC 2007.

Original paper: [arXiv:1608.00182](https://arxiv.org/abs/1608.00182)
Official Caffe code: [ppengtang/fishernet](https://github.com/ppengtang/fishernet)

## Presets

| Preset | Backbone | FV Dim | Epochs | Optimizer | Description |
|--------|----------|--------|--------|-----------|-------------|
| `smoke` | AlexNet | 1024 | 1 | AdamW | Quick smoke test |
| `alexnet-paper-like` | AlexNet | 16384 | 10 | SGD | Paper-like AlexNet |
| `vgg16-paper-like` | VGG16 | 16384 | 16 | SGD | **Paper reproduction target** |
| `official-res101-like` | ResNet101 | 16384 | 10 | SGD | Official Caffe ResNet config |

## Setup

```powershell
pip install -r requirements.txt
```

## Usage

### 1. GMM Initialization

```powershell
# VGG16 (paper reproduction)
python scripts/init_gmm.py --preset vgg16-paper-like --data-root data --pretrained --num-workers 4

# AlexNet
python scripts/init_gmm.py --preset alexnet-paper-like --data-root data --pretrained --num-workers 4
```

### 2. Train

```powershell
# VGG16 paper-like training (paper's stage-2 settings)
python scripts/train.py --preset vgg16-paper-like --data-root data ^
    --output-dir outputs/vgg16_paper ^
    --fisher-init outputs/fishernet_voc2007/fisher_gmm.pt ^
    --pretrained --num-workers 4
```

### 3. Evaluate

```powershell
# Single-scale (baseline)
python scripts/evaluate.py --preset vgg16-paper-like --data-root data ^
    --checkpoint outputs/vgg16_paper/best.pt

# Multi-scale (paper: 5 scales + flip)
python scripts/evaluate.py --preset vgg16-paper-like --data-root data ^
    --checkpoint outputs/vgg16_paper/best.pt ^
    --test-scales 480 576 688 864 1200

# SVM post-processing (paper's pipeline)
python scripts/evaluate.py --preset vgg16-paper-like --data-root data ^
    --checkpoint outputs/vgg16_paper/best.pt ^
    --test-scales 480 576 688 864 1200 --fit-svm --svm-C 1.0
```

## Key Architecture Details

- **VGG16**: removes pool5 (1/16 spatial stride), reuses fc6+fc7, adds 256-dim reduction head
- **Fisher Layer**: K=32 GMM components, D=256 patch dim → 16384-dim FV descriptor
- **Patch extraction**: 7 scales {64,96,128,160,192,224,256}, stride 32, max 800 patches
- **Training augmentation**: random 5-scale resize + 50% horizontal flip
- **Optimizer**: SGD + momentum (0.9), multistep LR (gamma=0.1 @ 60% epoch)
- **Three-tier LR**: backbone=0.001, classifier=0.1, fisher=0.0001

## Project Structure

```
fishernet/
├── configs.py          # Preset configuration system
├── data/
│   ├── voc.py          # VOC dataset + multi-scale augmentation
│   └── patches.py      # Dense patch proposal
├── models/
│   ├── fisher_layer.py # Differentiable Fisher Vector layer
│   └── fishernet.py    # AlexNet/VGG16/ResNet FisherNet models
└── utils/
    └── metrics.py      # mAP evaluation
scripts/
├── train.py            # Training entrypoint
├── evaluate.py         # Evaluation (single/multi-scale + SVM)
├── init_gmm.py         # GMM initialization for Fisher layer
└── benchmark_step.py   # Step time benchmark
reports/
├── handoff_4090.md     # 4090 migration handoff document
└── implementation_report.md
external/
└── official-fishernet  # Official Caffe reference (subrepo)
```

For detailed training strategy comparison, see [training_strategy_comparison.md](training_strategy_comparison.md).
