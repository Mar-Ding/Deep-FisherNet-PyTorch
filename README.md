# Deep FisherNet for Object Classification

PyTorch reproduction of **Deep FisherNet for Object Classification** (NIPS 2016).

> **Paper**: [Deep FisherNet for Object Classification](https://arxiv.org/abs/1608.00182)
> **Official Caffe**: [ppengtang/fishernet](https://github.com/ppengtang/fishernet)

---

## Overview

FisherNet replaces the global average pooling / fully-connected classifier head with a **differentiable Fisher Vector layer**. Local CNN patch descriptors are extracted densely from the feature map, softly assigned to learnable Gaussian mixture components, and aggregated via first-order and second-order Fisher scores into a fixed-size image descriptor. This bridges classic bag-of-visual-words encoding with end-to-end deep learning.

**Supported backbones**: AlexNet, VGG16, ResNet-101  
**Dataset**: PASCAL VOC 2007 (20-class multi-label classification, mAP metric)

---

## Architecture

```
Input Image
    ↓
Backbone CNN (VGG16 / ResNet / AlexNet)
    ↓  (remove spatial pooling, keep dense feature map)
Patch Proposals (7-scale dense crops, stride 32)
    ↓
PCA / Reduction Head (256-dim)
    ↓
┌─────────────────────────────────┐
│     Fisher Vector Layer         │
│  GMM (K=32) → soft assignment   │
│  Σ (first-order + second-order) │
│  → 16384-dim image descriptor   │
└─────────────────────────────────┘
    ↓
L2 Normalize + FC Classifier
    ↓
20-way multi-label sigmoid
```

| Component | Configuration |
|-----------|--------------|
| Backbone | VGG16 (remove pool5, keep conv5_3 at 1/16 stride) |
| Patch dim | 256 (reduction head on fc6/fc7 features) |
| GMM components | K = 32 |
| FV descriptor | 2 × K × D = 2 × 32 × 256 = **16384-dim** |
| Patch scales | 7 scales {64, 96, 128, 160, 192, 224, 256} |
| Training aug | Random 5-scale resize + 50% horizontal flip |
| Optimizer | SGD + momentum (0.9), multistep LR |
| Three-tier LR | backbone 1e-3, classifier 1e-1, fisher 1e-4 |

---

## Quick Start

### Setup

```bash
pip install -r requirements.txt
```

### 1. GMM Initialization

Collect local patch descriptors from a pretrained backbone and fit a diagonal GMM for initializing the Fisher Layer:

```bash
# VGG16 (paper reproduction target)
python scripts/init_gmm.py --preset vgg16-paper-like --data-root data --pretrained

# AlexNet
python scripts/init_gmm.py --preset alexnet-paper-like --data-root data --pretrained
```

### 2. Train

```bash
python scripts/train.py --preset vgg16-paper-like --data-root data \
    --output-dir outputs/vgg16_paper \
    --fisher-init outputs/fishernet_voc2007/fisher_gmm.pt \
    --pretrained
```

### 3. Evaluate

```bash
# Single-scale baseline
python scripts/evaluate.py --preset vgg16-paper-like --data-root data \
    --checkpoint outputs/vgg16_paper/best.pt

# Multi-scale evaluation (paper: 5 scales + horizontal flip)
python scripts/evaluate.py --preset vgg16-paper-like --data-root data \
    --checkpoint outputs/vgg16_paper/best.pt \
    --test-scales 480 576 688 864 1200

# With SVM post-processing (paper's final pipeline)
python scripts/evaluate.py --preset vgg16-paper-like --data-root data \
    --checkpoint outputs/vgg16_paper/best.pt \
    --test-scales 480 576 688 864 1200 --fit-svm --svm-C 1.0
```

---

## Presets

| Preset | Backbone | FV Dim | Epochs | Optimizer | Description |
|--------|----------|--------|--------|-----------|-------------|
| `smoke` | AlexNet | 1024 | 1 | AdamW | Quick smoke test |
| `alexnet-paper-like` | AlexNet | 16384 | 10 | SGD | Paper-like AlexNet |
| `vgg16-paper-like` | VGG16 | 16384 | 16 | SGD | **Paper reproduction** |
| `official-res101-like` | ResNet-101 | 16384 | 10 | SGD | Official Caffe config |

---

## Project Structure

```
fishernet/
├── configs.py              # Preset configuration system
├── data/
│   ├── voc.py              # VOC 2007 multi-label dataset
│   └── patches.py          # Dense patch proposal generation
├── models/
│   ├── fisher_layer.py     # Differentiable Fisher Vector layer
│   └── fishernet.py        # AlexNet / VGG16 / ResNet FisherNet
└── utils/
    └── metrics.py          # Per-class AP and mAP evaluation

scripts/
├── train.py                # Training entrypoint
├── evaluate.py             # Single / multi-scale evaluation + SVM
├── init_gmm.py             # GMM initialization for Fisher layer
├── benchmark_step.py       # Per-step timing benchmark
└── smoke_test.py           # Forward / backward shape verification

reports/
├── deep_fishernet_assessment_report.md
└── deep_fishernet_reproduction_final_report.md

external/
└── official-fishernet      # Official Caffe reference implementation
```

---

## Key Implementation Details

- **Fisher Vector layer** at `fishernet/models/fisher_layer.py`: differentiable GMM soft-assignment with learnable component means (`bias = -μ`) and diagonal precision (`weight = 1/σ`). Supports first-order and second-order statistics with optional prior learning.
- **Multi-scale dense patches** at `fishernet/data/patches.py`: sliding window proposal over the feature map at multiple scales with configurable stride and maximum patch count.
- **Three-tier learning rate**: backbone features, Fisher Layer parameters, and classifier weights are each assigned separate learning rates to account for their different roles in the architecture.

---

## References

- [[1608.00182] Deep FisherNet for Object Classification](https://arxiv.org/abs/1608.00182)
- [ppengtang/fishernet — Official Caffe implementation](https://github.com/ppengtang/fishernet)
- [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) — Training/validation/test dataset
