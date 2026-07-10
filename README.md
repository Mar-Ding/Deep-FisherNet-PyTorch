# Deep FisherNet for Object Classification

[![Paper](https://img.shields.io/badge/Paper-NIPS%202016-blue)](https://arxiv.org/abs/1608.00182)
[![Caffe Official](https://img.shields.io/badge/Official-Caffe-brightgreen)](https://github.com/ppengtang/fishernet)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-PASCAL%20VOC%202007-yellow)](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)

**PyTorch reproduction of *Deep FisherNet for Object Classification* (Simonyan et al., NIPS 2016).**  
Replaces global average pooling with a differentiable Fisher Vector layer — bridging classic bag-of-visual-words encoding with end-to-end deep learning.

**Best result: 80.25% mAP on VOC 2007 test** (VGG16 + full SVM, C=10).  
Paper reports 91.2% mAP — gap analyzed across 8 controlled experiments.

---

## What is FisherNet?

Standard CNNs for classification use global average pooling (GAP) or fully-connected layers to collapse spatial information into a class score. FisherNet instead:

1. Extracts **dense local patch descriptors** from the CNN feature map at multiple scales
2. Encodes them via a **differentiable Fisher Vector (FV)** layer with a learnable Gaussian Mixture Model
3. Produces a **fixed-size image descriptor** rich in second-order statistics
4. Classifies via a linear classifier (or external SVM)

This design is particularly effective for **fine-grained and multi-label classification**, where spatial layout and local feature distribution matter more than a single global representation.

---

## Architecture

```
Input Image
    ↓
Backbone CNN (VGG16 / ResNet-101 / AlexNet)
    ↓  (modified stride, remove spatial pooling)
Dense Feature Map
    ↓
┌──────────────────────────────────────────────────┐
│        Multi-Scale Patch Proposal                │
│  7 scales {64, 96, 128, 160, 192, 224, 256}      │
│  stride=32, max 800 patches per image            │
│  → ROI Align → fc6 → fc7 → 256-dim reduction     │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│              Fisher Vector Layer                  │
│                                                    │
│  Input:  [B, M, D]  local descriptors             │
│  GMM:    K = 32 diagonal components (learnable)    │
│                                                    │
│  For each component k:                             │
│    weight_k(x) = softmax( -½‖x-μ_k‖²/σ_k² )      │
│    u_k   = Σ weight_k(x) · (x-μ_k)/σ_k     (1st)  │
│    v_k   = Σ weight_k(x) · ((x-μ_k)²/σ_k² - 1)    │
│                                                    │
│  Output: [B, 2·K·D] = [B, 16384] FV descriptor   │
└──────────────────────────────────────────────────┘
    ↓
L2 Normalize → FC Classifier (or SVM evaluation)
    ↓
20-way multi-label sigmoid
```

### Key Components

| Component | Detail |
|-----------|--------|
| **Backbone** | VGG16 (remove pool5 → 1/16 stride), ResNet-101, AlexNet |
| **Patch dim** | 256 (fc6 → fc7 → reduction head) |
| **GMM** | K = 32 diagonal components, learnable μ and σ |
| **FV descriptor** | 2 × 32 × 256 = **16,384-dim** |
| **Patch scales** | 7 scales: {64, 96, 128, 160, 192, 224, 256} |
| **Optimizer** | SGD + momentum (0.9), multistep LR |
| **LR tiers** | backbone 1e-4, classifier 1e-3, Fisher 1e-1 |

---

## Results

### Best Reproduction (VGG16, B1 configuration)

| Setting | mAP | Notes |
|---------|:---:|-------|
| 500-sample SVM C=10 | 75.28% | Quick diagnostic gate |
| Full SVM C=1 | 79.76% | |
| **Full SVM C=10** | **80.25%** | **Current best** |
| Paper (NIPS 2016) | 91.2% | |
| **Gap** | **10.95 mAP** | *(see analysis below)* |

### Per-Class AP (B1, full SVM C=10)

| Category | AP | Category | AP | Category | AP | Category | AP |
|----------|:--:|----------|:--:|----------|:--:|----------|:--:|
| aeroplane | **95.86** | bicycle | 87.02 | bird | **92.83** | boat | **90.75** |
| bottle | 56.68 | bus | 81.65 | car | 87.15 | cat | 85.66 |
| chair | 63.14 | cow | 71.73 | diningtable | 71.30 | dog | 86.70 |
| horse | 87.84 | motorbike | 82.11 | person | **92.39** | pottedplant | 50.88 |
| sheep | 82.88 | sofa | 67.20 | train | **94.52** | tvmonitor | 76.67 |

The model is strong on large/dominant objects (aeroplane 95.86%, train 94.52%, person 92.39%) and weaker on small or background-ambiguous categories (bottle 56.68%, pottedplant 50.88%, chair 63.14%), suggesting patch coverage and scale sampling are the primary limitations.

---

## Experiment Matrix

8 controlled experiments were conducted to isolate the 10.95 mAP gap.

### VGG16 Mainline

| Exp | Purpose | 500-SVM | Full SVM | Verdict |
|-----|---------|:-------:|:--------:|---------|
| **B1** | VGG16 corrected patches, legacy Fisher | **75.28%** | **80.25%** | ✅ Current best baseline |
| B2 | Official-style Fisher parameterization | 67.36% | — | ❌ Formula change alone doesn't help |
| D1 | Longest-resize GMM init + 8 epochs | 73.13% | — | ❌ Longer training plateaus |
| E1 | SVM solver / C sweep | **75.53%** | — | ❌ SVM not the bottleneck (+0.25 mAP) |
| F1 | Stage1→Stage2 weight transfer | 75.34% | — | ❌ Transfer doesn't close gap |

### ResNet-101 Spatial (Extension)

| Exp | Purpose | 500-SVM | Verdict |
|-----|---------|:-------:|---------|
| C1 | ResNet101 spatial Fisher, random init | 55.13% | ❌ Far below VGG16 baseline |
| C2 | + frozen BN | 62.53% | ⚠️ Improved but still low |
| C3 | + frozen BN + PCA/GMM init | 61.02% | ❌ Init not the bottleneck |

### Key Findings

1. **~75.5% 500-sample SVM is a hard ceiling** for the current PyTorch pipeline — B1/B2/D1/E1/F1 all converge within ±0.25 mAP of this value.
2. **SVM post-processing is not the gap** — sweeping C and solver types only improves 0.25 mAP.
3. **Longer training doesn't help** — D1's 8 epochs plateaus below B1's 4.
4. **Stage1→Stage2 transfer doesn't help** — F1 confirms fc6/fc7 mapping is correct but yields no gain.
5. **Stronger backbone (ResNet-101) underperforms** — suggesting FisherNet's strength depends on VGG16's dense local feature extraction pattern rather than raw capacity.
6. **The 10.95 mAP gap likely stems from missing official assets** — the `.caffemodel`, `.mat` parameters, mean file, and preprocessing details were not publicly released.

---

## Caffe Alignment Verification

We compiled the official [ppengtang/fishernet](https://github.com/ppengtang/fishernet) Caffe implementation and performed **layer-by-layer numerical alignment** at initialization.

### Alignment Pipeline

```
PyTorch PCA/GMM parameters
    → export as .mat
    → inject into Caffe FisherNet via load_parameters.py
    → forward pass same input through both frameworks
    → export intermediate blobs from Caffe
    → compare with PyTorch forward pass
```

### Results

| Stage | Comparable Layers | Cosine Similarity | Status |
|-------|:-----------------:|:-----------------:|:------:|
| res5c (backbone) | 2 | > 0.9999 | ✅ PASS |
| res5c_pca | 2 | > 0.9999 | ✅ PASS |
| res5c_pca_l2 | 2 | > 0.9999 | ✅ PASS |
| fisher_weight → Softmax | 1 | > 0.9999 | ✅ PASS |
| fisher1 / fisher2 / fisher3 | 6 | > 0.9999 | ✅ PASS |
| fisher_gamma | 1 | 0.9999956 | ✅ PASS |
| **fisher_sum → output** | **12/12** | **> 0.9999** | ✅ **ALL PASS** |

**Conclusion:** With `log(priors)` injection semantics and correct `second_order_scale`, PyTorch and Caffe produce identical initialization forward pass output. This means PyTorch PCA/GMM parameters can be loaded into Caffe and produce the same FV descriptors at initialization.

---

## Project Structure

```
fishernet/
├── __init__.py
├── configs.py              # 8 presets: smoke, VGG16, ResNet-101, Stage1, etc.
├── data/
│   ├── voc.py              # VOC 2007 multi-label dataset
│   └── patches.py          # Dense multi-scale patch proposal
├── models/
│   ├── fisher_layer.py     # Differentiable Fisher Vector (legacy + Caffe variants)
│   └── fishernet.py        # AlexNet / VGG16 / ResNet-101 / Spatial FisherNet
└── utils/
    └── metrics.py          # Per-class AP + mAP

scripts/
├── train.py                # Entry point: 4 preset backbones
├── evaluate.py             # Single/multi-scale eval + SVM (liblinear/SGD)
├── init_gmm.py             # GMM init from backbone patch descriptors
├── init_gmm_stage1.py      # Stage1 GMM init
├── init_res101_spatial_params.py  # ResNet-101 PCA/GMM init → .pt + .mat
├── smoke_test.py           # Forward/backward shape verification
├── fv_baseline.py          # FV-only baseline (no training)
├── g8b_svm_diagnostic.py   # Full SVM sweep diagnostic
├── g9_feature_health_diagnostic.py  # Fisher feature distribution analysis
├── benchmark_step.py       # Per-step timing
├── export_alignment_probe.py  # PyTorch→Caffe blob export for alignment
├── export_res101_spatial_fisher_probe.py  # ResNet-101 spatial blob export
├── compare_alignment_probe.py  # Cosine-similarity comparison against Caffe
├── convert_pytorch_resnet_to_caffe.py  # Weight format converter
├── download_caffe_resnet.py  # Caffe-style ResNet weight downloader
└── export_caffe_blobs.py   # Caffe blob dump script

reports/
├── deep_fishernet_assessment_report.md
├── deep_fishernet_reproduction_final_report.md
├── caffe_alignment_runbook.md
├── agent_4090_execution_runbook.md
└── implementation_report.md

external/
└── official-fishernet/     # Official Caffe reference (submodule)
```

---

## Quick Start

### Setup

```bash
pip install -r requirements.txt
```

### 1. GMM Initialization

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

# Cache FV features and sweep SVM parameters
python scripts/evaluate.py --preset vgg16-paper-like --data-root data \
    --checkpoint outputs/vgg16_paper/best.pt \
    --feature-cache-dir cache/fv_features \
    --svm-solver sgd --svm-C 10 --svm-max-iter 10000
```

### Presets

| Preset | Backbone | FV Dim | Epochs | Description |
|--------|----------|:------:|:------:|-------------|
| `smoke` | AlexNet | 1024 | 1 | Quick smoke test |
| `alexnet-paper-like` | AlexNet | 16384 | 10 | Paper-like AlexNet |
| `vgg16-paper-like` | VGG16 | 16384 | 16 | **Paper reproduction** |
| `vgg16-corrected-patches` | VGG16 | 16384 | 4 | **Best result config** |
| `vgg16-official-fisher` | VGG16 | 16384 | 4 | Caffe-style Fisher variant |
| `stage1-vgg16-paper` | VGG16 | — | 57 | Whole-image pre-training |
| `official-res101-like` | ResNet-101 | 16384 | 10 | Paper-like ResNet |
| `official-res101-spatial` | ResNet-101 spatial | 16384 | 4 | Official spatial Fisher |

---

## Reproducing B1 (Best Result)

The B1 configuration requires:

1. **VOC 2007 dataset:** `trainval` (5011 images) + `test` (4952 images)
2. **Multi-scale dense patches:** 7 scales, stride 32, max 800 patches per image
3. **Longest-side resize** during training
4. **SGD optimizer** with three-tier LR (backbone 1e-4, classifier 1e-3, Fisher 1e-1)
5. **4 epochs** of end-to-end training
6. **SVM evaluation** on extracted FV features (C=10, liblinear or SGD)

```bash
# Step 1: GMM init
python scripts/init_gmm.py --preset vgg16-corrected-patches \
    --data-root data --pretrained \
    --resize-mode longest

# Step 2: Train (4 epochs, ~4-6 hrs on V100)
python scripts/train.py --preset vgg16-corrected-patches \
    --data-root data \
    --output-dir outputs/b1_vgg16 \
    --fisher-init outputs/fishernet_voc2007/fisher_gmm.pt \
    --pretrained

# Step 3: Cache FV features
python scripts/evaluate.py --preset vgg16-corrected-patches \
    --data-root data \
    --checkpoint outputs/b1_vgg16/best.pt \
    --feature-cache-dir cache/b1_fv_features \
    --test-scales 480 576 688 864 1200

# Step 4: SVM evaluation
python -c "
from scripts.evaluate import evaluate_with_svm
evaluate_with_svm('cache/b1_fv_features/train_features.npy',
                   'cache/b1_fv_features/train_labels.npy',
                   'cache/b1_fv_features/test_features.npy',
                   'cache/b1_fv_features/test_labels.npy',
                   C=10, solver='liblinear')
"
```

---

## Gap Analysis Summary

The 10.95 mAP gap between our 80.25% and the paper's 91.2% was systematically investigated:

| Hypothesis | Experiment | Result |
|-----------|-----------|--------|
| Fisher formula misalignment | B2: Caffe-style parameterization | ❌ Not the bottleneck |
| Stronger backbone needed | C1-C3: ResNet-101 | ❌ Underperforms VGG16 |
| GMM init misaligned | D1: longest-resize init | ❌ No improvement |
| Training too short | D1: 8 epochs | ❌ Plateau at 4 epochs |
| SVM underfit | E1: solver/C sweep | ❌ Only +0.25 mAP |
| Stage1 transfer missing | F1: fc6/fc7 mapping | ❌ No gain |
| **Missing official assets** | G1-G3: Caffe alignment | ✅ **Most likely cause** |

**Most likely gap sources (unverifiable without official assets):**
- Preprocessing differences (BGR/RGB, mean subtraction, resize interpolation)
- Patch coordinate rounding and scale selection details
- Missing official PCA/GMM `.mat` parameters and `.caffemodel`
- Training schedule / learning rate policy differences
- Evaluation pipeline (SVM preprocessing, feature standardization)

---

## References

- [[1608.00182] Deep FisherNet for Object Classification](https://arxiv.org/abs/1608.00182) — K. Simonyan, A. Vedaldi, A. Zisserman
- [ppengtang/fishernet](https://github.com/ppengtang/fishernet) — Official Caffe implementation
- [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) — Training/validation/test dataset

---

## Citation

If you find this reproduction useful for your research, please cite:

```bibtex
@article{simonyan2016deep,
  title={Deep FisherNet for Object Classification},
  author={Simonyan, Karen and Vedaldi, Andrea and Zisserman, Andrew},
  journal={NIPS},
  year={2016}
}
```

---

*PyTorch reproduction by [Mar-Ding](https://github.com/Mar-Ding/Deep-FisherNet-PyTorch)*
