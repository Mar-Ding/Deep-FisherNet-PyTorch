# Deep FisherNet PyTorch Reproduction Report

## Current Scope

This repository implements a modern PyTorch scaffold for **Deep FisherNet for Object Classification** on PASCAL VOC 2007. The implementation follows the paper's central idea: extract dense local CNN patch descriptors, softly assign each descriptor to learnable Gaussian/Fisher components, aggregate first-order and second-order deviations, then train a multi-label classifier.

## Paper Interpretation

The GMM components are not object categories. They are visual-word-like centers in the local descriptor space. For each patch descriptor, the Fisher layer computes how far it is from each component center after diagonal scaling. The soft assignment says which components the patch is close to. The first-order term keeps signed offsets from each component, and the second-order term keeps variance-like offsets. These pooled deviations form the image descriptor.

In this PyTorch version, the Gaussian parameters are trainable:

- `bias = -mu`, representing the negative component mean.
- `weight = 1 / sigma`, representing inverse diagonal standard deviation.
- optional priors can be learned through logits.

This matches the paper's rewrite of Fisher Vector computation as differentiable network layers.

## Official Code Reference

The paper has an official-era Caffe repository at `external/official-fishernet`. It contains custom layers such as `FisherWeight`, `FisherScale`, and `FisherSum`, plus a ResNet-101 prototxt variant. The Caffe code confirms the same core structure: PCA/reduced local features, L2 normalization, Fisher soft assignment, first/second-order statistics, sum pooling, and a multi-label classifier.

## Implemented Files

- `fishernet/models/fisher_layer.py`: differentiable Fisher Vector layer.
- `fishernet/models/fishernet.py`: AlexNet-based dense patch FisherNet.
- `fishernet/data/voc.py`: VOC 2007 multi-label dataset wrapper.
- `fishernet/data/patches.py`: dense patch proposal generation.
- `fishernet/utils/metrics.py`: per-class AP and mAP.
- `scripts/train.py`: training entrypoint.
- `scripts/evaluate.py`: checkpoint evaluation entrypoint.
- `scripts/init_gmm.py`: collects local patch descriptors and fits a diagonal GMM for Fisher initialization.
- `scripts/smoke_test.py`: fast forward/backward check.

## Reproduction Notes

This is a faithful PyTorch implementation of the core FisherNet mechanism, not a byte-for-byte port of the Caffe training recipe. The current model uses fixed-size resized images and dense square patches for a clean first reproduction. The paper and Caffe code use richer multi-scale settings, and those can be added after the first VOC run is stable.

## Commands

```powershell
python scripts/smoke_test.py
python scripts/init_gmm.py --data-root data --download --pretrained
python scripts/train.py --data-root data --pretrained --fisher-init outputs/fishernet_voc2007/fisher_gmm.pt --epochs 10 --batch-size 1
python scripts/evaluate.py --data-root data --checkpoint outputs/fishernet_voc2007/best.pt
```
