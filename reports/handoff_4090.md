# Deep FisherNet 4090 Handoff Log

Date: 2026-07-09
Session: B1 C=10 full SVM / B2 gate / C1 gate / C2 frozen BN gate

## Summary

All four candidate schemes from AGENT_4090_RUNBOOK.md have been evaluated:

| Scheme | Backbone | Config | 500-sample SVM | Full SVM | Status |
|--------|----------|--------|:---:|:---:|:--------|
| **B1** | VGG16 | corrected-patches, C=10 | 75.28% | **80.25%** | ✅ Best |
| B2 | VGG16 | official-fisher, 4ep | 67.36% | - | ❌ Gate failed |
| C1 | ResNet101-Spatial | official config, 4ep | 55.13% | - | ❌ Gate failed |
| C2 | ResNet101-Spatial | frozen BN, C=10 | 62.53% | - | ❌ Gate failed |

## B1: VGG16 corrected-patches (BEST RESULT)

**Checkpoint:** `outputs/b1_vgg_corrected_4ep/best.pt`
**Training:** 4 epochs, default SGD, preset `vgg16-corrected-patches`

### Full SVM C=10 Result
**mAP: 80.25%**
```
aeroplane    0.9586    bicycle      0.8702    bird       0.9283
boat         0.9075    bottle       0.5668    bus        0.8165
car          0.8715    cat          0.8566    chair      0.6314
cow          0.7173    diningtable  0.7130    dog        0.8670
horse        0.8784    motorbike    0.8211    person     0.9239
pottedplant  0.5088    sheep        0.8288    sofa       0.6720
train        0.9452    tvmonitor    0.7667
```

Log: `outputs/agent_logs/b1_svm_full_C10.log` (727 KB)

### C-parameter sweep (500 samples)
| C | mAP |
|---|:---:|
| 0.01 | 72.46% |
| 0.1 | 72.46% |
| 1 | 74.15% |
| **10** | **75.28%** |
| 100 | 75.03% |

Best C=10 confirmed on full data → 80.25%.

## B2: VGG16 official-fisher (GATE FAILED)

**Checkpoint:** `outputs/b2_vgg_official_fisher_4ep/best.pt`
**1epoch 500-sample SVM:** 68.80% ✅ Passed gate (needed >67.28%)
**4epoch 500-sample SVM:** 67.36% ❌ Below B1 500-sample - 1% (74.28%)

Training:
```
epoch=1: train_loss=0.4874  val_mAP=0.1155
epoch=2: ...
epoch=3: train_loss=0.3504  val_mAP=0.1103
epoch=4: train_loss=0.3407  val_mAP=0.1090
```

Logs: `outputs/agent_logs/b2_train_4ep.log`, `outputs/agent_logs/b2_svm_1ep_500.log`
Decision: Stop. Caffe-style Fisher math (log-det, sum pooling, learned priors) doesn't help within 4 epochs.

## C1: ResNet101-Spatial (GATE FAILED - POOR)

**Checkpoint:** `outputs/c1_res101_spatial_4ep/best.pt` (330 MB)
**Training:** 4 epochs, only 3:02 total (very fast)
**500-sample SVM:** 55.13% ❌ Far below B1 baseline

Training:
```
epoch=1: train_loss=0.4865  val_mAP=0.0839
epoch=2: train_loss=0.3265  val_mAP=0.0911
epoch=3: train_loss=0.2953  val_mAP=0.0922
epoch=4: train_loss=0.2918  val_mAP=0.0940
```

Decision: Stop. spatial Fisher on ResNet101 performs much worse than patch Fisher on VGG16. Possible cause: removing RoI Align/patch sampling loses spatial invariance that helps classification.

## C2: ResNet101-Spatial Frozen BN (GATE FAILED)

**Checkpoint:** `outputs/c2_res101_spatial_frozenbn_4ep/best.pt` (330 MB)
**Training:** 4 epochs with frozen BN, preset `official-res101-spatial-frozenbn`
**500-sample SVM C=10:** 62.53% ❌ Below 70% fail threshold

Training:
```
epoch=1: train_loss=0.4872  val_mAP=0.1085
epoch=2: train_loss=0.3265  val_mAP=0.1597
epoch=3: train_loss=0.2953  val_mAP=0.1719
epoch=4: train_loss=0.2918  val_mAP=0.1805
```

Frozen BN improved val_mAP (0.0940→0.1805, +92%) and SVM mAP (55.13%→62.53%, +7.4%) vs C1, but still far below B1 baseline. Likely root cause: ResNet101 spatial Fisher lacks the dense patch sampling that VGG16 patch Fisher uses.

Per-class AP:
```
aeroplane    0.8741  bicycle    0.3769  bird       0.8088
boat         0.7839  bottle     0.3814  bus        0.7581
car          0.8453  cat        0.6143  chair      0.4079
cow          0.4834  diningtab  0.4686  dog        0.5705
horse        0.8750  motors     0.8217  person     0.7937
pottedplant  0.4108  sheep      0.3696  sofa       0.5599
train        0.9326  tvmonitor  0.3706
```

Log: `outputs/agent_logs/c2_svm_4ep_500.log` (73 KB)
Decision: Stop. Frozen BN helps but insufficient. ResNet101-spatial architecture is the bottleneck.

## GPU Usage

- RTX 4090 24GB
- B1/B2: ~6.5 GB during training/eval
- C1: ~4.2-4.9 GB during training, ~1.7 GB idle during eval hang
- Peak observed: ~22.5 GB (vgg16_full eval OOM - multiscale)

## Key Files

Local: `F:/paper_ws/Deep FisherNet for Object Classification/`
Remote: `/root/autodl-tmp/Deep-FisherNet-PyTorch/`

### Logs (copied to reports/)
- `reports/b1_svm_full_C10.log` - B1 full SVM C=10 result (80.25%)
- `reports/b1_svm_full.log` - B1 full SVM C=1 result (79.76%)
- `reports/c1_train_4ep.log` - C1 training log
- `reports/c1_svm_4ep_500.log` - C1 500-sample SVM (55.13%)

- `reports/c2_train_4ep.log` - C2 frozen BN training log
- `reports/c2_svm_4ep_500.log` - C2 500-sample SVM C=10 (62.53%)

### Remote agent_logs/ (not synced, ~2 MB each)
- b2_train_4ep.log, b2_svm_1ep_500.log
- a0_smoke.log, a1_visualize.log, a2_forward_stats.log
- b1_train.log, b1_svm_*.log (C sweep variants)

## Next Steps (Suggested)

1. B1 more epochs (8-16) to see if mAP improves beyond 80.25%
2. Debug C1: why spatial Fisher fails (no patch sampling?)
3. Try multi-scale evaluation on B1 best checkpoint
