# 4090 Agent Runbook

This repo is prepared so the remote agent only needs to run diagnostics/training and return logs.

Cost rule: RTX 4090 rental is estimated at 2.2 yuan/hour. Any command expected to exceed 30 minutes must be estimated and approved by the user before running.

> **Status after session 2026-07-09:** All 4 schemes evaluated.
> ✅ B1 (best): 80.25% mAP (VGG16 corrected-patches, full SVM C=10)
> ❌ B2: gate failed (67.36% 500-sample SVM)
> ❌ C1: gate failed (55.13% 500-sample SVM)
> ❌ C2: gate failed (62.53% 500-sample SVM, frozen BN)
> See `reports/handoff_4090.md` for full results. GPU instance released.

## Environment

```bash
export DATA_ROOT=/path/to/VOCdevkit_parent
export PYTHONPATH=$PWD
mkdir -p outputs/agent_logs outputs/debug_patches
```

`DATA_ROOT` must be the parent directory that contains `VOCdevkit/VOC2007`.

## First Remote Session

Run these first. Do not start long training yet.

```bash
python scripts/train.py \
  --preset smoke \
  --data-root "$DATA_ROOT" \
  --output-dir outputs/a0_smoke \
  --pretrained \
  --max-train-batches 2 \
  --max-val-batches 2 \
  --num-workers 2
```

```bash
python scripts/visualize_patches.py \
  --data-root "$DATA_ROOT" \
  --output-dir outputs/debug_patches/longest \
  --image-set trainval \
  --image-size 480 \
  --resize-mode longest \
  --max-samples 20
```

```bash
python scripts/debug_forward_stats.py \
  --preset vgg16-corrected-patches \
  --data-root "$DATA_ROOT" \
  --output outputs/agent_logs/a2_vgg_corrected_stats.json \
  --max-batches 20 \
  --device cuda
```

## Candidate Training Schemes

### B1: corrected VGG patch pipeline

Fixes longest-side resize and per-scale boxes, keeps legacy Fisher math.

Estimated time: 20-40 minutes for 4 epochs; 1-2 hours if followed by full SVM eval.

```bash
python scripts/train.py \
  --preset vgg16-corrected-patches \
  --data-root "$DATA_ROOT" \
  --output-dir outputs/b1_vgg_corrected_4ep \
  --pretrained \
  --num-workers 4 \
  2>&1 | tee outputs/agent_logs/b1_train.log
```

```bash
python scripts/evaluate.py \
  --preset vgg16-corrected-patches \
  --data-root "$DATA_ROOT" \
  --checkpoint outputs/b1_vgg_corrected_4ep/best.pt \
  --fit-svm \
  --max-samples 500 \
  --device cuda \
  2>&1 | tee outputs/agent_logs/b1_svm_500.log
```

### B2: VGG with official-like Fisher math

Adds Caffe-style Fisher parameterization, log-det term, learned priors, and SumPooling.

```bash
python scripts/train.py \
  --preset vgg16-official-fisher \
  --data-root "$DATA_ROOT" \
  --output-dir outputs/b2_vgg_official_fisher_4ep \
  --pretrained \
  --num-workers 4 \
  2>&1 | tee outputs/agent_logs/b2_train.log
```

```bash
python scripts/evaluate.py \
  --preset vgg16-official-fisher \
  --data-root "$DATA_ROOT" \
  --checkpoint outputs/b2_vgg_official_fisher_4ep/best.pt \
  --fit-svm \
  --max-samples 500 \
  --device cuda \
  2>&1 | tee outputs/agent_logs/b2_svm_500.log
```

### C1: official Res101 spatial Fisher

Closest to the official `models/Res-101/train.prototxt`: `res5c -> 1x1 conv -> spatial Fisher`.

Estimated time: 30-60 minutes for 4 epochs; 1.5-3 hours if followed by full multiscale SVM eval.

```bash
python scripts/train.py \
  --preset official-res101-spatial \
  --data-root "$DATA_ROOT" \
  --output-dir outputs/c1_res101_spatial_4ep \
  --pretrained \
  --num-workers 4 \
  2>&1 | tee outputs/agent_logs/c1_train.log
```

```bash
python scripts/evaluate.py \
  --preset official-res101-spatial \
  --data-root "$DATA_ROOT" \
  --checkpoint outputs/c1_res101_spatial_4ep/best.pt \
  --fit-svm \
  --max-samples 500 \
  --device cuda \
  2>&1 | tee outputs/agent_logs/c1_svm_500.log
```

## Stop Conditions

Stop and report immediately if:

- smoke test fails;
- patch overlays obviously do not cover the image;
- forward stats contain NaN/Inf;
- a 4-epoch run still has FC mAP near 0.1 and 500-sample SVM does not improve;
- any unapproved command is expected to exceed 30 minutes.

## Report Back

Return command, elapsed time, GPU model, peak memory if available, train loss, val mAP, SVM mAP, per-class AP, and paths to logs.
