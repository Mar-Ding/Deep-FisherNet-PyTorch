#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT="${OUTPUT:-outputs/vgg16_paper_final}"
WORKERS="${WORKERS:-4}"

mkdir -p "$OUTPUT/logs" "$OUTPUT/eval_cache"

on_error() {
  status=$?
  echo "Pipeline failed with status $status at $(date --iso-8601=seconds)" | tee -a "$OUTPUT/logs/failure.log"
  nvidia-smi 2>&1 | tee -a "$OUTPUT/logs/failure.log" || true
  df -h "$OUTPUT" 2>&1 | tee -a "$OUTPUT/logs/failure.log" || true
  exit "$status"
}
trap on_error ERR

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "python=$PYTHON_BIN"
  echo "data_root=$DATA_ROOT"
  echo "output=$OUTPUT"
  "$PYTHON_BIN" -c 'import platform, torch; print(f"platform={platform.platform()}"); print(f"python_version={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}")'
  nvidia-smi
  df -h "$OUTPUT"
} 2>&1 | tee "$OUTPUT/logs/environment.log"

echo "=== 0/5 Static paper-path audit ==="
"$PYTHON_BIN" scripts/audit_paper_pipeline.py --strict \
  2>&1 | tee "$OUTPUT/logs/audit_static.log"

echo "=== 1/5 Stage 1: pretrained VGG16 whole-image fine-tuning, exactly 9000 updates ==="
"$PYTHON_BIN" scripts/train.py \
  --stage1 \
  --preset stage1-vgg16-paper-final \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT/stage1" \
  --pretrained \
  --num-workers "$WORKERS" \
  2>&1 | tee "$OUTPUT/logs/stage1.log"

echo "=== 2/5 Fit the 32-component diagonal GMM from transferred 256-d patch features ==="
"$PYTHON_BIN" scripts/init_gmm_stage1.py \
  --preset vgg16-paper-final \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT/fisher_gmm.pt" \
  --checkpoint "$OUTPUT/stage1/last.pt" \
  --hflip-prob 0 \
  --max-descriptors 100000 \
  --descriptors-per-image 100 \
  --gmm-max-iter 150 \
  --batch-size 1 \
  --num-workers "$WORKERS" \
  2>&1 | tee "$OUTPUT/logs/gmm.log"

echo "=== 3/5 Artifact and Fisher-assignment gate ==="
"$PYTHON_BIN" scripts/audit_paper_pipeline.py \
  --data-root "$DATA_ROOT" \
  --stage1-checkpoint "$OUTPUT/stage1/last.pt" \
  --gmm "$OUTPUT/fisher_gmm.pt" \
  --output "$OUTPUT/pipeline_audit.json" \
  --max-images 16 \
  --max-descriptors 10000 \
  --num-workers "$WORKERS" \
  --device cuda \
  --strict \
  2>&1 | tee "$OUTPUT/logs/audit_artifacts.log"

echo "=== 4/5 Stage 2: end-to-end FisherNet, exactly 40000 updates ==="
"$PYTHON_BIN" scripts/train.py \
  --preset vgg16-paper-final \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT/stage2" \
  --stage1-weights "$OUTPUT/stage1/last.pt" \
  --fisher-init "$OUTPUT/fisher_gmm.pt" \
  --num-workers "$WORKERS" \
  2>&1 | tee "$OUTPUT/logs/stage2.log"

echo "=== 5/5 Paper evaluation: normalize each FV, mean five scales, no flip, LIBLINEAR C=1 ==="
"$PYTHON_BIN" scripts/evaluate.py \
  --preset vgg16-paper-final \
  --data-root "$DATA_ROOT" \
  --checkpoint "$OUTPUT/stage2/last.pt" \
  --fit-svm \
  --feature-cache-dir "$OUTPUT/eval_cache" \
  --metrics-output "$OUTPUT/final_metrics.json" \
  --num-workers "$WORKERS" \
  2>&1 | tee "$OUTPUT/logs/final_eval.log"

echo "Done. Final metrics: $OUTPUT/final_metrics.json"
