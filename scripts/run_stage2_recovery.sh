#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

MODE="${1:?usage: run_stage2_recovery.sh gate|full}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/Deep-FisherNet-PyTorch/data}"
OUTPUT="${OUTPUT:-outputs/vgg16_paper_final_20260711}"
WORKERS="${WORKERS:-4}"

mkdir -p "$OUTPUT/logs"

check_checkpoint() {
  local checkpoint="$1"
  "$PYTHON_BIN" - "$checkpoint" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu")
state = checkpoint.get("model", checkpoint)
bad = [name for name, value in state.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
if bad:
    raise SystemExit(f"non-finite checkpoint tensors ({len(bad)}): {bad[:10]}")
print(f"checkpoint_finite=true tensors={len(state)} step={checkpoint.get('global_step')}")
PY
}

case "$MODE" in
  gate)
    GATE_DIR="$OUTPUT/stage2_gate_caffe_fused"
    GATE_LOG="$OUTPUT/logs/stage2_gate_caffe_fused.log"
    rm -rf "$GATE_DIR"
    "$PYTHON_BIN" scripts/train.py \
      --preset vgg16-paper-final \
      --data-root "$DATA_ROOT" \
      --output-dir "$GATE_DIR" \
      --stage1-weights "$OUTPUT/stage1/last.pt" \
      --fisher-init "$OUTPUT/fisher_gmm.pt" \
      --max-steps 500 \
      --max-val-batches 20 \
      --log-every-steps 20 \
      --finite-check-every-steps 20 \
      --num-workers "$WORKERS" \
      --no-progress \
      2>&1 | tee "$GATE_LOG"
    check_checkpoint "$GATE_DIR/last.pt" | tee -a "$GATE_LOG"
    ;;
  full)
    STAGE2_DIR="$OUTPUT/stage2"
    STAGE2_LOG="$OUTPUT/logs/stage2.log"
    if [[ -e "$STAGE2_DIR" ]]; then
      echo "Refusing to overwrite existing $STAGE2_DIR" >&2
      exit 2
    fi
    "$PYTHON_BIN" scripts/train.py \
      --preset vgg16-paper-final \
      --data-root "$DATA_ROOT" \
      --output-dir "$STAGE2_DIR" \
      --stage1-weights "$OUTPUT/stage1/last.pt" \
      --fisher-init "$OUTPUT/fisher_gmm.pt" \
      --log-every-steps 20 \
      --finite-check-every-steps 20 \
      --num-workers "$WORKERS" \
      --no-progress \
      2>&1 | tee "$STAGE2_LOG"
    check_checkpoint "$STAGE2_DIR/last.pt" | tee -a "$STAGE2_LOG"
    "$PYTHON_BIN" scripts/evaluate.py \
      --preset vgg16-paper-final \
      --data-root "$DATA_ROOT" \
      --checkpoint "$STAGE2_DIR/last.pt" \
      --fit-svm \
      --feature-cache-dir "$OUTPUT/eval_cache" \
      --metrics-output "$OUTPUT/final_metrics.json" \
      --num-workers "$WORKERS" \
      2>&1 | tee "$OUTPUT/logs/final_eval.log"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
