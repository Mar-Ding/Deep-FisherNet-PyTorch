#!/usr/bin/env bash
# Full ResNet101 training (official-like): Train → Multi-scale eval → SVM
set -euo pipefail

OUTPUT="outputs/res101_full"
EPOCHS=4

echo "=== ResNet101 FisherNet training ($EPOCHS epochs) ==="
python3 scripts/train.py \
    --preset official-res101-like \
    --data-root data \
    --output-dir "$OUTPUT" \
    --fisher-init outputs/official_res101_like/fisher_gmm.pt \
    --pretrained --epochs $EPOCHS \
    --num-workers 4

echo ""
echo "=== Single-scale evaluation ==="
python3 scripts/evaluate.py \
    --preset official-res101-like \
    --data-root data \
    --checkpoint "$OUTPUT/best.pt" \
    --num-workers 4

echo ""
echo "=== Multi-scale evaluation ==="
python3 scripts/evaluate.py \
    --preset official-res101-like \
    --data-root data \
    --checkpoint "$OUTPUT/best.pt" \
    --test-scales 336 504 672 840 1008 \
    --num-workers 4

echo ""
echo "=== SVM post-processing ==="
python3 scripts/evaluate.py \
    --preset official-res101-like \
    --data-root data \
    --checkpoint "$OUTPUT/best.pt" \
    --test-scales 336 504 672 840 1008 \
    --fit-svm --svm-C 1.0 \
    --num-workers 4

echo ""
echo "=== Done! ==="
echo "Checkpoint: $OUTPUT/best.pt"
