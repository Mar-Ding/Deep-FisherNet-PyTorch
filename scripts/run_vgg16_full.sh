#!/usr/bin/env bash
# Full VGG16 training (paper-like): Stage 1 → Stage 2 → Multi-scale eval → SVM
set -euo pipefail

OUTPUT="outputs/vgg16_full"

echo "=== Stage 1: Whole-image CNN finetune (9k iters ≈ 57 epochs) ==="
python3 scripts/train.py \
    --stage1 \
    --preset stage1-vgg16-paper \
    --data-root data \
    --output-dir "$OUTPUT/stage1" \
    --pretrained --num-workers 4

echo ""
echo "=== Stage 2: FisherNet end-to-end (40k iters ≈ 16 epochs) ==="
python3 scripts/train.py \
    --preset vgg16-paper-like \
    --data-root data \
    --output-dir "$OUTPUT/stage2" \
    --stage1-weights "$OUTPUT/stage1/last.pt" \
    --fisher-init outputs/vgg16-paper-like/fisher_gmm.pt \
    --pretrained --num-workers 4

echo ""
echo "=== Multi-scale evaluation (5 scales + flip) ==="
python3 scripts/evaluate.py \
    --preset vgg16-paper-like \
    --data-root data \
    --checkpoint "$OUTPUT/stage2/best.pt" \
    --test-scales 480 576 688 864 1200 \
    --num-workers 4

echo ""
echo "=== Full paper pipeline: multi-scale + SVM ==="
python3 scripts/evaluate.py \
    --preset vgg16-paper-like \
    --data-root data \
    --checkpoint "$OUTPUT/stage2/best.pt" \
    --test-scales 480 576 688 864 1200 \
    --fit-svm --svm-C 1.0 \
    --num-workers 4

echo ""
echo "=== Done! ==="
echo "Checkpoints: $OUTPUT/stage2/best.pt"
