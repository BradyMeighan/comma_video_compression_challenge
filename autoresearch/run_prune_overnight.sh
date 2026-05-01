#!/usr/bin/env bash
# Overnight 3090 work: train a baseline model, then explore pruning methods.
# Total runtime: ~6.5 hours.
#
# Phase 1 (~1.5h): train baseline with our best config (CONFIG=B + EMA 0.99 + cosine LR)
# Phase 2 (~5h): explore 6 pruning methods, save results to CSV
#
# Note: model trained in 1.5h won't be fully converged (Colab gets ~8h on A100 = ~16h-3090-equiv).
# That's OK — we're testing pruning METHODS, not absolute scores. Best method here will be best
# method on the converged Colab model too.
set -e
cd "$(dirname "$0")/.."

OUT=autoresearch/overnight_3090
mkdir -p "$OUT"

echo "═══════════════════════════════════════════════════════════"
echo "  PHASE 1: Train baseline model (~1.5h)"
echo "═══════════════════════════════════════════════════════════"
echo "  Config: CONFIG=B, EMA=0.99, cosine LR, full data"
echo "  Save:   $OUT/gen.pt"
echo ""

env CONFIG=B \
    EMA_DECAY=0.99 \
    COSINE_LR=1 \
    FULL_DATA=1 \
    TRAIN_BUDGET_SEC_OVERRIDE=5400 \
    SAVE_MODEL_PATH="$OUT/gen.pt" \
    CHECKPOINT_INTERVAL_SEC=1800 \
    PYTHONUNBUFFERED=1 \
    python autoresearch/train.py 2>&1 | tee "$OUT/train.log"

if [ ! -f "$OUT/gen.pt" ]; then
    echo "ERROR: training failed, no gen.pt produced. Aborting."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  PHASE 2: Pruning exploration (~5h)"
echo "═══════════════════════════════════════════════════════════"
echo "  Model:  $OUT/gen.pt"
echo "  Output: $OUT/prune_results.csv"
echo "  Methods: static / per-layer / channel / N:M / random / iterative+retrain"
echo ""

env MODEL_PATH="$OUT/gen.pt" \
    OUTPUT_DIR="$OUT" \
    FULL_DATA=1 \
    ITER_RETRAIN_SEC=1200 \
    PYTHONUNBUFFERED=1 \
    python autoresearch/prune_explore.py 2>&1 | tee "$OUT/prune.log"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  DONE. Results in $OUT/prune_results.csv"
echo "═══════════════════════════════════════════════════════════"
