#!/bin/bash
# Warm-restart from gen_3090.pt.e80.ckpt (current best 0.2988).
# Lower LR (2e-6 vs prior 1e-5), bigger batch (8 vs 4), lower pose weight (30 vs 60),
# checkpoint every 5 epochs (more granular for fine-tuning gains).
# 2h budget — plenty for fine-tuning since gains will be small.

set -uo pipefail

OUT_DIR="autoresearch/colab_run/3090_run_v2"
SOURCE_CKPT="autoresearch/colab_run/3090_run/gen_3090.pt.e80.ckpt"
mkdir -p "$OUT_DIR"

if [ ! -f "$SOURCE_CKPT" ]; then
  echo "FATAL: $SOURCE_CKPT not found" | tee "${OUT_DIR}/runner.log"
  exit 1
fi

echo "[v2] start $(date), source=$SOURCE_CKPT" | tee "${OUT_DIR}/runner.log"

# launch watcher in background
WATCH_DIR="$OUT_DIR" RESULTS_CSV="${OUT_DIR}/eval_log.csv" POLL_SEC=60 \
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  python autoresearch/eval_watcher.py > "${OUT_DIR}/watcher.log" 2>&1 &
WATCHER_PID=$!
echo "[v2] watcher PID=$WATCHER_PID" | tee -a "${OUT_DIR}/runner.log"

# warm-restart training
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  MODEL_PATH="$SOURCE_CKPT" \
  SAVE_MODEL_PATH="${OUT_DIR}/gen_3090_v2.pt" \
  TRAIN_BUDGET_SEC_OVERRIDE=7200 \
  JT_LR_OVERRIDE=2e-6 \
  BATCH_SIZE=4 \
  CHECKPOINT_INTERVAL_SEC=600 \
  CHECKPOINT_EPOCH_INTERVAL=5 \
  EMA_DECAY=0.9995 \
  COSINE_LR=1 \
  POSE_WEIGHT=30 \
  GRAD_CLIP_OVERRIDE=0.5 \
  python autoresearch/continue_train.py 2>&1 | tee "${OUT_DIR}/train.log"

touch "${OUT_DIR}/STOP_WATCHER"
sleep 3
kill "$WATCHER_PID" 2>/dev/null || true

echo "[v2] done $(date)" | tee -a "${OUT_DIR}/runner.log"
echo "[v2] eval_log:"
cat "${OUT_DIR}/eval_log.csv"
