#!/bin/bash
# 3090 continuation training — 4 hours, JT_LR=1e-5 (10x lower than yesterday's H100 overshoot),
# bs=4 (3090-safe), checkpoint every 10 epochs. Eval watcher runs in parallel.
#
# Output: autoresearch/colab_run/3090_run/
#   gen_3090.pt              — final model
#   gen_3090.pt.e{N}.ckpt    — every 10 epochs
#   gen_3090.pt.ckpt         — latest (overwrites)
#   train.log                — training stdout
#   eval_log.csv             — per-checkpoint score from watcher
#   watcher.log              — watcher stdout

set -uo pipefail

OUT_DIR="autoresearch/colab_run/3090_run"
mkdir -p "$OUT_DIR"

echo "[3090] start $(date)" | tee "${OUT_DIR}/runner.log"

# launch watcher in background first so it catches checkpoints as they appear
WATCH_DIR="$OUT_DIR" RESULTS_CSV="${OUT_DIR}/eval_log.csv" POLL_SEC=60 \
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  python autoresearch/eval_watcher.py > "${OUT_DIR}/watcher.log" 2>&1 &
WATCHER_PID=$!
echo "[3090] watcher PID=$WATCHER_PID" | tee -a "${OUT_DIR}/runner.log"

# launch training (foreground in this script)
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  MODEL_PATH=autoresearch/colab_run/gen_continued.pt \
  SAVE_MODEL_PATH="${OUT_DIR}/gen_3090.pt" \
  TRAIN_BUDGET_SEC_OVERRIDE=14400 \
  JT_LR_OVERRIDE=1e-5 \
  BATCH_SIZE=4 \
  CHECKPOINT_INTERVAL_SEC=600 \
  CHECKPOINT_EPOCH_INTERVAL=10 \
  EMA_DECAY=0.999 \
  COSINE_LR=1 \
  POSE_WEIGHT=60 \
  GRAD_CLIP_OVERRIDE=0.5 \
  python autoresearch/continue_train.py 2>&1 | tee "${OUT_DIR}/train.log"

# tell the watcher to exit cleanly
touch "${OUT_DIR}/STOP_WATCHER"
sleep 3
kill "$WATCHER_PID" 2>/dev/null || true

echo "[3090] done $(date)" | tee -a "${OUT_DIR}/runner.log"
echo "[3090] eval_log:"
cat "${OUT_DIR}/eval_log.csv"
