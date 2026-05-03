#!/bin/bash
# v3 — warm restart from latest ckpt (epoch 16) with INLINE EVAL every 10 epochs.
# No separate watcher process — eval is called inside the training loop, blocking,
# so no GPU contention. Eval log written to {SAVE_MODEL_PATH}.eval_log.csv.

set -uo pipefail

OUT_DIR="autoresearch/colab_run/3090_run_v3"
SOURCE_CKPT="autoresearch/colab_run/3090_run/gen_3090.pt.e80.ckpt"
mkdir -p "$OUT_DIR"

if [ ! -f "$SOURCE_CKPT" ]; then
  echo "FATAL: $SOURCE_CKPT not found" | tee "${OUT_DIR}/runner.log"
  exit 1
fi

echo "[v3] start $(date), source=$SOURCE_CKPT" | tee "${OUT_DIR}/runner.log"

# Inline-eval training: NO separate watcher.
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  MODEL_PATH="$SOURCE_CKPT" \
  SAVE_MODEL_PATH="${OUT_DIR}/gen_3090_v3.pt" \
  TRAIN_BUDGET_SEC_OVERRIDE=5400 \
  JT_LR_OVERRIDE=2e-6 \
  BATCH_SIZE=4 \
  CHECKPOINT_INTERVAL_SEC=600 \
  CHECKPOINT_EPOCH_INTERVAL=10 \
  EVAL_EPOCH_INTERVAL=10 \
  EMA_DECAY=0.9995 \
  COSINE_LR=1 \
  POSE_WEIGHT=30 \
  GRAD_CLIP_OVERRIDE=0.5 \
  python autoresearch/continue_train.py 2>&1 | tee "${OUT_DIR}/train.log"

echo "[v3] done $(date)" | tee -a "${OUT_DIR}/runner.log"
echo "[v3] eval_log:"
cat "${OUT_DIR}/gen_3090_v3.pt.eval_log.csv" 2>/dev/null
