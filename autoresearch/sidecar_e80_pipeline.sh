#!/bin/bash
# Run the full sidecar pipeline against e80 ckpt:
#   1. v2_cache_builder.py — builds X2 + CMA-ES + RGB cache
#   2. v2_per_pair_select.py — per-pair selection + LZMA2 wrap
# Both use MODEL_PATH=e80 ckpt.

set -uo pipefail

LOG_DIR="autoresearch/sidecar_results"
RUNNER_LOG="${LOG_DIR}/sidecar_e80_pipeline.log"
MODEL_E80="autoresearch/colab_run/3090_run/gen_3090.pt.e80.ckpt"

if [ ! -f "$MODEL_E80" ]; then
  echo "FATAL: $MODEL_E80 not found" | tee "$RUNNER_LOG"
  exit 1
fi

echo "[e80] start $(date), model=$MODEL_E80" | tee "$RUNNER_LOG"

run_step() {
  local script="$1"
  local logname="${script%.py}_e80"
  local logfile="${LOG_DIR}/${logname}.log"
  echo "[e80] === ${script} START $(date) ===" | tee -a "$RUNNER_LOG"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    MODEL_PATH="$MODEL_E80" \
    OUTPUT_DIR="$LOG_DIR" \
    python "autoresearch/${script}" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "[e80] !!! ${script} FAILED rc=${rc}" | tee -a "$RUNNER_LOG"
    return 1
  fi
  echo "[e80] === ${script} DONE $(date) ===" | tee -a "$RUNNER_LOG"
}

run_step "v2_cache_builder.py" || exit 1
run_step "v2_per_pair_select.py" || exit 1

echo "[e80] ALL DONE $(date)" | tee -a "$RUNNER_LOG"
echo "=== final result ===" | tee -a "$RUNNER_LOG"
cat "${LOG_DIR}/v2_per_pair_select_results.csv" | tee -a "$RUNNER_LOG"
