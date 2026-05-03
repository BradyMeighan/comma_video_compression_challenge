#!/bin/bash
# v2 runner — orchestrates the four v2 sidecar experiments.
#   STAGE 0: cache builder (X2 + CMA-ES + frames + RGB) — run ONCE
#   C1:      LZMA2 / cross-stream compression (CPU only, fastest)
#   C3:      pose-vector deltas on top 200 (medium GPU)
#   S2:      variable-pattern CMA-ES (heavier GPU)
#   C2:      3x3 blocks on X2-residual (heaviest GPU)
#
# All experiments load from v2_cache/ to skip ~50 min of redundant X2/CMA-ES per run.

set -uo pipefail

LOG_DIR="autoresearch/sidecar_results"
RUNNER_LOG="${LOG_DIR}/v2_runner.log"
CACHE_DIR="${LOG_DIR}/v2_cache"
mkdir -p "$LOG_DIR"

echo "[v2] starting at $(date)" | tee "$RUNNER_LOG"

wait_for_python_done() {
  while tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q "python.exe"; do
    sleep 15
  done
  echo "[v2] no python at $(date)" >> "$RUNNER_LOG"
}

run_exp() {
  local script="$1"
  local logname="${script%.py}"
  local logfile="${LOG_DIR}/${logname}.log"
  echo "[v2] === ${script} START $(date) ===" | tee -a "$RUNNER_LOG"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    MODEL_PATH=autoresearch/colab_run/gen_continued.pt \
    OUTPUT_DIR="$LOG_DIR" \
    python "autoresearch/${script}" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "[v2] !!! ${script} FAILED rc=${rc} at $(date)" | tee -a "$RUNNER_LOG"
  else
    echo "[v2] === ${script} DONE $(date) ===" | tee -a "$RUNNER_LOG"
  fi
}

if [ ! -f "${LOG_DIR}/baseline_patches.pkl" ]; then
  echo "[v2] FATAL: ${LOG_DIR}/baseline_patches.pkl missing — run baseline first" | tee -a "$RUNNER_LOG"
  exit 1
fi

# Stage 0 — cache builder (X2 + CMA-ES + frames + RGB)
if [ ! -f "${CACHE_DIR}/cache_meta.json" ]; then
  run_exp "v2_cache_builder.py"
  wait_for_python_done
else
  echo "[v2] === v2_cache_builder.py SKIP (cache present) ===" | tee -a "$RUNNER_LOG"
fi

# Stage 1 — CPU-only LZMA2 / cross-stream compression test
if [ ! -f "${LOG_DIR}/v2_c1_lzma2_results.csv" ]; then
  run_exp "v2_c1_lzma2.py"
  wait_for_python_done
else
  echo "[v2] === v2_c1_lzma2.py SKIP (results already present) ===" | tee -a "$RUNNER_LOG"
fi

# Stage 2 — pose-vector deltas (cached X5)
run_exp "v2_c3_pose_vector.py"
wait_for_python_done

# Stage 3 — variable-pattern CMA-ES (cached X2)
run_exp "v2_s2_strip_cmaes.py"
wait_for_python_done

# Stage 4 — 3x3 blocks on X2-residual (cached X2 + cmaes)
run_exp "v2_c2_3x3_residual.py"
wait_for_python_done

echo "[v2] === SUMMARY $(date) ===" | tee -a "$RUNNER_LOG"
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  OUTPUT_DIR="$LOG_DIR" \
  python autoresearch/v2_summary.py 2>&1 | tee -a "$RUNNER_LOG"

echo "[v2] ALL DONE at $(date)" | tee -a "$RUNNER_LOG"
