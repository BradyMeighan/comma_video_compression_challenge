#!/bin/bash
# Overnight orchestrator: run experiments sequentially.

LOG_DIR="autoresearch/sidecar_results"
echo "[overnight] starting at $(date)" > "$LOG_DIR/overnight_master.log"

wait_for_python_done() {
  while tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q "python.exe"; do
    sleep 30
  done
  echo "[overnight] no python at $(date)" >> "$LOG_DIR/overnight_master.log"
}

run_exp() {
  local script="$1"
  local extra_env="$2"
  local logname="${3:-$(basename $script .py)}"
  echo "[overnight] === $script start at $(date) ===" >> "$LOG_DIR/overnight_master.log"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    MODEL_PATH=autoresearch/colab_run/gen_continued.pt \
    $extra_env \
    python "autoresearch/$script" 2>&1 | tee "$LOG_DIR/${logname}.log"
  echo "[overnight] === $script done at $(date) ===" >> "$LOG_DIR/overnight_master.log"
}

# Wait for currently-running push.py to finish
wait_for_python_done

# Stage 1: ITERATIVE mask <-> RGB
run_exp "sidecar_iterative.py"
wait_for_python_done

# Stage 2: CHANNEL-ONLY RGB patches
run_exp "sidecar_channel_only.py"
wait_for_python_done

# Stage 3: POSE INPUT patches
run_exp "sidecar_pose_input.py"
wait_for_python_done

# Stage 4: bigger n_candidates for mask
run_exp "sidecar_combined_lean.py" "N_CANDIDATES=20" "lean_n20"
wait_for_python_done

run_exp "sidecar_combined_lean.py" "N_CANDIDATES=30" "lean_n30"
wait_for_python_done

echo "[overnight] ALL DONE at $(date)" >> "$LOG_DIR/overnight_master.log"
