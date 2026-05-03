#!/bin/bash
# Phase 2: research-level approaches. Waits for phase 1 (master_overnight.sh) to finish.
# Then runs O1, O3, O4, O2 sequentially. Then enters continuous loop to pick up
# any explore_x*.py the user adds.

LOG_DIR="autoresearch/sidecar_results"
MASTER_LOG="${LOG_DIR}/master_overnight.log"
PHASE2_LOG="${LOG_DIR}/phase2_runner.log"

echo "[phase2] starting at $(date)" > "$PHASE2_LOG"

wait_for_phase1() {
  while ! grep -q "ALL DONE" "$MASTER_LOG" 2>/dev/null; do
    sleep 60
  done
  echo "[phase2] phase1 done at $(date)" >> "$PHASE2_LOG"
}

wait_for_python_done() {
  while tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q "python.exe"; do
    sleep 30
  done
  echo "[phase2] no python at $(date)" >> "$PHASE2_LOG"
}

run_exp() {
  local script="$1"
  local logname="${script%.py}"
  echo "[phase2] === ${script} START $(date) ===" >> "$PHASE2_LOG"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    MODEL_PATH=autoresearch/colab_run/gen_continued.pt \
    python "autoresearch/${script}" 2>&1 | tee "${LOG_DIR}/${logname}.log"
  echo "[phase2] === ${script} DONE $(date) ===" >> "$PHASE2_LOG"
}

# Wait for phase 1 to be completely done
echo "[phase2] waiting for phase 1 to finish..." >> "$PHASE2_LOG"
wait_for_phase1
wait_for_python_done

# Stage 5: research-level approaches in priority order
run_exp "explore_o1_gumbel.py"
wait_for_python_done

run_exp "explore_o3_hessian.py"
wait_for_python_done

run_exp "explore_o4_admm.py"
wait_for_python_done

run_exp "explore_o2_cmaes_mask.py"
wait_for_python_done

echo "[phase2] STAGE 5 COMPLETE at $(date)" >> "$PHASE2_LOG"

# Stage 6: continuous loop — keep running any new explore_x*.py user adds
echo "[phase2] entering continuous mode" >> "$PHASE2_LOG"
while true; do
  for script in autoresearch/explore_x*.py; do
    if [ -f "$script" ]; then
      sname=$(basename "$script")
      if ! grep -q "$sname DONE" "$PHASE2_LOG" 2>/dev/null; then
        run_exp "$sname"
        wait_for_python_done
      fi
    fi
  done
  sleep 300
done
