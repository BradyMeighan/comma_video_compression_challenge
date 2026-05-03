#!/bin/bash
# Master overnight orchestrator: runs all explore_*.py experiments sequentially
# after the baseline patches are built. Updates OVERNIGHT_RESEARCH_LOG.md.

LOG_DIR="autoresearch/sidecar_results"
MASTER_LOG="${LOG_DIR}/master_overnight.log"
RESEARCH_DOC="OVERNIGHT_RESEARCH_LOG.md"

echo "[master] starting at $(date)" > "$MASTER_LOG"

wait_for_python_done() {
  while tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q "python.exe"; do
    sleep 30
  done
  echo "[master] no python at $(date)" >> "$MASTER_LOG"
}

run_exp() {
  local script="$1"
  local logname="${script%.py}"
  echo "[master] === ${script} START $(date) ===" >> "$MASTER_LOG"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    MODEL_PATH=autoresearch/colab_run/gen_continued.pt \
    python "autoresearch/${script}" 2>&1 | tee "${LOG_DIR}/${logname}.log"
  echo "[master] === ${script} DONE $(date) ===" >> "$MASTER_LOG"
}

# Wait for baseline builder to finish
wait_for_python_done

# Stage 1: ENTROPY CODING (highest expected impact, cheapest to test)
run_exp "explore_e1_ans.py"
wait_for_python_done

run_exp "explore_e2_zstd.py"
wait_for_python_done

run_exp "explore_e3_zopfli.py"
wait_for_python_done

# Stage 2: ALGORITHMIC patch experiments (don't depend on each other)
run_exp "explore_a4_dim_bits.py"     # quick: just analyzes existing patches
wait_for_python_done

run_exp "explore_a5_diff_coords.py"  # quick: just re-encodes existing
wait_for_python_done

run_exp "explore_a2_3chan_scoring.py"   # re-finds RGB with 3-channel scoring
wait_for_python_done

run_exp "explore_a3_compound.py"        # re-finds with compound channel patches
wait_for_python_done

run_exp "explore_a1_subpixel.py"        # re-finds with sub-pixel coords
wait_for_python_done

# Stage 3: ARCHITECTURAL changes (slowest, biggest potential)
run_exp "explore_m1_codebook.py"        # k-means cross-pair codebook
wait_for_python_done

# Stage 4: TINY MLP (most ambitious, only if time permits) — placeholder
if [ -f "autoresearch/explore_m2_tiny_mlp.py" ]; then
  run_exp "explore_m2_tiny_mlp.py"
  wait_for_python_done
fi

echo "[master] ALL DONE at $(date)" >> "$MASTER_LOG"
