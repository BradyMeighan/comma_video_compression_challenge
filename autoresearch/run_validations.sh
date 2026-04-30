#!/usr/bin/env bash
# Validation runner: 4 candidate configs at 1-hour budget on full 500/100 dataset.
# Total runtime: ~4 hours sequentially.
#
# Configs:
#   A: HEAD baseline (current arch, no algorithmic changes)
#   B: + boundary-weighted focal loss in anchor + Lion optimizer
#   C: joint-from-epoch-0 (single stage) + boundary-weighted focal
#   D: + per-dim normalized pose MSE (replaces smooth_l1)
set -e
cd "$(dirname "$0")/.."

mkdir -p autoresearch/validations
BUDGET=3600  # 1 hour

for cfg in A B C D; do
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Running CONFIG=$cfg  (budget=${BUDGET}s, full dataset)"
    echo "═══════════════════════════════════════════════════════════"
    log="autoresearch/validations/run_${cfg}.log"
    PYTHONUNBUFFERED=1 \
        CONFIG=$cfg \
        TRAIN_BUDGET_SEC_OVERRIDE=$BUDGET \
        FULL_DATA=1 \
        python autoresearch/train.py > "$log" 2>&1 || echo "  → FAILED, see $log"
    echo "  Result:"
    grep -E "^score:|^seg_term:|^pose_term:|^rate_term:|^model_bytes:|^total_epochs:" "$log" | sed 's/^/    /'
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════════════════════"
printf "%-8s %-10s %-10s %-10s %-10s\n" "config" "score" "seg" "pose" "rate"
for cfg in A B C D; do
    log="autoresearch/validations/run_${cfg}.log"
    score=$(grep "^score:" "$log" | awk '{print $2}')
    seg=$(grep "^seg_term:" "$log" | awk '{print $2}')
    pose=$(grep "^pose_term:" "$log" | awk '{print $2}')
    rate=$(grep "^rate_term:" "$log" | awk '{print $2}')
    printf "%-8s %-10s %-10s %-10s %-10s\n" "$cfg" "${score:-FAIL}" "${seg:-?}" "${pose:-?}" "${rate:-?}"
done
