#!/usr/bin/env bash
# HP tuning v2: stacked combinations of best HPs from v1.
# E5 = B + EMA 0.999 + cosine LR
# E6 = B + EMA 0.99 + cosine LR
# E7 = B + EMA 0.999 + cosine LR + GRAD_CLIP 1.0
# Total: ~1.5 hours.
set -e
cd "$(dirname "$0")/.."
mkdir -p autoresearch/validations
BUDGET=1800

run_one() {
    local name=$1; shift
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Running $name  (budget=${BUDGET}s, full dataset)"
    echo "  ENV: $@"
    echo "═══════════════════════════════════════════════════════════"
    log="autoresearch/validations/run_${name}.log"
    env "$@" \
        PYTHONUNBUFFERED=1 \
        CONFIG=B \
        TRAIN_BUDGET_SEC_OVERRIDE=$BUDGET \
        FULL_DATA=1 \
        python autoresearch/train.py > "$log" 2>&1 || echo "  → FAILED, see $log"
    echo "  Result:"
    grep -E "^score:|^seg_term:|^pose_term:|^rate_term:|^model_bytes:|^total_epochs:" "$log" | sed 's/^/    /'
}

run_one E5 EMA_DECAY=0.999  COSINE_LR=1
run_one E6 EMA_DECAY=0.99   COSINE_LR=1
run_one E7 EMA_DECAY=0.9999 COSINE_LR=1

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  HP TUNING v2 SUMMARY"
echo "═══════════════════════════════════════════════════════════"
printf "%-8s %-10s %-10s %-10s %-10s %-10s\n" "config" "score" "seg" "pose" "rate" "epochs"
for cfg in E1 E2 E3 E4 E5 E6 E7; do
    log="autoresearch/validations/run_${cfg}.log"
    [ -f "$log" ] || continue
    score=$(grep "^score:" "$log" | awk '{print $2}')
    seg=$(grep "^seg_term:" "$log" | awk '{print $2}')
    pose=$(grep "^pose_term:" "$log" | awk '{print $2}')
    rate=$(grep "^rate_term:" "$log" | awk '{print $2}')
    epochs=$(grep "^total_epochs:" "$log" | awk '{print $2}')
    printf "%-8s %-10s %-10s %-10s %-10s %-10s\n" "$cfg" "${score:-FAIL}" "${seg:-?}" "${pose:-?}" "${rate:-?}" "${epochs:-?}"
done
