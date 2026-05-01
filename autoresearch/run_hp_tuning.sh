#!/usr/bin/env bash
# HP screening: 4 candidates @ 30min each on full data, all derived from CONFIG=B (winner).
# Total runtime: ~2 hours.
#
# E1: B + EMA 0.99
# E2: B + EMA 0.999
# E3: B + cosine LR (all stages, decays to 0.1x base)
# E4: B + GRAD_CLIP 1.0
set -e
cd "$(dirname "$0")/.."

mkdir -p autoresearch/validations
BUDGET=1800  # 30 minutes

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

run_one E1 EMA_DECAY=0.99
run_one E2 EMA_DECAY=0.999
run_one E3 COSINE_LR=1
run_one E4 GRAD_CLIP_OVERRIDE=1.0

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  HP TUNING SUMMARY (vs B@1hr=0.747; B equivalent at 30min ≈ ?)"
echo "═══════════════════════════════════════════════════════════"
printf "%-8s %-10s %-10s %-10s %-10s %-10s\n" "config" "score" "seg" "pose" "rate" "epochs"
for cfg in E1 E2 E3 E4; do
    log="autoresearch/validations/run_${cfg}.log"
    score=$(grep "^score:" "$log" | awk '{print $2}')
    seg=$(grep "^seg_term:" "$log" | awk '{print $2}')
    pose=$(grep "^pose_term:" "$log" | awk '{print $2}')
    rate=$(grep "^rate_term:" "$log" | awk '{print $2}')
    epochs=$(grep "^total_epochs:" "$log" | awk '{print $2}')
    printf "%-8s %-10s %-10s %-10s %-10s %-10s\n" "$cfg" "${score:-FAIL}" "${seg:-?}" "${pose:-?}" "${rate:-?}" "${epochs:-?}"
done
echo ""
echo "(For reference, B at 1hr scored 0.747. At 30min the baseline B should score ~0.85-0.90.)"
