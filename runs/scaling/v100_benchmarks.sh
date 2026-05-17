#!/bin/bash
# Post-sweep benchmarking: run base_eval on every model_tag produced by
# runs/scaling/v100_min.sh and collect CORE-metric + bpb + qualitative
# samples into a single CSV/log directory.
#
# This is what answers "is there any basic intelligence / reasoning emerging
# at this scale?" — by running the same DCLM CORE benchmark suite that
# nanochat uses as its primary capability measure.
#
# CORE is the centered/normalized accuracy across ~22 ICL tasks
# (ARC-{Easy,Challenge}, HellaSwag, OBQA, MMLU subsets, etc.).
# - random model => CORE ≈ 0
# - GPT-2 (1.6B) => CORE ≈ 0.256
# - GPT-3-davinci => CORE ≈ 0.5+

set -uo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
if [ -z "${NANOCHAT_DATA_DIR:-}" ] && [ -d "$NANOCHAT_BASE_DIR/base_data_clean" ]; then
    export NANOCHAT_DATA_DIR="$NANOCHAT_BASE_DIR/base_data_clean"
fi
export WANDB_MODE="${WANDB_MODE:-disabled}"
: "${http_proxy:=http://sys-proxy-rd-relay.byted.org:8118}"
: "${https_proxy:=http://sys-proxy-rd-relay.byted.org:8118}"
: "${NO_PROXY:=localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8}"
export http_proxy https_proxy NO_PROXY

source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_v100"
BENCH_DIR="$RESULTS_DIR/benchmarks"
mkdir -p "$BENCH_DIR"
BENCH_CSV="$BENCH_DIR/benchmarks.csv"

if [ ! -f "$BENCH_CSV" ]; then
    echo "flops_budget,depth,model_tag,step,val_bpb,core_metric,train_bpb,eval_time_sec" > "$BENCH_CSV"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }
exists() {
    local tag=$1
    grep -q ",${tag}," "$BENCH_CSV" 2>/dev/null
}

# Walk the results.csv to figure out which (flops, depth, tag) tuples to eval.
RESULTS_FILE="$RESULTS_DIR/results.csv"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "missing $RESULTS_FILE — run runs/scaling/v100_min.sh first" >&2
    exit 1
fi

# Match the device-batch picker in v100_min.sh
device_batch_for_depth() {
    local d=$1
    if   [ "$d" -le 6  ]; then echo 16
    elif [ "$d" -le 10 ]; then echo 8
    else echo 4
    fi
}

# Skip header, iterate
tail -n +2 "$RESULTS_FILE" | while IFS=, read -r flops depth model_dim p_wte p_ve p_lm p_tr p_sc p_to niters tokens val_bpb tt; do
    if [ -z "$val_bpb" ]; then
        log "skip empty val_bpb row for d=$depth flops=$flops"
        continue
    fi
    TAG="v100_min_${flops}_d${depth}"
    if exists "$TAG"; then
        log "skip $TAG (already in benchmarks.csv)"
        continue
    fi
    DBS=$(device_batch_for_depth "$depth")
    log "eval $TAG  (CORE + bpb + samples)"

    LOG="$BENCH_DIR/eval_${TAG}.log"
    START=$(date +%s)
    # Use --max-per-task=200 to cap CORE eval runtime; CORE is already centered/normalized
    # so per-task subsampling doesn't bias the metric much.
    python -m scripts.base_eval \
        --model-tag="$TAG" \
        --eval=core,bpb,sample \
        --max-per-task=200 \
        --device-batch-size="$DBS" \
        --split-tokens=$((5 * 524288)) \
        2>&1 | tee "$LOG" || { log "FAILED $TAG (see $LOG)"; continue; }
    END=$(date +%s)
    ELAPSED=$((END - START))

    # Extract metrics from log
    CORE=$(grep -E "CORE metric:" "$LOG" | tail -1 | awk '{print $NF}')
    VALBPB=$(grep -E "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')
    TRBPB=$(grep -E "Train bpb:|Training bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')
    STEP=$(grep -E "step.*\(100\.00%\)|Step .* | Validation bpb" "$LOG" | tail -1 | grep -oP '\d{5}' | head -1)
    [ -z "$STEP" ] && STEP=$(grep -oP 'step \K\d+' "$LOG" | tail -1)

    echo "${flops},${depth},${TAG},${STEP:-?},${VALBPB:-?},${CORE:-0},${TRBPB:-?},${ELAPSED}" >> "$BENCH_CSV"
    log "  -> CORE=$CORE val_bpb=$VALBPB time=${ELAPSED}s"
done

log "================================================"
log "benchmarks complete — results at $BENCH_CSV"
log "================================================"
column -t -s',' "$BENCH_CSV"
