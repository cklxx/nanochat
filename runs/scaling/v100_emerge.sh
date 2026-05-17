#!/bin/bash
# Progressive emergence sweep on top of the vocab=8K tokenizer.
#
# Goal: find the *minimum* compute at which the DCLM CORE metric crosses
# +0.20 (3× our previous best, ~80 % of GPT-2 1.6B's 0.256). The starting
# point is the previous sweep's best run (3e16 FLOPs d=4, CORE=+0.072 with
# vocab=32K). We rerun that point with the smaller vocab as a fresh
# baseline, then progressively raise the compute budget along the d=4
# frontier (with d=6 / d=8 sanity points at the largest budgets).
#
# After each run, run CORE eval and append a row to emerge.csv. As soon as
# CORE crosses the threshold we touch a STOP file so the next iteration
# bails out.
#
# All artefacts live in $NANOCHAT_BASE_DIR/scaling_v100_emerge/.

set -uo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_TOKENIZER_DIR="${NANOCHAT_TOKENIZER_DIR:-$NANOCHAT_BASE_DIR/tokenizer_v8k}"
if [ -z "${NANOCHAT_DATA_DIR:-}" ] && [ -d "$NANOCHAT_BASE_DIR/base_data_clean" ]; then
    export NANOCHAT_DATA_DIR="$NANOCHAT_BASE_DIR/base_data_clean"
fi
export WANDB_MODE="${WANDB_MODE:-disabled}"
: "${http_proxy:=http://sys-proxy-rd-relay.byted.org:8118}"
: "${https_proxy:=http://sys-proxy-rd-relay.byted.org:8118}"
: "${NO_PROXY:=localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8}"
export http_proxy https_proxy NO_PROXY

source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_v100_emerge"
mkdir -p "$RESULTS_DIR"
EMERGE_CSV="$RESULTS_DIR/emerge.csv"
STOP_FILE="$RESULTS_DIR/STOP"
TARGET_CORE="${TARGET_CORE:-0.20}"

if [ ! -f "$EMERGE_CSV" ]; then
    echo "flops_budget,depth,model_dim,params_total,num_iterations,tokens_trained,val_bpb,train_bpb,core_metric,train_time_sec,eval_time_sec" > "$EMERGE_CSV"
fi

cat > "$RESULTS_DIR/manifest.json" <<JSON
{
  "label": "v100_emerge_v8k",
  "tokenizer_dir": "$NANOCHAT_TOKENIZER_DIR",
  "vocab_size": 8192,
  "target_core": $TARGET_CORE,
  "gpu": "Tesla V100-SXM2-32GB (single)",
  "compute_dtype": "fp16",
  "schedule": "progressive — stops when CORE >= TARGET_CORE"
}
JSON

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# device-batch picker (V100 32GB, fp16). Same as v100_min.sh but per the
# smaller embedding we can be a touch more generous at large depth.
device_batch_for_depth() {
    local d=$1
    if   [ "$d" -le 6  ]; then echo 16
    elif [ "$d" -le 10 ]; then echo 8
    else echo 4
    fi
}

# Each entry is "flops:depth". Strategy (from RECIPE.md): walk small
# budgets first to confirm the trend, then commit to big-compute runs once
# we know the slope. d=4 at 3e16 acts as a vocab-comparable baseline
# against the previous (vocab=32K) sweep.
SWEEP=(
  "3e16:4"     # ~13 min — vocab=8K baseline vs old sweep's d=4@3e16 (CORE +0.072)
  "1e17:4"     # ~40 min — first compute step up
  "1e17:6"     # ~40 min — does N* move once embedding is smaller?
  "3e17:6"     # ~1.5 h — most informative mid-budget
  "3e17:4"     # ~1.0 h — d=4 frontier at same budget
  "1e18:6"     # ~5 h   — big push #1
  "1e18:8"     # ~5 h   — big push #2, alt depth
  "3e18:6"     # ~15 h  — final stretch ONLY if STOP not yet triggered
)

eval_core() {
    local tag=$1 dbs=$2
    log "  eval CORE for $tag"
    python -m scripts.base_eval \
        --model-tag="$tag" \
        --eval=core,bpb,sample \
        --max-per-task=200 \
        --device-batch-size="$dbs" \
        --split-tokens=$((5 * 524288)) \
        2>&1 | tee "$RESULTS_DIR/eval_${tag}.log"
}

for entry in "${SWEEP[@]}"; do
    if [ -f "$STOP_FILE" ]; then
        log "STOP file present — emergence reached, exiting sweep"
        break
    fi
    flops="${entry%%:*}"
    depth="${entry##*:}"
    TAG="v100_emerge_v8k_${flops}_d${depth}"
    if grep -q ",${flops},${depth}," "$EMERGE_CSV" 2>/dev/null \
       || grep -q "^${flops},${depth}," "$EMERGE_CSV" 2>/dev/null; then
        log "skip $TAG (already in emerge.csv)"
        continue
    fi
    DBS=$(device_batch_for_depth "$depth")
    log "================================================"
    log "train  $TAG  (dbs=$DBS)"
    log "================================================"

    TR_START=$(date +%s)
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
        --depth="$depth" \
        --target-flops="$flops" \
        --target-param-data-ratio=-1 \
        --window-pattern="L" \
        --run=dummy \
        --model-tag="$TAG" \
        --eval-tokens=$((5 * 524288)) \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --device-batch-size="$DBS" \
        2>&1 | tee "$RESULTS_DIR/train_${TAG}.log" || { log "TRAIN FAILED $TAG"; continue; }
    TR_END=$(date +%s)
    TRAIN_TIME=$((TR_END - TR_START))

    LOG="$RESULTS_DIR/train_${TAG}.log"
    P_TO=$(grep "^total " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    NITERS=$(grep "Calculated number of iterations" "$LOG" | tail -1 | sed 's/.*: //' | tr -d ',')
    BSIZE=$(grep "Total batch size" "$LOG" | tail -1 | grep -oP 'Total batch size \K[\d,]+' | tr -d ',')
    TOKENS=$((NITERS * BSIZE))
    MODEL_DIM=$((depth * 64))
    VAL_BPB=$(grep "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')

    EV_START=$(date +%s)
    eval_core "$TAG" "$DBS"
    EV_END=$(date +%s)
    EVAL_TIME=$((EV_END - EV_START))

    ELOG="$RESULTS_DIR/eval_${TAG}.log"
    CORE=$(grep -E "^CORE metric:" "$ELOG" | tail -1 | awk '{print $NF}')
    EVAL_VAL=$(grep -E "^val bpb:" "$ELOG" | tail -1 | awk '{print $NF}')
    EVAL_TR=$(grep -E "^train bpb:" "$ELOG" | tail -1 | awk '{print $NF}')

    log "  -> $TAG : params=$P_TO  iters=$NITERS  tokens=$TOKENS  val_bpb=${EVAL_VAL:-$VAL_BPB}  CORE=$CORE  train=${TRAIN_TIME}s  eval=${EVAL_TIME}s"

    echo "${flops},${depth},${MODEL_DIM},${P_TO},${NITERS},${TOKENS},${EVAL_VAL:-$VAL_BPB},${EVAL_TR:-?},${CORE:-0},${TRAIN_TIME},${EVAL_TIME}" >> "$EMERGE_CSV"

    # Threshold check using awk for floating-point comparison
    if [ -n "$CORE" ] && awk "BEGIN{exit !($CORE >= $TARGET_CORE)}"; then
        log "🎯 CORE $CORE >= $TARGET_CORE — emergence reached. Writing STOP file."
        touch "$STOP_FILE"
    fi
done

log "================================================"
log "emerge sweep complete — emerge.csv at $EMERGE_CSV"
log "================================================"
column -t -s',' "$EMERGE_CSV"
