#!/bin/bash
# Final emergence run on V100 with vocab=8K + expanded data corpus.
#
# Background: the 4-run probe (3e16/1e17 d=4, 1e17/3e17 d=6) plateaued at
# CORE +0.077 because, with only 8 shards (~440M tokens), the bigger
# budgets fell into a multi-epoch overtraining regime. This script:
#
#   1. Merges newly-downloaded shards from base_data_climbmix_extra/ into
#      base_data_clean/ while *preserving* the original val shard
#      (renamed to shard_99999 so the alphabetical "last = val" rule
#      keeps the same validation set across runs).
#   2. Optionally cleans the new raw shards using the same pipeline.
#   3. Launches a single 1e18 d=8 run — the largest budget that fits
#      the remaining 24h budget on a single V100.
#   4. Evaluates CORE/bpb/samples and appends a row to emerge.csv.
#
# Re-runnable. The merge step is idempotent.

set -uo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_TOKENIZER_DIR="${NANOCHAT_TOKENIZER_DIR:-$NANOCHAT_BASE_DIR/tokenizer_v8k}"
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_clean}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
: "${http_proxy:=http://sys-proxy-rd-relay.byted.org:8118}"
: "${https_proxy:=http://sys-proxy-rd-relay.byted.org:8118}"
: "${NO_PROXY:=localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8}"
export http_proxy https_proxy NO_PROXY

source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_v100_emerge"
CLEAN_DIR="$NANOCHAT_DATA_DIR"
RAW_EXTRA="$NANOCHAT_BASE_DIR/base_data_climbmix_extra"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# Step 1: protect the existing val shard. The dataloader treats the
# alphabetically-last parquet as val. Rename our current val shard_00010
# to shard_99999 once so it always remains val even after we add 11+.
if [ -f "$CLEAN_DIR/shard_00010.parquet" ] && [ ! -f "$CLEAN_DIR/shard_99999.parquet" ]; then
    log "renaming clean shard_00010 -> shard_99999 (val pin)"
    mv "$CLEAN_DIR/shard_00010.parquet" "$CLEAN_DIR/shard_99999.parquet"
fi

# Step 2: clean any new raw shards from base_data_climbmix_extra/ into
# base_data_clean/. We pipe through the same data_pipeline filters.
if [ -d "$RAW_EXTRA" ] && [ "$(ls -A "$RAW_EXTRA" 2>/dev/null | grep -c '\.parquet$' || true)" -gt 0 ]; then
    log "cleaning extra shards from $RAW_EXTRA into $CLEAN_DIR"
    python -m scripts.scaling.data_pipeline clean \
        --in "$RAW_EXTRA" \
        --out "$CLEAN_DIR" 2>&1 | tee "$RESULTS_DIR/clean_extra.log" | tail -20
fi

# Step 3: count effective tokens after merge
N_TRAIN=$(ls "$CLEAN_DIR"/*.parquet 2>/dev/null | grep -v shard_99999 | wc -l)
log "$N_TRAIN train shards in $CLEAN_DIR (val pinned at shard_99999.parquet)"

# Step 4: run the final big training.
TAG="v100_emerge_v8k_1e18_d8_xdata"
LOG_TRAIN="$RESULTS_DIR/train_${TAG}.log"
LOG_EVAL="$RESULTS_DIR/eval_${TAG}.log"

if grep -q ",${TAG}," "$RESULTS_DIR/emerge.csv" 2>/dev/null \
   || grep -q "^1e18,8," "$RESULTS_DIR/emerge.csv" 2>/dev/null; then
    log "$TAG already in emerge.csv — skipping training"
else
    log "================================================"
    log "train $TAG (1e18 FLOPs, d=8, vocab=8K, extended data)"
    log "================================================"
    TR_START=$(date +%s)
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
        --depth=8 \
        --target-flops=1e18 \
        --target-param-data-ratio=-1 \
        --window-pattern=L \
        --run=dummy \
        --model-tag="$TAG" \
        --eval-tokens=$((5 * 524288)) \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --device-batch-size=8 \
        2>&1 | tee "$LOG_TRAIN"
    TRAIN_TIME=$(( $(date +%s) - TR_START ))

    log "================================================"
    log "eval CORE for $TAG"
    log "================================================"
    EV_START=$(date +%s)
    python -m scripts.base_eval \
        --model-tag="$TAG" \
        --eval=core,bpb,sample \
        --max-per-task=200 \
        --device-batch-size=8 \
        --split-tokens=$((5 * 524288)) \
        2>&1 | tee "$LOG_EVAL"
    EVAL_TIME=$(( $(date +%s) - EV_START ))

    # Parse metrics
    P_TO=$(grep "^total " "$LOG_TRAIN" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
    NITERS=$(grep "Calculated number of iterations" "$LOG_TRAIN" | tail -1 | sed 's/.*: //' | tr -d ',')
    BSIZE=$(grep "Total batch size" "$LOG_TRAIN" | tail -1 | grep -oP 'Total batch size \K[\d,]+' | tr -d ',')
    TOKENS=$((NITERS * BSIZE))
    CORE=$(grep "^CORE metric:" "$LOG_EVAL" | tail -1 | awk '{print $NF}')
    VAL_BPB=$(grep "^val bpb:" "$LOG_EVAL" | tail -1 | awk '{print $NF}')
    TR_BPB=$(grep "^train bpb:" "$LOG_EVAL" | tail -1 | awk '{print $NF}')

    log "RESULT: $TAG params=$P_TO tokens=$TOKENS val_bpb=$VAL_BPB CORE=$CORE train=${TRAIN_TIME}s eval=${EVAL_TIME}s"
    echo "1e18,8,512,${P_TO},${NITERS},${TOKENS},${VAL_BPB},${TR_BPB},${CORE:-0},${TRAIN_TIME},${EVAL_TIME}" >> "$RESULTS_DIR/emerge.csv"
fi

log "final emerge results:"
column -t -s',' "$RESULTS_DIR/emerge.csv"
