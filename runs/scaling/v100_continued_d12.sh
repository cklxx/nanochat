#!/bin/bash
# Continued pretraining from the d=12 / 1.5e18 vocab=8K checkpoint.
#
# Strategy: initialize weights from the just-finished v100_emerge_v8k_1.5e18_d12_xdata
# checkpoint, but start a fresh Muon+AdamW optimizer and a fresh
# warmup→constant→warmdown schedule for the new training horizon. This is
# cleaner than --resume-from-step because the original run already
# fully warmed down to LR≈0; reloading the optimizer state would lock us
# in to that minimum, while a fresh schedule lets us bring LR back up and
# meaningfully train on new data.
#
# Configuration:
#   - depth=12, vocab=8K (same architecture)
#   - target_flops = 1.5e18 (a second "chinchilla unit" of compute,
#     ~5.5 h on a single V100 at ~61K tok/sec for d=12)
#   - corpus = whatever is currently in $NANOCHAT_DATA_DIR — should be
#     the expanded 80+ shard corpus the data subagent is preparing, so
#     this run has ≥5B fresh tokens to read.
#   - model-tag = v100_emerge_v8k_1.5e18_d12_cont
#
# After training we evaluate CORE+bpb+samples and append to emerge.csv.

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

INIT_TAG="${INIT_TAG:-v100_emerge_v8k_1.5e18_d12_xdata}"
TAG="${TAG:-v100_emerge_v8k_1.5e18_d12_cont}"
FLOPS="${FLOPS:-1.5e18}"

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_v100_emerge"
mkdir -p "$RESULTS_DIR"
LOG_TRAIN="$RESULTS_DIR/train_${TAG}.log"
LOG_EVAL="$RESULTS_DIR/eval_${TAG}.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

if grep -q ",${TAG}," "$RESULTS_DIR/emerge.csv" 2>/dev/null; then
    log "$TAG already in emerge.csv — skipping"
    exit 0
fi

# Pre-flight: make sure the seed checkpoint exists
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$INIT_TAG"
if [ ! -d "$CKPT_DIR" ]; then
    log "ERROR: seed checkpoint dir not found: $CKPT_DIR" >&2
    exit 1
fi

n_train_shards=$(ls "$NANOCHAT_DATA_DIR"/*.parquet 2>/dev/null | grep -v shard_99999 | wc -l)
log "================================================"
log "continued pretrain: $TAG"
log "  seed       : $INIT_TAG"
log "  FLOPs      : $FLOPS"
log "  train dir  : $NANOCHAT_DATA_DIR ($n_train_shards train shards + 1 val pin)"
log "================================================"

TR_START=$(date +%s)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --target-flops="$FLOPS" \
    --target-param-data-ratio=-1 \
    --window-pattern=L \
    --init-from-checkpoint-tag="$INIT_TAG" \
    --run=dummy \
    --model-tag="$TAG" \
    --eval-tokens=$((5 * 524288)) \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --device-batch-size=4 \
    2>&1 | tee "$LOG_TRAIN"
TRAIN_TIME=$(( $(date +%s) - TR_START ))

log "eval CORE for $TAG"
EV_START=$(date +%s)
python -m scripts.base_eval \
    --model-tag="$TAG" \
    --eval=core,bpb,sample \
    --max-per-task=200 \
    --device-batch-size=4 \
    --split-tokens=$((5 * 524288)) \
    2>&1 | tee "$LOG_EVAL"
EVAL_TIME=$(( $(date +%s) - EV_START ))

P_TO=$(grep "^total " "$LOG_TRAIN" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
NITERS=$(grep "Calculated number of iterations" "$LOG_TRAIN" | tail -1 | sed 's/.*: //' | tr -d ',')
BSIZE=$(grep "Total batch size" "$LOG_TRAIN" | tail -1 | grep -oP 'Total batch size \K[\d,]+' | tr -d ',')
TOKENS=$((NITERS * BSIZE))
CORE=$(grep "^CORE metric:" "$LOG_EVAL" | tail -1 | awk '{print $NF}')
VAL_BPB=$(grep "^val bpb:" "$LOG_EVAL" | tail -1 | awk '{print $NF}')
TR_BPB=$(grep "^train bpb:" "$LOG_EVAL" | tail -1 | awk '{print $NF}')

log "RESULT $TAG params=$P_TO tokens=$TOKENS val_bpb=$VAL_BPB CORE=$CORE train=${TRAIN_TIME}s eval=${EVAL_TIME}s"
# Use a label that captures both the seed and the new flops so v3 stays distinct in CSV
echo "${FLOPS}_cont,12,768,${P_TO},${NITERS},${TOKENS},${VAL_BPB},${TR_BPB},${CORE:-0},${TRAIN_TIME},${EVAL_TIME}" >> "$RESULTS_DIR/emerge.csv"

column -t -s',' "$RESULTS_DIR/emerge.csv"
