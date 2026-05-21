#!/bin/bash
# v3 Run A': 3e18 FLOPs, d=12, fp16, from scratch on prose+code mix.
#
# This run pushes the **FLOPs axis** — the only "same architecture,
# change one thing" axis we haven't validated yet (depth was flat at
# 1.5e18, continued pretraining saturates after one round, data
# doubling is at the ceiling). We also add ~10% Python code shards to
# the corpus to see if a small code injection helps capability.
#
# Configuration:
#   - depth=12, hidden=768, vocab=8K (same arch as the v2 baseline)
#   - target_flops = 3e18 (2× our prior largest run)
#   - corpus = $NANOCHAT_BASE_DIR/base_data_clean
#       * 396 prose train shards (~9.4 B tokens of ClimbMix prose)
#       * + N code shards (codeparrot Python, cleaned via --code mode)
#       * + 1 val shard (pinned to shard_99999)
#   - tokenizer = unchanged v8k (code compresses ~3× worse vs prose
#     but we keep tokenizer constant to isolate the FLOPs axis)
#   - mid-train diagnostic: CORE@50 every 500 steps, save every 1000
#   - model-tag = v100_emerge_v8k_3e18_d12_code
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

TAG="${TAG:-v100_emerge_v8k_3e18_d12_code}"
FLOPS="${FLOPS:-3e18}"

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_v100_emerge"
mkdir -p "$RESULTS_DIR"
LOG_TRAIN="$RESULTS_DIR/train_${TAG}.log"
LOG_EVAL="$RESULTS_DIR/eval_${TAG}.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

if grep -q ",${TAG}," "$RESULTS_DIR/emerge.csv" 2>/dev/null; then
    log "$TAG already in emerge.csv — skipping"
    exit 0
fi

n_prose=$(ls "$NANOCHAT_DATA_DIR"/shard_[0-9]*.parquet 2>/dev/null | grep -v shard_99999 | wc -l)
n_code=$(ls "$NANOCHAT_DATA_DIR"/shard_code_*.parquet 2>/dev/null | wc -l)
log "================================================"
log "v3 Run A' (FLOPs axis): $TAG"
log "  FLOPs           : $FLOPS"
log "  data dir        : $NANOCHAT_DATA_DIR"
log "  prose shards    : $n_prose"
log "  code shards     : $n_code"
log "================================================"

TR_START=$(date +%s)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --target-flops="$FLOPS" \
    --target-param-data-ratio=-1 \
    --window-pattern=L \
    --run=dummy \
    --model-tag="$TAG" \
    --eval-tokens=$((5 * 524288)) \
    --core-metric-every=500 \
    --core-metric-max-per-task=50 \
    --sample-every=-1 \
    --save-every=1000 \
    --device-batch-size=8 \
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
echo "${FLOPS}_code,12,768,${P_TO},${NITERS},${TOKENS},${VAL_BPB},${TR_BPB},${CORE:-0},${TRAIN_TIME},${EVAL_TIME}" >> "$RESULTS_DIR/emerge.csv"

column -t -s',' "$RESULTS_DIR/emerge.csv"
