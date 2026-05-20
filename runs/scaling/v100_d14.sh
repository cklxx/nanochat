#!/bin/bash
# v3 Run B — depth scan: 1.5e18 FLOPs at d=14 from scratch.
#
# Motivation: cont2 (RESULTS_cont2.md) showed that stacking continued
# pretraining rounds saturates and then reverses CORE despite improving
# val_bpb. The v3 plan (RECIPE_v3.md §5) therefore prioritises a depth
# scan over more same-data compute.
#
# Configuration:
#   - depth=14, vocab=8K  (~200M params, +49% over d=12)
#   - target_flops = 1.5e18 (same compute budget as v2 best so v1↔v2↔v3
#     are directly comparable on the FLOPs axis)
#   - corpus = whatever is in $NANOCHAT_DATA_DIR — by the time this runs
#     after extra3 cleanup we expect ~196 train shards ≈ 4.6 B tokens
#   - model-tag = v100_emerge_v8k_1.5e18_d14
#
# Why d=14 specifically:
#   v2 IsoFLOP fit puts N*(1.5e18) above 100M params; cont1 confirmed
#   135M was already in the right neighbourhood. d=14 (200M) lets us
#   measure whether N* keeps shifting up with compute or has plateaued.
#
# Why from scratch (not from xdata ckpt):
#   d=12 weights can't be reshaped into d=14, and the cont chain proved
#   negative anyway. Fresh training is the only clean signal for
#   "does deeper help".
#
# Expected wallclock: ~7h on a single V100 (more FLOPs/token than d=12
# but fewer iterations: 2463 vs 3697).

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

TAG="${TAG:-v100_emerge_v8k_1.5e18_d14}"
FLOPS="${FLOPS:-1.5e18}"
DEPTH="${DEPTH:-14}"
DBS="${DBS:-4}"

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_v100_emerge"
mkdir -p "$RESULTS_DIR"
LOG_TRAIN="$RESULTS_DIR/train_${TAG}.log"
LOG_EVAL="$RESULTS_DIR/eval_${TAG}.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

if grep -q ",${TAG}," "$RESULTS_DIR/emerge.csv" 2>/dev/null; then
    log "$TAG already in emerge.csv — skipping"
    exit 0
fi

n_train_shards=$(ls "$NANOCHAT_DATA_DIR"/*.parquet 2>/dev/null | grep -v shard_99999 | wc -l)
log "================================================"
log "v3 Run B — depth scan: $TAG"
log "  depth        : $DEPTH"
log "  FLOPs        : $FLOPS"
log "  device batch : $DBS"
log "  train dir    : $NANOCHAT_DATA_DIR ($n_train_shards train shards + 1 val pin)"
log "================================================"

TR_START=$(date +%s)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth="$DEPTH" \
    --target-flops="$FLOPS" \
    --target-param-data-ratio=-1 \
    --window-pattern=L \
    --run=dummy \
    --model-tag="$TAG" \
    --eval-tokens=$((5 * 524288)) \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --device-batch-size="$DBS" \
    2>&1 | tee "$LOG_TRAIN"
TRAIN_TIME=$(( $(date +%s) - TR_START ))

log "eval CORE for $TAG"
EV_START=$(date +%s)
python -m scripts.base_eval \
    --model-tag="$TAG" \
    --eval=core,bpb,sample \
    --device-batch-size="$DBS" \
    2>&1 | tee "$LOG_EVAL"
EVAL_TIME=$(( $(date +%s) - EV_START ))

log "done. train=${TRAIN_TIME}s eval=${EVAL_TIME}s"
log "next: append a row to $RESULTS_DIR/emerge.csv and write RESULTS_d14.md"
