#!/bin/bash
# v3 Run A' on A100 80GB — same recipe as v100_d12_3e18_code.sh but
# adapted for native bf16 + larger device batch.
#
# Hardware: A100-SXM4-80GB (SM 8.0). Native bf16 tensor cores +
# Flash Attention 2. Expected ~5-7h for 3e18 / d=12 (vs ~16h on V100).
#
# Key changes vs the V100 variant:
#   - NANOCHAT_DTYPE=bfloat16 (native + numerically stable for long runs)
#   - device-batch-size=64 (4 grad-accum steps vs 32 on V100, far less
#     per-step overhead)
#   - No byted proxy env vars (A100 has direct HF access)

set -uo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_TOKENIZER_DIR="${NANOCHAT_TOKENIZER_DIR:-$NANOCHAT_BASE_DIR/tokenizer_v8k}"
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$NANOCHAT_BASE_DIR/base_data_clean}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

source .venv/bin/activate

TAG="${TAG:-a100_emerge_v8k_3e18_d12_code}"
FLOPS="${FLOPS:-3e18}"
DEVICE_BATCH="${DEVICE_BATCH:-64}"

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
log "v3 Run A' (A100): $TAG"
log "  FLOPs           : $FLOPS"
log "  dtype           : $NANOCHAT_DTYPE"
log "  device_batch    : $DEVICE_BATCH"
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
    --device-batch-size="$DEVICE_BATCH" \
    2>&1 | tee "$LOG_TRAIN"
TRAIN_TIME=$(( $(date +%s) - TR_START ))

log "eval CORE for $TAG"
EV_START=$(date +%s)
python -m scripts.base_eval \
    --model-tag="$TAG" \
    --eval=core,bpb,sample \
    --max-per-task=200 \
    --device-batch-size=32 \
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
echo "${FLOPS}_a100,12,768,${P_TO},${NITERS},${TOKENS},${VAL_BPB},${TR_BPB},${CORE:-0},${TRAIN_TIME},${EVAL_TIME}" >> "$RESULTS_DIR/emerge.csv"

column -t -s',' "$RESULTS_DIR/emerge.csv"
