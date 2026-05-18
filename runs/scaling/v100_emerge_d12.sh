#!/bin/bash
# Sideways bet: d=12 at 1.5e18 FLOPs with vocab=8K + extended corpus.
#
# Reasoning: 1e18 d=8 hit CORE +0.119. Pushing the same model to 3e18
# would be too long (15h+) and probably hit the same multi-epoch ceiling
# we just escaped. Going bigger (d=12, ~104M total params, ~91M effective)
# at a moderate compute bump (1.5e18) probes a different angle of the
# (N, D) plane — more parameters, fewer epochs. 22-shard corpus (~1.3B
# tokens) keeps the data:param ratio reasonable at ~30:1.
#
# Expected: ~8h training + eval, fits the remaining 15h V100 budget.

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
mkdir -p "$RESULTS_DIR"
TAG="v100_emerge_v8k_1.5e18_d12_xdata"
LOG_TRAIN="$RESULTS_DIR/train_${TAG}.log"
LOG_EVAL="$RESULTS_DIR/eval_${TAG}.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

if grep -q ",${TAG}," "$RESULTS_DIR/emerge.csv" 2>/dev/null; then
    log "$TAG already in emerge.csv — skipping"
    exit 0
fi

log "================================================"
log "train $TAG (1.5e18 FLOPs, d=12, vocab=8K, 22-shard corpus)"
log "================================================"
TR_START=$(date +%s)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --target-flops=1.5e18 \
    --target-param-data-ratio=-1 \
    --window-pattern=L \
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
echo "1.5e18,12,768,${P_TO},${NITERS},${TOKENS},${VAL_BPB},${TR_BPB},${CORE:-0},${TRAIN_TIME},${EVAL_TIME}" >> "$RESULTS_DIR/emerge.csv"

column -t -s',' "$RESULTS_DIR/emerge.csv"
