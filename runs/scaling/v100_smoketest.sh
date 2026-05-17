#!/bin/bash
# Single tiny training run to verify the V100 pipeline end-to-end before the
# full sweep. Should complete in well under a minute on a V100. Emits the
# usual stats and writes its log to $NANOCHAT_BASE_DIR/scaling_v100/smoke.log
# Re-run as you like.

set -uo pipefail
export OMP_NUM_THREADS=1
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"   # V100 tensor cores
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
if [ -z "${NANOCHAT_DATA_DIR:-}" ] && [ -d "$NANOCHAT_BASE_DIR/base_data_clean" ]; then
    export NANOCHAT_DATA_DIR="$NANOCHAT_BASE_DIR/base_data_clean"
fi
mkdir -p "$NANOCHAT_BASE_DIR/scaling_v100"

source .venv/bin/activate

LOG="$NANOCHAT_BASE_DIR/scaling_v100/smoke.log"
echo "[smoke] writing log to $LOG"
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=4 \
    --num-iterations=20 \
    --target-param-data-ratio=-1 \
    --target-flops=-1 \
    --window-pattern=L \
    --run=dummy \
    --model-tag=smoke_d4 \
    --eval-every=10 \
    --eval-tokens=$((524288)) \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --device-batch-size=8 \
    --total-batch-size=32768 \
    2>&1 | tee "$LOG"

echo "[smoke] done"
grep -E "Validation bpb:|total |Total batch|tok/s|step" "$LOG" | tail -20
