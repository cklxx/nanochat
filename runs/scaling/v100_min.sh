#!/bin/bash
# Minimum-scale scaling law validation on a single Tesla V100-SXM2-32GB.
#
# Why these settings:
#   - V100 is SM 7.0: no FP8 (Hopper) and no bf16 tensor cores (Ampere+).
#     nanochat's autodetect would fall back to FP32. We override with
#     NANOCHAT_DTYPE=float16 because V100 has *fp16 tensor cores*
#     (~125 TFLOPS peak vs ~15 TFLOPS FP32) and the training loop already
#     wires up `torch.amp.GradScaler()` whenever COMPUTE_DTYPE==float16.
#     This is what lets the sweep finish on a single V100 in ~hours.
#   - No Flash Attention 3 (Hopper-only). We force window_pattern=L so SDPA
#     is used on a full-context causal mask (sliding-window fallback is much
#     slower).
#   - Single GPU --nproc_per_node=1, gradient accumulation is automatic.
#   - We sweep depth in {4,6,8,10,12} at four IsoFLOP budgets to mirror the
#     Chinchilla-style methodology used by the original runs/scaling_laws.sh,
#     but with budgets ~3 orders of magnitude smaller so it actually finishes
#     on one V100 within ~ a few hours.
#
# Inputs:
#   - $NANOCHAT_BASE_DIR must contain base_data_clean/ with at least
#     a handful of parquet shards (the data pipeline produces those).
#   - .venv must be set up via uv (the script does NOT install deps).
#
# Outputs (all under $NANOCHAT_BASE_DIR/scaling_v100/):
#   results.csv             — one row per finished run
#   run_<tag>_train.log     — full stdout of each training run
#   manifest.json           — config used for the sweep
#
# Re-runnable: rows already in results.csv are skipped.

set -uo pipefail

LABEL="v100_min"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
WANDB_RUN="${WANDB_RUN:-dummy}"

# IsoFLOP design (very small budgets so each fits on one V100 in FP32):
FLOPS_BUDGETS=(
    1e15
    3e15
    1e16
    3e16
)
DEPTHS=(4 6 8 10 12)

# Evaluation budget (val tokens). 5 * 524288 ≈ 2.6M tokens — enough for stable
# val_bpb at this scale, cheap enough not to dominate training time.
EVAL_TOKENS=$((5 * 524288))

# Single GPU on V100 has 32GB. With FP16 activations, batches can be bigger.
device_batch_for_depth() {
    local d=$1
    if   [ "$d" -le 6  ]; then echo 16
    elif [ "$d" -le 10 ]; then echo 8
    else echo 4
    fi
}

export OMP_NUM_THREADS=1
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"   # V100 tensor cores
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
# If a cleaned data dir exists, prefer it so training runs on the curated
# corpus. Otherwise fall back to the raw shards.
if [ -z "${NANOCHAT_DATA_DIR:-}" ] && [ -d "$NANOCHAT_BASE_DIR/base_data_clean" ]; then
    export NANOCHAT_DATA_DIR="$NANOCHAT_BASE_DIR/base_data_clean"
fi

# proxy (only used if env not already set)
: "${http_proxy:=http://sys-proxy-rd-relay.byted.org:8118}"
: "${https_proxy:=http://sys-proxy-rd-relay.byted.org:8118}"
: "${NO_PROXY:=localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8}"
export http_proxy https_proxy NO_PROXY

source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_v100"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,model_dim,params_wte,params_value_embeds,params_lm_head,params_transformer,params_scalars,params_total,num_iterations,tokens_trained,val_bpb,train_time_sec" > "$RESULTS_FILE"
fi

cat > "$RESULTS_DIR/manifest.json" <<JSON
{
  "label": "$LABEL",
  "gpu": "Tesla V100-SXM2-32GB (single)",
  "compute_dtype": "fp16 (NANOCHAT_DTYPE=float16; V100 tensor cores via GradScaler)",
  "window_pattern": "L",
  "flops_budgets": [$(IFS=,; echo "${FLOPS_BUDGETS[*]}")],
  "depths": [$(IFS=,; echo "${DEPTHS[*]}")],
  "eval_tokens": $EVAL_TOKENS,
  "nproc_per_node": $NPROC_PER_NODE
}
JSON

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }
run_exists() {
    local flops=$1 depth=$2
    grep -q "^${flops},${depth}," "$RESULTS_FILE" 2>/dev/null
}

for flops in "${FLOPS_BUDGETS[@]}"; do
    log "================================================"
    log "Compute budget: $flops FLOPs"
    log "================================================"
    for d in "${DEPTHS[@]}"; do
        if run_exists "$flops" "$d"; then
            log "skip d=$d at $flops (already in results)"
            continue
        fi
        TAG="${LABEL}_${flops}_d${d}"
        DBS=$(device_batch_for_depth "$d")
        log "train d=$d at $flops, device_batch=$DBS"

        START=$(date +%s)
        # Note: --target-param-data-ratio=-1 disables data:param mode so that
        # target-flops controls the horizon. window-pattern=L disables sliding
        # window (no FA3 on V100). No --fp8.
        torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
            --depth="$d" \
            --target-flops="$flops" \
            --target-param-data-ratio=-1 \
            --window-pattern="L" \
            --run="${WANDB_RUN}_${TAG}" \
            --model-tag="$TAG" \
            --eval-tokens="$EVAL_TOKENS" \
            --core-metric-every=-1 \
            --sample-every=-1 \
            --save-every=-1 \
            --device-batch-size="$DBS" \
            2>&1 | tee "$RESULTS_DIR/run_${TAG}.log"
        END=$(date +%s)
        TRAIN_TIME=$((END - START))

        LOG="$RESULTS_DIR/run_${TAG}.log"
        P_WTE=$(grep "^wte " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        P_VE=$(grep "^value_embeds " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        P_LM=$(grep "^lm_head " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        P_TR=$(grep "^transformer_matrices " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        P_SC=$(grep "^scalars " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        P_TO=$(grep "^total " "$LOG" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        NITERS=$(grep "Calculated number of iterations" "$LOG" | tail -1 | sed 's/.*: //' | tr -d ',')
        BSIZE=$(grep "Total batch size" "$LOG" | tail -1 | grep -oP 'Total batch size \K[\d,]+' | tr -d ',')
        TOKENS=$((NITERS * BSIZE))
        MODEL_DIM=$((d * 64))
        VAL_BPB=$(grep "Validation bpb:" "$LOG" | tail -1 | grep -oP '[\d.]+$')

        log "  d=$d done — params=$P_TO iters=$NITERS tokens=$TOKENS val_bpb=$VAL_BPB time=${TRAIN_TIME}s"
        echo "$flops,$d,$MODEL_DIM,$P_WTE,$P_VE,$P_LM,$P_TR,$P_SC,$P_TO,$NITERS,$TOKENS,$VAL_BPB,$TRAIN_TIME" >> "$RESULTS_FILE"
    done
done

log "================================================"
log "v100 scaling sweep complete — results at $RESULTS_FILE"
log "================================================"
column -t -s',' "$RESULTS_FILE"
