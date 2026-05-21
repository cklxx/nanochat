#!/bin/bash
# Resume an A100 training run after a Colab session restart, using the
# Drive backup at /content/drive/MyDrive/nanochat_a100/.
#
# Prereqs (do these on the fresh container before running this script):
#   1. Mount Drive in the Colab notebook:
#        from google.colab import drive; drive.mount('/content/drive')
#   2. Get SSH access to the container.
#   3. Clone the repo:
#        cd /root && git clone https://github.com/cklxx/nanochat.git
#        cd /root/nanochat && git checkout scaling-law-v100-validation
#   4. Install deps:
#        uv sync
#        uv pip install nvidia-cusparselt-cu12 nvidia-nvshmem-cu12 \
#            nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 \
#            nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
#            nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 \
#            hf_transfer
#
# Then: bash runs/scaling/a100_resume_from_drive.sh
#
# This script will:
#   1. Pull tokenizer back from Drive
#   2. Pull the most recent checkpoint back from Drive
#   3. Re-download data shards (faster than rsyncing 38 GB through Drive)
#   4. Clean (10-worker parallel for prose, single thread for code)
#   5. Re-arm the watchdog
#   6. Resume training with --init-from-checkpoint-tag pointing at the
#      restored checkpoint

set -uo pipefail
DRIVE=/content/drive/MyDrive/nanochat_a100
CACHE=/root/.cache/nanochat
PY=/root/nanochat/.venv/bin/python
LOG=$CACHE/scaling_v100_emerge
mkdir -p $LOG

log() { echo "[ $(date '+%H:%M:%S') ] $1"; }

# 0. sanity
test -d "$DRIVE" || { echo "Drive not mounted at $DRIVE"; exit 1; }
test -d "$DRIVE/tokenizer_v8k" || { echo "Drive tokenizer missing"; exit 1; }

# 1. tokenizer
log "restore tokenizer"
mkdir -p $CACHE/tokenizer_v8k
rsync -a $DRIVE/tokenizer_v8k/ $CACHE/tokenizer_v8k/

# 2. checkpoint
log "restore latest checkpoint(s)"
mkdir -p $CACHE/base_checkpoints
rsync -a $DRIVE/checkpoints/ $CACHE/base_checkpoints/
LATEST_TAG=$(ls -dt $CACHE/base_checkpoints/*/ 2>/dev/null | head -1 | xargs -I{} basename {})
log "  latest checkpoint tag: ${LATEST_TAG:-NONE}"

# 3. data re-download (CPU/network-bound, much faster than rsync from Drive)
log "downloading prose (397 shards)"
NANOCHAT_DATA_DIR=$CACHE/base_data_climbmix HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 \
    $PY -m nanochat.dataset -n 397 > $LOG/resume_download_prose.log 2>&1 &
prose_pid=$!

log "downloading code (3 passes in parallel)"
for tuple in "40 0 0" "80 40 200000" "80 120 500000"; do
    read NS SI SD <<< "$tuple"
    NANOCHAT_DATA_DIR=$CACHE/base_data_code_raw HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 \
        $PY -m scripts.scaling.download_code \
        --dataset codeparrot/codeparrot-clean-train --config "" --text-column content \
        --num-shards $NS --start-idx $SI --skip-docs $SD \
        > $LOG/resume_download_code_si${SI}.log 2>&1 &
done

# wait for all downloads (poll by shard count, not pgrep — race-safe)
log "waiting for downloads to land..."
while true; do
    n_prose=$(ls $CACHE/base_data_climbmix/ 2>/dev/null | wc -l)
    n_code=$(ls $CACHE/base_data_code_raw/ 2>/dev/null | wc -l)
    if [ "$n_prose" -ge 398 ] && [ "$n_code" -ge 200 ]; then break; fi
    log "  prose=$n_prose/398 code=$n_code/200"
    sleep 30
done

# 4. clean
log "cleaning prose (10 workers)"
NANOCHAT_BASE_DIR=$CACHE PYTHONUNBUFFERED=1 \
    $PY -m scripts.scaling.data_pipeline clean \
    --in $CACHE/base_data_climbmix --out $CACHE/base_data_clean \
    --workers 10 > $LOG/resume_clean_prose.log 2>&1
log "  prose cleaned: $(ls $CACHE/base_data_clean/shard_[0-9]*.parquet 2>/dev/null | wc -l) shards"

log "cleaning code"
NANOCHAT_BASE_DIR=$CACHE PYTHONUNBUFFERED=1 \
    $PY -m scripts.scaling.data_pipeline clean \
    --in $CACHE/base_data_code_raw --out $CACHE/base_data_clean \
    --code > $LOG/resume_clean_code.log 2>&1
log "  code cleaned: $(ls $CACHE/base_data_clean/shard_code_*.parquet 2>/dev/null | wc -l) shards"

# 5. re-arm watchdog (idempotent — kill any old one first)
pkill -f "rsync.*drive/MyDrive/nanochat_a100" 2>/dev/null
sleep 1
nohup bash -c '
while true; do
    rsync -a /root/.cache/nanochat/base_checkpoints/ /content/drive/MyDrive/nanochat_a100/checkpoints/ 2>/dev/null
    rsync -a /root/.cache/nanochat/scaling_v100_emerge/ /content/drive/MyDrive/nanochat_a100/logs/ 2>/dev/null
    rsync -a /root/.cache/nanochat/base_eval/ /content/drive/MyDrive/nanochat_a100/results/ 2>/dev/null
    rsync -a /root/.cache/nanochat/scaling_v100/ /content/drive/MyDrive/nanochat_a100/data_report/ 2>/dev/null
    sleep 300
done
' > /content/drive/MyDrive/nanochat_a100/watchdog.log 2>&1 &
log "watchdog PID=$!"

# 6. relaunch training from checkpoint if found, else from scratch
if [ -n "${LATEST_TAG:-}" ]; then
    log "resuming training from $LATEST_TAG (--init-from-checkpoint-tag)"
    cd /root/nanochat
    INIT_TAG=$LATEST_TAG TAG=${LATEST_TAG}_resume \
        nohup bash runs/scaling/a100_d12_3e18_code.sh \
        > $LOG/train_a100_resume.log 2>&1 &
else
    log "no checkpoint found — starting fresh"
    cd /root/nanochat
    nohup bash runs/scaling/a100_d12_3e18_code.sh > $LOG/train_a100.log 2>&1 &
fi
log "training PID=$!"
sleep 10
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
log "RESUME COMPLETE"
