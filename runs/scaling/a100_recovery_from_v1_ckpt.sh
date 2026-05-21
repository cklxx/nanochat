#!/bin/bash
# A100 recovery #2 — init from V1 step 2000 ckpt.
set -uo pipefail
LOG=/root/.cache/nanochat/scaling_v100_emerge
mkdir -p $LOG

echo "[ $(date) ] step1: clone + sync repo"
cd /root
test -d nanochat || git clone https://github.com/cklxx/nanochat.git
cd /root/nanochat
git checkout scaling-law-v100-validation 2>&1 | tail -2
git pull 2>&1 | tail -2

echo "[ $(date) ] step2: uv sync + cuda libs + triton<3.6"
uv sync 2>&1 | tail -3
uv pip install nvidia-cusparselt-cu12 nvidia-nvshmem-cu12 nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 \
    hf_transfer "triton<3.6" 2>&1 | tail -3
.venv/bin/python -c "import torch, triton; print(torch.__version__, triton.__version__)"

echo "[ $(date) ] step3: restore tokenizer + V1 ckpts (has step 2000)"
mkdir -p /root/.cache/nanochat/tokenizer_v8k /root/.cache/nanochat/base_checkpoints
rsync -a /content/drive/MyDrive/nanochat_a100/tokenizer_v8k/ /root/.cache/nanochat/tokenizer_v8k/
rsync -a /content/drive/MyDrive/nanochat_a100/checkpoints/a100_emerge_v8k_3e18_d12_code/ \
    /root/.cache/nanochat/base_checkpoints/a100_emerge_v8k_3e18_d12_code/
ls /root/.cache/nanochat/base_checkpoints/a100_emerge_v8k_3e18_d12_code/ | grep model_

PY=/root/nanochat/.venv/bin/python
cd /root/nanochat

echo "[ $(date) ] step4: download data (4 parallel: 1 prose + 3 code passes)"
NANOCHAT_DATA_DIR=/root/.cache/nanochat/base_data_climbmix HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 \
    nohup $PY -m nanochat.dataset -n 397 > $LOG/v3_dl_prose.log 2>&1 &
for tuple in "40 0 0" "80 40 200000" "80 120 500000"; do
    read NS SI SD <<< "$tuple"
    NANOCHAT_DATA_DIR=/root/.cache/nanochat/base_data_code_raw HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 \
        nohup $PY -m scripts.scaling.download_code \
        --dataset codeparrot/codeparrot-clean-train --config "" --text-column content \
        --num-shards $NS --start-idx $SI --skip-docs $SD \
        > $LOG/v3_dl_code_si${SI}.log 2>&1 &
done

echo "[ $(date) ] step5: poll for downloads (by shard count, race-safe)"
while true; do
    p=$(ls /root/.cache/nanochat/base_data_climbmix/ 2>/dev/null | wc -l)
    c=$(ls /root/.cache/nanochat/base_data_code_raw/ 2>/dev/null | wc -l)
    if [ "$p" -ge 398 ] && [ "$c" -ge 200 ]; then break; fi
    echo "  prose=$p/398 code=$c/200"
    sleep 30
done
echo "[ $(date) ] downloads done"

echo "[ $(date) ] step6: clean prose (10 workers) + code (single thread)"
NANOCHAT_BASE_DIR=/root/.cache/nanochat PYTHONUNBUFFERED=1 \
    $PY -m scripts.scaling.data_pipeline clean \
    --in /root/.cache/nanochat/base_data_climbmix --out /root/.cache/nanochat/base_data_clean \
    --workers 10 > $LOG/v3_clean_prose.log 2>&1
NANOCHAT_BASE_DIR=/root/.cache/nanochat PYTHONUNBUFFERED=1 \
    $PY -m scripts.scaling.data_pipeline clean \
    --in /root/.cache/nanochat/base_data_code_raw --out /root/.cache/nanochat/base_data_clean \
    --code > $LOG/v3_clean_code.log 2>&1
echo "clean done. prose=$(ls /root/.cache/nanochat/base_data_clean/shard_[0-9]*.parquet | wc -l) code=$(ls /root/.cache/nanochat/base_data_clean/shard_code_*.parquet | wc -l)"

echo "[ $(date) ] step7: re-arm watchdog"
cat > /tmp/watchdog.sh << 'WATCHEOF'
#!/bin/bash
while true; do
    rsync -a /root/.cache/nanochat/base_checkpoints/ /content/drive/MyDrive/nanochat_a100/checkpoints/ 2>/dev/null
    rsync -a /root/.cache/nanochat/scaling_v100_emerge/ /content/drive/MyDrive/nanochat_a100/logs/ 2>/dev/null
    rsync -a /root/.cache/nanochat/base_eval/ /content/drive/MyDrive/nanochat_a100/results/ 2>/dev/null
    sleep 300
done
WATCHEOF
chmod +x /tmp/watchdog.sh
nohup /tmp/watchdog.sh > /content/drive/MyDrive/nanochat_a100/watchdog.log 2>&1 &
echo "watchdog PID=$!"

echo "[ $(date) ] step8: launch training (INIT_TAG = V1 step 2000)"
INIT_TAG=a100_emerge_v8k_3e18_d12_code TAG=a100_emerge_v8k_3e18_d12_code_resume2 \
    nohup bash runs/scaling/a100_d12_3e18_code.sh > $LOG/train_resume2.log 2>&1 &
TR_PID=$!
sleep 10
echo "training PID=$TR_PID"
ps -ef | grep base_train | grep -v grep | head -2
echo "[ $(date) ] RECOVERY COMPLETE"
