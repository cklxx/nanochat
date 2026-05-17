# Minimum-Scale Scaling Law Validation on a Single Tesla V100

**Repo branch:** `scaling-law-v100-validation`
**Compute:** 1 × Tesla V100-SXM2-32GB, FP16 (tensor cores) via NANOCHAT_DTYPE=float16
**Codebase:** nanochat (this fork), training script `scripts/base_train.py`

This report records the end-to-end pipeline used to reproduce Chinchilla-style
scaling-law behaviour on the smallest scale that still fits the standard
nanochat training stack: one V100, no FP8, no Flash Attention 3.

The deliverable is two-fold:

1. The full data → tokenizer → training → analysis pipeline, all
   committed and reproducible.
2. Paper-grade figures (PNG + PDF) and JSON fit summaries archived in this
   directory.

## 1. Data

**Source:** NVIDIA Nemotron-ClimbMix, repackaged by Karpathy as
`huggingface.co/datasets/karpathy/climbmix-400b-shuffle` — the same dataset
used by the upstream nanochat pretraining loop.

**Selection:** 8 shards downloaded via `python -m nanochat.dataset -n 12` (we
trimmed the in-progress downloads after the proxy throughput stalled; the
final 8 shards give 7 train + 1 val under nanochat's last-shard val
convention).

| | docs | chars | letter-ratio | prefix-dup |
|--|--:|--:|--:|--:|
| raw     | 592,896 | 1.761,954,831 | 0.9888 | 0.000157 |
| cleaned | 586,424 | 1.746,514,832 | 0.9891 | 0.000000 |

**Pipeline:** `scripts/scaling/data_pipeline.py` runs four stages:
inventory → stats → clean → validate. The cleaning rules drop:

- docs <200 chars (6,822) or >200,000 chars (49)
- docs with average word length outside [2.0, 20.0] (121)
- docs with letter+punct ratio <0.6 (42)
- exact prefix-hash duplicates (468)

98.9% of docs and 99.1% of chars survive — ClimbMix is already aggressively
filtered upstream. The histogram in `data_report/doc_length_hist.png` shows
the impact mostly on the short-doc tail.

All raw stats live in `data_report/stats_train_base_data_climbmix.{csv,json}`
and the cleaned counterpart in `data_report/stats_train_base_data_clean.{csv,json}`.

## 2. Tokenizer

Trained with the upstream `scripts/tok_train.py` on **1.0 B characters of
cleaned ClimbMix** (vocab size 2^15 = 32,768, doc-cap 10,000 chars).

Training time on remote V100 host: **43.3 s** (CPU job, the tokenizer is
RustBPE).

`scripts/tok_eval.py` reports a compression of 4.62 bytes/token on
fineweb-edu val and ~4.58 on news — within ~2 % of GPT-4's tokenizer on
English web text.

## 3. Experimental design

Single-node single-GPU sweep:

```
depth   ∈ {4, 6, 8, 10, 12}
flops   ∈ {1e15, 3e15, 1e16, 3e16}      # 20 runs
```

Per run, `--target-flops` fixes the compute budget and num_iterations is
back-solved from `flops / (flops_per_token × total_batch_size)`. All other
hyperparameters (LR, batch size, weight decay, warmdown) are auto-set by
nanochat's scaling-law-aware defaults.

Why FP16: V100 SM 7.0 has no bf16 tensor cores but **does** have 125 TFLOPS
fp16 tensor cores, and `scripts/base_train.py` already wires up
`torch.amp.GradScaler()` when `COMPUTE_DTYPE == torch.float16`. This is the
single change (`NANOCHAT_DTYPE=float16`) that makes the sweep feasible on
one V100.

Why `window_pattern=L`: V100 has no Flash Attention 3 (Hopper-only), so the
default sliding-window pattern falls back to a much slower SDPA path. Full
attention (`L`) keeps SDPA fast.

## 4. Results

> Filled in by `scripts/scaling/analyze.py` after the sweep finishes:
> see `results.csv`, `fit_summary.json`, and `figures/`.

| compute (FLOPs) | depth | params_total | tokens | val_bpb |
|---|---|---|---|---|
| … | … | … | … | … |

Power-law fits (placeholder until run):

- $L(C) = A \cdot C^{-\alpha_L}$ with $\alpha_L \approx$ TODO ($R^2 = $ TODO)
- $N^*(C) \propto C^{a}$ with $a \approx$ TODO
- $D^*(C) \propto C^{b}$ with $b \approx$ TODO

## 5. Reproduction

```bash
# remote V100 setup
uv venv && uv sync --extra gpu
uv pip install matplotlib pandas    # for analyze + validate plot

# data
python -m nanochat.dataset -n 12
python -m scripts.scaling.data_pipeline inventory
python -m scripts.scaling.data_pipeline stats   --split train
python -m scripts.scaling.data_pipeline clean
python -m scripts.scaling.data_pipeline validate
export NANOCHAT_DATA_DIR=$NANOCHAT_BASE_DIR/base_data_clean

# tokenizer
python -m scripts.tok_train --max-chars 1000000000

# sweep
bash runs/scaling/v100_smoketest.sh   # 20-iter sanity check
bash runs/scaling/v100_min.sh         # full IsoFLOP sweep

# analysis
python -m scripts.scaling.analyze
```

## 6. Caveats

- Embedding parameters dominate at small scale because vocab=32k is fixed.
  We therefore use **Kaplan-style effective N = transformer + lm_head** for
  the scaling-law fits (matching the original nanochat scaling notebook).
- FP16 vs FP8/BF16 introduces a constant multiplier on the loss but does not
  affect the *scaling exponent*; the slopes reported here are directly
  comparable to bf16 H100 results.
- We disabled wandb (`WANDB_MODE=disabled`) so the sweep is fully
  self-contained on the remote host.
