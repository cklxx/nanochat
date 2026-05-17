# V100 minimum-scale scaling law validation

Reproducible recipe for validating Chinchilla-style scaling on a **single**
Tesla V100-SXM2-32GB, using only nanochat's own training loop. Everything is
committed so it can be re-run end-to-end.

## What we run

Sweep over depth ∈ {4, 6, 8, 10, 12} at four compute budgets
C ∈ {3e15, 1e16, 3e16, 1e17} FLOPs (IsoFLOP design).

Per run we record total params, transformer/embedding/lm_head breakdown,
iteration count, total tokens, train wallclock, and final `val_bpb`.

## Why this size

V100 lacks both FP8 (Hopper-only) and Flash Attention 3, so:

- `--window-pattern L` (no sliding window — SDPA fallback is fine on full attn)
- bf16 compute dtype (V100 supports it; throughput < H100 but works)
- single-GPU `torchrun --nproc_per_node=1` with gradient accumulation
- conservative `--device-batch-size` per depth (16/8/4)

Each run completes in seconds to a few minutes; the full sweep finishes well
inside one V100-day.

## Data

Uses the existing `nanochat.dataset` ClimbMix shards. The cleaning &
validation pipeline lives in `scripts/scaling/data_pipeline.py`:

```bash
# 1. inventory local shards
python -m scripts.scaling.data_pipeline inventory --dir $NANOCHAT_BASE_DIR/base_data_climbmix
# 2. compute raw stats
python -m scripts.scaling.data_pipeline stats --dir $NANOCHAT_BASE_DIR/base_data_climbmix --split train
# 3. apply filters & write cleaned shards
python -m scripts.scaling.data_pipeline clean \
    --in  $NANOCHAT_BASE_DIR/base_data_climbmix \
    --out $NANOCHAT_BASE_DIR/base_data_clean
# 4. recompute stats on cleaned & diff vs raw + plot histograms
python -m scripts.scaling.data_pipeline validate \
    --raw-dir   $NANOCHAT_BASE_DIR/base_data_climbmix \
    --clean-dir $NANOCHAT_BASE_DIR/base_data_clean
```

All reports land under `$NANOCHAT_BASE_DIR/scaling_v100/data_report/`.

## Run

```bash
bash runs/scaling/v100_min.sh
```

Outputs in `$NANOCHAT_BASE_DIR/scaling_v100/`:

- `results.csv`            — one row per finished run
- `manifest.json`          — sweep config (budgets / depths / dtype / etc.)
- `run_<tag>.log`          — full training log per run

## Analyze & plot

```bash
python -m scripts.scaling.analyze
```

Writes paper-quality figures (PNG + PDF) to
`$NANOCHAT_BASE_DIR/scaling_v100/figures/`:

- `isoflop.{png,pdf}`           IsoFLOP curves with U-shape quadratic fits
- `loss_vs_compute.{png,pdf}`   power law L(C) on the empirical frontier
- `optimal_N_vs_C.{png,pdf}`    N*(C) power law fit
- `optimal_D_vs_C.{png,pdf}`    D*(C) power law fit
- `fit_summary.json`            all slopes / intercepts / R²
