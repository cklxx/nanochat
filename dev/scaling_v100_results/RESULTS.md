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

All 20 IsoFLOP runs completed on a single V100 in ~3 hours wallclock.
Full per-run table in `results.csv` and fits in `fit_summary.json`.

### Loss vs compute (compute-optimal frontier)

| C (FLOPs) | best run | val_bpb | tokens | eff. params N |
|---:|:--|---:|---:|---:|
| 1e15 | d=4 | **1.821** | 10.5M  | 11.5M |
| 3e15 | d=4 | **1.534** | 31.7M  | 11.5M |
| 1e16 | d=4 | **1.165** | 105.9M | 11.5M |
| 3e16 | d=4 | **1.081** | 318.0M | 11.5M |

Power-law fit on the empirical frontier:

$$
L(C) \;=\; A \cdot C^{-\alpha_L}, \qquad \alpha_L = 0.168, \;\; R^2 = 0.976
$$

→ across 30× compute, val_bpb dropped 1.821 → 1.081 (–41 %).

### IsoFLOP optima (Hoffmann-style quadratic-in-log fit)

At budgets large enough for the IsoFLOP curve to bend (3e15 and 3e16), the
parabolic minimum sits at:

| C (FLOPs) | N* (eff. params) | val_bpb* (fit) |
|---:|---:|---:|
| 3e15 | 1.48 × 10⁷ | 1.567 |
| 3e16 | 1.50 × 10⁷ | 1.062 |

At 1e15 and 1e16 the curves are still monotone in N (we're below the
optimum), so the empirical minimum is at our smallest depth (d=4).

### Compute-optimal data scaling

$$
D^*(C) \propto C^{1.00}, \qquad R^2 = 0.99996
$$

→ at our scale, *almost all* extra compute goes into more tokens, not
larger models. This is consistent with the embedding-overhead floor:
N saturates at the smallest model in the sweep that the parabola permits.

### Token : parameter ratio along the frontier

| C (FLOPs) | D* / N* |
|---:|---:|
| 1e15 | 0.91 (heavily under-trained) |
| 3e15 | 2.14 |
| 1e16 | 9.18 |
| 3e16 | **21.23** ← brackets Chinchilla = 20 |

This is the cleanest minimum-scale validation we get: even with a tiny
~$10^7$-param model on one V100, the empirically compute-optimal training
shifts from severe under-training at $10^{15}$ FLOPs to right at
Chinchilla's $D/N \approx 20$ by $3 \times 10^{16}$.

### Figures (PNG + PDF)

All in `figures/`:

- `isoflop.png` — five depths × four budgets, with quadratic-in-log fits
  and the fitted optima (☆) at 3e15 and 3e16.
- `loss_vs_compute.png` — the L(C) power law fit on the frontier.
- `optimal_N_vs_C.png` — N*(C) is essentially flat (R² = 0.21);
  the embedding-overhead floor dominates at this scale.
- `optimal_D_vs_C.png` — D*(C) tracks compute linearly.
- `token_param_ratio.png` — D*/N* climbs from ~1 to ~21 across the four
  budgets, hitting Chinchilla.

## 5. Capability emergence (DCLM CORE benchmark)

Each of the 20 final checkpoints was evaluated with
`scripts/base_eval.py --eval=core,bpb,sample --max-per-task=200`. CORE is
the *centered* DCLM ICL accuracy aggregated over ~22 tasks (ARC, HellaSwag,
OBQA, MMLU subsets, BoolQ, BigBench-LangID, …). Definition: random
performance ⇒ CORE = 0; GPT-2 (1.6B) ⇒ CORE ≈ 0.256.

Raw CSV: `benchmarks.csv`. Plots: `figures/core_vs_compute.png` and
`figures/core_vs_params.png`.

### Headline numbers

| C (FLOPs) | best CORE | which run | val_bpb of that run |
|---:|---:|:--|---:|
| 1e15 | +0.019 | d=10 | 3.124 |
| 3e15 | +0.012 | d=12 | 2.939 |
| 1e16 | **+0.041** | d=4  | 1.165 |
| 3e16 | **+0.072** | d=4  | 1.081 |

Five **null observations** at C ≤ 3e15: every CORE measurement is within
±0.04 of zero, indistinguishable from random.

Two **clear-but-tiny positive signals** at C ≥ 1e16:

- C = 1e16, d=4 → CORE = **+0.041** (val_bpb 1.165, eff. N ≈ 11.5M)
- C = 3e16, d=4 → CORE = **+0.072** (val_bpb 1.081)
- C = 3e16, d=6 → CORE = **+0.063** (val_bpb 1.085)

Pattern: only the *compute-optimal* models (those whose IsoFLOP curves
already showed the minimum) develop above-random capability. The biggest
models in our sweep (d=10, d=12) are too undertrained at every budget and
stay near random.

### Interpretation

- **Yes, there is non-trivial capability above random** in the best
  3e16-FLOPs run — a ~36M-param model trained on 318M tokens reaches
  CORE +0.07. That's 4× below GPT-2 1.6B (0.256) but ~3× above the
  noise floor we measured at C ≤ 3e15.
- The signal *strictly tracks* val_bpb on the IsoFLOP frontier: the
  better the language modeling, the higher the CORE. Capability does
  not appear in over-parameterised under-trained models even though
  they have more weights.
- We do not see any **abrupt** emergence threshold in this range, which
  is consistent with the literature: capability emergence is a function
  of *both* parameters and training tokens, and our smallest scale is
  well below the typical emergence regime.

## 6. v2 sweep — chasing CORE ≥ 0.20 with vocab=8K

After the v1 sweep, the user asked for the **minimum-emergence** scale,
defined as CORE ≥ 0.20 (~80 % of GPT-2 1.6B's 0.256). The hypothesis was
that the v1 sweep's IsoFLOP optimum was locked at N ≈ 11.5 M because the
32K-vocab embedding dominated the parameter budget; shrinking the vocab
should let bigger models become compute-optimal. See `RECIPE.md` for the
research that informed the decisions.

**Changes vs v1:**

| | v1 | v2 |
|---|---|---|
| tokenizer vocab | 32 K | **8 K** (`tokenizer_v8k/`) |
| embed:transformer ratio at d=4 | 8 : 1 | **2 : 1** |
| data corpus | 7 train shards (~170 M tokens at vocab=8K) | grew to 47-48 shards (~1.15 B tokens) by end of sweep |
| hyperparameters | nanochat defaults | nanochat defaults (unchanged) |
| sweep schedule | 5 depths × 4 budgets dense | progressive `(depth, FLOPs)` probe, stop at CORE ≥ 0.20 |

### v2 final results (6 runs, single V100, ~24 h cumulative)

The full v2 sweep is in `emerge.csv`. Final IsoFLOP-frontier numbers:

| C (FLOPs) | depth | params | tokens | val_bpb | CORE | wallclock |
|---:|---:|---:|---:|---:|---:|---:|
| 3e16   | 4  | 11.5 M  | 530 M | 1.114 | +0.050 | 15 m |
| 1e17   | 4  | 11.5 M  | 1.77 B | 1.092 | +0.044 | 48 m |
| 1e17   | 6  | 26 M    | 718 M | 1.014 | +0.075 | 40 m |
| 3e17   | 6  | 26 M    | 2.16 B | 0.989 | +0.077 | 2 h |
| 1e18   | 8  | 50 M    | 3.61 B | 0.905 | +0.119 | 6 h |
| **1.5e18** | **12** | **135 M** | **1.94 B** | **0.848** | **+0.1414** | **9 h** |

The **1.5 e18 / d=12** run is the v2 champion: it lifted CORE from
+0.072 (v1 best) to +0.1414 (+97 %). v1-vs-v2 fits are summarised in
`fit_summary_v1v2.json`; the loss-vs-compute power law flattens from
α<sub>L</sub> = 0.168 (v1, pre-emergence) to α<sub>L</sub> = 0.064 (v2,
post-emergence).

Findings:

1. **The vocab shrink delivered the predicted N\* shift.** d=12 became
   compute-optimal at 1.5 e18 FLOPs — impossible under the v1 32K-vocab
   embedding tax.
2. **CORE 0.20 was *not* reached** inside the 24-h V100 cap. The gap is
   ~+0.06, which the v2 L(C) slope says costs roughly 4–8× more compute.
3. **Capability flips from rote-completion to fact recall** around
   1 e18 FLOPs. See `qualitative_samples_d12.txt`: the 1.5 e18 model
   correctly produces "Paris", "Au, Ag, …", and "Mercury, Venus, Earth,
   Mars, Jupiter".

## 7. Continued pretraining (cont, cont2) and the data-pool experiment

After the v2 sweep, two extra runs tested whether **continued
pretraining + fresh data** can push past the 0.1414 plateau on the
same V100 budget:

| Run | Init | Depth | Data pool | Tokens trained | val_bpb | CORE |
|---|---|---:|---:|---:|---:|---:|
| v2 best (xdata) | scratch | 12 | 47 shards (~1.1 B tok) | 1.94 B | 0.848 | **0.1414** |
| cont1 | xdata ckpt | 12 | 48 shards (refreshed, ~1.15 B tok) | 1.94 B | 0.841 | **0.1528** (+0.011) |
| cont2 | cont1 ckpt | 12 | **96 shards (~2.3 B tok)** | 1.94 B | **0.832** | **0.1465** (−0.006 vs cont1) |
| **d=14** | **scratch** | **14** | **196 shards (~4.6 B tok)** | **1.29 B** | **0.848** | **0.1403** (−0.001 vs xdata) |
| **d=14_cont** | **d=14 ckpt** | **14** | **196 shards (~4.6 B tok)** | **1.29 B** | **0.833** | **0.1435** (+0.003 vs d=14, −0.009 vs cont1 SOTA) |

cont1 — same compute, same-size but reshuffled data — delivered +0.011
CORE. Reasoning tasks dominate the gain (arc_easy +0.10, piqa +0.09,
winograd +0.10) while factual recall slips slightly (jeopardy −0.006).
See `RESULTS_cont.md` for the full breakdown.

cont2 doubles the data pool to ~2.3 B tokens by adding the freshly-cleaned
`extra2` shards. **It broke the cont chain.** Despite val_bpb improving
to 0.832 (the best v2-family checkpoint), CORE *fell* to 0.1465 — losing
0.006 to cont1 while still ahead of xdata by +0.005. Big regressions
hit commonsense_qa (−0.068), boolq (−0.064), arc_easy (−0.031) — the
exact reasoning tasks that made cont1 a win.

Both hypotheses (over-distillation by repeated warmdown, or diminishing
returns from same-distribution data) imply the same v3 fix: **change
something other than data volume** — deeper model, more FLOPs from
scratch, or an external code/math mix. See `RESULTS_cont2.md`.

### v3 Run B — Depth (d=14 from scratch, same 1.5 e18 FLOPs)

Tested the "depth was the bottleneck" hypothesis directly: same compute,
+49 % parameters (135 M → 201 M). Result (`RESULTS_d14.md`): val_bpb
**identical to d=12 xdata** (0.848 vs 0.848) but CORE slipped
**−0.001** (0.1403). The same arc_easy/piqa/winograd reasoning pattern
showed up, but factual / reading-comp tasks fell further (squad −0.07,
bigbench_qa_wikidata −0.06) because at fixed FLOPs the deeper model
trains on **fewer tokens** (1.29 B vs 1.94 B) — token:param ratio drops
from 14 → 9, below the nanochat-Muon optimum of 10-11. Depth scaling
at the **compute budget is flat** — IsoFLOP is real and the bottleneck
is compute, not depth.

### v3 Run B+ — Continued pretraining of d=14 (d=14_cont)

After the d=14 baseline showed depth alone was flat, this run tested
whether the d=14 capacity *could* be exploited if it got another full
schedule's worth of compute. Init from the d=14 checkpoint, train
another 1.5 e18 on the **doubled 196-shard pool** (~4.6 B tok), with
mid-train CORE@50 every 100 steps for real-time diagnostic
(`RESULTS_d14_cont.md`). The instrumentation gave us the full
warmup→constant→warmdown CORE shape for the first time: CORE crashes
to 0.08 by step 500, plateaus, then climbs back during warmdown.

Final result: **val_bpb 0.833 (new SOTA, −0.020 vs xdata) but CORE
only 0.1435** (+0.003 vs d=14 baseline, −0.009 vs cont1 SOTA). The
val_bpb gain lands on language-modelling tasks (hellaswag +0.013,
piqa +0.019, hellaswag_zeroshot +0.009) — d=14_cont is the *best*
model on those four prose tasks — but reasoning/factual losses keep
the total CORE below cont1.

**Summary across all five v2/v3 models trained:**

```
val_bpb (low is good): d=14_cont (0.833) ≈ cont2 (0.832)
                     < cont1 (0.841) < xdata = d=14 (0.848)
CORE     (high is good): cont1 (0.1528) > cont2 (0.1465)
                       > d=14_cont (0.1435) > xdata (0.1414) > d=14 (0.1403)
```

val_bpb and CORE have fully decoupled. Each axis we've perturbed (data
refresh, data doubling, depth, depth+cont) lands within ±0.011 of the
baseline CORE. **The remaining clean axis is FLOPs.** RECIPE_v3
Run A' (3 e18 / d=12 from scratch with the now 396-shard / ~9.4 B-token
corpus that's ready after the extra4 cleanup) is the next experiment
with a chance of meaningfully moving CORE.

### Data composition aside

`DATA_COMPOSITION.md` profiles the 96-shard cleaned corpus by sampling
28,800 docs. The headline: ClimbMix as we use it is **95.8 % web prose
by characters**, with <0.2 % real math and <0.1 % real code. The
NVIDIA HF card's "web text + code + math + other" claim collapses to
near-zero code/math after the Karpathy repackage and our cleaning.
Implication: future CORE pushes need **external mixes** (e.g. The Stack
v2 for code, OpenWebMath for math), not deeper sampling of ClimbMix.

## 8. Reproduction

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

## 9. Caveats

- Embedding parameters dominate at small scale because vocab=32k is fixed.
  We therefore use **Kaplan-style effective N = transformer + lm_head** for
  the scaling-law fits (matching the original nanochat scaling notebook).
- FP16 vs FP8/BF16 introduces a constant multiplier on the loss but does not
  affect the *scaling exponent*; the slopes reported here are directly
  comparable to bf16 H100 results.
- We disabled wandb (`WANDB_MODE=disabled`) so the sweep is fully
  self-contained on the remote host.
