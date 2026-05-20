# v3 Run B+ — Continued Pretraining of d=14 on the Doubled-Data Pool

> *Follow-up to* `RESULTS_d14.md`. *Init from the d=14 / 1.5e18 / from-
> scratch checkpoint (CORE 0.1403), run another full 1.5e18 schedule
> on the now 196-shard cleaned corpus (~4.6 B tokens, 2× the pool the
> baseline d=14 saw). Mid-train CORE evaluated every 100 steps with
> max-per-task=50 for real-time diagnostic.*

## TL;DR

| metric | d=14 baseline | **d=14_cont** | Δ |
|---|---:|---:|---:|
| val_bpb (train-time) | 0.848 | **0.828** | **−0.020** ↓ |
| val_bpb (eval-time)  | 0.853 | **0.833** | **−0.020** ↓ |
| train_bpb            | 0.853 | 0.836 | −0.017 |
| **CORE**             | 0.1403 | **0.1435** | **+0.0032** ↑ |
| cum. wallclock       | 8.2 h | 16.5 h | +8.2 h |
| cum. FLOPs           | 1.5 e18 | 3.0 e18 | +1.5 e18 |
| init                 | from scratch | d=14 ckpt | — |
| data pool            | 196 shards (~4.6 B tok) | 196 shards (~4.6 B tok) | same |

The val_bpb improved a full 0.020 — **the biggest val_bpb gain of any
single training run in v2/v3** — but the CORE bump is only +0.003.
The "val_bpb keeps improving while CORE saturates" pattern from cont2
reappeared.

## 1. Recipe

```
init_from_tag:  v100_emerge_v8k_1.5e18_d14   (CORE 0.1403 ckpt)
target_flops:   1.5e18 (full schedule on top of the checkpoint)
depth:          14, hidden 896, 9 heads, 3 KV heads (GQA-free)
data:           base_data_clean — 196 train + 1 val shards
                (the 96 from cont2 + 100 freshly cleaned extra4 shards
                downloaded and cleaned the same morning)
device_batch:   4 (peak GPU mem 8.7 GB)
iterations:     2463
lr_schedule:    full peak (lrm=1.0) → constant → warmdown (0.65)
core_metric_every:        100  (mid-train CORE diagnostic)
core_metric_max_per_task: 50   (~5 min per quick eval)
sample_every:             250
save_every:               500  (incremental ckpts at 500/1000/1500/2000)
```

## 2. Loss + CORE trajectory (mid-train diagnostic)

The `--core-metric-every=100 --core-metric-max-per-task=50` instrumentation
gave us 24 mid-training CORE data points — first time we have the full
shape of how CORE evolves during a continued-pretraining schedule:

```
step    val_bpb   CORE@50    LR phase
   0    0.848     0.140      init (= d=14 baseline, full eval)
 100      —       0.078      warmup ramp (lrm 1.0 by step ~50)
 200      —       0.096      constant LR
 250    0.920      —         (val_bpb +0.07 — model thrown out of basin)
 500    0.942     0.082      constant LR — CORE bottoms out, val_bpb peak
 750    0.928      —         (val_bpb starts climbing back)
1000    0.911     0.109      warmdown starts at step 862
1200      —       0.148      first CORE > baseline ✓
1500    0.875     0.114      val_bpb compressing fast
1750    0.859      —
2000    0.845     0.152      CORE essentially tied with cont1 SOTA
2250    0.834      —
2400      —       0.146
2463    0.828     0.148      final ckpt
```

Two-phase behaviour, identical to cont1/cont2:
1. **Steps 0-500:** warmup destroys CORE (0.140 → 0.082) and val_bpb
   (0.848 → 0.942). The model is dragged out of its converged basin.
2. **Steps 500-1500:** constant LR keeps CORE around 0.10, val_bpb
   barely creeps down to 0.875.
3. **Steps 1500-2463:** warmdown takes over. val_bpb plummets 0.875 →
   0.828, CORE oscillates 0.11–0.15 with high noise from
   `max-per-task=50` (each point has ±0.02 std-dev).

The final mid-train CORE@50 at step 2463 (0.148) over-estimates the
true CORE@500 by ~0.005 — a small noise correction that lands us at
the full-eval result of **0.1435**.

## 3. Full CORE breakdown vs the v2/v3 family

Centered scores. **Bold** = best across all five models we've trained.

```
Task                    xdata   cont1   cont2   d=14   d=14_cont  best
                        d=12    d=12    d=12    d=14   d=14
hellaswag_zeroshot      0.167   0.162   0.172   0.145  0.176*    d=14_cont
jeopardy                0.015*  0.009   0.009   0.004  0.003     xdata
bigbench_qa_wikidata    0.225*  0.214   0.215   0.165  0.202     xdata
arc_easy                0.260   0.362*  0.331   0.321  0.348     cont1
arc_challenge           0.060*  0.049   0.027   0.014  0.046     xdata
copa                    0.140   0.140   0.180*  0.140  0.100     cont2
commonsense_qa          0.119*  0.093   0.025   0.101  0.026     xdata
piqa                    0.250   0.338   0.347   0.335  0.354*    d=14_cont
openbook_qa             0.073   0.099   0.120*  0.083  0.115     cont2
lambada_openai          0.330*  0.322   0.328   0.316  0.330*    tied
hellaswag               0.160   0.155   0.164   0.138  0.168*    d=14_cont
winograd                0.120   0.223*  0.216   0.194  0.209     cont1
winogrande              0.000   0.045*  0.026   0.018  0.040     cont1
bigbench_dyck_languages 0.165*  0.111   0.084   0.134  0.128     xdata
agi_eval_lsat_ar        0.075*  0.049   0.016   0.054  0.027     xdata
bigbench_cs_algorithms  0.395   0.405   0.430   0.435* 0.390     d=14
bigbench_operators      0.150*  0.124   0.119   0.148  0.110     xdata
bigbench_repeat_copy    0.000   0.000   0.031*  0.000  0.000     cont2
squad                   0.220*  0.220*  0.215   0.150  0.174     tied
coqa                    0.210   0.228*  0.226   0.188  0.212     cont1
boolq                  -0.197  -0.169* -0.233  -0.181 -0.177    cont1
bigbench_lang_id        0.175   0.181   0.174   0.185* 0.177     d=14
─────────────────────────────────────────────────────────────────
CORE                    0.1414  0.1528* 0.1465  0.1403 0.1435   cont1
```

**Wins-per-model count:** xdata 7, cont1 7, cont2 4, d=14 2, d=14_cont 4
(plus 1 tied). The five models actually split the task wins fairly evenly
— no single recipe dominates everywhere.

## 4. Where d=14_cont specifically wins

| Task | d=14_cont | second best | gain |
|---|---:|---:|---:|
| hellaswag_zeroshot | **0.176** | 0.172 (cont2) | +0.004 |
| hellaswag (10-shot) | **0.168** | 0.164 (cont2) | +0.004 |
| piqa | **0.354** | 0.347 (cont2) | +0.007 |
| lambada_openai | **0.330** | 0.330 (xdata) | tied |

All four wins are **language-modeling / commonsense-prose tasks** —
exactly the surface that a lower val_bpb buys you. The
0.020-bpb improvement (0.848 → 0.828) shows up cleanly here.

What d=14_cont **gives up** vs cont1: 9 of cont1's task gains shrink
back partially (arc_easy 0.362 → 0.348, arc_challenge 0.049 → 0.046,
winograd 0.223 → 0.209, coqa 0.228 → 0.212, …). Net of those losses
plus the modest LM gains: −0.009 CORE vs cont1.

## 5. Updated cross-experiment summary

```
val_bpb (lower is better):
   d=14_cont (0.833) ≈ cont2 (0.832) < cont1 (0.841) < xdata = d=14 (0.848)
CORE (higher is better):
   cont1 (0.1528) > cont2 (0.1465) > d=14_cont (0.1435) > xdata (0.1414) > d=14 (0.1403)
```

**val_bpb and CORE have fully decoupled in v3.** The three best val_bpb
models (cont2, d=14_cont, cont1) include the *worst* CORE among the
non-d=14 cohort (cont2 at 0.1465). Going below val_bpb 0.84 buys little
or nothing on CORE; the gains land on a handful of prose tasks while
factual / reasoning tasks erode.

## 6. What v3 actually proved

Across the five v2/v3 models we've trained, every axis tried *other
than raw compute* has now been ruled out as the binding constraint:

| Axis perturbed | Result | Cumulative CORE Δ |
|---|---|---:|
| baseline (xdata d=12, 1.5 e18) | reference | 0.1414 |
| **+1 cont round (refresh data)** | **best so far** | **+0.011** |
| +1 more cont round (cont2, 2× data) | overcooks | −0.006 |
| **+2 layers at same FLOPs (d=14)** | flat / slight loss | −0.001 |
| +1 cont round on d=14 (this run, 2× data) | partial recovery | +0.003 |

All four perturbations to the "same compute, change something else"
recipe land within ±0.011 of each other. Continued pretraining helps
exactly once, depth scaling at fixed FLOPs is neutral, and doubling
prose-only data has reached its ceiling.

**The remaining clean axis is FLOPs.** RECIPE_v3 Run A' (3 e18 / d=12
from scratch with the 396-shard / ~9.4 B-token corpus that's now ready
after the extra4 cleanup) is the next experiment that has a chance
of meaningfully moving CORE.

## 7. Files

```
emerge_d14_cont.csv             one-row table
d14_cont_core_breakdown.csv     22-task CORE detail
RESULTS_d14_cont.md             this writeup
```
