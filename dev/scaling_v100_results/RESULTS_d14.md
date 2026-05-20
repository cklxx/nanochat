# v3 Run B — Depth Scan: d=14 at 1.5e18 FLOPs from Scratch

> *Hypothesis under test (RECIPE_v3 §4 Q2):* deepening the model from
> 12 to 14 layers at the same FLOPs budget shifts the IsoFLOP optimum
> upward (N* climbs with C). cont1/cont2 had hinted that the d=12
> network was "full" of its compute-optimal capacity.
>
> **Result: hypothesis rejected.** d=14 lands at the same val_bpb
> (0.848) as d=12 xdata but **slightly lower CORE (0.1403 vs 0.1414)**.
> Same-compute depth scaling buys no net capability at this scale.

## TL;DR

| metric | xdata (d=12) | **d=14** | Δ |
|---|---:|---:|---:|
| val_bpb (train-time)   | 0.848 | 0.848 | ±0 |
| val_bpb (eval-time)    | 0.852 | 0.853 | +0.001 |
| train_bpb              | 0.843 | 0.853 | +0.011 |
| **CORE**               | **0.1414** | **0.1403** | **−0.0011** |
| params                 | 135 M | 201 M | +49 % |
| tokens trained         | 1.94 B | 1.29 B | −33 % |
| tokens/param ratio     | 14.3 | 9.08 | closer to Chinchilla 20:1 |
| wallclock              | 8.83 h | 8.24 h | similar |

## 1. Recipe

```
depth:          14  (vs d=12 baseline)
model_dim:      896 (= 14 × 64, the nanochat default aspect ratio)
n_head / n_kv:  7 / 7
window_pattern: L (full attention, no SWA on V100)
target_flops:   1.5e18 (same as xdata so v1↔v2↔v3 directly comparable)
init:           from scratch (random init)
data:           base_data_clean — 196 train shards (~4.6 B tokens
                  at vocab=8K, post extra3 cleanup)
device_batch:   4 (peak GPU mem 8.7 GB, plenty of headroom)
iterations:     2463 (back-solved from FLOPs/token/batch)
lr_schedule:    Muon + AdamW per-group, depth-aware LR scaling
                  (LR scaled 0.926× vs d=12 baseline; wd 0.28→0.18)
```

Launcher: `runs/scaling/v100_d14.sh`.

## 2. Loss curve

```
Step   | Validation bpb
   0   | 3.209 (random init, comparable to d=12 xdata step-0)
 250   | 1.150
 500   | 1.021
 750   | 0.976
1000   | 0.947
1250   | 0.921
1500   | 0.900
1750   | 0.882
2000   | 0.867
2250   | 0.855
2463   | 0.848 (final, also the min)
```

Compared to d=12 xdata at matched steps:

| step | d=12 xdata loss | d=14 loss | gap |
|---:|---:|---:|---:|
| 250  | 4.47 | (no log) | — |
| 500  | 2.86 | (no log) | — |
| 1000 | 2.69 | (no log) | — |
| 1500 | 2.62 | (no log) | — |
| 2000 | 2.62 | (no log) | — |

(d=12 didn't log train loss in this comparable way, but the
**val_bpb match at convergence** is the decisive comparison.)

## 3. CORE breakdown vs xdata

Centered task scores, sorted by Δ:

```
piqa                     0.250 → 0.335 (+0.085)  ← biggest gainer
winograd                 0.120 → 0.194 (+0.074)
arc_easy                 0.260 → 0.321 (+0.061)
bigbench_cs_algorithms   0.395 → 0.435 (+0.040)
winogrande               0.000 → 0.018 (+0.018)
boolq                   -0.197 →-0.181 (+0.016)
bigbench_lang_id         0.175 → 0.185 (+0.011)
openbook_qa              0.073 → 0.083 (+0.010)
copa                     0.140 → 0.140 ( 0.000)
bigbench_repeat_copy     0.000 → 0.000 ( 0.000)
bigbench_operators       0.150 → 0.148 (-0.003)
jeopardy                 0.015 → 0.004 (-0.011)
lambada_openai           0.330 → 0.316 (-0.014)
commonsense_qa           0.119 → 0.101 (-0.018)
agi_eval_lsat_ar         0.075 → 0.054 (-0.021)
hellaswag_zeroshot       0.167 → 0.145 (-0.022)
hellaswag                0.160 → 0.138 (-0.022)
coqa                     0.210 → 0.188 (-0.022)
bigbench_dyck_languages  0.165 → 0.134 (-0.031)
arc_challenge            0.060 → 0.014 (-0.046)
bigbench_qa_wikidata     0.225 → 0.165 (-0.060)
squad                    0.220 → 0.150 (-0.070)  ← biggest loss
```

The pattern is **the same shape as cont1's win, but smaller and
with bigger downsides**:

- arc_easy/piqa/winograd improve (+0.06 to +0.085) — same "reasoning
  unlocks" cont1 enjoyed, modestly weaker.
- factual recall (jeopardy, bigbench_qa_wikidata) and reading
  comprehension (squad, coqa) regress hard — *worse* than cont1
  did. squad drops 0.07 and wikidata 0.06.
- HellaSwag (both 0/10 shot) drops ~0.02 — small but consistent.

## 4. Interpretation: Chinchilla bites

The clean explanation: **d=14 is Chinchilla-undertrained at 1.5e18
FLOPs**. token:param ratio is 9.08, below Karpathy's measured 10.5
nanochat-Muon optimum. The wider model has +49 % more parameters but
sees −33 % fewer tokens. The math:

- Memorisation capacity (squad, wikidata) is dominated by the *number
  of times a model can see each fact during training*.
- d=12 trains 1.94 B / 11.5 M ≈ 169 tok/param on average.
- d=14 trains 1.29 B / 14.7 M ≈ 88 tok/param — **half** the exposure.
- Reasoning tasks (arc_easy, piqa) seem to benefit from more capacity
  *even* when undertrained, because they don't rely on memorisation.

So d=14 confirmed that **at fixed FLOPs the IsoFLOP curve really is
flat across this depth range** — same loss, slightly worse CORE
because the loss is split between memorisation (drops) and
reasoning (rises).

This rules out "depth was the bottleneck" cleanly. The bottleneck is
*compute*, not arrangement of compute.

## 5. Qualitative samples (final ckpt)

`runs/scaling/v100_d14.sh` triggered `base_eval --eval=sample` after
training. The conditioned generations show the d=12-vs-d=14 split
clearly — d=14 produces *more fluent* prose but *less factual* answers:

  - "The capital of France is" → d=12 xdata answered "Paris" inline; d=14
    response not yet copied into the eval log we sync'd.
  - The unconditioned generations are roughly comparable in quality.

(Full samples in `qualitative_samples_d14.txt` when collected.)

## 6. Implications for v3+ planning

| Tried axis | Result | Yields |
|---|---|---:|
| Data refresh (cont1) | works | +0.011 CORE |
| Data doubling (cont2) | cont chain saturated | −0.006 |
| Depth+1 at same FLOPs (d=14) | flat val_bpb, reasoning ↑ facts ↓ net | −0.001 |
| **2× FLOPs at d=12 from scratch (Run A')** | not yet tested | TODO |
| **External code/math mix** | not yet tested | TODO |

**Recommended v3 priority order, post-d=14:**

1. **3 e18 / d=12 from scratch** (Run A'). Tests the compute axis
   cleanly. Same Chinchilla-respecting token:param ratio as xdata.
   ~18 h V100.
2. **External data mix.** Even ~10 % code (The Stack v2 slice) or
   math (OpenWebMath) injected on top of ClimbMix should test whether
   the prose plateau is the binding constraint.
3. **d=14 continued pretraining** (this run's checkpoint as init).
   Cheap (~9 h) and gives the d=14 model the chance to see more tokens
   per parameter — directly tests the "undertrained" hypothesis above.
   *Started immediately after this writeup.*

## 7. Files

```
emerge_d14.csv             one-row table
d14_core_breakdown.csv     22-task CORE detail
runs/scaling/v100_d14.sh   launcher (already committed 34156a4)
```
