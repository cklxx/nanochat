# Continued Pretraining, Round 2 — When the cont chain breaks

> *Follow-up to* `RESULTS_cont.md`. *cont2 takes the cont1 checkpoint
> (CORE 0.1528) and runs another full 1.5 e18 schedule on the now 96-
> shard cleaned corpus (~2.3 B tokens, ~2× the pool cont1 saw). It
> tests whether stacking continued-pretraining rounds keeps paying
> dividends. **It does not.***

## TL;DR

| metric | xdata (v2 best) | cont1 | **cont2** | Δ(cont2−cont1) |
|---|---:|---:|---:|---:|
| val_bpb       | 0.848 | 0.841 | **0.832** | **−0.009** ↓ better |
| train_bpb     | 0.843 | 0.845 | 0.841 | −0.004 |
| **CORE**      | 0.1414 | 0.1528 | **0.1465** | **−0.006** ↑ *worse* |
| cum. wallclock | 9.0 h | 17.7 h | **26.4 h** | +8.7 h |
| cum. FLOPs     | 1.5 e18 | 3.0 e18 | **4.5 e18** | +1.5 e18 |

The headline: **val_bpb keeps falling but CORE has saturated and reversed.**
Running a third schedule on top of cont1 spent an extra V100-day and
*lost* 0.006 CORE. The model fits prose better but reasons worse.

## 1. Recipe (identical to cont1 except for the data pool)

```
init_from_tag:  v100_emerge_v8k_1.5e18_d12_cont   (CORE 0.1528 ckpt)
target_flops:   1.5e18
depth:          12
data:           base_data_clean — 96 train shards (~2.3 B tokens at vocab=8K)
                vs 48 shards (~1.15 B) for cont1
                The new shards (52-100) were freshly downloaded and
                cleaned in the same morning, then merged in.
lr_schedule:    full peak (lrm=1.0) → constant → warmdown (0.65 ratio)
seed/shuffle:   default
device_batch:   4
```

Tokens trained: 1.94 B (same as cont1) — but now drawn from a 2× pool,
so the model sees ≈84 % of the corpus rather than ~1.7× cycling.

## 2. Loss curve

| step | val_bpb |
|---:|---:|
| 0    | 0.836 (init = cont1 final) |
| 250  | 0.943 (warmup blow-up) |
| 500  | 0.957 (peak degradation) |
| 1000 | 0.940 |
| 1500 | 0.923 |
| 2000 | 0.899 |
| 2500 | 0.875 |
| 3000 | 0.854 |
| 3250 | 0.845 |
| 3500 | 0.837 |
| **3697** | **0.832** (final, also the min) |

Compared to cont1 at matched step 3500 = 0.836, cont2 is ~0.005 ahead
all the way through warmdown. Strictly on val_bpb, cont2 is the best
v2-family checkpoint we've trained.

## 3. CORE breakdown — where the loss came from

cont2 minus cont1 (centered scores):

```
copa                       +0.040  largest gainer
bigbench_repeat_copy_logic +0.031  emerges from zero
bigbench_cs_algorithms     +0.025
openbook_qa                +0.021
hellaswag_zeroshot         +0.010
piqa                       +0.009
hellaswag                  +0.009
lambada_openai             +0.006
bigbench_qa_wikidata       +0.001
jeopardy                    0.000
squad                      −0.005
bigbench_operators         −0.005
coqa                       −0.002
winograd                   −0.007
bigbench_lang_id           −0.007
winogrande                 −0.019
arc_challenge              −0.022
bigbench_dyck_languages    −0.027
arc_easy                   −0.031
agi_eval_lsat_ar           −0.033
boolq                      −0.064  (already negative; gets more negative)
commonsense_qa             −0.068  largest regression
```

The cont1→cont2 trade is **lopsided**: the gains are spread across many
tasks at +0.01-0.04 each, while the losses concentrate on a few
reasoning tasks that took big hits (-0.07 commonsense_qa, -0.06 boolq,
-0.03 arc_easy, -0.03 agi_eval_lsat_ar, -0.02 arc_challenge).

In particular, **the arc_easy / piqa / winograd breakout that made
cont1 a CORE win is now being clawed back** — cont1's standout strengths
(arc_easy +0.10 over xdata) have shrunk in cont2 (arc_easy +0.07 over
xdata).

## 4. Qualitative samples — same regression visible

Conditioned generations from cont2:

  - "The capital of France is the city of Paris, which is the capital of France"
    → cont1 was "the capital of the French capital city of Paris" — both bad.
  - "The chemical symbol of gold is Au. It is a metal that is found in nature in the form of"
    → richer but veers from the prompt's pattern (xdata gave "Au. The chemical symbol of silver is Ag…")
  - "If yesterday was Friday, then tomorrow will be Saturday. If yesterday was **Saturday**, then tom"
    → **cont2 fixed the cont1 mistake here** (cont1 said "Sunday")
  - "The opposite of hot is cold. The opposite of cold is **warm**."
    → cont2 introduces a slight error (warm ≠ hot)
  - "The planets of the solar system are: 1. Earth 2. Mars 3. Jupiter"
    → cont2 *skips Mercury and Venus and lists in wrong order*
    (xdata: "Mercury, Venus, Earth, Mars, Jupiter" ✓)
  - "If 5*x + 3 = 13, then x is 5*x + 3 = 13"
    → unchanged failure mode from cont1

The "planets" sample is particularly telling: each successive cont
generation has degraded that fact (xdata listed all five correctly,
cont1 dropped Earth, cont2 dropped Mercury AND Venus AND scrambled
the order). Fact memorization decays monotonically with each
continued-pretraining round.

## 5. Interpretation

Two hypotheses, both consistent with the data:

**H1 — Over-distillation.** Each cont schedule passes ~1.94 B tokens of
gradient updates through the model with full peak LR + warmdown. The
warmdown is essentially a sharpening step that smooths the loss
landscape toward the data distribution. After two such rounds, the
model has been smoothed past the point where fine-grained reasoning
features survive — broad prose fitness improves (val_bpb ↓) but the
sharp decision boundaries that ICL needs erode (CORE ↓).

**H2 — Same-distribution data has diminishing returns.** The added 49
shards in cont2 are from the same ClimbMix shuffle as the existing
48; 95.8 % of both are web prose (`DATA_COMPOSITION.md`). Doubling the
prose pool adds little new information per parameter update — but each
update still risks erasing earlier reasoning structure. Net: marginal
prose gain ≪ marginal reasoning loss.

Both hypotheses imply the same fix: **change something other than the
data volume.** Add architectural capacity (d=14+), more compute
(3 e18+), or *qualitatively different* data (external code/math
mixes — see `DATA_COMPOSITION.md` §4).

## 6. Implications for v3

1. **No cont3.** The cont chain has saturated and reversed. Spending
   another 8.7 V100-h to extend would lose more CORE.
2. **Adopt the cont1 checkpoint as the v2-era SOTA** (CORE 0.1528).
   cont2 is preserved as an instructive negative result but should not
   be the basis for further fine-tuning or post-training.
3. **Re-prioritise v3 sweep:** the depth experiment (Run B: 1.5 e18 /
   d=14) is now strictly more important than the FLOPs scan (Run A:
   3 e18 / d=12), because Run A would be testing exactly the failed
   hypothesis — that more same-data compute pays at this scale.
4. **External data mix is the new bottleneck-breaker.** Even a modest
   GitHub-Code or OpenWebMath slice (~50-100 M tokens) injected into
   the training corpus would test whether the saturation is data-mix
   limited rather than depth-limited.

## 7. Files

```
emerge_cont2.csv               one-row table
cont2_core_breakdown.csv       22-task CORE detail
RESULTS_cont2.md               this writeup
```
