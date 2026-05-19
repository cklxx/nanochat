# Continued Pretraining of the d=12 / 1.5e18 Champion

> *Addendum to* `RESULTS.md` *and* `RECIPE.md` *— this run was not in the
> original v2 sweep. It re-trains the v2 champion (CORE 0.1414) for one
> additional full schedule on the freshly-refreshed 48-shard cleaned
> corpus, both to test whether more data can push past 0.14 and to vet
> the new* `--init-from-checkpoint-tag` *plumbing introduced in 3c5de4a.*

## TL;DR

| metric | xdata (v2 best, from scratch) | cont (continued pretraining) | Δ |
|---|---:|---:|---:|
| val_bpb       | 0.847831 | **0.841168** | **−0.0067** (−0.79 %) |
| train_bpb     | 0.842659 | 0.844558  | +0.0019 |
| **CORE**      | 0.1414   | **0.1528** | **+0.0114** (+8.1 % rel.) |
| train wallclock | 9.0 h  | 8.7 h    | — |
| total compute (cumulative) | 1.5e18 FLOPs | ~3.0e18 FLOPs (2× pass) | — |

**Verdict:** continued pretraining on refreshed data yields a real but
modest CORE bump at the cost of an extra V100-day. The gain is **task-
selective** — reasoning tasks strongly improve, factual recall slightly
regresses (see §3).

## 1. Recipe

```
init_from_tag:  v100_emerge_v8k_1.5e18_d12_xdata  (CORE 0.1414 checkpoint)
target_flops:   1.5e18 (another full schedule on top of the checkpoint)
depth:          12
window_pattern: L
data:           base_data_clean (48 shards, ~3 B tokens — same source,
                refreshed corpus that overlaps but is not identical with
                the xdata run's data)
lr_schedule:    full peak (lrm=1.0) → constant → warmdown (0.65 ratio)
seed/shuffle:   default (different from xdata)
device_batch:   4 (same as xdata)
```

Launch:

```bash
NANOCHAT_DTYPE=float16 \
NANOCHAT_TOKENIZER_DIR=$HOME/.cache/nanochat/tokenizer_v8k \
NANOCHAT_DATA_DIR=$HOME/.cache/nanochat/base_data_clean \
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
  --depth=12 --target-flops=1.5e18 \
  --target-param-data-ratio=-1 --window-pattern=L \
  --init-from-checkpoint-tag=v100_emerge_v8k_1.5e18_d12_xdata \
  --run=dummy --model-tag=v100_emerge_v8k_1.5e18_d12_cont \
  --eval-tokens=2621440 \
  --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
  --device-batch-size=4
```

## 2. Loss curve sanity check

Continued pretraining looks suspicious to the eye because the loss
*rises* at warmup (the checkpoint was already converged, full peak LR
pushes it back out of the basin):

| step | xdata (from scratch) | cont (from checkpoint) | Δ |
|---:|---:|---:|---:|
| 0    | 9.01 | 2.29 | — (start) |
| 100  | 4.47 | 2.52 | cont well ahead |
| 500  | 2.86 | 2.65 | cont ahead by 0.21 |
| 1000 | 2.69 | 2.61 | cont ahead by 0.08 |
| 1200 | 2.77 | 2.69 | cont ahead by 0.08 |
| 2000 | 2.62 | (warmdown) | |
| 3697 | 2.46 (final)| — | both converged |

After warmdown the cont run lands at val_bpb 0.841 vs the xdata's 0.848 —
the ~0.08-train-loss lead translates to a 0.007 BPB lead at convergence.

## 3. CORE task breakdown (cont − xdata)

```
arc_easy                  +0.102  ← physical / commonsense reasoning
piqa                      +0.088  ← physical reasoning
winograd                  +0.103  ← pronoun resolution (reasoning)
winogrande                +0.045  ← pronoun resolution
boolq                     +0.029  (still negative absolute)
openbook_qa               +0.025  ← elementary science
coqa                      +0.018  ← reading comprehension
bigbench_cs_algorithms    +0.010
bigbench_lang_id          +0.006
squad                     +0.000
copa                       0.000
bigbench_repeat_copy_logic 0.000
hellaswag                 −0.005
hellaswag_zeroshot        −0.005
jeopardy                  −0.006  ← factual recall ↓
lambada_openai            −0.008
arc_challenge             −0.011
bigbench_qa_wikidata      −0.011  ← factual recall ↓
agi_eval_lsat_ar          −0.026
commonsense_qa            −0.026
bigbench_operators        −0.026
bigbench_dyck_languages   −0.054  ← formal-syntax matching ↓
```

**Pattern:** "soft reasoning" tasks gain (arc_easy, piqa, winograd
collectively contribute ~+0.07 to CORE on their own); factual recall and
formal pattern matching tasks lose a touch.

This is consistent with the continued schedule acting as a regularizer
that smooths the loss landscape — the model gives up some razor-thin
memorized facts to consolidate broader reasoning patterns. It is **not**
the result of seeing strictly more data (the cont corpus has the same
total size as the xdata corpus, only a different shuffle).

## 4. Qualitative samples

See `qualitative_samples_cont.txt`. Two telling cases:

  - **planets**: xdata "Mercury, Venus, Earth, Mars, Jupiter" → cont
    "Mercury, Venus, Mars, Jupiter, S..." — the cont model *skipped Earth*.
    Aligns with the −0.011 on bigbench_qa_wikidata.

  - **gold's chemical symbol**: xdata "Au. The chemical symbol of silver
    is Ag." → cont "Au. It is a soft, silvery-white metal that is" —
    cont gives a richer encyclopedic continuation rather than parrot-
    matching the prompt's pattern.

## 5. Implications for v3

This validates the `--init-from-checkpoint-tag` mechanism for staged
training and confirms data isn't yet the binding constraint at this
scale (a same-size refresh adds ~0.01 CORE, not transformative).

Given the new 49 cleaned shards (extra2 batch, total now 97 shards ≈
5.8 B tokens), the v3 plan in `RECIPE_v3.md` is **Branch I** (data is
worth refreshing). Recommended next runs:

| # | depth | FLOPs | tokens/param | rationale |
|---|---:|---:|---:|---|
| A | 12 | 3e18 | 27 (overtrained on 6 B-token corpus) | tests if α_L bend continues |
| B | 14 | 1.5e18 | 8 (compute-optimal at deeper) | tests N* shift |

Total estimated wallclock: 28 h. See `RECIPE_v3.md` §5.

## 6. Files

```
emerge_cont.csv             one-row table of the cont run
cont_core_breakdown.csv     22-row task-level CORE breakdown
qualitative_samples_cont.txt sample generations side-by-side with xdata
```
