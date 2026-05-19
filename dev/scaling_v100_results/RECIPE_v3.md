# v3 Sweep Recipe — Pushing Past CORE 0.14 on 1×V100

> *Drafted 2026-05-19 while the v2-cont run (continued pretraining of
> 1.5e18/d=12 on refreshed data) is in flight. Decisions below depend on
> its outcome.*

## 1. State of the art (entering v3)

| Sweep | Best CORE | Best val_bpb | Config | Notes |
|---|---:|---:|---|---|
| v1 (vocab=32K, raw) | **0.0715** | 1.081 | d=4, 3e16 FLOPs | bottleneck = vocab 吞参数 |
| v2 (vocab=8K, clean) | **0.1414** | 0.848 | d=12, 1.5e18 FLOPs, 1.94B tok | emergence: 模型可背事实 |
| v2-cont (refresh data, 2nd schedule) | **0.1528** | 0.841 | init from xdata + 8.7h V100 | +0.011 abs / +8.1% rel, reasoning ↑ facts ↓ |

**Gap to GPT-2 (CORE 0.256):** still 0.115. ~1.8× CORE remaining.
Loss-vs-compute slope (v2 fit): αL = 0.064 (post-emergence flatten —
**not** the v1 slope of 0.168). So pure FLOPs scaling alone will only
deliver 0.84 → 0.74 bpb at 10× more compute — and CORE doesn't track
bpb linearly past emergence.

## 2. What we learned from v1+v2

1. **vocab=8K + cleaned data = +0.07 CORE** (v1→v2). Most of the v2 win.
2. **Depth matters more than width past emergence** —
   d=4→d=12 (3×) at fixed-ish compute (3e16→1.5e18 = 50×) gave the +0.07,
   suggesting depth is doing real work, not just compute.
3. **Token:param ratio 14:1** worked at d=12/1.5e18. Below Chinchilla
   optimal of 20:1 — i.e., we are still under-trained, not overtrained.
   More tokens should still help.
4. **Continued pretraining curve looks normal** —
   warmup pushes the loss back up to mid-training level (~2.7) then
   warmdown re-converges. Cont at step 1200 is ~0.08 ahead of original
   at matched step. **Modest gain expected, not transformative.**

## 3. New resources

- **5.8 B tokens available** (as of 2026-05-19): 48 existing cleaned
  shards + **49 fresh cleaned shards** (extra2 batch — shard 51-100;
  shard 51 was a 54 MB partial download and was quarantined as
  `.corrupt`). Total **97 cleaned shards, 9.3 GB on disk** ≈ 5.8 B
  tokens at vocab=8K. Up from 1.94 B tokens used in the best v2 run.
- **`--init-from-checkpoint-tag` works**: validated by the cont run —
  staged training without restarting from scratch is now a usable tool.

## 4. Open questions for v3

| Q | Why it matters | How to test |
|---|---|---|
| Q1: 更多 FLOPs 上 d=12 还能涨吗? | 测 αL 在 emergence 后斜率 | 3e18 / d=12 |
| Q2: 更深的模型 (d=14, d=16) 在同等 FLOPs 是否更优? | 测 N* (compute-optimal params) 是否随 C 上移 | 1.5e18 / d=14 |
| Q3: 数据扩到 6B 是否解锁更高 CORE? | 配合 Q1, 比较 token-rich vs token-starved 同 FLOPs 性能 | 3e18 / d=12 用 6B 数据 |

## 5. v3 sweep plan

**两轮，按 wallclock 排:**

| # | depth | FLOPs | params | tokens | tokens/param | est. wallclock | est. CORE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 12 | 3e18 | 135M | 3.7B | 27 | ~18h | 0.16-0.18 |
| B | 14 | 1.5e18 | ~180M | 1.5B | 8 | ~10h | 0.15-0.17 |

**Total: ~28h V100 = 1.5 nights.**

Optional Run C (only if A+B both yield CORE ≥ 0.16 to motivate):
| # | depth | FLOPs | params | tokens | tokens/param | est. wallclock | est. CORE |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 14 | 3e18 | ~180M | 3B | 17 | ~21h | 0.18-0.21 |

## 6. Decision gates

**v2-cont eval result (2026-05-19 11:55):** CORE 0.1528, val_bpb 0.841 →
**Branch I committed.** Continued pretraining gave +0.011 CORE on a
same-size data refresh, so a true 2× data refresh (6 B tokens) is
worth the cleaning step before v3 runs. Reasoning-task gains dominated
the bump (arc_easy +0.10, piqa +0.09, winograd +0.10) while factual-
recall tasks slid slightly (jeopardy −0.006, bigbench_qa_wikidata −0.011).
Implication: at this scale, *broader exposure* helps reasoning more than
*more compute on same data* helps memorization.

Status of the other branches (kept for the writeup):

- **Branch II (CORE ≈ 0.14, data not the bottleneck):** ruled out.
- **Branch III (CORE < 0.14, cont hurts):** ruled out — the full-peak
  LR schedule worked fine because the warmdown phase still re-converges
  the model.

## 7. Implementation notes

**Run A (3e18 / d=12):**
```bash
NANOCHAT_DTYPE=float16 torchrun --standalone --nproc_per_node=1 \
  -m scripts.base_train -- \
  --depth=12 --target-flops=3e18 \
  --target-param-data-ratio=-1 --window-pattern=L \
  --run=v3_3e18_d12 --model-tag=v100_emerge_v8k_3e18_d12 \
  --eval-tokens=2621440 --core-metric-every=-1 \
  --sample-every=-1 --save-every=-1 \
  --device-batch-size=4
```

**Run B (1.5e18 / d=14):**

⚠️ **d=14 not in current sweep code** — need to verify nanochat scales
LR/wd/batch correctly at this depth. Karpathy's d=12→d=20 sweep
flagged that d12-tuned hparams can hurt d=20; d=14 should be safer but
test val_bpb at ~5% step matches expected ~3.5 first.

```bash
NANOCHAT_DTYPE=float16 torchrun --standalone --nproc_per_node=1 \
  -m scripts.base_train -- \
  --depth=14 --target-flops=1.5e18 \
  --target-param-data-ratio=-1 --window-pattern=L \
  --run=v3_1.5e18_d14 --model-tag=v100_emerge_v8k_1.5e18_d14 \
  --eval-tokens=2621440 --core-metric-every=-1 \
  --sample-every=-1 --save-every=-1 \
  --device-batch-size=4
```

`device-batch-size=4` proven safe on V100 32GB at d=12. At d=14
(~33% more params), drop to `--device-batch-size=3` if OOM.

## 8. What this nets for the paper

Two new data points on the CORE-vs-compute curve at 1.5e18 + 3e18 FLOPs,
covering depths 12 and 14. With v1 (5 depths × 4 FLOPs) + v2 (6 runs) + v3,
total: **31 IsoFLOP runs across 7 orders of magnitude (1e15 → 3e18)**.

Sufficient for:
- **Theorem 1:** loss-vs-compute power law on V100 (αL pre-emergence: 0.17,
  post-emergence: 0.06, transition near 1e17)
- **Theorem 2:** CORE emerges between 1e17 and 1e18, plateaus at depth-12
  with single V100 budget
- **Observation:** at this scale (24h V100), the bottleneck is **depth +
  arch**, not data. CORE 0.20 likely needs d≥14 + 3e18+ FLOPs, or
  arch change (e.g., FA3 + bf16 on H100).

## 9. Risks / open issues

1. **LR for cont was full-peak** — if v2-cont gives CORE 0.13, that's a
   negative result for the cont approach; need to redo with lower peak LR.
2. **d=14 may OOM** at batch 4 — preview by running 50 warmup steps first.
3. **3e18 run is ~18h** — if V100 host gets bumped (process killed), need
   checkpoint-based resume. `--init-from-checkpoint-tag` flag (3c5de4a)
   enables this; verify save-every ≠ -1 for long runs.
4. **CORE 0.20 still uncertain on 1×V100** — Run C (3e18/d=14, 21h)
   is the only path to that, and only justified if A+B both push past 0.16.
