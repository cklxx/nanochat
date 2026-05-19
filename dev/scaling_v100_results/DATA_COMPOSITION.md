# Data Composition Analysis — ClimbMix-400B (as we use it)

> *What's actually inside the data we're feeding into every v1/v2/v3
> training run, measured by sampling 28,800 docs (300 per shard ×
> 96 train shards) from* `base_data_clean/`.

## 1. Headline numbers

By document count (28,800 sampled out of ~4 M total):

| Category | docs | % |
|---|---:|---:|
| **Prose (general web text)** | 28,285 | **98.21 %** |
| Math (with LaTeX) | 246 | 0.85 % * |
| Structured (lists / glossaries) | 234 | 0.81 % |
| Markup (HTML fragments) | 15 | 0.05 % |
| Code (real source code) | 14 | 0.05 % |
| Code inside markdown | 6 | 0.02 % |

*The 0.85 % math figure is inflated by false positives — the heuristic
treats inline `$...$` as math, but USD price strings like `"$15. ... $42"`
trigger it. After spot-checking, the **true LaTeX-math fraction is well
under 0.2 %**.

By chars (proxy for tokens):

| Category | chars | % | avg doc len |
|---|---:|---:|---:|
| Prose | 83,088,479 | **95.81 %** | 2,938 |
| Math | 2,693,853 | 3.11 % * | 10,951 |
| Structured | 790,833 | 0.91 % | 3,380 |
| Code | 67,482 | 0.08 % | 4,820 |
| Markup | 51,373 | 0.06 % | 3,425 |
| Code in MD | 31,995 | 0.04 % | 5,332 |

Math docs are unusually long (~10 k chars vs 3 k for prose), which is why
their share rises from 0.9 % of docs to ~3 % of chars — though again,
this is inflated by USD prices.

## 2. Per-shard stability

96 train shards, mean ± standard deviation of each category's share:

```
prose             : 98.21 % ± 0.77 %
math              :  0.85 % ± 0.55 %
structured        :  0.81 % ± 0.49 %
code              :  0.05 % ± 0.13 %
markup            :  0.05 % ± 0.12 %
code_in_md        :  0.02 % ± 0.08 %
```

±0.77 % across 96 shards is a strong signal that **ClimbMix is deeply
mixed at the shuffle step**, not source-bucketed like RedPajama. You
can't "take a code-only subset of ClimbMix" because there isn't one in
the parquet.

## 3. What this means in practice

NVIDIA's HuggingFace card for `nvidia/Nemotron-ClimbMix` advertises
"web text, code, math, and other sources". In the repackage that
Karpathy uses (`karpathy/climbmix-400b-shuffle`) and after our doc-level
cleaning, the dataset is **effectively a pure-prose web-text corpus**:

- 95-96 % web prose by chars
- <0.2 % real math
- <0.1 % real code
- 0.9-1 % structured (lists, vocabularies, headlines)
- Tiny HTML / markdown fragments

This is consistent with the qualitative samples our d=12 models produce:
fluent web prose, no math reasoning, no code completion. Capabilities
we *do* see (CORE arc_easy 0.36, piqa 0.34, winograd 0.22) all rely on
general-knowledge web text.

## 4. Implications for v3+ data planning

If we want to push CORE beyond ~0.18-0.20 on a single V100, **adding
more ClimbMix is just adding more prose**. The token:param ratio gain
saturates because the model has already learned the prose distribution
well at the 0.85 val_bpb level.

The interventions that *would* change capability mix on this corpus are
**external mixes**, not deeper sampling of ClimbMix:

| Target capability | Suggested external mix | Cost |
|---|---|---|
| Code (HumanEval-like) | `bigcode/the-stack-v2` slice, ~50-200 M tok | new tokenizer? |
| Math (GSM8K-like) | `EleutherAI/proof-pile-2`, OpenWebMath | LaTeX tokenization |
| Reasoning bump | StarCoder QA, Wikipedia | minimal |
| Long-context | books (Project Gutenberg) | seq_len budget |

None of these are in scope for the current single-V100 sweep, but they
are the natural next move if v3 confirms that prose-only ClimbMix
plateaus around CORE 0.18.

## 5. How to reproduce

```bash
# 28,800 docs sampled (300 × 96 train shards), random seed 42:
python /tmp/classify_climbmix_v2.py \
  --dir $NANOCHAT_BASE_DIR/base_data_clean \
  --per-shard 300 --seed 42
```

Output JSON at `/tmp/classify_summary_v2.json` includes per-category
example docs and per-shard counts. The classifier (`/tmp/classify_climbmix_v2.py`)
is heuristic — it intentionally trades recall for precision on `code`,
errs on inclusion for `math` (USD-price false positive noted above), and
treats everything else as prose.

## 6. Caveats

- The classifier is heuristic, not learned. Real code embedded in long
  prose docs (e.g. "here's how to write a for loop in Python: ...") is
  classified as prose because the code spans <30 % of indented lines.
  True code-by-substring fraction may be a few % higher.
- The math false-positive (USD prices) means the absolute math fraction
  is likely 0.1-0.2 %, not the 0.85 % reported by the doc counter.
- No measurement of language: we assume English. ClimbMix is overwhelmingly
  English, but a small non-English tail probably exists.
- We did not bin by topic (news vs forum vs Wikipedia). A topic
  classifier was out of scope.
