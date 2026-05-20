# Architecture Comparison: nanochat d=12 vs SmolLM2-135M

> *Same parameter count (~135 M), same target capability range, but
> drastically different training budgets and architectural choices.
> SmolLM2-135M is the strongest public open-weight model at this
> size and serves as the SOTA reference point for our V100-scale
> experiments.*

## 1. Side-by-side architecture

Source: HuggingFace `HuggingFaceTB/SmolLM2-135M/config.json` + nanochat
`base_checkpoints/v100_emerge_v8k_1.5e18_d12_xdata/meta_003697.json`.

| Dim | nanochat d=12 (our v2 best) | SmolLM2-135M |
|---|---|---|
| Family | nanochat (GPT-style) | **Llama 3.x** |
| Params (total) | 135.27 M | **134.83 M** |
| **n_layer** | **12** | **30** (2.5× deeper) |
| hidden_size | 768 | 576 |
| n_attention_heads | 6 | 9 |
| n_kv_heads | 6 (full attention) | **3 (GQA, 3:1 ratio)** |
| head_dim | 128 | 64 |
| intermediate_size | ~2048 (ReLU² FFN) | **1536 (SwiGLU)** |
| activation | **ReLU²** (Karpathy-tuned) | **SiLU/SwiGLU** (mainstream) |
| normalization | RMSNorm | RMSNorm (rms_norm_eps=1e-5) |
| vocab | **8 192** (we shrank from 32 K) | 49 152 |
| max ctx | 2 048 | 8 192 |
| RoPE θ | (nanochat default) | 100 000 |
| **tie_word_embeddings** | **False** (separate lm_head) | **True** (saves 28 M) |
| **value_embeddings** | **True** (37.7 M extra) | **False** |
| optimizer | **Muon + AdamW per-group** | AdamW |
| precision | float16 (V100) | bfloat16 (H100) |
| attention impl | SDPA fallback | scaled_dot_product_attention |
| dropout | 0 | 0 |

## 2. Parameter-budget breakdown

How each architecture allocates the same 135 M:

### nanochat d=12 (our model)

| Component | Params | % of total |
|---|---:|---:|
| wte (token embedding) | 6.29 M | 4.65 % |
| **value_embeds** (6 per layer × 768 × 8192) | **37.75 M** | **27.9 %** |
| lm_head (separate) | 6.29 M | 4.65 % |
| Transformer matrices (Q/K/V/O + MLP × 12) | 84.94 M | 62.8 % |
| Scalars (norms etc.) | 0.00 M | <0.01 % |
| **Total** | **135.27 M** | 100 % |

### SmolLM2-135M (Llama style)

| Component | Params | % of total |
|---|---:|---:|
| Tied embedding (used as both wte and lm_head) | 28.31 M | 21.0 % |
| Per layer: Q=576², KV=576×192, O=576², MLP=3×576×1536 | 3.55 M | (×30) |
| Transformer matrices (30 layers × 3.55 M) | 106.49 M | 79.0 % |
| Final RMSNorm | <0.01 M | — |
| **Total** | **~134.8 M** | 100 % |

### Implications

1. **SmolLM2 has 1.25× more transformer capacity per dollar**: 106 M
   transformer matrices vs our 85 M. We spent 28 % of the budget on
   value embeddings (a Karpathy 320-experiment winner at GPT-2 scale)
   that SmolLM2 skips entirely.
2. **SmolLM2 is 2.5× deeper** (30 layers vs 12). Deep+narrow is the
   modern small-model recipe (Pythia, OpenELM, MobileLLM all do this).
3. **SmolLM2 uses GQA 3:1** — modest KV memory savings, not a big
   capacity win at this size. We don't use it.
4. **Vocab gap matters**: 49 K vs 8 K. SmolLM2's bigger vocab gives
   ~16 % better bytes/token compression — but they could afford it
   because tied embeddings + deep architecture mean the embedding
   table isn't dominating their budget.
5. **Activation: ReLU² (ours) vs SwiGLU (SmolLM2)** — Karpathy's LOG
   notes ReLU² beat SwiGLU in his 320-experiment sweep at GPT-2 scale.
   That decision is one of the few places we diverge from "modern
   small-model defaults" and it's deliberate.

## 3. Training-budget gap

This is the elephant in the room.

| | nanochat d=12 xdata | SmolLM2-135M |
|---|---|---|
| Compute | 1.5 × 10¹⁸ FLOPs | ~9 × 10²⁰ FLOPs (est.) |
| Hardware | 1 × V100 fp16, 8.8 h | 64 × H100 bf16, "weeks" |
| Pretraining tokens | **1.94 B** | **2 000 B** = 2 T |
| **Token : param ratio** | 14.3 | **14 815** |
| Data | ClimbMix 400 B (96 % web prose) | FineWeb-Edu + DCLM + The Stack + filtered datasets |

SmolLM2 saw **1 031 × more tokens** than our d=12. That's the dominant
explanation for the gap on every benchmark — at fixed architecture
choices, what changes scoring at this scale is **how many tokens the
parameters get to see**, not Muon vs AdamW or ReLU² vs SwiGLU.

## 4. Benchmark gap (vs our v2 best CORE 0.1414)

SmolLM2-135M official numbers (raw accuracy, not centered):

| Task | SmolLM2-135M | nanochat d=12 xdata raw | gap |
|---|---:|---:|---:|
| HellaSwag (10-shot) | 42.1 | 37.0 | +5.1 pp |
| ARC easy + challenge (avg) | 43.9 | ~37 | +7 pp |
| PIQA | 68.4 | 62.5 | +5.9 pp |
| MMLU (cloze, 5-shot) | 31.5 | not measured | n/a |
| CommonsenseQA | 33.9 | 29.5 | +4.4 pp |
| Winogrande | 51.3 | 50.0 | +1.3 pp |
| OpenBookQA | 34.6 | 30.5 | +4.1 pp |

A SmolLM2-135M-on-our-22-task-CORE-suite eval (using
`scripts.base_eval --hf-path HuggingFaceTB/SmolLM2-135M`) is the next
step. Centered CORE for SmolLM2-135M should land in the **0.20-0.25**
range based on the published HellaSwag / ARC / PIQA numbers, which
would put it solidly above our v2 best (0.1414) and approaching
GPT-2-1.6B's 0.256.

## 5. What this tells us for v3 planning

1. **Architecture changes alone can't close the gap.** Switching to
   Llama-style (drop value_embeds, tie embeddings, deepen to 30 layers,
   SwiGLU, GQA) might buy us ~+1-2 % on capability at fixed compute,
   but the **gap is mostly tokens**.
2. **The "data is the bottleneck" reframing.** Even if we max our
   V100 at d=12 and 3-4 e18 FLOPs, we'd still be running ~3-4 B
   tokens — three orders of magnitude below SmolLM2. To get close
   we'd need either: (a) a much bigger compute budget, or (b) better
   *signal per token* via curriculum / quality filtering / external
   high-density mixes.
3. **Karpathy's specific choices (ReLU², value_embeds, Muon) are
   probably right at our compute regime** but become less defensible
   as compute grows — the value_embed 28 % parameter overhead is a
   *gift* at our overtraining regime but a *tax* on SmolLM2's
   undertraining regime.

## 6. Reproducing the eval

```bash
# On the V100 host, after d=14_cont finishes:
NANOCHAT_DTYPE=float16 \
NANOCHAT_TOKENIZER_DIR=$HOME/.cache/nanochat/tokenizer_v8k \
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat \
http_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
https_proxy=http://sys-proxy-rd-relay.byted.org:8118 \
python -m scripts.base_eval \
    --hf-path HuggingFaceTB/SmolLM2-135M \
    --eval=core,bpb,sample \
    --device-batch-size=4 \
    > $HOME/.cache/nanochat/scaling_v100_emerge/eval_smollm2_135m.log 2>&1
```

The base_eval script accepts `--hf-path` directly. The 22-task CORE
pipeline is identical to what we ran on our checkpoints, so the
resulting score is directly comparable.

## 7. References

- HF model card: <https://huggingface.co/HuggingFaceTB/SmolLM2-135M>
- Paper: Allal et al., "SmolLM2: When Smol Goes Big" arXiv:2502.02737
- Training corpus: FineWeb-Edu, DCLM, The Stack (subsets)
- License: Apache 2.0
- See `DATA_COMPOSITION.md` for our ClimbMix profile (96 % web prose
  by chars, <0.2 % math, <0.1 % code — SmolLM2's mix has explicit
  code from The Stack which ours doesn't).
