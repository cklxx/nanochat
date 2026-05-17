# Best-Practice Recipe for Minimum-Emergence Sweep on 1×V100

> *Research summary before launching the v2 sweep targeting CORE ≥ 0.20.*

## 1. 数据 (Data) — 不动

**ClimbMix 是当前 nanochat 已知最优数据集**，连作者 Karpathy 自己尝试 5 次替换 FineWeb-Edu 都没胜出，最终是 NVIDIA Nemotron-ClimbMix 让 nanochat GPT-2 speedrun 时间从 2:46 缩到 2:01 (–27%)。([`dev/LOG.md` 2026-03-04 entry](../LOG.md))

业界对比 (2025-2026):
- **SmolLM2-135M** ([HF blog](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)) 混合用了 FineWeb-Edu + DCLM + Stack + 滤后小数据集，2T token 训练
- **DCLM** vs **FineWeb-Edu**: DCLM 在等 token 预算下普遍更优 ([llm.c discussion](https://github.com/karpathy/llm.c/discussions/664))
- **ClimbMix** 是 NVIDIA 基于 CLIMB 框架自动迭代搜出的混合配方，作者声明"等 token 预算下超过 DCLM" ([HF](https://huggingface.co/papers/2504.13161))

**结论:** 数据保持 ClimbMix。我们已下载 7 个 shard (~1.75B chars ≈ 440M tokens with vocab=8K)，对小模型多 epoch 复用足够。

## 2. Tokenizer — 改小

**当前 vocab=32K 在我们的尺度上让 embedding 吞掉大半参数预算**:

| Config | depth=4 emb参数 | d=4 transformer | 比例 |
|---|---:|---:|---:|
| vocab=32K (原) | 25 M | 3 M | 8 : 1 |
| **vocab=8K (新)** | **6 M** | **3 M** | **2 : 1** |

vocab=8K 在 fwe-val 上压缩比 3.96 bytes/token (vs 32K 的 4.62) — 损失 ~16% 数据效率，但换来 4× 的 embedding 缩减，让 IsoFLOP 最优点能从 d=4 移开。

**已训好 `tokenizer_v8k/`，43 秒，已 commit。**

业界小模型 vocab 选择 (参考):
- SmolLM2-135M: **49,152** (大，但因为它有 2T token 预算)
- Qwen3-0.6B: **151,936** (overtrain 60,000:1, 极端规模)
- GPT-2 (124M): 50,257
- 我们 36M 参数尺度 + 几百 M tokens: **8K 是合适的折中**

## 3. Token : Param 比例 — 不绑 Chinchilla

经典 Chinchilla 20:1 是**算力最优**，但现代小模型业界都做**推理最优** = 重 overtraining:

| 模型 | tokens/param | 来源 |
|---|---:|---|
| Chinchilla 算力最优 | 20 | DeepMind 2022 |
| **nanochat (Muon 优化) 实测最优** | **10.5** | nanochat dev/LOG.md |
| Llama-2-7B | 257 | (1.8T tok / 7B) |
| Llama-3-8B | 1,875 | (15T / 8B) |
| **SmolLM2-135M** | **14,815** | (2T / 135M) |
| Qwen3-0.6B | 60,000 | 极端 |

Sardana et al. 实验: loss 在 1000–10000 tokens/param 范围继续下降 ([Databricks](https://www.databricks.com/blog/how-long-should-you-train-your-language-model))。

**对我们 (单 V100, 24h, ~3e17–1e18 FLOPs 预算):**

| 配置 | params | optimal tokens (FLOPs/6N) | ratio |
|---|---:|---:|---:|
| 1e17 / d=4 / 5M eff | 5M | 3.3B | 660 |
| 3e17 / d=8 / 29M eff | 29M | 1.7B | 60 |
| 1e18 / d=8 / 29M eff | 29M | 5.7B | 200 |
| 1e18 / d=6 / 16M eff | 16M | 10.4B | 650 |

→ **重 overtraining 是我们这个尺度的正解**。d=4 / d=6 在 1e18 FLOPs 自然落到 200–650 tokens/param。

**实操:** 用 nanochat 的 `--target-flops`，让 num_iterations 自动算出来。不去手动 force `--target-param-data-ratio`。

## 4. 架构 + 超参 — 全部保留 nanochat 调好的默认值

Karpathy 在 [`dev/LOG.md`](../LOG.md) 里做了 ~320 实验调优 (d12→d20 三轮 sweep):

| 项目 | 值 | 出处 |
|---|---|---|
| aspect_ratio | 64 (model_dim = depth × 64) | "128 worse than 64, LLM prefers thinner+longer" |
| Value Embeddings | 开 | "models love them, all attempts to reduce fail" |
| Bigram embeddings | 关 | "small benefit, not justified at scale" |
| activation | ReLU² (非 SwiGLU) | "SwiGLU negative" |
| optimizer | Muon + AdamW per-group | Karpathy 320-exp sweep |
| weight_decay (Muon) | 0.28 + 线性 schedule | "cautious wd best" |
| warmdown_ratio | 0.65 | (d12-20 tuned) |
| batch size | auto B ∝ D^0.383 | Cerebras Power Lines |
| LR scaling | η ∝ √(B/B_ref) | AdamW theory |
| WD scaling | λ ∝ √(B/B_ref) · (D_ref/D) | T_epoch framework |

**注意一个负面信号**: Karpathy 报告 "d12 微调过的超参在 d20 反而伤性能"。意思是: nanochat 的默认值是有 **scale-aware muP 风格的自动 LR/wd/batch 缩放**，跨深度自动调整 — 我们 d=4-8 用默认值就行。

不改任何超参，把精力放在 **"算力 × 模型尺寸" 二维网格**上。

## 5. V100 现实算力评估

GPT-2 (CORE 0.256) 的训练算力对比:

| 平台 | 时间 | 等效 FLOPs |
|---|---|---|
| OpenAI 2019 (32× TPUv3) | 168 h | ~2.4 × 10²¹ |
| nanochat-modern (8× H100, fp8) | 1.65 h | ~4 × 10¹⁹ |
| **1× V100 fp16 (我们)** | 24 h | **~3 × 10¹⁸** ← |

V100 fp16 张量核 ~125 TFLOPS 峰值, 30-50% MFU 算来 ≈ 37 TFLOPS 持续 → 24h × 37e12 × 3600 ≈ **3.2 × 10¹⁸ FLOPs**。

这是 nanochat-modern-GPT-2 的 **1/12**。

按 L(C) ∝ C^-0.168 (我们已拟合) 外推:
- 当前 best val_bpb = 1.08 @ 3e16 FLOPs (vocab=32K)
- 期望 val_bpb @ 1e18 ≈ 1.08 × (3e16/1e18)^0.168 = **0.71** (理论上)
- 期望 val_bpb @ 3e18 ≈ **0.61**

GPT-2 baseline 是 val_bpb=0.748 + CORE=0.256。

→ **理论上 24h V100 可以接近 GPT-2 val_bpb**。但实际 CORE 不一定线性跟上 — emergence 是阈值现象，可能要更大模型才出。

## 6. 最终配方

**单点假设 (先验)**: 最小智能配置约 **d=6, vocab=8K, 1e18 FLOPs**。
- 总参数 ~ 17M
- 训练 tokens ≈ 1e18 / (6 × 17M) ≈ 10B → 22 epochs × 440M tokens
- 预计 wallclock ≈ 5h

**Sweep 设计 (按由小到大):**

| # | depth | FLOPs | 预估 wallclock | 预估 CORE |
|---|---:|---:|---:|---:|
| 1 | 4 | 1e17 | 30 min | ~0.05 |
| 2 | 6 | 3e17 | 1.5h | ~0.10 |
| 3 | 4 | 3e17 | 1h | ~0.08 |
| 4 | 6 | 1e18 | 5h | ~0.15 |
| 5 | 8 | 1e18 | 5h | ~0.15-0.18 |
| 6 | 6 | 3e18 | 15h ← 接近 24h 上限 | ~0.20? |

总预估: ~28h，**会超过 24h 预算**。所以需要边跑边 CORE eval，CORE ≥ 0.20 立刻停。

## 7. 风险声明

**CORE ≥ 0.20 在单 V100 24h 内不一定能达成**。理由:
1. nanochat-modern 用 8× H100 fp8 跑了 ~3.3 h 到 CORE 0.256，等效 ~26 V100-h。我们有 24 V100-h，差不多 1× 接近。
2. emergence 是阈值现象，scaling law 外推不保证连续提升
3. 改 vocab 引入了 ~16% 数据效率损失，需要稍多算力补偿

如果跑到 24h 还没到 CORE 0.20，我会:
- 记录最佳达到值 (CORE 多少 + 对应配置)
- 给出"补到 0.20 还差多少算力"的外推
- 明确说明在这台机器上下一步该做什么 (例如换 A100)

## 8. 决策清单

| 项目 | 决定 |
|---|---|
| 数据集 | ClimbMix 不变 |
| Tokenizer | **vocab=8K** (已训好) |
| 数据清洗 pipeline | 复用上次 (已 commit) |
| 架构 | nanochat 默认 (aspect 64, VE 开, ReLU², 全注意力) |
| Optimizer | nanochat 默认 (Muon+AdamW + auto-scale) |
| Token:param 比 | 不绑 Chinchilla, 用 `--target-flops` 自动 |
| Sweep 范围 | depth ∈ {4,6,8} × FLOPs ∈ {1e17, 3e17, 1e18, 3e18} |
| 停止条件 | CORE ≥ 0.20 或 24h 用完 |
