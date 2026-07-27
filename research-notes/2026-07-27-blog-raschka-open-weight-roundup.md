# Raschka「本周值得注意的几个开放权重模型」精读(2026-07-26)

- **Date:** 2026-07-27
- **Tags:** LLM架构, MoE, 开放权重, 注意力机制, 参数共享, LoRA, Raschka, blog-deep-dive
- **原文:** Sebastian Raschka, [A Few Notable Open-Weight Models This Week](https://sebastianraschka.com/blog/2026/notable-open-weight-models-this-week.html)(2026-07-26,Substack note 的轻编辑版)

## TL;DR

Raschka 一次点评 **6 个新开放权重模型**的架构。他自己都感叹 "Yes, one of those weeks!"——在大家都还在等 Kimi K3 与 Ling 3.0 权重的同时,这一周又冒出一批。六个模型恰好横跨了 2026 架构演进的**五条主线**:参数共享/循环、稀疏与混合 MoE、线性/delta 注意力混合、KV 压缩的 latent attention 变体、以及小型任务专用模型与 LoRA adapter 的复兴。

> 这篇是我维护的 [[2026-07-21-raschka-llm-architecture-comparison]](Raschka 架构横评活文档)的**增量更新**——其中 Motif-3-Beta 我已在那篇的"补充二"里收录,这次拿到了 Raschka 对其 GDLA 机制的解读。

## 六个模型逐个看

### 1. Nanbeige 4.2 3B —— 「looped depth sharing」

- **机制:** 同一个 **22 block 的 stack 跑两遍**,得到 **44 个有效层但不增加权重**——"2× 的 transformer block 计算量,同样的显存占用"。
- **为什么是两遍:** 技术报告 §2.1 说两遍是最佳权衡,保留了标准架构 **约 75% 的 token efficiency**;再加更多遍收益微弱,却让训练更慢更贵。
- **归类:** 参数共享/循环计算路线(与我此前追踪的 Ouro looped LM 同族)。

### 2. Laguna S 2.1(poolside) —— Raschka 私心最爱

- **规格:** **118B 稀疏 MoE / 8B 激活 / 1M token 上下文**。
- **注意力:** **36 层滑窗 + 12 层 global (gated-)GQA**;其余"相当标准"。
- **Raschka 的私人角度:** "能在我的 DGX Spark 上跑(占用 **< 80 GB RAM**)",是"对我个人而言最有意思的模型",可能取代他日用的 Qwen3.6-35B——虽然"大 3 倍因而稍慢"。他补了句克制的话:"还在等更多独立性能基准。"
- **意义:** 1M 上下文 + 8B 激活 + 80GB 可跑,是"本地可用旗舰"的一个新标杆。

### 3. Motif-3-Beta —— GDLA(分组差分潜注意力)

- **规格:** **314B-A13B 稀疏 MoE**,部分借鉴 **DeepSeek V4** 的 mHC 与 latent attention。
- **GDLA(Grouped Differential Latent Attention),灵感来自 MLA:**
  - 标准 MLA 把 K/V 压缩成 latent 形式,主要为**缩小 KV cache**;
  - GDLA 同样做低秩压缩,但**把注意力头分组**,并**为每组学一个 noise head,再把这个 noise 减掉做过滤**。
- **解读:** 这是把 **differential attention(差分注意力)** 的"减噪"思想与 **latent attention 的压缩**合到了一处——我在架构横评笔记里标注过"Raschka 全文未提差分注意力,这是新增维度",现在他自己补上了这一维。

### 4. Solar Open 2(Upstage) —— 混合注意力

- **规格:** **250B-A15B 混合 MoE**。
- **层模式:** **每 3 层 Kimi Delta Attention(KDA)交错 1 层 GQA**。
- **解读:** KDA 从 Kimi Linear(48B)→ Kimi K3(2.8T)→ 现在被**第三方(Upstage)采用**,说明 delta 类线性注意力正在成为可复用组件,而非单家特色。

### 5. Antares 1B(Cisco) —— 小模型的任务专用后训练

- **规格:** 1B(另有 0.3B 变体),建在 **IBM Granite 4.0 1B** 骨干上。
- **后训练:** **SFT + GRPO,专攻终端环境下的网络安全任务**。
- **Raschka 评价:** "在真正小的模型上做任务专用后训练的好例子。"

### 6. BTL-3 —— LoRA 在 2026 依然有用

- **不是完整模型:** 而是给 **Qwen3.6-27B 的 rank-32 LoRA adapter**,面向编码 agent 与结构化工具调用。
- **结论:** 基准成绩不错,说明 **"LoRA adapter 在 2026 仍是有用的工具/技术"**。

## 横向对比

| 模型 | 规格 | 关键架构点 | 归属主线 |
|---|---|---|---|
| Nanbeige 4.2 | 3B 密集 | 22 block 跑 2 遍 = 44 层 | 参数共享/循环 |
| **Laguna S 2.1** | 118B-A8B, 1M ctx | 滑窗:global = 36:12 | 稀疏 MoE + 滑窗混合 |
| **Motif-3-Beta** | 314B-A13B | **GDLA**(分组 + noise head 过滤) | latent attn + 差分注意力 |
| Solar Open 2 | 250B-A15B | **KDA:GQA = 3:1** | 线性/delta 混合 |
| Antares 1B | 1B / 0.3B | Granite 骨干 + SFT+GRPO | 小模型任务专用 |
| BTL-3 | rank-32 LoRA | Qwen3.6-27B 适配器 | Adapter 复兴 |

## 我的反思

1. **"周更级"开源发布已成常态。** Raschka 这种"一周点评六个"的格式本身就是信号——2026 的开源模型密度已经高到无法逐个写长文,只能做批量架构速览。这也印证了我那篇架构横评必须做成**活文档**的判断。

2. **注意力创新在"混合比例"上收敛。** 本批三个模型给了三种比例:Laguna 36:12(滑窗:global)、Solar Open 2 的 3:1(KDA:GQA)、以及此前 Gemma 4 / MiMo 的 5:1。**没人再用纯全局注意力,但混什么、混多少还完全没有共识**——这正是 Raschka 那条"线性注意力钟摆"主线的延续。

3. **GDLA 值得单独盯。** 把差分注意力的减噪(为每组学 noise head 再减掉)嫁接到 MLA 的压缩上,是个组合创新。若 Motif 正式版验证有效,可能引发一波"latent + differential"的跟随。

4. **小模型 + adapter 的路线没死。** Antares 1B(网安专用)与 BTL-3(LoRA)与那些百亿千亿 MoE 同周发布,提醒我们**能力前沿之外还有巨大的"任务专用"市场**,且 LoRA 这种 2023 的技术在 2026 仍然够用。

5. **Raschka 的"本地可跑"标准很实用。** 他判断模型的第一反应是"能不能在我的 DGX Spark 上跑(<80GB)"——这个视角比刷榜更贴近开发者真实决策。

## Open Questions

- Laguna S 2.1 的独立基准何时出?1M 上下文是架构能力还是实际可用(与 GLM-5.2 同样的疑问)?
- Nanbeige 的 looped depth sharing 保留 75% token efficiency——这个折损在更大规模下会放大还是收窄?
- GDLA 的 noise head 到底过滤掉了什么?与 differential transformer 原始论文的机制差异有多大?
- KDA 被第三方采用(Solar Open 2)后,会不会像 RoPE/GQA 一样成为标准组件?

## 引用关系与跟进路径

- **上游:** MLA(DeepSeek V3)、differential attention、Kimi Linear 的 KDA、DeepSeek V4 的 mHC
- **本仓库关联:** [[2026-07-21-raschka-llm-architecture-comparison]](架构横评活文档,Motif-3 已收录)、[[2026-07-17-inkling-glm52-kimik3]](三大开放模型)、[[2026-07-27-hf-daily-papers-jul25-27]](本周论文)
- **跟进:** Raschka 把这六个都加进了他的 [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/);待 Kimi K3 / Ling 3.0 权重释出后应有新一轮更新
- **arXiv 入库:** 本文提及的模型多为技术报告/模型卡而非 arXiv 论文(Nanbeige 4.2 技术报告有 §2.1 但原文未给 arXiv id),**故本次无新增 arXiv 入库**;Motif-3 相关已在架构横评笔记中记录

> 引用须可验证:以上架构细节均引自 Raschka 原文;未凭记忆补充原文未提及的参数(他明确没写归一化与位置编码,本笔记也不臆测)。
