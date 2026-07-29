# Raschka《The Big LLM Architecture Comparison》总结 + 2026-07 新模型补充

- **Date:** 2026-07-21
- **Tags:** LLM架构, MoE, 注意力机制, MLA, 稀疏注意力, 线性注意力, 归一化, Raschka, 综述
- **原文:** Sebastian Raschka, "The Big LLM Architecture Comparison"(2025-07-19 首发,持续更新,截至 2026-04-02 更新到 Gemma 4 / 第 23 节)

## Context

Raschka 这篇是**活文档式**的架构横评——不比跑分,只拆**架构选择**:同样是 Transformer,2025–2026 这批开源模型在注意力、MoE、归一化、位置编码上到底改了什么。核心提问贯穿全文:

> 这些究竟是**突破性变化**,还是我们只是在**打磨同一套架构地基**(RoPE / GQA / SwiGLU 的微调)?

本笔记先按原文总结覆盖的模型与主线(1–23 节,原文更新至 Gemma 4 / 2026-04),再用他的架构词汇**补上原文尚未覆盖的更新模型**——包括 7 月三大新模型(Inkling / GLM-5.2 / Kimi K3),以及经 HF trending 核实、原文遗漏的旗舰(**DeepSeek-V4-Pro**、Tencent Hy3、Motif-3),覆盖至 **2026-07-21**。与 [[Inkling / GLM-5.2 / Kimi K3 三大开放模型深度调研]] 一篇互为补充。

## 五条贯穿全文的主线

1. **注意力效率光谱**:MHA → GQA → MLA → 滑窗(sliding-window)→ 线性/混合注意力。
2. **MoE 全面复兴**:2025 起几乎人手一个 MoE,内部争论两点——**「少而大」vs「多而小」专家**、**要不要共享专家(shared expert)**。
3. **归一化位置**是反复被动用的稳定性杠杆:Pre-Norm / Post-Norm / QK-Norm。
4. **线性注意力复兴**(MiniMax-M1、Qwen3-Next、DeepSeek V3.2、Kimi Linear),而 **MiniMax-M2 又反向退回 full attention**——是本主线里最耐人寻味的反转。
5. **宽 vs 深**的取舍(gpt-oss 偏宽 vs Qwen3 偏深)。

## 覆盖模型与架构要点(按原文顺序)

| # | 模型 | 注意力 | MoE | 归一化/其它亮点 |
|---|---|---|---|---|
| 1 | **DeepSeek V3/R1** | **MLA**(K/V 压缩到低维再缓存,称优于 GQA) | 671B/37B,256 专家,1 共享+8 路由 | 奠定后续多家基座 |
| 2 | **OLMo 2** | MHA(32B 变体用 GQA) | 密集 | **Post-Norm** + **QK-Norm** 稳训练 |
| 3 | **Gemma 3** | 滑窗+GQA,窗口 4096→**1024**,比例 **5:1** | 密集 | Pre+Post-Norm 并用;3n 有 PLE + MatFormer |
| 4 | **Mistral Small 3.1** | 标准 GQA,**弃用滑窗** | 密集 | 砍 KV cache 与层数降延迟 |
| 5 | **Llama 4** | MoE+GQA,MoE/密集隔层交替 | 400B/17B,**2 活跃专家×8192** | 与 DeepSeek「多而小」相反,走「少而大」 |
| 6 | **Qwen3**(密集+MoE) | GQA | 30B-A3B / 235B-A22B,8 专家,**无共享专家** | 比 Llama 3 更深 |
| 7 | **SmolLM3** | **NoPE**(每 4 层一次,无位置信号) | 密集(3B) | 靠因果掩码保序,长度泛化更好 |
| 8 | **Kimi K2 / Thinking** | MLA(比 DeepSeek 头更少) | **1T** 参数,专家更多 | **Muon 优化器**;Thinking 上下文 128k→256k |
| 9 | **GPT-OSS**(20b/120b) | 隔层滑窗 | **32 专家、4 活跃(少而大)** | 更宽(emb 2880);attention bias + **attention sinks** |
| 10 | **Grok 2.5** | — | 270B,8 个大专家 | 常开的类共享专家 SwiGLU(中间维翻倍) |
| 11 | **GLM-4.5**(+Air 106B) | 带 attention bias | 355B,含共享专家 | **MoE 前先放 3 个密集层**稳训练 |
| 12 | **Qwen3-Next** | **Gated DeltaNet + Gated Attention 混合 3:1** | 80B-A3B,更多专家+共享 | 原生 262k;**MTP** 多 token 预测 |
| 13 | **MiniMax-M2** | **退回 full attention**(放弃 M1 的 lightning) | ~230B/10B(比 Qwen3 更稀疏 2×) | 每头独立 QK-Norm;partial RoPE;无共享专家 |
| 14 | **Kimi Linear**(48B) | **KDA**(Gated DeltaNet 通道门控)线性:full = 3:1 | — | full 层用带门 MLA;MLA 层用 NoPE |
| 15 | **Olmo 3 Thinking** | 7B(MHA+滑窗)/ 32B(GQA) | — | Post-Norm;仅全局层用 YaRN 扩到 64k |
| 16 | **DeepSeek V3.2** | V3 + **稀疏注意力**提效(与 GPT-5.1/Gemini 3 Pro 部分基准持平) | 同 V3 | 单独长文详解 |
| 17 | **Mistral 3**(Large 673B/39B) | 与 DeepSeek V3 **完全相同架构** | 专家×2 大、数量÷2(少而大) | Mistral 自 Mixtral 后首个 MoE;自有 tokenizer,与 NVIDIA 合作优化 Blackwell 吞吐 |
| 18 | **Nemotron 3 Nano/Super**(NVIDIA) | **Mamba-2 + Transformer 混合**,仅少数层用 GQA | Nano 30B-A3B(128 专家,1 共享+6 路由);Super 120B-A12B | **latent MoE**(4096→1024→4096)+ **MTP 推理期投机解码**;数据+代码全开 |
| 19 | **Xiaomi MiMo-V2-Flash** | **滑窗 5:1**,窗口仅 **128**(比 Gemma 3 的 1024 小 8×,史上最大滑窗模型) | 309B/15B MoE | MTP;性能对标 DeepSeek V3.2、参数减半 |
| 20 | **Arcee AI Trinity Large**(400B/13B) | 局部:全局滑窗 **3:1**、窗口 4096(似 Olmo 3)+ **gated attention** | DeepSeek 式多小专家、调粗提吞吐 | **QK-Norm + 全局层 NoPE**;**depth-scaled sandwich norm**(4 个 RMSNorm,第二个增益按 1/√L 初始化) |
| 21 | **GLM-5**(z.AI,744B/40B) | **MLA + DeepSeek 稀疏注意力(DSA)** | **256 专家**(GLM-4.5 为 160),含共享专家 | emb 5120→**6144**、层数 **92→78**(减层降推理成本);基准对标 GPT-5.2/Gemini 3 Pro/Claude 4.6 Opus |
| 22 | **Jan–Feb 2026 十连发** | — | — | Kimi K2.5、Step 3.5 Flash、Qwen3-Coder-Next、**MiniMax M2.5**、**Qwen 3.5**、蚂蚁 **Ling/Ring 2.5 1T**、Cohere Tiny Aya…(单独长文) |
| 23 | **Gemma 4**(31B / MoE 26B-A4B) | **GQA + 滑窗 5:1**(几乎同 Gemma 3);全局层 **values=keys** 复用降 KV;**p-RoPE**(仅 25% 频率对带位置) | 另有 MoE 变体 | 独特 Pre+Post-Norm 保留;架构几无变、但基准较 Gemma 3 大跃升,31B 对标 Qwen3.5-397B |

### 几个最值得记的细节

- **MLA vs GQA**:DeepSeek 把 K/V 压缩到低维再缓存,Raschka 认为建模质量优于 GQA——这是 2025 注意力侧最有影响的选择,Kimi K2 直接沿用。
- **「少而大」vs「多而小」专家**:DeepSeek/Qwen3 走多而小(8–256 个小专家),Llama 4 / GPT-OSS / Grok 走少而大(2–8 个大专家)。没有定论,是当前 MoE 设计的主要分歧。
- **共享专家的去留**:DeepSeek/GLM-4.5/Grok 保留;Qwen3/MiniMax-M2 不用。Qwen3 开发者自己都说"不确定"。
- **线性注意力的钟摆**:MiniMax-M1 上了 lightning attention,M2 又退回 full attention——说明线性注意力在长上下文提效和质量间还没稳态。
- **Post-Norm + QK-Norm**(OLMo 2)是低调但有效的稳训组合,被多家借鉴。

---

## 补充:2026-07 三大新模型(原文未覆盖)

Raschka 更新到 **Gemma 4(2026-04,第 23 节)**,而 **7 月中旬两周内又出了三个旗舰开放权重模型**,正好在原文覆盖范围之外。这里用他的架构词汇把它们接上(细节见 [[Inkling / GLM-5.2 / Kimi K3 三大开放模型深度调研]])。

> 衔接点:Raschka 第 21 节已详解 **GLM-5(744B/40B,MLA+DSA,256 专家,层数 92→78)**——它正是我们 7 月调研的 **GLM-5.2(753B)的直接前身**,架构同系微调放大。所以下面 GLM-5.2 那行可直接继承他对 GLM-5 的拆解。

| 模型 | 注意力 | MoE | 归一化/亮点 | 放进 Raschka 框架 |
|---|---|---|---|---|
| **Inkling**(TML,975B/41B) | 滑窗+全局 **5:1**、8 KV heads、**相对位置编码**(称优于 RoPE) | 256 路由+2 共享,激活 6;**多而小** | K/V 投影后加短卷积;encoder-free 多模态 | 沿 DeepSeek-V3 式「多而小 + 共享专家」;滑窗比例同 Gemma 3 |
| **GLM-5.2**(智谱,753B) | **DSA(稀疏)+ IndexShare** 跨层复用 indexer | 256 专家(GLM-5 系),含共享专家 | MoE 前置密集层(承 GLM-4.5) | **稀疏注意力主线**延伸:比 DeepSeek V3.2 更进一步做「跨层索引复用」 |
| **Kimi K3**(月之暗面,2.8T) | **KDA(Kimi Delta Attention)** + AttnRes | 16/896 专家激活,Stable LatentMoE | **Per-Head Muon**、SiTU、Gated MLA;自 SFT 起 MXFP4/8 QAT | 直接把 [[Kimi Linear]] 的 KDA 从 48B 推到 **2.8T**;延续 Kimi 的 Muon 传统 |

### 补充二:Raschka 4月后至今的其它旗舰(HF trending 核实,截至 2026-07-21)

除 7 月三大模型外,查 HF trending 又发现几个原文未收、架构上有料的旗舰级开放模型(数据来自各自 HF 模型卡/config,非编造):

| 模型 | 参数 | 注意力 | MoE | 亮点 / 放进 Raschka 框架 |
|---|---|---|---|---|
| **DeepSeek-V4-Pro** ⭐ | **1.6T / 49B** | **CSA + HCA 混合**(Compressed Sparse + Heavily Compressed Attention) | FP4 专家 + FP8 | **正是 Raschka 第 23 节"还在等的 DeepSeek-V4"——已发布!** 1M 上下文下推理 FLOPs 仅 V3.2 的 **27%**、KV cache 仅 **10%**;mHC 超连接 + Muon;MIT |
| **Tencent Hunyuan Hy3** | 295B / 21B | GQA(8 KV heads) | **192 专家 top-8** | 80 层 + 1 MTP 层(投机解码);`reasoning_effort` 可调(no_think/low/high);256K;Apache-2.0 |
| **Motif-3-Beta** | ~314B / 13B | **GDLA(Grouped Differential Latent Attention)** | **384 专家 top-8 + 1 共享** | 自研差分+latent 混合注意力;Grouped PolyNorm、Modified mHC;256K;仅非商用 |

**这几个把 Raschka 的主线又往前推了一步**:
- **DeepSeek-V4-Pro** 延续"稀疏/压缩注意力"主线到极致(CSA+HCA 双压缩),并引入 **mHC(流形约束超连接)**——残差连接的新变体,Motif-3 也用了 Modified mHC,值得盯。
- **差分注意力落地**:Motif-3 的 GDLA 把 differential attention 和 latent attention 合起来——Raschka 全文未提差分注意力,这是新增维度。
- **MTP + reasoning_effort 可调**成了国产 MoE 标配(Hy3、Nemotron 3 Super、Kimi K3 都有)。

### 补充三:原文遗漏的其它代表性系列(按厂 namespace 核查)

进一步逐厂核查 HF namespace,又发现几个**代表性系列**原文/前面补充里没收(数据来自 HF 模型卡/config):

| 模型 | 参数 | 注意力 | MoE | 放进 Raschka 框架 |
|---|---|---|---|---|
| **Qwen3.5 全家**(2026-02) ⭐ | 密集 0.8/2/4/9/27B + MoE **35B-A3B / 122B-A10B** | GQA | 有密集也有 MoE | **原文只到 Qwen3;Qwen3.5 已是当前最主流基座**(满屏衍生:Ornith/Bonsai/Qwythos 都基于它);现已多模态(image-text-to-text) |
| **Qwen3.6-27B** | 27B | — | — | 更新的密集旗舰(Ternary-Bonsai 等基于它) |
| **Mistral Small 4**(119B-A6.5B) | 119B/**6.5B** | **MLA**(FLASH_ATTN_MLA) | **128 专家 top-4** | **Mistral 从 Mistral 3 的 dense/DeepSeek式转向自己的 MLA+多小专家 MoE**;统一 Instruct/Reasoning/Devstral 三家,reasoning_effort 可调;256K |
| **Mistral Medium 3.5**(128B) | 128B | — | mistral3 架构 | 闭源权重级中端;EAGLE 投机解码变体 |
| **Qwen-AgentWorld-35B-A3B** | 35B-A3B | 基于 Qwen3.5-MoE | — | Qwen 的**世界模型/环境模拟**方向(arXiv:2606.24597) |

**已确认没有新主力的**(避免误判遗漏):
- **Meta Llama**:namespace 最新仍是 **Llama 4** Maverick/Scout(2025-04)+ Guard 安全模型,**无 Llama 5**。Llama 4 已在原文第 5 节。
- **OpenAI**:最新仍是 **gpt-oss-20b/120b**(2025-08)+ safeguard 变体,**无新 gpt-oss**。已在原文第 9 节。

> Raschka 明说全文"不是穷举,只挑亮点";上面这些补录同理——**只收有明确架构数据的旗舰/代表系列,finetune 与量化版一律排除**。

### 补充四:再扫一轮各厂 namespace(Google / MiniMax / Moonshot / xAI)

| 模型 | 参数 | 注意力 | MoE | 放进 Raschka 框架 |
|---|---|---|---|---|
| **MiniMax-M3** ⭐ | 428B/23B | **MSA(MiniMax Sparse Attention)** | MoE | **注意力钟摆再摆一次!** M1 线性→M2 退回 full→**M3 又转稀疏**;1M 上下文 prefill 9×/decode 15× 提速(vs M2),per-token 算力降到 1/20;原生多模态 |
| **DiffusionGemma-26B-A4B** ⭐ | 25.2B/3.8B | **encoder-decoder**,decoder 对 canvas 做双向注意力 | 128 专家 8 活跃+1 共享 | **Google 的离散扩散语言模型**——与 LLaDA2.0-Uni 同路线;multi-canvas 并行去噪,block-autoregressive,1100+ tok/s(H100 FP8);⚠️ 准确率略逊标准 Gemma 4,拿精度换速度 |
| **MiniMax-M2.5 / M2.7** | (M2 系) | full attention(承 M2) | MoE | 原文第 13 节 M2 的迭代版 |
| **Kimi K2.5 / K2.6 / K2.7-Code** | (K2.5 架构) | — | MoE | Kimi 在 K2→K3 之间的密集迭代支线;K2.5 原文第 22 节提过 |

**已确认没有新主力的**:
- **xAI**:HF 上最新公开权重仍是 **Grok 2**(2025-08),**无 Grok 3+ 开放权重**。Grok 2.5 已在原文第 10 节。
- **Google 主力**:仍是 **Gemma 4**(原文第 23 节)+ 这次的 DiffusionGemma 支线,无 Gemma 5。

**两个新主线增量**:
1. **MiniMax 注意力三连跳**(M1 线性 → M2 full → M3 稀疏 MSA)是 Raschka"线性注意力钟摆"主线的最强注脚——同一家三代三种注意力。
2. **扩散语言模型进入大厂**:DiffusionGemma(Google)+ LLaDA2.0-Uni(蚂蚁)说明离散扩散不再是学术玩具,但都还在"拿精度换速度"阶段。

### 三点观察(补全 Raschka 的主线)

1. **稀疏/线性注意力主线在放大**:Raschka 收尾于 DeepSeek V3.2 的稀疏注意力与 Kimi Linear(48B)。GLM-5.2 的 **IndexShare** 把稀疏 indexer 的 O(L²) 冗余进一步做成**跨层复用**(1M 上下文 per-token FLOPs 降 2.9×);Kimi K3 则把 Kimi Linear 的 **KDA 从 48B 直接推到 2.8T**——证明线性注意力已敢上万亿规模主力模型。

2. **Muon 优化器从 Kimi 扩散出去**:原文提到 Kimi K2 用 Muon;现在 **Inkling(大矩阵权重用 Muon)、Kimi K3(Per-Head Muon)** 都在用,Muon 正从"Kimi 特色"变成通用选择。

3. **共享专家 + 前置密集层成为"稳训标配"**:Inkling(2 共享专家)、GLM-5.2(承 GLM-4.5 的 MoE 前置密集层)都延续了 Raschka 标记的稳训技巧;与不用共享专家的 Qwen3/MiniMax-M2 形成持续分歧,尚无收敛。

### 原文最新主线(17–23 节,agent-browser 抓全后补)

读全后半段,又浮现两条 Raschka 的新观察:

- **DeepSeek V3 架构成了"事实标准"**:Kimi K2(1T)、**Mistral 3 Large(673B,与 V3 完全相同架构,仅专家÷2 变大)**、Arcee Trinity、GLM-5 都在复用 DeepSeek V3 的 MLA+MoE 骨架。Raschka 原话:"why change what ain't broke?"——秘方越来越在训练管线与推理 scaling,而非架构本身。
- **混合/线性注意力从边缘走向主力**:Nemotron 3(Mamba-2 + Transformer 混合,仅少数层 attention)比 Qwen3-Next/Kimi Linear 更激进;Xiaomi MiMo 用史上最小滑窗(128);Gemma 4 的 p-RoPE(仅 25% 频率带位置)+ values=keys 复用。**MTP(多 token 预测)也从训练期扩到推理期做投机解码**(Nemotron 3 Super)。

### 待补

- ~~Kimi K3 技术报告出后补齐层数/tokenizer/KDA 设计~~ → ✅ **已补齐**([arXiv:2607.24653](https://arxiv.org/abs/2607.24653),2026-07-28):**93 层、2.78T/104.2B、69 KDA + 24 MLA、896 专家激活 16、SiTU-GLU、词表 160K、MoonViT-V2 401M 视觉塔、1M 训练上下文**。另有 **AttnRes**(用学习到的伪查询对所有前序 block 输出算注意力,重构深度方向信息流)——与 DeepSeek-V4 的 mHC、Motif-3 的 Modified mHC 同属"重构残差连接"这条新主线。详见 [[2026-07-29-hf-daily-papers-jul28-29]];**KDA 由此完成 48B → 2.8T 的跨两个数量级验证**;
- ~~DeepSeek-V4(Raschka 明确在等)~~ → **已发布 DeepSeek-V4-Pro**(见补充二),待读其技术报告 arXiv:2606.19348 补 CSA/HCA/mHC 细节;
- Motif-3 正式版(现为 beta)、Hunyuan Hy3 完整技术报告发布后更新。

## Open Questions

- 「少而大 vs 多而小」专家之争,到万亿规模(Kimi K2 1T、K3 2.8T)是否出现新证据?
- 线性/稀疏注意力(KDA、DSA、IndexShare)在 1M+ 上下文的质量-效率权衡,谁会成为事实标准?
- Muon 会不会全面取代 AdamW 成为大模型默认优化器?
- MiniMax-M2「退回 full attention」是个例,还是预示线性注意力在某些规模/任务上仍不成熟?

## References

- Sebastian Raschka, "The Big LLM Architecture Comparison," https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison(2025-07-19,更新至 2026-04-02)
- 本仓库:[[Inkling / GLM-5.2 / Kimi K3 三大开放模型深度调研]](research-notes/2026-07-17-inkling-glm52-kimik3.md)
- 相关:GLM-5 技术报告 arXiv:2602.15763、IndexCache arXiv:2603.12201

> 注:原文 1–23 节已用 agent-browser(headless Chromium)抓全;补充部分的三大新模型(Inkling/GLM-5.2/Kimi K3)数据来自各家一手来源(见链接笔记)。
