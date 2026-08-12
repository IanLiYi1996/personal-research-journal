# 深度调研：Meta Muse Glimmer 30B（开放权重、本地 agent 向）

- **Date:** 2026-08-12
- **Tags:** #模型调研 #开放权重 #本地部署 #agent #多模态 #投机解码 #meta
- **模型:** [meta-models/Muse-Glimmer-30B](https://hf.co/meta-models/Muse-Glimmer-30B)（Apache-2.0，1,109 likes）
- **一手材料:** 官方 model card / `config.json`（主模型 + drafter）/ GGUF 仓库实际文件大小 / [HF 官方博文](https://huggingface.co/blog/muse-glimmer) / ⭐ **官方评测方法学报告 PDF（7 页，第二轮）** / DFlash 论文正文（第二轮）
- **我做过的独立核算:** ⭐ 从 `config.json` 复算了 KV cache 预算与 24GB 卡的总占用（见 §4），并核实了「global 层无位置编码」这一架构主张
- ⭐⭐ **§11 是第二轮**（读官方评测方法学报告 + DFlash 论文之后）——**其中 §11.1 更正了我第一轮的一处过头批评，§11.2 是本次调研最重要的发现**
- ⭐⭐ **§12 是第三轮**（追 Muse Spark 安全报告）——**用对照实验证明 model card 引用的那份报告链接已失效，并整理出「Cyber 定级无法端到端核验」的完整链条**

---

## 0. TL;DR

**Muse Glimmer 是 Meta Superintelligence Lab 于 2026-08 释出的 30B 开放权重多模态模型，从 Muse Spark 蒸馏而来，明确面向「在消费级硬件上跑自主 agent」这个场景。**

⭐⭐⭐ **我认为它最值得记的不是跑分，而是「本地 agent」这个目标如何反向决定了整套架构选择** —— 而这一点可以被算术验证，不必依赖厂商说法：

| 我的核算结论 | 数值 |
|---|---|
| 权重（17GB 量化 LM + DFlash drafter + 视觉塔） | **18.43 GiB** |
| **满 131,072 上下文的 KV cache** | ⭐ **仅 1.70 GiB** |
| 合计 | **20.13 GiB** → 24 GiB 卡尚余 **3.87 GiB** |
| ⚠️ 若换成朴素 MHA + 全层全局注意力，同上下文 KV 需 | ⭐⭐ **104 GiB（61 倍）** |

> ⭐⭐⭐ **也就是说「装得进一张 3090」不是量化技巧的结果，而是架构决策的结果。** GQA 16:1 加上「三层滑窗 + 一层全局」的 3:1 结构，把 KV cache 从 104 GiB 压到 1.7 GiB —— **这是整个模型设计的支点，而不是一个附带优化。**

**其余几条要点:**
- ⭐ **架构上一个干净的分工:滑窗层用 RoPE 管局部顺序，全局层用 NoPE（无位置编码）管全局信息** —— 我在 `config.json` 里核实过（`layer_rope_theta` 是 `[5e5, 5e5, 5e5, 0, …]` 的显式数组）
- ⭐⭐ **自带一个 2.56B 的 DFlash 投机解码 drafter，是「块扩散」而非自回归**:一次前向预测 16 个 token。**实测 RTX 5090 上 74.9 → 233.4 tok/s（3.1×）**
- ⚠️ **基准上有一个清晰的分裂:它在「调工具 / 多轮 agentic」上大幅领先同级，但在「操作电脑 / 终端」上明显落后 Qwen3.6-27B**（OSWorld-Verified 65.9 vs 75.6、TerminalBench 51.7 vs 60.7）
- ⚠️ **model card 里直接报了 prompt injection 的攻击成功率:AgentDojo ASR 28.4%** —— 对企业落地这是比任何能力分都重要的一个数
- ⚠️⚠️ **Preparedness 里 Cyber 与 Loss-of-Control 被标为「推断」，而三轮追查下来这条定级在公开信息里无法端到端核验:** 第二轮发现 Meta 其实定义并运行了两个网络安全基准（CyberGym / CyberBench）**但分数在三处公开文档里全都没有**（§11.1）；第三轮发现推断所依赖的 **Muse Spark 安全报告链接已失效** —— 我用「同一浏览器会话下 `ai.meta.com/blog/` 能渲染、报告路径渲染出 0 字符」的对照实验排除了反爬（§12.1），且 **Muse Spark 权重未公开、不在 HF 上**。⭐ **这不构成「风险被低估」的证据（我没有反面证据），只说明这条公开论证链是断的**
- ⭐⭐⭐ **第二轮最重要的发现（§11.2）:领先项与落后项的证据来源不同** —— 领先最多的 MCP-Atlas(+13.0) 与 DeepSearchQA 是 **Meta 自己跑 + LLM 判分**，落后最多的 Terminal-Bench(−9.0) 是**第三方 Artificial Analysis** 测的；而 Meta 自己写了一条免责声明说「我们的 agent 工具与 system prompt 可能未针对第三方模型调优」——**这正是 A²E/ProMax 测出的 scaffold 效应，由厂商自己承认**

---

## 1. 身份与定位

| 项 | 内容 |
|---|---|
| **作者** | **Meta Superintelligence Lab** |
| 发布 | 2026 年 8 月（HF 仓库创建 08-09，官方博文 08-10） |
| **许可** | ⭐ **Apache 2.0**（不是 Llama 那种自定义社区许可） |
| 来源 | ⭐ **从 Muse Spark 蒸馏（distilled from Muse Spark）** |
| 参数 | HF API 报 **29,776.6M**；card 写 ~29.6B；HF 博文拆为 **2B 视觉 + 28B 文本** |
| 模态 | 输入 文本 + 图像，输出 文本（⚠️ **不支持音频**） |
| 上下文 | **131,072+** |
| **知识截止** | **2026-01-04** |
| 多语言 | 100+ 语言的训练数据（⚠️ 但 card 自陈「未在预训练数据包含的所有语言上评估过」） |
| 推理力度 | ⭐ 通过 system prompt 里的 `Reasoning strength: <value>` 控制，支持 **low / medium / high / xhigh** |
| 采样建议 | temperature 1.0 / top_p 0.95 / top_k 64 |
| 声明兼容的 scaffold | ⭐ **OpenClaw、Hermes Agent** 及其他 agentic orchestration 模式 |

**官方的一句话定位（我的翻译）:**
> 「一个 300 亿参数的因果语言模型，带一个专用的感知编码器，从 Muse Spark 蒸馏而来，**为消费级硬件上的自主 agentic 任务专门构建**。该模型把多步推理、可靠的工具使用、多模态理解与**失败恢复**整合进单一模型，**在本地运行而不需要云基础设施或网络访问**。」

⭐ **card 明确把「能力清单」写成了 agent 的组成要件**：端到端任务完成、可靠工具调用、多步推理、**失败恢复**（工具调用失败时「诊断错误并重试而不是停下」）、多模态输入、**scaffold 兼容性**、可控力度、多语言。
> ⭐⭐ **注意「scaffold 兼容性」被列为一项模型能力。** 这与我这两周追的 harness 主线正好对上 —— [[2026-08-11-hf-daily-papers-aug10-11]] 里 DCAS 测出「在某个 scaffold 上微调会让模型换 scaffold 就退化」，而这里 Meta 把「跨 scaffold 可用」当成一个要交付的属性写进了 card。**这是我见过厂商第一次把它列为模型规格。**

---

## 2. 架构（逐项，含我核实过的部分）

### 2.1 语言模型

| 项 | 值 | 我的核实 |
|---|---|---|
| 类型 | Dense Causal Transformer（**非 MoE**） | `MuseGlimmerForConditionalGeneration` |
| Hidden dim | **6,656** | ✅ config |
| 层数 | **52** | ✅ config |
| **注意力模式** | ⭐ **[Local, Local, Local, Global] 重复 13 次** | ✅ `layer_types` 数组逐项核对 |
| 滑窗大小 | **2,048** | ✅ config |
| **位置编码** | ⭐⭐ **RoPE(θ=500,000) 只在滑窗层；全局层 NoPE** | ✅ **`layer_rope_theta` = `[5e5,5e5,5e5,0,…]`，全局层显式为 0** |
| Q/KV heads | **32 / 2 → GQA 16:1** | ✅ config |
| Head dim | 128 | ✅ config |
| FFN | SwiGLU，intermediate **19,968** | ✅ config（`hidden_activation: silu`） |
| 词表 | **202,048**（200k BPE + 2,048 special） | ✅ config |
| Gated attention | Yes | card |

**⭐⭐ HF 博文给出了 NoPE 设计的明确理由（我的翻译）:**
> 「(SWA, SWA, SWA, Full) 重复 13 次共 52 层。**这让模型既能用 RoPE 保留相对顺序与距离信息，又能用 NoPE 在全局层面保存信息。**」

> ⭐⭐⭐ **这条值得单独记，因为它是「位置编码分工」这个趋势的一个干净实例。** 我在 [[2026-07-27-blog-raschka-open-weight-roundup]] 与 K3 架构笔记里记过「Kimi K3 全面弃用 RoPE」；**Muse Glimmer 的做法不是弃用而是分层指派 —— 局部层要顺序信息所以给 RoPE，全局层要「不因距离衰减」所以不给。** 这比「用不用 RoPE」这个二元问题信息量大得多。

**⭐ 两个我从 config 里挖出来、card 没写的细节:**

| config 字段 | 值 | 我的解读 |
|---|---|---|
| `qk_scale_factor` | **3.87** | ⭐ HF 博文解释了它：**先对每个 Q/K head 做 RMS 归一化以稳定 attention logits，之后再把 queries 乘一个 scale factor 来设定归一化后的目标 logit 尺度** —— 博文说它「**在 softmax 层面表现得像一个逆温度**」 |
| `output_multiplier` | 0.19611613513818404 | ⚠️ **恰好等于 1/√26，而 26 = 52/2（层数的一半）**。这个数值关系我确认过（1/√26 = 0.196116135138184），**但 card 与博文都没解释它，所以我只记录这个巧合，不推断其用意** |
| `final_logit_softcapping` | 20.0 | logit 软截断 |
| `post_norm_eps` | 1e-08（与 `rms_norm_eps` 1e-05 并存） | 暗示除 pre-norm 外还有一个 post-norm |
| `tie_word_embeddings` | false | 输入输出 embedding 不共享（30B 规模下合理） |

### 2.2 感知编码器（Perception Encoder）

| 项 | 值 |
|---|---|
| 规模 | **~1.8B**（HF 博文写「2B」）ViT-G/14 |
| 层数 / 宽度 / patch | **50 层 / 1,536 / 14** |
| ⭐ 注意力模式 | **同样是「三层 window + 一层 full」的 3:1 结构**（我在 `vision_config.layer_types` 里核实过） |
| 位置编码 | 可插值的**绝对位置 embedding**（学习得到的位置表）+ 注意力层内对 Q/K 用 **2D RoPE** |
| 单图最大视觉 token | **4,096** |
| ⭐ token 压缩 | **pixel shuffle 把相邻 2×2 的空间 token 拼接 → 图像 token 数减少 4×，且不丢弃通道** |

**它源自 Meta 自己 2025 年的工作:** [Perception Encoder: The best visual embeddings are not at the output of the network](https://arxiv.org/abs/2504.13181)（arXiv:2504.13181，2025-04-17，Bolya / Huang / Sun / Cho 等）。
> ⭐ **注意 HF 博文特别指出「与其他 VLM 里相对较小的视觉编码器不同，这是一个相当大的 2B ViT 类模型」** —— 在一个 30B 的总预算里拿出 ~6% 给视觉塔，是个明确的取舍。

**⭐⭐ 一处 card 与 config 的表面矛盾，我查清了:**
- **card 的 Limitations 写:「模型未针对视频显式优化；视频输入按单帧处理。」**
- **但 `config.json` 里有 `video_token_id: 200091`。**
- ⭐ **HF 博文解决了这个矛盾:确实存在一条定义好的视频通路** —— 视频逐帧过同一个编码器，**processor 目标 2 帧/秒、把片段上限截到均匀采样的 96 帧**，并**生成带时间戳的视频占位符，形如 `Time: 0.0s <|video|>` 与文本交错**。
> ⭐ **所以准确的说法是「有视频路径但未针对视频做优化」，而不是「不支持视频」。** 这个区分对做视频 agent 的人是实质性的。

### 2.3 DFlash 投机解码 drafter（单独的 2.56B 模型）

**这是本次发布里我认为工程上最有意思的部分。** 它作为独立仓库释出：[meta-models/Muse-Glimmer-30B-assistant](https://hf.co/meta-models/Muse-Glimmer-30B-assistant)（**2,556.0M 参数**，架构 `muse_glimmer_assistant`）。

| 项 | 值 | 核实 |
|---|---|---|
| 机制 | ⭐⭐ **块扩散（block diffusion）**：**一次前向预测整块 16 个 token**，主模型并行验证、接受正确的、纠正错误的 | card + `block_size: 16` |
| Draft 层数 | **5** | ✅ config |
| ⭐ 读取目标模型的哪几层 | **`target_layer_ids: [1, 13, 25, 37, 49]`**（52 层里均匀取 5 层） | ✅ config |
| 注意力 | 全部滑窗 2,048 | ✅ config |
| Q/KV heads | 32 / **8**（GQA 4:1，⭐ **比主模型的 16:1 宽松**） | ✅ config |
| Hidden / interm | 6,656 / 19,968（**与主模型同宽**，故可直接消费目标模型的 hidden state） | ✅ config |
| ⭐ `mask_token_id` | **201818** | ✅ config —— **这是「掩码块扩散」的直接证据:drafter 一次填充 16 个被掩码的位置** |

**它基于:** [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)（arXiv:2602.06036，2026-02-05，Jian Chen / Yesheng Liang / Zhijian Liu）。

**⭐⭐ 实测加速（card 给的数字）:**

| 硬件 | 无投机 (tok/s) | 带 DFlash (tok/s) | 加速 |
|---|---:|---:|---:|
| **Nvidia RTX 5090** | 74.9 | **233.4** | ⭐ **3.1×** |
| Apple M4 Max | 23.7 | 37.8 | 1.5× |
| Apple M5 Max | 26.6 | 50.2 | 1.8× |

*（batch size 1、greedy decoding；M4/M5 用 **ExecuTorch**，RTX 用 **llama.cpp**）*

> ⭐⭐⭐ **这里有一条跨来源的时间线，是我这个仓库能提供的独有视角:**
> | 时间 | 事件 | 我的记录 |
> |---|---|---|
> | **2026-02-05** | DFlash 论文发表 | — |
> | **2026-07-06** | NVIDIA Developer Blog：「在 Blackwell 上用 DFlash 投机解码把推理性能提升**最高 15×**」 | [[tech-blogs/2026-W28]] |
> | 2026-08-06 | NVIDIA 再次宣传（称最高 15×） | [[tech-blogs/2026-W32d]] |
> | ⭐ **2026-08-09/10** | **Meta 在 Muse Glimmer 里出货 DFlash drafter，实测消费卡 3.1×** | 本份 |
>
> ⭐⭐ **「最高 15×」与「实测 3.1×」的差距值得记，而且不必解读成谁在夸大** —— NVIDIA 那个数字是 Blackwell 数据中心卡上的上界（很可能配合大 batch 与特定负载），Meta 这个是**消费卡、batch=1、greedy** 的平均值。**但它是一个很好的「厂商 up-to 数字 vs 部署实测」对照案例**，我会把它记进成本主线。
> ⭐ **另一点:一项技术从论文到被另一家厂商写进旗舰开放权重发布，用了 6 个月。**

---

## 3. 量化与「装得下」的具体形态

**card 的说法:** 用量化把权重压到约 4-bit，**语言模型降到 20 GB 以下**，为 KV cache、感知编码器与投机解码 drafter 同时运行留出余量，**目标 24 GB 或 32 GB 的显存包线**。

| | Full Precision | K-Quant-Dynamic | K-Quant-17GB |
|---|:---:|:---:|:---:|
| 退化* | — | **0.2%** | **1.0%** |
| 目标硬件 | 64 GB VRAM | 32 GB VRAM | **24 GB VRAM** |

*\* card 注明：退化是**在 15 个常见基准上准确率指标的平均**。*

**⭐ 我核对了官方 GGUF 仓库的实际文件大小（不依赖 card 的描述）:**

| 文件 | 字节 | GiB |
|---|---:|---:|
| `muse-glimmer-30B-kquant-17gb.gguf` | 16,756,681,056 | **15.61** |
| `muse-glimmer-30B-kquant-dynamic.gguf` | 19,653,957,984 | **18.30** |
| `dflash-kquant.gguf`（drafter） | 1,631,205,312 | **1.52** |
| `mmproj-kquant.gguf`（视觉投影/塔） | 1,400,328,928 | **1.30** |

> ⭐ **「17GB」是十进制 GB 的命名（16.76 GB = 15.61 GiB）**，「under 20 GB」对应 dynamic 变体的 19.65 GB。**两个名字都准确，但读的时候要注意 GB / GiB 的区别** —— 差 7%，在 24 GiB 卡上不是可忽略的量。

---

## 4. ⭐⭐⭐ 我自己算的一遍：为什么它真能装进 24 GB

**这一节是本份最有价值的部分 —— 用 `config.json` 直接算，不依赖厂商叙述。**

KV cache 每 token 每层 = `num_kv_heads(2) × head_dim(128) × 2(K,V) × 2 bytes(bf16)` = **1,024 字节**。
52 层中 **13 层全局**（需要全上下文）、**39 层滑窗**（只需 2,048 窗口）。

| 上下文长度 | 全局层 KV | 滑窗层 KV | **合计** |
|---|---:|---:|---:|
| 8,192 | 0.10 GiB | 0.08 GiB | **0.18 GiB** |
| 32,768 | 0.41 GiB | 0.08 GiB | **0.48 GiB** |
| **131,072（满）** | 1.62 GiB | 0.08 GiB | ⭐ **1.70 GiB** |

**24 GB 卡上的完整预算（17GB 变体 + drafter + 视觉 + 满 128K 上下文）:**

```
权重  15.61 (LM) + 1.52 (drafter) + 1.30 (vision) = 18.43 GiB
KV    满 131,072 上下文                          =  1.70 GiB
────────────────────────────────────────────────────────────
合计                                              = 20.13 GiB
24 GiB 卡剩余                                     =  3.87 GiB
```

**⭐⭐ 对照:如果换成朴素 MHA（32 个 KV head）且所有层都是全局注意力，同样 131,072 上下文的 KV cache 需要 104 GiB —— 是实际设计的 61 倍。**

> ⭐⭐⭐ **这个对照是我认为理解这个模型的钥匙:**
> **「30B 模型跑在一张 3090 上」这件事，媒体和社区归因于 4-bit 量化，但量化只解决了权重那 15.6 GiB。真正让它可行的是 KV cache 从 104 GiB 变成 1.7 GiB —— 而那来自两个架构决策（GQA 16:1、3:1 滑窗/全局），不是来自量化。**
> ⭐ **换一个说法:如果 Meta 只做量化而不改注意力结构，这个模型在消费卡上连 8K 上下文都跑不动**（52 层全局 + 32 KV heads 在 8,192 上下文下 KV 就要 6.5 GiB，而 131K 根本不可能）。
> ⭐⭐ **这也解释了为什么 card 把「本地部署」写在架构之前 —— 目标确实反向决定了设计。**

⚠️ **我这个核算的边界:** 它算的是**理论 KV 上限**，未计入 llama.cpp / ExecuTorch 的实现开销、激活内存、图像 token 占用的上下文（单图最多 4,096 视觉 token，会挤占 KV 预算）、以及分页/碎片。**所以 3.87 GiB 的余量是乐观值，实际可用余量更小。** 但结论的方向（KV 不是瓶颈、权重是）不受影响。

---

## 5. 基准：赢在哪、输在哪

**对照组是 Gemma4-31B 与 Qwen3.6-27B，两者都用 Thinking Mode，Muse Glimmer 用 High Reasoning。** card 与 HF 博文的表格数字一致（我逐项核对过）。

**⭐ Muse Glimmer 领先的:**

| 类别 | 基准 | Muse Glimmer | Gemma4-31B | Qwen3.6-27B |
|---|---|---:|---:|---:|
| 通用 agentic | ⭐ **MCP Atlas** | **75.5** | 54.2 | 62.5 |
| | DeepSearch QA | **74.6** | 61.7 | 71.1 |
| | τ³-Banking | **23.5** | 15.1 | 16.7 |
| | WildClawBench | **47.6** | 37.6 | 43.2 |
| | Gaia2 | **43.3** | 36.4 | 40.0 |
| agentic 编码 | **SWE-Bench Pro** | **51.2** | 36.9 | 50.2 |
| | SciCode | **43.6** | 43.4 | 39.8 |
| 多模态 | Charxiv Reasoning | **78.8** | 77.7 | 78.4 |
| 通用推理 | IFBench / AIME 2026 | **77.0** / **94.7** | 76.0 / 89.2 | 70.8 / 94.1 |
| 长上下文 | ⭐ **AA-LCR** / **Beam128K** | **80.0** / **65.1** | 68.3 / 58.2 | 73.3 / 63.0 |

**⚠️ Muse Glimmer 落后的:**

| 基准 | Muse Glimmer | 最好者 | 差距 |
|---|---:|---:|---|
| ⚠️ **OSWorld-Verified** | 65.9 | **75.6**（Qwen） | **−9.7** |
| ⚠️ **TerminalBench 2.1** | 51.7 | **60.7**（Qwen） | **−9.0** |
| GDPval-AA v2 | 953 | **1141**（Qwen） | −188 |
| SkillsBench (with skills) | 44.3 | **46.6**（Qwen） | −2.3 |
| SWE-Bench Verified | 76.0 | **77.2**（Qwen） | −1.2 |
| OmniDocBench v1.5 | 75.8 | **77.8**（Qwen） | −2.0 |
| GPQA Diamond / HLE Text | 83.5 / 22.0 | **85.7 / 23.6**（Gemma） | −2.2 / −1.6 |

> ⭐⭐⭐ **这张表里有一个我认为很清晰的模式，而 card 与博文都没点出来:**
>
> **它在「通过 API/工具接口完成多轮任务」上大幅领先**（MCP Atlas +13.0 over Qwen、DeepSearch QA、τ³-Banking、WildClawBench、Gaia2 全胜），**但在「直接操作电脑与终端」上明显落后**（OSWorld-Verified −9.7、TerminalBench −9.0）。
>
> **一个合理的解释是这两类任务的瓶颈不同:** 工具调用考的是 schema 遵循与多轮状态维持；OSWorld / TerminalBench 考的是 GUI 定位与长序列 shell 操作。**而 ScreenSpot Pro（GUI 定位）它是 75.4 vs 76.1 —— 基本持平**，所以差距不在「看得懂屏幕」，更可能在**长序列操作的执行与恢复**上。
> ⚠️ **这只是我的假设，card 未做归因分析。** 但它对选型有直接含义：⭐ **如果你的场景是「agent 调你的 API」，这个模型很强；如果是「agent 替你操作电脑」，Qwen3.6-27B 在同级里更好。**

**⭐⭐ 另外注意长上下文两项（AA-LCR 80.0 vs 73.3、Beam128K 65.1 vs 63.0）都领先** —— 这与 §2 的 NoPE 全局层设计是自洽的：**全局层不带位置衰减，理论上更利于远距离检索。** ⚠️ 但这是相关而非因果，card 未做消融。

---

## 6. ⚠️ 安全与 Preparedness：两处必须指出的地方

### 6.1 ⭐ 它在 model card 里直接报了 prompt injection 的攻击成功率

| 基准 | Muse Glimmer | Gemma4-31B | Qwen3.6-27B |
|---|---|---|---|
| **Siren AgentDojo** | ⚠️ **ASR(↓) 28.4** / Utility **94.2** | ASR **25.6** / Utility 90.8 | ASR 40.3 / Utility 92.7 |
| **CI Memories** | Violation(↓) 26.4 / Coverage 64.8 | Violation **12.1** / Coverage 53.0 | Violation 53.4 / Coverage **66.9** |

> ⭐⭐ **把攻击成功率印在 model card 上是值得肯定的做法** —— 大多数发布不报这个。
> ⚠️⚠️ **但要把这个数读对:28.4% 的攻击成功率意味着每四次注入攻击有超过一次成功。** 三个模型里最好的是 Gemma 的 25.6%。**也就是说这一代同级开放权重模型在 AgentDojo 上都还在 1/4 以上。**
> ⭐⭐⭐ **这直接接上我这两周的主线:** [[2026-08-11-hf-daily-papers-aug11b]] 记的加密推理块可做**不可见 prompt injection**、[[tech-blogs/2026-W33]] 记的 Databricks「Innocent until combined」把 lethal trifecta 防御做成产品、以及 [[2026-W33-reddit-hot]] 记的用户侧「让 Claude 用 WebFetch 要小心」。
> **对企业落地的含义很直接:模型层的注入抵抗力不足以作为防线，必须在权限与执行层做约束。** 而 card 自己也这么说 —— 它「强烈建议不要把 Muse Glimmer 当作端点本身部署，而应作为带额外护栏的整体 AI 系统的一部分」，并特别建议「对不可逆动作加人在环确认」。

⭐ **训练期缓解里有一条我没在别处见过的表述:**
> 「**Appropriate information flows（适当的信息流）:数据敏感性识别、最小化、以及 local-first 执行的原则，通过专门的合成训练数据被直接嵌入模型权重。**」
> ⭐ **把「本地优先执行」当作一种要写进权重的价值取向** —— 这与它的产品定位一致，且是我第一次看到厂商这样描述。它对应 card 的隐私风险轴「**Privacy（Appropriate Information Flows）—— 受 CI（contextual integrity）理论启发**」。

### 6.2 ⚠️⚠️ Cyber 与 Loss-of-Control 两项风险等级是「推断」的，不是实测

**card 原文（我的翻译）:**
> Muse Glimmer **不落入 Meta 的 Advanced AI Scaling Framework（AAISF）对「Frontier AI」的定义**，因为它总体上不如 Muse Spark 能干。不过出于审慎，Preparedness 团队评估了它的风险画像并给出以下定级：
> - Chem/Bio：中等或更低；
> - **Cyber：中等或更低（推断）**；
> - **Loss of Control：中等或更低（推断）**。
>
> 「Cyber 与 Loss of Control 的风险等级**被推断为**中等或更低，因为 Muse Glimmer 整体弱于 Muse Spark 1.0，而后者在这些领域获得了相同的风险定级。」

> ⚠️⚠️⚠️ **【第二轮更正】本节的批评需要收窄 —— 我读了官方的评测方法学报告后发现 Meta 确实做了网络安全评测（CyberGym + CyberBench），只是没公布结果。详见文末 §11.1。以下保留原文以留痕，但请连带读那一节。**
>
> ⚠️⚠️ **我认为这个推断链条在本周的语境下值得明确指出，原因有两个:**
> 1. ⭐ **「A 弱于 B，B 是中等，所以 A 是中等」这个论证依赖「弱于」是在相关维度上成立的。** 而 card 自己的数据显示 Muse Glimmer 在**某些**维度上并不弱于同级（WMDP-Bio 86.5 高于 Gemma4 的 85.9、Lab Bench ProtocolQA 80.2 高于 75.8）。**它与 Muse Spark 的比较未给出网络安全维度上的具体分数。**
> 2. ⚠️⚠️ **而 Muse Spark 正是 [[tech-blogs/2026-W32d]] 里那几起「AI 意外攻击真实目标」案例之一的模型系列**（Meta Muse Spark 与英国 AISI 的记录）。**把它作为「同为中等风险」的锚点，需要那个锚点本身是可靠的。**
>
> ⭐ **公平地说:Chem/Bio 是实测的**（给了 MBCT / HPCT / VCT / WMDP / Lab Bench 六项，并把 Kimi K3 放进来作参照），**结论「大致与同级持平、严格低于更大的开放权重模型」有数据支撑。** 我的保留只针对被标注为「inferred」的那两项。
> ⚠️ **同时注意:此前 Reddit 已记「Meta 即将释出 Muse Spark 1.2 的权重」**（[[2026-W33-reddit-hot]]）—— 若成立，那么「用 Muse Spark 的定级来推断 Glimmer」这条链会变成一个被更多人检视的对象。

---

## 7. 生态：一天之内长出完整的量化矩阵

**官方 day-0 支持（HF 博文）:** `transformers`（`AutoModelForMultimodalLM` + `AutoProcessor`，**主模型与 drafter 都支持**）、**llama.cpp**、**vLLM**、Inference Endpoints；同一段代码在 **NVIDIA (CUDA) 与 AMD (ROCm)** 上不改即可跑。

**官方释出的四类 artifact（全部 Apache 2.0）:** BF16 全精度权重 / 两个 4-bit 量化变体 / **DFlash drafter head** / **感知编码器（冻结的 ViT-G/14）**。

**⭐ 官方还额外提供:** `-GGUF`（207 likes）、`-assistant`（drafter，45）、⭐ **`-ExecuTorch-PTE`（22）—— PyTorch Edge 的端侧导出**。

**社区在 1–2 天内产出的（我从 HF 检索到的）:**

| 方向 | 仓库 |
|---|---|
| GGUF | **unsloth（311 likes）**、lmstudio-community、prithivMLmods、AtomicChat（imatrix）、weareapexcreators |
| Apple MLX | **mlx-community 的 bf16 / 4 / 5 / 6 / 8bit / mxfp4 / mxfp8 / nvfp4** 全套；RadixArk 的 q4/q4km/q4k-dynamic；OsaurusAI JANG_4M |
| NVIDIA FP4 | RadixArk NVFP4（mxfp8/modelopt/sglang）、Inferact **NVFP4-W4A4** |
| bitsandbytes | unsloth-bnb-4bit |
| ⭐ 浏览器 | **webml-community/muse-glimmer-webgpu-kernels**（WebGPU kernels） |
| 托管推理 | **together**（live） |

> ⭐⭐ **这个矩阵本身就是「本地 agent」定位被生态接受的证据:** 出现的不是通用的服务端量化，而是**MLX（Apple 本地）+ GGUF（llama.cpp 本地）+ ExecuTorch（端侧）+ WebGPU（浏览器）**四条本地路线全覆盖。
> ⭐ 这印证了我在 [[2026-W33-reddit-hot]] 记的那条社区流程：「**发布 → 量化 → 确认单卡可跑**在同一天内完成」，与 Qwen3.8-27B、MiniMax H3 是同一个已经稳定下来的模式。

---

## 8. 我的看法

**1. ⭐⭐⭐ 这是我见过第一个「部署约束明确地反向决定了架构」的旗舰开放权重模型。**
不是「训了一个模型，然后量化一下让它能本地跑」，而是**先定 24/32 GB 的包线，再选 GQA 16:1、3:1 滑窗/全局、pixel shuffle 4× token 压缩、自带 drafter** —— 每一项都在压 KV 或压激活。**§4 的算术（1.70 GiB vs 104 GiB）是这个判断的证据。**

**2. ⭐⭐ 它把「scaffold 兼容性」写成模型规格，这件事的意义超过它本身。**
[[2026-08-11-hf-daily-papers-aug10-11]] 里 DCAS 刚测出「在单一 scaffold 上微调会导致跨 scaffold 退化」，[[2026-08-11-hf-daily-papers-aug11b]] 里 A²E 刚测出「九个 harness 在单轮任务上分数完全相同、只有多轮任务能区分」。**现在厂商把「跨 scaffold 可用」列为要交付的属性。**
> ⭐ **但注意 card 只声明兼容（OpenClaw / Hermes Agent），没有给跨 scaffold 的分数。** 按 A²E 的结论，**要证明这一点需要在多轮任务上跨多个 harness 报分** —— 这正是我会向 Meta 提的问题。

**3. ⭐⭐ 「工具调用强、操作电脑弱」这个分裂对选型比总分有用得多。**
MCP Atlas +13.0、DeepSearch QA +3.5，但 OSWorld-Verified −9.7、TerminalBench −9.0。**而 ScreenSpot Pro 基本持平说明差距不在视觉定位。** ⭐ 这与我在 HF digest 里记的 SWE-Bench ProMax 的发现同调：**失效模式往往不是"看不见"而是"长序列执行撑不住"。**

**4. ⚠️ 我对两件事保持保留。**
- **Preparedness 里 Cyber/LoC 用「弱于 Muse Spark」来推断**，而 Muse Spark 恰是本仓库记过的意外攻击案例系列（§6.2）。
- **AgentDojo ASR 28.4%** 说明模型层的注入抵抗远不够用；⭐ **这不是 Muse Glimmer 的问题（三家都在 25%+），而是这一代同级模型的共同状态** —— 但它意味着**任何把它接进有真实权限的工作流的方案，都必须在执行层设约束**。

**5. ⭐ 一个可以拿去用的成本论据。**
本周我记了四次「成本买不到分数」。**Muse Glimmer 提供的是另一个方向的数据点:一个 Apache-2.0、可完全本地运行、在通用 agentic 基准上领先同级的 30B 模型，硬件门槛是一张 24 GB 消费卡。** 对数据不能出境、或推理量大到 API 成本不可接受的客户，这是一个真实可评估的选项 —— ⚠️ **但要连带说明 ASR 28.4% 与 OSWorld 落后这两件事。**

---

## 9. Open Questions

- ⭐⭐ **跨 scaffold 的实际分数是多少？** card 声明兼容 OpenClaw / Hermes Agent，但没给跨 scaffold 对比。按 A²E 的方法（同骨干、同配置、同 task ID、多轮任务）测一遍，才能验证「scaffold 兼容性」这个声明。
- ⭐⭐ **OSWorld-Verified 与 TerminalBench 落后 9 分的原因是什么？** ScreenSpot Pro 持平排除了视觉定位。**是长序列执行、失败恢复，还是 shell 工具使用？card 未做归因。**
- ⭐ **`output_multiplier` = 1/√26 的用意？** 数值关系我核实了（26 = 层数/2），但 card 与博文都未解释。
- ⭐⭐ **Cyber / LoC 的「推断」有实测数据支撑吗？** 尤其在本周 OpenAI 公开承认 Astra 触及 Critical 级网络能力的语境下，**用「弱于 Muse Spark」作锚点需要那个锚点本身的网络安全评测是公开的。**
- ⭐ **1.0% 的量化退化是怎么分布的？** card 说是「15 个常见基准上准确率的平均」。**平均 1.0% 可能掩盖某个 agentic 基准上的大幅下降** —— 而 card 说「验证了压缩在 agentic 任务上引入最小到无退化」，但没给分项。**这正是我这周反复记的「一个数掩盖一个结构」。**
- **`research.meta.ai/static/muse-glimmer-methodology` 那份方法学报告里有什么？** 我本次未取（本份已基于 card + config + GGUF + HF 博文四个一手来源）。**若要做第二轮，这是首选。**
- **DFlash 的接受率是多少？** 3.1× 加速是结果，但**块大小 16 下的 token 接受率**才是理解它适用范围的关键（card 说 drafter「特别适合结构化内容生成如编码」，暗示接受率与内容类型强相关）。

## 10. 一手来源与核实状态

| 来源 | 我做了什么 |
|---|---|
| [meta-models/Muse-Glimmer-30B](https://hf.co/meta-models/Muse-Glimmer-30B) model card | ✅ 全文读取（约 17KB） |
| `config.json`（主模型） | ✅ 逐字段核对；**核实了 `layer_rope_theta` 的 0 值与 `layer_types` 的 3:1 模式** |
| `config.json`（drafter） | ✅ 核实 `block_size: 16`、`target_layer_ids`、`mask_token_id` |
| GGUF 仓库文件清单 | ✅ 取到**实际字节数**，据此复算 GiB/GB |
| [HF 官方博文](https://huggingface.co/blog/muse-glimmer) | ✅ 读取架构与生态两节；**基准表与 card 逐项一致** |
| [arXiv:2504.13181](https://arxiv.org/abs/2504.13181) Perception Encoder | ✅ 经 arXiv API 核实标题、作者、日期 |
| [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) DFlash | ✅ 同上 |
| KV cache 与显存预算 | ⭐ **我自己从 config 复算**（见 §4，含边界说明） |

⚠️ **未核实/未做的:**
1. **所有基准分数均为 Meta 自报**（HF 博文注明「Scores are reported as published」），**我未独立复现任何一项**。
2. **两篇被引论文我只核实了标题/作者/日期，未读正文** —— 因此 Perception Encoder 与 DFlash 的机制描述转述自 model card 与 HF 博文。
3. **`research.meta.ai` 的方法学报告与 Muse Spark 安全报告未读。**
4. **性能数字（tok/s）为 card 自报**，我无相应硬件复测。
5. **「distilled from Muse Spark」的具体蒸馏方法未披露**，card 只给了这个结论。
6. **§5 关于「工具调用强 / 操作电脑弱」的解释是我的假设**，card 未做归因分析。

---

# 11. 第二轮：读了官方评测方法学报告之后

**第一轮我把 `research.meta.ai/static/muse-glimmer-methodology` 列为「未读、第二轮首选」。现在读了 ——它是一份 7 页 / 22,353 字符的 PDF（标题 `Muse Glimmer Eval Methodology`，Google Docs 导出），逐基准说明了评测设定。**

⭐ **结论:它显著改变了我对这份发布的两处判断 —— 一处是我批评过头了，一处是我此前没看到的更严重的问题。**

## 11.1 ⚠️⚠️ 更正：Cyber 确实被评测了，只是结果没公布

**第一轮我在 §6.2 批评「Cyber 与 Loss-of-Control 是推断而非实测」，并暗示他们没做网络安全评测。这个暗示是错的。**

**方法学报告里有一个完整的 "Cyber Domain" 小节，列了两个基准:**

| 基准 | 内容（原文要点） |
|---|---|
| **CyberGym** | 「衡量在**真实世界开源软件**中自动发现漏洞的公开基准。⭐ **它作为 Cyber 2 track 的代理（proxies the Cyber 2 track）**，聚焦模型在复杂真实代码库中识别并利用漏洞的能力。」指标是**平均漏洞发现率** |
| **CyberBench** | 「一个**对抗鲁棒性**基准，评估模型抵抗越狱与绕过安全措施的能力。测量**单轮与多轮**对抗场景下的 ASR」，同时跟踪拒答率与安全响应质量，**结果在多次独立对抗测试运行上取平均** |

> ⭐ **所以准确的说法是:Meta 定义并（据方法学报告）运行了两个网络安全评测，其中 CyberGym 明确对应其框架里的 "Cyber 2 track"。**
> ⚠️⚠️ **但真正的问题变了，而且我认为比原来那个更值得指出:两个基准的方法都写了，结果一个数都没有。** model card 的表格里没有 CyberGym / CyberBench 的分数，方法学报告也只描述方法不给数值 —— **而 card 同时把 Cyber 风险等级标为「inferred」。**
>
> ⭐⭐ **修正后的批评:不是「没测」，而是「测了、定义了指标、却既不公布数值也不用它来支撑定级，反而回退到『弱于 Muse Spark 所以同级』的推断」。** 这比我原本以为的情况更奇怪 —— **因为数据显然存在。**
> ⭐ 我第一轮提的那个 Open Question（「Cyber/LoC 的推断有实测数据支撑吗？」）**现在有了部分答案:有对应的评测，但支撑链条没有被展示出来。**

## 11.2 ⭐⭐⭐ 最重要的新发现：领先项与落后项的证据来源不同

**方法学报告逐基准写了「分数从哪来」。我把它整理成一张表 —— 这是第一轮完全看不到的结构:**

| 基准 | Muse 相对表现 | 分数来源 | 判分方式 | 重复次数 |
|---|:---:|---|---|:---:|
| **MCP-Atlas** | ⭐ **+13.0** | **Meta 内部跑全部 3 模型** | ⚠️ LLM judge = **Gemini 2.5 Pro**，阈值 0.75 | 4 |
| **DeepSearchQA** | ⭐ +3.5 | **Meta 内部跑全部 3 模型** | ⚠️ LLM judge = **gpt-oss-120b**，F1 | 4 |
| **GAIA2** | ⭐ +3.3 | **Meta 内部**，⭐ **用 OpenClaw harness** | ⚠️ LLM judge = gpt-oss-120b（容器内） | 3 |
| SkillsBench | −2.3 | Meta 内部 | ✅ **确定性 pytest 隐藏测试** | 4 |
| **OSWorld-Verified** | ⚠️ **−9.7** | **Meta 内部** | ✅ 执行式（程序化 checker 查 VM 终态） | 4 |
| SWE-Bench Pro | ⭐ +1.0 | Meta 内部（⭐ **明确不用 Qwen 自报分，因其在 refined 版本上测的**） | ✅ 测试通过率 | 4 |
| SWE-Bench Verified | −1.2 | ⚠️ **Qwen 用的是自报分** | ✅ 测试通过率 | 4 |
| ⚠️ **Terminal-Bench 2.1** | ⚠️ **−9.0** | ⭐⭐ **第三方 Artificial Analysis** | Terminus 2 harness / E2B 沙箱 | AA 方法学 |
| SciCode | +0.2 | ⭐ **第三方 Artificial Analysis** | 单元测试 | AA 方法学 |
| τ³-Banking | ⭐ +6.8 | ⭐ **第三方 Artificial Analysis** | per-task 断言 | AA 方法学 |
| OmniDocBench | −2.0 | Meta 内部，⚠️ **改过评分协议**（见 11.4） | 编辑距离 | 4 |
| MMMU-Pro / GPQA / HLE | −1〜−2 | ⭐ **第三方 Artificial Analysis** | 选择题 | AA 方法学 |
| CIMemories | 混合 | Meta 内部 | ⚠️ LLM judge = **Claude 4.6 Sonnet** | 4 |
| **Siren AgentDojo** | 中等 | Meta 内部 | ✅ **确定性规则判分** | — |
| IFBench | ⭐ +1.0 | Meta 内部（Gemma 用自报分） | ✅ 可验证约束 | 4 |
| AIME 2026 | ⭐ +0.6 | Meta 内部（Gemma/Qwen 用自报分） | ✅ 精确答案 | ⭐ **10** |

> ⭐⭐⭐ **这张表里有一个我认为必须说出来的模式:**
>
> **Muse Glimmer 领先最多的三项（MCP-Atlas +13.0、τ³-Banking +6.8、DeepSearchQA +3.5）里，有两项是 Meta 自己跑的、且用 LLM judge 判分；而它落后最多的两项里，Terminal-Bench 2.1（−9.0）是第三方 Artificial Analysis 测的。**
>
> ⚠️ **而 Meta 自己在报告开头写下了一条正好适用于此的免责声明（我的翻译）:**
> > 「**对第三方模型的 agentic 评测**：这些结果代表我们对第三方模型的尽力评测，使用与我们内部模型相同的评测框架以保证一致性。**我们注意到我们的评测设置（例如 agent 工具与 system prompt）可能并未针对专有第三方模型专门调优。因此这些结果可能不反映这些模型在为其特定优势定制的环境中的最佳表现。**」
>
> ⭐⭐⭐ **这条免责声明就是 A²E 与 ProMax 测出来的那个效应，由厂商自己写出来。** [[2026-08-11-hf-daily-papers-aug11b]] 里 A²E 证明 **harness 差异只在多轮任务上显形**；[[2026-08-11-hf-daily-papers-aug10-11]] 里 ProMax 测出**同一模型换 scaffold 分数接近翻倍（21.8%→41.2%）**。
> **而 MCP-Atlas / DeepSearchQA / GAIA2 恰恰全是多轮 agentic 任务 —— 也就是 harness 效应最强的那一类。** 所以：**这三项的领先幅度里有多少来自模型、有多少来自「用 Meta 的工具与 system prompt」，从公开信息无法拆开。**
>
> ⭐ **要公平地说清另一面:Meta 在这份报告里做了三件明显偏向对手的选择，值得记账。**
> 1. ⭐⭐ **「对于 Muse Glimmer 以外的模型，我们报告自报分数与我们内部复现之间**更有利的那个**。」** —— 取对手的**较好**成绩。
> 2. ⭐ **判分用的是竞争对手的模型**（Gemini 2.5 Pro、gpt-oss-120b、Claude 4.6 Sonnet），而不是自家模型当裁判。
> 3. ⭐ **SWE-Bench Pro 上主动弃用 Qwen 的自报分**，理由是那个分数是在 refined 版本上测的、与自己用的原始任务集不可比 —— **这是在放弃一个对自己有利的对比。**
>
> ⭐⭐ **我的综合判断:这份评测的披露质量明显高于行业均值，偏向性主要不在选择性报告，而在「内部 agentic 评测的 harness 是自家的」这个结构性因素上 —— 而 Meta 自己把它写了出来。** ⭐ **正确的读法是:把内部跑的多轮 agentic 领先当作「在 Meta 的 harness 下」，把第三方测的落后当作更接近独立结论。**

## 11.3 ⭐⭐ 一个我该收回的抱怨：他们的重复次数报得很充分

**我这一周反复批评 agent 评测领域「不报重复次数与区间」**（Evo-Bench 只跑一次、A²E 每格 5 任务、ProMax 未报）。**Meta 这份报告在这一点上做得比那些论文都好:**

- **多数基准明确写了 3–4 次运行取平均**，理由都写作 "to reduce variance"
- ⭐ **AIME 2026 用了 10 次运行**（30 题的基准，方差大，10 次是合理的）
- OSWorld / SkillsBench 是 **4 次 attempt 取平均**
- GAIA2 是 **3 次独立运行**，且注明「匹配公开 leaderboard 惯例」

> ⭐ **所以在「重复次数」这一项上我要给它正面评价。** ⚠️ **但仍然没有区间/标准差** —— 只有均值。**而 SWE-Bench Verified 上 76.0 vs 77.2 这种 1.2 分的差距，在 4 次运行的均值上是否显著，无法判断。** ⭐ 这与我从 Evo-Bench 记的教训一致：**跑了多次却只报均值，等于把最有用的那部分信息扔掉了。**

## 11.4 ⚠️ 两处此前没看到的具体保留

**① OmniDocBench 用了改过的评分协议。**
> 报告原文要点：用「**内部实现的原始评分协议，但做了修改**」—— 把官方的三分量评分公式（文本 / 表格 / 公式）**替换成两分量加权平均，把公式折进文本组用编辑距离评分**，而不是用 Character-and-Delimiter Match 单独评；并且**用较简单的匈牙利算法做二部图元素匹配，而不是官方 v1.6 的 MGAM（多粒度自适应匹配）**。

> ⚠️ **三个模型用的是同一套修改后协议，所以内部可比；但它与其他地方报的 OmniDocBench v1.5 分数不可直接比较。** ⭐ card 与 HF 博文的表格都只写「OmniDocBench v1.5」，**没有标注这个协议修改** —— 只有方法学报告里有。

**② ⭐⭐ AgentDojo 的 28.4% 是在「自适应攻击者」下测的 —— 这改变了这个数的含义。**
> 报告原文要点：97 个良性任务 × 4 个环境（银行、旅行预订、工作区管理、在线商店）+ 35 个恶意任务 = **949 个 prompt injection 场景**。⭐⭐ **「LLM 驱动的攻击者（我们用 Claude Opus 4.6）会迭代改进它的攻击，直到成功或达到最多 6 次迭代上限。」** 判分是**确定性规则判分**。

> ⭐⭐⭐ **第一轮我把 28.4% 读成「每四次注入攻击有超过一次成功」。更准确的说法是:「在一个能改写自己载荷、最多试 6 轮的自适应攻击者面前，28.4% 的场景最终失守。」**
> ⭐ **这两个读法的实践含义不同:** 前者像是一个固定载荷的通过率；后者是**对抗性上界**。⚠️ **对企业落地，后者其实更贴近真实威胁**（真实攻击者会迭代），**但也意味着这个数不能直接和那些用静态载荷测出的 ASR 比较。**
> ⭐ **值得肯定的是判分是确定性规则而非 LLM judge** —— 在安全指标上这很重要，因为 [[2026-08-07-hf-daily-papers-aug05-07b]] 记的 OSReward 已经证明 LLM 裁判在困难样本上会大量 over-accept。

## 11.5 ⭐ OSWorld 的行动空间是被控制的——这让我第一轮的假设更站得住

**第一轮我猜「OSWorld/TerminalBench 落后不是视觉定位问题，而是长序列执行问题」，依据是 ScreenSpot Pro 基本持平。方法学报告补上了一个关键控制变量:**

> 「**对 Qwen3.6-27B 和我们的模型，我们采用 Claude computer-use 行动空间（版本 `computer_20251124`）**，归一化 0–1000 坐标空间加一个独立 stop 动作。对 Gemma4-31B，我们用 Gemini 2.5 Flash 的 computer-use 接口。」
> 其余设定：361 任务split（排除 8 个 Google Drive 任务）、**仅 GUI 无 shell**、1920×1080 截图、**每 episode 上限 200 步**、**截图历史截断保留最近 19 张**、4 次 attempt 取平均。

> ⭐⭐ **也就是说 Muse Glimmer 与 Qwen3.6-27B 在 OSWorld 上用的是同一套行动空间** —— **所以 65.9 vs 75.6 这 9.7 分不是行动空间差异造成的。** 配合 ScreenSpot Pro 的 75.4 vs 76.1（基本持平），**我第一轮的假设（差距在长序列执行与恢复，而非视觉定位）现在有两个控制变量支撑了。**
> ⚠️ **仍未被排除的解释:200 步上限与「只保留最近 19 张截图」的历史截断，可能对不同模型的长程记忆策略影响不同。** 这一点报告没做消融。

## 11.6 ⭐⭐ 读了 DFlash 论文之后：为什么是块大小 16

**第一轮我把「DFlash 的 token 接受率」列为 Open Question。论文回答了，而且解释了 Meta 为何选 `block_size: 16`:**

> 论文摘要（我的翻译）：「通过在**单次前向传播**中生成 draft token，DFlash 实现了高效起草；通过**让 draft 模型条件于从目标模型提取的上下文特征**，它以更高的接受率获得高质量草稿。实验表明 DFlash 实现了**超过 6× 的无损加速**，比 SOTA 投机解码方法 **EAGLE-3 高出最多 2.5×** 的加速。」

⭐ **「条件于从目标模型提取的上下文特征」正是 Meta config 里 `target_layer_ids: [1,13,25,37,49]` 的机制** —— 论文的设计，Meta 的实例化。

**⭐⭐ 关于块大小的关键实验（我的翻译）:**
> 「当训练与推理的块大小匹配时（8→8 与 16→16），**块大小 16 的模型在数学与编码任务上取得显著更高的接受长度**。Math500 上的接受直方图显示，**块大小 8 的模型经常整块被完全接受（35.7%）**，说明块大小 8 常常没被充分利用。相比之下，**块 16 的模型呈现更分散的接受分布与更高的平均接受长度**，说明更有效地利用了较大的块。」

> ⭐⭐⭐ **这解释了两件事:**
> 1. **Meta 为什么选 16** —— 论文实测 16 在数学与编码上优于 8，而「块 8 有 35.7% 的时候整块全接受」是一个**天花板信号**（说明还能推更大）。
> 2. ⭐ **card 那句「drafter 特别适合结构化内容生成如编码」的来源** —— 论文的收益正是在数学与编码上最明显。**这不是营销措辞，是有实验依据的适用范围声明。**

**⭐ 另一个论文里的设计洞察（解释了 5 层 drafter 为何可行）:**
> 论文指出**并行起草让起草成本对块大小 γ 几乎不敏感**（`T_parallel ≪ γ · t_step`），因为现代 GPU 执行这类并行操作远比多次串行前向高效。**「因为起草成本不再随 γ 扩展，这从根本上改变了设计空间。」**

> ⭐⭐ **这才是 DFlash 与 EAGLE-3 这类自回归 drafter 的本质区别:自回归 drafter 每多起草一个 token 就多一次串行前向，所以块不能大；块扩散 drafter 一次前向出整块，所以块可以大而几乎不加成本。** 这也是为什么 Meta 敢用 **5 层**（而 EAGLE-3 常用 1 层）—— 起草成本的瓶颈不在深度上。

**⭐⭐⭐ 而这让第一轮那个「up-to 阶梯」变成了三层，比我原来记的更完整:**

| 来源 | 声称/实测 | 设定 |
|---|---|---|
| **DFlash 论文**（2026-02） | **> 6× 无损加速** | 论文自选的模型与任务范围 |
| **NVIDIA**（2026-07，[[tech-blogs/2026-W28]]） | **最高 15×** | Blackwell 数据中心卡 |
| ⭐ **Meta 实测**（2026-08） | ⭐ **3.1×**（RTX 5090）/ 1.5×（M4 Max）/ 1.8×（M5 Max） | 消费级硬件、batch=1、greedy |

> ⭐⭐ **三个数字都可能是对的，因为设定完全不同 —— 但它们放在一起是一个很好的教学案例:同一项技术的「加速倍数」在论文、芯片厂商与实际出货者手里相差 5 倍。**
> **对客户沟通的含义:任何「投机解码能加速 N 倍」的说法，必须连带问硬件、batch size、解码策略与任务类型。**

## 11.7 第二轮之后仍未解决的

- ⭐⭐⭐ **CyberGym / CyberBench 的分数是多少？** 方法与指标都定义了，**数值在 card 与方法学报告里都没有**。这是我现在最想要的一个数。
- ⭐⭐ **1.0% 量化退化的分项分布？** 仍然只有「15 个基准的平均」。⭐ 结合 11.2 的表可以推断那 15 个基准里应该包含 agentic 项，**但没有分项就无法判断「agentic 任务上最小到无退化」这个声明的强度。**
- ⭐ **各基准的方差/区间？** 跑了 3–10 次却只报均值（见 11.3）。
- ⭐ **MCP-Atlas / DeepSearchQA 的领先在换成对手偏好的 harness 后还剩多少？** 按 Meta 自己的免责声明，这个问题是开放的 —— **而按 A²E 的方法（同骨干、同配置、同 task ID、跨多 harness）是可以测的。**
- **OSWorld 的 200 步上限与 19 张截图历史截断，对不同模型是否不等价？** 未做消融。
- **Muse Spark 安全与 Preparedness 报告**（`ai.meta.com/static-resource/muse-spark-safety-and-preparedness-report/`）—— ⚠️ **我这次尝试取它返回 HTTP 400**；它是 Cyber/LoC 推断所依赖的锚点，值得换路径再试（sitemap / 浏览器兜底）。

## 11.8 第二轮的来源与核实状态

| 来源 | 状态 |
|---|---|
| `research.meta.ai/static/muse-glimmer-methodology` | ✅ **PDF（7 页 / 22,353 字符），已用 pymupdf 抽取并读完全文** |
| DFlash 论文 [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) | ✅ 读了摘要、引言要点、块大小消融与起草成本论证（**未读完整实验表**） |
| Muse Spark 安全报告 | ❌ **HTTP 400，未取到** |
| Perception Encoder 论文 | ⚠️ 本轮仍未读正文（优先级低于上述几项） |

⚠️ **第二轮的引用局限:**
1. **方法学报告的所有引文均为我对英文原文的中译**；关键结构（各基准的分数来源、判分方式、重复次数）我逐条对照原文整理成 11.2 的表。
2. ⚠️ **11.2 表里的「Muse 相对表现」是我从 card 的分数自行计算的差值**，方法学报告本身不给差值。
3. ⚠️ **CyberGym「proxies the Cyber 2 track」这句我按原文转述，但我不清楚 Meta AAISF 里 "Cyber 2" 的确切定义** —— 未在本轮核实。
4. **DFlash 的 6× 与 EAGLE-3 对比来自论文摘要**，我未核实其实验细节；**块大小消融的 35.7% 数字来自论文正文，已回原文核对。**

---

# 12. 第三轮：追 Muse Spark 安全报告 —— 那条引用链是断的

**第二轮末我把「Muse Spark 安全与 Preparedness 报告（HTTP 400 未取到）」列为第三轮首选，因为它是 Cyber/LoC 定级推断所依赖的锚点。这一轮我把它追到底了。**

## 12.1 ⭐⭐⭐ 结论：model card 引用的那份报告，链接是坏的

**我做了六种尝试:**

| 尝试 | 结果 |
|---|---|
| 原 URL（curl，桌面 UA） | **HTTP 400**，1,542 字节 Meta 通用错误页「Sorry, something went wrong」 |
| 去掉/加上尾斜杠、加 `.pdf` 后缀 | 均 **400**，同样 1,542 字节 |
| `research.meta.ai` 上的对应路径 | **404** |
| `ai.meta.com/research/publications/...` | **400** |
| `llama.com` / `ai.meta.com/muse/` / `ai.meta.com/muse-spark/` | 均 **400** |
| ⭐ **Xvfb + 非 headless 本地 Chromium**（对 openai.com 有效的那套） | ⚠️ **页面打开但 `document.body.innerText.length === 0`、title 为空** |

**⭐⭐⭐ 关键是我做了一个对照实验，它把「被反爬拦」和「链接本身坏了」区分开了:**

| | curl | 非 headless 浏览器 |
|---|---|---|
| `ai.meta.com/blog/` | **400** | ✅ **渲染成功（title "AI at Meta Blog"，1,401 字符）** |
| `ai.meta.com/`（站点根） | **400** | —— |
| **报告路径** | **400** | ❌ **0 字符** |

> ⭐⭐⭐ **推论:curl 被整站拦截（连站点根都 400），所以不能用 curl 的 400 下任何结论。但同一个浏览器会话里 `/blog/` 正常渲染而报告路径渲染出空白 —— 这说明报告那个资源本身不可用，而不是我被挡在门外。**
>
> **也就是说:Muse Glimmer model card 在 Preparedness 一节里，用来支撑其风险评估方法学的官方引用，是一个失效链接。**

**⚠️ 另两条相关事实:**
- **Muse Spark 不在 HF 上**（我用 HF 检索 API 查过，`meta-models` 名下与全站均无结果）—— 与 [[2026-W33-reddit-hot]] 记的「Meta **即将**释出 Muse Spark 1.2 权重」一致：**权重尚未公开。**
- **web.archive.org 从本机连续返回 429**，多次退避后仍然限流，**所以我无法确认这份报告是否曾被存档** —— 这一条是我的能力限制，不是证据。

## 12.2 ⭐⭐ 于是「Cyber: Moderate or lower」这条定级的完整链条是这样

把三轮查到的东西接起来：

```
model card: 「Cyber: Moderate or lower risk (inferred)」
   ↓ 依据
「Muse Glimmer 整体弱于 Muse Spark 1.0，后者在这些领域获得相同定级」
   ↓ 那 Muse Spark 的定级依据？
「详见 Muse Spark Safety & Preparedness Report」
   ↓
⛔ 链接失效（§12.1 已用对照实验确认不是反爬）
   ↓ 那 Muse Spark 本身能被独立检视吗？
⛔ 权重未公开（不在 HF 上）
   ↓ 那 Muse Glimmer 自己的网络安全评测结果呢？
⛔ CyberGym 与 CyberBench 的方法与指标在方法学报告里有定义，
   但分数在 (a) 主 model card (b) GGUF model card
   (c) 评测方法学报告 三处都没有出现
```

**⭐ 我这一轮特意去查了第三处** —— GGUF 仓库的 README 有 25,216 字节（比主 card 的约 17KB 更长），我读完了全文：**多出来的部分全是 llama.cpp 的部署说明，Preparedness 一节与主 card 逐字相同，同样没有 cyber 分数。**

> ⭐⭐⭐ **所以第三轮的净结论是:「Cyber: Moderate or lower」这个定级在公开信息里无法被端到端核验 —— 不是因为哪一环刻意隐瞒，而是因为四条可能的核验路径同时不可用（推断链的锚点文档失效、锚点模型未开源、自身的 cyber 分数未公布、方法学报告只给方法不给数）。**
>
> ⚠️ **我要把这个结论的边界说清楚:** 这**不构成**「该模型网络安全风险被低估」的证据 —— 我没有任何反面证据。**它只说明这条特定的公开论证链是断的。** ⭐ 而这在本周的语境下值得记，因为同一周 OpenAI 公开承认无法排除 Astra 的 Critical 级网络能力、并公布了 GPT‑5.6‑Cyber 的 95%/1.5% 对照（[[tech-blogs/2026-W33b]]）—— **两家厂商在网络安全能力披露的颗粒度上出现了明显反差。**
> ⭐ **公平地说 Chem/Bio 那一半是实测且给了六项分数**（还把 Kimi K3 放进来作参照），**所以问题局限在被标注 inferred 的那两项。**

## 12.3 ⭐⭐ 一个意外收获：Meta 自己的内存表交叉验证了我 §4 的算术

GGUF 仓库的 README 给了一张「权重加上一个工作上下文」的粗略内存表，**这是我第一轮没看到的:**

| Build | 纯文本 | + 视觉 | **+ 视觉 + drafter** |
|---|---|---|---|
| `17gb` | ~17 GB | ~19 GB | ⭐ **~20 GB** |
| `dynamic` | ~20 GB | ~22 GB | ~23 GB |

**⭐⭐ 与我 §4 独立算出的数字对照:**

| | 我的核算 | Meta 的表 |
|---|---|---|
| 17gb + 视觉 + drafter 的**权重** | **18.43 GiB = 19.8 GB** | ⭐ **「~20 GB」——吻合** |
| 再加**满 131,072 上下文**的 KV | +1.70 GiB（+1.83 GB）→ 20.13 GiB = **21.6 GB** | （其表只含「一个工作上下文」） |

> ⭐⭐⭐ **两个来源互相印证，而且合起来比任一方更有信息量:**
> **Meta 的「~20 GB」基本等于权重；我的核算显示即使把上下文开到满 131,072，也只再加 1.70 GiB。**
> ⭐ **这恰恰是我 §4 那个论点的最强形式:KV 小到「工作上下文」与「满上下文」在内存表上几乎看不出差别 —— 所以 Meta 的表可以不区分二者。** 换成朴素 MHA，这张表根本写不出来（同上下文需 104 GiB）。

## 12.4 ⭐ 部署上几条硬性前提（第三轮新得，实用）

GGUF 仓库的 README 前半段是 llama.cpp 部署说明，有几条是**会直接卡住人**的：

- ⚠️⚠️ **必须 llama.cpp build `b10353` 或更新。** 支持于 **2026-08-10** 合入（PR **#26841**，commit `62bf73d`），首次随 release `b10353` 发布。⭐ **`b10344` 及更早的版本「根本不注册这个架构」，会直接拒绝加载。**
  - 自检：`./llama-cli --version`（build 号 ≥ 10353）；源码检出可用 `grep -c LLM_ARCH_MUSE_GLIMMER src/llama-arch.cpp`（**期望 ≥ 1，得到 0 说明检出早于支持**）
- ⚠️ **两个文本 build 单独都是纯文本的** —— `mmproj-kquant.gguf` 是**图像输入的必需件**，`dflash-kquant.gguf` 是可选件（+约 1.6 GB）
- ⭐ 官方 `llama-server` 示例里直接用 **`-c 131072 -np 4`** —— 满上下文是他们自己文档的默认示范（与 §12.3 的 KV 结论一致）
- ⭐ **思考内容走单独的 `reasoning_content` 字段**；⚠️ 若 `content` 开头出现 `to=self<|message|>`，说明你的 build 早于 chat parser
- ⭐ 开投机解码加 `-md dflash-kquant.gguf -ngld 99`；⚠️ **启动时的 `[spec] failed to measure draft model memory` 警告是无害的**（官方明说）
- ⭐ ExecuTorch 那个仓库是**预导出的程序，覆盖 NVIDIA CUDA 与 Apple Silicon 两者**（不只是 Apple）

## 12.5 三轮之后仍未拿到的

- ⛔ **CyberGym / CyberBench 的分数** —— 三处公开文档皆无（§12.2）。**这已经不是「我没找到」，而是「公开材料里没有」。**
- ⛔ **Muse Spark 安全与 Preparedness 报告** —— 链接失效，且我无法用 wayback 确认是否曾存档（本机被 429 限流）。
- ⭐ **1.0% 量化退化的分项分布** —— 仍只有「15 个基准的平均」。⭐ 三轮下来我倾向认为**这 15 个基准很可能就是 card 表里那批**，但没有依据，所以不写成结论。
- ⭐ **各基准的方差/区间** —— 跑了 3–10 次却只报均值（§11.3）。
- **Perception Encoder 论文正文** —— 三轮都没读（优先级一直低于上面几项）。若要做第四轮，这是唯一还没碰的一手来源。

## 12.6 第三轮的来源与核实状态

| 动作 | 结果 |
|---|---|
| Muse Spark 报告 6 种 URL 变体（curl） | ❌ 全 400 / 404 |
| ⭐ **Xvfb + 非 headless Chromium 打开报告路径** | ❌ 0 字符 |
| ⭐⭐ **同浏览器会话打开 `ai.meta.com/blog/` 作对照** | ✅ 1,401 字符 → **证明是链接坏而非反爬** |
| HF 全站检索 Muse Spark | ❌ 无结果（权重未公开） |
| web.archive.org（CDX + availability API，多次退避） | ⚠️ **持续 429，未能查询** |
| ⭐ **GGUF 仓库 README 全文（25,216 字节）** | ✅ 读完 —— 补上部署前提与内存表，**Preparedness 一节与主 card 逐字相同、无 cyber 分数** |
| 我 §4 算术 vs Meta 内存表 | ✅ **互相印证（19.8 GB vs「~20 GB」）** |

⚠️ **第三轮的局限:**
1. **「链接失效」的判定基于我的对照实验**（同会话下 `/blog/` 成功、报告路径 0 字符）。⭐ 这是我能做到的最强证据，**但不排除该资源对特定地区/登录状态可见**。
2. ⚠️ **我未能查询 wayback**（429），所以**不能断言这份报告「从未存在过」** —— 只能说**现在从我这里取不到**。
3. **§12.2 的流程图是我对三轮所得的整合**，其中每一环都在正文有出处，但**「链条是断的」这个整体判断是我的结论，不是任何一份官方文档的表述。**
