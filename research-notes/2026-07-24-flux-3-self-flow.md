# FLUX.3 与 Self-Flow:BFL 的多模态统一基础模型

- **Date:** 2026-07-24
- **Tags:** FLUX, Black-Forest-Labs, 多模态, flow-matching, 扩散模型, 视频生成, 具身智能, Self-Flow
- **来源:** [bfl.ai/blog/flux-3](https://bfl.ai/blog/flux-3)(2026-07-23 发布公告)+ Self-Flow 研究页元数据

## Context

Black Forest Labs(BFL,Stable Diffusion / FLUX 原班团队)于 **2026-07-23** 发布 **FLUX.3**,从"文生图模型"跃升为**多模态基础模型**——在统一架构里联合学习**图像 / 视频 / 音频**,底层方法叫 **Self-Flow**。

> [!WARNING]
> **可信度边界:** FLUX.3 页面是**发布公告,不是技术报告**。参数量、训练数据/算力规模、Self-Flow 的公式与训练目标官方**均未披露**("will release more technical details later"),基准全标 "preliminary"。Self-Flow 研究页与论文正文为 JS 渲染,本次仅取到**标题+作者**元数据,机制部分按已公开信息 + flow matching 通用原理谨慎描述,凡推断处均标注。

## 一、FLUX.3 是什么

| 维度 | 内容 |
|---|---|
| 发布 | 2026-07-23,现 Early Access |
| 定位 | 多模态基础模型,"real-world visual intelligence" |
| 底层 | **Self-Flow**(multimodal flow matching),统一架构联合学图/视频/音频 |
| 变体 | FLUX 3 Video / Image / **FLUX-mimic(Action)** / **FLUX 3 Dev(开放权重,计划中)** |

### 四个变体

- **FLUX 3 Video** — 视频+原生音频生成/编辑;单次最长 **20 秒**;文/图/视频→视频、关键帧→视频、多语对话、片段的 **agentic 串联**。(Early Access:API + 私有权重)
- **FLUX 3 Image** — 多风格/比例/分辨率,复杂 prompt + **多语言高精度文字渲染**改进。(未来几周开放)
- **FLUX-mimic / FLUX 3 Action** — 与 mimic robotics 合作的视频-动作模型;两条路线:①动作预测**原生集成进 FLUX 3**;②把预训练视频骨干当**动力学感知基座**,专用动作模型少量数据微调。
- **FLUX 3 Dev** — 开放权重多模态骨干,**尚未发布**(FLUX.2 是 Apache-2.0,可期待)。

### 基准(preliminary,视频胜率偏好)

| 对手 | FLUX 3 胜率 |
|---|---|
| Luma Ray 3.2 | **93%** |
| Runway Gen-4.5 | 77% |
| Grok Imagine Video | ≤69% |
| Kling v3 Pro | 60% |
| Seedance 2.0 / Gemini Omni Flash | 52% |

另有 Self-Flow 内部基准:生成误差(Fréchet 距离)归一到 **FM=100**(越低越好)+ 四类任务组的操作成功率(微调后,越高越好)——**具体数值在图中,未提取为文本**。

## 二、Self-Flow 论文

- **标题:** *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*
- **作者:** Hila Chefer、Patrick Esser(共同一作)、Dominik Lorenz、Dustin Podell、Vikash Raja、Vinh Tong、Antonio Torralba、Robin Rombach
- **机构:** Black Forest Labs × **MIT**
- **一句话定位(官方):** 在**同一底层架构**里高效对齐**多模态生成与理解**。
- ⚠️ 正文(方法/公式/基准数值)未取到;下文"技术背景"是理解这篇论文所需的**基础知识铺垫**,非论文原文。

## 三、技术背景与相关技术(理解 Self-Flow 的前置知识)

### 1. Flow Matching(流匹配)——FLUX 的生成范式基座

- **是什么:** 一种训练连续生成模型的框架。不像 DDPM 那样学"逐步去噪",而是直接学一个**速度场(velocity field)** $v_\theta(x_t, t)$,把噪声分布沿一条概率路径"流"向数据分布。
- **Rectified Flow / 线性路径:** FLUX/SD3 系用的是把噪声 $x_0$ 与数据 $x_1$ 用**直线插值** $x_t = (1-t)x_0 + t x_1$,训练目标是回归速度 $v = x_1 - x_0$(简单、少步采样友好)。本周 HF digest 里 **Tuna-2 的像素空间 flow matching**、**LLaDA2.0-Uni 的扩散解码器**都是同一族。
- **为什么重要:** flow matching 采样步数少、训练目标干净,是当前视觉生成的主流(取代了纯 DDPM),也是 FLUX.3 "multimodal flow matching model" 的字面含义。

### 2. 从单模态扩散 → 统一多模态

FLUX.3 的核心跳跃是"图/视频/音频统一在一个架构 + 一个 flow matching 目标"。这条路线的相关技术:

- **统一 token 化:** 把不同模态编码成可在同一序列里处理的表示(图像 patch、视频时空 token、音频频谱/波形 token)。呼应本周 [[2026-07-21-raschka-llm-architecture-comparison]] 里 **Inkling 的 encoder-free 多模态**(dMel 音频 + 40×40 图像块)与 **LLaDA2.0-Uni 的 SigLIP-VQ 语义离散 token**。
- **生成 × 理解统一:** Self-Flow 的卖点是"生成与理解同架构对齐"。业界同期做法有 **LLaDA2.0-Uni**(扩散 LLM 统一理解+生成)、**Tuna-2**(像素空间统一)、**DiffusionGemma**(Google 离散扩散)——见 [[2026-07-21-raschka-llm-architecture-comparison]] 补充四。**FLUX.3 是从"视觉生成侧"切入统一,而非从 LLM 侧。**
- **"Self-Supervised" 的含义(推断):** 标题里的 self-supervised 很可能指——不依赖大量"生成任务标注",而用**模型自身对多模态数据的重建/预测**作监督信号来对齐生成与理解(类似 MAE / masked prediction 的思路)。⚠️ 此为基于标题的合理推断,待论文正文确认。

### 3. 视频/世界模型 → 具身动作(FLUX-mimic 的脉络)

FLUX.3 把视频生成骨干当机器人**动力学基座**——这是 2026 的明确主线,与本周多篇呼应:

- **世界模型:** [[2026-07-24-hf-daily-papers-jul21-24]] 的 **ABot-World-0**(单卡无限交互世界)、AlayaWorld;
- **具身基座:** 同篇的 **RynnBrain 1.1**(跨本体 VLA,建在 Qwen3.5);
- **共同逻辑:** 视频模型隐式学到了"世界如何随动作演变"的动力学,因此可作为动作预测/机器人策略的预训练基座——FLUX-mimic 走的正是"视频骨干 → 少量数据微调出动作模型"这条路。

### 4. FLUX 家族演进

| 版本 | 时间 | 定位 | 开放权重 |
|---|---|---|---|
| FLUX.1 | 2024 | 文生图 DiT(dev/pro/schnell) | dev 开放 |
| FLUX.2 | ~2026 上半 | 图像(klein-4B/9B) | Apache-2.0(HF 有大量社区 LoRA) |
| **FLUX.3** | 2026-07 | **多模态(图/视频/音频)+ 具身** | Dev 计划中,未发 |

## 我的看法

1. **BFL 从"最强开源文生图"转型"多模态基础模型",是被 Sora/Veo/Kling 逼出来的战略升维**——纯图像市场已卷到头,视频+音频+具身才是增量。93% vs Luma、77% vs Runway 的胜率(即便 preliminary)说明视频侧已有竞争力。
2. **"生成与理解统一"是 2026 全行业共识**,但路径分两派:LLM 派(LLaDA2.0-Uni/DiffusionGemma 从语言侧长出多模态)vs 视觉派(FLUX.3/Tuna-2 从生成侧长出理解)。谁的统一更彻底,取决于 Self-Flow 论文揭示的对齐机制。
3. **视频→具身是最被低估的一条线**:FLUX-mimic、RynnBrain、ABot 三者从不同起点(生成/VLA/世界模型)汇向"用视频动力学驱动机器人",值得持续追踪。

## Open Questions(待技术报告)

- Self-Flow 到底怎么"对齐生成与理解"?"self-supervised" 具体指什么监督信号?参数量/训练规模?
- FM=100 基准的真实数值,以及与 Sora/Veo 3/Kling v3 的可比口径?
- FLUX 3 Dev 何时发、什么协议?能否像 FLUX.1/.2 那样催生开源生态?
- FLUX-mimic 的动作预测是 VLA 式(直接出动作 token)还是 world-model 式(先想象后规划)?

## References

- FLUX.3 发布公告 — https://bfl.ai/blog/flux-3(2026-07-23)
- Self-Flow 研究页 — https://bfl.ai/research/self-flow(论文:*Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*,BFL × MIT)
- FLUX.2 开放权重(佐证家族谱系)— https://hf.co/black-forest-labs/FLUX.2-klein-9B
- 相关本仓库笔记:[[2026-07-21-raschka-llm-architecture-comparison]](统一多模态/扩散 LM)、[[2026-07-24-hf-daily-papers-jul21-24]](世界模型×具身)

> 引用须可验证:FLUX.3 数据来自 BFL 官方页;Self-Flow 论文因正文未取到,仅记录已公开的标题/作者/定位,技术背景部分为通用原理铺垫并已标注推断处;未凭记忆编造参数或基准数值。
