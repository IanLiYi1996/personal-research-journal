---
title: "两周内，三家开源了 975B、753B、2.8T 三个巨兽：Inkling、GLM-5.2、Kimi K3 到底谁强"
description: "2026 年 7 月中旬，Thinking Machines、智谱、月之暗面在两周内接连放出旗舰开放权重模型。一手资料扒了个遍，含 Kimi 官方六模型横评表。"
author: "Li Yi"
cover: "imgs/cover-three-models.png"
---

# 两周内三个开源巨兽：975B 的 Inkling、753B 的 GLM-5.2、2.8T 的 Kimi K3，我把一手资料扒了个遍

> 2026 年 7 月，两周之内。
> Thinking Machines Lab 放出 **975B** 的 Inkling。
> 智谱放出 **753B** 的 GLM-5.2。
> 月之暗面公告 **2.8T** 的 Kimi K3，自称"全球首个开源 3T 级模型"。
> 而且——全是**开放权重**。

---

## 先把话说在前面

这篇不是二手解读的搬运。我把三家的官网公告、Hugging Face 模型卡、arXiv 技术报告、Kimi 官方博客全翻了一遍，能对上一手来源的才写进来。

有一点必须先摆明：**除了极少数第三方评测，绝大多数分数都是各家自己报的**，峰值算力设置、评测口径都不一样，横着比要非常小心。所以下面我会明确标注"谁报的数"，而不是把它们混在一张榜单上骗你。

好，开始。

---

## 一、Inkling：975B 的多模态开放权重，作者自己说"不是最强"

Thinking Machines Lab（就是 Mira Murati 那家）7 月 15 日放出了他们第一个开放权重模型 **Inkling**，一个从零训练的 MoE Transformer。

先看骨架：

> **975B 总参数 / 41B 激活**，256 路由专家 + 2 共享专家，每 token 激活 6 个。
> 还有个小号 **Inkling-Small：276B / 12B**。

它最有意思的是**多模态是"从零一起训"的**——45 万亿 token 里就含文本/图像/音频/视频，而且走的是 **encoder-free** 路线：音频输入 dMel 频谱图，图像切成 40×40 像素块，都用轻量 embedding 层直接和文本 token 一起处理，没有传统的独立视觉编码器。

训练配方也挺"新派"：大矩阵权重用 **Muon** 优化器、其余用 Adam，硬件是 NVIDIA **GB300 NVL72**，RL 扩到了 **3000 万次 rollout**。

跑分（effort=0.99 峰值设置）：

| 基准 | 分数 |
|---|---|
| AIME 2026 | 97.1% |
| GPQA Diamond | 87.2%（第三方 Artificial Analysis）|
| SWEBench Verified | 77.6%（bash-only harness）|

但最让我意外的是官方的态度：

> **"Inkling is not the strongest overall model available today."**（Inkling 不是当今整体最强的模型。）

一家实验室发旗舰模型，自己主动说不是最强——它的卖点其实是**开放权重 + 多模态 + 可微调**：权重挂 Hugging Face（还给了 NVFP4 量化版），TogetherAI、Fireworks、Modal、Databricks、Baseten 五家都能调 API，能在自家 Tinker 平台微调。

---

## 二、GLM-5.2：753B、MIT 协议，为"长时域任务"而生

智谱这次很低调，约 7 月 2 日就上了 **GLM-5.2**，**753B 参数、MIT 完全开源、无地域限制**。官方标题直接写：**"Built for Long-Horizon Tasks"**（为长时域任务而生）。

它没有单独的技术报告——架构依据是同系的 GLM-5 报告（arXiv:2602.15763），可以理解为 GLM-5（744B/40B、80 层、256 专家）的同系放大。两个技术亮点值得记住：

**IndexShare / IndexCache**：稀疏注意力（DSA）的 indexer 每层都要算 top-k，本身是 O(L²)，很贵。但相邻层的 top-k 选择高度相似——于是把层分成"自己算索引"和"复用最近层索引"两类。官方说法是**每 4 个稀疏注意力层复用同一个 indexer，1M 上下文下 per-token FLOPs 降 2.9×**。

**异步 RL 后训练**：基于 slime 框架，把生成和训练解耦。这块下面 Kimi 会有个呼应，先按下不表。

> ⚠️ 一个诚实提醒：GLM-5.2 官方宣称 1M 上下文，但这个说法我在做对抗验证时被质疑过，**实际可用长度以实测为准**。

---

## 三、Kimi K3：2.8T 的"3T 级"，但技术报告还没发

月之暗面 7 月 16 日公告了 **Kimi K3**，**2.8T 参数、激活 16/896 专家**，自称"全球首个开源 3T 级模型"，比 K2 号称 scaling 效率提升 2.5×。架构上堆了一串新东西：Kimi Delta Attention、Stable LatentMoE、Per-Head Muon……

但有两个关键事实得说清楚：

> **1. 权重计划 7 月 27 日才释出。**
> **2. 完整技术报告还没发**——层数、tokenizer、预训练 token 数官方都没披露，页面只写"随技术报告一起发布"。

不过 Kimi 干了件很仗义的事：**官方直接放了一张六模型横评表**，把 K3 和 Claude Fable 5、GPT 5.6 Sol、Opus 4.8、GPT 5.5、**还有 GLM-5.2** 摆在同一口径下比。这可能是目前唯一一组"同口径"的对比数据。

摘几行（effort=max，加粗为该行最高）：

| 基准 | K3 | Fable 5 | GPT 5.6 Sol | GLM-5.2 |
|---|---|---|---|---|
| DeepSWE | 67.5 | 70.0 | **73.0** | 46.2 |
| Terminal Bench 2.1 | 88.3 | 84.6 | **88.8** | 82.7 |
| BrowseComp | **91.2** | 88.0 | 90.4 | — |
| GPQA-Diamond | **93.5** | 92.6 | **94.1** | 91.2 |
| OmniDocBench | **91.1** | 89.8 | 85.8 | 89.4 |

读表要点：

- K3 在 **BrowseComp、OmniDocBench、SWE Marathon、Automation Bench** 等项**领先全场**（含两家闭源旗舰）；
- 但在 FrontierSWE、HLE、多数 Agentic 工具项上**仍落后 Claude Fable 5**，官方自己也承认整体不及 Fable 5 / GPT 5.6 Sol；
- **K3 vs GLM-5.2**：双方都有数的项上，K3 基本全面领先（DeepSWE 67.5 vs 46.2、Toolathlon 73.2 vs 59.9）。

> ⚠️ 但这张表是**月之暗面单方**做的，选哪些 benchmark、用什么 harness 都可能对自己有利，GLM-5.2 还有不少项是 `—`（没列）。真实排名得等 Artificial Analysis 这类第三方补齐。

---

## 三家横向速览

| | Inkling | GLM-5.2 | Kimi K3 |
|---|---|---|---|
| 厂商 | Thinking Machines | 智谱 | 月之暗面 |
| 发布 | 07-15 | ~07-02 | 07-16（权重 07-27）|
| 参数 | 975B/41B | 753B | 2.8T，16/896 激活 |
| 许可 | Apache 2.0 | MIT | 自称"首个开源 3T"|
| 权重 | 已发 | 已发 | 待 07-27 |

---

## 我读完想到的几件事

1. **稀疏注意力 + 百万上下文 + 长时域 Agent + 异步 RL，是三家共押的同一条主线。** 谁都在解决"1M 上下文下注意力太贵"和"多步任务怎么训"这两个问题。

2. **量化正在"原生化"。** Inkling 直接发 NVFP4、Kimi 从 SFT 起就 MXFP4/MXFP8 训练——低精度不再是事后压缩，而是训练时的一等公民。

3. **开源前沿正在逼近闭源旗舰。** 两周内三个巨兽开放权重、挂 HF、多云可调、可微调可自托管。Kimi 自己承认还追不上 Claude Fable 5，但差距肉眼可见地在缩小。

4. **对"官方跑分"要留一个心眼。** 三份榜单来自三家自己，口径各异。Inkling 甚至主动说自己不是最强——这种诚实反而更可信。

等 Kimi K3 权重 7 月 27 日释出、技术报告补齐，我会再写一轮第三方评测的横评。

---

*我是 Li Yi，每周读一些有意思的 AI 论文/发布，写成普通人也看得懂的笔记。点个在看，下期见。*
