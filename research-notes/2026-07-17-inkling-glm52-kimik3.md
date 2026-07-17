# Inkling / GLM-5.2 / Kimi K3 三大开放模型深度调研

- **Date:** 2026-07-17
- **Tags:** LLM, MoE, 稀疏注意力, 长时域Agent, 多模态, 开源模型, Inkling, GLM-5.2, Kimi-K3

## Context

2026 年 7 月中旬,三家实验室在两周内接连放出旗舰级**开放权重**大模型:

- **Thinking Machines Lab —— Inkling**(2026-07-15,Apache 2.0)
- **智谱 Zhipu —— GLM-5.2**(约 2026-07-02,MIT)
- **月之暗面 Moonshot —— Kimi K3**(2026-07-16 公告,权重定于 07-27 释出)

三者共同押注的主线高度一致:**MoE + 稀疏注意力 + 百万上下文 + 长时域 Agent 能力 + 异步 RL 后训练**。本笔记基于各家一手来源(官网公告、Hugging Face 模型卡、arXiv 技术报告)整理,并对第一方声明与第三方评测做了区分标注。

> [!WARNING]
> **可信度提示**:除少数第三方(Artificial Analysis)评测外,绝大多数基准分数为**厂商第一方自报**,峰值算力设置、评测 harness 各不相同,横向不可直接比较。发布均在两周内,社区独立复现尚不充分。Kimi K3 权重尚未释出(计划 07-27),其数据均来自官方博客声明。

## 论文总览表

| | **Inkling** | **GLM-5.2** | **Kimi K3** |
|---|---|---|---|
| 厂商 | Thinking Machines Lab | 智谱 Zhipu AI | 月之暗面 Moonshot |
| 发布/公告 | 2026-07-15 | ~2026-07-02 | 2026-07-16(权重 07-27) |
| 总参数 | **975B**(Small 276B) | **753B** | **2.8T** |
| 激活参数 | **41B**(Small 12B) | 40B(据 GLM-5 报告*) | 16/896 专家激活 |
| 架构基座 | DeepSeek-V3 式 MoE | MoE + DSA(`glm_moe_dsa`) | KDA + Stable LatentMoE |
| 稀疏注意力 | 滑窗/全局 5:1 交错 | DSA + IndexShare | Kimi Delta Attention |
| 上下文 | 架构 1M(Tinker 64K/256K) | 声称 1M | 1M |
| 多模态 | 文/图/音/视频输入 | 文本为主 | 文/图/文档(视觉基准) |
| 许可证 | **Apache 2.0** | **MIT**(无地域限制) | "首个开源 3T 级"(许可未明示) |
| 权重 | HF(BF16 + NVFP4) | HF `zai-org/GLM-5.2` | 计划 2026-07-27 |
| 可信度 | 高(一手齐全) | 高(HF 卡 + 双 arXiv) | 中(权重未出,仅公告) |

\* GLM-5.2 模型卡本身未列专家数/激活参数;40B 激活来自同系 GLM-5 技术报告(744B/40B),GLM-5.2 为 753B,推测同架构微调放大。

---

## 一、Inkling(Thinking Machines Lab)

### 是什么

TML 首个开放权重模型,2026-07-15 发布。从零训练的 MoE Transformer,主打**多模态输入 + 长上下文 + 可微调**,并同步放出小号 **Inkling-Small(276B/12B)**,共享同一套后训练栈。

### 架构(largely follows DeepSeek-V3)

| 组件 | 细节 |
|---|---|
| 专家 | 256 路由专家 + 2 共享专家,每 token 激活 6 路由专家 |
| 路由 | sigmoid router + auxiliary-loss-free 负载均衡 bias;路由/共享专家分数联合归一化 |
| 注意力 | 滑窗层与全局层 **5:1 交错**,8 个 KV heads |
| 位置编码 | 相对位置编码(实测比 RoPE 外推更好) |
| 卷积 | K/V 投影后 + 注意力/MLP 残差分支上加短卷积 |

### 训练配方

- **数据**:45T token 预训练,含文本/图像/音频/视频;多模态部分从零训练。
- **优化器**:混合 —— 大矩阵权重用 **Muon**,其余用 **Adam**;weight decay 与学习率平方耦合(灵感来自其 modular manifolds 研究)。
- **后训练**:先用开放权重模型(**含 Kimi K2.5**)的合成数据做 SFT 冷启动 → RL 扩到 **30M+ rollouts**,reward 呈 log-linear 提升(held-out reasoning 聚合从 SFT 初始 0.264 → 释出版 0.356)。CoT 随训练**自发变短**。
- **硬件**:NVIDIA **GB300 NVL72** 系统。

### 多模态编码(encoder-free)

- **音频**:输入 dMel 频谱图。
- **图像**:40×40 像素块,经四层 hMLP 编码。
- 二者均用轻量 embedding 层,与文本 token 联合处理;推理时可调 Python 工具做图像缩放/裁剪。

### 上下文

架构支持**最高 1M**;托管平台 Tinker 提供 **64K / 256K** 两档。

### 基准(effort=0.99, temp=1.0;编码评测 256K 轨迹上限)

| Benchmark | Score | 备注 |
|---|---|---|
| AIME 2026 | **97.1%** | 第一方 |
| GPQA Diamond | 87.2% | 第三方 Artificial Analysis |
| HLE(纯文本 / 带工具) | 29.7% / 46.0% | |
| SWEBench Verified | 77.6% | **bash-only harness**,口径特殊 |
| SWEBench Pro Public | 54.3% | |
| Terminal Bench 2.1 | 63.8% | 内部 harness |
| MMMU Pro | 73.5% | 多模态 |
| VoiceBench / MMAU | 91.4% / 77.2% | 音频 |
| StrongREJECT | 98.6% | 安全 |

> TML **自己承认**:"Inkling is not the strongest overall model available today." 亮点在开放权重 + 多模态 + 可微调,而非刷榜第一。effort 从 0.2 扫到 0.99;在 Terminal Bench 上以约 1/3 token 量匹配 Nemotron 3 Ultra。

### 可用性

- **权重**:HF `thinkingmachines/inkling`,原始 checkpoint + **NVFP4**(Blackwell 高效推理)。
- **API**:TogetherAI、Fireworks、Modal、Databricks、Baseten。
- **推理/RL**:SGLang+Miles、vLLM(Inferact)、TokenSpeed、llama.cpp(Unsloth)、transformers。
- **微调**:Tinker 平台(64K/256K,限时 5 折);另放出 `tml-renderer`(工具调用/推理/多模态采样)。
- Inkling-Small 全权重待测试完成后释出。

---

## 二、GLM-5.2(智谱 Zhipu AI)

### 是什么

约 2026-07-02 发布,**753B 参数** MoE,官方定位标题即 **"GLM-5.2: Built for Long-Horizon Tasks"**(为长时域任务而生),**MIT 完全开源、无地域限制**。

> **技术报告**:GLM-5.2 **没有单独的技术报告**,其架构/训练依据仍是 GLM-5 系列报告 **arXiv:2602.15763**(HF 模型卡亦将其列为关联论文)。因此下文架构均以该报告口径为准,GLM-5.2 视为同系迭代放大(744B/40B → 753B)。

### 架构(GLM-5 技术报告口径,GLM-5.2 同系放大)

| 组件 | GLM-5 报告值 |
|---|---|
| 参数 | 744B 总 / 40B 激活(GLM-4.5 为 355B/32B,翻倍) |
| 层数 | 80(减少层数以降低 expert parallelism 通信开销) |
| 专家 | 256 |
| 注意力 | **DSA(DeepSeek Sparse Attention)** + MLA-256("Muon Split" 配方,head dim 192→256、head 数减 1/3);indexer k=2048 |
| MTP | 3 层 MTP 训练时共享参数,accept length 2.76(vs DeepSeek-V3.2 的 2.55) |

- **DSA**:用"动态、细粒度选择"替代传统 dense O(L²) 注意力——不像滑窗那样固定,而是**"看内容"决定哪些 token 重要**,长序列注意力计算减约 1.5–2×,号称 "lossless by construction"。
- **预训练**:基座 28.5T token;数据含 Web(DCLM + World Knowledge 分类器)、Code(去重后独特 token +28%)、Math & Science。中训练把上下文 4K→200K 分阶段拉伸(32K@1T → 128K@500B → 200K@50B);SWE 数据约 1000 万 issue–PR 对、约 160B 独特 token。

### 技术亮点 1 —— IndexShare / IndexCache(arXiv:2603.12201)

跨层复用稀疏注意力索引,消除 DSA indexer 的 O(L²) 冗余:

- DSA 的 lightning indexer 每层都选 top-k、自身仍是 O(L²);而**相邻层的 top-k 选择高度相似**。
- 把层分成 **Full layers**(少数,跑自己的 indexer)和 **Shared layers**(多数,直接复用最近 Full layer 的 top-k 索引)。GLM-5.2 卡描述其 IndexShare **"每 4 个稀疏注意力层复用同一 indexer,在 1M 上下文下 per-token FLOPs 降低 2.9×"**。
- 两种配置:**训练无关**(greedy search 按校准集 LM loss 选保留层)与**训练感知**(多层蒸馏 loss,让简单交错模式也能匹配 full-indexer 精度)。
- 效果(30B DSA 模型):削减 **75% indexer 计算**、prefill 最高 **1.82×**、decode **1.48×** 加速,质量损失可忽略。⚠️ 零损失需配合 greedy 搜出的层模式,均匀交错在 Long Avg 上损失约 7.2 分。已在生产级 GLM-5 上做过初步验证(Figure 1)。

### 技术亮点 2 —— 异步 RL 后训练(GLM-5 报告)

- 基于 **slime** 框架,**解耦生成与训练**以最大化 GPU 利用率:训练/推理引擎分设不同 GPU,推理持续产出轨迹、攒够阈值成批送训,每 K 次梯度更新同步权重(更新后重置 optimizer)。
- 关键组件:**Multi-Task Rollout Orchestrator**(支持 1k+ 并发 rollout)、**TITO Gateway**、PD(Prefill-Decode)分离、心跳容错、DP-aware 路由。
- 异步 Agent RL 算法:
  - **TITO(Token-in-Token-out)**:直接消费精确 tokenization 与解码 token 流,避免再 tokenize 失配。
  - **Direct Double-sided Importance Sampling**:token 级 [1−εℓ, 1+εh] 裁剪,复用 rollout log-prob 作行为代理,免去追踪 πθold(类似 IcePop 但更简)。
  - 丢弃 off-policy/噪声样本(stale w′−w0 > τ、环境崩溃失败)。
  - 骨干 Reasoning RL:GRPO + IcePop,去 KL 项,β=2、εlow=0.2、εhigh=0.28、group size 32。

### 基准(GLM-5.2 模型卡自报)

| 类别 | 分数 |
|---|---|
| AIME 2026 | 99.2 |
| GPQA-Diamond | 91.2 |
| HLE(/带工具) | 40.5 / 54.7 |
| HMMT Feb.2026 | 92.5 |
| SWE-bench Pro | 62.1 |
| Terminal Bench 2.1(最佳 harness) | 82.7 |
| FrontierSWE(Dominance) | 74.4 |
| MCP-Atlas(Public) | 76.8 |

> Intelligence Index v4.0:GLM-5 得 50(GLM-4.7 为 42);Vending-Bench 2 终局余额 $4,432。
> ⚠️ **矛盾点**:官方宣称 1M 上下文,但该声明在前序对抗性验证中被质疑(实际可用长度待实测);上表分数为第一方,社区独立复现未充分。

### 可用性

- **权重**:HF `zai-org/GLM-5.2`(BF16·F32),108 个量化版(llama.cpp/LM Studio/Jan/Ollama 兼容)、15 个微调版;月下载约 51 万。
- **推理框架**:SGLang(v0.5.13.post1+)、vLLM(v0.23.0+)、Transformers、KTransformers、Unsloth;Ascend NPU 走 vLLM-Ascend/xLLM/SGLang。
- **入口**:chat.z.ai(对话 + API)。
- 关联论文:arXiv:2602.15763(GLM-5 技术报告)、2603.12201(IndexCache)。

---

## 三、Kimi K3(月之暗面 Moonshot)

> [!NOTE]
> 官方博客 `kimi.com/blog/kimi-k3` 日期 2026-07-16,**权重计划 2026-07-27 释出**。以下均为**官方公告声明**,尚无第三方复现或开放权重可核实——数据置信度中,谨慎对待。(此前深度调研的对抗性验证曾拒绝这些数字,原因是彼时无法独立核实一手文档;现博客一手源已确认存在。)

### 是什么

月之暗面公告的 **2.8T 参数** MoE,自称**"全球首个开源 3T 级模型"**。较 Kimi K2 号称整体 scaling 效率提升 **2.5×**。

### 架构与训练

| 组件 | 细节 |
|---|---|
| 参数 | 2.8T 总,激活 **16/896 专家**(激活参数未明示) |
| 注意力 | **Kimi Delta Attention(KDA)** + Attention Residuals(AttnRes) |
| MoE | **Stable LatentMoE** 框架 + Quantile Balancing |
| 其它 | Per-Head Muon、Sigmoid Tanh Unit(SiTU)、Gated MLA |
| 上下文 | 1M token |
| 量化训练 | 自 SFT 阶段起 QAT:**MXFP4 权重 + MXFP8 激活** |
| 推理 | 默认 max thinking effort(计划出 low/high 档);保留 thinking history 模式;建议 64+ 加速器 supernode |

### 基准(Kimi K3 max,官方自报)

| Benchmark | Score |
|---|---|
| GPQA-Diamond | 93.5 |
| BrowseComp | 91.2 |
| Terminal Bench 2.1 | 88.3 |
| FrontierSWE | 81.2 |
| DeepSWE | 67.5 |
| HLE-Full | 43.5 |
| MathVision | 94.3 |
| MMMU-Pro | 81.6 |
| OmniDocBench | 91.1 |
| GDPval-AA v2(Elo) | 1668.0 |

> 官方坦承 K3 整体仍**落后 Claude Fable 5 与 GPT 5.6 Sol**。

### 官方六模型横向对比表(effort=max, temp=1.0, top-p=1.0)

这是本次调研拿到的最有价值的一手数据:月之暗面官方把 K3 与两家闭源旗舰 + Opus 4.8 + GPT 5.5 + **GLM-5.2** 放在同一口径下评测(K3 与 GLM-5.2 直接可比,GLM-5.2 部分项 `—` 为官方未列)。`*` / `—` 按原表保留。

| Benchmark | **K3** | Fable 5 | GPT 5.6 Sol | Opus 4.8 | GPT 5.5 | **GLM-5.2** |
|---|---|---|---|---|---|---|
| **Coding** | | | | | | |
| DeepSWE | 67.5 | 70.0 | **73.0** | 59.0 | 67.0 | 46.2 |
| Program Bench | 77.8 | 76.8 | 77.6 | 71.9 | 70.8 | 63.7 |
| Terminal Bench 2.1 | 88.3 | 84.6 | **88.8** | 84.6 | 83.4 | 82.7 |
| FrontierSWE | 81.2 | **86.6** | 71.3 | 66.7 | 64.9 | 67.3 |
| SWE Marathon | **42.0** | 35.0 | 39.0 | 40.0 | 14.0 | 13.0 |
| PostTrain Bench | 36.6 | **41.4** | 34.6 | 34.1 | 28.4 | 34.3 |
| MLS Bench | 48.3 | **49.9** | 46.2 | 42.8 | 35.5 | 40.4 |
| Kimi Code Bench 2.0(内部) | 72.9 | **76.9** | 64.8 | 71.7 | 69.0 | 64.2 |
| **Agentic** | | | | | | |
| GDPval-AA v2(Elo) | 1668.0 | **1760.0** | 1748.0 | 1600.0 | 1494.0 | 1514.0 |
| BrowseComp | **91.2** | 88.0 | 90.4 | 84.3 | 84.4 | — |
| DeepSearchQA(f1) | **95.0** | 94.2 | — | 93.1 | — | — |
| Toolathlon-Verified | 73.2 | **77.9** | 74.9 | 76.2 | 73.5 | 59.9 |
| MCP Atlas | 84.2 | **84.7** | 83.6 | 83.6 | 82.8 | 82.6 |
| Automation Bench | **30.8** | 29.1 | 29.7 | 27.2 | 22.7 | 12.9 |
| Job Bench | 52.9 | **57.4** | 46.5 | 48.4 | 38.3 | 43.4 |
| AA-Briefcase(Elo) | 1548.0 | **1583.0** | 1495.0 | 1354.0 | 1158.0 | 1260.0 |
| APEX-Agents | 37.6 | **43.3** | 39.9 | 39.4 | 38.5 | 35.6 |
| Office QA Pro | 63.3 | 69.9* | 63.2* | 63.9* | 60.9* | 41.4 |
| SpreadsheetBench 2 | **34.8** | 34.7* | 32.4* | 31.6* | 29.1* | 28.1 |
| DECK-Bench(内部) | 73.5 | 73.0 | **74.7** | 66.9 | 68.2 | 68.6 |
| **Reasoning & Knowledge** | | | | | | |
| GPQA-Diamond | **93.5** | 92.6 | **94.1** | 91.0 | 93.5 | 91.2 |
| HLE-Full | 43.5 | **53.3** | 44.5 | 49.8* | 41.4* | — |
| HLE-Full w/ tools | 56.0 | **63.0** | 58.0 | 57.9* | 52.2* | — |
| **Vision** | | | | | | |
| MMMU-Pro | 81.6 | 81.2 | **83.0** | 78.9 | 81.2 | — |
| MathVision | 94.3 | 94.8 | **95.8** | 86.7 | 92.2 | — |
| OmniDocBench | **91.1** | 89.8 | 85.8 | 87.9 | 89.4 | — |
| WorldVQA ForceAnswer | 51.0 | **56.7** | 41.8 | 39.1 | 38.5 | — |
| ZeroBench_main(pass@5) | **23.0** | **23.0** | 17.0 | 17.0 | 22.0 | — |

**读表要点**(⚠️ 全为月之暗面单方评测,选表/口径可能对己有利):
- K3 在 **SWE Marathon、BrowseComp、DeepSearchQA、Automation Bench、OmniDocBench、SpreadsheetBench 2** 等项**领先全场**(含两家闭源旗舰);
- 但在 **FrontierSWE、GDPval-AA、HLE、多数 Agentic 工具项**上仍**落后 Claude Fable 5**,官方自己也承认整体不及 Fable 5 / GPT 5.6 Sol;
- **K3 vs GLM-5.2**:在双方都有数的项上 **K3 基本全面领先**(如 DeepSWE 67.5 vs 46.2、Toolathlon 73.2 vs 59.9、Automation 30.8 vs 12.9),仅个别接近——这是本笔记里唯一一组"同口径"的国产双雄直接对比。

> ⚠️ **技术报告尚未发布**:页面明确写完整细节"will be released alongside the Kimi K3 technical report",目前**无 arXiv/PDF/HF 链接**(仅 GitHub org 主页 `github.com/MoonshotAI`)。层数、tokenizer、预训练 token 数、具体 RL 方法官方均未披露,待技术报告 + 07-27 权重释出。

### 可用性

- 入口:Kimi.com、Kimi Work 桌面版(v3.1.0+,Win/Apple silicon)、Kimi Code(`/model`)、Kimi API(`kimi-k3`);移动端 iOS/Android/HarmonyOS;Kimi Enterprise。
- **API 定价**:cache-hit 输入 \$0.30/MTok、cache-miss 输入 \$3.00/MTok、输出 \$15.00/MTok。
- 许可证:自称"首个开源模型",但页面**未明示具体协议名**(MIT/Apache 待权重释出确认)。

---

## 趋势分析:三家押注同一条主线

1. **稀疏注意力成为长上下文标配,且开始"跨层复用索引"**。GLM 的 DSA→IndexShare、Kimi 的 KDA、Inkling 的滑窗/全局交错——三家都在解决"1M 上下文下注意力/indexer 的 O(L²) 成本",GLM 的 IndexCache 更把优化推进到**跨层复用 top-k 索引**这一新维度(per-token FLOPs 降 2.9×)。

2. **"长时域 Agent"是共同叙事**。GLM-5.2 标题直接写 Long-Horizon Tasks;Kimi K3 强调 thinking history + agentic 编码(FrontierSWE 81.2);Inkling 冲 SWEBench/Terminal Bench。评测重心从"知识问答"转向"端到端软件工程 + 多步工具调用"。

3. **异步 RL 后训练 + 巨量 rollout 成为分水岭**。GLM 的 slime 异步框架(解耦生成/训练、TITO、1k+ 并发 rollout)与 Inkling 的 30M+ rollout 都指向:后训练的**工程基础设施**已是核心竞争力,不再只是算法。

4. **量化原生化**。Inkling 直接发 NVFP4 checkpoint、Kimi K3 从 SFT 起就 MXFP4/MXFP8 QAT——低精度不再是事后压缩,而是训练时一等公民,瞄准 Blackwell/大规模推理成本。

5. **开放权重竞赛白热化**。同两周内 975B(Apache)、753B(MIT)、2.8T(自称首个开源 3T)接连开放,且都挂 HF + 多推理框架 + 多云 API,可微调可自托管——开源前沿正快速逼近闭源旗舰。

## Open Questions

- **Kimi K3 权重 07-27 能否如期释出?** 具体开源协议(能否商用)、激活参数量、KDA/Stable LatentMoE 的完整技术报告是否会跟进?
- **GLM-5.2 的 1M 上下文是架构能力还是实际可用?** IndexShare 是否已在生产版真正启用?实测有效上下文多长?
- **Inkling 45T 多模态各模态占比?** 音频(语音识别/生成)、视频理解的实际能力天花板在哪?encoder-free 路线相比传统 vision encoder 的代价?
- **三者真实推理成本/延迟横向对比?** 975B/753B/2.8T MoE 在中等规格云服务器上的部署可行性与 tokens/s?
- **第一方基准的可复现性**:待 Artificial Analysis 等第三方补齐 GLM-5.2/Kimi K3 的独立评测后,实际排名会如何变化?

## References

- Thinking Machines Lab, "Introducing Inkling" — https://thinkingmachines.ai/news/introducing-inkling/
- Inkling 产品页 — https://thinkingmachines.ai/inkling/
- Hugging Face — `thinkingmachines/inkling`
- GLM-5.2 模型卡 — https://huggingface.co/zai-org/GLM-5.2
- GLM Team, "GLM-5: from Vibe Coding to Agentic Engineering," arXiv:2602.15763
- Bai et al., "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse," arXiv:2603.12201
- Moonshot AI, "Kimi K3" — https://www.kimi.com/blog/kimi-k3
- 深度调研工作流(99 agents,对抗性验证)于 2026-07-17 执行

> 引用须可验证:以上均为一手来源;第一方自报分数已明确标注,未经第三方独立审计的数字请勿直接用于横向排名。
