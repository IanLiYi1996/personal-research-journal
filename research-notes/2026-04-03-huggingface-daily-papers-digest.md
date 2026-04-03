# Hugging Face Daily Papers Digest: 2026-04-03

- **Date:** 2026-04-03
- **Tags:** #daily-papers #huggingface #agent #document-parsing #speech-synthesis #multimodal #RAG #LLM-serving

## Context

本文对 2026 年 4 月 3 日 Hugging Face Daily Papers 上榜的 15 篇论文进行系统梳理与分析。当日论文覆盖 AI Agent 框架与训练、文档解析与 OCR、语音与多模态生成、RAG 与推理基础设施、自动化科学发现、时间序列预测、世界模型等多个方向。通过逐篇解析核心论点、方法论和主要结论，归纳当前 AI 研究的热点趋势。

## 今日论文总览

| 排名 | 论文 | 票数 | 机构 | 领域 |
|------|------|------|------|------|
| 1 | MinerU2.5 | 155 | 上海 AI Lab | 文档解析 |
| 2 | VibeVoice | 149 | Microsoft Research | 语音合成 |
| 3 | Agent Lightning | 140 | Microsoft Research | Agent RL 训练 |
| 4 | MetaClaw | 136 | UNC-Chapel Hill | Agent 持续学习 |
| 5 | PaddleOCR-VL | 123 | Baidu/PaddlePaddle | OCR/文档理解 |
| 6 | daVinci-MagiHuman | 120 | 上海交大/Sand.ai | 音视频生成 |
| 7 | AgentScope 1.0 | 61 | Alibaba | Agent 框架 |
| 8 | Mem0 | 50 | Mem0.ai | Agent 长期记忆 |
| 9 | vLLM/PagedAttention | 50 | UC Berkeley | LLM 推理服务 |
| 10 | AgentScope 大规模仿真 | 41 | Alibaba/人民大学 | 多 Agent 仿真 |
| 11 | LightRAG | 37 | 北邮/港大 | RAG |
| 12 | TradingAgents | 35 | Tauric Research | 金融 Agent |
| 13 | AI Scientist-v2 | 21 | Sakana AI | 自动科学发现 |
| 14 | TimesFM | 18 | Google Research | 时间序列预测 |
| 15 | LeWorldModel | 18 | - | 世界模型/JEPA |

---

## 第一部分：Agent 框架与训练（5 篇）

本日最大主题。5 篇 Agent 相关论文覆盖了从训练框架、持续学习、应用框架到长期记忆的完整技术栈。

### 1.1 Agent Lightning — 用 RL 训练任意 AI Agent

**arXiv:** 2508.03680 | **票数:** 140 | **机构:** Microsoft Research

**核心论点：** Agent Lightning 是首个实现 AI Agent 执行与 RL 训练完全解耦的框架。它允许使用任何现有框架（LangChain、OpenAI Agents SDK、AutoGen 等）开发的 Agent，以几乎零代码修改的方式接入 RL 训练流程。

**方法论：**
- **统一数据接口：** 将 Agent 执行抽象为状态（State）和调用（Call）的序列，其中状态由语义变量组成，调用包含元信息、输入、输出三元组
- **MDP 建模：** 将每次 LLM 完整输出作为一个动作（而非单个 token），状态空间为 Agent 快照，支持部分可观测 MDP
- **LightningRL 算法：** 分层 RL 方法，先将 episode 级回报通过信用分配分解到各动作，再通过 GRPO/PPO/REINFORCE++ 分解到 token 级
- **系统架构（TA Disaggregation）：** Lightning Server 管理 RL 训练并暴露 OpenAI 风格 API；Lightning Client 通过 OpenTelemetry 无侵入式采集数据
- **AIR 机制：** 自动中间奖励，将系统监控信号转化为中间奖励以缓解稀疏奖励问题

**主要结论：**
- 在 Text-to-SQL（Spider）、RAG（MuSiQue）、Math QA（Calc-X）三个任务上均展示持续稳定的性能提升
- 支持多 Agent 系统中选择性优化特定 Agent
- 基于 transition 的数据组织方式解决了上下文累积导致的长序列问题

**局限性：** 当前信用分配仅采用均等策略；实验仅使用 Llama-3.2-3B 等较小模型；多 LLM 联合优化尚未实验验证。

---

### 1.2 MetaClaw — 在实际部署中自主进化的 Agent

**arXiv:** 2603.17187 | **票数:** 136 | **机构:** UNC-Chapel Hill, CMU, UC Santa Cruz, UC Berkeley

**核心论点：** MetaClaw 是持续元学习框架，使已部署的 LLM Agent 在实际使用中自主进化。核心洞察是两种不同时间尺度的适应机制天然互补：

1. **技能驱动的快速适应**（秒级）：从失败轨迹中提取可复用行为指令，无需停机立即生效
2. **机会主义策略优化**（小时级）：在用户空闲时段通过 OMLS 调度器检测睡眠、键盘不活动、日历会议，触发基于 RL 的云端 LoRA 微调

**方法论：**
- **元模型定义：** M = (θ, S)，θ 为 LLM 策略参数，S 为技能库。推理时通过嵌入相似度检索相关技能注入系统提示
- **技能进化：** LLM evolver 分析失败轨迹合成新行为指令，纯粹在 prompt 层面操作，零停机
- **策略优化：** GRPO + 过程奖励模型（PRM）+ 云端 LoRA 微调，仅在空闲窗口触发
- **技能代版本控制：** 严格区分 support data 和 query data，防止过时奖励污染梯度

**主要结论：**
- MetaClaw-Bench（934 题，44 模拟工作日）：技能适应将准确率从 21.4% 提升至 28.3%（+32.2%）；完整流水线提升至 40.6%，接近 GPT-5.2 基线（41.1%）
- 端到端任务完成率提升 8.25 倍（2.0% → 16.5%）
- 较弱模型从 MetaClaw 中获益更大，能力差距可通过框架大幅弥补

**局限性：** 空闲窗口检测依赖用户配置；MetaClaw-Bench 为模拟基准；策略优化自然滞后于技能适应。

---

### 1.3 AgentScope 1.0 — 以开发者为中心的 Agent 应用框架

**arXiv:** 2508.16279 | **票数:** 61 | **机构:** Alibaba

**核心论点：** AgentScope 1.0 是以开发者为中心的智能体应用框架，采用 ReAct 范式作为核心架构，提供统一模型接口、灵活工具管理、短期与长期记忆系统及企业级部署能力。

**方法论：**
- **基础组件层：** 统一消息原语（多模态）、模型抽象（OpenAI/Anthropic/Gemini/DashScope/Ollama 统一接口）、双层记忆（InMemoryMemory + 基于 Mem0 的 LongTermMemory）、分组式工具管理（解决"选择悖论"问题）+ MCP 集成
- **Agent 基础设施：** ReAct Agent 支持实时引导、并行工具调用、状态持久化、非侵入式 Hook 定制
- **内置 Agent：** Deep Research Agent（树状搜索+反思）、Browser-use Agent、Meta Planner（层次化任务分解+动态 Worker 编排）
- **开发者体验：** 评估模块（Task-Solution-Metric-Benchmark）、Studio（对话界面+OpenTelemetry 追踪）、Runtime（FastAPI 部署+Google A2A 协议+安全沙箱）

**主要结论：** 提供从消息传递到生产部署的完整技术栈；分组式工具管理有效解决工具过多导致的性能下降；支持 Google A2A 协议实现跨系统互操作。

---

### 1.4 AgentScope 大规模仿真 — 百万 Agent 并行

**arXiv:** 2407.17789 | **票数:** 41 | **机构:** Alibaba / 中国人民大学

**核心论点：** 解决基于 LLM 的大规模多智能体仿真中的可扩展性、群体多样性和管理困难三大挑战，在 4 台设备上实现百万智能体仿真。

**方法论：**
- **Actor-based 分布式机制：** 无依赖关系的 Agent 并行执行，`to_dist` 函数自动转换集中式→分布式工作流
- **Agent-环境交互：** 环境抽象为特殊 Agent，支持高并发、双向交互和嵌套环境
- **异构配置：** 指定人口总量和各维度分布（年龄/性别/职业/国籍/教育），自动生成多样化背景

**主要结论：**
- 100 万 Agent 在 4 台设备上 12 分钟完成（串行需约 12 天）
- 线性可扩展性：设备数量增加时运行时间成比例减少
- "猜 2/3 平均值"博弈验证行为真实性：思维链使 Agent 趋向纳什均衡；教育程度越高数字越低

---

### 1.5 Mem0 — 面向生产的 Agent 长期记忆

**arXiv:** 2504.19413 | **票数:** 50 | **机构:** Mem0.ai

**核心论点：** LLM 固定上下文窗口是跨会话一致性的根本障碍。Mem0 提出以记忆为中心的架构，动态提取、整合、检索对话关键信息，实现持久记忆。进一步提出 Mem0^g（图记忆增强版），用有向标记图捕获实体间复杂关系。

**方法论：**
- **Mem0 基础架构：** 两阶段流水线——提取阶段用 LLM 从消息对中提取显著记忆；更新阶段对每条候选事实检索语义相似记忆，LLM 通过 tool call 决定 ADD/UPDATE/DELETE/NOOP
- **Mem0^g 图记忆：** 有向标记图 G=(V,E,L)，双策略检索（实体中心法+语义三元组法）
- **实现：** GPT-4o-mini + Neo4j + 向量数据库

**主要结论：**
- LOCOMO 基准 J 分数：Mem0 66.88，Mem0^g 68.44，显著超 RAG（~61%）、OpenAI 记忆（52.90）、Zep（65.99）
- 相比全上下文方法（J=72.90），p95 延迟降低 91%（1.44s vs 17.1s），token 节省超 90%
- 记忆占用仅 ~7K token/对话（Zep 超过 600K）

**局限性：** 全上下文方法纯准确率仍略高；图操作引入额外延迟；图记忆对多跳推理未带来预期增益。

---

## 第二部分：文档解析与 OCR（2 篇）

两篇论文不约而同地用不到 2B 的小模型达到 SOTA，展现了"小而精"的工程路线。

### 2.1 MinerU2.5 — 解耦式高分辨率文档解析

**arXiv:** 2509.22186 | **票数:** 155（当日最高）| **机构:** 上海 AI Lab / 北京大学 / 上海交大

**核心论点：** 端到端高分辨率文档解析面临 O(N²) 复杂度瓶颈。MinerU2.5 提出"粗到精"的解耦两阶段策略，以仅 1.2B 参数实现 SOTA 精度，同时保持高效推理。

**方法论：**
- **模型架构：** Qwen2-Instruct 0.5B 解码器 + NaViT 675M 视觉编码器 + Pixel-unshuffle patch merger
- **两阶段解析：** 第一阶段在降采样图像（1036×1036）做布局分析；第二阶段在原始分辨率裁剪区域做细粒度识别
- **三阶段训练：** Stage 0 模态对齐（1.2M 样本）→ Stage 1 文档预训练（6.9M 样本）→ Stage 2 精调（630K 样本，IMIC 策略自动挖掘困难样本）
- **创新设计：** PageIoU 布局评测指标；ADR 公式原子分解重组框架；OTSL 表格结构语言（序列长度减少约 50%）

**主要结论：**
- OmniDocBench 总分 90.67（超越 MonkeyOCR-pro-3B 的 88.85）
- 吞吐量 2.12 页/秒（A100 80GB），比 MonkeyOCR-Pro-3B 快 4 倍，比 dots.ocr 快 7 倍
- 文本编辑距离 0.047、公式 CDM 88.46、表格 TEDS 88.22，均为最佳

**局限性：** 裁剪分辨率上限 2048×28×28；极端嵌套公式仍有挑战；主要针对中英文训练。

---

### 2.2 PaddleOCR-VL — 0.9B 超紧凑多语言文档解析

**arXiv:** 2510.14528 | **票数:** 123 | **机构:** Baidu/PaddlePaddle

**核心论点：** 以仅 0.9B 参数的视觉语言模型支持 109 种语言的文本、表格、公式和图表识别，在极低资源消耗下取得多基准 SOTA。

**方法论：**
- **两阶段流水线：** PP-DocLayoutV2（RT-DETR）版面分析 + PaddleOCR-VL-0.9B 元素识别
- **模型架构：** NaViT 动态分辨率视觉编码器 + 2 层 MLP 投影器 + ERNIE-4.5-0.3B 语言模型 + 3D-RoPE 位置编码
- **训练：** 预训练 2900 万图文对 + 微调 270 万精选样本（OCR/表格/公式/图表四类任务）

**主要结论：**
- OmniDocBench v1.5 综合得分 92.86（SOTA）
- olmOCR-Bench 总通过率 80.0%（最高）
- 支持 109 种语言
- A100 上 FastDeploy 达 1.6184 页/秒，比 MinerU2.5 吞吐量高 53.1%

**局限性：** 两阶段设计增加系统复杂度；图表评估仅限内部基准。

**MinerU2.5 vs PaddleOCR-VL 对比：** MinerU2.5 走解耦 VLM 路线（单模型端到端识别），PaddleOCR-VL 走传统流水线路线（版面分析+识别分离）。PaddleOCR-VL 在吞吐量和语言覆盖上更优，MinerU2.5 在架构简洁性上更优。两者在 OmniDocBench 不同版本上各有领先。

---

## 第三部分：语音与多模态生成（2 篇）

### 3.1 VibeVoice — 长篇多说话人语音合成

**arXiv:** 2508.19205 | **票数:** 149 | **机构:** Microsoft Research

**核心论点：** 在 64K 上下文窗口内合成长达 90 分钟、最多 4 个说话人的对话语音。核心创新是超低帧率（7.5 Hz）因果语音分词器，实现 3200 倍压缩率（比 Encodec 提高约 80 倍）。

**方法论：**
- **双分词器设计：** 声学分词器采用 sigma-VAE 实现 7.5 tokens/秒；语义分词器通过 ASR 代理任务训练
- **模型架构：** Qwen2.5（1.5B/7B）LLM + 4 层扩散头，采用 next-token diffusion 预测连续 VAE 特征
- **训练策略：** 冻结分词器，课程学习逐步将序列长度从 4096 增加到 65536

**主要结论：**
- 主观评估中，VibeVoice-7B 在现实感（3.71）、丰富度（3.81）和偏好度（3.75）上均优于 Gemini 2.5 Pro Preview TTS 和 Elevenlabs v3 alpha
- WER 最低（Whisper: 1.29%），说话人相似度 SIM 0.692
- 声学分词器以 7.5 Hz 取得最优 PESQ（3.068）和 UTMOS（4.181）

**局限性：** 仅支持中英文；不处理背景噪音/音乐/音效；不支持重叠语音。

---

### 3.2 daVinci-MagiHuman — 单流架构音视频生成

**arXiv:** 2603.21986 | **票数:** 120 | **机构:** 上海交大 SII-GAIR / Sand.ai

**核心论点：** 采用单流（single-stream）15B Transformer 在统一 token 序列中联合处理文本、视频、音频，避免多流架构复杂性。支持多语言（中英日韩德法）有声视频生成。

**方法论：**
- **单流 Transformer：** 40 层三明治布局（首尾各 4 层模态特定，中间 32 层共享）
- **无时间步去噪：** 不使用显式时间步嵌入，模型直接从含噪输入推断去噪状态
- **推理加速四件套：** 潜空间超分辨率 + Turbo VAE 解码器 + MagiCompiler 全图编译 + DMD-2 蒸馏（8 步去噪，无需 CFG）

**主要结论：**
- 视觉质量 4.80、文本对齐 4.18（均最高）；WER 14.60%（Ovi 1.1 为 40.45%）
- 人工评估：对 Ovi 1.1 胜率 80.0%，对 LTX 2.3 胜率 60.9%
- 单张 H100 生成 5 秒 256p 视频仅需 2 秒，5 秒 1080p 仅需 38.4 秒
- 完整开源

**局限性：** 物理一致性得分略低于 LTX 2.3；未与闭源模型直接比较。

---

## 第四部分：RAG 与 LLM 推理基础设施（2 篇）

### 4.1 LightRAG — 图结构增强的轻量 RAG

**arXiv:** 2410.05779 | **票数:** 37 | **机构:** 北京邮电大学 / 香港大学

**核心论点：** 现有 RAG 的扁平数据表示缺乏上下文感知。LightRAG 将图结构融入索引和检索，通过双层检索范式（低层细粒度实体 + 高层跨实体主题）实现全面高效检索。

**方法论：**
- **图增强索引：** LLM 提取实体和关系 → LLM Profiling 生成 key-value 对 → 去重合并
- **双层检索：** 低层（特定实体属性/关系）+ 高层（跨实体主题信息）+ 混合模式
- **增量更新：** 新文档直接与现有图做 union 合并，无需重建

**主要结论：**
- UltraDomain 四个数据集上全面超越 NaiveRAG、RQ-RAG、HyDE、GraphRAG
- 检索 token 消耗不足 100（GraphRAG 需 610,000），仅需 1 次 API 调用（GraphRAG 需数百次）
- 消融实验：仅用图（去除原始文本）性能下降很小甚至有所提升

**局限性：** 评估主要用 LLM-as-Judge；实体提取质量受 LLM 限制；简单事实查询可能有不必要开销。

---

### 4.2 vLLM/PagedAttention — KV Cache 高效内存管理

**arXiv:** 2309.06180 | **票数:** 50 | **机构:** UC Berkeley | **发表:** SOSP 2023

**核心论点：** 现有系统因碎片化和冗余复制浪费 60%-80% 的 KV cache 内存。借鉴 OS 虚拟内存与分页技术，提出 PagedAttention 实现高效灵活的 KV cache 管理。

**方法论：**
- 将 KV cache 分成固定大小"块"，通过块表实现逻辑-物理映射（类似 OS 页表）
- 物理块按需分配，浪费率低于 4%
- Copy-on-Write 共享机制：并行采样场景减少 55% 内存

**主要结论：**
- 相比 HuggingFace Transformers 提升高达 24 倍吞吐量
- 相比 TGI 提升 3.5 倍，相比 FasterTransformer/Orca 提升 2-4 倍
- LMSYS Chatbot Arena 实际部署：30 倍吞吐提升，GPU 使用量减少 50%

**局限性：** 主要关注单机场景；间接寻址对极小模型有少量延迟开销。

---

## 第五部分：其他前沿方向（4 篇）

### 5.1 TradingAgents — 多 Agent 金融交易框架

**arXiv:** 2412.20138 | **票数:** 35 | **机构:** Tauric Research

**核心论点：** 模拟真实交易公司组织架构，七类专业 Agent 角色（基本面/情绪/新闻/技术分析师、看多/看空研究员、交易员、风险管理团队）通过辩论和协作综合决策。

**主要结论：** 2024 年 1-3 月测试，AAPL 累计收益 26.62%，Sharpe 比率 8.21，最大回撤 0.91%，显著优于所有基线策略。

**局限性：** 仅 3 个月 3 支股票，样本有限；未实盘验证；API 成本较高。

---

### 5.2 AI Scientist-v2 — Workshop 级自动化科学发现

**arXiv:** 2504.08066 | **票数:** 21 | **机构:** Sakana AI

**核心论点：** 完全自主的科学发现系统，通过渐进式 Agent 树搜索探索研究假设空间，集成 VLM 反馈迭代优化图表，内置审稿机制。

**主要结论：** 向 ICLR Workshop 提交 3 篇自主生成论文，其中 1 篇超过人类平均接受阈值并被接收——AI 历史上的重要里程碑。

---

### 5.3 TimesFM — 时间序列基础模型

**arXiv:** 2310.10688 | **票数:** 18 | **机构:** Google Research

**核心论点：** 受 LLM 启发，设计 200M 参数的 decoder-only 时间序列基础模型，零样本预测性能接近针对每个数据集单独训练的 SOTA 监督模型。

**方法论：**
- **Patching：** 将时间序列分成 patch（类似 NLP 的 token），输入 patch 长度 32，输出 128
- **预训练数据 O(100B) 时间点：** Google Trends、Wikipedia Pageviews、合成 ARMA 数据等
- 80% 真实 + 20% 合成数据混合训练

**主要结论：** Monash 数据集零样本排名第一，超过 N-BEATS 等监督模型；性能随 FLOPs 单调递减（17M → 200M），展现良好缩放规律。

**局限性：** 仅 200M 参数；聚焦单变量预测；预训练数据领域偏差。

---

### 5.4 LeWorldModel — 稳定的端到端 JEPA 世界模型

**arXiv:** 2603.19312 | **票数:** 18

**核心论点：** 仅用两个损失项实现稳定的端到端 JEPA 训练，大幅简化现有方法（可调超参数从 PLDM 的 6 个减少到 1 个）。仅 15M 参数，单 GPU 训练，规划速度比基础模型方法快 48 倍。

**方法论：**
- **架构：** ViT-Tiny 编码器（~5M）+ Transformer 预测器（~10M，AdaLN 动作条件化）
- **训练目标：** L_LeWM = L_pred + λ × SIGReg(Z)，SIGReg 基于 Cramer-Wold 定理通过随机投影防止表示坍塌
- **规划：** 交叉熵方法（CEM）+ 模型预测控制（MPC）

**主要结论：** PushT 任务比 PLDM 高 18%；出现涌现的物理理解——潜在路径自发"拉直"，违反期望实验证实模型捕捉了物理结构。

**局限性：** 短规划视野（约 5 步）；依赖离线数据集；需要显式动作标注。

---

## 趋势分析

### 1. Agent 技术全链路成熟化

今日 5 篇 Agent 论文覆盖了完整技术栈：**训练**（Agent Lightning 的通用 RL 框架）→ **持续进化**（MetaClaw 的元学习）→ **应用框架**（AgentScope 1.0 的生产级工具）→ **长期记忆**（Mem0 的图增强记忆）→ **大规模仿真**（AgentScope 百万 Agent）。这标志着 Agent 研究已从"单点突破"进入"系统工程"阶段。

### 2. 小模型文档理解的实用主义路线

MinerU2.5（1.2B）和 PaddleOCR-VL（0.9B）都在用不到 2B 的参数达到 SOTA。两者策略不同但目标一致：在边缘设备和成本敏感场景下实现高精度文档解析。这与大模型的"规模至上"形成鲜明对比，反映了工业界对部署效率的强烈需求。

### 3. 统一架构的多模态生成

VibeVoice 用 next-token diffusion 统一语音生成，daVinci-MagiHuman 用单流 Transformer 统一文本/视频/音频。两者都在追求"一个架构搞定一切"，并都选择了 LLM 骨干网络（Qwen2.5）作为基础。这暗示未来多模态生成模型将进一步趋同于 LLM 架构。

### 4. 生产级基础设施持续吸引关注

vLLM（75K GitHub stars）和 LightRAG（31K stars）作为相对成熟的项目仍在获得大量投票，说明社区对可直接部署的工具有持续且旺盛的需求。Mem0（51K stars）的上榜也印证了这一趋势——"能用"比"新奇"更重要。

### 5. AI 自主性边界持续扩展

AI Scientist-v2 实现了首篇 AI 全自主论文被会议接收；MetaClaw 让 Agent 在部署后自主进化；TradingAgents 模拟完整交易公司。AI 系统的自主性正从单一任务执行向复杂工作流自主运作扩展。

## Open Questions

- Agent RL 训练（Agent Lightning）与 Agent 持续学习（MetaClaw）能否结合？前者提供离线训练基础，后者提供在线适应能力
- 小模型文档解析路线（<2B）的天花板在哪里？当前任务是否已接近饱和？
- 单流 Transformer 统一多模态生成是否是最终架构？还是会出现更高效的模态交互方式？
- AI Scientist-v2 被 Workshop 接收是否意味着 AI 自主科研的实质性突破，还是仅限于特定领域的初步探索？

## References

- [MinerU2.5](https://huggingface.co/papers/2509.22186) — 上海 AI Lab, 2025
- [VibeVoice](https://huggingface.co/papers/2508.19205) — Microsoft Research, 2025
- [Agent Lightning](https://huggingface.co/papers/2508.03680) — Microsoft Research, 2025
- [MetaClaw](https://huggingface.co/papers/2603.17187) — UNC-Chapel Hill et al., 2026
- [PaddleOCR-VL](https://huggingface.co/papers/2510.14528) — Baidu, 2025
- [daVinci-MagiHuman](https://huggingface.co/papers/2603.21986) — SJTU/Sand.ai, 2026
- [AgentScope 1.0](https://huggingface.co/papers/2508.16279) — Alibaba, 2025
- [Mem0](https://huggingface.co/papers/2504.19413) — Mem0.ai, 2025
- [vLLM/PagedAttention](https://huggingface.co/papers/2309.06180) — UC Berkeley, SOSP 2023
- [AgentScope 大规模仿真](https://huggingface.co/papers/2407.17789) — Alibaba/人民大学, 2024
- [LightRAG](https://huggingface.co/papers/2410.05779) — 北邮/港大, 2024
- [TradingAgents](https://huggingface.co/papers/2412.20138) — Tauric Research, 2024
- [AI Scientist-v2](https://huggingface.co/papers/2504.08066) — Sakana AI, 2025
- [TimesFM](https://huggingface.co/papers/2310.10688) — Google Research, 2023
- [LeWorldModel](https://huggingface.co/papers/2603.19312) — 2026
