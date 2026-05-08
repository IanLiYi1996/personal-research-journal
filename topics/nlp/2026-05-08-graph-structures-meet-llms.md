# 图结构 × 大模型：从依存句法到 GraphRAG 的演化谱系

- **创建日期**: 2026-05-08
- **Tags**: #nlp #graph-structures #amr #dependency-parsing #srl #rst #graphrag #graph-of-thoughts #knowledge-graph #cot-graph
- **类型**: 专题笔记（年型 / 持续更新）

> 本文梳理 NLP 中 "**用图来表达语言/知识/推理**" 这条研究脉络在大模型时代的延续与变形。重点不是堆砌论文清单，而是把**经典语言学结构**（句法/语义/篇章依存）和**LLM 时代的图工作**（CoT graph / GraphRAG / Graph-of-Thoughts / KG-LLM）放在同一张地图上，看它们各自解决什么问题、互相之间什么关系。

## 0. 一张总图

```
                         图结构 × 语言 / 推理
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
   [语言学结构]                [LLM 推理图]              [外部知识图]
        │                         │                         │
   词↔词依存                  思想↔思想依存             实体↔实体关系
        │                         │                         │
   ┌────┼────┬─────┬───┐       ┌──┴──┐                ┌──┴──┐
   句法  语义   语义   篇章       CoT-G  GoT/ToT         KG    GraphRAG
  依存   依存   角色   关系
  (UD)   (SDP)  (SRL/  (RST/
                AMR)   PDTB)
```

三条线看似独立，**核心问题都是同一个**：自然语言是序列，但其底层的语义、推理、知识却是图状的 — 怎么把图表征**"嫁接"**到只擅长处理序列的模型上？

经典 NLP 时代的答案是**显式建图、专门解析器**（dependency parser、AMR parser…）。
LLM 时代的答案分两派：
- **隐式派**：相信结构已被 LLM 内化（KisMATH 的 CCG 即是事后从 CoT 中抽出的图）
- **显式派**：构图后再喂给 LLM 或让 LLM 在图上推理（GraphRAG、GoT、KG-LLM）

下文按四大子领域展开。

---

## 1. 推理图 (Reasoning DAG / CoT Graph)

### 1.1 经典前身：从 Proof Tree 到 Discourse Graph

- **逻辑/定理证明的 proof tree**：每一步推理是节点，边是逻辑蕴含。从 Robinson 1965 归结法、Prolog 的 SLD-tree 一脉相承
- **认知科学中的 Discourse Representation Theory (DRT)**（Kamp 1981）：把多句话编码为含变量的逻辑结构，可视为推理图的前身

### 1.2 LLM 时代的两条路：事前生成 vs 事后抽取

#### Tree-of-Thoughts (ToT, Yao et al. 2023, NeurIPS)

- 提示阶段让 LLM 显式生成多个候选思路，用搜索（BFS/DFS）维护一棵树
- 关键贡献：把 CoT 的"一条线"扩展为"一棵树"，引入**剪枝/回溯**机制
- 局限：仍是树（无环），子问题无法重用

#### [Graph of Thoughts (GoT, Besta et al. 2023, AAAI 2024)](https://arxiv.org/abs/2308.09687)

- 把 LLM 思路单元抽象为图节点，**任意有向边**：可合并、聚合、循环反馈
- 在排序任务上比 ToT 提升 62% 质量同时降 31% 成本
- **意义**：第一次明确把"思想空间"建模为通用图，呼应了人类大脑的递归网络结构
- **局限**：图结构由 prompt-engineer 预设，不是模型自发生成

#### KisMATH (Saha et al. 2025) — 事后抽图派

- 详见 [独立笔记](../../papers/2025-kismath-causal-cot-graph.md)
- 不预设图，**从 LLM 自然产生的 CoT 中用 SymPy 解析自动抽取因果 DAG**
- 证明：LLM 内部其实已经"实现"了类似图的结构 — 沿 CCG 路径的 token 概率系统性高于随机路径

#### [Thought Anchors (Bogdan et al. 2025)](https://arxiv.org/abs/2506.19143)

- 句子级因果分析，发现存在 "thought anchors" — 通常是 planning / backtracking 句子，对后续推理有不成比例影响
- 用 attention suppression 给出可靠的因果代理（明确指出 attention aggregation 不可靠）

### 1.3 2025-2026 推理图新进展

#### [Topologies of Reasoning (Besta et al. 2024)](https://arxiv.org/abs/2401.14295)

- GoT 团队的后续，**第一个推理拓扑学综述与分类法**
- 把 Chain / Tree / Graph 各种 prompt 范式抽象成"图拓扑 + 节点函数 + 调度策略"三元组
- 核心贡献：定义了 **structure-enhanced LLM reasoning** 的统一蓝图，可作为后续工作的元框架

#### [Adaptive Graph of Thoughts (AGoT, Pandey et al. 2025)](https://arxiv.org/abs/2502.05078)

- 关键升级：**动态构建 DAG**，递归分解复杂查询为子问题
- 只有"需要进一步分析"的子问题才被展开 → 计算资源按需分配
- 在多跳检索 / 科学推理 / 数学问题上**最高 +46.2%**
- **意义**：把 GoT 的"静态预设图"升级为"运行时生长的图"，更接近人类思考方式

#### [Atom of Thoughts (AoT, Teng et al. 2025)](https://arxiv.org/abs/2502.12018)

- 观察：复杂推理常可分解为一系列**自包含、可验证**的"原子子问题"
- 把每次状态转移建模为：**当前问题 → DAG 分解 → 收缩为新原子问题**
- 借鉴**马尔可夫过程**思想：原子状态是无记忆的，避免历史信息干扰
- 与 KisMATH 共振：CCG 末端节点也具有自包含性

#### [Knowledge Graph of Thoughts (KGoT, Besta et al. 2025)](https://arxiv.org/abs/2504.02670)

- 把 GoT 与动态构建的 KG 结合：LLM 推理 + 工具增强（数学求解器、爬虫、Python）填充 KG
- **GAIA benchmark 上 GPT-4o-mini + KGoT 相比 HF Agents 提升 29%**，同时**成本降低 36 倍**
- **意义**：让小模型借助显式图也能完成复杂任务，对降低 agent 成本有重大启示

#### [Graph-R1 (Liu et al. 2025)](https://arxiv.org/abs/2508.20373)

- 大胆设想：用 **NP-hard 图问题作为合成 Long CoT 训练语料**
- NPH 问题天然要求**深推理 + 探索 + 反思**，正是 Long CoT 的核心特征
- Two-stage post-training：在拒绝采样的 NPH 实例上 SFT → 细粒度 reward 的 RL
- Graph-R1-7B 在数学/代码/STEM/逻辑上**超越 QwQ-32B**
- **意义**：跳出"必须用人工数学/代码题"的 RLVR 数据瓶颈

#### [Graph-Based CoT Pruning (Wang et al. 2026)](https://arxiv.org/abs/2604.05643)

- 把每条 linear CoT 转换为带依赖边的 DAG → 双重剪枝（branch + depth）
- 解决 RL 后训练带来的 "overthinking" — 模式化盲目反思 / 重复确认
- 三阶段 pipeline: SFT (concise) → DPO (correct-yet-shorter) → ...
- **与 KisMATH 直接呼应**：CCG 不仅可以做事后分析，还可以**主动用于训练时压缩冗余反思**

### 1.4 与经典语言学结构的对照

| 维度 | 句法依存树 | AMR | CoT graph (KisMATH) | GoT/AGoT/AoT |
|---|---|---|---|---|
| 节点 | 词 token | 概念 (PropBank frame) | 数学表达式 | 思想单元（句/段/原子问题）|
| 边 | 句法关系（subj/obj/...）| 语义角色（ARG0/1/...）| 因果依赖 | 任意聚合/反馈/分解 |
| 构造 | 监督训练的 parser | 监督 parser + 手工本体 | 符号解析自动 | LLM + prompt / 运行时生长 |
| 任务粒度 | 单句 | 单句 | 单题 | 单任务 |
| LLM 关系 | 输入特征 / probing 目标 | 输入/输出格式 | 事后分析工具 | 推理时显式 |

**有意思的观察**：CoT graph 在精神上**最接近 AMR** — 都是把表面文本里的概念抽出来作为节点，加上语义/因果边。区别是 AMR 强调单句完整语义，CoT graph 强调多步推理依赖。**AGoT 与 AoT 的"运行时动态生长 DAG"则更进一步 — 不依赖人工本体也不依赖完整轨迹后抽取，更像人类的"思考时构造图"**。

---

## 2. 知识图与检索 (KG / GraphRAG)

### 2.1 经典前身：Semantic Networks 到 Knowledge Graphs

- **Semantic Networks** (Quillian 1968)：节点=概念，边=关联类型 — 认知科学起源
- **Frame Semantics** (Fillmore 1976) → **FrameNet** / **PropBank**：动词作为框架，论元作为槽
- **DBpedia / Wikidata / YAGO**：大规模 RDF 三元组（s, p, o）；本质上是有向多重图
- **Knowledge Graph Embedding**：TransE (Bordes 2013) → ComplEx → RotatE 等，把图嵌入连续空间

LLM 之前，KG 与 NLP 的结合主要是 entity linking、KGQA、relation extraction 这些任务。LLM 之后，关系发生了根本变化。

### 2.2 LLM × KG 三种范式（Pan et al. 2023）

[Unifying Large Language Models and Knowledge Graphs: A Roadmap](https://arxiv.org/abs/2306.08302)（路线图论文）把 LLM × KG 关系总结为三类：

1. **KG-enhanced LLM**：用 KG 增强 LLM 的预训练/推理（如 ERNIE、K-BERT）
2. **LLM-augmented KG**：用 LLM 做 KG 补全、构建、文本到图（如 LLM 抽取三元组）
3. **Synergized**：双向，互相增强（GraphRAG 是这一类的代表）

### 2.3 代表工作

#### [Microsoft GraphRAG (Edge et al. 2024)](https://arxiv.org/abs/2404.16130)

- **核心思想**：先用 LLM 从语料抽实体 + 构建 entity KG → 用 community detection 聚合 → 预生成 community summaries → 查询时多级聚合
- **解决的问题**：朴素 RAG 在"全局问题"（"这个数据集的主题是什么"）上失效；GraphRAG 通过图结构提供 corpus-level sensemaking
- **意义**：把 RAG 从"向量检索"升级为"结构检索 + 多级摘要"
- 已开源 Python 实现，在 1M token 级别语料上实证有效

#### [Reasoning on Graphs (RoG, Luo et al. 2024)](https://arxiv.org/abs/2310.01061)

- planning–retrieval–reasoning 框架
- LLM 先生成 KG 中的关系路径作为忠实计划 → 用计划去 KG 检索有效推理路径 → LLM 在路径上做忠实推理
- **关键贡献**：把 KG 的**结构信息**而非仅事实信息也用上 — 之前的 KG-LLM 方法把 KG 当查找表，没用到图拓扑

#### [HippoRAG (Gutiérrez et al. 2024)](https://arxiv.org/abs/2406.14550)

- 灵感来自人脑海马体的 indexing 理论
- 构建 entity graph + Personalized PageRank 做检索
- 在多跳 QA 上比标准 RAG 提升 ~20%

#### [GraphReader (Li et al. 2024)](https://arxiv.org/abs/2406.14550) (注：同 ID 列错，下面校正)

- 4k 上下文窗口的 GraphReader 在 16k–256k 长文本任务上**击败 GPT-4-128k**
- 把长文档结构化为图，用 agent 主动遍历

### 2.4 2025-2026 GraphRAG 关键进展

#### [MindMap (Wen et al. 2024 ACL)](https://arxiv.org/abs/2308.09729)

- 把 KG 作为 prompt 输入，引出 LLM 的 "mind map" 推理路径
- 关键发现：**GPT-3.5 + MindMap 在多个 QA 数据集上稳定超越 GPT-4**
- mind map 既是 LLM 推理过程的可视化，也成为 probing 工具
- **桥接意义**：把 KG 推理与 GoT 推理联系起来 — 一个外部图，一个内部图

#### [TCR-QF (Wu et al. 2025)](https://arxiv.org/abs/2501.15378)

- 解决 GraphRAG 的核心痛点：**KG 抽取过程中的信息损失**
- 三元组的局限：(主语, 谓词, 宾语) 丢掉了原文上下文细节
- 方法：Triple Context Restoration（恢复原文上下文）+ Query-driven Feedback（查询驱动反向补全）
- 在 5 个 QA 基准上 **EM +29.1%, F1 +15.5%**

#### [BYOKG-RAG (Wang et al. 2025)](https://arxiv.org/abs/2507.04127)

- "Bring Your Own KG" — 处理用户自定义 KG（schema/标签都未必标准）
- LLM 不只生成查询，还生成"图工件"（候选实体、推理路径、OpenCypher 查询）
- 多策略图工具协同检索 → LLM 迭代精炼
- 解决 entity-linking 错误传播问题，对**异构 KG 跨域 QA** 鲁棒

#### [PolyG (Zhang et al. 2025)](https://arxiv.org/abs/2504.02112)

- 自适应图遍历：根据查询类型动态选择策略
- 关键观察：**没有一种检索策略对所有查询类型都最优** — 局部 vs 全局、广度 vs 深度需要 case-by-case
- 这是 GraphRAG 走向"工业级鲁棒"的必要一步

### 2.5 与经典 KG 的对照

| 维度 | 传统 KG (Wikidata 等) | GraphRAG | RoG |
|---|---|---|---|
| 构造 | 人工 + crowdsource | LLM 自动抽 | 已有 KG |
| 节点 | 实体（统一 ID）| 实体 mention | 实体 |
| 边 | 有限谓词集 | 自由文本关系 | 有限谓词 |
| 查询方式 | SPARQL | 自然语言 + 多级摘要 | LLM 生成路径模板 |
| 准确性保证 | 强 schema 约束 | 弱（依赖 LLM）| 强（KG ground）|

**演进趋势**：从"严格 schema、人工标注、SPARQL 查询"转向"LLM 抽取、自由 schema、自然语言查询" — 灵活性和覆盖率上升，但**事实严谨性下降**。RoG 这类工作试图找回严谨性。

---

## 3. 语义角色与谓词-论元 (SRL / AMR / SDP)

### 3.1 经典工作

#### Semantic Role Labeling (SRL)

- 起源：FrameNet (Baker 1998)、PropBank (Palmer 2005)
- 任务：找出句子中的谓词及其论元（Agent/Patient/Instrument...）
- 经典系统：Gildea & Jurafsky 2002 的概率模型；后来过渡到 BiLSTM、BERT-based
- **关键公开数据集**：CoNLL-2005, OntoNotes 5.0, FrameNet

#### Abstract Meaning Representation (AMR, Banarescu et al. 2013)

- "对一句话画一张图"：节点是概念（动词框架 + 实体 + 修饰），边是 PropBank 语义角色
- 强调**抽象**：忽略表面句法、时态、可数性等
- AMR parsing 先后经历 transition-based、graph-based、seq2seq、LLM 时代
- AMR-to-text 是反向任务

#### 语义依存 (SDP, Semantic Dependency Parsing)

- DELPH-IN MRS、CCG、Semantic Dependencies (SemEval-2014/15)
- 比句法依存更"深"（捕捉论元共享、ellipsis 等）

### 3.2 LLM 时代的复活与挑战

#### [Ettinger et al. 2023 — LLMs Cannot Reliably AMR-Parse](https://arxiv.org/abs/2305.04535)

- 测试 GPT-3.5/4 在 AMR parsing 上的表现
- 发现 LLM 能产出"看起来像 AMR"的结构，但**关系类型、概念抽象、对齐都频繁出错**
- 暗示：LLM 的内部表示**未必遵循 AMR 的本体论**

#### LLM 用作 AMR generator 然后做下游任务

- 一些工作把 AMR 作为中间表示喂给 LLM，希望提升 faithfulness（如对话状态追踪、QA）
- 结果**喜忧参半**：在某些任务上 AMR 帮助，在另一些任务上反而增加错误传播

#### 是不是该放弃 AMR？

- 反方观点（Stechly 2025 风味）：LLM 已经在内部学到了某种"准 AMR"，显式 AMR parsing 是冗余的
- 正方观点：在需要**可验证、可解释**的任务（法律、医疗、科学）中，显式语义结构仍是不可替代的

### 3.3 2025-2026 AMR 实证再评估

#### [When Does Meaning Backfire? (Pelekhov et al. 2025)](https://arxiv.org/abs/2506.14613)

- **关键的反向证据**：在 NLI 任务上系统测试 AMR 的作用
- 发现：
  - **Fine-tuning 时加 AMR**：损害模型泛化能力
  - **Prompting 时加 AMR**：在 GPT-4o 上有轻微涨点
  - 但消融研究揭示：涨点其实来自"放大表面差异"，**不是真的语义推理**
  - 这种放大反而会误导模型，把核心语义保留的句子判为非蕴含
- **意义**：给"AMR 一定有用"的乐观情绪泼了冷水。在 LLM 时代，**额外的符号化结构未必比模型自己的隐式表征更好**
- 支持 KisMATH 暗示的"结构已被 LLM 内化"立场

#### [Emphasising Structured Information (Park et al. 2024)](https://arxiv.org/abs/2404.01129)

- 反向：把 AMR 作为输入特征**增强**对话评估
- 在开放域对话评估中显式注入 AMR → 显著超越无结构 baseline
- 与上一篇形成有趣对比：**AMR 是否有用，强烈依赖任务和注入方式**

#### 是否还有 AMR 的位置？

综合 2024-2025 证据，可分三种情况：
1. **任务需要可验证语义**（医疗、法律 QA）→ AMR 仍有价值
2. **任务是 NLI、对话生成等**：AMR 可能反而损害
3. **任务是数学/代码推理**：CoT graph 这类领域专用图比通用 AMR 更有效

### 3.4 与推理图的潜在桥接

KisMATH 的 **CCG ≈ 数学版 AMR**：
- 节点：数学表达式（vs AMR 的概念）
- 边：因果包含（vs AMR 的语义角色）

如果说 AMR 是"句子级语义图"，那么 CoT graph 就是"段落级推理图"。**两者本质上是同一个研究纲领在不同粒度上的延伸**。一个开放问题：能否设计一个统一的图本体，从词级（依存）到句级（AMR）到段级（CoT graph）连续过渡？

---

## 4. 篇章结构与话语分析 (RST / PDTB / Coh-Metrix)

### 4.1 经典工作

#### Rhetorical Structure Theory (RST, Mann & Thompson 1988)

- 把篇章看作树状结构：每个节点是 EDU (Elementary Discourse Unit)，边是修辞关系（Cause, Contrast, Elaboration...）
- **核心区分**：nucleus（核）vs satellite（卫星）— 一个句子在篇章中的功能不对等

#### Penn Discourse TreeBank (PDTB, Prasad 2008)

- 不预设全树，标注**句间连接**（显式连词 + 隐式关系）
- 关系类别：Temporal, Contingency, Comparison, Expansion
- 比 RST 更接近"图"而非"树"

#### Coh-Metrix (Graesser et al. 2004)

- 计算文本连贯性的 ~100 个指标
- 词汇/句法/篇章多层级

### 4.2 LLM 时代

#### LLM 生成长文本的 discourse 结构

- 早期观察：GPT-2/3 在长文本中常常出现局部连贯但全局散乱的"discourse drift"
- LLM 时代用 RST-style annotation 评估生成质量（如 Atalla et al. 2023）

#### [GraphReader (Li et al. 2024)](https://arxiv.org/abs/2406.14550)

- 把长文档结构化为图（节点为关键事实，边为篇章关系）让 agent 遍历
- 4k context 击败 128k 直接 attention — 暗示**显式图遍历比隐式长 context 更高效**

#### Discourse 结构作为 LLM 评估维度

- Bamberg & Liang 2024 用 RST 评估 LLM summarization 的结构合理性
- 发现 LLM 生成的摘要在 nucleus-satellite 平衡上接近人类，但在长距离修辞关系上较弱

### 4.3 2025-2026 篇章结构新进展

#### [RST-LoRA (Li et al. 2024 NAACL)](https://arxiv.org/abs/2405.00657)

- **第一个把 RST 显式注入参数高效微调的工作**
- 四种 RST-aware 变体：把 RST 关系类型 + 不确定性融入 LoRA
- 长文档摘要任务上**超越 vanilla LoRA、全参微调和之前 SOTA**
- **意义**：证明经典语言学结构（RST）在 LoRA 时代仍然有用 — 给 PEFT 时代的"反直觉发现"

#### [QUDsim (Wang et al. 2025)](https://arxiv.org/abs/2504.09373)

- 基于 **Questions Under Discussion (QUD)** 理论的话语相似度度量
- 解决一个尖锐问题：**LLM 生成的内容看似多样，实际话语结构高度雷同**
- 实证：LLMs 在不同 prompt 下复用同一套 discourse 结构的频率**显著高于人类**
- **暗示**：LLM 创作能力不足的核心可能不是词汇/句法多样性，而是**话语层面的模式坍塌**
- 这给 RLVR 后训练压平 fork tokens 的故事（见 [CoT Mini-Survey](../../research-notes/2026-05-08-cot-mechanism-mini-survey.md)）提供了篇章层面的旁证

#### [Disco-RAG (Liu et al. 2026)](https://arxiv.org/abs/2601.04377)

- 把 GraphRAG 与 RST/PDTB 风格的篇章结构合体
- 同时构建：**chunk 内的 discourse tree + chunk 间的 rhetorical graph**
- 这两层结构联合作为生成的"plan blueprint"
- 在 QA 和长文档摘要 benchmark 上 SOTA，**无需 fine-tuning**
- **意义**：经典 RST 终于在 RAG 时代找到了生产级应用 — 它解决的就是 RAG 长期忽视的"chunk 结构断裂"问题

### 4.4 与 KisMATH 的呼应

KisMATH 的 R-paths（最长 Q→A 路径）从功能上类似 RST 的 **nucleus chain** — 篇章中"骨架"路径。这暗示：**评价 LLM 推理质量的一个统一思路是看其图骨架的清晰度**，而不是看表面的句子流畅度。

---

## 5. 跨主题：LLM 能否"原生"处理图？

如果上面四条线说明"图结构对语言/推理至关重要"，那一个根本问题是：**LLM 自己能直接处理图输入吗？** 这有专门的研究线：

#### [NLGraph (Wang et al. 2023)](https://arxiv.org/abs/2305.10037)

- 29,370 个图问题，自然语言描述，覆盖 8 个图任务（连通性、最短路、最大流、模拟 GNN 等）
- 发现：GPT-3/4 有初步图推理能力，但**任务复杂度上升时 prompt 增益消失**；对图描述的扰动很脆弱
- 提出 **Build-a-Graph Prompting** 和 **Algorithmic Prompting** — 涨 3-17%

#### [GraphText (Zhao et al. 2023)](https://arxiv.org/abs/2310.01089)

- 把图翻译成自然语言（graph-syntax tree → 序列）
- ChatGPT 配合 GraphText 做图任务，**ICL 即可媲美/超越监督训练的 GNN**
- 启示：**图任务的瓶颈不在"图能力"，而在"图→文本的表示"**

#### Graph Foundation Models（趋势性）

- GraphGPT、LLaGA、GraphLLM 等：尝试把 GNN encoder 接到 LLM 输入端
- 现状：在节点分类/链接预测上 OK，但远未达到 LLM 在文本上那种通用性

#### [GL-Fusion (Wei et al. 2024)](https://arxiv.org/abs/2412.06849)

- **"深度融合"GNN 与 LLM** 的新架构
- 三个核心创新：
  1. **Structure-Aware Transformer**：把 GNN 的 message-passing 直接嵌入 LLM transformer 层 — 同时处理文本和结构
  2. 同时支持 GNN 输出和 LLM 输出（不再受限于分类任务）
  3. 解决之前 LLM-centered（丢图结构）和 GNN-centered（压缩文本）两条路的双向缺陷
- **意义**：可能是 graph foundation model 的下一代起点

#### [NT-LLM (Wang et al. 2024)](https://arxiv.org/abs/2410.10743)

- 提出 **Node Tokenizer** — 把图节点映射为 LLM 可消化的 token
- 保留拓扑信息的同时，避免 graph-to-text 转换的高昂开销
- 与 ViT 的 patch token 思路一致 — 是不是该把图节点也"token 化"？

### 5.5 LLM 原生图工作的两难

综合 NLGraph、GraphText、GL-Fusion、NT-LLM：

| 路线 | 优势 | 局限 |
|---|---|---|
| Graph-to-Text (GraphText) | LLM 零修改即可用 | 转换损失，拓扑信息部分丢失 |
| Token-Level Embedding (NT-LLM) | 保留拓扑 | 依赖额外 tokenizer 训练 |
| Architectural Fusion (GL-Fusion) | 端到端最完整 | 改 LLM 架构，预训练成本高 |

**目前共识**：暂时没有"图原生 LLM"的赢家。短期内最实用还是 Graph-to-Text + 强 prompting；长期可能 Architectural Fusion 胜出。

### 跨主题的核心张力

**显式图表示 vs 隐式图表示**：

- **显式派**主张：把图明确转换成文本/结构化输入再喂 LLM (GraphRAG、RoG、GraphText)
  - 优点：可解释、可验证、可控
  - 缺点：转换损失、依赖额外解析

- **隐式派**主张：LLM 内部已经自然处理了图依赖 (KisMATH、Wang 80/20、CoT 内化派)
  - 优点：通用、零额外开销
  - 缺点：不可控、faithfulness 难保证

**当前实证证据**（综合上文）：
- 简单结构任务，**隐式 OK**（NLGraph 上 GPT-4 有基础能力）
- 复杂或高 stake 任务，**显式更可靠**（医疗 KG QA、法律推理、定理证明）
- **混合派**正在兴起 — KisMATH 用因果图分析隐式结构，GraphRAG 用图增强检索，下一代可能是"LLM 自己内部生成图、外部验证图"的双向流动

---

## 6. 我自己的分析与未来方向

### 三个值得关注的研究方向

1. **跨粒度图本体的统一**：依存（词级）→ AMR（句级）→ CoT graph（段级）→ GraphRAG（语料级）目前各自为政。能否设计统一框架？这对 multi-document reasoning 至关重要。**Disco-RAG 已经把 chunk 内 discourse tree + chunk 间 rhetorical graph 联合，是一个早期信号**

2. **图的"可微"利用**：当前 GraphRAG/RoG 的图操作大多是离散的（检索、遍历），不能端到端梯度回传。能否设计可微的图诱导推理？— **soft graph attention** 或 graph-aware LoRA 是潜在方向。**RST-LoRA 已经用 LoRA 把语言学结构融入参数高效微调；GL-Fusion 把 GNN message-passing 嵌入 transformer 层，是更激进的尝试**

3. **图结构作为 RL 信号**：呼应 [CoT mechanism mini-survey](../../research-notes/2026-05-08-cot-mechanism-mini-survey.md) 中讨论的 RLVR 代价，能否用 R-path 分布形态（钟形 vs 指数型）作为 reward 信号，让 RL 不再压平 fork tokens？**Graph-Based CoT Pruning 已经做了相关尝试 — 用 DAG 结构剪枝 RL 后训练带来的 overthinking**

4. **(新增) 用图问题做合成 RL 数据**：Graph-R1 的"NP-hard 图问题作为 Long CoT 训练语料"是一条全新的路线 — 用**算法难题的天然结构**绕开人工标注瓶颈。值得追踪：合成图问题能否拓展到 agent / 多模态 / 科学推理？

5. **(新增) 篇章结构作为 LLM 创造力的解药**：QUDsim 揭示 LLM 在话语层面坍塌；如果显式注入 RST/QUD 结构能否提升真正的写作多样性，而不是表面词汇多样性？

### 与本笔记库其他笔记的关联

- [KisMATH 论文笔记](../../papers/2025-kismath-causal-cot-graph.md) — CoT graph 的最强实证之一
- [CoT 机制 Mini-Survey](../../research-notes/2026-05-08-cot-mechanism-mini-survey.md) — 推理机制论辩，与本专题第 1 节深度交叉
- [Agentic RL Survey](../../papers/2025-agentic-rl-survey.md) — Agent 行为图与本专题中"长上下文 graph agent"线相关

### 三个开放问题

- **图是否真的存在于 LLM 内部？** KisMATH 给出了数学领域的"是"，但跨领域尚未验证。能否设计语言学领域的"AMR mediation 实验"？
- **显式图能否成为 alignment 工具？** 用图结构约束 RLHF 输出（让 R-path 必须经过特定节点），能否提升 faithfulness？
- **图的"语言"是什么？** 当前主流是把图转 JSON/triple/edge-list 再喂 LLM。能否设计一种**图原生 token**（类似 ViT 的 patch token），让模型直接消化图结构？

## 7. 持续更新领域

下面是近期重点跟踪的几个垂直方向，与主线四大子领域形成补充。

### 7.1 事件图与时序推理 (Event Graph / TKG)

经典基础：**TimeML / TimeBank** (Pustejovsky 2003) — 标注事件、时间表达和它们的关系（BEFORE/AFTER/INCLUDES）；**Event Causality** (Mostafazadeh 2016) — 事件因果链。

LLM 时代的核心挑战：模型在含**多实体、复合时序算子、演化事件序列**的问题上系统性失败。

#### [Narrative-of-Thought (Zhang et al. 2024)](https://arxiv.org/abs/2410.05558)

- 关键发现：**temporal graph generation** 这一任务暴露了 LLM 的全局推理短板，<10B 模型比大模型落后 50%
- NoT 方法：先把事件集合编码为 Python class → 让小模型生成"时序定锚的叙事" → 再用叙事引导生成时序图
- 技术亮点：通过"语言（叙事）↔ 形式（图/类）"的来回切换，把全局推理拆解为局部叙事

#### [GETER (Wang et al. 2025)](https://arxiv.org/abs/2505.15245)

- 完整的"图-文本融合"框架：
  - 时序 KG 编码器抽取结构特征
  - **structure-text prefix adapter** 把图结构特征映射到文本嵌入空间
  - LLM 用 soft graph token + 指令调优 prompt token 一起生成解释
- 同时给出 explainable temporal reasoning 的细粒度基准
- **意义**：提供了一种可微的图注入方式，与 GL-Fusion 思路殊途同归

#### [MemoTime (Liu et al. 2025)](https://arxiv.org/abs/2510.13614)

- 最完整的 TKG-LLM 推理框架
- 把复杂时序问题分解为 **Tree of Time**（层级化、operator-aware）
- 强制单调时间戳，多实体共约束，跨问题积累 reasoning experience
- 解决了之前 TKG-RAG 在多跳 / 多实体时序同步上的脆弱性

#### [TAG-EQA (Chen et al. 2025)](https://arxiv.org/abs/2510.01391)

- 极简方法：**把因果事件图转换成自然语言陈述插入 prompt**
- 9 种 prompt 配置（zero/few-shot × text/graph/text+graph）
- 在 TORQUESTRA 上零样本 +12%，graph-augmented CoT +18%
- **重要消息**：fine-tuning 不必要，结构注入 prompt 即可有效

### 7.2 多模态图（Scene Graph / 3D Scene Graph）与 VLM

经典基础：**Visual Genome** (Krishna 2016) — 图像中的对象、属性、关系标注；**Scene Graph Generation** 是 2017-2020 的核心 CV-NLP 交叉任务。

#### [3DGraphLLM (Zemskova et al. 2024)](https://arxiv.org/abs/2412.18450)

- 首次系统地把**3D scene graph**（不只是 2D）作为可学习表征喂给 LLM
- 关键贡献：**显式建模物体间语义关系**，不止是坐标
- 在 ScanRefer / RIORefer / Multi3DRefer 上 SOTA — embodied AI 任务受益最大
- **意义**：把 GraphRAG 思路从文本推到 3D 空间

#### [ZING-3D (Patil et al. 2025)](https://arxiv.org/abs/2510.21069)

- **Zero-shot 增量式 3D scene graph 构建**：用 VLM 在线生成场景图
- 解决了 3D 场景图构建依赖大规模标注的痛点
- 是 robotics + 长期记忆代理的潜在基础设施

#### 与经典 Scene Graph 的对照

- 经典工作：先训专用 detector → 抽对象 → 训 relation classifier
- LLM 时代：VLM 直接生成图，零样本可用，但**关系类型不受控、可能虚构**
- 趋势：scene graph 不再是中间产物，而是**世界模型表征** — 与 [Causal World Models (Mahajan 2024)](https://arxiv.org/abs/2410.19923) 这条路线汇合

### 7.3 代码图：AST / Call Graph / 仓库依赖图

经典基础：**AST (Abstract Syntax Tree)、CFG、PDG** — 编译器和静态分析的核心；**code2vec / GraphCodeBERT** (2019-2020) 是 transformer-graph 早期融合。

#### [Code Graph Model (CGM, Tan et al. 2025)](https://arxiv.org/abs/2505.16901)

- **首个把仓库级 code graph 嵌入 LLM 注意力机制**的开源工作
- 把函数/文件作为节点，依赖作为边 → 通过专用 adapter 映射到 LLM 输入空间
- 配合 **agentless graph RAG**（不需要 agent 框架）
- 在 SWE-bench Lite 上用 Qwen2.5-72B 达到 **43%** — 开源模型第一名
- **意义**：repo-level 代码任务不必依赖 GPT-4 + 复杂 agent，开源 + 图结构即可

#### [GraphSkill (Liu et al. 2026)](https://arxiv.org/abs/2603.06620)

- 文档驱动的层级 RAG + 自调试代码 agent
- 解决两个被忽视的痛点：(i) 技术文档的层级结构被当成扁平文本；(ii) debug 只看运行时错误，忽视逻辑错误
- 用 top-down 文档遍历 + 自动小测例迭代 refine

#### 与经典 AST 工作的对照

- 经典 AST 嵌入：tree-LSTM、tree-transformer、GraphCodeBERT — 把图当成额外信号
- LLM 时代：AST 作为输入特征效果有限（LLM 已经隐式学到语法）；**真正有效的是 repo 级别的 call graph + import graph** — LLM 单凭文本无法获取
- 启示：经典语言学结构（句法/语义）的层级，对应代码领域是 AST / CFG / Call Graph 的层级 — 越上层的图越有用

### 7.4 GNN + LLM 的"真"融合

GNN-LLM 集成在 2023-2024 出现了一波架构创新，2025 后开始走向"训练范式"创新。

#### 早期：架构融合
- **GraphGPT** (Tang 2023): GNN encoder → LLM
- **LLaGA** (Chen 2024): adapter + 节点序列化
- **GL-Fusion** (Wei 2024)（前已介绍）: 把 message-passing 嵌入 transformer 层
- **NT-LLM** (Wang 2024)（前已介绍）: graph token

#### 新一代：训练范式驱动

##### [G1 (Liu et al. 2025)](https://arxiv.org/abs/2505.18499)

- **用 RL 在合成图任务上训练 LLM**（与 Graph-R1 思路一致但更早）
- 构建 **Erdős 数据集**：50 个图论任务、100k 训练 + 5k 测试，全来自真实图
- **3B 模型经 RL 后超越 Qwen2.5-72B**（24× 体量差距）
- 强 zero-shot 泛化：未见任务、未见域、未见图编码方式
- **意义**：不需要架构改造，**纯靠 RL + 合成数据**即可让 LLM 获得图能力

##### [GraphICL Benchmark (Zhao et al. 2025)](https://arxiv.org/abs/2501.15755)

- 第一个**专为评估图能力的 prompt-only benchmark**
- 揭示了一个尴尬事实：很多"专用 graph LLM"在公平 prompt 设计下并不超过通用 LLM
- 给整个领域提供了校准基线

#### 张力总结

| 路线 | 代表 | 优势 | 缺点 |
|---|---|---|---|
| 架构融合 | GL-Fusion, NT-LLM | 端到端、保留拓扑 | 改 LLM 架构，预训练成本高 |
| Adapter 嵌入 | 3DGraphLLM, GETER, CGM | 可加在现成 LLM 上 | 表达力受限 |
| Prompt 注入 | TAG-EQA, GraphText, GraphICL | 零修改、零样本 | 转换损失 |
| RL 合成训练 | G1, Graph-R1 | 不改架构、强泛化 | 依赖任务设计 |

**当前共识**：**RL + 合成图任务**（G1 / Graph-R1）和 **adapter 注入**（CGM / GETER）是性价比最高的两条路。架构改造留给少数有预训练资源的团队。

---

## 8. 三个开放问题的现有解法

第 6 节列了三个开放问题，这里整理近期的尝试。**注意：现有解法大多是初步探索，不是定论**。

### 8.1 图是否真的存在于 LLM 内部？— Causal Probing 的进展

KisMATH 在数学领域给出了"是"，问题是能否扩展到语言学结构。**好消息：2025 年已有可信工作。**

#### [Causal Interventions Reveal Shared Structure Across Filler-Gap Constructions (Tucker et al. 2025)](https://arxiv.org/abs/2505.16002)

- **直接对应 KisMATH 在语言学领域的版本**
- 研究英语 filler-gap 依存（疑问句、关系从句等）
- 用 **Distributed Interchange Interventions (DII)** — 一种比 attention suppression 更精细的因果干预技术
- 关键发现：
  - LLMs 内部对不同 filler-gap 构式收敛到**类似的抽象分析**
  - 这种抽象与语言学家几十年来的理论基本一致
  - 但同时**揭示了被忽略的因素**（频率、filler 类型、上下文）— 反过来推动语言学理论修正
- **意义**：第一次明确证明**机制可解释性方法可以反哺语言学理论**

#### [CAT: Causal Attention Tuning (Lin et al. 2025)](https://arxiv.org/abs/2509.01535)

- 不是 probing，而是**主动注入因果结构**
- 自动从人类先验生成 token 级因果信号 → Re-Attention 机制引导训练
- 在 OOD 场景下显著优于直接训练 — **说明 LLM 默认学到的是相关而非因果**
- 提出 Spurious Token Game (STG) 基准

#### 当前共识

- **数学领域**: KisMATH 强证据 → 内部有图
- **句法领域**: Tucker 2025 强证据 → 内部有抽象语言学结构
- **代码领域**: CGM 间接证据 → 内部对 repo 拓扑表征不足，需外部注入
- **常识/事件领域**: 弱证据 → 大多需要外部图（TKG / event graph）增强
- **总体趋势**: **越接近形式化的领域（数学、句法），LLM 内化越完整；越偏世界知识/时序的领域，越需要显式图**

### 8.2 显式图能否成为 alignment 工具？— 结构化 RL/RLHF 的进展

KisMATH 末尾观察到 RLVR 压平 fork tokens；问题是能否反过来用图结构约束 RL 让它不要压平？

#### [G1 (Liu et al. 2025)](https://arxiv.org/abs/2505.18499)

- 间接答案：**RL 在图任务上训练时，反而扩展了模型的探索能力**（强 zero-shot 泛化）
- 与 Yue 2025 "RL 不引入新能力"形成有趣对比 — **可能是因为图任务本身要求探索**
- 暗示：**reward 设计如果对应图结构（路径多样性、节点覆盖），RL 不会压平 fork**

#### [Graph-Based CoT Pruning (Wang et al. 2026)](https://arxiv.org/abs/2604.05643)（前面介绍过）

- 直接答案：用 DAG 显式剪枝 RL 后训练带来的 overthinking
- 三阶段 SFT → DPO → 进一步精炼
- **是当前最直接的"图作 alignment 工具"实践**

#### [Causal World Models meet LLMs (Mahajan et al. 2024)](https://arxiv.org/abs/2410.19923)

- 走得更远：把因果表示学习（CRL）学到的因果世界模型作为 LLM 推理的"模拟器"
- LLM 通过自然语言 query 这个图结构的世界模型
- 在因果推断 + 规划任务上有效
- **意义**：图不只是 alignment 工具，是 LLM 的**外置因果推理引擎**

#### 当前共识

- **直接用图约束 RLHF 的 reward**：尚未看到主流工作（机会窗口）
- **用图引导 RL 数据合成（G1/Graph-R1）**：已被验证有效
- **用图剪枝 RL 副作用（Graph-Based CoT Pruning）**：早期但有希望
- **用图作为外部世界模型**（Causal World Models）：最雄心勃勃，技术成熟度最低

**研究空白**：能否用 KisMATH 的 R-path 分布形态作为 PPO/GRPO 的辅助 reward？这是一个具体的、可立即尝试的方向。

### 8.3 图原生 token：图能否被像 ViT 的 patch 那样消化？

#### [NT-LLM (Wang et al. 2024)](https://arxiv.org/abs/2410.10743)（前面介绍过）

- 把节点直接编码为 LLM token，保留拓扑信息
- 思路接近 ViT 的 patch token，但只覆盖节点不覆盖整图
- **属于早期探索 — 还没有"GraphFormer"那种统一架构**

#### [GETER (Wang et al. 2025)](https://arxiv.org/abs/2505.15245)（前面介绍过）

- soft graph token + 文本 prompt token 拼接
- 通过 prefix adapter 把图特征映射到文本空间
- **这是当前最接近"图原生 token"的实践方案**

#### [3DGraphLLM (Zemskova et al. 2024)](https://arxiv.org/abs/2412.18450)（前面介绍过）

- 类似思路，但针对 3D scene graph
- 把节点关系（不只是节点本身）也 token 化

#### 当前共识

- **节点级 token 化**: NT-LLM、GETER 已基本成熟
- **完整图 token 化**: 未解决 — 节点数动态、拓扑信息难以一维序列化
- **多层次（节点/子图/全图）token**: 是一个开放设计空间，目前只有零星探索
- **训练范式**: 是否需要类似 MAE 的"masked graph modeling"预训练任务？尚未看到定论

**最大空白**：还没有一个像 ViT 那样"统一所有图任务"的 graph foundation tokenizer。这是 1-2 年内最有可能出现重大突破的方向。

---

## 9. 四篇代表性论文深度解读

前面的笔记是地图，本节是 4 个标的物的特写。挑选标准：每篇代表本专题的一个核心范式 — **(1) 内化派的最强证据 (2) 推理图的最新形态 (3) 工业级显式图标杆 (4) 不改架构让 LLM 学图**。

### 9.1 Tucker et al. 2025 — Causal Interventions Reveal Shared Structure Across English Filler-Gap Constructions

> **代表的范式**：内化派（KisMATH 在语言学领域的对应物）

#### 研究问题

英语中的 **filler-gap 依存** 是语言学的经典话题：把句子的某个成分挪到非典型位置，留下"空位"。例子：

- 嵌入 wh-questions: "I wonder **what** the man liked __."
- 关系从句: "The boy **who** the man liked __ was..."
- 分裂句 (cleft): "It was the **boy** that the man liked __."
- 主题化: "Actually, the **boy** the man liked __."

语言学家几十年来争论：这些表面不同的结构**底层是不是共享一套语法机制**？(Fodor 1989 等支持共享，Ross 1967、Culicover 1999 等强调差异)

**LLM 给我们一个新答案路径**：如果模型内部确实在用同一套机制处理它们，那么**从一种结构上学到的"机制"应该可以迁移到另一种**。

#### 方法 — Distributed Alignment Search (DAS)

核心干预公式（这是与 KisMATH 的 attention suppression 不同的、更精细的因果干预）：

$$\mathbf{b} + (\mathbf{s}\mathbf{a}^\top - \mathbf{b}\mathbf{a}^\top)\mathbf{a}$$

- $\mathbf{b}$: base 例子的 transformer block 输出
- $\mathbf{s}$: source 例子的对应输出
- $\mathbf{a}$: 学习得到的方向向量（**只有它需要训练，LLM 参数全冻结**）

**直觉**：在旋转过的特征空间里定义一个方向，沿这个方向把 base 替换为 source 的值，**保持其他正交维度不变**。这是一种"软干预"，比注意力屏蔽更精确。

学习目标：在干预后让模型预测 source 期望的 continuation。如果能学到这样的 $\mathbf{a}$，就证明**该位置的激活里编码了被干预的特征**。

#### 实验设计

7 种构式 × 2 种生命性 (animate/inanimate) × 1 或 2 个从句边界 = 28 种模板，共 200×2 = **400 平衡对子**用于训练。

两种干预模式：
1. **Single-source**：从一种构式上训练干预，迁移到其他
2. **Leave-one-out**：从其他所有构式上训练，看能否泛化到留出的那一个

模型：Pythia 1.4B / 2.8B / 6.9B（开源开数据）。

**Odds metric**：干预后 counterfactual 标签的相对概率提升。

#### 发现

| 发现 | 含义 |
|---|---|
| 跨构式机制**强迁移** | LLM 确实用同一套抽象机制处理所有 filler-gap |
| **Lexical boost**：相同生命性时迁移更强 | 与心理语言学的 syntactic priming 一致 |
| 共享语言学特征的构式间迁移更强 | 例如 wh- 类构式之间 |
| 存在 **source constructions** | 它们的机制能广泛迁移到其他构式 |
| 存在 **sink constructions** | 它们持续受益于被迁移的机制 |
| **不能跨从句边界迁移** | 揭示了一个 architectural 限制 |
| 揭示频率/filler 类型/上下文的作用 | 这些因素被传统语言学理论忽视，应予以纳入 |

#### 与 KisMATH 的对比

| 维度 | KisMATH (数学) | Tucker 2025 (句法) |
|---|---|---|
| 任务领域 | 数学推理 | filler-gap 依存 |
| 因果工具 | Attention Suppression | Distributed Alignment Search |
| 图的本质 | 表达式间因果 DAG | 跨构式共享的抽象机制 |
| 证据类型 | "推理节点是中介" + "R-path 概率高于随机" | "机制可在构式间迁移" |
| 结论 | LLM 内部存在数学推理图 | LLM 内部存在共享语法抽象 |

**两篇加起来构成的论断**：在数学和形式语法这两个**形式化领域**，LLM 不只是表面模仿，**内部确实存在抽象结构**。

#### 局限与启示

- 实验在 Pythia 上做（开源开数据），未在最新前沿模型上验证 — 但作者在附录里报告 1.4B/2.8B/6.9B 结果**质性一致**，暗示规律稳健
- "不能跨从句边界"是个有意思的负结果，可能反映 transformer 注意力对长距离 binding 的固有限制
- **方法学贡献最大**：DAS + 跨构式迁移的实验范式可推广到 AMR、SRL、discourse 等其他语言学结构

---

### 9.2 Pandey et al. 2025 — Adaptive Graph of Thoughts (AGoT)

> **代表的范式**：推理图最新形态（test-time 动态生长 DAG，性价比最高）

#### 研究问题

CoT、ToT、GoT 都在 prompt 阶段**预设结构**：CoT 是链，ToT 是树，GoT 是图。但实际推理任务复杂度差异巨大 — **简单子问题不需要展开，复杂子问题才需要进一步分解**。能不能让推理结构在运行时按需生长？

替代方案有：fine-tuning（成本高）、RL post-training（数据贵）。AGoT 给出第三条路：**纯 test-time、不改模型**。

#### 核心设计

**思想 (thought)**：信息单元，是图节点。
**边**：A → B 表示 A 的输出会影响 B 的处理。
**复杂节点**：被 LLM 判断为"需要进一步展开"的节点 → 触发**嵌套 AGoT** 子图。

#### 算法（简化）

```
def AGoT(query, parent_graph):
    G = empty_graph()
    while not done:
        # Layer 生成
        new_nodes = LLM_generate_subproblems(query, G)
        for node in new_nodes:
            if LLM_classifies_complex(node):
                # 递归调用，生成嵌套图
                node.nested = AGoT(node.content, G)
            else:
                node.answer = LLM_evaluate(node.content, G)
        # 自终止判断
        if LLM_judges_high_quality():
            return final_answer
```

两个核心机制：
1. **每层节点数动态**：LLM 决定每层生成多少节点
2. **递归嵌套**：复杂节点直接产生嵌套子图，深度 = 任务复杂度

#### 与前辈对比

| 框架 | 拓扑 | 生成时机 | 自适应能力 |
|---|---|---|---|
| CoT (Wei 2022) | 链 | 一次性 | 无 |
| ToT (Yao 2024) | 树 | 搜索时扩展 | 弱（固定 BFS/DFS）|
| GoT (Besta 2023) | 图 | prompt 预设 | 中（人工配置）|
| AIoT (Radha 2024) | 准线性 | 迭代 | 中 |
| **AGoT (本文)** | **嵌套 DAG** | **运行时生长** | **强（递归 + 复杂度判断）**|

#### 关键结果

GPT-4o-mini + AGoT 在 GPQA（科学推理）上 **+46.2%**，可与 RL 蒸馏效果（DeepSeek-R1 系列声称的提升幅度）持平 — 但 AGoT **不改模型，不要数据**。

在 reasoning / retrieval / explorative 三大类任务上一致涨点，**explorative 任务涨最多**（这类任务本来就需要分解 + 多路探索）。

#### 与 KisMATH 的关系

KisMATH 说 LLM 内部已经隐含一个 CCG，AGoT 说**与其事后挖出来，不如让模型显式构造它**。两者其实是同一个洞察的两面：
- KisMATH：LLM 内部图的存在 → 推理可信
- AGoT：把这个内部图外显化 → 计算可控

#### 局限

- 依赖"复杂度判断"的 prompt — 判断不准的话，简单问题被过度展开浪费算力，复杂问题被低估漏掉细节
- 结构纯由 LLM 生成，**没有验证机制**（与 KisMATH 用 SymPy 解析的严谨度有差距）
- 在数学这类有形式答案的任务上，AGoT 有时不如直接 CoT — 暗示动态生长 DAG 在**开放探索任务**上优势更大

---

### 9.3 Edge et al. 2024 — From Local to Global: GraphRAG (Microsoft)

> **代表的范式**：工业级显式图标杆

#### 研究问题

朴素 RAG 在"全局问题"上失效。例如问 "this dataset 的主要主题是什么"、"过去十年跨学科研究如何影响科学发现"，**没有任何单一文档片段能回答**。这是 **query-focused summarization (QFS)** 任务，不是 retrieval 任务。

但是经典 QFS 方法不能 scale 到 RAG 处理的语料规模（百万 token 级）。GraphRAG 要同时获得：
1. RAG 的可扩展性（百万 token 语料）
2. QFS 的全局理解能力

#### 整体 Pipeline

```
源文档
   ↓ (LLM 抽取)
实体 + 关系 + claims
   ↓ (构图)
全局知识图
   ↓ (Leiden community detection)
社区层级（嵌套）
   ↓ (LLM 摘要)
社区摘要（每层一份）
   ↓ (查询时)
Map: 每个社区摘要独立生成部分回答
   ↓
Reduce: 合成最终全局回答
```

关键设计：
1. **图作为索引**，不是知识库 — 不查询图，而是从图导出 community summary
2. **Leiden 算法**做社区检测 — 比 Louvain 收敛性更好
3. **递归 community summary** — 高层 summary 由低层 summary 生成
4. **Map-Reduce 查询** — 每个社区独立部分回答，最后合并

#### 评估方法（这是论文的另一个贡献）

QFS 没有 ground truth，作者设计了 **adaptive benchmarking**：
1. 用 LLM 生成"用户角色"
2. 用角色生成 corpus-specific 的 sensemaking 问题
3. 用 LLM-as-a-judge 比较 GraphRAG vs vector RAG

评估维度：comprehensiveness（广度）和 diversity（多样性）— 都是全局 sensemaking 才需要的指标。

#### 关键结果

在两个真实数据集（百万 token 级别）上，**GraphRAG 在 comprehensiveness 和 diversity 上都显著优于 vector RAG**。GPT-4 作为底层 LLM 时优势最明显。

#### 工程影响

- 已开源在 https://github.com/microsoft/graphrag
- LangChain、LlamaIndex、NebulaGraph、Neo4J 都集成了变体
- **是 2024-2025 最有影响力的 RAG 工作之一** — 后续 TCR-QF、BYOKG-RAG、PolyG、Disco-RAG 都是在 GraphRAG 框架上的迭代

#### 与本专题其他工作的关系

| 角度 | GraphRAG | KisMATH | AGoT |
|---|---|---|---|
| 图的来源 | LLM 从语料抽取 | 解析器从 CoT 抽取 | LLM 在推理时生成 |
| 图的用途 | 索引 + 摘要 | 事后分析 | 推理调度 |
| 任务规模 | 百万 token 语料 | 单题 | 单查询 |
| 验证机制 | LLM-as-judge | 因果干预 | 自评估 |

**核心 insight**：图的"显式 vs 隐式"在不同尺度对应不同需求 — **语料规模需要显式（GraphRAG）；单题规模隐式即可（KisMATH）；单查询规模运行时生长（AGoT）**。

#### 局限

- LLM 抽取实体/关系不可避免有错（schema 不严格）
- Community summary 是有损压缩 — 局部细节查询反而不如 vector RAG
- 索引构建非常昂贵（每篇文档都要 LLM 调用）— 与 vector RAG 的轻索引形成强对比
- TCR-QF 之后的工作（2025）正是为了解决这些痛点

---

### 9.4 Guo et al. 2025 — G1: Teaching LLMs to Reason on Graphs with RL

> **代表的范式**：不改架构、不要标注，让 LLM 学会图推理

#### 研究问题

LLM 在图任务上**糟糕得令人吃惊**：OpenAI o1 在简单的图连通性任务上准确率仅 **58.49%**。前人的解决方案：
- 各种自然语言图编码（adjacency list / edge list / 真名网络…）— 涨幅有限
- Graph foundation model 预训练 — 受限于"通用图表示"的 scarcity
- 监督指令微调 — 数据贵且容易过拟合

G1 提出第三条路：**RL + 合成图任务**，与 DeepSeek-R1 在数学/代码上的成功类比。

#### 核心创新 — Erdős 数据集

为什么用图论问题作 RL training corpus？
- **可验证答案**：NetworkX 提供 ground truth solver → 自动 verifier
- **难度可控**：从 node counting 到 NP-hard 的 TSP / Maximal Independent Set
- **多样性**：50 个不同任务、各种答案类型（bool / int / float / 节点列表 / 边列表 / 映射）
- **真实分布**：从 Network Repository 采样真实图，而非随机生成

数据集统计：100k 训练 + 5k 测试，4 个难度等级（Easy / Medium / Hard / Challenging）。

#### 训练范式

- 基础模型：Qwen2.5-3B / 7B / 72B-Instruct
- 算法：RLVR（rule-based reward，对照 ground truth 或 verifier 程序）
- 关键设计：选用 **edge list** 单一格式作为图描述方式（实验发现单一格式训练后能正向迁移到其他格式）

#### 关键结果

**任务内表现**：
- Base model (Qwen2.5-7B): Easy 57% / Medium 43% / Hard 19% / Challenging 3%
- G1-7B: Easy **95%** / Medium **89%** / Hard **50%** / Challenging **24%**

**G1-3B（3B 模型）超过 Qwen2.5-72B-Instruct（24× 体量差）**

**零样本泛化**（这是 G1 最有意思的部分）：
- 迁移到 GraphWiz / GraphArena（其他图基准）：稳定提升
- 迁移到 Cora / PubMed（真实图节点分类、链接预测）：base model 也涨
- 不损害通用能力：GSM8K / MATH / MMLU-pro 几乎不掉

#### 与 Graph-R1 (NPG-Muse) 的对比

Graph-R1 (Liu et al. 2025, arXiv 2508.20373) 也用 NP-hard 图问题做 SFT+RL：

| 维度 | G1 | Graph-R1 (NPG-Muse) |
|---|---|---|
| 训练目标 | **图任务能力**本身 | **Long CoT 一般能力** |
| 训练任务 | 50 类图任务 | 3 类 NP-hard（GED, TSP, MCP） |
| Pipeline | 直接 RL | SFT + RL 两阶段 |
| 验证 | NetworkX | 改造 GraphArena 验证 |
| 评估 zero-shot | 跨图任务 / 跨域 / 跨编码 | 跨数学/代码/逻辑/图 |
| 关键发现 | 3B 超 72B 是体量 | 7B 超过 QwQ-32B |

**两者其实是互补的**：G1 证明 RL+图任务**能让模型学会图**；Graph-R1 证明图任务**能教 Long CoT 通用能力**。前者是对图能力本身的训练，后者是把图作为通用推理 catalysist。

#### 与 KisMATH "RL 缩小探索边界"的关系

[KisMATH](../../papers/2025-kismath-causal-cot-graph.md) 末尾观察到 RLVR (Qwen3 32B) 把分布压成指数型 → pass@k 上限下降。但 G1 / Graph-R1 表现出**反向证据**：
- G1 在零样本任务上**扩展了**能力边界
- Graph-R1 的 pass@k 显著上升（表明探索增加）

**可能的统一解释**：
1. KisMATH 观察的 Qwen3 32B 用的是**通用 RLVR**（math + code）— 任务相对低 entropy
2. G1 / Graph-R1 用的是**图任务 RLVR** — 天然要求探索（NP-hard 没有捷径）
3. **关键变量是 reward 是否对应需要高熵 fork token 的任务** — 如果是，RL 不会压平 fork

如果这个假说成立，那么 [Wang 2025 高熵 token 是 RL 真正作用对象](https://arxiv.org/abs/2506.01939) + KisMATH + G1 一起暗示：**"RL 损害探索"不是 RL 的本质，而是 reward 设计的问题。** 用图任务这种天然要求探索的 reward signal，可以避免压平。

#### 局限

- 50 个任务都来自 NetworkX — 局限于经典图论问题，**没有覆盖知识图、scene graph、AST 这类语义图**
- edge list 格式优势在小图（5-35 节点）；超过这个规模可能失效
- 与 GraphRAG 这类应用工作正交：G1 是在让 LLM 内化图能力，GraphRAG 是在外部用图。两者结合（用 G1 训练的 LLM 跑 GraphRAG）尚未被验证

---

### 9.5 四篇深读的整体观察

| 论文 | 派别 | 关键发现 | 对本专题的核心贡献 |
|---|---|---|---|
| Tucker 2025 | 内化派 | LLM 跨构式共享 filler-gap 机制 | 把 KisMATH 的"内化"结论扩展到形式语法 |
| AGoT 2025 | 显式派（test-time） | 运行时生长 DAG = RL 蒸馏的廉价替代 | 显式图操作的最佳性价比方案 |
| GraphRAG 2024 | 显式派（pre-built） | 图索引 + community summary 解决全局 sensemaking | 工业级显式图的标杆 |
| G1 2025 | 训练派 | RL + 合成图任务比预训练/SFT 高效 | "图能力可教"，且不损害通用能力 |

**贯穿四篇的统一信号**：

1. **隐式图 vs 显式图的张力可以缓解** — 在不同尺度（单题/单查询/语料/任务大类）选不同方案
2. **RL 不必损害探索** — 选对 reward 信号（图任务、需要 fork 的任务），RL 反而扩展能力边界
3. **合成数据是图能力的主要瓶颈解药** — 不论是 Erdős（G1）还是 NPH 题目（Graph-R1），都比真实标注更可扩展
4. **因果干预方法学正在从数学迁移到语言学** — Tucker 用 DAS 干预 Pythia，与 KisMATH 用 attention suppression 干预数学 LLM 是同一精神

---

## 10. 持续跟踪计划

### 高优先级监控
- 把图原生 tokenizer / GraphFormer 类工作（接 ViT 类比）
- KisMATH 风格的因果图分析向其他领域扩展（已在句法领域看到 Tucker 2025）
- 图作 RLHF reward 的具体方法
- 图驱动的 RL 数据合成（Graph-R1、G1 之后）

### 已有但需深入的领域
- **时序/事件图**：MemoTime / GETER 之后下一代
- **代码仓库图**：CGM 之后的开源跟进者
- **多模态 scene graph**：3DGraphLLM / ZING-3D 与 robotics 的结合

## References

### 经典语言学结构
- Mann & Thompson 1988. Rhetorical Structure Theory.
- Banarescu et al. 2013. Abstract Meaning Representation for Sembanking.
- Palmer et al. 2005. The Proposition Bank: An Annotated Corpus of Semantic Roles.
- Prasad et al. 2008. The Penn Discourse Treebank 2.0.

### LLM × 推理图
- [Yao et al. 2023. Tree of Thoughts (NeurIPS)](https://arxiv.org/abs/2305.10601)
- [Besta et al. 2023. Graph of Thoughts](https://arxiv.org/abs/2308.09687)
- [Besta et al. 2024. Topologies of Reasoning](https://arxiv.org/abs/2401.14295)
- [Pandey et al. 2025. Adaptive Graph of Thoughts (AGoT)](https://arxiv.org/abs/2502.05078)
- [Teng et al. 2025. Atom of Thoughts (AoT)](https://arxiv.org/abs/2502.12018)
- [Besta et al. 2025. Knowledge Graph of Thoughts (KGoT)](https://arxiv.org/abs/2504.02670)
- [Liu et al. 2025. Graph-R1](https://arxiv.org/abs/2508.20373)
- [Wang et al. 2026. Graph-Based CoT Pruning](https://arxiv.org/abs/2604.05643)
- [Saha et al. 2025. KisMATH](https://arxiv.org/abs/2507.11408)
- [Bogdan et al. 2025. Thought Anchors](https://arxiv.org/abs/2506.19143)

### LLM × KG / GraphRAG
- [Pan et al. 2023. Unifying LLMs and Knowledge Graphs: A Roadmap](https://arxiv.org/abs/2306.08302)
- [Wen et al. 2024 ACL. MindMap](https://arxiv.org/abs/2308.09729)
- [Edge et al. 2024. From Local to Global: A GraphRAG Approach](https://arxiv.org/abs/2404.16130)
- [Luo et al. 2024. Reasoning on Graphs (RoG)](https://arxiv.org/abs/2310.01061)
- [Gutiérrez et al. 2024. HippoRAG](https://arxiv.org/abs/2405.14831)
- [Li et al. 2024. GraphReader](https://arxiv.org/abs/2406.14550)
- [Wu et al. 2025. TCR-QF (Triple Context Restoration)](https://arxiv.org/abs/2501.15378)
- [Zhang et al. 2025. PolyG](https://arxiv.org/abs/2504.02112)
- [Wang et al. 2025. BYOKG-RAG](https://arxiv.org/abs/2507.04127)

### LLM × AMR / SRL
- [Park et al. 2024. Emphasising Structured Information (AMR for dialogue eval)](https://arxiv.org/abs/2404.01129)
- [Pelekhov et al. 2025. When Does Meaning Backfire? (AMR in NLI)](https://arxiv.org/abs/2506.14613)

### LLM × 篇章结构
- [Li et al. 2024 NAACL. RST-LoRA](https://arxiv.org/abs/2405.00657)
- [Wang et al. 2025. QUDsim](https://arxiv.org/abs/2504.09373)
- [Liu et al. 2026. Disco-RAG](https://arxiv.org/abs/2601.04377)

### LLM × 图原生处理
- [Wang et al. 2023. NLGraph](https://arxiv.org/abs/2305.10037)
- [Zhao et al. 2023. GraphText](https://arxiv.org/abs/2310.01089)
- [Wang et al. 2024. NT-LLM (Node Tokenizer)](https://arxiv.org/abs/2410.10743)
- [Wei et al. 2024. GL-Fusion](https://arxiv.org/abs/2412.06849)
- [Liu et al. 2025. G1 (RL on Erdős)](https://arxiv.org/abs/2505.18499)
- [Zhao et al. 2025. GraphICL Benchmark](https://arxiv.org/abs/2501.15755)

### 事件图 / 时序推理
- [Zhang et al. 2024. Narrative-of-Thought](https://arxiv.org/abs/2410.05558)
- [Wang et al. 2025. GETER](https://arxiv.org/abs/2505.15245)
- [Liu et al. 2025. MemoTime](https://arxiv.org/abs/2510.13614)
- [Chen et al. 2025. TAG-EQA](https://arxiv.org/abs/2510.01391)

### 多模态图 / Scene Graph
- [Zemskova et al. 2024. 3DGraphLLM](https://arxiv.org/abs/2412.18450)
- [Patil et al. 2025. ZING-3D](https://arxiv.org/abs/2510.21069)

### 代码图 / AST / 仓库依赖图
- [Tan et al. 2025. CGM](https://arxiv.org/abs/2505.16901)
- [Liu et al. 2026. GraphSkill](https://arxiv.org/abs/2603.06620)

### 因果干预 / Causal Probing（开放问题 1）
- [Tucker et al. 2025. Causal Interventions on Filler-Gap Constructions](https://arxiv.org/abs/2505.16002)
- [Lin et al. 2025. CAT (Causal Attention Tuning)](https://arxiv.org/abs/2509.01535)

### 图作世界模型 / Alignment 工具（开放问题 2）
- [Mahajan et al. 2024. Causal World Models meet LLMs](https://arxiv.org/abs/2410.19923)
