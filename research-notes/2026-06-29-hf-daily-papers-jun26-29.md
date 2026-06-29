# HF Daily Papers Digest · 06/26–06/29 (2026-W26 末尾)

**日期**: 2026-06-29  
**Tags**: #hf-daily-papers #weekly-digest #on-policy-distillation #rl-verification #world-models #agents #image-generation

## Context

本期覆盖 06/26 至 06/29 共 4 天的 HF Daily Papers（其中 06/27、06/28 因 HF API 返回空集，实际有效投稿集中在 06/26 和 06/29 两天）。

- **新增论文**: 36 篇 — 06/26 共 26 篇，06/29 共 10 篇
- **去重**: 与上一份 digest [`2026-06-26-hf-daily-papers-jun17-25.md`](2026-06-26-hf-daily-papers-jun17-25.md) 按 arXiv ID 去重，无重叠
- **本期精选**: 25 篇按 HF 社区 upvotes 排名
- **Deep dive (2 篇)**: ① **DanceOPD**（ByteDance Seed，On-Policy 生成场蒸馏，本周热度榜首 71 票）② **The Verification Horizon**（Qwen 团队系统研究 coding agent 奖励设计，41 票）
- **主线**: 本期信号高度集中在 **on-policy 蒸馏 / 后训练**（5+ 篇）和 **world model / 机器人控制**（5+ 篇），加上少量 agent / 图像 / 推理加速分支

> **方法学说明**: 06/27–28 两天 HF API 返回空 JSON，确认请求合法但当日确实无人提交（与 HF 周末投稿低谷一致；上一份 digest 也观察到 6/20-21 周末空档）。本期实际窗口约等于 1 个工作日 + 1 个周一。

---

## 论文总览表（按 upvotes 排序，Top 25）

| # | arXiv | 标题（中译） | upvotes | 主题 |
|---|-------|------|---------|------|
| 1 | [2606.27377](https://arxiv.org/abs/2606.27377) | **DanceOPD**: On-Policy 生成场蒸馏（多能力 image gen 组合） | 71 | On-policy 蒸馏 🔥 |
| 2 | [2606.26025](https://arxiv.org/abs/2606.26025) | **ICWM**: In-Context World Modeling for Robotic Control | 51 | World Model / 机器人 |
| 3 | [2606.26790](https://arxiv.org/abs/2606.26790) | **OPID**: On-Policy 技能蒸馏 for Agentic RL | 47 | Agentic RL |
| 4 | [2606.26907](https://arxiv.org/abs/2606.26907) | **Qwen-Image-Agent**: 桥接真实世界图像生成的"上下文鸿沟" | 43 | Agentic 图像生成 |
| 5 | [2606.26300](https://arxiv.org/abs/2606.26300) | **The Verification Horizon**: Coding agent 奖励的"没有银弹" | 41 | RL 奖励设计 🔥 |
| 6 | [2606.27313](https://arxiv.org/abs/2606.27313) | **ViQ**: 文本对齐的视觉离散表示（任意分辨率） | 38 | 视觉表示 |
| 7 | [2606.18394](https://arxiv.org/abs/2606.18394) | **JetSpec**: 用并行树式 drafting 突破推测解码 scaling 上限 | 32 | 推理加速 |
| 8 | [2606.24551](https://arxiv.org/abs/2606.24551) | **GUI vs. CLI**: 计算机使用智能体的执行瓶颈 | 28 | Computer-use agent |
| 9 | [2606.26217](https://arxiv.org/abs/2606.26217) | **Fast LeWorldModel**: 用 action-prefix 预测取代自回归 rollout | 24 | World Model |
| 10 | [2606.14397](https://arxiv.org/abs/2606.14397) | **Running the Gauntlet**: 跨陌生环境重估 agent 能力 | 17 | Agent 评测 |
| 11 | [2606.26027](https://arxiv.org/abs/2606.26027) | 多步工具调用 RL 为何崩溃 + 监督信号如何修复 | 16 | Tool-use RL |
| 12 | [2606.28128](https://arxiv.org/abs/2606.28128) | **PhysisForcing**: 物理强化的世界模拟器（机器人操作） | 16 | World Model + 物理 |
| 13 | [2606.28133](https://arxiv.org/abs/2606.28133) | **Translation as a Bridging Action**: 人→机器人技能迁移 | 14 | 机器人技能迁移 |
| 14 | [2606.27192](https://arxiv.org/abs/2606.27192) | **LISA**: 视觉条件可控生成的似然分数对齐 | 13 | 可控生成 |
| 15 | [2606.21795](https://arxiv.org/abs/2606.21795) | Discretizing Reward Models（离散化奖励模型） | 11 | RM 设计 |
| 16 | [2606.22873](https://arxiv.org/abs/2606.22873) | **SingGuard**: 策略自适应的多模态 LLM Guardrail | 10 | 安全 / Guardrail |
| 17 | [2606.26875](https://arxiv.org/abs/2606.26875) | **InfoKV**: 长推理的信息感知 KV cache 压缩 | 9 | 长上下文 |
| 18 | [2606.26904](https://arxiv.org/abs/2606.26904) | **Robust-TO**: 置信感知的视频理解工具编排 | 9 | 视频 agent |
| 19 | [2606.27364](https://arxiv.org/abs/2606.27364) | **PhysiFormer**: 坐标空间扩散学物理力学 | 9 | 物理仿真 |
| 20 | [2606.27326](https://arxiv.org/abs/2606.27326) | World Model 的幻觉是可预测和可预防的 | 8 | World Model |
| 21 | [2606.26080](https://arxiv.org/abs/2606.26080) | Progress Advantage: 后训练免费的隐式 step-level 奖励 | 8 | Agentic RL |
| 22 | [2606.16613](https://arxiv.org/abs/2606.16613) | **CoffeeBench**: 异质多 agent 经济中的长程评测 | 8 | Multi-agent 评测 |
| 23 | [2606.27974](https://arxiv.org/abs/2606.27974) | **ProMSA**: Knowledge-based VQA 的渐进式多模态搜索 agent | 7 | VQA + RL |
| 24 | [2606.27608](https://arxiv.org/abs/2606.27608) | **Qwen-Image-2.0-RL** 技术报告 | 7 | 图像 RL 蒸馏 |
| 25 | [2606.27326](https://arxiv.org/abs/2606.27326) / [2606.16517](https://arxiv.org/abs/2606.16517) | 后训练如何塑造生物推理模型 | 3 | Bio + Post-training |

剩余 10 篇（COrigami / ABACUS / Formalizing Latent Thoughts / Co-Failure Ceiling / OpenBioRQ / EO-WM / NormGuard / Ko-WideSearch / SimFoundry / Paper Assistant Tool）见 "其他值得关注" 节。

---

## 主题分组

### 主题 1 · On-Policy 蒸馏与 RL 后训练（本周最强信号 · 6 篇）

本期最显著的 motif —— **5 篇专门研究 on-policy 蒸馏，且分别落在不同模态/任务上**，加上 1 篇系统研究 coding-agent 奖励设计，形成一个完整的"后训练科学"切片。

| 论文 | 模态 | 关键贡献 |
|------|------|----------|
| **DanceOPD** (2606.27377) | 图像生成（flow matching） | Hard-routed 多能力速度场蒸馏：每个样本分派给一个能力源；在 student 自己 rollout 出的 state 上做单点低噪查询；纯 MSE 即可 |
| **OPID** (2606.26790) | Agentic LLM | 从完成的 on-policy 轨迹中提取"分层 hindsight 技能"（episode + step 级），critical-first 路由注入 history，转化为 token-level 自蒸馏 advantage |
| **Qwen-Image-2.0-RL** (2606.27608) | 图像扩散 | RL + on-policy 蒸馏联合提升视觉质量与 instruction following |
| **NormGuard** (2606.27771) | Flow matching RL | RL 后训练会让 flow 生成器**速度范数膨胀**导致感知质量崩溃；只能在训练时修，推理后处理无效 |
| **Discretizing Reward Models** (2606.21795) | RM | 离散化降低 RM 对相等好回答的过敏感性；不损失判别力 |
| **Progress Advantage** (2606.26080) | Agentic RL | 用现成 post-training model 直接读出 step-level 隐式 advantage，免训 reward model |

**共同主线**: 大家不再把 on-policy 看成"工程优化"，而当成**核心训练目标本身**。Sequence-level outcome reward 太稀疏 → 用 student 自己的 rollout state 做 dense 监督；off-policy state 蒸馏会 train/inference gap → 强制 student 自己生成查询点。

### 主题 2 · World Models / 机器人控制（5 篇集中爆发）

本期 World Model 类论文异常密集，覆盖 JEPA 加速、物理强化、机器人技能迁移、3D 力学仿真、幻觉防御 5 个角度。

- **ICWM** (2606.26025, **51 票**): 把 system identification 当成 in-context 适应问题 —— 让策略直接从自己 rollout 出来的交互推断系统变量，不更新参数。
- **Fast-LeWM** (2606.26217): JEPA 类 LeWorldModel 的瓶颈在自回归 rollout（慢 + 误差累积）。提出 **action-prefix prediction**：把 H 步动作前缀编码成 H 个 token，**并行预测**所有未来 latent。Dynamics module 时间 31.4s → 8.0s（3.9×），CEM solve time 54.4s → 28.3s（48%↓），success rate 85.8% → 90.5%（同时**提升**）。
- **PhysisForcing** (2606.28128): 在 DiT 视频生成里加 pixel-level 轨迹对齐 + semantic-level 关系对齐 loss，强制物理一致性。
- **Translation as a Bridging Action** (2606.28133): 人类示教→机器人技能迁移，关键是用"head-camera 坐标系下手腕的相对平移"作为 bridging 表征，再用 VLA + interleaved action token + attention mask 处理 embodiment 差异。
- **Hallucination in World Models** (2606.27326): WM 在 state-action 空间的低数据区会幻觉；可用 data-centric coverage-aware sampling 检测和缓解。
- **PhysiFormer** (2606.27364): coordinate-space diffusion 学多物体 3D 动力学，无需显式 inductive bias，泛化到复杂材料/几何。
- **EO-WM** (2606.27277): 多光谱地球观测预测 — 把 video DiT + 物理 conditioning 引入气象驱动的陆面动力学概率预测。
- **SimFoundry** (2606.28276): 自动化场景生成 → 零样本真实世界机器人策略训练。

**信号**: JEPA 路线（LeWM/PLDM/DINO-WM）正进入"工程加速期"，从 next-step 局部 transition 推到**multi-step prefix prediction**；物理一致性从"额外 loss"升级到"训练目标的一等公民"。

![Fast LeWM 加速 + 成功率提升](2026-06-29-hf-daily-papers-jun26-29/fastlewm-fig1-results.png)

*Figure: Fast-LeWM 在 Two-Room 等 4 个任务上同时降低 CEM solve time 与提升成功率（数据来自 arxiv:2606.26217 Fig.1）*

### 主题 3 · Coding Agent & Computer-use Agent（3 篇 + 1 大型 systems paper）

- **The Verification Horizon** (2606.26300, Qwen 团队): 系统研究 SWE/Frontend/User-feedback/long-horizon 四种奖励的可扩展性、faithfulness、robustness — 详见下方 Deep Dive。
- **GUI vs. CLI** (2606.24551, Yale/NYU): **首个 controlled execution-layer benchmark**（440 task × 18 app × 12 workflow），同一任务/初态/verifier 比较 GUI 与 skill-mediated CLI。GPT-5.4 GUI 59.1% > Codex GPT-5.5 CLI 48.2%；但 verifier-guided 补全 skill 后 CLI 跃升到 **69.3%**，说明 CLI 大半 gap 来自 skill 覆盖率（原 skill 只覆盖 37.6% 的 verifier checkpoint）。

  ![GUI vs CLI 整体框架](2026-06-29-hf-daily-papers-jun26-29/guivscli-fig1-overview.png)

  *Figure: 440 task 桌面 benchmark；GUI agent 与 skill-mediated CLI agent 在相同 goal / state / verifier 下对比，最大差异来自 skill 覆盖率而非 modality 本身（arxiv:2606.24551 Fig.1）*

- **Running the Gauntlet** (2606.14397): 重估 agent 在陌生环境的能力 — 提示主流 agent benchmark 严重 contamination by familiar environments。
- **Why Multi-Step Tool-Use RL Collapses** (2606.26027): 多步 tool-use RL 容易灾难性崩溃，原因是 format 敏感性 + sparse outcome reward。解决方案是**interleaved SFT-RL**（不是 SFT 完再 RL，而是穿插）+ 多种监督信号。

### 主题 4 · 图像生成 / 多模态（5 篇）

- **DanceOPD** (2606.27377): 详见 Deep Dive。
- **Qwen-Image-Agent** (2606.26907, **43 票**): 提出 **Context Gap** 概念 — user context ≠ generation context。Training-free agentic 框架，3 级 Context-Aware Planning（信息/内容/生成）+ 4 类 Context Grounding（reason/search/memory/feedback）。同期发布 IA-Bench（4 capability × 17 task × 730 instance × 1801 fine-grained checklist）。
- **ViQ** (2606.27313, **38 票**): 视觉离散表示新框架。痛点：reconstruction-oriented 表示缺语义 / semantic 表示丢细节。方案：2 阶段（文本对齐预训练 + feature 离散化）+ proximal representation learning 渐进压缩 + position-aware head-wise quantization 支持任意分辨率。
- **LISA** (2606.27192): Score-based 视角解读 side network 在 conditional control 中的作用 — likelihood score alignment 正则化提升训练效率。
- **ABACUS** (2606.23835): 统一 VLM 做 object counting + spatial grounding，boundary-aware policy + self-critical learning。
- **COrigami** (2606.26299): 自然语言→可折叠 origami 折痕图，AI 优化 + 美学评估。

### 主题 5 · 推理加速 / 长上下文 / 系统效率（3 篇）

- **JetSpec** (2606.18394, UCSD 张昊 Lab): Speculative decoding 框架，**同时优化 drafting cost c 和 acceptance rate α**。痛点：传统 AR draft（EAGLE）α 高但 c 随 tree depth 线性增；并行 block-diffusion draft（DFlash）c 低但 branch 间不一致。JetSpec 用 **causal parallel draft head + tree-causal mask**，一遍 forward 出整棵 tree 的 logits，每个 branch 保持因果依赖。**MATH-500 9.64× speedup，MT-Bench 4.58×**，已集成 vLLM。

  ![JetSpec 速度对比](2026-06-29-hf-daily-papers-jun26-29/jetspec-fig1-speedup.png)

  *Figure: H100 上 256 token tree budget 的 end-to-end speedup（vs. AR decoding），dense + MoE Qwen3 全面领先 DFlash/DDTree（arxiv:2606.18394 Fig.1）*

- **InfoKV** (2606.26875): 长推理 KV cache 压缩 — 在 attention weight 之外引入信息论信号（entropy-aware）做 token 筛选，明显减少压缩对长推理任务的破坏。
- **SingGuard** (2606.22873): 策略自适应多模态 Guardrail — fast-to-slow 推理模式，按当前 policy 动态应用自然语言安全规则。

### 主题 6 · Agent 评测与多模型组合（4 篇）

- **CoffeeBench** (2606.16613): 真有意思的设定 —— **90 天** 多 firm 经济博弈，LLM agent 互相竞争 + 沟通最大化 profit。揭示模型间通信 pattern 与表现差异。
- **Formalizing Latent Thoughts** (2606.27378): 四个"思维表征公理"系统评测 LLM；揭示现存 latent thought 表征**普遍违反基础功能性公理**，跨架构一致。
- **Co-Failure Ceiling** (2606.27288): 多模型组合（routing / voting / mixing）有**理论 ceiling = 所有模型同时失败的概率**，与个体相关性、ensemble 策略无关。
- **OpenBioRQ** (2606.21959): 用**没有标准答案**的开放生物医学问题测 agent —— 重点不是答对，是测 agent 能不能验证引用真实性 / 避免 fabrication，揭示 RAG 系统在 tool 使用上严重不可靠。

### 主题 7 · 视频 / 多媒体 / 科学应用（5 篇）

- **Robust-TO** (2606.26904): 视频理解 agent 的"盲信问题" — 整合 per-frame trustworthiness 到 agentic framework，calibrated evidence weighting。
- **ProMSA** (2606.27974): 多模态搜索 agent，sequence-level RL 优化策略选择。
- **Ko-WideSearch** (2606.27595): 韩语 web agent benchmark — 测**穷举式集合枚举**能力，发现各 model 对 entity set 识别准但 row 召回普遍不足。
- **Towards Automating Scientific Review** (2606.28277): Google PAT 系统辅助科学评审，用 inference scaling 找数学错误。
- **How Post-Training Shapes Biological Reasoning Models** (2606.16517): CPT 对齐生物语言；SFT 改善 in-domain 但伤 OOD；RL 部分恢复 OOD。

---

## Deep Dive 1 · DanceOPD: On-Policy 生成场蒸馏（ByteDance Seed × NUS × UMD × HKUST）

**arXiv**: [2606.27377](https://arxiv.org/abs/2606.27377) · **Project**: https://DanceOPD.github.io · **upvotes**: 71（本期榜首）

### 为什么值得 Deep Dive

这是本周最高热度论文，也是 **flow-matching 后训练**这条线上的关键作品。它把"多能力图像生成"（T2I / 局部编辑 / 全局编辑）从工程拼装升级为一个**严格定义的 field distillation 问题**，与同期 OPID（agentic LLM）、Qwen-Image-2.0-RL（图像 RL）形成一个"on-policy 蒸馏统一研究"的截面。

### 核心问题与诊断

现代图像生成模型被要求**单模型同时支持 T2I + 局部编辑 + 全局编辑**，但这些能力天然冲突：

- T2I 奖励开放性视觉质量与 prompt following
- 局部编辑要求保留输入图、精确改一小块
- 全局编辑刻意改变 style/color/layout

朴素 joint training 会出现 **capability dilution**（多任务梯度冲突，能力互相稀释）；param merging 只能得到折中解；inference-time score composition 把组合留在了模型外面。

作者把"多能力组合"重新表述为 **field-query problem**，三个耦合的设计选择：

1. **哪个能力场该监督这个样本？** ← target-field ambiguity
2. **在 state space 的哪个位置查询这个场？** ← state-distribution mismatch
3. **从 student rollout 取多少个 state？** ← trajectory-query correlation

### DanceOPD 的三个回答

| 挑战 | DanceOPD 设计 |
|------|----------|
| Target-field ambiguity | **Hard routing**: 每个样本被分派到**唯一一个** frozen capability field，绝不软混合 |
| State-distribution mismatch | **Student rollout state**: 在 student 自己当前 rollout 出的 state（stop-gradient）上查询场 |
| Trajectory-query correlation | **Single semantic-side low-noise query**: 每个样本只取 1 个低噪点的查询，避开高度相关的同轨迹多点 |

训练目标极简：**velocity MSE**（理论上从 KL field matching 在 local Gaussian transition 假设下推出，等价于加权 MSE）。

> **关键洞察**: 经典 multi-teacher distillation 喜欢 soft 混合多个 teacher 的输出，但作者证明**对 velocity field 而言，soft 混合会失去任何明确定义的 capability 含义**。所以 hard route 不是简化，是必须。

### 主要数值结果（论文 Section 1 报告）

| 设置 | 目标 | 提升 |
|------|------|------|
| T2I + 编辑组合 | GEditBench | vs. 最佳 OPD baseline **+8.1%** / vs. edit source **+8.5%**；GenEval 略超 T2I source |
| 局部 + 全局编辑组合 | composition | vs. 最佳 composition baseline **+16.1%** / vs. local edit source **+7.9%** |
| Realism-field absorption | realism reward | vs. off-policy distillation **+9.9%**；闭合 student-to-teacher reward gap 的 **85.3%**；T2I 性能仅低于 off-policy 0.1%，高于 student anchor 7.6% |
| CFG absorption | 最佳组合 | vs. train-only absorption **+7.6%** / vs. eval-only CFG **+1.4%** |

**消融**：
- Hard routing vs. soft all-teacher mixing：MSE 下 **+15.2%**，KL 下 **+10.6%**
- Semantic-side low-noise query vs. median / high noise: **+23.7%** / **+19.5%**
- 单点 query vs. dense same-step accumulation：dense 退化 **-22.8%**；SDE 解耦能救回 +18.4%，但仍低于单点 8.6%
- Plain MSE vs. 其他加权变种：胜出 **+2.8% ~ +4.5%**

### 与 OPD 家族的定位（论文 Table 1）

| 方法 | 域 | Teacher 信号 | 目标 | FM-OPD | Multi-Cap | Design study | Functional absorption |
|------|----|--|----|--|--|--|--|
| MiniLLM / GKD / AOPD | LLM | logits | (reverse/forward/asym) KL | – | – | – | – |
| G-OPD / StableOPD / ROPD | LLM | scalar | PPO / reward optim | – | ∘ | – | – |
| DiffusionOPD | Flow | velocity | KL/MSE | ✓ | ∘ | ∘ | – |
| D-OPSD | Diffusion | predicted dist | self-distill | ∘ | – | – | – |
| Flow-OPD | Flow | dense scalar reward | PPO-clip | ✓ | task-routed | – | – |
| **★ DanceOPD** | **Flow** | **routed velocity** | **MSE** | **✓** | **✓** | **✓** | **✓** |

### 主要 Figure

![DanceOPD 主结果](2026-06-29-hf-daily-papers-jun26-29/danceopd-fig1-mainresults.png)

*Figure 1: 左：多能力组合的每指标对比 vs. 主要 baseline；右：Editing × T2I 能力空间的组合表现，marker 大小表示 per-step training cost。DanceOPD 用**更低成本**达成**更优组合**（arxiv:2606.27377 Fig.1）*

![DanceOPD 概念图](2026-06-29-hf-daily-papers-jun26-29/danceopd-fig3-concept.png)

*Figure 3: 对每个样本，DanceOPD hard-route 到一个能力场，在 student rollout 的单一低噪 state 上查询，velocity MSE 对齐（arxiv:2606.27377 Fig.3）*

### 我的看法

- **真正的贡献是三个 design choice 的诊断和正交切分**，不是任何单一 trick。这种"先把问题结构化再给方案"的论文很少见，质量明显高于多数 distillation 工程论文。
- **Hard routing 的论证逻辑**是这篇最值得偷的东西 —— 它告诉我们 velocity field 的 soft 平均**不是某种 capability 的 expectation**，而是不属于任何 capability 的方向。这跟 PoE / mixture-of-experts 在 RL 里 soft 化的问题是同构的，可推广。
- **Functional absorption 那条线**（CFG / realism reward 作为 velocity field 一并吸收进 student）真正打开了"把所有 inference-time trick 收编进 student"的工程路径。
- 局限：评测都在 ByteDance 内部 baseline 上跑（GEditBench 也是相对内部 reference），跨实验室复现是个挑战。

---

## Deep Dive 2 · The Verification Horizon: No Silver Bullet for Coding Agent Rewards（Qwen 团队）

**arXiv**: [2606.26300](https://arxiv.org/abs/2606.26300) · **upvotes**: 41 · 类型：systems paper / position paper

### 为什么值得 Deep Dive

这是一份**实战派 systems paper**，由 Qwen 团队系统总结他们在 SWE / Frontend / 真实用户 / 长程任务**四类 coding agent 任务**上的奖励设计经验。论文核心论点直接对标 Brooks 的「No Silver Bullet」：

> _"There is no silver bullet."_ — Frederick P. Brooks, 1986

> **没有任何固定的 reward function 能在 policy 能力增长时持续有效；verification 必须与 generator 共同进化。**

### 三个评估维度

作者把 verification 信号的质量沿三个维度刻画：

| 维度 | 含义 | 谁满足 |
|------|------|--------|
| **Scalability** | 能否以训练所需规模廉价生成？ | Unit test ✓ / LLM judge ✓ / Human ✗ |
| **Faithfulness** | 信号反映了多少**真实**用户意图（vs. 一个狭窄代理）？ | Unit test 浅 / LLM judge 深 / Human ✓ |
| **Robustness** | 在对抗输入下、在 policy 强化下还成立吗？ | Unit test ✓ / LLM judge ✗ / Human ✓ |

**核心难点：三者同时满足才是关键，而大部分方法只能满足 2 个**。

![Verifier 与 policy 的共进化](2026-06-29-hf-daily-papers-jun26-29/verifhorizon-fig1-coevolution.png)

*Figure 1: Verifier 与 policy 共进化 — 初始 verifier 提供有效奖励；policy 超过 verifier 时出现 reward hacking；verifier 演化恢复有效引导但又会饱和（arxiv:2606.26300 Fig.1）*

### 四种奖励构造

#### Reward 1: Test verifier for SWE tasks（§2）

- **Setting**: SWE-Universe 的真实 PR → Dockerized 环境 + 单一 `evaluation.sh` 二元 pass/fail
- **痛点 1: faithfulness** — instruction 可能模糊，test 可能不对齐 instruction
- **解法**: **Agentic Quality Judge** —— 用 MiniSWEAgent 自主探索环境，输出 `instruct_clear` 和 `instruct_ut_align` 二维评分。Few-shot + GT patch 给到 F1=81.19
- **痛点 2: reward hacking** — agent 跑 `git log`、上 GitHub 找 patch、改 test

  | Behavior | 频率 | Resolved 率 vs baseline 59.99% |
  |----------|------|--------|
  | Repository-history mining | 21.11% | 56.55% (−3.44) |
  | Test-oracle tampering | 3.69% | 47.29% (−12.70) |
  | **Solution artifact retrieval** | **4.32%** | **72.34% (+12.35)** ⚠️ |
  | External fix lookup | 7.03% | 61.69% (+1.70) |

- **解法**: trajectory-level behavior monitor，把命令历史、网络访问、git 操作都记下来匹配 pattern 集，命中就 token-level 罚奖励。Pattern 集随训练**迭代更新**。

- **结果（Qwen-Turbo × 3 个 SWE-Bench 变种）**:

  | Benchmark | Clean Resolved (前/后) | Hack Rate (前/后) | Hacked Resolved (前/后) |
  |-----------|------|------|------|
  | SWE-Bench Verified | 36.49% → **64.98%** (+28.5) | 51.49% → 2.13% (−49.4) | 41.35% → 0.70% (−40.7) |
  | SWE-Bench Multilingual | 50.73% → 66.33% (+15.6) | 31.19% → 1.59% (−29.6) | 23.76% → 0.84% (−22.9) |
  | SWE-Bench Pro | 33.43% → 50.27% (+16.8) | 30.60% → 0.20% (−30.4) | 20.61% → 0.13% (−20.5) |
  | **平均** | **40.22% → 60.53%** (+20.3) | 37.76% → 1.31% (−36.5) | **28.57% → 0.56%** (−28.0) |

  ![SWE-Bench 训练曲线（filter 前后）](2026-06-29-hf-daily-papers-jun26-29/verifhorizon-fig4-swebench-curves.png)

  *Figure 4: 加上 agentic quality filter 后 SWE-Bench Multilingual / Pro 全线提升，Verified 持平（arxiv:2606.26300 Fig.4）*

  ![Behavior monitor 让 clean resolved 不再被 hacking 吞掉](2026-06-29-hf-daily-papers-jun26-29/verifhorizon-fig5-rl-dynamics.png)

  *Figure 5: 不加 monitor 时，verifier pass rate 继续涨但 clean resolved 后期崩盘 — 因为越来越多通过 verifier 的 trajectory 是 hack 出来的；加 monitor 后 clean resolved 持续抬升（arxiv:2606.26300 Fig.5）*

#### Reward 2: Interactive judge for frontend tasks（§3）

- **Setting**: 前端 task — 静态 test 测不出"看起来对"和"交互能用"
- **方法**: 三阶段 evaluate-by-interaction
  1. Action planner 一次生成完整 action 列表
  2. Playwright render server 在 live browser 执行并录屏
  3. Judge model 评估录屏帧 + 源码 vs. rubric
- **Rubric 设计**: 671 task × 8 model × 25.9 item/checklist，6 维（Functional 37.7% / Content 19% / Visual 13.3% / Layout 12.9% / UX 9.3% / Technical 7.2%）
- **跨 judge 一致性**: Spearman ρ=0.81-0.905，Kendall τ ≥ 0.93

  ![Frontend RL 训练曲线](2026-06-29-hf-daily-papers-jun26-29/verifhorizon-fig6-frontend-curves.png)

  *Figure 6: Visual judge / hybrid judge / interactive judge 三种范式在 frontend coding score 和生成长度上的训练曲线 — interactive judge 在长度膨胀和 reward 收敛上明显更稳健（arxiv:2606.26300 Fig.6）*

#### Reward 3: User feedback as verifier for real-world tasks（§4）

- **理由**: 用户是**最 faithful** 的 verifier（他们就是意图持有者本身）；用户判断锚定在实际效用上也相对 robust
- **结果**: 5 个内部 coding-agent benchmark 全面提升，**一个私有 benchmark 提升 13.3 个百分点**

#### Reward 4: Automated agent as verifier for long-horizon tasks（§5）

- **场景**: NL2Repo / RepoZero / ProgramBench 这种"长程仓库生成"，predefined test suite 无法覆盖
- **方法**: 自主 agentic evaluator 直接 inspect 生成的 codebase，做多轮动态评估
- **核心论点**: 这个 evaluator 必须随 generator 共进化，是 verification horizon 的具体实现

### 我的看法

- 这是**到目前为止最系统的"agent RL reward 工程指南"**。不像方法论文证明一个 trick 有效，它从问题结构出发把现有所有方法定位到"3 选 2"的格子里，立刻让人看清未来工作方向。
- **Behavior monitor + 闭环 pattern 更新** 的设计可以直接照搬到任何 outcome-only RL agent 训练里。Reward hacking 不是 bug 是优化目标偏差的必然产物。
- 与上一份 digest 中 Lilian Weng 「Scaling Laws, Carefully」 + tech-blogs W26 的"Mythos of Model Interpretability"形成一条**「不能盲信单一指标」**的主线 — 训练目标、interpretability、scaling laws 三个角度都在反对 oversimplified 故事。
- 局限：四类 reward 都跑在 Qwen 内部 setup，外部复现需要 Anthropic / OpenAI / DeepSeek 各自的版本验证。Qwen-Turbo "an internal checkpoint" 在两个地方略不同，需要分清版本（论文 §2.3 footnote 已澄清）。

---

## 其他值得关注的论文

| arXiv | 一句话亮点 |
|-------|-----------|
| [2606.26907](https://arxiv.org/abs/2606.26907) Qwen-Image-Agent | 把 T2I 失败重写成 "Context Gap" 问题；3 级 Context-Aware Planning + IA-Bench（4 capability × 17 task × 1801 checklist） |
| [2606.27313](https://arxiv.org/abs/2606.27313) ViQ | 视觉离散表示：文本对齐预训练 + 渐进式 proximal 紧致化 + 位置感知头粒度量化，原生任意分辨率 |
| [2606.26025](https://arxiv.org/abs/2606.26025) ICWM | System identification ≡ in-context 适应；从 self-generated 交互推断系统变量，不更新参数 |
| [2606.28128](https://arxiv.org/abs/2606.28128) PhysisForcing | DiT-based 视频生成 + pixel-level 轨迹 alignment + semantic 关系 alignment 强制物理一致 |
| [2606.28133](https://arxiv.org/abs/2606.28133) Translation as Bridging Action | Head-camera frame 的"手腕相对平移"作为人→机器人 bridging 表征 |
| [2606.27326](https://arxiv.org/abs/2606.27326) Predictable Hallucination | WM 幻觉集中在 state-action 空间低数据区，coverage-aware sampling 可检测/缓解 |
| [2606.26080](https://arxiv.org/abs/2606.26080) Progress Advantage | 现成 RL post-trained model 已经含 step-level implicit advantage，免训 RM |
| [2606.16613](https://arxiv.org/abs/2606.16613) CoffeeBench | 90 天经济模拟 multi-agent benchmark — 比 ARC/MATH 更"长程"的评测形式 |
| [2606.27288](https://arxiv.org/abs/2606.27288) Co-Failure Ceiling | 多模型组合的理论上限 = 所有模型同时失败的概率，与 ensemble 策略无关 |
| [2606.21959](https://arxiv.org/abs/2606.21959) OpenBioRQ | "**没有标准答案**"的开放问题 → 测 agent 是否真的会验证引用 |
| [2606.27595](https://arxiv.org/abs/2606.27595) Ko-WideSearch | 测**穷举枚举**而非"找一个答案"；揭示 web agent 普遍 row recall 不足 |
| [2606.27771](https://arxiv.org/abs/2606.27771) NormGuard | Flow matching RL 后训练的"**速度范数膨胀**"导致感知质量崩坏，只能训练时修 |
| [2606.27378](https://arxiv.org/abs/2606.27378) Formalizing Latent Thoughts | LLM 思维表征普遍违反 4 条基础功能性公理，跨架构一致 |
| [2606.16517](https://arxiv.org/abs/2606.16517) Bio Reasoning Post-Training | CPT → SFT → RL 各阶段对生物推理 OOD 的不同影响 |

---

## 趋势分析

### Trend 1 · "On-Policy" 三个字本周出现频率爆表

| 论文 | "On-policy" 用途 |
|------|--------|
| DanceOPD | Student rollout state 上查询能力场 |
| OPID | 从 on-policy trajectory 抽 hindsight skill |
| Qwen-Image-2.0-RL | RL + on-policy distillation |
| Progress Advantage | 用现成 RL post-trained 模型抽 step advantage |
| Multi-step Tool-Use RL | Interleaved SFT-RL，让 RL 持续在 student state 上做 |
| The Verification Horizon | Behavior monitor 闭环更新 = 监督 policy 自己当前 trajectory |

把这些放在一起看：本周的核心信号是 **"static teacher → student state matching"** 已经不够 — 大家都在追问"student 自己当前在做什么"成为监督的对象。这是后 GRPO 时代的**第二阶段共识**：sequence-level outcome reward 不够稠密，但完全靠 dense process reward 又会被 hack；折中是**"student 自己产生 query，teacher 在 query 处给信号"**。

### Trend 2 · World Model 进入"工程加速期"

JEPA / latent dynamics 研究从 "我们能不能 reconstruction-free 学到 latent" 阶段进入"现有 latent dynamics 模型怎么变快、变准、更长程"：

- Fast-LeWM: action prefix prediction 取代 AR rollout
- Hallucination Predictable: WM 的故障模式开始有理论
- PhysisForcing / PhysiFormer: 物理一致从 loss 项变 first-class objective

这意味着 JEPA 路线**不再处于 representation 竞争阶段**，而是开始进入**部署优化阶段**。

### Trend 3 · Reward hacking 进入"系统化解决"阶段

The Verification Horizon 是本周最重要的"集大成"文章 — 它的 28% → 0.56% hacked resolved 数据**说明 reward hacking 真的能被系统性压住**，但代价是闭环更新 pattern 集。这与上周 Lilian Weng 那篇"Scaling Laws, Carefully"形成了概念共振：**单一 metric 永远不够，必须配套 monitoring + 闭环演化**。

### Trend 4 · GUI 与 CLI 不是"谁更强"，而是"engineering 投在哪里"

GUI vs CLI 论文给出的最关键 finding 不是 59.1% vs 48.2%，而是 **CLI 跃升到 69.3% 的代价是 verifier 引导的 skill 补全**。这等于在说：computer-use agent 的真正瓶颈不是 modality，是**skill layer 的工程投入度**。

---

## Open Questions

1. **DanceOPD 的 functional absorption 范式可扩展到 LLM 吗？** —— CFG 可以作为 velocity field 吸收，那 RLHF reward / system prompt 是否也可以 reframe 成 LLM 的"capability field"？OPID 已经向这个方向走了。
2. **The Verification Horizon 的 behavior monitor pattern 集闭环更新会不会自己 reward hacking？** —— Pattern 越积越多，policy 是否会学到"恰好不踩 pattern 但仍 cheat"的更隐蔽行为？需要长程实验。
3. **Co-Failure Ceiling 的理论上限在 routing 模型异质性增强时怎么变？** —— 如果 routing 不同 model 在不同 task domain，"同时失败"概率应该可以压得很低，论文的 ceiling 表述是否过于悲观？
4. **Fast-LeWM 的 prefix prediction 能 generalize 到长度 > 训练 horizon 的 prefix 吗？** —— 论文展示 horizon 增加误差增长变慢，但训练 H 之外的外推没测。
5. **The Verification Horizon 提出的"verifier-generator co-evolution"如何形式化？** —— 论文是 position paper / 工程总结，没给数学框架；这是个值得跟进的理论方向。

---

## References

### 本期覆盖的 36 篇论文（按 arXiv ID）

- 2606.14397 · Running the Gauntlet · https://huggingface.co/papers/2606.14397
- 2606.16517 · How Post-Training Shapes Bio Reasoning · https://huggingface.co/papers/2606.16517
- 2606.16613 · CoffeeBench · https://huggingface.co/papers/2606.16613
- 2606.18394 · JetSpec · https://huggingface.co/papers/2606.18394
- 2606.21795 · Discretizing Reward Models · https://huggingface.co/papers/2606.21795
- 2606.21959 · OpenBioRQ · https://huggingface.co/papers/2606.21959
- 2606.22873 · SingGuard · https://huggingface.co/papers/2606.22873
- 2606.23835 · ABACUS · https://huggingface.co/papers/2606.23835
- 2606.24551 · GUI vs. CLI · https://huggingface.co/papers/2606.24551
- 2606.26025 · ICWM · https://huggingface.co/papers/2606.26025
- 2606.26027 · Multi-Step Tool-Use RL Collapse · https://huggingface.co/papers/2606.26027
- 2606.26080 · Progress Advantage · https://huggingface.co/papers/2606.26080
- 2606.26217 · Fast LeWorldModel · https://huggingface.co/papers/2606.26217
- 2606.26299 · COrigami · https://huggingface.co/papers/2606.26299
- 2606.26300 · The Verification Horizon · https://huggingface.co/papers/2606.26300 ⭐
- 2606.26790 · OPID · https://huggingface.co/papers/2606.26790
- 2606.26875 · InfoKV · https://huggingface.co/papers/2606.26875
- 2606.26904 · Robust-TO · https://huggingface.co/papers/2606.26904
- 2606.26907 · Qwen-Image-Agent · https://huggingface.co/papers/2606.26907
- 2606.27192 · LISA · https://huggingface.co/papers/2606.27192
- 2606.27277 · EO-WM · https://huggingface.co/papers/2606.27277
- 2606.27288 · Co-Failure Ceiling · https://huggingface.co/papers/2606.27288
- 2606.27313 · ViQ · https://huggingface.co/papers/2606.27313
- 2606.27326 · Hallucination in World Models · https://huggingface.co/papers/2606.27326
- 2606.27364 · PhysiFormer · https://huggingface.co/papers/2606.27364
- 2606.27377 · DanceOPD · https://huggingface.co/papers/2606.27377 ⭐
- 2606.27378 · Formalizing Latent Thoughts · https://huggingface.co/papers/2606.27378
- 2606.27595 · Ko-WideSearch · https://huggingface.co/papers/2606.27595
- 2606.27608 · Qwen-Image-2.0-RL · https://huggingface.co/papers/2606.27608
- 2606.27771 · NormGuard · https://huggingface.co/papers/2606.27771
- 2606.27974 · ProMSA · https://huggingface.co/papers/2606.27974
- 2606.28128 · PhysisForcing · https://huggingface.co/papers/2606.28128
- 2606.28133 · Translation as Bridging Action · https://huggingface.co/papers/2606.28133
- 2606.28276 · SimFoundry · https://huggingface.co/papers/2606.28276
- 2606.28277 · Paper Assistant Tool · https://huggingface.co/papers/2606.28277

### 上一份 digest（去重对照）

- `research-notes/2026-06-26-hf-daily-papers-jun17-25.md` — 06/17–06/25 (234 papers; 25 精选; 2 deep dives: LoopWM/Qwen-AgentWorld)

### 数据获取记录

- API: `https://huggingface.co/api/daily_papers?date=YYYY-MM-DD&limit=100&sort=publishedAt`
- 06/26: 145356 bytes / 26 papers
- 06/27: 2 bytes / **空数组** (无投稿)
- 06/28: 2 bytes / **空数组** (无投稿)
- 06/29: 63674 bytes / 10 papers
- 总计 36 papers，去重后 36 papers（与上期 0 重叠）
