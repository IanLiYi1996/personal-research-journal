# Two 2026-05 Harness Surveys: ETCLOVG vs Code-as-Substrate

- **Date:** 2026-05-26
- **Tags:** #harness #survey #etclovg #code-as-substrate #agent-systems #comparative-reading

## Context

2026-05 同月里，两份独立的 agent harness engineering 综述同时挂出 arXiv / OpenReview：

| | A · ETCLOVG | B · Code-as-Substrate |
|---|---|---|
| 标题 | *Agent Harness Engineering: A Survey* | *Code as Agent Harness* |
| 作者 | Junjie Li, Xi Xiao, Yunbei Zhang, Chen Liu et al. (CMU / Yale / JHU / Tulane / UAB / OSU / Virginia Tech / Amazon · 19 人) | Xuying Ning, Katherine Tieu, Dongqi Fu et al. (UIUC + 多个工业界 · 42 人) |
| 平台 | TMLR under review | arXiv 2605.18747 |
| 提交 | 2026-05 | 2026-05-18 |
| 我的笔记 | [深度解读](./2026-05-25-agent-harness-engineering-survey.md) | （本文 + references.md） |

两份综述 6 个月前都还不存在；现在同月撞车。**这本身就是个数据点**——说明 harness engineering 已经从工业 buzzword 跨过了"学术追认"的门槛，进入"两个流派同时竞争 taxonomy" 的阶段。

这篇笔记不是另一份深度解读，是<u>对比阅读</u>——回答两个问题：

1. **两份综述的 framing 差在哪？** 哪个更适合什么读者？
2. **作为 harness 工程师 / 研究员，应该信哪一个？**（剧透：都信，互补）

---

## Main Content

### 1. 两个根本差异 · 切入点

#### A · ETCLOVG（Li et al.）：把 harness 当作<u>多层结构</u>

> "What does each layer in the system do?"

ETCLOVG = **E**xecution / **T**ool / **C**ontext / **L**ifecycle / **O**bservability / **V**erification / **G**overnance。

把 harness 切成 **4 层结构主干**（E·T·C·L）+ **3 层控制平面**（O·V·G），然后把 170+ 开源项目映射到这 7 层，统计每层的项目密度——L 层（lifecycle）47 个最热，C 层（context）9 个最稀缺。

切入视角是**工程师的视角**：我要造 harness，需要知道有哪些层、每层有什么 component。

#### B · Code-as-Substrate（Ning et al.）：把 harness 当作<u>统一媒介</u>

> "What is the operational substrate that lets an agent reason, act, model, and verify?"

中心论点：

> *"Code is no longer only a target output. It increasingly serves as an operational substrate for agent reasoning, acting, environment modeling, and execution-based verification."*

这句话翻译过来就是——**code 同时是 4 个东西**：

| 角色 | Agent 拿 code 做什么 |
|---|---|
| Reasoning | 看 code（含 spec / config / repo state）推断当前状态 |
| Action | 通过写 / 执行 code 改变环境 |
| Environment modeling | code 把外部世界（依赖 / 状态 / 接口）钉死 |
| Verification | 通过 execute → assert 自动判断"做对没" |

切入视角是**理论家的视角**：harness 看似多层，但底下的 substrate 是<u>同一个</u>——code。

#### 一个比喻

A 像化学：把分子分解成原子，告诉你 7 种原子怎么组合。
B 像物理：告诉你这些"原子" 其实都是同一种粒子（code）的不同状态。

两个都对。化学解释组成，物理解释本质。

---

### 2. 三层 vs 七层 · 哪个更精细？

| Li 七层（ETCLOVG） | Ning 三层（Code-as-Harness） |
|---|---|
| E · Execution | **Layer 1 · Harness Interface** —— code 怎么连接 reasoning / action / environment |
| T · Tool | （部分映射到 Interface） |
| C · Context | **Layer 2 · Harness Mechanisms** —— planning / memory / tool use / feedback control |
| L · Lifecycle | （部分映射到 Mechanisms / Scaling） |
| O · Observability | （映射到 Mechanisms 的 feedback 子层） |
| V · Verification | （映射到 Mechanisms 的 verification 子层） |
| G · Governance | **Layer 3 · Scaling** —— shared code artifacts 协调 multi-agent |

Ning 的三层是**纵向** framing（从 single agent 到 multi-agent 的成长曲线）；Li 的七层是**横向** framing（同一个 system 里的并列模块）。

**含义**：
- 想"找一个空白方向" → 用 Li，统计哪一层项目少（C/G/T 都 ⚠）
- 想"理解 system 演进路径" → 用 Ning，从 single → multi-agent 的 substrate 变化

---

### 3. 应用清单对比 · Ning 覆盖更广

Li 的项目映射偏 coding agent / DevOps（170+ 项目大多是 Cline / Aider / Claude Code 这一类）。

Ning 的应用清单覆盖：

- coding assistants ✅
- GUI / OS automation ✅
- **embodied agents** ← Li 没覆盖
- **scientific discovery** ← Li 没覆盖
- **personalization & recommendation** ← Li 没覆盖
- DevOps / enterprise workflows ✅

**对我的含义**：如果要把"AI 做科研" 论证为 harness 工程的<u>合法应用</u>，Ning 的引用比 Li 的更直接——他明确把 scientific discovery 列为 harness 应用之一。

---

### 4. 5 个开放问题 · 两份综述的研究路线图差异

#### Li 的 5 题（更偏 system / infra）：
1. 强化和扩展 execution environment（cost / portability / surrogate sim）
2. 长程 state 管理（recast as state estimation）
3. Trace-native 失败诊断
4. 标准化交接协议（cross-layer handoff contract）
5. 随模型能力提升的自适应简化（meta-harness / adaptive simplification）

#### Ning 的 5 题（更偏 evaluation / safety）：
1. **Evaluation beyond final-task-success** — final score 不够，要看 trace
2. **Verification under incomplete feedback** — 信号稀疏怎么办
3. **Regression-free harness improvement** — 改 harness 不能让以前能跑的崩
4. **Consistent shared state across agents** — multi-agent 共享 state
5. **Human oversight for safety-critical actions** — 安全关键场景的人类审批

#### 两份的<u>共同空白</u>：

把 Li #2（state estimation）+ Ning #3（regression-free）+ Ning #4（shared state）合起来 ⇒ **"Harness 的 reproducibility / regression / state"问题**。这是<u>下一篇 survey 大概率会专攻的方向</u>——也就是说，谁先在这块出系统化工作，谁就有下一个 taxonomy 命名权。

#### 两份不同的优先级：

- Li 偏 *infrastructure-economics*（cost / scale / contracts）
- Ning 偏 *epistemics*（怎么知道 harness 改完更好 / 安不安全）

---

### 5. 两份对"模型 vs harness" 的态度

#### Li · binding-constraint thesis（强论断）

> "**Benchmark variance may be driven as much by the execution harness as by the model itself.**"

并给出 3 个跨独立团队的硬数据（Bölük 10× / LangChain +13.7pp / Stanford-MIT 76.4%）。这是**有立场的综述**——明确告诉学术界"应该把投资从模型挪到 harness"。

#### Ning · 没有强论断，更像 *positioning paper*

Ning 没有声明"harness 比模型重要"——只是说 code 是 substrate，并把这个 substrate 怎么用做了个 taxonomy。是**中立综述**。

**含义**：
- Li 适合**说服怀疑论者**（"harness 是不是大厂自吹"）——直接给数据
- Ning 适合**做研究路线图**（已经入门 harness，要选课题）——给空间架构

---

### 6. 我应该怎么用这两份综述

按用途选：

| 我在做什么 | 用哪份 |
|---|---|
| 给师弟师妹分享 / 工业界讲 | **Li**（binding-constraint thesis 的 3 个数最有冲击） |
| 找研究方向 | **两份都看**（Li 的层级密度 + Ning 的 5 个开放问题） |
| 论证 SDD / spec-driven 的合法性 | **Ning**（"code as substrate" 给 SDD 学术背书） |
| 论证"多 agent 协作的工程地位" | **Ning**（layer 3 · scaling 直接讲这个） |
| 论证"harness 是工程<u>新</u>分支" | **Li**（明确把 harness 跟 prompt / context 的三阶段演化分开） |
| 讨论安全 / 评测 / 回归 | **Ning**（5 题里 3 题在这块） |
| 讨论 governance / observability 工程 | **Li**（7 层里 G + O 各自有独立讨论） |

---

### 7. 这两份综述的<u>命名</u>差异本身是个观察

- Li 把这个领域叫 **"Agent Harness Engineering"** —— 强调"工程"
- Ning 把这个领域叫 **"Code as Agent Harness"** —— 强调"媒介"

学科命名争夺通常是早期阶段的标志。**6-12 个月内大概率会出现第三个综述**，可能叫：

- "Agent Operating Systems"（强调 system 视角）
- "Cognitive Infrastructure"（强调认知层）
- "Agent Substrates"（融合 Ning 的 substrate + Li 的 layer）

如果是我赌，**最终命名权可能落在 Ning 这一派**——因为"code as substrate" 比"7 层分类" 更<u>本体论</u>，更适合作为长期共识。

但短期内（6-18 个月），Li 的 ETCLOVG 因为<u>更具操作性</u>（你可以直接说"我做 V 层"），会被工业界更广泛使用。

---

## 我的判断

### 不需要选边

两份综述<u>互补</u>，应该一起读：

- 先读 **Ning §1 + Li §1** 看两个 framing 的根本差异
- 再读 **Li §3-§10**（ETCLOVG 各层）建立"模块认知"
- 最后读 **Ning §3 (Mechanisms) + §4 (Scaling)** 把模块串成"演进路径"

#### 给我自己的 ai-research-harness 仓库

按 [v1.4.0 changelog](../sharing/CHANGELOG.md)：

- references.md 加 Ning 作为 "学术综述 B"
- 05-spec-driven.md 加 "code as substrate" 段
- 07-glossary.md 加 #21 · Code as Substrate

不写第二份深度解读——一份就够，避免对学术的 over-commitment。

---

## Open Questions

- 两份综述都 6 个月内挂出，**为什么没互引**？是同时进行不知情，还是有意避开？
- Ning 的 42 位作者里很多都是 Hanghang Tong / Jingrui He 实验室——这是 UIUC 系统派的"集团军"作业，意味着<u>这个流派会持续输出</u>。Li 的 19 位作者更分散——意味着第二代 ETCLOVG 可能就停了。<u>谁的 taxonomy 能赢，可能取决于哪个团队继续作业</u>。
- 两份都没回答的问题：**"harness 改进有没有 scaling law？"** Li 的 binding-constraint 给的是单点数据（10× / +13.7pp / 76.4%），不是连续曲线。如果有人能画出 "harness 投入 vs benchmark 提升" 的曲线，会是 next major contribution。
- Ning §4（scaling）讲 multi-agent shared code artifacts，但没回答**"shared state 怎么 version control？" / "concurrent agent 改同一份 code 怎么 conflict resolution？"** 这是 git 已经有 30 年答案的领域，但 multi-agent 场景是不是该重新设计 git？

---

## References

### 两份核心综述

- **Li et al. 2026** · *Agent Harness Engineering: A Survey* · TMLR under review · [openreview.net/forum?id=eONq7FdiHa](https://openreview.net/forum?id=eONq7FdiHa)
- **Ning et al. 2026** · *Code as Agent Harness* · arXiv 2605.18747 · [arxiv.org/abs/2605.18747](https://arxiv.org/abs/2605.18747)

### 我之前的相关笔记

- [Agent Harness Engineering: A Survey 深度解读](./2026-05-25-agent-harness-engineering-survey.md) — Li et al. 的 350 行深读
- [Harness 工程：让 AI Agent 从"能用"到"好用"的系统设计](./2026-04-14-harness-engineering-analysis.md) — 4 篇博客综合
- [AI-DLC 深度研究：把 SDLC 重写成 AI 可执行的 Markdown](./2026-05-07-aidlc-deep-dive.md) — AWS AIDLC 方法论
