# Harness 工程：让 AI Agent 从"能用"到"好用"的系统设计

- **Date:** 2026-04-14
- **Tags:** #harness #agent #claude-code #codex #system-design #long-running-agents

## Context

2025-2026 年，Anthropic 和 OpenAI 不约而同地提出了同一个概念：**Harness Engineering**（脚手架工程）。两家公司分别从自己的产品实践（Claude Code 和 Codex）出发，发表了多篇技术博客，描述了一个共同的发现：**模型能力的上限不取决于模型本身，而取决于包裹在模型外部的系统设计。**

本文综合四篇核心文献和 Claude Code 源码分析，系统阐述 Harness 工程是什么、为什么重要、以及怎么做。

**信息来源：**
- [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — Justin Young (Anthropic), 2025-11-26
- [Harness Engineering: Leveraging Codex in an Agent-First World](https://openai.com/index/harness-engineering/) — Ryan Lopopolo (OpenAI), 2026-02-11
- [Unlocking the Codex Harness: How We Built the App Server](https://openai.com/index/unlocking-the-codex-harness/) — Celia Chen (OpenAI), 2026-02-04
- [Harness Design for Long-Running Application Development](https://www.anthropic.com/engineering/harness-design-long-running-apps) — Prithvi Rajasekaran (Anthropic), 2026-03-24
- Claude Code 源码（512K+ 行 TypeScript）

---

## 一、什么是 Harness？

### 1.1 定义

**Harness（脚手架/套具）** 是包裹在 AI 模型外部的程序化框架，负责管理 Agent 的运行环境、上下文、状态、工具和生命周期。它不改变模型本身的能力，而是通过**工程手段**让模型在复杂任务中发挥出更接近上限的表现。

一个形象的类比：如果 LLM 是一位能力出众的工程师，那么 Harness 就是这位工程师的工作环境——包括项目管理工具、代码仓库、CI/CD 流水线、Code Review 流程、团队协作协议等。**单独一个厉害的工程师和一个配备完善工程体系的团队，产出质量天差地别。**

OpenAI 的 Ryan Lopopolo 用一句话概括了这种新范式：

> **"Humans steer. Agents execute."**

### 1.2 为什么不能只靠模型本身？

**Anthropic 视角——模型的内在缺陷：**

Justin Young 观察到：

> "Agents still face challenges working across many context windows... each new session begins with no memory of what came before."

这就像一个工程师每天上班都失忆。Prithvi Rajasekaran 进一步识别了两个系统性问题：

- **Context Anxiety（上下文焦虑）：** 模型在接近上下文窗口限制时，会"begin wrapping up work prematurely"——提前收尾、草草了事。
- **Self-Evaluation Bias（自我评估偏差）：** "When asked to evaluate work they've produced, agents tend to respond by confidently praising the work — even when, to a human observer, the quality is obviously mediocre."

**OpenAI 视角——环境的根本欠缺：**

Ryan Lopopolo 从另一个角度发现了同样的问题：

> "Early progress was slower than we expected, not because Codex was incapable, but because the environment was underspecified."

Agent 不是不够聪明，而是**缺乏做事的条件**——没有工具、没有抽象层、没有内部结构。当任务失败时，"the fix was almost never 'try harder.'" 正确的反应是问：**"What capability is missing, and how do we make it both legible and enforceable for the agent?"**

两家公司殊途同归：**这些问题不是通过"更好的 prompt"就能解决的，它们需要系统层面的工程方案——这就是 Harness。**

### 1.3 两种 Harness 视角的统一

Anthropic 和 OpenAI 对 Harness 的侧重点不同，但本质互补：

| 维度 | Anthropic 侧重 | OpenAI 侧重 |
|------|----------------|-------------|
| **核心关注** | 单次任务的质量上限 | 持续开发的工程效率 |
| **典型问题** | Agent 如何在 6 小时内构建一个可用的 DAW？ | 如何用 Agent 团队维护百万行代码仓库？ |
| **关键方案** | 多 Agent 架构（Planner-Generator-Evaluator） | 环境可读性 + 仓库知识体系 + 架构约束 |
| **复杂度哲学** | 从简到繁，模型升级时做减法 | 前期投入重，用架构约束换长期速度 |
| **衡量标准** | 任务完成度、成本、质量评分 | 吞吐量（3.5 PR/人/天）、代码规模（100 万行） |

**统一理解：Harness 工程是关于"如何设计 Agent 的工作环境"的系统工程学科。** 它包含两个层面：
1. **任务层面（Anthropic）：** 如何让 Agent 高质量地完成一个复杂任务
2. **项目层面（OpenAI）：** 如何让 Agent 团队可持续地开发和维护软件

---

## 二、Harness 的演进

### 2.1 第一代：文件驱动的状态管理（Anthropic, 2025-11）

Justin Young 提出的最早方案是一个两阶段架构：

```
┌─────────────────────────────────────────────┐
│              Initializer Agent               │
│  ┌─────────┐  ┌──────────────┐  ┌────────┐ │
│  │ init.sh │  │ progress.txt │  │features│ │
│  │ (环境)   │  │  (进度追踪)   │  │ .json  │ │
│  └─────────┘  └──────────────┘  └────────┘ │
│                     ↓                        │
│              Coding Agent                    │
│  1. pwd → 确认工作目录                        │
│  2. 读 git log + progress.txt               │
│  3. 读 features.json → 选下一个任务           │
│  4. 运行 init.sh → 启动开发服务器              │
│  5. 实现一个功能 → 测试 → 提交                 │
└─────────────────────────────────────────────┘
```

**核心设计决策：**

| 决策 | 原因 |
|------|------|
| 用 JSON 而非 Markdown 存储 feature list | "The model is less likely to inappropriately change or overwrite JSON files" |
| 每次只做一个 feature | 防止 Agent 在上下文耗尽前试图"一把梭" |
| 用 Puppeteer 做端到端测试 | Agent "would fail to recognize that the feature didn't work end-to-end" |
| 强硬的 prompt 措辞 | "It is unacceptable to remove or edit tests" |

简单有效，但有局限：**没有独立的质量评估环节**，自我评估偏差问题依然存在。

### 2.2 第二代：Generator-Evaluator 多 Agent 架构（Anthropic, 2026-03）

Prithvi Rajasekaran 受 GAN 启发，引入了三 Agent 分离架构：

```
┌──────────┐     Spec      ┌──────────┐    Sprint     ┌──────────┐
│ Planner  │──────────────→│Generator │──────────────→│Evaluator │
│          │               │          │←──────────────│          │
│ 1-4句话  │               │ 按Sprint │   评分+反馈    │ Playwright│
│ → 完整Spec│               │ 逐步实现  │               │ 交互测试  │
└──────────┘               └──────────┘               └──────────┘
```

**实验结果——复古游戏制作器：**

| 方案 | 时长 | 成本 | 结果 |
|------|------|------|------|
| 单 Agent | 20 min | $9 | 界面能渲染但游戏不能玩，实体不响应输入 |
| 三 Agent Harness | 6 hr | $200 | 16 个功能，10 个 Sprint，含动画、音效、AI 生成器，游戏可玩 |

22 倍成本，18 倍时间，但产出从"不可用"变为"可交付"。**Harness 不是让简单任务变贵，而是让不可能的任务变为可能。**

随着 Opus 4.6 的发布，Prithvi 又做了关键的**减法**——移除 Sprint 分解，Evaluator 改为最终单次评审——因为更强的模型不再需要那么密集的外部检查点。

### 2.3 第三代：Agent-First 全栈工程体系（OpenAI, 2026-02）

OpenAI 团队做了一个更激进的实验：**从空仓库开始，零人工代码**，完全用 Codex 构建一个内部产品。

> 5 个月，~1500 个 PR，3 名工程师 → ~100 万行代码，3.5 PR/人/天

这不再是"用 Harness 辅助 Agent 完成单个任务"，而是**用 Harness 重新定义软件开发流程**。Ryan Lopopolo 的核心发现：

**工程师的角色彻底转变：**
- 不再写代码，而是设计环境、描述意图、构建反馈循环
- 当 Agent 失败时，不是"try harder"，而是问"什么能力缺失了"
- "Humans may review pull requests, but aren't required to. Over time, we've pushed almost all review effort towards being handled agent-to-agent."

**这一代 Harness 的关键创新不在多 Agent 架构，而在于将整个仓库变成了 Agent 的操作系统。**

---

## 三、Harness 核心设计模式

综合 Anthropic 和 OpenAI 的实践，可以提炼出以下跨公司共识的设计模式：

### 模式 1：仓库即 Agent 的世界（Repository as the Agent's World）

**OpenAI 的核心发现：**

> "From the agent's point of view, anything it can't access in-context while running effectively doesn't exist."

Slack 里的讨论、Google Docs 里的决策、人脑中的假设——**对 Agent 来说都不存在**。只有仓库里版本化的文件（代码、Markdown、Schema、执行计划）才是它的全部认知。

OpenAI 的做法是把一切都推进仓库：
- 架构对齐的 Slack 讨论？→ 写成 design doc 提交
- 团队达成的 convention？→ 写成 linter 规则提交
- 产品需求？→ 写成 product spec 提交

**Anthropic 的对应实践：**
- Claude Code 的 `.claude/scratchpad/` 目录作为跨 Agent 持久知识
- `MEMORY.md` + 记忆文件系统实现跨会话状态
- `CLAUDE.md` 作为项目级指令文件

**两家的共识：Agent 的世界边界就是仓库的边界。扩大 Agent 能力的方式不是给它更好的模型，而是给它更丰富的仓库。**

### 模式 2：渐进式披露（Progressive Disclosure）

**问题：** 给 Agent 太多上下文会适得其反。

OpenAI 的惨痛教训——"one big AGENTS.md" approach 失败了：
- 上下文是稀缺资源，巨大的指令文件挤占了任务、代码和文档
- "When everything is 'important,' nothing is"
- 单体文档立即腐烂，Agent 无法分辨什么还是对的

**解决方案：AGENTS.md 作为目录表，而不是百科全书。**

```
AGENTS.md           ← ~100 行，只是"地图"，指向更深的知识源
ARCHITECTURE.md     ← 顶层架构概览
docs/
├── design-docs/    ← 设计文档（含验证状态和核心信念）
├── exec-plans/     ← 执行计划（活跃/已完成/技术债务）
│   ├── active/
│   ├── completed/
│   └── tech-debt-tracker.md
├── product-specs/  ← 产品需求规格
├── references/     ← 外部技术参考（llms.txt 格式）
├── DESIGN.md
├── FRONTEND.md
├── QUALITY_SCORE.md
├── RELIABILITY.md
└── SECURITY.md
```

Agent 从小而稳定的入口开始，被教会"下一步去哪里看"，而不是一上来就被淹没。

**Claude Code 的对应实现：**
- `MEMORY.md` 限制 200 行，只做索引
- `findRelevantMemories()` 用 Sonnet 做语义排序，每次最多召回 5 个相关文件
- Skills 系统的三层加载（bundled → disk → MCP），按需加载而非全量注入

### 模式 3：机械化约束（Mechanical Enforcement）

**问题：** 文档和 convention 在高吞吐量下会迅速腐烂。

**OpenAI 的回答：把品味编码为代码。**

> "Documentation alone doesn't keep a fully agent-generated codebase coherent."

他们的做法是建立严格的分层架构，然后用 custom linter（当然也是 Codex 生成的）机械化地执行：

```
每个业务域内部的分层规则：
Types → Config → Repo → Service → Runtime → UI
↑ 只能向前依赖，不能反向
Cross-cutting concerns 只能通过 Providers 接口进入
```

> "This is the kind of architecture you usually postpone until you have hundreds of engineers. With coding agents, it's an early prerequisite: the constraints are what allows speed without decay."

**关键洞见：** 在人类工程中，这种严格约束会被嫌"太死板"。在 Agent 工程中，它们变成了**乘数**——"once encoded, they apply everywhere at once."

并且，自定义 linter 的报错信息本身就是对 Agent 的上下文注入——"Because the lints are custom, we write the error messages to inject remediation instructions into agent context."

**Anthropic 的对应实践：**
- Claude Code 的分层权限系统：Global → Project → Harness → Tool
- Hook 系统的 PreToolUse/PostToolUse 在工具执行前后注入控制逻辑
- Skills 的 frontmatter 声明工具范围限制（allowed_tools）

### 模式 4：应用可读性（Application Legibility）

**问题：** Agent 能写代码，但不能"看到"运行中的应用。

**OpenAI 的方案——把应用本身变得对 Agent 可读：**
- 每个 git worktree 可独立启动一个应用实例
- Chrome DevTools Protocol 接入 Agent runtime → Agent 可以操作 DOM、截图、导航
- 本地可观测性栈（logs + metrics + traces）对 Agent 暴露 → Agent 可以用 LogQL 查日志、用 PromQL 查指标

> "Prompts like 'ensure service startup completes in under 800ms' or 'no span in these four critical user journeys exceeds two seconds' become tractable."
>
> "We regularly see single Codex runs work on a single task for upwards of **six hours**."

**Anthropic 的对应实践：**
- Evaluator Agent 通过 Playwright MCP 像真实用户一样操作应用
- Claude Code 的 `webapp-testing` Skill 和 `agentcore` Browser Tools
- Justin Young 发现 Agent 在做了代码修改后会用 curl 或单元测试验证，但无法发现 E2E 集成问题 → Puppeteer 成为标配

### 模式 5：Generator-Evaluator 分离

**问题：** Agent 对自己的产出自我感觉良好。

**Anthropic 的经典方案：** 将"做事"和"评判"分给不同的 Agent。受 GAN 启发。

Prithvi 发现 "out of the box, Claude is a poor QA agent"——Agent 先识别出真正的问题，然后"talk itself into deciding they weren't a big deal"。

**OpenAI 的实践将此推向极致：**
- Agent 写代码 → Agent review 代码 → 循环直到所有 Agent reviewer 满意
- 这被称为"Ralph Wiggum Loop"——一个 Agent 驱动的反馈循环
- "Humans may review pull requests, but aren't required to."

**Claude Code 的实现：** Coordinator 模式中 Lead Agent 审查 Worker Agent 的产出，Worker 只能访问受限的工具子集。

### 模式 6：熵管理与垃圾回收（Entropy & Garbage Collection）

这是 OpenAI 独特的贡献——当 Agent 产出的代码量级达到百万行时，一个全新的问题出现了：

> "Codex replicates patterns that already exist in the repository—even uneven or suboptimal ones. Over time, this inevitably leads to drift."

最初，OpenAI 团队每周五花 20% 的时间清理 "AI slop"。这显然不可持续。

**解决方案——Golden Principles + 自动化垃圾回收：**
- 将品质原则编码进仓库（如：偏好共享工具包而非手写 helper；不用 YOLO 式探测，必须在边界做校验）
- 定期用后台 Codex 任务扫描偏差、更新质量评分、开重构 PR
- "Most of these can be reviewed in under a minute and automerged"

> "Technical debt is like a high-interest loan: it's almost always better to pay it down continuously in small increments than to let it compound."

这本质上是把**代码库的健康维护也交给了 Agent**——人类定义标准一次，Agent 持续在每一行代码上执行。

---

## 四、Harness 作为协议与基础设施

### 4.1 OpenAI Codex App Server：Harness 的协议层

Celia Chen 的文章揭示了 Harness 的另一个维度——**Harness 不仅是设计模式，还需要标准化的协议**。

Codex 的 Harness（Agent 循环 + 工具执行 + 状态管理）需要被多个客户端（Web、CLI、VS Code、Xcode、Desktop App）统一访问。这催生了 **App Server**：一个 JSON-RPC over stdio 的双向协议。

**三层会话原语（Conversation Primitives）：**

| 原语 | 说明 | 生命周期 |
|------|------|---------|
| **Item** | 原子级输入/输出单元（消息、工具调用、diff、审批请求） | `started` → `delta`(可选流式) → `completed` |
| **Turn** | 一次 Agent 工作单元，由用户输入触发 | 包含多个 Item 的有序序列 |
| **Thread** | 持久化的会话容器 | 可创建、恢复、分叉、归档 |

```
Client                              App Server
  │                                      │
  │─── initialize ──────────────────────→│
  │←── initialize response ─────────────│
  │                                      │
  │─── create thread + turn ───────────→│
  │←── thread/started ──────────────────│
  │←── turn/started ────────────────────│
  │←── item/started (user message) ─────│
  │←── item/started (tool call) ────────│
  │                                      │
  │←── approval request ────────────────│  ← Server 主动请求
  │─── allow/deny ──────────────────────→│  ← Client 回应
  │                                      │
  │←── item/started (agent message) ────│
  │←── item/delta (streaming) ──────────│
  │←── item/completed ──────────────────│
  │←── turn/completed ──────────────────│
```

**关键设计决策：**
- 协议是**双向的**——Server 可以主动请求 Client 审批（如工具执行权限）
- 用事件流而非请求-响应——一个用户请求可以触发多个 Item 的流式更新
- Thread 可跨 session 恢复——Web 标签页关闭后重连不丢状态

### 4.2 Claude Code：Harness 的全栈实现

Claude Code 的 512K+ 行代码本质上是一个极其完备的 Harness 实现，与 Codex 的方案形成有趣的对照：

| 架构层 | Codex App Server | Claude Code |
|--------|-----------------|-------------|
| **协议层** | JSON-RPC over stdio (JSONL) | React + Ink 终端 UI（直接集成） |
| **会话模型** | Item → Turn → Thread | Task 系统（6 种类型）+ Session State |
| **Agent 协调** | Ralph Wiggum Loop（Agent review Agent） | Coordinator 模式（Lead + Workers） |
| **工具扩展** | MCP server + Skills | Skills（bundled + disk + MCP 三层） |
| **权限控制** | 审批请求-响应协议 | Hook 系统（PreToolUse/PostToolUse） |
| **状态持久** | Thread 持久化 | Memory 系统（4 类型分类法） |
| **知识管理** | AGENTS.md + docs/ 目录 | CLAUDE.md + .claude/scratchpad/ |
| **客户端** | VS Code, Web, CLI, Xcode, Desktop | CLI, VS Code, JetBrains, Web |

**Claude Code 的 Harness 架构：**

```
┌─────────────────────────────────────────────────────────────────┐
│                      Claude Code Harness                        │
│                                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌────────────┐  │
│  │ Bootstrap │  │  Context  │  │  Skills   │  │   Memory   │  │
│  │ 状态初始化 │  │ 上下文管理 │  │ 行为扩展  │  │ 跨会话记忆 │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬──────┘  │
│        │              │              │              │          │
│  ┌─────┴──────────────┴──────────────┴──────────────┴─────┐   │
│  │                    Hook 系统                            │   │
│  │  UserPromptSubmit → PreToolUse → 执行 → PostToolUse    │   │
│  └─────┬──────────────────────────────────────────────────┘   │
│        │                                                       │
│  ┌─────┴─────────────────────────────────────────────────┐    │
│  │                  Coordinator                           │    │
│  │  Lead Agent ←→ Worker Agents (scoped tools)           │    │
│  │       ↓              ↓              ↓                  │    │
│  │  LocalAgent   InProcessTeammate  RemoteAgent          │    │
│  │  (子进程隔离)   (AsyncLocalStorage) (物理隔离)          │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │                  Task System                           │    │
│  │  Shell │ Agent │ Teammate │ Remote │ Dream │ Workflow │    │
│  │  pending → running → completed/failed/killed          │    │
│  └───────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、关键工程洞见

### 5.1 Harness 复杂度与模型能力的动态平衡

**Anthropic 的表述：**
> "Every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing."
>
> "The space of interesting harness combinations doesn't shrink as models improve. Instead, it moves."

Prithvi 的 V1→V2 演进完美说明了这一点：Opus 4.6 发布后 Sprint 分解变得多余。**模型升级时，Harness 应该做减法。**

**OpenAI 的表述：**
> "Our most difficult challenges now center on designing environments, feedback loops, and control systems."

OpenAI 发现工程师的工作不是消失了，而是**上移了一个抽象层**——从写代码变成设计环境。

**两家的共识：Harness 不会趋于消亡。旧脚手架拆除，新脚手架建起。模型越强，Harness 能释放的价值越大。**

### 5.2 "无聊"的技术选型反而是好的

OpenAI 的一个反直觉发现：

> "Technologies often described as 'boring' tend to be easier for agents to model due to composability, API stability, and representation in the training set."

他们甚至选择让 Codex 重新实现一些功能，而不是引入第三方库：

> "Rather than pulling in a generic p-limit-style package, we implemented our own map-with-concurrency helper: it's tightly integrated with our OpenTelemetry instrumentation, has 100% test coverage, and behaves exactly the way our runtime expects."

**原因：** Agent 需要能完全理解和推理它使用的依赖。不透明的上游行为（opaque upstream behavior）对 Agent 来说就是黑洞。

### 5.3 文件是 Agent 间通信的最佳媒介

四篇文章和 Claude Code 不约而同地选择了**文件系统**作为 Agent 间通信的主要方式：

- Justin Young: `features.json` + `claude-progress.txt`
- Prithvi: "One agent would write a file, another agent would read it"
- Ryan Lopopolo: 整个 `docs/` 目录体系 + execution plans
- Claude Code: `.claude/scratchpad/` + `MEMORY.md` + 记忆文件

文件的优势：**持久化**（不受 session 限制）、**可审计**（人类可检查）、**结构化**（JSON/Markdown 约束）、**去耦合**（Agent 不需要同时在线）。

### 5.4 吞吐量改变了一切

OpenAI 团队发现，当 Agent 的产出速度远超人类审查速度时，传统工程规范变得适得其反：

> "In a system where agent throughput far exceeds human attention, corrections are cheap, and waiting is expensive."

这意味着：
- 最小化阻塞式 merge gate
- PR 生命周期极短
- Test flake 用 follow-up 修复而非阻塞进度

> "This would be irresponsible in a low-throughput environment. Here, it's often the right tradeoff."

### 5.5 端到端测试不可替代

Anthropic 和 OpenAI 都独立得出了同样的结论：Agent 的自我验证不够，必须有外部的端到端验证。

- **Anthropic:** Puppeteer/Playwright 做浏览器自动化
- **OpenAI:** Chrome DevTools Protocol + 可观测性栈（LogQL/PromQL）

Justin Young: Agent "would fail to recognize that the feature didn't work end-to-end."
Prithvi: "Out of the box, Claude is a poor QA agent."

---

## 六、设计原则总结

| 原则 | 说明 | 来源 |
|------|------|------|
| **Humans steer, Agents execute** | 人类定义意图和标准，Agent 执行实现 | OpenAI |
| **仓库即世界** | Agent 看不到的就不存在，把一切推进仓库 | OpenAI |
| **渐进式披露** | AGENTS.md 是目录表不是百科全书 | OpenAI + Claude Code |
| **机械化约束** | 品味编码为 linter，convention 编码为 CI，一次定义永久执行 | OpenAI |
| **从简到繁** | "Find the simplest solution possible, and only increase complexity when needed" | Anthropic |
| **模型升级时做减法** | Harness 的每个组件都编码了"模型做不到"的假设，要定期质疑 | Anthropic |
| **生成与评估分离** | 自己做、自己评会导致偏差，必须用独立 Agent 评估 | Anthropic + OpenAI |
| **增量进展** | 每次只做一个 feature，避免上下文耗尽 | Anthropic |
| **应用可读性** | 让 Agent 能"看到"运行中的应用（DOM、日志、指标） | OpenAI + Anthropic |
| **熵管理** | 定期用 Agent 扫描和修复代码库退化，像垃圾回收一样持续运行 | OpenAI |
| **无聊技术优先** | 可组合、API 稳定、训练集中表示充分的技术对 Agent 更友好 | OpenAI |
| **文件优先通信** | Agent 间通过文件传递状态，持久、可审计、去耦合 | 全部 |

---

## Open Questions

- **Harness 工程是否会标准化？** Codex App Server 的 JSON-RPC 协议和 Claude Code 的 Hook/Skills 体系代表两种不同的标准化路径。是否会出现跨厂商的 Harness 协议？
- **"零人工代码"的边界在哪里？** OpenAI 做到了百万行代码零人工编写，但这是否只适用于特定类型的项目（内部工具、Web 应用）？对于底层系统、安全关键软件呢？
- **熵管理能否完全自动化？** OpenAI 的 Golden Principles + 自动化垃圾回收是一个优雅的方案，但当 Agent 生成的代码量进一步增长到千万行级别时，"人类定义标准"这个环节本身能否被 Agent 接管？
- **多 Agent 协作的组织结构上限是什么？** Anthropic 的 Planner-Generator-Evaluator 是三节点线性流，OpenAI 的 Ralph Wiggum Loop 是同级循环。更复杂的拓扑（层级化、P2P、专家委员会）是否有价值？
- **Harness 的成本效率曲线？** Anthropic 的 V1（$200/6h）→ V2（$125/4h）显示成本在下降，但 OpenAI 没有披露成本数据。在什么规模下，Harness 工程的投入回报最优？

## References

- Justin Young, [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents), Anthropic Engineering Blog, 2025-11-26
- Ryan Lopopolo, [Harness Engineering: Leveraging Codex in an Agent-First World](https://openai.com/index/harness-engineering/), OpenAI Engineering Blog, 2026-02-11
- Celia Chen, [Unlocking the Codex Harness: How We Built the App Server](https://openai.com/index/unlocking-the-codex-harness/), OpenAI Engineering Blog, 2026-02-04
- Prithvi Rajasekaran, [Harness Design for Long-Running Application Development](https://www.anthropic.com/engineering/harness-design-long-running-apps), Anthropic Engineering Blog, 2026-03-24
- Claude Code 源码，512K+ 行 TypeScript
- Anthropic, [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents), 2024
