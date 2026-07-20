# Blog Deep Dive: Harness Engineering for Self-Improvement（Lilian Weng）

**日期**: 2026-07-08
**Tags**: #blog-deep-dive #harness-engineering #recursive-self-improvement #agent #context-engineering
**来源**: [Lilian Weng — Lil'Log](https://lilianweng.github.io/posts/2026-07-04-harness/)（发布于 2026-07-04，阅读时长 28 分钟）

## TL;DR

Lilian Weng 综述 35 篇论文，系统梳理"harness engineering"（智能体运行时系统工程）如何构成通往 recursive self-improvement（RSI，递归自我改进）的实际路径。核心论点：**近期 RSI 不太可能从模型直接重写自己的权重开始，而更可能从优化 harness（模型周围的运行时系统——上下文管理、工作流、权限控制、持久化状态）开始**。文章按"被优化的对象"梯度组织全文：instruction prompts → 结构化上下文 → workflow → harness 代码 → optimizer 代码，随着模型越来越强，可优化的目标也越来越通用化、代码化。

## 为什么值得深读

这篇文章直接命中本项目自身的运行处境——本项目日常使用的 Claude Code 本身就是一个"harness"（工具集 + 权限系统 + 持久化状态 + 子代理调度），而 CLAUDE.md、skills、workflow 脚本等都是"harness 工程"的具体实践。Weng 把这些零散的工程决策放进一个统一的理论框架里，值得作为长期参考。

## 历史脉络

- **RSI 的起源**：I. J. Good (1965) 提出"ultraintelligent machine"概念；Yudkowsky (2008) 提出"recursive self-improvement"这个具体术语——AI 用当前智能改进产生自身智能的认知机器。
- **现代 RSI 的两条路径**：(1) 模型直接重写自己的权重；(2) 更广义地，模型改进训练管线和部署系统，从而催生更强的后继模型。Weng 强调"部署系统"这一层与模型原始智能一样重要——Claude Code、Codex 等成功的 coding agent 产品证明了这一点。
- **Harness 的定义**：围绕基础模型的系统，负责编排执行、决定模型如何思考和规划、调用工具、感知和管理上下文、存储产物、评估结果。

## Harness Design Patterns（三个基础模式）

1. **Workflow Automation**：定义一个 plan → execute → observe/test → improve → execute 的目标导向循环，直到目标达成。Karpathy 的 autoresearch repo 是一个干净的示例。
2. **File System as Persistent Memory**：不应该把整个工作流和所有日志塞进 context，而应该把持久状态保存在文件里（实验日志、代码 diff、论文摘要、错误 trace、历史 rollout 轨迹）。读写文件系统（通常通过 bash）是 LLM 的基础技能。
3. **Sub-agent and Backend Jobs**：harness 可以生成多个子代理并行执行、监控后台任务。关键设计选择是让并行性显式化、可检视——子代理的输出如果只存在于短暂的 chat context 中会很快过时且隐藏；如果存成文件、日志、状态记录，模型就能在中断后恢复并对自己的执行历史进行推理。

**Coding Agent Harness 案例研究**：Claude Code、Codex、OpenCode、Cursor 类 agent 的核心接口已经趋同为一套稳定的工具分组——文件系统操作（glob/grep/read/write/edit/apply_patch）、shell 执行、IO（lsp/git 工具）、外部上下文（MCP/Skills/web_search）、artifacts、后端进程（CronCreate 等）、agent 委派（spawn_agent/resume_agent 等）。

## Harness Layer vs Core Intelligence

Weng 的预测：harness engineering 会朝"meta-methodology"方向演化——优化获取更好答案的机制，而不只是优化答案本身。harness 系统本身成为优化目标，启发式规则减少、通用机制增多。反过来，成熟的 harness 使能 auto-research 的模型自我改进循环，更聪明的模型则防止 harness 过度工程化，保持系统可持续。

类比 prompt engineering 的演化史：随着 instruction tuning 和模型推理能力提升，手工 prompt 技巧变得不那么核心，但**指定目标、约束、上下文、评估的需求并未消失**——只是被内化了。Weng 预计 harness 改进最终也会被内化进核心模型行为，但与外部上下文和工具的接口应该会保留。

## Context Engineering：三个层次的递进

- **ACE（Agentic Context Engineering，Zhang et al. 2025）**：把上下文当作演化的 playbook 而非不断变长的 prompt。Generator（生成任务轨迹）→ Reflector（从成功/失败轨迹中提炼洞察）→ Curator（增量更新结构化上下文）三组件循环。关键设计：curator 不重写整段 prompt，而是输出结构化的 (identifier, description) 条目集合，用确定性逻辑合并——防止"context collapse"和"brevity bias"。

- **MCE（Meta Context Engineering，Ye et al. 2026）**：把"如何管理上下文的机制"和"上下文里的内容"分离，在元优化层面进行技能演化，在基础层面进行上下文优化。定义 MCE skill $s$ 为一个上下文函数 $c_s=(\rho_s,F_s)$，双层优化：

$$\text{Inner: }c_s^*=\arg\max_{c_s}J_\text{train}(c_s;s)\quad\text{Outer: }s^*=\arg\max_{s\in\mathcal{S}}J_\text{val}(c_s^*)$$

  技能数据库追踪历史 $\mathcal{H}_{k-1}$，元层 agent 对先前技能做 agentic crossover 生成新技能。

  ![Meta Context Engineering 框架：元层技能演化搜索上下文管理机制，基层优化任务上下文](2026-07-08-blog-harness-engineering/mce.png)

- **Meta-Harness（Lee et al. 2026）**：再深一层——优化对象变成了**决定该存储、检索、呈现什么信息给模型的代码本身**。"Meta-"意味着这是一个用于优化 harness 的 harness。proposer 本身是一个 coding agent，最终输出是 Pareto frontier 上的 harness 候选集合。整个执行历史通过文件系统访问，coding agent 用 grep/cat 而不是把一切塞进单个 prompt。

  ![Meta-Harness 外层优化算法循环](2026-07-08-blog-harness-engineering/meta-harness-outer-loop.png)

## Self-Improving Harness：从上下文到代码

核心洞察：**代码是定义程序和系统的通用语言**。如果 LLM 能优化执行 agent 的代码，它就能访问比手写 prompt 大得多的设计空间。

- **STOP（Self-Taught Optimizer, Zelikman et al. 2023）**：早期递归 scaffolding 改进的例子。目标不是直接改进解 $s$，而是**改进改进器 $I$ 本身**。元效用 $\hat{u}(I)$ 定义为改进器在任务集合上的平均效用，递归更新 $I_t=I_{t-1}(\hat{u},I_{t-1};M)$。**关键警示**：STOP 用 GPT-4 时能提升下游表现，但用更弱的模型（GPT-3.5、Mixtral）时反而**性能下降**——递归结构本身不够，base model 必须强到足以真正改进机制。

- **Self-Harness（Zhang et al. 2026）**：三阶段循环——weakness mining（把失败聚类成 verifier-grounded 失败模式）→ harness proposal（基于挖掘出的失败模式提出有界的 harness 编辑）→ proposal validation（在 held-in/held-out 数据上做回归测试，只接受两边都无回归的候选）。在 MiniMax M2.5、Qwen3.5-35B-A3B、GLM-5 上测试于 Terminal-Bench-2，学到了针对不同 base model 弱点的模型特定 harness 指令。

  ![Self-Harness 的 weakness mining → harness proposal → validation 三阶段循环](2026-07-08-blog-harness-engineering/self-harness.png)

  Weng 提出的担忧：如果一个程序被允许编辑操作系统层，抽象边界就被打破了。可编辑的表面需要被恰当设计，权限控制和安全层需要活在这个循环之外——reward hacking 相关的所有挑战依然存在。

## Evolutionary Search 与 Darwin Gödel Machine

演化搜索适合搜索空间巨大或形状怪异、难以用梯度直接优化但容易评估解的场景。AlphaEvolve（Novikov et al. 2025）维护一个候选程序池，用冻结的 LLM 生成 diff 来改进；Darwin Gödel Machine（DGM, Zhang et al. 2025）则显式针对**可编辑的 harness 代码仓库**演化——用 Claude 3.5 Sonnet 作为 base LLM，DGM 发现的 agent 在 SWE-bench Verified 上从 20% 提升到 50%，在 Polyglot 上从 14.2% 到 30.7%，与手工构建的 agent 相当甚至更优。

这一整个方法家族在候选解可自动评估、fitness 容易量化时效果好（矩阵乘法、GPU kernel 优化、算法竞赛），在评估缓慢/模糊/主要靠启发式的领域会遇到困难。

## Future Challenges（论文点出的 7 个开放瓶颈）

1. **弱且模糊的评估器**——研究品味、新颖性、长期科学价值远比可验证奖励难以衡量
2. **上下文与记忆的生命周期**——Weng 认为 context engineering 应该成为智能本身的核心部分，而不只是停留在软件系统层
3. **负面结果的缺失**——文献偏向发表成功结果，LLM 可能不擅长决定何时该放弃一个假设
4. **多样性坍缩**——演化和 RL 循环倾向于利用已知高奖励模式
5. **Reward hacking**——评估器和权限控制应该活在演化 harness 的循环之外
6. **长期成功**——coding agent 能完成眼前任务，但难以保护由数百上千工程师共同维护的 repo 的长期健康
7. **人类的角色**——人类应该在栈中"向上移动"而非被移出循环，需要在正确的抽象层级和正确的时间点提供监督

## 我的反思

这篇文章最有价值的部分是它把"harness 工程"明确地从"写更好的 prompt"提升为"设计一套可被搜索、可被优化的系统"，并给出了一个清晰的复杂度梯度（prompt → context → workflow → harness code → optimizer code）。这个框架对理解本项目自身正在做的事情（CLAUDE.md 里积累的规则、skills 的沉淀、workflow 脚本的编排）很有启发——本质上都是在这个梯度上做手工版本的"harness 优化"，只是优化者是人类而不是模型自己。

STOP 论文的警示值得记住：**递归结构本身不构成自我改进的充分条件，base model 的能力是硬约束**。这对判断"什么时候该投入精力做自动化 harness 优化 vs. 手工调优"是一个有用的判断标准——如果驱动优化循环的模型本身不够强，自动化很可能不会带来净收益，甚至可能倒退。

另一个值得警惕的点是 Self-Harness / DGM 提出的"允许 agent 编辑自己的 harness"这个方向——Weng 自己也提到这打破了抽象边界的担忧。对于个人项目而言，这提示了一个实践原则：**任何允许 agent 自我修改运行环境的设计，都应该把评估器和权限控制显式地放在被修改的范围之外**，否则难以排除 reward hacking 类的失控。

## Open Questions

1. Harness 优化的"内化"过程（类比 prompt engineering 的历史）具体会以什么形式发生？是模型通过预训练/后训练直接学会更好的自我管理策略，还是会一直依赖外部 harness 层？
2. 对于评估模糊的任务（研究品味、代码可维护性），是否存在比"held-in/held-out 回归测试"更好的验证机制来支撑 harness 自我改进循环？
3. 本项目自身的 workflow 脚本设计是否也能应用文中的"三个基础模式"（workflow automation / file system as memory / sub-agent delegation）来系统化改进，而不是逐个任务临时设计？

## 涉及的 arXiv 论文（已入库）

本文引用的 20 篇 arXiv 论文已通过 `scripts/add_paper.py` 注册到文献库：2309.16797（Promptbreeder）、2401.10020（Self-Rewarding Language Models）、2410.07095（MLE-bench）、2502.10517（KernelBench）、2505.03335（Absolute Zero）、2505.22954（Darwin Gödel Machine）、2506.13131（AlphaEvolve）、2507.19457（GEPA）、2509.19349（ShinkaEvolve）、2511.16072（Early science acceleration experiments with GPT-5）、2511.23473（ThetaEvolve）、2601.03315（Why LLMs Aren't Scientists Yet）、2601.16175（Learning to Discover at Test Time）、2601.21557（Meta Context Engineering）、2603.19461（Hyperagents）、2603.28052（Meta-Harness）、2605.11328（Epistemic Uncertainty for Test-Time Discovery）、2605.26340（ScientistOne）、2605.27276（SIA）、2606.09498（Self-Harness）、2606.25996（Autodata）。

## 跟进阅读路径

- Lilian Weng 的前一篇文章 [Scaling Laws, Carefully](/research-notes/2026-06-26-lilian-weng-scaling-laws.md)（本项目已有笔记）—— 两篇文章共同勾勒出"预训练规模化收益递减 + 后训练/harness 层承接更多优化压力"的叙事
- Darwin Gödel Machine（2505.22954）与 Hyperagents（2603.19461）——如果要深入理解"harness 自我演化"的具体实现，这两篇是核心起点
- AlphaEvolve（2506.13131）——演化搜索在真实工程问题（矩阵乘法、GPU kernel）上的落地案例，与 harness 演化的方法论高度相关
