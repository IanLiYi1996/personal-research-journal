# Strands Decider 与 Strands Box：如何帮助客户构建 Agent

阅读主文已于 2026-10-13 重写为[技术背景、相关产品及行业 assets 适配判断](2026-10-13-strands-decider-box-assets-fit.md)，结合 Mingfei 附件与当前 GitLab 项目核验。本文件保留为 10 月 12 日的详细证据版，其中版本状态应按新版补充理解。

- 日期：2026-10-12（Asia/Shanghai）
- 标签：Strands、Agent Building、Decision Model、Agent Sandbox、Dogwood、Agent Governance
- 调研范围：用户指定的 [Strands Box 用户指南](https://strandsagents.com/docs/user-guide/box/) 与 [Strands Decider 仓库](https://github.com/strands-labs/strands-decider)，并交叉核验官方发布文章、模型卡、源码和 AWS 文档。
- 方法：文档与源码调研；没有在客户环境部署，没有运行模型性能测试或沙箱逃逸测试。文中性能数字均为维护者公布的测量，不是本次独立复测。
- 阅读说明：`[D*]`、`[B*]`、`[G*]` 对应文末一手来源；“建议”“推断”“待验证”不是产品承诺。

## 1. 结论先行

这两个项目值得纳入客户 Agent Building 的技术选项，但它们解决的不是同一个问题，也不是两个完整的 Agent 平台。

**Decider 是概率性的语义决策组件；Box 是独立于 Agent 进程的动作约束组件。** 前者适合把高频、窄范围、可枚举的判断从通用生成模型中拆出来；后者适合让文件、网络、命令和工具访问受到进程外的限制。Decider 不是生成模型，Box 不是业务结果正确性的保证。[D1][D2][B1][B2]

对客户更有意义的组合是：

> 大模型负责复杂理解和规划；Decider 辅助选择、检查与评分；确定性规则负责授权；Box 限制可执行动作；业务系统仍校验身份、资源和审批。

这是本文的参考架构，不是官方已经提供的 Decider + Box 一体化产品。

| 客户问题 | Decider 的作用 | Box 的作用 | 仍需要客户或平台补齐 |
|---|---|---|---|
| 每个路由判断都调用昂贵模型 | 窄任务分类、工具/模型候选选择 | 通常不是主要优化点 | 业务数据集、回退机制、计算成本测量 |
| Agent 猜测参数后直接调用工具 | 检查参数是否有用户上下文支持 | 限制目标资源和实际动作 | 参数 schema、业务校验、人工确认 |
| Agent 可读日志，但不能改生产 | 可辅助判断请求意图 | 限制文件、HTTP、命令和 MCP 调用 | IAM、服务端授权、可信身份 |
| 先审批、再执行，或控制调用频率 | 不能把自然语言推断当成审批证据 | Dogwood 可根据已记录事件约束时序/频次 | 可信审批事件、完整拦截路径 |
| 敏感凭证不能暴露给 Agent | 本地推理可减少判定阶段的数据外送 | 网关可在转发时注入凭证 | 最小权限、目的地约束、密钥轮换 |
| 多租户托管 Agent | 不提供租户隔离 | 文档明确不提供租户隔离 | 托管环境的租户边界、配额与运维 |

表中能力依据 [D1–D5][B1–B4]；场景映射为本文建议，不是客户案例。

**优先级建议：**

1. 对有大量工单、任务路由或语义检查的客户，先做 Decider 影子评测。
2. 对正在开放 shell、文件、API 或 MCP 的客户，先做 Box 的隔离与策略试验。
3. 对退款、支付、生产变更等高后果动作，先建设可信授权与审批，不应先用模型替代这些规则。
4. 两者均不应仅凭发布演示直接承诺生产级 SLA、完整防泄漏或合规认证。

## 2. 项目定位与版本基线

### 2.1 Decider：不能只看 10 月 1 日的发布版本

官方 10 月 1 日发布文章介绍的是基于 Qwen3.5-2B 的 hobson-v19。调查时仓库已经在 10 月 9 日发布新命名的 Qwen3.5 v1 与四个 Gemma 4 v1 模型。[D1][D2][D3]

本报告对 Decider 源码及文档采用以下固定版本：

`0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534`，提交时间 `2026-10-09T23:33:05Z`。

本次查到的主要模型为：

- `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`
- `StrandsAgents/strands-decider-E2B-gemma4-v1-2610`
- `StrandsAgents/strands-decider-E4B-gemma4-v1-2610`
- `StrandsAgents/strands-decider-12B-gemma4-v1-2610`
- `StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610`

旧的 hobson-v19、hobson-v21 仍可获取。新版本采用新的命名方式，而非简单把旧研究编号当作对外产品版本。[D2][D3][D7]

### 2.2 Box：Developer Preview / 0.1.x 预发布软件

AWS 10 月 7 日公告称 Box 为 developer preview；用户指南进一步标注为 0.1.x pre-release，并提醒配置键、策略动作和 CLI 参数可能跨版本改变，需要固定 `BOX_VERSION`。[B1][B2]

沿用户指南核验到的官方仓库是 `strands-agents/box`。本报告对 Box 实现采用固定提交 `93c4749c377a64fb7a04f4d5a07afd880a290b38`，提交时间为上海时间 10 月 12 日 01:48:04；同时单独检查 `v0.1.0` release。开发分支中的实现与修复不自动等于发行包中的能力。[B8][B10]

**含义：**客户可以开展受控试点，但应把版本锁定、策略回归、升级检查和平台兼容验证纳入项目计划，而不是假定接口已经稳定。

## 3. Decider 深度分析

### 3.1 技术机制：把“生成答案”换成“对候选答案评分”

Decider 复用预训练 decoder 模型主体，移除语言建模输出头，加入 pointer head。模型通过一次前向传播比较问题答案位置与各候选选项位置的隐藏状态，输出候选项分布，不执行逐 token 生成循环。[D4]

它的标签在请求中定义，而不是固定在模型权重中。因此客户可增加队列、工具候选或等级描述，而不必为每次标签变化重新训练。文档也明确，候选顺序的影响并不完全为零，schema 的选项上限是 255。[D4]

| 原语 | 输出 | 适合的客户问题 |
|---|---|---|
| `noul` | `P(true)` | 参数是否来自用户陈述？当前是否需要澄清？ |
| `choice` | 选项、分布、confidence | 路由到哪个队列？在给定工具中选择哪个？ |
| `score` | 有序等级的期望值、分布、confidence | 紧急程度、答案充分性、反馈情绪等级 |

原语定义见 [D4][D5]；具体业务映射为建议。

这里最重要的区别是：**它输出结构化判断，不输出生成式解释。** 若客户需要写邮件、解释合同条款、生成 SQL 或完整规划，仍应由生成模型或其他组件完成。类型合法也不意味着判断正确。[D1][D4]

### 3.2 最值得客户尝试的六类用途

#### A. 工单与任务分流：优先试点

适合队列有限、标签明确、反馈闭环容易收集的业务，例如售后问题、产品咨询、内部服务台和事件分级。[D1][D2]

建议采用：

- 把 `other`、`need_clarification` 或 `human_review` 纳入候选，避免封闭候选集强迫选错。
- 对低确定性的结果回退到大模型或人工。
- 用客户自己的中文、简称、混合语言和历史类别变更样本测量。
- 不以总体 accuracy 作为唯一指标，分别看误路由、漏掉高优先级工单和自动处理覆盖率。

这些是工程建议；本次未找到足以直接证明客户中文场景表现的专项评测。

#### B. 模型路由：有潜力，但需计入自托管成本

可先判定任务是 FAQ、标准分类、复杂分析还是需要人工，选择不同的模型或处理路径。官方把模型路由列为用途，但没有给出适用于所有客户的降本比例。[D1][D2]

需要测量的是“含 Decider 的端到端成本”，不是仅比较一次模型 API 的价格。

#### C. 工具调用前的语义检查：减少“参数编造”

官方仓库的例子在 `InterventionHandler.before_tool_call` 中，把对话和拟议工具调用交给 Decider，检查参数是否有依据、调用是否过早。业务代码据此返回 `Guide` 或 `Proceed`，引导 Agent 先问用户城市，而不是猜一个城市查天气。[D6]

建议将此模式推广到：

- 客户是否真正给出了订单号或账户 ID？
- Agent 是否把搜索得到的信息误认为用户确认？
- 用户要求查询，Agent 是否准备执行更新？
- 在尚未获得关键输入前是否应澄清？

**边界：**这些仍是模型推断，不能代替订单归属校验、身份认证、审批签名或合法的资源访问权限。官方例子明确是实验演示，问题、阈值和策略均不推荐照搬生产。[D6]

#### D. RAG 和上下文处理：作为辅助筛选器

可以设计成“候选片段是否相关”“答案是否充分”等窄问题，再交由大模型生成回答。官方列出 guardrails、evaluations、memory 与 context management 等方向，但不代表现成提供了完整的 RAG、记忆或上下文产品。[D1]

**建议：**先选择可人工标注、错误代价可控的筛选步骤；不要让筛选器误删唯一证据，也不要把“认为有依据”当成引用真实性证明。

#### E. 低成本评测：适合大量预筛选，不适合唯一裁判

可用 `score` 或 `noul` 评估回答是否充分、是否满足简单 rubric。v1 的官方结果中，HelpSteer2 adequacy 为 0.709，generated adequacy 为 0.828，说明评测场景也存在明显误差。[D3]

建议使用“小模型筛选 + 抽样人工 / 强模型复审”，并为失败模式单独建集，不应将它作为高风险合规结论的唯一来源。

#### F. 行业微调：在通用模型验证后再投入

仓库提供训练和校准代码，可为客户的标签、业务语汇和判断边界做适配。公开文档给出了单 RTX 3090 与多 GPU 的训练路径。[D2][D8]

但截至本报告版本，**Qwen v1 新数据混合的构建器尚未全部进入 main，不能从 main 完整重建该训练数据**。已有配方可研究，并不等于最新模型训练全过程已能一键复现。[D3]

建议先评估不微调的效果，再判断增量收益是否覆盖标注、训练、校准、维护和许可证审查成本。

### 3.3 性能：有价值，但不能把宣传数字当作客户 SLA

官方在固定模型、窗口和硬件下给出如下结果：

| 模型 | JevBench 正确数 / 231 | Brier | ECE | Gemma GPU 权重内存，bf16 |
|---|---:|---:|---:|---:|
| Qwen3.5 2B v1 | 180 / 231，77.92% | 0.280 | 0.072 | 此表不估算 |
| Gemma 4 E2B v1 | 180 / 231，77.92% | 0.313 | 0.053 | 约 9 GB |
| Gemma 4 E4B v1 | 187 / 231，80.95% | 0.243 | 0.037 | 约 15 GB |
| Gemma 4 12B v1 | 200 / 231，86.58% | 0.173 | 0.061 | 约 23 GB |
| Gemma 4 26B-A4B v1 | 207 / 231，89.61% | 0.162 | 0.051 | 约 51 GB |

正确数、Brier 和 ECE 来自 [D3]；百分比为本文换算；内存来自 [D5]，仅为权重，激活等开销另计。Brier 与 ECE 越低越好，但这里并非对客户数据的校准保证。

需要区分两种证据：

- Qwen v1 明确有通过 main 上服务代码进行的 L4 发布验证，结果为 180/231。
- Gemma 表中完整 JevBench 结果来自维护者的训练/评测记录；公开服务路径的 CPU 对齐检查覆盖每个模型的 50 项，而非完整 231 项。[D3]

**延迟也必须带版本和条件：**

- 发布文章中的约 115 ms 是旧模型路径在 RTX 3090 上的测量背景，不可直接当成 v1 在客户硬件上的 SLA。[D1]
- Gemma 文档在一张 H100 上给出中位数：E2B 102 ms、E4B 139 ms、12B 136 ms、26B-A4B 245 ms。[D5]
- CPU 可以运行，但同一文档的 Gemma CPU 测量已到数秒级，且采用特定线程与并行运行条件。[D5]
- 增加输入长度会增加计算；小任务的中位数不能代表长文档 p95 或峰值并发。

建议模型选择：先比较 Qwen 2B v1 与一个满足预算的 Gemma 模型；只有业务指标确实提升时才扩大尺寸，不能仅凭总榜分数决定。

### 3.4 多个判断共享同一 state，是比较独特的工程价值

Decider 可编码一次 state，在问题之间共享前缀状态，只分别计算较短问题后缀。每个问题独立评估，不能看到其他问题的文本。[D4][D5]

这适合一个工具调用同时做若干检查：

1. 参数是否有上下文依据？
2. 是否需要澄清？
3. 属于哪个风险类别？
4. 是否建议人工复核？

建议把这些问题放进同一个请求，避免重复编码对话。不过多个问题仍不是免费，受 suffix 长度、批大小、缓存开销和显存限制影响。

### 3.5 Confidence：能用于分流，不能充当安全证明

`choice` 的 confidence 由选项分布计算：

`confidence = (N × p_max − 1) / (N − 1)`

它不是“判断正确的保证”；`score` 又使用另一种基于分布标准差的度量，`noul` 直接返回 `P(true)`，不能在三者之间简单复制同一个阈值。[D4]

例如三选一的最高概率为 0.90 时，choice confidence 是 0.85；不能把 response 中的两个字段混用。

官方文档提到的 0.9 / 0.5 路由分档属于示例约定，并要求按动作后果和自身流量测量。[D4]

建议为每个业务判断绘制 risk–coverage 曲线：自动处理多少样本时，会放过多少错误？身份、支付、合规等高后果动作不能以模型高分为授权条件。

### 3.6 接入方式与生产缺口

官方提供 CLI 和 HTTP `POST /v1/systemone`，输入一个 state 和多种 typed questions。默认服务监听 `127.0.0.1`，没有认证；文档明确并发请求行为未经验证，应用于本地实验。[D5][D9]

请求中的 `model` 字段不意味着动态切换模型；示例客户端注明服务按实际加载的 checkpoint 回答。客户应检查 `/health`，并通过实例、版本和路由明确区分模型。[D10]

长输入默认会裁掉 state 的前部，为问题与选项保留空间。`--strict-window` 可对超窗请求返回 422。对上下文中的早期授权、账户信息和证据，静默截断尤其危险。[D5]

建议生产封装至少包括：

- 身份认证、TLS、私网访问、请求体和资源限制。
- 串行队列或经验证的进程级隔离，不能把“一进程一个 worker”误当成线程安全保证。
- 单 state 多问题的批处理和显存限制。
- 冷启动预热、超时、重试上限、回退到人工或更强模型。
- 对高风险路径在判定不可用时拒绝或暂停，避免 fail-open。
- 固定源码、依赖、模型 revision、base model revision 和校准参数。
- 对输入截断、设备数值偏差、中文、提示改写和候选顺序建立回归集。
- 不记录未经脱敏的完整客户对话。

这些是对 [D5][D9][D10] 所揭示缺口的工程建议，不是现成功能清单。

Gemma 当前需要 main 上代码，PyPI 包尚无其完整支持；E2B 的缓存后缀注意力路径曾有 CUDA 数值问题，文档要求使用修复后的 eager attention 路径。[D5]

### 3.7 许可证与数据：区分代码、模型、原始数据

代码仓库为 Apache-2.0，Qwen v1 模型许可证也声明 Apache-2.0；模型卡说明 v1 去掉了一批具有商业使用限制的数据行。[D2][D7]

但训练来源仍包括 ShareAlike、定制或未明确条款的数据；Gemma 系列与 Qwen v1 的训练混合也不相同。不能由“代码开源”推出“所有训练数据都能任意重新分发”。[D3][D7][D8]

具体而言，Gemma 模型卡仍列出 AG News 等训练来源，而仓库的数据清单记录 AG News 原始来源为 non-commercial。Qwen v1 则明确移除了旧混合中的一批商业用途受限行。这里存在值得单独审查的来源差异，不能把 Qwen v1 的数据改进外推到所有新模型；也不能仅凭某项数据条款，未经审查就自行断言模型权重的法律结论。[D3][D7][D8]

建议客户分别记录代码、模型、base model、训练数据和衍生数据的来源，并在重新训练或分发之前交由内部合规流程审查。本文仅记录来源声明，不作法律意见。

## 4. Box 深度分析

### 4.1 两层设计：可达范围与可执行动作

Box 把 containment 与 policy 分开：

- containment：限制进程自身能访问哪些文件、程序、网络与宿主机资源。
- policy：对通过 Box 受控路径的请求做允许或拒绝决策。[B1][B2][B3]

Agent 进程外的可信 Box 进程持有 Dogwood Local Engine。shell、Python、网络和 MCP 的拦截点共享事件历史，因此可以依据前序行为约束后续请求。[B1][B2]

**客户价值：**从“提示 Agent 不要做危险动作”升级成“危险动作没有被授权就不能沿受控路径执行”。但这种收益取决于配置、拦截覆盖和可信事件，不保证所有已允许的业务动作都正确。

### 4.2 四类受控路径及其边界

| 路径 | 控制能力 | 客户必须注意的边界 |
|---|---|---|
| Egress gateway | 目标 host/port、HTTP 方法和路径、凭证注入、远端 MCP | 已允许目的地仍可能成为数据外送通道；显式放宽 egress 会扩大边界 |
| Strands Shell | shell 命令、文件动作、启动主机工具 | 是受控 shell，不是任意宿主 shell 的透明替代 |
| Monty Python | Python 子集的文件操作和 `fetch()` | 不是完整 CPython，不支持任意 PyPI 包 |
| MCP broker | 本地/远端 MCP 请求、发现的具体工具动作 | 允许一个工具，不等于验证该工具内部的每一个副作用 |

能力与边界见 [B1–B5]。直接授予进程的文件访问属于 containment，并不自动形成 Dogwood 文件动作历史。[B1][B3]

### 4.3 最独特的价值：基于历史的动作治理

Dogwood 不只判断“现在可不可以调用工具”，还可描述某个动作是否已经发生、顺序是否正确、是否在时间窗口内，或累计频次/数量是否达到上限。[G1]

官方用两个例子解释这一价值：

- 只有最近 15 分钟内测试通过、且之后没有测试失败，才允许 git push。
- 允许 Agent 向事件频道更新进展，但不能超过规定时间窗口的发送频次。[G1][B1]

可映射到客户的流程包括“先取授权后导出”“先查询订单再发起处理”“限制 Agent 的通知量”。不过这些是参考设计，必须先确认业务字段确实可由拦截点获得。

事件类型也影响语义：`::request` 包括被拒绝的尝试；固定版本的历史契约把 `::response` 定义为动作已发生或可能发生，把 `::error` 定义为确定未发生。因此“曾经请求审批”或“曾经请求测试”不能当成“已批准”或“测试成功”；还需要可信响应中的明确结果。Box 当前还拒绝超过 24 小时的 temporal window，不能直接据此实现月度额度。[B4][B9]

**历史是按 Box 目录持久化，不是每次 run 清空。** 同一个 `<box_dir>/private/dogwood.redb` 会在重启时重开，以免重启释放预算；不同 Box 不共享这个历史。新增或修改的历史规则从空状态开始，未改变的规则保留状态。外部副作用与 outcome 记录也不是一个事务，操作完成后崩溃可能缺少 response。[B9]

因此 quota 要区分“限制尝试”与“限制完成”：request 计数可在串行 admission 下约束尝试，但拒绝重试也计入；response 计数只看已经结束的动作，多个在途请求可能一起通过。严格金额预算仍应由服务端事务或可信账本管理。[B9]

**不能直接推导出的能力：**

- 观察到 Agent 文本说“已审批”，不等于审批系统确实批准。
- 检查了一次测试通过，不等于随后 push 的代码未被修改。
- 根据 HTTP 方法/路径放行，不等于已经按 JSON body 中的金额、租户、行级数据做了业务授权。
- 同一 Box 可跨运行复用历史，但不自动等于跨 Box、跨机器、跨租户的全局业务配额。

这些区别需要应用侧的可信审批、资源/版本绑定和服务端事务控制。

Dogwood Local Engine 自身是判定库，不直接执行阻断；集成层必须保证拒绝时不执行，并且事件不能由不可信 Agent 伪造。[G1]

### 4.4 凭证代理：不把真实 secret 放进 Agent 环境

官方说明网关可使用 placeholder，在放行转发时替换或注入真实凭证；支持 bearer、自定义 header、HTTP Basic、query 参数及 AWS SigV4。[B1]

这可帮助客户降低“Agent 读取环境变量后泄漏 secret”的风险，但不能阻止它滥用一个已被授权的 API。

还有一个容易漏掉的边界：**凭证边界是整个 Box，而不是单独一个工具。** 用户指南明确，每个 Agent、tool 和 stdio MCP server 都可以使用该 Box 声明的凭证绑定，网关不能据请求区分具体来源进程。需要不同凭证权限的组件，应放进不同 Box，再在服务端施加资源权限。[B2][B3]

建议：

- 用最小权限 IAM / 服务账号，避免网关注入高权限通用凭证。
- 绑定有限目的地和必要动作。
- 按用户与租户建立授权关系，而不是让所有 Agent 共享一把广泛 token。
- 对支付、外发、删除等动作继续做服务端授权与审批。
- 验证凭证在 Agent 文件、环境、日志与返回值中均不可见。

凭证注入还有两个接入条件：仅在 TLS 请求上附加凭证；`aws://<profile>` 支持指定 profile 中的静态 keys，并不支持完整 AWS credential chain，SSO、role_arn、source_profile、web identity 和 credential_process 等会被拒绝。临时身份有 `credsd://` 路径，但不能直接声称自动继承 ECS / AgentCore execution role。[B11]

### 4.5 Python 子集是客户选型的实际分水岭

Monty 支持有限的标准库。官方明确指出，`subprocess`、`socket`、`urllib` 及所有 PyPI 包不能直接导入。[B5]

因此以下需求不应默认认为开箱即用：

- pandas / NumPy / SciPy 数据分析；
- boto3 或其他服务 SDK；
- Playwright 浏览器自动化；
- 任意项目的 Python 测试与 build；
- 需要 native extension 或安装依赖的代码执行。

建议将这些工作封装为明确授权的工具、MCP 或独立执行服务，或者另选完整运行时沙箱。使用宿主工具时，需要单独评审其文件和网络权限，不能为了兼容而把隔离边界整体放开。

### 4.6 审计能力：有基础，不等于完整合规记录

Box 可记录策略 decision 的动作、permit/deny、原因、命中规则、文件路径、网络目标等信息，并提供 telemetry 与 OTLP 接入。[B6]

但直接文件授权的访问不会自动变成策略历史；Agent 上报的 telemetry 与 Box 自己产生的记录也有不同来源。[B1][B3][B6]

更重要的是，官方安全模型明确指出：**decision records 是 best effort，不防篡改，可能丢失，没有签名/链式完整性保护，也不自动轮转；不应直接作为 audit trail。** telemetry 的 loopback 接口没有认证，同一账号下的其他进程可提交内容。这里适合调试与可观测，不等于监管级审计存储。[B3]

建议：

- 使用可关联的 request / run / trace ID，把 Agent 意图、Decider 判定、Box 策略结果和工具业务结果连起来。
- 为高风险动作建立服务端审计记录，不仅依赖 Agent trace。
- 对 telemetry 做脱敏、保留策略、权限与不可篡改存储设计。
- 测试拒绝是否真实阻止副作用，而不仅仅生成了一条 deny 日志。

关联也不是自动完成的：官方 Strands CLI telemetry 教程指出，CLI 未传递 trace context 时，每条 Box decision 有自己的 trace，Agent traces 与 decisions 只是并列出现；同文件不等于同一因果链。需要针对实际 harness 实现和验证上下文传递。[B12]

### 4.7 安全边界与成熟度

Box 的安全模型明确说明，它保护宿主机，**不保护已经授予给 workload 的 Box 内部内容**；数据可按已授予的读写权限被读取或修改，也可被发往策略允许的目的地。托管环境的租户隔离不由 Box 提供。[B3]

因此以下说法不成立：

- “使用 Box 就能全面阻止 prompt injection。”
- “Box 能保证敏感内容不能发给已经允许的大模型 API。”
- “授权一个二进制工具，工具所有内部文件动作都会逐个进入 Dogwood。”
- “Box 可替代 IAM、业务审批或云平台租户隔离。”

正确定位是：缩小 prompt injection 或错误规划能造成的动作范围；不是让 Agent 的语义行为天然可信。[B1][B3]

此外，客户上线评审至少需要覆盖：

- **云 metadata / SSRF：**网关没有默认目的地黑名单，官方要求显式禁止 metadata host / address；`contain_egress = false` 的工具不会经过这些网络策略。[B3][B4]
- **资源耗尽：**Box 不提供全局进程数量、CPU、总内存或磁盘配额；虽然解释器有局部时长/分配限制，仍必须由运行环境约束整个工作负载。[B3][B14]
- **宽泛 MCP permit：**per-tool action 是 `mcp:call` 的细化；没有匹配的 per-tool 规则时，通用 `mcp:call` 结果仍成立。不能靠添加一个“允许某工具”的 permit 来收窄已有宽泛 permit，应按实际 fallback / forbid 语义测试。[B4]
- **同账号宿主进程：**egress gateway 的 loopback 端口不按连接进程认证，外部同账号进程也可沿网关发请求；策略仍会判定，但来源归因与 credential 注入边界需考虑。[B3]
- **远程服务：**允许远端 MCP / HTTP 请求之后，Box 不能约束服务内部执行和数据处理。[B3]

### 4.8 平台支持：公开指南与开发分支须区分

截至调研时，用户指定的公开指南写明：Apple silicon 上 macOS 15+ supported，Linux not supported yet，Windows unsupported。[B2]

版本固定后的结果是：

| 层次 | 核验结果 |
|---|---|
| 官方公开支持 | macOS 15+、Apple silicon；README 也仍写本地 macOS preview。[B2][B8] |
| 已发布包 | `v0.1.0` 的二进制资产只有 `aarch64-apple-darwin`。[B10] |
| Linux 实现 | 固定 SHA 的 facade 只接受 `aarch64`，其他 Linux 架构明确返回 unsupported；包括 x86_64。[B13] |
| Linux 容器与云平台 | 尚未完成客户平台验证；官方 issue 有 namespace、`/proc` mount 与 Kata 容器权限的直接问题报告，不是生产资格认证。[B15] |

因此“存在 Linux backend”不等于“发行版支持 Linux”，更不等于客户现有 x86_64、AgentCore、ECS 或 Kubernetes 可直接用。

已核验 Linux backend 使用 namespaces、mount view、capability drop 与 seccomp，不是 KVM hypervisor。外层若选择 VM / microVM，其虚拟化条件属于外层平台；本次未测任何目标主机。[B13][B15]

建议从 Mac 开发者试点开始。对 Linux 云端，先核验版本、架构、user namespaces、proc mounts、libc、网络代理与容器权限；不要为了启动新组件而放宽宿主保护或直接要求客户使用 privileged pod。

### 4.9 网络兼容性：harness-agnostic 不等于零改造

当前网关为 HTTP/1.1、每连接一个请求，请求/响应体上限 16 MiB，拒绝 protocol upgrade，包括 WebSocket。它会处理代理和 TLS CA；客户 SDK 的 HTTP/2、gRPC、WebSocket、SSE 和长流式场景必须逐项验证，不能承诺透明支持。普通 HTTP 可以经过网关，要求 TLS 的是凭证绑定路径。[B11]

普通 `http:request` schema 看见 host、port、method、path、字节数等，但看不到 query、headers 或业务请求体值。`body_bytes` 不是金额，也不是 SQL、租户或 AWS action。因此“每天退款最多 100 美元”不能仅靠现有普通 REST 策略实现，需要可信、带金额语义的工具/MCP 与服务端授权。[B4][B11][B16]

原生 Agent SDK 的 `open()` 和 HTTP 请求应分开看：前者可以直接使用 OS grant，后者可能经过 egress；不能笼统说所有 SDK 操作都经过 Dogwood。官方 Strands SDK 例子刻意把三个工具的文件/命令访问转交给 `zsh -c`，正是为了走受控 Shell 路径。[B17]

## 5. 两者组合：建议的客户参考架构

```text
客户请求
  → 大模型理解、规划
  → 提议工具调用
  → Decider 做窄范围语义检查
      ├─ 信息不足 / 不确定：澄清、强模型复核或人工
      └─ 判定可继续
           → 应用校验身份、资源、schema、可信审批
           → Box containment + Dogwood 受控路径
                ├─ deny：不执行，并反馈可恢复原因
                └─ permit：执行受限工具 / API / MCP
                     → 服务端校验与审计
                     → 结果反馈、下一轮规划
```

这是本文建议，接入点以 [D6] 的 intervention 模式和 [B1–B4] 的执行边界为依据。**Decider 的高分不能绕过 Box 或业务授权；Box 放行也不能绕过服务端的租户与资源权限。**

### 5.1 不应假定的“自动集成”

本次查阅的资料没有证明 Box 会自动把 Decider 输出变成策略事实。若要组合：

1. 应用在 before-tool-call 处调用 Decider。
2. 应用对语义判定做风险分流。
3. 对高后果动作获取可信审批和业务校验。
4. 通过 Box 已知的受控路径执行，而不是另外开放绕过路径。
5. 将请求和结果按统一标识关联。

这是客户需要实现的集成工作，而不是配置一个开关就完成。

以下是对应 `POST /v1/systemone` 的最小请求结构，展示一次请求做多个窄判断的接入方式；中文问题的效果未验证，不是推荐的生产 rubric。[D5][D10]

```json
{
  "state": "用户只要求查询订单状态；Agent 拟调用 update_order，订单号为其自行猜测。",
  "questions": {
    "args_grounded": {
      "type": "noul",
      "instructions": "拟议调用的每个关键参数是否都来自用户明确提供的信息？",
      "criteria": {
        "true": "所有关键参数均能追溯至用户明确陈述",
        "false": "至少一个关键参数被猜测，或缺少明确来源"
      }
    },
    "intent": {
      "type": "choice",
      "instructions": "用户明确要求的动作属于哪一类？",
      "criteria": {
        "query": "只读取，不改变业务状态",
        "update": "明确要求改变业务状态",
        "unclear": "信息不足或意图不明确"
      }
    }
  }
}
```

应用读取返回值后，应先做澄清或回退，而不是把判断直接写成 `permit`。即使 Decider 认为用户要求更新，业务服务仍要校验订单归属、角色权限和审批。

不要让 Agent 自己写入 `approved=true`，然后让策略据此放行；审批事实应来自受信系统，并绑定操作、主体、资源和有效期。

### 5.2 与 AWS AgentCore 的分工

AgentCore Runtime 官方文档介绍了框架无关的托管和 microVM 会话隔离；其 Code Interpreter 提供完整的托管代码执行能力与预装运行时/库。[G2][G3]

建议按职责选择：

| 层 | 负责什么 | 不应被误当成什么 |
|---|---|---|
| Strands Harness / SDK | 模型调用、Agent loop、工具接口与 interventions | 宿主机强隔离 |
| Decider | 窄范围语义选择、判断和评分 | 身份或业务授权 |
| Box / Dogwood | 进程外可达范围、受控动作与历史规则 | 跨租户托管和业务数据正确性 |
| AgentCore Runtime 等托管平台 | 托管、会话/环境边界、运行生命周期 | 细粒度业务操作审批 |
| Code Interpreter / 独立工具服务 | 完整代码环境、明确的计算能力 | 任意业务动作均安全 |
| IAM 与业务服务端 | 主体、资源、租户、审批和事务规则 | 语义理解组件 |

其中 AWS 能力见 [G2][G3]，其他项目见各自前文；职责映射为建议。

AWS Box 发布文章把 AgentCore、ECS、Kubernetes 的部署列为未来目标。客户不能仅凭这段路线图，把 Box 在这些平台上的落地视为已验证集成。[B1]

若客户已有生产 AgentCore / Kubernetes：

- Decider 可作为独立推理服务或专门实例，由 Agent 调用；需验证目标设备、版本与访问控制。
- 对 Box 先验证宿主机机制、权限、网络拦截、工具兼容和容器边界。
- 不为使用新组件而撤销已有网络、IAM、会话或租户隔离。
- 若只需一个业务工具的时序治理，可评估直接集成 Dogwood Local Engine；但判定执行与事件完整性需自行实现。[G1]

AWS 文档 MCP 在本次会话中未检索到，因此 AWS 服务事实以公开官方文档交叉核验；未安装或修改任何 AWS 连接。

## 6. 优先客户场景

以下为本文的选型建议，不是已验证客户案例。

### 场景 A：客服 / 运营 Agent

- 痛点：大量重复路由与澄清判断，模型调用开销高；Agent 易猜测订单或账户参数。
- Decider：队列分类、缺失信息检查、判断是否应澄清。
- Box：限制查询与更新 API、CRM/MCP 工具、凭证访问和外发目的地。
- 仍需：订单归属、用户身份、退款资格和金额上限在服务端校验。
- 优先级：高；先只读与影子评测，写动作后置。

### 场景 B：只读运维 / On-call Agent

- 痛点：希望自主查日志、读取监控、总结故障，但不能改生产或泄漏凭证。
- Decider：问题分类、工具候选选择、是否需要更多诊断数据。
- Box：限制 shell/tool/network/MCP 路径，凭证代理、通知频次控制。
- 仍需：只读 IAM、不可变审计、生产网络边界和授权主体。
- 优先级：高；与 Box 官方事件排查示例较接近。[B1]

### 场景 C：内部 Coding Agent

- 痛点：需要编辑、执行命令和调用 Git，但风险随自主性增长。
- Decider：判断是否缺少要求、是否需要人工复核，作为辅助而非安全闸门。
- Box：限制工作目录、启动工具和网络；探索测试/发布顺序规则。
- 仍需：可信测试结果、测试与被提交代码的版本绑定、CI 权限和制品链。
- 优先级：中高；先测实际项目的 shell、编译器、Python 和 MCP 兼容性。

### 场景 D：合同 / 金融资料分析 Agent

- 痛点：长文档、多条规则、敏感数据与高错误代价。
- Decider：文件分类、初步筛选；不宜仅凭总体榜分数直接做最终判断。
- Box：数据与目的地访问边界；不保证信息不会沿已允许目的地外送。
- 仍需：长上下文处理、引用核验、强模型或人工复审、合规流程。
- 优先级：中；高后果结论先保持人工在环。

### 场景 E：完整 Python 数据分析 Agent

- 痛点：pandas、NumPy、SDK、数据库驱动、图表与依赖安装。
- Decider：任务分流或低风险评测，可独立使用。
- Box：默认 Monty 路径不能满足完整 PyPI 生态；需工具化或不同运行环境。[B5]
- 优先级：对 Box 的默认 Python 路径偏低；优先评估完整执行沙箱，如 Code Interpreter。[G3]

### 场景 F：多租户 SaaS Agent 平台

- 痛点：隔离、跨租户权限、规模化、统一策略与审计。
- Decider：共享推理服务可能有价值，需处理数据边界与并发。
- Box：可评估作为单工作负载内的一层，不能承担租户隔离。[B3]
- 仍需：租户身份、实例/会话边界、配额、持久化、策略控制面和审计存储。
- 优先级：从隔离试验开始，不推荐直接承担完整生产安全边界。

## 7. 成本与收益：如何证明“帮到客户”

### 7.1 Decider 的收益取决于被替换的判断

建议按如下模型核算：

```text
基线成本 = 通用模型判断费用 + 原流程处理费用
新流程成本 = Decider 分摊计算费用
           + 回退比例 × 回退判断费用
           + 人工复核费用
           + 新增运维与评测费用
```

这是核算框架，不是产品报价。

若 Decider 仅作为额外检查，未替换任何生成模型调用，它首先是质量/风险投入，不一定降本。若调用量小、冷启动多、GPU 长时间闲置，也可能没有成本优势。

### 7.2 Box 的收益更偏风险控制与可运维性

建议测量：

- 高风险测试动作实际被阻断的比例。
- 正常任务的误拒绝率与因拒绝导致的重试次数。
- 每个正常任务的策略开销与整体完成时间。
- 配置一个真实项目所需时间。
- 新增工具、版本升级和策略变更所需回归成本。
- 凭证是否真的不进入 Agent 可见域。
- telemetry 能否解释“为什么拒绝”并关联到业务结果。

不要给未量化的风险控制收益套用固定百分比，也不要把 deny 数量越多当成越安全。

## 8. 建议的四周 POC

这是执行建议，排期需按客户数据与安全审批调整。

### 第 1 周：建立业务基线和威胁模型

- 选一个窄场景：工单分流或只读事件排查。
- 建议收集 500–2,000 条经脱敏、人工标注的代表性样本；这是起步建议，不是统计充分性的保证。
- 按时间、客户或工单实体隔离训练/校准/测试，避免数据泄漏。
- 包括中文、英文、简称、缺失字段、冲突信息、提示注入和分布外样本。
- 定义“什么必须拦截”“什么可以自动”“什么必须人工”。
- 记录原系统准确率、失败模式、p50/p95/p99、成本及成功率。

### 第 2 周：Decider 影子评测

- 固定源码与模型 revision；至少测试 Qwen 2B v1。
- 与确定性规则、原生成模型和人工标签比较。
- 分别测准确率、危险动作误放行、误拒绝、risk–coverage 和回退比例。
- 测完整请求耗时、输入长度、问题数量、冷/热启动与目标并发。
- 对中文做独立分析，不能由英文榜分数推定。
- 不让影子判定影响真实业务写动作。

### 第 3 周：Box 受控实验

- 固定 Box release，采用真实客户项目的必要最小权限。
- 建立允许/拒绝的成对测试：允许正常查询；拒绝越界读、未经授权写、未知目的地和不允许的工具。
- 区分 Agent 直接 grant、受控 shell/Python、主机工具和 MCP。
- 验证凭证、文件、网络、symlink、子进程、拒绝后的实际副作用。
- 用实际依赖确认 Monty 和 Strands Shell 能否支持工作流。
- 对时间/频率策略测试重启、并行、边界时刻及失败响应。
- 只在测试资源上验证，不对生产做破坏性尝试。

### 第 4 周：组合灰度与 Go / No-go

- 在低风险、只读流程接入 Decider + 业务校验 + Box。
- 判定服务失败时按风险暂停或人工回退。
- 将 Decider、Box、工具服务端的日志按 ID 关联。
- 评审成本、质量、兼容性、安全测试和运维职责。
- 达标后扩大低风险流量；高后果写动作单独审批上线。

### POC 验收表

| 维度 | 必测指标 | 不可替代的判断 |
|---|---|---|
| Decider 质量 | 逐类 precision/recall、risk–coverage、中文和 OOD、校准 | 总体 accuracy 高不等于高风险误放行足够低 |
| Decider 性能 | 目标长度与并发下端到端 p50/p95/p99、冷启动、内存、费用 | 本地单请求中位数不是生产 SLA |
| Box 安全 | 越界动作阻断、凭证可见性、拦截覆盖、真实副作用 | deny 日志不等于没有执行 |
| Box 可用性 | 正常流程成功率、依赖兼容、误拒绝与重试 | 安全但做不了业务不能算 POC 成功 |
| 平台工程 | 身份、租户、配额、版本回滚、审计与故障处理 | Box 不承担整个托管平台职责 |

## 9. 客户访谈必须问的七个问题

1. Agent 主要做固定工具调用，还是允许任意 shell / Python？
2. 高频语义判断有哪些？现在的模型与人工成本是什么？
3. 客户数据主要是中文还是多语言？有没有可用标签与负例？
4. 哪些错误只是体验问题，哪些可能导致资金、数据或生产损失？
5. 工具是否已在服务端做身份、资源和租户校验？
6. 运行环境是什么？是否已有隔离和可观测体系？
7. 客户希望扩大自主性、降低费用，还是先满足治理要求？

不知道这些条件时，不应承诺某个组合一定降低成本或符合生产安全要求。

## 10. 待验证与调研限制

- 尚未独立测量性能、并发、中文准确率和校准。
- 尚未复测 Box 的 containment 或逃逸边界。
- 没有提供客户生产环境下的 SLA 或安全保证。
- Decider 的发布文章、main 源码、PyPI 包与模型 revision 存在时间差，接入必须固定版本。[D3][D5]
- Box 的已发布二进制、开发分支中的 Linux 实现和云部署倡议需要逐版本对照，不能混用后作承诺。[B1][B8][B10][B13]
- 本次未查到足以支持“Decider + Box 已有成熟客户规模化部署”的一手案例。
- 对业务审批、全局配额、跨会话历史和租户隔离，必须明确可信事件与平台责任。

## 11. 一手来源

### Decider

- **D1**：[官方发布文章，2026-10-01](https://strandsagents.com/blog/introducing-strands-decider/)。
- **D2**：[README，固定提交](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/README.md)。
- **D3**：[官方 evaluation/results.md，固定提交](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/evaluation/results.md)。
- **D4**：[架构与 confidence 定义，固定提交](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/docs/architecture.md)。
- **D5**：[推理、服务、硬件、并发与窗口限制，固定提交](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/docs/inference.md)。
- **D6**：[Strands tool-call intervention 源码，固定提交](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/examples/strands/tool_call_intervention.py)。
- **D7**：[Qwen v1 模型卡](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610)、[Qwen v1 LICENSE.md](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/blob/main/LICENSE.md)、[Gemma 12B LICENSE.md](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610/blob/main/LICENSE.md)。模型仓库 main 会变化，客户部署应进一步固定 revision。
- **D8**：[训练说明](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/training/README.md)、[数据来源及许可证清单](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/data/sources.md)。
- **D9**：[HTTP 服务源码，固定提交](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/src/strands_decider/server.py)。
- **D10**：[示例客户端，固定提交](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/examples/strands/_client.py)。
- **D11**：[JevBench 评测解释、版本和场景差异，固定提交](https://github.com/strands-labs/strands-decider/blob/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534/evaluation/jevbench.md)。

### Box

- **B1**：[AWS 官方发布文章，2026-10-07](https://aws.amazon.com/blogs/opensource/introducing-strands-box-ai-agent-sandboxes-powered-by-dogwood/)。
- **B2**：[用户指定的 Box 用户指南](https://strandsagents.com/docs/user-guide/box/)。
- **B3**：[Box 安全模型](https://strandsagents.com/docs/user-guide/box/security/)。
- **B4**：[策略动作参考](https://strandsagents.com/docs/user-guide/box/reference/policy-actions/)。
- **B5**：[Python / Monty 参考与限制](https://strandsagents.com/docs/user-guide/box/reference/python/)。
- **B6**：[策略决策与 telemetry](https://strandsagents.com/docs/user-guide/box/guides/record-telemetry/)。
- **B7**：[Box 整体设计背景](https://strandsagents.com/blog/strands-box-the-big-picture/)。
- **B8**：[Box README，固定提交](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/README.md)、[提交时间](https://github.com/strands-agents/box/commit/93c4749c377a64fb7a04f4d5a07afd880a290b38)。
- **B9**：[策略、事件、历史持久化与并发契约，固定提交](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/policy.md)。
- **B10**：[Box v0.1.0 发行版](https://github.com/strands-agents/box/releases/tag/v0.1.0)。
- **B11**：[网络目的地、凭证与协议限制](https://strandsagents.com/docs/user-guide/box/guides/network-and-credentials/)。
- **B12**：[Strands CLI telemetry 与 trace correlation 限制](https://strandsagents.com/docs/user-guide/box/guides/strands-cli-telemetry/)。
- **B13**：[平台选择 facade，固定提交](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/containment/src/facade.rs)、[Linux namespace 后端](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/containment/src/backend/linux/namespace/mod.rs)。
- **B14**：[实现限制与非目标，固定提交](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/limitations.md)、[Monty 局部限制](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/box/src/run/broker/python/mod.rs)。
- **B15**：[Linux user namespace probe](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/containment/src/backend/linux/namespace/probe.rs)、[容器直接问题报告 #26](https://github.com/strands-agents/box/issues/26)。issue 为外部使用者观察，不是本次独立复测。
- **B16**：[网关拦截接口与可见字段，固定提交](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/egress-gateway/README.md)、[策略网络 schema](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/policy/schema/actions.cedarschema)。
- **B17**：[Strands SDK 与 Box 的具体集成示例](https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/examples/strands-box/strands-sdk-agent/agent.py)。

### Dogwood 与 AWS 平台

- **G1**：[Dogwood Local Engine 官方发布文章，2026-09-30](https://aws.amazon.com/blogs/opensource/introducing-the-dogwood-local-engine-temporal-governance-for-agent-actions/)。
- **G2**：[AgentCore Runtime 官方文档](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html)。
- **G3**：[AgentCore Code Interpreter 官方文档](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html)。

## 12. 对客户的一句话定位

**用 Decider 承接高频语义判断、验证降本提速的空间，用 Box 限制 Agent 的实际动作；复杂推理、可信授权和生产隔离仍由各自适合的组件承担。**

下一步不应是先大规模微调或把现有平台替换掉，而是选一个只读、高频、易测量的客户流程，完成业务基线、影子评测、隔离验证和灰度验收。
