# Strands Box：背景与相关产品边界比较

截至 2026-10-13 检索时点（Asia/Shanghai），仅依据公开一手资料；不是全天结束快照。未读取私有 GitLab、用户附件或复做项目适配审计；无子代理、安装、部署、付费测试或外部写入。下文“事实”指文档/源码证据，“推断”不代表实测集成。

## 1. 版本与定位

【事实】Box 是 Strands 的开源、与 agent harness 无关的本地沙盒预览：OS 限制直接访问，外部可信进程中的 Shell、Monty、网络网关和 MCP broker 共用 Dogwood 判权。[背景][box]

【事实】指定基线 `93c4749c377a64fb7a04f4d5a07afd880a290b38` 之后，公开 main 仅新增 `223e3959d539110eaacfacb6f973c03928dc39b0`（上海时间 10-13 04:07:37）：只删除 `crates/box/README.md` 三行旧 `rm/reset` 指引，无代码变化。[差异][delta] 发布仍是 v0.1.0，仅 Apple silicon/macOS 产物；源码虽有 Linux ARM64 namespace 后端，却受架构及 user namespace 实时探测门控，失败拒绝启动。[发布][release]、[源码门控][linux]、[探测][probe]【推断】Linux 只是实验性适配候选，不能把源码存在等同于发布支持，更不能套用 macOS 保证。

## 2. “关在哪里”与“允许做什么”不是一回事

【事实】OS 进程隔离是在同一宿主内核上限制系统调用，不是另开一台电脑；VM/microVM 有客体内核与虚拟化边界。microVM 是精简 VM，不是缩小的容器。

| 机制/产品 | 与 Box 的实质区别 |
| --- | --- |
| Docker / gVisor | Linux Docker 用 namespaces、cgroups 等隔离/限额，共享宿主内核；gVisor 增加用户态 Sentry 承接 Linux 系统调用，不是独立客体 VM。[Docker][docker]、[gVisor][gvisor] |
| Firecracker | KVM 驱动精简 VM，增加虚拟化边界；自身不是 SQL、支付或历史判权引擎。[机制][firecracker] |
| E2B Sandbox | 托管 Firecracker Linux 环境，可定制依赖、运行 pip 包。默认可出网，但已有 allow/deny 和沙盒外秘密注入；请求变换仍为公开 beta，不能说它“只有隔离”。秘密解析失败会省略对应 header，但继续转发，不能等同 Box 的凭证失败拒绝。[隔离][e2b-security]、[包][e2b-packages]、[网络][e2b-network]、[秘密][e2b-secret]、[Box 网关][gateway] |
| AgentCore | Code Interpreter 是 Python/JS/TS 执行工具，工具会话独占 microVM；Runtime 是整套 agent/tool 的托管宿主，microVM 模式支持 SSE/WebSocket、ARM64 容器，另有可共享实例的 Instances 模式，不能泛化其隔离保证。[解释器][aci]、[工具会话][aci-session]、[Runtime][runtime]、[协议][runtime-http] |

【事实】Box/macOS 用 Seatbelt；不提供租户隔离、CPU/进程/磁盘限额或抗内核漏洞保证。可信进程及解释器在 agent 沙盒外，仍属于必须信任的代码。[边界][containment]、[限制][limits]

【事实】授权回答“此身份能否对此资源做此动作”：Cedar/IAM 依据请求、实体与提供的上下文，不自动生成完整行为历史；Dogwood 兼容 Cedar，增加基于已记录事件的时间窗口/顺序规则。隔离负责约束可达范围，历史策略只覆盖被送到引擎的事件。[Cedar][cedar]、[IAM][iam]、[Box 策略][policy]

重要更新：AgentCore Policy 也已支持 Dogwood，作用于 Gateway 工具调用及 policy session，而非自动监控 Runtime 内每次文件/SQL 操作。不能把时间策略当成 Box 独有，也不能照搬事件语义：Box 的 `::request` 包含拒绝尝试，AWS 文档则将拒绝记为 `::error`。[AWS 策略][aws-policy]、[事件][aws-temporal]、[Box 事件][policy]

## 3. 两类 agent 的改造风险

【事实→推断】ADB client 默认连接本机 5037 server；设备操作不是 HTTP API。Box 默认网络只通网关/遥测端口，因此授予 `adb` 执行权不等于能连既有 server，更不等于约束 Android 内部动作。远程 E2B/AgentCore 也不能仅凭“有 Linux”推定能访问本地 USB/模拟器。候选是外部窄 ADB 工具，限制设备与命令；未验证。[ADB][adb]、[Box 网络][egress]

【事实→推断】Bun 官方 `fetch` 支持代理环境变量与流式响应，但这不证明具体 Bun agent 能在 Box 启动、信任其 CA 或保持流式行为。Box 默认 TLS 通路使用 HTTP/1.1，完整缓冲请求/响应，拒绝协议升级；默认响应上限 16 MiB、上游交换期限 300 秒。HTTP/1.1 本来能承载 SSE，问题是这里的缓冲实现：有限流可能延迟到结束，长期流可能超时；WebSocket、h2/gRPC 及原生数据库协议不能假定兼容。[Bun][bun]、[实现][http1]、[默认限额][gateway]

【事实→推断】Monty 是无 CPython 底座的 Python 子集，不能 import PyPI 包；pandas、数据库驱动等不能直接迁入。显式宿主 CPython 工具可成为候选，但其直接文件读取不再逐次进入 Dogwood。SQL 更宜经窄查询服务；这只是改造方向。[Python][monty]、[直接访问][containment]

【事实→推断】Box HTTP 判权主要看目的地、方法、路径、正文长度，不看行级数据或支付正文 schema；MCP 可检查声明的工具参数，但不自动理解 SQL AST、数据库行权限或付款合法性。允许目的地仍可能泄露可读数据。行级权限应在数据库/服务端落实；支付需显式金额、对象、审批及幂等约束，不能以“沙盒内”替代。[可见字段][gateway]、[MCP][mcp]、[限制][limits]

## 4. 凭证与历史的底线

【事实】Box 将真秘密留在可信进程、通过网关注入，但同一 box 的 agent、工具、本地 MCP 都能使用其凭证路由；隐藏值不等于禁止使用，不同凭证权限需分 box。`contain_egress=false` 绕过网关检查、注入与流量记录；宽泛解释器 `fs:read` 还可能读取宿主凭证文件。[凭证][credentials]、[限制][limits] AWS 执行角色则可通过 MMDS 被 VM 内代码取得，网络隔离不阻止它；入口 IAM 鉴权不能替代执行角色最小权限。[AWS 凭证][aws-credentials]

Box 的 `dogwood.redb` 历史可跨重启恢复，损坏或缺失按规则拒绝；但只限单 box，窗口最多 24 小时，修改历史规则可能重置状态。直接文件/工具内部访问不入历史，操作与结果记录非原子，也无防篡改证据。部署迁移要保全历史与记录身份，不能把对话记忆、VM 快照当成完整审计。[持久历史][policy]

【推断】Box 更像本地“可达范围＋语义闸门”，不是托管执行平台替代品。将其叠在 microVM 中只是候选架构；ARM64/user namespace、ADB 通路、CPython、流式及历史恢复均须另行验证，本研究未测试任何组合。

[box]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/README.md
[delta]: https://api.github.com/repos/strands-agents/box/compare/93c4749c377a64fb7a04f4d5a07afd880a290b38...223e3959d539110eaacfacb6f973c03928dc39b0
[release]: https://github.com/strands-agents/box/releases/tag/v0.1.0
[linux]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/containment/src/facade.rs
[probe]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/containment/src/backend/linux/namespace/probe.rs
[docker]: https://docs.docker.com/engine/security/
[gvisor]: https://gvisor.dev/docs/architecture_guide/intro/
[firecracker]: https://github.com/firecracker-microvm/firecracker#what-is-firecracker
[e2b-security]: https://e2b.dev/security
[e2b-packages]: https://docs.e2b.dev/quickstart/install-custom-packages
[e2b-network]: https://docs.e2b.dev/network/internet-access
[e2b-secret]: https://docs.e2b.dev/secrets/inject
[aci]: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html
[aci-session]: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-session-characteristics.html
[runtime]: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html
[runtime-http]: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html
[containment]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/design/containment.md
[limits]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/design/limitations.md
[cedar]: https://docs.cedarpolicy.com/auth/authorization.html
[iam]: https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic.html
[policy]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/design/policy.md
[aws-policy]: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-core-concepts.html
[aws-temporal]: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-temporal-authoring.html
[adb]: https://developer.android.com/tools/adb
[egress]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/design/egress.md
[bun]: https://bun.com/docs/runtime/networking/fetch
[http1]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/egress-gateway/src/intercept/mitm/http1.rs
[gateway]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/crates/egress-gateway/README.md
[monty]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/design/monty.md
[mcp]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/design/mcp.md
[credentials]: https://github.com/strands-agents/box/blob/223e3959d539110eaacfacb6f973c03928dc39b0/docs/design/credentials.md
[aws-credentials]: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/security-credentials-management.html
