# 主流托管 Agent 沙箱：替代关系与治理边界

核查基准：2026-10-13，Asia/Shanghai。仅使用下列第一方公开文档／源码，按当日在线文档核查，不把滚动文档当作历史发布日期证明。使用 research、aws-agentic-ai 技能；无子代理，AWS 文档 MCP 不可调用，回退官方文档。未访问私有资产、外部评估报告或账户，未安装、部署、付费测试；不讨论本地编码代理沙箱，不推断安全认证。

## 1．E2B：直接替代通用远程执行，环境自由度高

公开运行时明确采用每沙箱一台 Firecracker microVM，外围还有 cgroup、网络 namespace、nftables 出口防火墙；不是仅把代码放进普通容器。模板由镜像和构建步骤生成，Code Interpreter 可扩展官方模板预装 pip／npm 包，Bun 有正式模板示例。因此，需真实 Linux、CPython 原生扩展、系统库或长期工具进程时，它是真正的执行层替代，不要求更换模型或整个代理框架。（E1、E2）

pause 默认保存文件与内存，恢复可保留变量和进程；也可选择只保存磁盘。暂停无限保留，但默认超时行为是 kill，必须主动配置 auto-pause。持续运行上限 Hobby 一小时、Pro 二十四小时；暂停恢复会重置连续运行计时，不能把它解释为总任务寿命。暂停会断开客户端连接；快照拥塞时的 auto-pause 还可能降级为仅保存磁盘。长流程必须设计重连和状态恢复，而非假设所有连接一起复活。（E3、E4）

出口默认开放，可动态替换 IP／CIDR／域名规则；私网和链路本地地址被平台强制阻断，不能直接连私网 ADB。公开 URL 默认无认证，须禁用公开流量并要求每沙箱 token，尤其解释器端口。域名过滤限 HTTP 80、TLS 443，自动放行指定 DNS，且共享 CDN 有目的地混淆风险。新增 HTTPS 转换 Beta、外部秘密注入和短期工作负载身份已超出“只有隔离”：每次请求读取当前秘密，但解析失败仍转发、只删除对应头，须由目标拒绝未认证访问。主机许可不等于 API 操作获批。（E5、E6、E8）

命令 stdout／stderr 支持回调流，仍需重连；平台限制同时运行数量和创建速率，Hobby 二十并发／每秒一次，Pro 基础一百并发／每秒五次，可扩额度。控制面身份、出口变更和可观测事件不自动组成“先审批后付款”等历史授权状态机；应另配可信工具代理与业务账本。（E1、E4、E7）

## 2．Daytona：持久工作区与交互工具强，但须先选沙箱类别

当前已经不能笼统称 Daytona 为容器方案：Container 类是 namespaces 与硬资源限制，沙箱内可为 root；Linux VM／Windows 类有独立内核和硬件虚拟化边界；GPU 类是独占 GPU 的隔离容器。公开文档没有据此证明所有类别采用同一种 hypervisor。镜像／Dockerfile 快照可定义运行时、系统包与项目环境，适合仓库、开发服务、交互工具；与 E2B 相比，迁移决策必须显式选类，不能从相同 SDK 推断相同安全边界。（D1、D2）

普通容器 stop／start 保留文件和已装包，但清空内存；VM 才有 pause／resume、热快照、fork 保留运行状态。默认容器闲置十五分钟停止，VM 六十分钟暂停；GPU 停止即删除，结果须写外部 volume。持久 Python interpreter context 与普通 stateless code_run 是不同接口，后者不会因为工作区持久就保留变量。PTY、独立 shell session、日志流适合外部命令，但恢复后哪些进程需重启要逐类设计。（D3、D4）

Tier 1／2 默认限制出网，Tier 3 及以上默认开放；另有每沙箱封网、CIDR／域名白名单及动态更新。沙箱间默认不通；preview／SSH 默认认证，可显式公开 preview。Secrets 现已在沙箱外替换 HTTPS 请求头，并清洗回显响应；然而省略 hosts 就允许向任意主机注入，write:sandboxes 可挂载组织内任何秘密，manage:secrets 可修改其目的地主机。它解决明文暴露，不自动解决凭据使用者的业务授权；body／query／被 Base64 改写的占位符也不适用。（D1、D5）

并发取决于组织 CPU／内存／磁盘池，不是一个统一沙箱数；创建速率按 tier 为每分钟三百至六百。停止容器仍占磁盘额度，VM 暂停不占，影响大量待恢复会话的容量规划。已有管理操作 audit logs，不能说无审计；但这不等同逐条 shell 语义授权，也未从上述文档找到内建业务时序策略引擎。（D6、D7）

## 3．Modal Sandboxes：计算与数据依赖替代，能力组合有硬约束

当前有两种运行时：gVisor 在用户态实现 Linux syscall 接口；VM 通过 KVM 提供独立 Linux 内核。不要再概括成“Modal 永远只有 gVisor”。自定义镜像、apt／pip、Volume、流式 exec 可承载完整 CPython 数据栈；GPU 只支持 gVisor，VM 更适合 FUSE、嵌套 cgroup 等完整 Linux 行为，嵌套虚拟化需相应计划和申请。故它是复杂数据／计算工具的真替代，但 GPU 与内核兼容性要求未必能在同一类别满足。（M1、M2、M3）

默认生命周期五分钟，最长二十四小时；控制端需保存沙箱／快照标识并安排重建。文件快照默认三十天，可改 TTL；内存快照 Alpha 只有七天、会终止原沙箱、关闭 TCP，且不能保存正在 exec 的命令或可靠恢复 exec 启动的后台进程，也不支持 GPU。不能将“快照”承诺为无限期、多工具进程的透明续跑。stdout／stderr 流在 exec 完成后仅可读取十分钟，须及时另存审计证据。（M1、M4、M5）

出网默认开放，入网默认不接收连接，且沙箱默认无权访问其他 Modal 资源；可封网、CIDR 白名单、TLS 域名白名单 Beta、认证 HTTP／WebSocket Connect Tokens。域名规则不解密正文，DNS 例外及 domain fronting 风险被官方明确列出。动态收紧网络 Alpha 能切断不再允许的现有连接，但须创建时预先启用对应规则类型。（M2）

秘密注入 Beta 在外部 HTTPS 代理替换头，而普通 Secret 环境变量对代码可读。关键代价是注入不能与网络 allowlist、block_network、内存快照组合；仅 HTTP/1.1，不支持 HTTP/2-only／gRPC，轮换后须更新运行沙箱的 policy 才取新值。CPU／内存可设置 request 与 burst 上限，规模受工作区容器／GPU 额度约束；不能套用 Function 的四千容器上限证明 Sandbox 容量。外部代理策略并非历史业务审批引擎。（M3、M6、M7）

## 4．AgentCore：托管执行、工具入口与凭据服务的组合

Runtime 托管的是代理／工具服务，不只是代码片段。serverless 模式每 session 专属 microVM，隔离 CPU、内存、文件系统，默认闲置十五分钟终止、最长八小时；HTTP 合约要求 ARM64，支持 SSE／WebSocket 及同会话命令执行。Bun 和 CPython 可随自定义镜像迁入，但须重建 ARM64 原生依赖并实现服务合约。V2 从初始化快照启动，并非用户会话内存续存；凭据、随机标识、当前时间及工具目录不能无条件冻结在启动快照中。（A1、A2、A3、A4）

不能继续说 Runtime 文件只能随 microVM 消失：Session Storage Preview 保留指定目录，闲置十四天过期且版本更新重置；也能通过 VPC 挂载 S3 Files／EFS，但共享文件系统会重新引入跨会话数据权限与并发问题。新 Instances 模式在客户账户内每 session 对应 EC2，同一实例的多个代理不互相隔离，必须互信，另有 EBS 持久卷。资源归属变化不是自动增强代理间隔离。（A5、A6）

Code Interpreter 是独立执行工具，每工具 session 专属 microVM，保留本会话变量／文件，默认十五分钟、最长八小时。已预装 pandas、NumPy、SciPy、DuckDB、PyArrow 等，提供 Python、Node／Deno 的 JS／TS、shell 与事件流；官方 runtime 列表没有 Bun，本文也未确认任意自定义镜像能力，不能将“自定义解释器资源”理解成任意执行环境。SANDBOX 模式仍可访问 S3；PUBLIC 可出网；VPC 的实际出口由安全组、路由、NAT／端点决定。执行角色的 AWS 权限必须收窄，不能假设微虚机隐藏已授予代码的凭据。（A7、A8、A9、A10）

规模也分层：Runtime 活跃会话默认在美东一／美西二五千、其他区域两千五百，新会话每秒二十五；Code Interpreter 默认一千活跃会话，每会话二 vCPU／八 GB。它更适合 AWS 服务化与预装数据分析；高度定制的工具／大内存计算应比较 Runtime Instances 或其他通用沙箱，而非硬塞进解释器。（A4）

组合时，Gateway 是外部工具的受控入口，Identity 管理目标凭据，Policy 决定入口处哪些调用获准。Gateway 可代持 API key／OAuth 凭据，或按支持的目标传递调用方身份；但共享 Gateway 角色的权限是所有已授权调用者可行使权限的上界，敏感度不同的工具应分入口。若代理直接调用 Identity 取钥 API，响应明确包含密钥值；这不是 E2B／Daytona 外部代理注入的等价保密边界。（A14、A15；组合建议）

只保留当前托管实现的治理边界：Policy 已支持时序授权、ENFORCE／LOG_ONLY，但执行点是 Gateway。历史按 policy session 分组，换 ID 重置计数；认证入口绑定 principal，无认证入口同 ID 共用历史；时间窗口最长二十四小时，策略更新会失效既有 session。自动 WAT 传播限同账户同区域，非 AgentCore 中间层须自行转发，依赖先前响应的动作必须等其完成。（A11、A12、A13）

所以托管组合需要同时限制运行环境的出口：直接 HTTP、SDK、ADB 或本地命令若绕过 Gateway，就不进入该策略事件链。跨 session 总预算和可信审批仍需应用设计，CloudWatch 决策日志不能补齐漏掉的执行路径。这是文档覆盖范围的推论，不是已测攻击。（A9、A11、A12、A13）

## 5．迁移判断：替换执行底座，不吞并其他职责

E2B／Daytona／Modal 是执行与工作区职责的真正替代；Runtime 是整代理服务部署层，Code Interpreter 是数据执行子工具，Policy 可与任一执行底座互补。保留原编排时需把远程文件传输、异步流、取消／超时、重连、工具连接和沙箱标识映射纳入适配器；换 SDK 不会自动复刻原治理语义。（据上述接口与边界综合判断）

本文 ADB 指 Android Debug Bridge：客户端／server／设备连接是独立链路，装 CLI 不代表云沙箱拥有 USB 真机或已配对设备。远程 TCP 需要显式端点授权，域名仅允许 443 的规则不能承载任意 ADB 端口；若需硬件加速模拟器，须单独核实嵌套虚拟化资格，Modal 也不是所有计划默认可用。Bun、CPython 与数据 C 扩展宜按架构锁定镜像、wheel 和系统库，再关闭安装阶段的广泛出口。（T1、E2、E5、M1、A2；迁移建议）

集成时不要复用一个 ID 就假设状态同步：工作区恢复、policy-history 和 audit 各有独立保留与失效条件。Runtime 工作区可能因版本更新重置，Policy session 也可能因规则变更失效，而 Daytona 管理审计和 Modal 十分钟输出流又不是同一类证据。适配层需分别管理映射、恢复与日志归档；快照恢复不自动恢复外部审批账本。（D7、M5、A5、A12、A13；设计推论）

## 第一方链接

### E2B

- E1：[公开运行时源码与隔离机制](https://github.com/e2b-dev/runtime)
- E2：[扩展 Code Interpreter 模板](https://docs.e2b.dev/code-interpreting/customize-template)；[Bun 模板](https://docs.e2b.dev/template/examples/nextjs-bun)
- E3：[暂停、恢复、持久化与降级边界](https://docs.e2b.dev/sandbox/persistence)
- E4：[计划与容量限制，不引用价格](https://docs.e2b.dev/billing)
- E5：[网络控制、动态更新、请求转换与 workload identity](https://docs.e2b.dev/network/internet-access)
- E6：[秘密注入及解析失败语义](https://docs.e2b.dev/secrets/inject)
- E7：[命令输出流与重连](https://docs.e2b.dev/commands/streaming)
- E8：[公开 URL 默认、解释器端口与强制私网阻断](https://docs.e2b.dev/network/restrict-public-access)

### Daytona

- D1：[按类别区分隔离、网络及组织权限](https://www.daytona.io/docs/en/isolation/)
- D2：[快照、镜像与运行环境](https://www.daytona.io/docs/en/snapshots/)
- D3：[文件／内存／外部存储与生命周期](https://www.daytona.io/docs/en/persistence/)
- D4：[无状态执行、Python context、shell 与流](https://www.daytona.io/docs/en/process-code-execution/)
- D5：[秘密代理、host allowlist 与权限风险](https://www.daytona.io/docs/en/secrets/)
- D6：[资源池、状态占额和创建限速](https://www.daytona.io/docs/en/limits/)
- D7：[管理操作审计日志](https://www.daytona.io/docs/en/audit-logs/)

### Modal

- M1：[生命周期、两种 runtime、自定义镜像与嵌套虚拟化](https://modal.com/docs/guide/sandboxes)
- M2：[网络安全、SNI 限制及动态策略](https://modal.com/docs/guide/sandbox-networking)
- M3：[资源上限、GPU runtime 与抢占](https://modal.com/docs/guide/sandbox-resources)
- M4：[文件／内存快照的 TTL 和约束](https://modal.com/docs/guide/sandbox-snapshots)
- M5：[exec 流与输出保留期限](https://modal.com/docs/guide/sandbox-spawn)
- M6：[秘密注入 Beta 的不可组合限制](https://modal.com/docs/guide/sandbox-secret-injection)
- M7：[工作区额度与 Function 专属扩容限制](https://modal.com/docs/guide/scale)

### AWS AgentCore

- A1：[microVM、会话与 V2](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html)
- A2：[HTTP、ARM64、SSE 与 WebSocket 合约](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html)
- A3：[V2 快照中的随机性、凭据和连接注意事项](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-v2-optimize.html)
- A4：[Runtime／Code Interpreter／命令执行／Policy 配额](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html)；[协议对照](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-service-contract.html)
- A5：[Session Storage、S3 Files、EFS 与 EBS](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-filesystem-configurations.html)
- A6：[Instances 隔离、角色凭据与会话用户绑定](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-instances-security.html)
- A7：[解释器会话的专属 microVM](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-session-characteristics.html)；[shell 与 AWS CLI 支持](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-resource-session-management.html)
- A8：[预装数据／系统依赖库](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-preinstalled-libraries.html)；[运行时选择](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-runtime-selection.html)
- A9：[解释器网络模式与执行角色](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-resource-management.html)；[VPC 路由、安全组和 NAT](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-vpc.html)
- A10：[解释器事件流](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-execute-code.html)
- A11：[Policy 执行点、Cedar／Dogwood 与决策记录](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy.html)
- A12：[时序策略、窗口、策略更新、区域及重置风险](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-temporal.html)
- A13：[policy session、principal 绑定、WAT 及多跳范围](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-session-based-temporal.html)
- A14：[Gateway 目标授权、代持凭据及共享角色权限上界](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-outbound-auth.html)
- A15：[Identity 取钥 API 明确返回 apiKey](https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_GetResourceApiKey.html)

### 工具链

- T1：[Android 官方 ADB 架构、USB／无线配对和 TCP 连接](https://developer.android.com/tools/adb)
