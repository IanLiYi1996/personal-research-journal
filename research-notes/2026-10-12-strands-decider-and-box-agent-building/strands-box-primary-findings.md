# Strands Box 一手证据工作报告

研究日：2026-10-12，Asia/Shanghai。范围仅为 Strands Box，不是 Box.com；不承担 Strands Decider 的技术结论。

证据分级：**事实**指当日读取的官方网页、日期固定的官方源码；**报告**指官方仓库 issue 中的直接观察，注明贡献者或外部报告者；**推论／建议**不当作已经实现的能力。本轮只进行了网络搜索和只读源码检查，没有安装、编译、运行 Box 或部署云资源。

## 1. 可供主报告直接采用的结论

- **事实：Box 是“OS 约束 + 受中介操作的确定性策略执行”，不是所有系统调用都经过 Dogwood。** 原生 SDK、自带文件工具、外部程序和 MCP 服务自身的文件操作，只受各自的 OS grants 约束，不产生逐次 `fs:*` 策略决定。需要逐文件策略时，必须去掉相应直接访问权，让操作走 Shell／Monty。[用户指南总览][guide]、[双路径设计][contain]
- **事实：当日可核验的发布支持仍为 Apple silicon、macOS 15+。** 固定 SHA 的 Linux facade 只选择 `aarch64`，明确拒绝 `x86_64`；发布包只有 macOS。Linux 后端源码存在不等于 Linux 发布支持，更不等于两个 Linux 架构都可用。[平台表][guide]、[架构选择源码][linux-facade]、[v0.1.0 发布][release]
- **事实：Box 的信任边界比 MicroVM 大。** Shell、嵌入的 Lua、Monty、网关和 Dogwood 在沙箱外的同一可信进程，旁边就是凭据与 TLS CA 私钥。解释器内存安全缺陷应视为可信进程代码执行风险；Rust 不构成“没有 unsafe／C 代码”的证明。[设计决定][decisions-tcb]
- **事实：时序历史按 Box 目录持久化，而不是每次 `run` 自动清空。** 新运行重开同一个 `private/dogwood.redb`；新 Box／删除目录／部分策略改变可以改变预算状态。历史不跨 Box 分享，也不与外部副作用原子提交。[历史设计][policy-history]
- **事实：普通 HTTP 策略只看到 host、port、method、path、长度等字段，看不到 query、headers 或业务请求体。** 所以“支付 API 每天最多 100 美元”不是现成 REST 网关能力；需要暴露金额语义的受控工具／MCP schema／额外业务授权。[HTTP 拦截接口][gateway-seam]、[策略 schema][network-schema]
- **事实：遥测是诊断记录，不是完整、抗篡改、保证送达的审计链。** 同一文件内出现 Agent spans 和 Box decisions，也不意味着已建立因果父子关系。[遥测限制][telemetry]、[CLI 因果关联限制][cli-telemetry]

**事实：**Box 以 Apache License 2.0 开源，官方定位 developer preview／0.1.x，配置 keys、policy actions、CLI flags 可在发布间变化。许可证允许使用和修改，不代表生产支持、兼容性或安全保证。[许可证说明][readme]、[预发布提示][guide]

## 2. 日期固定、仓库身份与平台矛盾

### 2.1 本报告的时间锚

GitHub API 首次返回的服务器时间为 `2026-10-12T09:17:50Z`，即上海时间 10 月 12 日 17:17:50。按该时刻执行 `commits?until=...`，得到最新提交：

`93c4749c377a64fb7a04f4d5a07afd880a290b38`，提交时间 `2026-10-11T17:48:04Z`，即上海时间 **10 月 12 日 01:48:04**。下文源码链接均固定此 SHA，除明确引用发布 tag 或某次修复提交外，不使用漂移的 `main`。[截至查询时刻的提交列表][asof-commits]、[固定提交][head]

| 证据 | UTC 时间／状态 | 上海时间／研究解释 |
|---|---|---|
| 初始公开提交 `158dade7…` | 2026-10-07 15:52:05 | 10 月 7 日 23:52:05；提交说明写 macOS Apple silicon，Linux planned。[初始提交][initial] |
| `v0.1.0` 指向 `81d241f9…` | 对应提交 2026-10-07 16:02:13 | 10 月 8 日 00:02:13。[tags API][tags] |
| `v0.1.0` 发布 | 2026-10-07 17:24:27 | 10 月 8 日 01:24:27；AWS 文章的“Oct 7 发布”是文章日期，不能换算成上海日历后仍写所有事件都在 10 月 7 日。[releases API][releases-api] |
| 修复每个 Box 独有 alias inode：`331d563e…` | 2026-10-08 21:31:35 | 10 月 9 日 05:31:35；已在上述源码 SHA 中，但不在 `v0.1.0` tag 中。[修复提交][alias-fix] |

**事实：**当日 API 列出 `v0.0.1`、`v0.1.0` 两个 tag，公开 release 仅 `v0.1.0`，资产为 `box-v0.1.0-aarch64-apple-darwin.tar.gz` 和 `SHA256SUMS.txt`。GitHub 的 `prerelease=false` 与用户指南的“pre-release 0.1.x／developer preview”并存；前者是发布元数据，不应被解释成 GA、安全认证或稳定兼容承诺。[tags][tags]、[release 元数据][releases-api]、[用户指南成熟度][guide]

### 2.2 官方仓库路径与平台证据

用户实际提供的是 Strands Box 官方用户指南，并未提供 labs Box 仓库或 Linux 支持声明。调查过程中曾出现 `strands-labs/strands-box` 这一待核验路径线索，**但该精确地址的 GitHub API 返回 404，未取得 README 或日期固定版本，因此不纳入平台支持证据。** AWS 发布文章和用户指南实际链接到 `strands-agents/box`，本报告以此官方仓库核验，不推断曾发生仓库迁移。[AWS 发布文章][aws-blog]、[待核验路径 API][labs-api]

当日可独立复核的证据如下：

| 来源 | 实际范围 | 能否支持“Linux x86_64 已支持” |
|---|---|---|
| AWS Oct 7 文章 “What’s next” | 从 macOS 起步，扩展 OS 是下一步。[AWS][aws-blog] | 不能 |
| 指定官方用户指南平台表 | macOS 15+ Apple silicon supported；Linux not supported yet；Windows unsupported。[指南][guide] | 不能 |
| 固定 SHA README | 本地 macOS Apple silicon preview。[README][readme] | 不能 |
| 固定 SHA facade L75–109 | Linux 仅 `aarch64` 进入 namespace backend；其他架构直接 `PlatformUnsupported`。[源码][linux-facade] | 明确反证 x86_64 |
| 固定 SHA 下载脚本 | 只接受 Darwin + arm64/aarch64。[下载脚本][download] | 不能 |
| 发布工作流 | 构建三种 tarball，但发布 bundle 只保留 macOS；Linux 构建 artifact 不等于 release asset。[工作流][release-workflow] | 不能 |
| Linux 待办 #45、#49 | 移除 x86_64 gate、补 legacy syscalls、发布 ARM64 包及修改下载器仍为 open；贡献者在 10 月 9 日创建。[#45][issue45]、[#49][issue49] | 不能 |

**建议主报告措辞：**“截至 2026-10-12 的已核验发布为 macOS 开发者预览；源码有实验性 Linux ARM64 backend，x86_64 被固定版本明确拒绝。”不要把源码中存在一个后端写成发行版已支持，也不要采用未取得的一段 README 作为支持声明。

官网网页是**当日读取值**，不是 Oct 7 网页存档；源码、tag 和提交时间才是本报告的可复现历史锚。源码已经修复的问题，也不能自动归到仍然下载得到的 `v0.1.0` 二进制。

## 3. 架构与精确拦截路径

### 3.1 两个独立授权路径

**事实：**`box.toml` 设置进程的直接文件、执行与网络可达范围，由 OS 在进程启动前固定；`policy.dw` 决定可信进程代执行的请求。Dogwood 的 `forbid` **不能撤销**已有 direct grant，`permit` 也**不会扩大** Agent 自己的 syscall 权限。`workspace`／`HOME` 本身不授予内容读取权。[OS 与策略路径][contain]、[配置参考][configuration]

**事实：**每个 `box run` 有一个可信进程、一个本地策略引擎、一个网关和一套该 Box 的凭据。Agent、每次外部 tool launch、本地 stdio MCP server 分别被 containment；可信进程不在这些沙箱内。同一个 `box_dir` 只能被一个运行持锁占用。[架构][architecture]

| 客户调用方式 | 实际路径／决定 | 重要缺口 |
|---|---|---|
| 自定义 Python `@tool` 内直接 `open()`，或 harness 的 Read／Write | Agent syscall → Agent OS grants | 不产生 `fs:*` 历史或逐次策略决定。[指南][sandboxes] |
| Agent 执行 `bash`／`zsh`／`sh` | alias → Unix socket `run/box.sock` → broker → Strands Shell | 不是宿主原生 shell；alias 不授权，broker 将其视为不可信输入。[架构][architecture] |
| Shell 内置 `cat`、`rm`、重定向等 | 展开、解析、程序解析 → `shell:exec` → 每个 `fs:*` → deny-only reach check → 实际操作 | 路径／操作语义可见；不是把原始命令字符串拿给 LLM 判断。[Shell 设计][shell-design] |
| Shell 启动 `git`／`cargo`／客户二进制 | `shell:spawn` → 匹配 tool table 或 exec entry → 独立 tool sandbox | 一次许可覆盖整个进程树，内部文件操作、hooks、子 shell 不产生新的 `fs:*`。[Shell 设计][shell-design] |
| Agent 直接运行已授予 `exec` 的程序 | 在 Agent sandbox 内启动 | 不经过 `shell:spawn`；不是每次 exec 都被 Dogwood 看到。[指南][sandboxes] |
| `python`／`python3` alias | broker → Monty；文件 effect 挂起 → Box 授权并执行 | 不等于拦截 Agent 本身的 CPython `exec()` 或任意 SDK 内部代码。[Monty 设计][monty-design] |
| SDK 的 HTTP(S)／模型请求 | OS 限制 + proxy → egress gateway → `net:connect`、`http:request` | HTTP 业务体不进入普通策略；客户端必须兼容代理和 CA。[网络指南][network-guide] |
| 配置的本地 MCP | server alias → broker → `shell:spawn` 启动 server → `mcp:call`／逐工具 refinement | server 自身 effects 主要受 OS grants，不由 MCP 参数策略追踪。[MCP 设计][mcp-design] |
| 配置的远程 HTTP MCP | 网关 → 网络两门 + `mcp:call` + 已发现工具／参数规则 | 远程服务不被 containment；允许调用后的服务端行为不受 Box 控制。[MCP 设计][mcp-design] |

**事实：**Shell policy 的动作适配发生于实际 effect admission；源码测试明确描述 `Shell::run → Mediated::admit → ShellPolicyInterceptor → PolicyEngine::decide`，之后执行并记录 outcome。Shell 先解析展开，`eval`、命令替换、`find -exec`、`xargs`、source、traps 等都回到同一命令 admission；文件对象身份在执行前复核，降低 symlink／TOCTOU 路径替换风险。[端到端测试源码][shell-e2e]、[Shell 设计][shell-design]

### 3.2 Python 与 shell 的适配成本

**事实：**Monty 是 Python 子集解释器，没有下层 CPython；其 I/O 是挂起点，由 Box 代办。支持部分标准库，PyPI 包、`subprocess`、`socket`、`urllib` 等不能直接 import；联网只有 `fetch()`。`os.environ`／`getenv`、`Path.resolve()`、`mkdir(parents=True)` 等也不支持。[Python 参考][python-guide]

**事实：**Shell 和 Monty 每次调用重建，不保留前一次导出变量或 Python 状态。Monty 对 symlink 路径先按脚本拼写判断策略，随后因其不等于 canonical identity 拒绝；Shell 则以解析后的目标路径判断。两者不能笼统描述成同一套“完整 Python／bash 环境”。[Monty 路径设计][monty-design]、[Shell 参考][shell-guide]

**事实：**源码限制包括 Shell 命令 30 秒、输入 256 KiB、内存 VFS 单文件 10 MiB；Monty 执行 tracker 120 秒、单次分配 128 MiB、10,000 次 host suspension、累计 sleep 180 秒。它们不是 Box 全局进程／CPU／磁盘／总内存限额；设计文档明确提醒小步分配仍可耗尽可信进程内存。[Shell 源码][shell-code]、[Monty limits 源码][monty-code]、[Monty residual risk][monty-design]

**客户建议：**完整 numpy／pandas／第三方 Python 环境可作为原生 Agent 或外部 tool 跑，但届时文件访问主要是 OS 层，不是 Monty 逐 effect 策略。不能既要求零改造任意 Python，又承诺所有文件操作都进入时序历史。

## 4. 网关：不是“任何网络协议都能审查”，也不是“一律 HTTPS-only”

**事实：**明文 HTTP 的 absolute-form 请求与 HTTPS `CONNECT` 都可进入受控路径。**仅对绑定了凭据的请求要求 TLS**；“Box 只支持 HTTPS”不准确。“HTTPS 可用”也不等于任意 TLS 隧道可通过：网关终止客户端 TLS，用每次运行新建的 CA 签证书，再建立并验证上游 TLS。[egress 设计][egress-design]

精确请求链：

1. `net:connect(host, port)` 在 DNS 前决定；拒绝时连 DNS 查询都不发出。
2. 解析一次并固定 IP；每个将拨号的 IP 再问一次 `net:connect`。
3. HTTPS 时接受 `CONNECT`，终止客户端 TLS；内部 request authority 必须与 CONNECT 目标一致。
4. 校验／替换 phantom 或 SigV4 signing；普通 HTTP 策略看到的是原始请求的非秘密语义字段。
5. `http:request` 放行后才向上游转发；响应经过凭据 scrub。[egress 设计][egress-design]

**事实：**目前 wire 为 **HTTP/1.1、每连接一个请求**，请求／响应体上限 **16 MiB**，protocol upgrade（含 WebSocket）拒绝。源码 materialize 请求和响应 body，不能据此承诺透明、低延迟的长期 SSE／HTTP/2／gRPC streaming；这类 workload 未在本轮运行验证。[网络指南协议限制][network-guide]、[HTTP/1 读取源码][http1-code]

**事实：**普通策略输入为 host、port、method、canonical path（不含 query）、`body_bytes`、`intercepted`；没有 header／body 值。请求体“字节数”不是金额、SQL 查询空间、AWS action 或 API 业务结果。回复状态记录为 `output.status`，但 HTTP reply 本身没有新的 Dogwood allow／deny 决定。[策略 action 参考][actions]、[接口源码说明][gateway-seam]

**事实：**标准 proxy／CA 环境变量由 Box 设置，OS 阻断 Agent 的其他直连；忽略代理不会自动绕过，通常是失败。官方 Strands CLI 教程仍需 Node preload：AWS JS SDK 不按 `HTTPS_PROXY` 走，HTTP/2 handler 要换成 HTTP/1.1 handler。因此 “harness-agnostic”不等于“所有 harness 原样即用”。[入门教程][getting-started]

**事实：**Box 网关上游验证采用 public roots；底层 `MitmConfig` 有扩展根的接口，但没有因此自动成为 `box.toml` 支持的用户选项。外部用户 #54 报告企业 TLS interception 场景失败，研究日仍 open；应当作为客户兼容风险，而非本轮确认的普遍故障。[egress 设计][egress-design]、[网关 crate 配置][gateway-readme]、[直接用户报告 #54][issue54]

## 5. Dogwood 语义与历史：可执行，但不等于业务真相

**事实：**Box 依赖 `dogwood-language =1.0.0`、`dogwood-local-engine =1.0.0`，并精确固定 Cedar 4.11.0。它是本地确定性授权引擎，不需要每次联网向模型判断。默认拒绝、匹配 permit 才允许、forbid 覆盖 permit；限制和预算应写为 forbid，不能靠一个较窄 permit 收窄另一个较宽 permit。[依赖源码][policy-deps]、[策略设计][policy]

**事实：**请求还会做 schema 校验；evaluation error、poisoned lock、durable fault 等走 deny。单个引擎的提交在 mutex 后串行，request 先 durable append，再返回 verdict；每个 decision 需要锁和存储同步，突发请求会排队。不存在“任意吞吐、零开销”的一手证明。[决定源码][dogwood-code]、[策略 residual risk][policy]

### 5.1 “per-run engine”与“durable per-box history”必须分开

**事实：**运行中的 engine 对象属于一个 run，但 reopen 相同 `box_dir/private/dogwood.redb` 复用历史，重启不应自动释放预算。历史丢失／损坏与 Box record 不一致时拒绝启动；时钟倒退不使事件时间倒退；历史按最长窗口 checkpoint／prune。最长 temporal window 为 24 小时。[历史契约][policy-history]

**事实：**历史不跨 Box 共享，没有 rewind，也不是抗篡改 audit。副作用与 outcome 记录不是一个事务：请求已出网／操作已完成后崩溃，可能缺少 response。改动的历史规则从空状态开始；Dogwood 1.0.0 的 temporal macro 还可能因为调用前文本／注释改变而丢掉累计状态，官方建议 inline temporal clause。[历史与 macro 风险][policy-history]、[被忽略的 regression test][macro-test]

**推论：**云平台如果每个任务新建 Box 目录，预算自然重新开始；同一客户多个 Box 的“每日总额”也不会自动合并。要跨 session 的总额度，需要外部可信账本／业务授权或明确持久化设计。不能把引擎 durable 直接解释成跨租户、跨副本、跨容器的 durable quota。

### 5.2 计数、成功、并发与示例的边界

**事实：**`::request` 包括拒绝尝试，且 request budget 包含当前正在判断的请求；`::response` 表示已发生或可能发生的操作，不自动表示业务成功；`::error` 是确定未发生。文件 response 另有 completed／descriptor_issued／indeterminate，Shell 有 exit status，HTTP 有上游 status。[详细 event 契约][policy-events]、[action 参考][actions]

**事实：**request 计数可在串行 admission 下严格限制尝试数，但拒绝重试也消耗；response 计数只有操作结束后更新，多个在途请求可能一起越过预算。没有外部副作用的原子 reservation／commit 保证。[计数与并发][policy-events]

**推论：**Big Picture 的“60 次／小时”示例使用 request count `>=60`；按用户指南“包括当前请求”的语义，第 60 个匹配尝试就会被拒，不能照搬为“恰好允许 60 次”。正确边界必须测试；本轮没有执行该测试。[博客示例][big-picture]、[当前请求计数说明][shell-policy-guide]

**推论：**AWS 的 Slack 示例按 HTTP 200 response 计数，只证明记录到该 endpoint 返回 200；普通 HTTP 策略没有读取 JSON `ok`、业务事务提交或金额，因此不能当成 Slack 业务成功证明，也不能当成严格并发预算。[AWS 示例][aws-blog]、[HTTP schema][network-schema]

**事实：**Big Picture 自己承认“测试后 git push”示例不完整，例如 `git switch` 可破坏“推送的是已测代码”的承诺。用户指南也指出只检查 `arg1=="push"` 会漏掉 `git -C … push`、`git -c … push`；Shell 结果应该使用 `shell:spawn`、固定 `program_path`，不能靠可由 workload 伪装的 shell function exit status。[博客限定][big-picture]、[策略编写指南][write-policy]、[工具成功条件][add-tool]

**客户建议：**真正的“仅推送被测代码”仍需可信 commit／tree hash 绑定、限制可修改测试与 hooks、精确命令允许列表或 trusted wrapper；Box 示例不是已完成的软件供应链保证。

## 6. MCP 的可见性、继承许可与发现机制

**事实：**server 级 `mcp:call` permit 已覆盖该 server 的方法和工具；逐工具 `<server>::Action::"<tool>"` 是 refinement，未匹配时继承 coarse permit，**只有匹配 forbid 才收窄**。仅列出几个 per-tool permit 不是工具 allowlist。该逻辑也有源码测试。[MCP 设计][mcp-design]、[admission 源码][mcp-admission]

**事实：**Box 被动观察 harness 的 `tools/list` 回复，将 JSON input schema 生成为 typed actions；最后一页会被扣住直到 schema／规则提交。未知工具、尚未 accepted catalog 的 calls 拒绝，避免 typed rule 尚未生效就放行。schema 变更会重新验证；server 失败／degraded 与其他 server 的可用性分开处理。[MCP discovery][mcp-discovery]

**事实：**本地 `initialize`、`ping`、`server/discover`、`subscriptions/listen` 和 notifications 不做 MCP 策略判断；远程 `initialize` 会判断。不可写“每一条 MCP frame 全都鉴权”。服务端自身发出的 roots／sampling 等请求，其 Agent reply 也有相应 `mcp:call` 路径。[action 参考][actions]

**事实：**`box policy generate-schema` 与运行时被动发现不是同一安全边界：前者会**在沙箱外、以操作者身份启动 stdio server**，并直接用真实凭据请求远程 server。只能对可信服务执行；本轮没有执行。[CLI 参考][cli]

**建议：**schema 只证明参数类型，不证明工具名与实际副作用一致；MCP server 可以在自身 grants 内做额外操作，远程 server 更不可 containment。金额／客户空间规则需要可信 schema、受控服务端实现与上游权限配合。

## 7. 安全负面细节与绕过边界

以下不是“发现并验证了任意逃逸”，而是官方明示的边界、源码行为或有身份区分的 issue 报告。

1. **直接 grants 是策略盲区。** 原生读后没有 `fs:read` 历史，“读敏感文件后禁联网”不会触发；外部 tool／MCP server 的内部读也一样。[局限][limits]
2. **解释器 reach 比 Agent direct grants 大。** 宽泛 `fs:read` permit 可通过沙箱外 Shell／Monty 读操作者 `~/.aws`、`~/.ssh`。direct-grant credential floor 不保护这条路；reach floor 主要保护 Box 状态、当前配置／策略和可达 roots。[策略放置][policy]、[用户安全页][security-guide]
3. **显式 credential-store grant 可获准。** Box 拒绝含凭据仓的宽树，但可允许点名凭据仓／具体文件并在启动时披露；不能说凭据路径在任何配置下都不可读。[文件授权指南][grant-files]
4. **凭据值隐藏 ≠ 凭据使用隔离。** 同 Box Agent、每个 tool、每个 stdio MCP server 都能使用所有 routes；网关无法按请求识别源进程，不能做 per-process credential allowlist。[凭据设计][credentials]
5. **loopback 网关不是只服务被 containment 的 Agent。** 同账户沙箱外程序可向网关发请求，仍由该 Box policy 判断并归因于该 Box；`inject="always"` 可以给这种请求附凭据。collector 也接受本地进程 telemetry。不能以日志中的 Box principal 证明实际调用主体。[官网安全页][security-guide]
6. **native egress 是明确的逃生口。** tool／本地 MCP 的 `network.contain_egress=false` 绕过网络策略、凭据注入与逐请求记录；Agent 不接受该选项。Agent 可控制 tool 参数，所以不要给通用 curl／解释器这个例外。metadata forbid 也覆盖不到 native egress。[网络指南][network-guide]、[tool sandbox][sandboxes]
7. **网关没有内建 metadata deny。** 操作者需写 hostname 与 resolved IP 的 forbid；DNS pinning 防重绑定，但不是“所有私网／元数据已默认硬封锁”。[网络指南][network-guide]
8. **外部程序的一个 spawn permit 覆盖整个进程树。** macOS tool sandbox 可运行可达 helpers、加载自身可写代码、查看较广 home metadata；git hooks、npm scripts 是被一起信任的代码。[tool sandbox][sandboxes]
9. **允许目的地仍是外泄渠道。** 模型 endpoint 也可能接收敏感数据；“读后禁其他 host，但豁免模型”不是完整 DLP。允许的写直接改变真实文件，退出不回滚；只 forbid delete 也不能防 truncate／恶意覆盖。[局限][limits]、[用户指南示例][guide]
10. **TCB 含复杂解释器与协议解析器。** Shell 嵌 Lua 5.4 C，Monty 有 unsafe；可信进程的缺陷不受 Agent sandbox 再 containment。[TCB 设计决定][decisions-tcb]
11. **无 Box 全局资源与租户隔离保证。** 进程、CPU、disk exhaustion、kernel exploit、侧信道等需要外围环境；解释器的局部限额不能替代 tenant/session resource boundary。[局限][limits]
12. **沙箱外生成代码不受后续保护。** Box 控制本轮 effects，不控制 Agent 写出的程序日后在外面被人执行，也不能 containment 已获准访问的远程服务。[shared responsibility][security-design]
13. **退出要区分受处理 signal 与杀死可信进程。** 文档要求避免对 `box run` 用 SIGKILL；#53 的贡献者待办指出缺 parent-death signal，不能假设可信进程死亡一定同步清理所有 workload。这不等于 workload 的 OS grants 自动消失。[CLI signal 契约][cli]、[贡献者 #53][issue53]
14. **macOS 兼容性并非透明。** 官方记录 timestamp-preserving copy／archive／sync 可能成功但 timestamp 错误，或写出部分内容再失败；pi 教程还需 process-title 与 timestamp probe workaround。[macOS 设计][macos]、[pi 教程][pi-guide]

### 7.1 凭据与文档矛盾

**事实：**配置源码支持 `env://`、`aws://`、`credsd://` 三类。设计凭据文档却仍写“只接受两类”；官网配置参考已列 credsd，10 月 12 日 08:09:13Z 创建的 #104 直接指出文档落后。故以固定 SHA parser 为实现证据，不把旧 design 文档的“两类”当成当前实情。[parser][credential-parser]、[旧 design 文档][credentials]、[官网配置][configuration]、[文档问题 #104][issue104]

**事实：**`aws://profile` 路径不等于完整 AWS credential chain：源码拒绝 `credential_process`、SSO、role_arn、source_profile、web_identity_token_file，支持明确的静态 keys（可含已有 session token）；动态临时身份有 credsd 路径，不能声称自动使用 ECS／AgentCore task role。[AWS profile 源码][aws-profile]、[网络指南][network-guide]

**报告＋源码交叉支持：**外部用户 #39 在 AgentCore ARM64 自建版本中观察 Agent 收到 phantom，但 brokered Shell 的 `$TOKEN` 为空；固定 SHA 的 Shell construction 只显式设置 PWD／HOME／USER，没有投影 credential placeholders。官方“curl 使用 `$TOKEN`”示例需实测，不能把 Agent environment 与 brokered Shell environment 混为一谈。用户说明未复现 macOS、未验证最新 HEAD；本轮也未运行。[用户 #39][issue39]、[Shell build 源码][shell-code]

**文档矛盾：**egress design／decisions 写 `inject="always"` 会 stderr warning；官网用户指南明确写没有 warning、decision record 不标 placeholder 缺失。未运行确认，主报告应保留差异，不承诺操作者一定能收到告警。[egress design][egress-design]、[官网网络指南][network-guide]

## 8. Linux：KVM、权限与容器可行性

**事实：**已读 Linux 后端机制是 user／PID／mount／network／IPC namespaces、mount view、capability drop、`no_new_privs` 与静态 seccomp；不是 KVM hypervisor。probe 在子进程尝试 `unshare(CLONE_NEWUSER)`；不允许就 fail closed。[后端源码][linux-backend]、[user namespace probe][linux-probe]

**推论：**Box 自身 Linux containment 没有从这些路径体现 KVM 依赖。若另用 Firecracker／Kata 提供外层 MicroVM，KVM／虚拟化能力属于外围平台条件，不能把“Box 需要 KVM”与“客户选择的 MicroVM 需要虚拟化条件”混写。本轮没有验证任何主机或平台的 KVM／nested virtualization 条件。

**事实／贡献者说明：**ARM64 backend 还不具备 macOS tool/MCP broad-exec parity；两个 Linux integration cases 被明确 quarantine。Linux CI advisory，工作流对 Ubuntu userns 的 AppArmor restriction 做环境调整；README 明示尚非完整 Linux qualification。[quarantine][quarantine]、[Linux CI][linux-tests]、[工作流][linux-ci]

**直接用户报告，不是官方生产认证：**#26 在 Kata 4.2.0、AL2023、guest kernel 6.18 的 masked `/proc` pod 中遇到 fresh proc mount EPERM，报告只有 privileged guest pod 可运行；`procMount: Unmasked`／`hostUsers:false` 与所用 Kata runtime 的组合又不兼容。此证据证明“常规 unprivileged Kubernetes container 一定可运行”不成立，但不能推广成所有容器都必须 privileged。[原始报告 #26][issue26]

**贡献者待办：**#49 指出 Linux workflow 构建的 glibc 2.35 包不能直接跑 AL2023 glibc 2.34，改用 AL2023 build／静态链接、发布 ARM64、下载器支持仍待实现。#45 的 x86_64 系统调用兼容也未完成；#51 列 TIOCSTI、set-id chmod、open_tree_attr 等 unmeasured／待收窄事项；#53 列 host-root 检测与 parent-death 缺口。这些是实验性 Linux 的风险证据，不是本轮执行出的漏洞利用。[#49][issue49]、[#45][issue45]、[#51][issue51]、[#53][issue53]

**建议：**不要建议客户为了让 Box 启动就解除宿主保护或开启普通宿主上的 privileged pod。若做实验，应明确 guest 内 privilege 与宿主 privilege 的边界，验证 namespace、proc mounts、glibc、runtime helpers 和正反向 containment tests；目前不能给 AgentCore／ECS／Kubernetes 一张通用“可部署”勾选表。

## 9. 云部署：当前、倡议与路线图区分

| 场景 | 当日证据与正确定位 |
|---|---|
| 本地 Mac 开发 Agent | 已有官方入门、Strands SDK／harness、Claude Code、Codex、pi 示例；preview，不是成熟生产 SLA。[入门][getting-started]、[README][readme] |
| Box + AgentCore／ECS／Kubernetes 打包 | AWS Oct 7 “What’s next”明确是希望将来让自建 Agent 带 Box 部署的方向。[AWS 路线图][aws-blog] |
| Box inside AgentCore Runtime／per-session MicroVM | Big Picture 推荐的 defense-in-depth 架构，不是当日发布包／安装器已支持这些目标的证明。[Big Picture][big-picture] |
| AgentCore ARM64 自行构建实验 | #39 提供直接用户实验线索，但仍有 phantom projection 问题，不能升级成官方生产支持。[报告][issue39] |
| EKS／Kata／容器 | #21、#26 有外部试验；架构 gate、masked proc、privilege 与 qualification 缺口并存。[#21][issue21]、[#26][issue26] |
| 多租户隔离、session resource limits | Box 明确不提供；属于外层平台。[shared responsibility][security-design] |

**核心张力：**两篇同日博客分别说“未来带到 deployed agents”和“最容易是放进 AgentCore agent”；合理解读是**已有架构建议与部署意图，但当前发行支持／完整验证尚未对齐**，不是以一篇覆盖另一篇。

**建议架构，不是已实现保证：**外层 per-session VM／MicroVM 负责租户隔离与资源预算；内层 Box 在受控平台上负责本地语义 policy；IAM／上游业务权限与独立 quota ledger 管实际授权／金额；遥测另送受保护的审计管道。Box 不替代其中任何一层。

## 10. 遥测与因果绑定

**事实：**默认记录在 `<box_dir>/private/telemetry/records.jsonl`，OTLP JSON，effective permit／deny 与 control-plane records 可和 Agent 自报 spans/logs/metrics 放在一起。collector 剥离 Agent payload 的 `strands.box.*` claims 并加 source stamp，因此不能通过正常遥测入口伪造一条 Box decision；但 Agent 自报内容仍不是可信事实。[遥测设计][telemetry]、[用户记录指南][record-telemetry]

**事实：**记录 best effort，运行时 target write 失败不重试、不报告丢失；无签名／hash chain，无自动文件轮转，改到 workload 可写目录会丢失默认保护，两 Box 共写一文件可撕裂行。官网已有 OTLP target，而 repo design 文档仍写“只有 file target”，也是文档漂移，不能把两者混为稳定版本契约。[用户记录指南][record-telemetry]、[配置参考][configuration]、[旧 design 文档][telemetry]

**事实：**durable policy history 与 telemetry 是两个存储，不可因 JSONL 丢记录便说 engine 预算丢失，也不可因 redb durable 便说 telemetry 必然完整。原生 syscall 的拒绝和访问本来就不进入 Dogwood decision log。[历史设计][policy-history]、[记录指南][record-telemetry]

**事实：**trace correlation 由 workload 提供，只是 hint，不是授权主体或审计证明。官方 Strands CLI telemetry 教程明确：CLI 没有设置来传 trace context，每条 Box decision 落到自己的 trace，Agent traces 与 decisions 只并列；Claude Code 示例则配置 traceparent propagation。故“可关联”取决于实际 harness 和传递路径，不是 Box 自动提供所有工具调用的完整 causal graph。[correlation 设计][telemetry]、[CLI 限制][cli-telemetry]、[Claude Code 示例][claude-guide]

**建议：**客户可用 records 排查策略为什么拒绝；正式审计要独立定义完整性、交付确认、保护、rotation 和主体认证。不能把同文件、同 trace id、或 `source=box` 等同于外部业务效果的可信证明。

## 11. 已有例子与客户适配建议

**事实：**日期固定的 Strands SDK 例子有三个 `@tool`：list_project、read_file、run_command，均用 `subprocess.run(["zsh","-c",…])` 走 Box Shell；Agent 是原生 CPython，项目只授予 list，SDK／runtime 单独 grants；policy 拒绝 `.env` 和删除。这是具体可检查的集成代码，不是“自动拦截任意 SDK 工具”。[Agent 源码][sdk-agent]、[Box 配置][sdk-box]、[policy][sdk-policy]

**事实：**官网 harness 教程进一步只保留 shell tool，关闭 session／memory／skills 的工作目录访问，并展示允许 README／统计行数、拒绝 `.env`／删除的输出。CLI 教程还要做代理 preload，pi 要 shellPath 与兼容 workaround；这些证明路线可行，也证明 integration 有工程成本。[harness 教程][harness-guide]、[CLI 入门][getting-started]、[pi][pi-guide]

**建议／适合：**Mac 上的受控编码 Agent、只读运维调查、生成小型 Shell／Python 程序、方法／路径层 API 限制、MCP 参数规则、尝试次数／冷却窗口，是较可信的 pilot 方向。先做 shell-only 工具集、窄 grants、窄网络 endpoint、分 Box 分凭据，再做 adversarial 测试。

**建议／不宜直接承诺：**Linux 云端 production parity、任意 Python 数据科学栈的逐 effect 策略、任意原生 SDK 业务体治理、严格金融金额日限额、全局跨 Agent quota、多租户安全、强审计、完全防外泄。上述都有具体的可见性、平台或事务边界。

没有在所读来源中取得“采用 Box 后某客户改善多少成功率／成本／安全事件”的可信量化证据。博客需求场景、官方 runnable example、外部 issue 集成报告应分别标记，不能改写成已认证客户案例。

## 12. 必须保留的未知项与主报告纠错清单

- 早期 labs 路径线索未取得正文或版本，**不纳入平台支持证据，也不归因于用户陈述**。
- Box 在具体 AgentCore、ECS/Fargate、Kubernetes distribution 上无需特权地运行：**未验证**，有 contrary container observations。
- 具体客户外层 KVM／nested virtualization／MicroVM entitlement：**未调查，不把 Box 本身与外层要求混同**。
- `inject="always"` warning 文档冲突、企业 CA 场景、broker Shell phantom 缺失的实际 release／平台矩阵：**未运行验证**。
- “每小时 60 次”、Slack 200 成功、测试绑定 push 的精确 adversarial 行为：有语义／覆盖缺口证据，**不是本轮跑过的结果**。
- Oct 7 后已合并的修复应写成“截至日源码包含”，不写成“v0.1.0 发布包已包含”。
- Box 当前策略 PDP 为 Dogwood；本报告没有确认 Box 调用 Decider 或二者有官方直接集成，不能在合并稿画成既成调用链。

研究方法采用 research 技能要求的一手证据回溯和单文件工作笔记；遇到文档与实现不一致，保留版本与可信度差异，不以推测补齐。主报告可直接复用本报告结论和精确来源，不需改动本工作报告的目录归属。

## 来源链接（正文引用均解析到精确来源）

[guide]: https://strandsagents.com/docs/user-guide/box/
[aws-blog]: https://aws.amazon.com/blogs/opensource/introducing-strands-box-ai-agent-sandboxes-powered-by-dogwood/
[big-picture]: https://strandsagents.com/blog/strands-box-the-big-picture/
[labs-api]: https://api.github.com/repos/strands-labs/strands-box
[asof-commits]: https://api.github.com/repos/strands-agents/box/commits?until=2026-10-12T09:17:50Z&per_page=20
[head]: https://github.com/strands-agents/box/commit/93c4749c377a64fb7a04f4d5a07afd880a290b38
[initial]: https://github.com/strands-agents/box/commit/158dade7ba1e015f08e7e5a6358f71f33793628f
[tags]: https://api.github.com/repos/strands-agents/box/tags
[releases-api]: https://api.github.com/repos/strands-agents/box/releases?per_page=10
[release]: https://github.com/strands-agents/box/releases/tag/v0.1.0
[alias-fix]: https://github.com/strands-agents/box/commit/331d563e8599b4359f9578ea377fc4f2d0dccce7
[readme]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/README.md
[download]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/download.sh#L30-L40
[release-workflow]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/.github/workflows/deploy-box-artifact.yml#L304-L360
[linux-facade]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/containment/src/facade.rs#L75-L109
[linux-backend]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/containment/src/backend/linux/namespace/mod.rs
[linux-probe]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/containment/src/backend/linux/namespace/probe.rs#L1-L34
[quarantine]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/test-integ/QUARANTINE.md
[linux-tests]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/test-integ/README.md#L40-L81
[linux-ci]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/.github/workflows/containment-deterministic.yml#L74-L125
[architecture]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/architecture.md
[contain]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/containment.md
[limits]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/limitations.md
[security-design]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/security.md
[security-guide]: https://strandsagents.com/docs/user-guide/box/security/
[sandboxes]: https://strandsagents.com/docs/user-guide/box/security/agent-and-tool-containment/
[configuration]: https://strandsagents.com/docs/user-guide/box/reference/configuration/
[actions]: https://strandsagents.com/docs/user-guide/box/reference/policy-actions/
[cli]: https://strandsagents.com/docs/user-guide/box/reference/cli/
[shell-guide]: https://strandsagents.com/docs/user-guide/box/reference/shell/
[shell-design]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/shell.md
[shell-code]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/box/src/run/broker/shell.rs#L23-L202
[shell-e2e]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/policy/tests/temporal_shell_e2e.rs
[python-guide]: https://strandsagents.com/docs/user-guide/box/reference/python/
[monty-design]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/monty.md
[monty-code]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/box/src/run/broker/python/mod.rs#L29-L63
[egress-design]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/egress.md
[network-guide]: https://strandsagents.com/docs/user-guide/box/guides/network-and-credentials/
[gateway-seam]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/egress-gateway/README.md#L37-L80
[gateway-readme]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/egress-gateway/README.md#L197-L223
[http1-code]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/egress-gateway/src/intercept/mitm/http1.rs#L553-L700
[network-schema]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/policy/schema/actions.cedarschema#L163-L199
[policy-deps]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/policy/Cargo.toml#L28-L57
[dogwood-code]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/policy/src/dogwood.rs#L390-L490
[policy]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/policy.md
[policy-events]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/policy.md#L110-L146
[policy-history]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/policy.md#L148-L224
[macro-test]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/policy/src/dogwood.rs#L2197-L2223
[write-policy]: https://strandsagents.com/docs/user-guide/box/guides/write-a-policy/
[shell-policy-guide]: https://strandsagents.com/docs/user-guide/box/guides/write-shell-policy/
[add-tool]: https://strandsagents.com/docs/user-guide/box/guides/add-a-tool/
[grant-files]: https://strandsagents.com/docs/user-guide/box/guides/grant-files/
[mcp-design]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/mcp.md
[mcp-discovery]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/mcp-discovery.md
[mcp-admission]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/box/src/run/broker/mcp/admission.rs#L28-L112
[credentials]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/credentials.md
[credential-parser]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/box/src/record/config/egress.rs#L371-L433
[aws-profile]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/crates/credentials/src/sources/aws_profile.rs#L14-L120
[decisions-tcb]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/decisions.md#L1253-L1266
[macos]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/macos-enforcement.md
[telemetry]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/docs/design/telemetry.md
[record-telemetry]: https://strandsagents.com/docs/user-guide/box/guides/record-telemetry/
[cli-telemetry]: https://strandsagents.com/docs/user-guide/box/guides/strands-cli-telemetry/
[getting-started]: https://strandsagents.com/docs/user-guide/box/getting-started/
[harness-guide]: https://strandsagents.com/docs/user-guide/box/guides/run-strands-harness/
[claude-guide]: https://strandsagents.com/docs/user-guide/box/guides/run-claude-code/
[pi-guide]: https://strandsagents.com/docs/user-guide/box/guides/run-pi/
[sdk-agent]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/examples/strands-box/strands-sdk-agent/agent.py
[sdk-box]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/examples/strands-box/strands-sdk-agent/box.toml
[sdk-policy]: https://github.com/strands-agents/box/blob/93c4749c377a64fb7a04f4d5a07afd880a290b38/examples/strands-box/strands-sdk-agent/policy.dw
[issue21]: https://github.com/strands-agents/box/issues/21
[issue26]: https://github.com/strands-agents/box/issues/26
[issue39]: https://github.com/strands-agents/box/issues/39
[issue45]: https://github.com/strands-agents/box/issues/45
[issue49]: https://github.com/strands-agents/box/issues/49
[issue51]: https://github.com/strands-agents/box/issues/51
[issue53]: https://github.com/strands-agents/box/issues/53
[issue54]: https://github.com/strands-agents/box/issues/54
[issue104]: https://github.com/strands-agents/box/issues/104
