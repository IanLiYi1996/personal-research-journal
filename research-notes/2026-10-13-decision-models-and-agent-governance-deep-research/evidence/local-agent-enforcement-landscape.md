# 本地编码代理的沙箱与强制执行路线

核查日期：2026-10-13，Asia/Shanghai。仅用公开主源、固定标签源码及深层官方文档；未登录、安装或运行产品测试。范围限本地执行，不含托管服务、私有资产审计及旧论文沿革。以下“主流”指已进入主要编码客户端或正式本地产品的交付路线，不代表采用率排名；主源不足以支持采用人数比较。

## 先区分三层

审批属于 harness：决定某次工具调用是否启动、是否扩大权限；命令获批后，其任意子进程和系统调用并不自动再次审批。OS 强制执行由内核限制文件、网络及进程能力，不能用提示词代替，但通常仍共享宿主内核。microVM 为每个环境另起客体内核，改变逃逸边界；它也不能撤销已共享目录中的写入，或判断一次获准上传是否符合用户意图。[1][14]

## Claude Code 与可独立嵌入的 SRT

核查可见 Claude Code 最新发布标签为 **2.1.296**，SRT 为 **0.0.79**，后者仍标为研究预览。Claude 客户端支持 macOS 13+、Ubuntu 20.04+、Debian 10+、Alpine 3.19+，但内置命令沙箱仅支持 macOS、Linux、WSL2；原生 Windows 能运行客户端，不代表能使用该沙箱。macOS 使用 Seatbelt／`sandbox-exec`；Linux 使用 bubblewrap、网络命名空间及 socat，另有可选 seccomp Unix-socket 阻断。[2–5]

内置沙箱默认关闭；开启后仅包围 shell 命令及其子进程。Read/Edit、WebFetch、MCP、hooks、LSP 等仍在外面。默认可写工作目录、添加目录和临时目录，配置、hooks 等路径另受写保护；读取和环境继承却较宽，不能假设 SSH 密钥或环境 token 已隐藏。自动批准与常规审批使用同一 OS 边界；不兼容、缺依赖或获准重试可能产生非沙箱执行。需要硬边界时，应在可信配置中要求可用性失败即退出、禁止非沙箱重试，并检查命令排除项；2.1.285 起还加强了仓库配置不能放宽管理员要求的规则。[3]

SRT 的不同定位是 CLI／TypeScript 库，可包围整个代理或本地 MCP 服务，而非只包围 Claude 的 Bash。0.0.79 要求 Node ≥22.12；macOS/Linux 均列 ripgrep 依赖，独立库未单列 macOS 最低版本。Linux 还需 bubblewrap ≥0.4、socat，特殊文件遮罩及部分清理语义需 ≥0.5。x64/arm64 提供 seccomp helper；源码实际组合网络、用户、PID 命名空间、只读挂载与过滤器，不是仅设置代理变量。[5][6] Linux 可运行性还依赖宿主允许所需命名空间，不能把“支持 Linux”当成所有受限容器都兼容。

独立 SRT **已有 Windows alpha**：独立 `srt-sandbox` 用户、受限 token、job object、NTFS ACE，以及按该用户 SID 生效的 WFP 出站围栏；x64/arm64 helper 需一次管理员部署。不要将它写成 Claude 原生 Windows 已支持，也不要写成 microVM。该版本未给统一 Windows 最低构建门槛；系统 DNS 服务、共享账号和初始化时展开的 glob 等仍有边界限制。[5][7]

SRT 默认广泛可读、写入按许可、网络按域名许可，经 HTTP/SOCKS 代理执行；Linux 无直接外网路径，不遵守代理的程序是连接失败，而非自动旁路。凭据已支持 deny／mask：移除环境值、遮罩文件，代理替换 sentinel；Claude 的文件 mask 自 2.1.221 起，Linux/WSL2 给遮罩副本，macOS 则阻读。HTTPS 替换需实验性 TLS termination 和可信注入目标；提取模式未匹配默认可能跳过，不能当成必然保密。[3][5] SRT 库还有解密请求过滤回调及运行中更新网络配置的接口，但文件边界仍在启动时固定。宽域名许可、危险 Unix socket、宿主后续执行被修改脚本，都可能扩大影响面。

## Codex：审批与权限 profile 分开演进

仅按 OpenAI 官方文档核对，当前 CLI 为 **0.162.1**。macOS 使用 Seatbelt；Linux 自 **0.115** 默认改为 bubblewrap＋seccomp，Landlock 留作兼容路径，WSL1 因此不再支持；WSL2 使用 Linux 路径。原生 Windows 当前优先 **MXC 进程隔离**，不是 Hyper-V microVM：需要 Windows 11 24H2 build **26100.9278** 或 25H2 **26200.9278** 的原生能力；0.162+ 文档提供单命令兼容性探测。MXC 要求托管网络允许本机 loopback，且前台命令结束会终止剩余子进程。[8–10]

旧 Windows `elevated` 后端通过管理员预配置低权限用户、ACL 和防火墙，执行命令本身不升为管理员；`unelevated` 使用当前用户受限 token，网络主要是环境层离线控制，较弱且不能实现 denied-read。Windows 11 为推荐基线，更新后的 Windows 10 ≥1809 仅尽力支持；不能把这一客户端基线当成 MXC 可用性。官方未在这些页给 macOS/Linux 全后端统一的最低 OS／内核版本，不另推断。[10]

旧 `workspace-write` 默认限制写入、关闭命令网络，保护 `.git/.agents/.codex`；它不等于“只能读工作区”。beta 权限 profiles 可组合明确的读、写、拒绝路径及域名规则；管理员 profile 限制的混合版本部署门槛为 0.138。关键陷阱是：开启网络不等于启用代理，未启用 `network_proxy` 或托管代理时，域名规则不限制直连。[9][11]

`on-request`、`never` 及细粒度审批决定何时询问，`never` 本身不移除沙箱；自动 reviewer 也是动作审批层，不是内核边界。获准完整升级可能绕过命令代理；MCP、连接器、浏览器、模型认证等另有边界。扩展以 profiles、执行规则和管理员要求为主；因此不应把命令沙箱视为覆盖任意工具的通用 broker。凭据保护需拒读路径及环境传递策略配合；现行环境文档甚至默认不自动排除名称含 KEY/SECRET/TOKEN 的变量，不能据“有沙箱”宣称凭据不进入命令进程。[9][11][12]

## Docker Sandboxes：当前是独立 microVM 产品

稳定版为 **sbx 0.47.0**；不把 0.48 RC 当稳定版，也不套用早期容器实现或“必须 Docker Desktop”的说明。当前不需宿主 Docker Desktop/Engine，官方安装流程仍要求 Docker 登录，本研究未登录。精确支持：macOS Sonoma **14+、Apple silicon**；Windows **11、x64** 且启用 Windows Hypervisor Platform；Linux **Ubuntu 24.04+、amd64/arm64、KVM**。其他发行版有发布包不等于获官方支持。[13]

Docker 自建 VMM 分别直接使用 Hypervisor.framework、WHP、KVM。每个 sandbox 有独立 Linux 内核及私有 Docker Engine，代理在 VM 内有 sudo；它不是普通 `docker run`，也不是挂宿主 docker.sock 的开发容器。共享工作区走 filesystem passthrough／virtiofs：默认直接挂载实时改宿主文件；clone 模式只读挂宿主仓库、在客体私有克隆中编辑。**只读不等于不泄露**：源挂载仍含未跟踪及被 gitignore 忽略的 `.env`。[14][15]

宿主代理约束出站 TCP；UDP 默认关、开启仍属实验功能，ICMP 阻断，内部 DNS 也执行策略。网络规则已支持域名/IP/端口及 HTTP 方法、路径，不能简化成仅域名表。无匹配许可时请求先被阻，再进入批准流程；组织拒绝不能由本地批准覆盖。[16] 代理管理的 API/OAuth 凭据使用 sentinel、在宿主注入；但宿主有可用 SSH agent 时默认转发，客体可申请签名；OAuth passthrough 或显式传入真值也会缩小保密保证，均须单独审查。[17]

可通过 OCI 模板和 kits 扩展环境、工具、凭据和网络能力。v3 kits 要求 sbx ≥0.45，仍属 Early Access；内置代理快捷名使用 v2，不能混入 v3 mixin。[18] 共享 skills 现行默认只读，但可改读写，形成跨 sandbox 信任连接；文档个别示意仍写默认读写，以 defaults 与 architecture 的一致说明为准。[15][19] MCP gateway 在宿主，注册的本地 stdio MCP 也在宿主而非 VM 内，是明确的可信集成出口。[15]

## Box／Dogwood 的差异在历史语义，不只隔离强度

Box **0.1.0** 发布支持 macOS Apple silicon，Linux 仍列计划；源码有 Linux 模块不等于已支持。它用 Seatbelt 包围代理，工具及本地 MCP 分别有独立 leaf sandbox；可信 broker、Strands Shell、Monty、egress gateway 在外。固定 OS grants 控制直接系统调用，Dogwood 则控制进入这些中介的操作；两者不是同一授权路径。凭据绑定到目的地并由外部 gateway 注入，但绑定本身不授予操作许可。[20–22]

Box 固定依赖 Dogwood **1.0.0**，以 Cedar 扩展的时序规则关联文件、命令、HTTP、MCP 事件，例如读敏感文件后五分钟拒绝联网、成功测试后才允许提交。请求先持久化再决策，历史跨重启保留；损坏、缺失或追加失败拒绝继续，窗口上限为24小时。应区分 attempt 与完成事件：被拒请求也进入 request 历史，并发完成计数不是原子预算。[22][23]

这不是完整系统调用审计，更不是信息流追踪：直接文件读取、leaf 工具自身读写不会进入 Dogwood history；操作与结果记录也非共同事务，记录没有防篡改证明。可信解释器的宽文件 permit 可越过代理自身 grants；Box 没有人工审批 verdict，也不提供资源配额、租户隔离或内核漏洞保护。[21][22][24]

## 设计判断

已产品化的主要路线是“客户端审批＋原生 OS 命令边界”和“整个开发环境置于本地 microVM”；SRT 把前者抽成可复用执行包装，Box 则押注中介化操作和统一持久历史。它们不是同一维度的替代：microVM 改变内核信任面，时序策略回答跨操作许可，审批回答用户授权。选型应先列谁在边界内、哪些事件被中介、哪些秘密仍可被使用；集成失败应缩小接口、申请明确资源或选择能执行原约束的后端，而非关闭 containment。

## 精确主源 URLs

以下编号对应正文；源码固定标签，在线文档按核查日阅读，不能保证未来页面不变。OpenAI 部分仅使用 OpenAI 官方域名。

- [1] OpenAI：审批与沙箱层次、OS 后端、升级及覆盖范围：https://learn.chatgpt.com/docs/agent-approvals-security
- [2] Claude Code 发布标签：https://github.com/anthropics/claude-code/releases/tag/v2.1.296
- [3] Claude 命令沙箱、默认值、失败处理、可信配置及凭据 mask：https://code.claude.com/docs/en/sandboxing
- [4] Claude 精确客户端 OS 基线及 WSL 对照：https://code.claude.com/docs/en/setup
- [5] SRT 0.0.79 实现说明、依赖、平台限制、代理及扩展接口：https://github.com/anthropics/sandbox-runtime/blob/v0.0.79/README.md
- [6] 已读 SRT Linux 后端源码，尤其 seccomp 依赖检查、命名空间及命令包装：https://github.com/anthropics/sandbox-runtime/blob/v0.0.79/src/sandbox/linux-sandbox-utils.ts
- [7] 已读 SRT Windows 后端源码，独立用户、WFP 验证、ACE 及启动语义：https://github.com/anthropics/sandbox-runtime/blob/v0.0.79/src/sandbox/windows-sandbox-utils.ts
- [8] OpenAI 官方版本日志，2026-10-09 CLI 0.162.1：https://learn.chatgpt.com/docs/changelog
- [9] OpenAI 当前本地沙箱、审批与代理安全文档：https://learn.chatgpt.com/docs/agent-approvals-security
- [10] OpenAI Windows MXC 精确构建门槛与 legacy 后端：https://learn.chatgpt.com/docs/windows/windows-sandbox
- [11] OpenAI beta profiles、0.138 管理门槛、代理开关及覆盖范围：https://learn.chatgpt.com/docs/permissions
- [12] OpenAI 环境传递策略及默认排除行为：https://learn.chatgpt.com/docs/config-file/config-advanced#shell-environment-policy
- [13] Docker 精确安装支持矩阵：https://docs.docker.com/ai/sandboxes/install/；稳定发布：https://github.com/docker/sbx-releases/releases/tag/v0.47.0
- [14] Docker 隔离层及 clone 模式的可读范围：https://docs.docker.com/ai/sandboxes/security/isolation/；自建 VMM 技术说明：https://www.docker.com/blog/why-microvms-the-architecture-behind-docker-sandboxes/
- [15] Docker 深层架构、virtiofs、持久化与宿主 MCP：https://docs.docker.com/ai/sandboxes/architecture/
- [16] Docker 本地网络与 HTTP 策略、批准语义：https://docs.docker.com/ai/sandboxes/governance/access-controls/network/
- [17] Docker 凭据代理、OAuth passthrough、绑定与 SSH agent：https://docs.docker.com/ai/sandboxes/configuration/credentials/
- [18] Docker kits v3／v2 兼容矩阵：https://docs.docker.com/ai/sandboxes/customize/；作者扩展接口：https://docs.docker.com/ai/sandboxes/customize/author/
- [19] Docker 当前默认安全姿态：https://docs.docker.com/ai/sandboxes/security/defaults/；共享 skills 的跨环境边界：https://docs.docker.com/ai/sandboxes/workflows/agent-skills/
- [20] Box 发布平台范围：https://github.com/strands-agents/box/releases/tag/v0.1.0
- [21] Box 两种授权路径、leaf 与可信进程：https://github.com/strands-agents/box/blob/v0.1.0/docs/design/containment.md
- [22] Box／Dogwood 时序、持久历史及其限制：https://github.com/strands-agents/box/blob/v0.1.0/docs/design/policy.md；凭据绑定：https://github.com/strands-agents/box/blob/v0.1.0/docs/design/credentials.md
- [23] 已读固定依赖源码：https://github.com/strands-agents/box/blob/v0.1.0/crates/policy/Cargo.toml
- [24] Box 明示不保护项：https://github.com/strands-agents/box/blob/v0.1.0/docs/design/limitations.md
