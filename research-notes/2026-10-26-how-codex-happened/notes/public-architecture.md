# How Codex Happened：公开架构核查笔记

查阅／截止日：2026-10-26。本文只核实六个概念，不审计全部代码，不覆盖 Every 正文、原访谈或模型／产品发布日期。技术证据仅来自 OpenAI 文档、官方工程博客和 `openai/codex`；未安装或运行 agent，未读取或复制用户环境配置。

公共仓库快照：`970111f6cb62cc02fe36a82ea7d71b37a2889c97`，提交时间为 2026-10-23 18:50:54 UTC。通过 GitHub API 查询 `main`、截止 2026-10-26 返回；这是可定位的源码快照，不是产品发布版本。[S12]

## 六个关键概念

1. **执行内核与产品界面分层。** 官方工程博客把模型交互、工具执行、状态管理归于共享 harness／`codex-core`，App Server 提供客户端接口，IDE／app 负责交互呈现。公开 Rust crate 的 README 也称 core 实现业务逻辑，可供多个 Rust UI 使用。博客在 2026-02-04 仍把 TUI 接入 App Server 写为后续计划；快照中的 TUI 已依赖其 client、daemon、protocol，故不能将旧博客的接线方式当成今天的静态事实。[S1][S2][S3] 这支持“共享内核、不同入口”，不证明整个桌面应用或云端服务全部开源。

2. **事件协议不是普通问答接口。** App Server 文档描述 JSON-RPC 2.0 语义；stdio 使用逐行 JSON，WebSocket 使用消息传输但仍标为实验性、不受支持，实际信封省略 `jsonrpc` 字段。thread 是会话容器，turn 是一轮工作，item 是消息、命令、文件修改等工作单元。客户端提交 `turn/start`，接收 `item/started`、文本增量、`item/completed`、`turn/completed`，并回应服务端审批请求。[S4] 分析：UI 可以重建进行中的工作；这条客户端事件流不应与模型服务侧的流式响应混为一谈。

3. **sandbox 与 approval 是两条控制轴。** 沙箱限制执行时能访问或修改的文件、网络等资源；审批策略决定何时需要额外授权／审查。官方安全文档分别解释两者，core README 列出不同操作系统的执行约束。[S2][S5] 因而“不询问”不必然代表“无沙箱”，同意一次操作也不代表解除所有限制。它们是权限执行机制，不是提示词；本地命令的沙箱证据也不能自动推广为所有远程工具的完整安全保证。

4. **worktree 隔离工作副本，支持并行但不消灭冲突。** app 文档说明可在独立 Git worktree 中处理任务，避免与主要工作目录互相干扰，并通过移交工作继续处理变更。[S6] 分析：它隔离的是文件工作副本，仍属于同一仓库，不等于容器或权限沙箱；多个任务修改同一逻辑仍需整合、审查。这里可核实并行任务的工作区机制，不能据此推断所有入口的子代理都采用同一种协作拓扑。

5. **AGENTS.md 与 skills 是上下文工程。** 官方文档说明项目指令按从根目录到当前目录的路径发现、合并，较近目录的内容可覆盖上层；skills 则先提供名称和描述等元数据，使用时才加载完整指令，支持显式或隐式选择。[S7][S8] 分析：前者为任务提供分层约束，后者按需注入流程知识，减少无关上下文。这不是给模型重新训练权重，也不能代替实际权限或安全边界；不宜声称每轮都会完整读取全部技能。

6. **compaction 延续上下文，测试形成反馈。** 官方 harness 文章解释长任务需要压缩历史以继续工作，并区分早期摘要与 `/responses/compact` 返回的加密压缩项；不能把所有路径都说成可读摘要。公开 `compact.rs` 可见构建替代历史、更新历史与 token 统计，并发出压缩 item 的开始／完成事件。[S9][S10] 官方长任务文章描述实施中运行验证；仓库另有压缩、审批和命令执行测试。[S11][S13] 分析：模型可根据命令和测试结果继续调整，形成“修改—验证—再修改”闭环；压缩不是无损永久记忆，测试文件的存在也不证明每次任务都自动运行测试或必然通过。

## 事实／分析与未知项

上文接口、crate 职责、依赖项及文档规则是可定位事实；关于分层价值、上下文工程、并行整合和反馈闭环的解释属于分析。没有证据确认每项机制的首次引入时间、各商业入口实际部署的精确 SHA、压缩的信息保真率，或所有任务的测试覆盖与协作策略。本次未运行代码，不能报告测试通过或进行安全认证。尤其不能用查阅日的公共实现，倒推这些细节在 2025 年已存在；历史叙事必须另找当时文档或提交证据。

## Source 表

日期规则：文档正文未显示发布／更新时间的，明确列为查阅日；不是功能上线日。源码按上述提交快照固定。部分 developers 文档入口现重定向至官方 ChatGPT Learn，已实际读取落地正文。链接均为一手来源，GitHub API 仅用于取得提交标识。

| 编号 | 标题／具体定位 | 日期 | URL |
| --- | --- | --- | --- |
| S1 | Unlocking the Codex harness: how we built the App Server | 发布 2026-02-04 | https://openai.com/index/unlocking-the-codex-harness/ |
| S2 | codex-core README：业务逻辑与平台沙箱 | 快照 2026-10-23 | https://github.com/openai/codex/blob/970111f6cb62cc02fe36a82ea7d71b37a2889c97/codex-rs/core/README.md |
| S3 | codex-tui Cargo.toml：App Server 依赖 | 快照 2026-10-23 | https://github.com/openai/codex/blob/970111f6cb62cc02fe36a82ea7d71b37a2889c97/codex-rs/tui/Cargo.toml |
| S4 | Codex App Server：协议、生命周期与审批 | 未显示发布日；查阅 2026-10-26 | https://developers.openai.com/codex/app-server/ |
| S5 | Agent approvals & security：sandbox 与 approval | 未显示发布日；查阅 2026-10-26 | https://developers.openai.com/codex/agent-approvals-security/ |
| S6 | Worktrees | 未显示发布日；查阅 2026-10-26 | https://developers.openai.com/codex/app/worktrees/ |
| S7 | Custom instructions with AGENTS.md | 未显示发布日；查阅 2026-10-26 | https://developers.openai.com/codex/guides/agents-md/ |
| S8 | Build skills：渐进加载与显式／隐式调用 | 未显示发布日；查阅 2026-10-26 | https://developers.openai.com/codex/skills/ |
| S9 | Unrolling the Codex agent loop：工具反馈与 compaction | 发布 2026-01-23 | https://openai.com/index/unrolling-the-codex-agent-loop/ |
| S10 | core/src/compact.rs：替代历史与压缩事件 | 快照 2026-10-23 | https://github.com/openai/codex/blob/970111f6cb62cc02fe36a82ea7d71b37a2889c97/codex-rs/core/src/compact.rs |
| S11 | Run long horizon tasks with Codex：验证与修复循环 | 发布 2026-02-23 | https://developers.openai.com/blog/run-long-horizon-tasks-with-codex |
| S12 | openai/codex commit：查询返回的公共快照 | 提交 2026-10-23；查阅 2026-10-26 | https://github.com/openai/codex/commit/970111f6cb62cc02fe36a82ea7d71b37a2889c97 |
| S13 | core/tests/suite：compact.rs、approvals.rs、unified_exec.rs | 快照 2026-10-23 | https://github.com/openai/codex/tree/970111f6cb62cc02fe36a82ea7d71b37a2889c97/codex-rs/core/tests/suite |

采集方式：通过网页搜索／读取工具（本会话暴露为 `_web_run`）实际打开官方开发者文档、官方博客及 GitHub README／源码；另用只读 HTTPS 请求补取固定 SHA 的 README、依赖清单、压缩／测试代码片段和提交元数据。没有将搜索摘要单独作为技术事实依据。
