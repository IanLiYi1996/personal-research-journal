# Always-on agent：运行时参考项目

核查日期：2026-11-20 UTC。范围仅为公开上游仓库与官方文档；未检索私有账户、未检查 agent-memory-kit 或其他 memory 库，未安装包、运行第三方代码、提交或推送。以下架构映射与选型建议属于本次分析，不是项目官方承诺。

## 1. Temporal — 重型持久执行依赖候选（GitHub）

- **身份与证据**：官方 `temporalio/temporal` 是 Temporal **服务端**源码。README 明确描述持久工作流、处理间歇性失败和失败操作重试，并将应用侧 Workflows / Activities / Workers 指向语言 SDK。
- **对当前 bootstrap 的价值**：当 Inbox / Timers 触发的任务需要可靠失败恢复时，可评估把长任务交给 Temporal；不是借用一个 memory 层。
- **重要限制**：须引入独立 Temporal 服务及应用侧 SDK / Worker，不能把该服务端仓库当作可直接嵌入现有循环的轻量库。是否值得增加这层基础设施，应由任务可靠性需求决定。
- **许可证**：所核查仓库为 MIT；若复制其代码，应保留版权与许可文本。其他 SDK 仓库须另行核查。
- **源码版本**：`a8f8328b001bd3302e658cdd062e18a5e92cf50f`，提交时间 `2026-11-19T23:11:18Z`。

来源：

- 仓库：https://github.com/temporalio/temporal
- 固定版本 README（功能与运行模型）：https://github.com/temporalio/temporal/blob/a8f8328b001bd3302e658cdd062e18a5e92cf50f/README.md
- 固定版本许可证：https://github.com/temporalio/temporal/blob/a8f8328b001bd3302e658cdd062e18a5e92cf50f/LICENSE

## 2. LangGraph — 状态工作流与审批恢复依赖候选（GitHub）

- **身份与证据**：官方 `langchain-ai/langgraph` 是 LangChain Inc 的低层有状态 agent 编排框架；README 列出 durable execution 与 human-in-the-loop，并明确可脱离 LangChain 使用。
- **具体机制**：官方 Interrupts 文档要求配置 checkpointer 与稳定的 `thread_id`。节点调用 `interrupt()` 保存状态并暂停，以同一线程提交 `Command(resume=...)` 恢复；生产使用应配置持久 checkpointer，而不是示例中的内存实现。适合承载 Permissions → 等待人工决定 → 继续执行。
- **重要限制**：恢复不是从任意代码行接着跑，而是从被中断节点开头重新执行；`interrupt()` 前的副作用会再次发生，必须幂等或移到审批之后。这不自动替代 Outbox 的外部副作用去重。
- **许可证**：所核查仓库为 MIT；若复制代码，保留版权与许可文本。不据此推断其他部署产品的许可证。
- **源码版本**：`58a6e1cc5b7dfc3619fc6e7e7573b040d554649b`，提交时间 `2026-11-20T01:00:31Z`。下列官方文档为实时页面，未取得与该 commit 对应的文档修订。

来源：

- 仓库：https://github.com/langchain-ai/langgraph
- 固定版本 README：https://github.com/langchain-ai/langgraph/blob/58a6e1cc5b7dfc3619fc6e7e7573b040d554649b/README.md
- 审批、checkpoint 与节点重跑规则：https://docs.langchain.com/oss/python/langgraph/interrupts
- 固定版本许可证：https://github.com/langchain-ai/langgraph/blob/58a6e1cc5b7dfc3619fc6e7e7573b040d554649b/LICENSE

## 3. GitLab — 后台作业幂等/去重的模式参考（真正托管在 GitLab）

- **身份与证据**：公开官方主项目为 `gitlab.com/gitlab-org/gitlab`，不是 GitHub 镜像。其 Sidekiq 开发文档描述失败重试、`idempotent!`、重复执行测试及两类去重策略：`until_executing` 在执行前释放锁，`until_executed` 在执行结束后释放锁；`reschedule_once` 可在去重后追加一次运行。
- **对当前 bootstrap 的价值**：研究这些机制用于 webhook Inbox 的重复事件处理和 Outbox 重试；尤其借鉴“执行两次仍验证真实副作用”的测试方式。此处是模式迁移建议，未核查或声称 GitLab webhook Worker 本身具有恰好一次保证。**不建议把整个 GitLab 产品作为 agent 运行时依赖。**
- **重要限制**：Redis 去重锁不能保证恰好一次。固定版本文档说明，异常遗留的键可能阻塞新作业直到 TTL 到期；当前默认 TTL 为 **10 分钟**，到期后又可能出现重复作业。因此仍需业务幂等，不应照搬锁机制作为副作用正确性保证。
- **许可证边界**：根 LICENSE 明确区分目录：一般代码为 MIT Expat，`doc/` 为 CC BY-SA 4.0，`ee/` 与 `domains/<name>/ee/` 等适用另行许可证，第三方组件沿用原许可。借代码应逐文件检查；不能把整个仓库或文档视为 MIT。
- **源码版本**：`2c1f3cacae4b908894a281a70f45f87f168960ac`，提交时间 `2026-11-20T03:46:54+01:00`。上述机制与 TTL 已对照同版本文档原文。

来源：

- 官方 GitLab 仓库：https://gitlab.com/gitlab-org/gitlab
- 官方在线文档：https://docs.gitlab.com/development/sidekiq/idempotent_jobs/
- 固定版本文档：https://gitlab.com/gitlab-org/gitlab/-/blob/2c1f3cacae4b908894a281a70f45f87f168960ac/doc/development/sidekiq/idempotent_jobs.md
- 固定版本许可证：https://gitlab.com/gitlab-org/gitlab/-/blob/2c1f3cacae4b908894a281a70f45f87f168960ac/LICENSE

## 合并时的建议

现阶段先研究 GitLab 的作业幂等测试和 LangGraph 的审批恢复边界；若要新增主运行时，再按基础设施承受能力评估 LangGraph 或 Temporal，不建议未经验证就同时叠加两者。Memory-kit 仍由主代理单独核查，本笔记不作相关判断。

调研限定为 3 个项目、约 5 个核心仓库/文档页面，另核对上述许可证和公开 commit 元数据；未扩展到其他候选。
