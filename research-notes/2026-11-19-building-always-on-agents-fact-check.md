# Always-on agents：三项生产边界事实核查

核查日期：2026-11-19。范围仅限生产保障；不重复原文分析，也不审计其具体实现。以下依据五份官方文档，将来源事实与工程推论分开。

## 1. 持久化 inbox 的重试，不等于外部副作用恰好执行一次

**来源事实：** Temporal 强烈建议可能重试的 Activity 具备幂等性，并明确举例：Activity 执行成功后，Worker 在通知服务前崩溃，重试可能造成重复扣款或创建重复资源（S1）。AWS Lambda durable execution 默认采用 at-least-once step 语义：已完成并 checkpoint 的 step 不会在重放时再次执行，但完成前失败或中断的 step 可能执行多次；官方要求业务逻辑幂等，并建议使用幂等键（S2）。

**工程推论：** 即使 inbox 能可靠保存、重投事件，也不能消除“外部写入成功，但本地尚未记录成功”的故障窗口。发邮件、支付、创建工单等操作需要各自的稳定幂等键与下游保障；只对 inbox 消息去重不足以证明所有外部效果恰好一次。

**限定：** S2 的 execution name 可以避免重复启动同一个 durable execution，但文档仍单独要求 step 业务逻辑幂等。执行启动去重、已完成 step 的重放跳过与外部副作用去重，是不同层面的保障。

## 2. timer 状态推进与 inbox 插入，需要原子性或等效的持久恢复机制

**来源事实：** AWS transactional outbox 文档描述了数据库更新与消息发送的双写不一致：一项成功、另一项失败会导致状态与通知脱节。其方案把业务更新与 outbox 记录放入同一事务，再异步发布；发布仍可能重复，因此消费者需要幂等处理（S3）。

**工程推论：** 将这一原理用于调度器：若先推进 `next_due` 再插入 inbox，二者之间崩溃可能漏掉该次触发；反向执行而缺少去重，则可能重复触发。可以采用同一数据库事务，或持久化触发意图、可恢复重放及稳定的触发标识（例如 timer ID + 原定触发时间）。关键是故障恢复后能找回每次到期触发，而不只是重试已经进入 inbox 的消息。

**限定：** 官方来源论证的是双写问题；上述 timer/inbox 映射是工程推论，不是 AWS 对原文实现的判断。若实现已有事务、日志或等效恢复机制，该风险可能已被覆盖。

## 3. 凭证注入／代理本身，不构成动作授权

**来源事实：** Anthropic 安全部署文档将 credential injection proxy 用于让 agent 在不接触原始凭证的情况下调用服务；同时另行列出端点 allowlist、最小 IAM 权限与代理请求验证等控制（S4）。Agent SDK 权限文档则单独提供工具级控制：permission modes、allow/deny rules、hooks 与 `canUseTool` 运行时决策（S5）。

**工程推论：** 凭证注入解决的是秘密暴露与请求携带身份的问题，并不自动回答“该 agent 是否获准对该资源执行该动作”。需要由可信的服务端／代理／工具权限层落实动作、参数、资源与租户范围约束，必要时要求人工批准；仅允许访问某域名，也不能证明域名下所有操作都获授权。

**限定：** 代理完全可以兼任授权执行点，但必须确实配置并执行相应策略。“代理可用于授权”与“用了代理便已授权”是两回事。此外，S5 明确指出已被前序步骤自动批准的调用会跳过 `canUseTool`；必须覆盖每次工具调用的检查应使用 `PreToolUse` hook，不能默认该 callback 是不可绕过的授权关卡。

## 官方来源（精确 URL）

- S1 — Temporal：Python SDK error handling / Activity 幂等性  
  https://docs.temporal.io/develop/python/best-practices/error-handling
- S2 — AWS Lambda：Durable execution idempotency  
  https://docs.aws.amazon.com/lambda/latest/dg/durable-execution-idempotency.html
- S3 — AWS Prescriptive Guidance：Transactional outbox pattern  
  https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html
- S4 — Anthropic：Securely deploying AI agents  
  https://platform.claude.com/docs/en/agent-sdk/secure-deployment
- S5 — Anthropic：Configure permissions  
  https://platform.claude.com/docs/en/agent-sdk/permissions
