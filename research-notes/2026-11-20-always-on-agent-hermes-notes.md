# Hermes：always-on agent 架构参考核对

核对日期：2026-11-20（UTC）；仅针对 NousResearch/hermes-agent，不涉及微信协议。
版本：[GitHub HEAD 固定快照](https://github.com/NousResearch/hermes-agent/commit/9f47677483d1b4de4d1a88d37b3b7ca18b18b215)：`9f47677483d1b4de4d1a88d37b3b7ca18b18b215`，提交时间 2026-11-19 23:18:10 UTC。
范围：抽查 6 个架构代码/文档文件，另检查根 LICENSE；只读上游文本，未安装、运行代码、登录或做故障测试。
证据等级：下文明确区分“代码核实”和“文档契约”；建议属于设计判断，不代表已选技术栈或获准构建应用。

## 1. Gateway / adapter 合同与标准化事件

- 代码核实：`BasePlatformAdapter` 要求实现 `connect`、`disconnect`、`send`；发送返回 `SendResult`，含成功、消息 ID、错误类别和重试提示，不是仅返回文本。[1]
- 消息入口接受 `MessageEvent`；另有标准化平台事件入口（稳定字典 + 内部 `SessionSource`）。后者将授权及插件派发交给 gateway runner，未注册回调时按接口契约关闭入口。[1]
- `SessionSource` 将平台、聊天、用户、线程、scope、profile 与显示名称分开；relay 信任标记不经 `to_dict` / `from_dict` 传输，不能由普通序列化载荷直接授予。[2]
- 借鉴：保留小型渠道接口、标准事件信封、独立发送结果与能力声明；身份与授权决策放在宿主入口，而不是交给模型根据消息文本判断。[1][2]
- 避免整套照搬：基类已经包含媒体、语音、忙时合并、中断及平台特例；本项目可借职责划分，不必继承全部运行时行为。[1]
- 具体限制：适配器自身的 pending、debounce、background task 等集合是进程内状态；接口统一不等于入站消息已持久化或崩溃后可重放。[1]

## 2. Sessions、长期 memory 与当前 context 分层

- 代码核实：`build_session_key` 按 profile 命名空间、平台、聊天类型、聊天/线程及可选用户隔离；群组默认按用户隔离，线程默认共享，不能笼统称“每个人完全独立”。`SessionStore` 声明 SQLite 存储接口及旧 JSONL 回退。[2]
- 当前会话 context 是单独构造的系统提示段；对名称/话题的“不可信”提醒属于模型指令，不是身份认证。路由原始 ID 与模型可见的可选脱敏 ID 分离。[2]
- 文档契约：`MEMORY.md` / `USER.md` 是 profile 范围的小容量跨会话记忆，默认上限 2,200 / 1,375 字符；会话开始时冻结注入。历史消息另存 SQLite，可用 `session_search` 按需检索。[3]
- 借鉴：分开存储会话路由/历史、经整理的长期事实、每轮上下文；群组共享与个人记忆权限应显式定义。[2][3]
- 避免整套照搬：不要把共享 profile 的个人记忆当成多用户隔离存储，也不要把聊天重启等同于新会话；官方明确警告不要让两个 agent 进程共写同一 home。[3]
- 具体限制：本轮更新的记忆不会刷新本会话的冻结系统提示快照，通常要到新会话才注入；“已记住”文本也不证明 `memory` 工具真的写入。[3]
- 背景任务另有限制：文档明确说明本地模型延后执行的 memory review 队列为内存态，退出会丢弃未执行项；它不是持久任务队列。[3]

## 3. 定时任务、执行与渠道投递

- 文档契约：任务定义写入 profile 的 `cron/jobs.json`（临时文件后 rename）；默认触发器是后台线程内的 `InProcessCronScheduler`，每 60 秒 tick，跨进程文件锁避免重叠扫描；不是外部 durable workflow engine。[4]
- 不能误写为“所有状态都在内存”：当前文档已有 occurrence / execution 账本、fire claim 与恢复策略；带 `execution_id` 的 agent 结果进入 `cron/deliveries.db` 持久投递队列，由 gateway housekeeping 经在线适配器发送。[4]
- 投递目标支持 `origin`、渠道 home、显式 chat/thread、`local`；普通 cron 用新会话，无前次对话历史，但会加载长期记忆。普通渠道投递不镜像到目标聊天历史；Bot Chat 是另行描述的入站任务例外。[4]
- 结果状态分开：`delivery_queued` 仅表示等待投递，不能报告“用户已收到”；执行失败、投递失败、发送结果未知都需单独呈现。[4]
- 借鉴：拆开“何时触发 / 本次执行 / 如何投递”；用执行 ID、投递账本和明确状态处理恢复，任务 prompt 自包含。[4]
- 避免整套照搬：不要直接采用其所有 profile/relay/托管调度拓扑；可插拔触发器只管触发，不能据此推导执行与外部副作用都具备持久事务保证。[4]
- 具体限制：发送中途 claiming gateway 死亡的队列项会被标为 `unknown` 且不重试；这是避重与可能漏送的取舍，不是端到端 exactly-once。[4]
- 验证边界：本轮对 cron 的上述核对来自官方内部文档，未逐项检查 scheduler、ledger、CAS 实现；即使文档称 occurrence at-most-once，也不应承诺业务副作用恰好执行一次。[4]

## 4. 审批与认证/授权边界

- 代码核实：gateway 的准入判断在 `_principal_authorized`，由 pairing、平台/全局 allowlist、角色或显式开放配置决定；常规用户无 ID 则拒绝。relay、webhook、`trusted_inbound` 等走受信上游分支，不是重新验证所有入口的万能认证器。[5]
- 与提示词分开：`SessionSource` 中的 trust 标记和准入分支是程序决策；“名称/话题不可信”的 context 提醒是模型行为约束，不能取代程序授权。[2][5]
- 文档契约：危险命令审批可为 `smart`（辅助 LLM 判断）、`manual` 或 `off`；cron / unattended 默认 `deny`，等待超时拒绝。hardline blocklist 是文档声明的代码级底线，连 YOLO 也不绕过；本轮未审其检测实现。[6]
- 审批载体有独立 `request_id`、`session_key` 和操作选项；共享聊天默认 `approvers: ["*"]` 允许已准入成员答审批，不能默认推导成“仅任务发起人可批准”。[1][6]
- 借鉴：区分渠道认证、调用者准入、工具执行权限与本次审批；审批绑定具体请求和身份，后台无人工可答时按确定性规则拒绝。[1][5][6]
- 避免整套照搬：不要将 smart LLM、shell 模式扫描或“请勿执行”提示当作硬沙箱；也不要直接照搬 allow-all、YOLO、永久模式 allowlist 与共享聊天审批默认值。[6]
- 具体限制：通用 terminal 仍以同一 OS 用户运行，可绕过文件工具写保护；`memory.write_approval` 同样只覆盖 memory 工具，不阻止前台通用文件/terminal 直接改记忆。真正能力隔离仍需 OS 权限、受限凭证和执行环境。[3][6]
- 认证核验边界：授权代码注释将 webhook HMAC、relay 上游认证作为前提；本轮未检查相应传输实现，因此不能声称所有渠道认证已通过安全审计。[5]

## 采用结论与许可证

适合借鉴的是四条职责边界：渠道适配、会话/记忆分层、触发/执行/投递分层、准入/审批/执行能力分层。保持技术选择开放；Hermes 是参考实现，不是可以直接代入生产保证的整体方案。
根 LICENSE 是 MIT，版权注明 Nous Research（2025）；复制代码或实质部分需保留版权及许可声明，并有无担保条款。本轮没有核查依赖、插件或其他资产许可证。[7]

## 固定版本一次资料

[1]: https://github.com/NousResearch/hermes-agent/blob/9f47677483d1b4de4d1a88d37b3b7ca18b18b215/gateway/platforms/base.py
[2]: https://github.com/NousResearch/hermes-agent/blob/9f47677483d1b4de4d1a88d37b3b7ca18b18b215/gateway/session.py
[3]: https://github.com/NousResearch/hermes-agent/blob/9f47677483d1b4de4d1a88d37b3b7ca18b18b215/website/docs/user-guide/features/memory.md
[4]: https://github.com/NousResearch/hermes-agent/blob/9f47677483d1b4de4d1a88d37b3b7ca18b18b215/website/docs/developer-guide/cron-internals.md
[5]: https://github.com/NousResearch/hermes-agent/blob/9f47677483d1b4de4d1a88d37b3b7ca18b18b215/gateway/authz_mixin.py
[6]: https://github.com/NousResearch/hermes-agent/blob/9f47677483d1b4de4d1a88d37b3b7ca18b18b215/website/docs/user-guide/security.md
[7]: https://github.com/NousResearch/hermes-agent/blob/9f47677483d1b4de4d1a88d37b3b7ca18b18b215/LICENSE
