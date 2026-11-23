# 个人微信接入：账号与授权边界核查

核查日期：2026-11-20（UTC）。正确上游为 `Tencent/openclaw-weixin`，不是不存在的 `Tencent/Weixin-ClawBot`。固定提交：[`0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e`](https://github.com/Tencent/openclaw-weixin/commit/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e)（提交日期 2026-11-09）。仅阅读 README、绑定升级说明、授权辅助函数这 **3 个文件**；未检查协议、运行代码或登录。该提交的绑定升级说明标注为 `3.0.0-beta-0` 预发布，不能视为稳定版或所有微信账号的可用性承诺。[来源](https://github.com/Tencent/openclaw-weixin/blob/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e/docs/release-notes-bot-realname-auth_zh_CN.md#L1-L5)

## 核查结论（5 项）

1. **支持的交互模型是个人微信用户连接、聊天于 ClawBot。** README 描述绑定本人或他人的微信 ClawBot；升级说明明确微信用户目标识别为“私聊”。这些证据支持“用户 ↔ Bot”的设计，**未证明 Bot 可以冒充用户的整个个人账号、读取全部好友聊天或向全部好友代发**；这些权限在所读文件中未知。[README](https://github.com/Tencent/openclaw-weixin/blob/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e/README.md#L88-L129)、[私聊说明](https://github.com/Tencent/openclaw-weixin/blob/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e/docs/release-notes-bot-realname-auth_zh_CN.md#L58-L68)

2. **绑定操作者不一定就是实际连接用户。** 当前 `promptBindOwner` 默认关闭；开启后，本人 Bot 直接扫码绑定；他人 Bot 则先由操作者用本人微信完成实名授权，再由实际连接用户扫码绑定。PIN 校验是否实际触发由服务端决定。这是绑定流程，不能据此把实名操作者自动当作聊天发送者或高风险操作审批人；资格、地区及账号开通范围未在所读文件中明确。[绑定与 PIN 说明](https://github.com/Tencent/openclaw-weixin/blob/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e/docs/release-notes-bot-realname-auth_zh_CN.md#L9-L43)

3. **授权辅助函数体现“消息准入”和“命令授权”的区别。** `ownerId` 非空时，只有 `senderId === ownerId` 才允许发送者；缺少或空白 `ownerId` 时，`senderAllowed` 为真，但需要计算命令授权时，`commandAuthorized` 为假。该函数没有配置式多人 allowlist；不能外推整个插件不存在其他策略。`ownerId` 的获取、持久化及与实名操作者的映射，以及所有调用方是否完整执行该结果，本次未核查。[完整授权函数](https://github.com/Tencent/openclaw-weixin/blob/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e/src/messaging/authorization.ts#L3-L25)

4. **群聊支持与具体限制：未知。** README 提到私聊会话隔离，升级说明确认微信用户目标属于私聊；两者均未明确声明群聊是否支持、群成员权限或群内审批规则。不能把私聊证据写成“明确不支持群聊”，也不能承诺群聊可用。[会话隔离](https://github.com/Tencent/openclaw-weixin/blob/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e/README.md#L67-L79)、[目标识别](https://github.com/Tencent/openclaw-weixin/blob/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e/docs/release-notes-bot-realname-auth_zh_CN.md#L58-L68)

5. **定时提醒与主动出站的支持条件：未知。** README 声明可收发消息，升级说明记录文字、媒体收发联调；但所读三文件未明确承诺定时提醒、无用户先发消息时的主动发送、可发送对象范围、有效期或配额。普通收发、错误提示及恢复发送不等于“任意时间向任意个人账号主动推送”。[收发声明](https://github.com/Tencent/openclaw-weixin/blob/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e/README.md#L8-L17)、[联调范围](https://github.com/Tencent/openclaw-weixin/blob/0ad1ae91347a938f52f1e4e2abd6dcda51e05f9e/docs/release-notes-bot-realname-auth_zh_CN.md#L128-L134)

**设计建议，非上游保证：** 将个人微信定位为待验证资格的 Bot 私聊入口，而非个人账号托管。高风险操作须由应用层单独校验“绑定 Bot/渠道账号＋已核实的审批人发送者 ID”，配置明确的审批白名单，并将确认绑定到具体操作；身份缺失或不匹配时拒绝审批。扫码绑定、实名完成或消息能够进入均不等同于高风险批准。群聊、提醒和主动发送先保留为待协议侧验证的能力条件。
