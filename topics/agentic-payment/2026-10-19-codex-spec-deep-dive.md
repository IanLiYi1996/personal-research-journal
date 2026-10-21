# Agentic Payment 协议规范与源码深挖：离线源码核查与证据缺口

研究日期：2026-10-19（UTC）。

状态：**部分完成。AP2、ACP、x402 已取得离线浅克隆并开展源码核查；最新上游默认分支、新增独立仓库和远端元数据仍无法确认。**

## 1. 结论与证据边界

第一阶段直接通过 GitHub HTTPS 获取源码的八条浅克隆命令全部失败，终端均报告 `Could not resolve host: github.com`，见 E1–E8。第二阶段发现三个既有本地浅克隆，通过 `git clone --depth 1 file:///...` 成功复制到任务指定的独立新目录；这些源码可以固定 SHA，但不能证明其等于 2026-10-19 的最新上游默认分支。下文严格区分“离线快照结论”和“仍无法确认”。

没有将搜索摘要、GitHub 网页或记忆中的协议内容替换成“已核查的克隆源码”。网页入口仅用于保留研究线索，不构成协议实现或规范的核查结论。离线快照中的每个结论采用 `路径@AP2`、`路径@ACP` 或 `路径@X` 引用；别名固定到下表的完整 SHA，不代表浮动分支。

本次没有执行、安装或构建任何目标仓库内容，没有修改任何项目代码。本次会话在项目中唯一写入的文件是本报告；此外仅执行任务授权的克隆及创建其父目录。既有本地快照只读使用，没有修改。

| 引用别名 | 指定目录内的克隆 | 固定提交 SHA | 提交时间（原时区） | 新鲜度限制 |
| --- | --- | --- | --- | --- |
| AP2 | `/tmp/codex-payment-repos/ap2` | `e1ea56db72a6385bce3e5c1112b3a56ce60acb43` | `2026-04-29T09:51:41-07:00` | 不等于确认了最新远端 |
| ACP | `/tmp/codex-payment-repos/acp` | `7fdd78df677a94dce04c770644b0fbbb1401272b` | `2026-07-17T21:51:51-07:00` | 不等于确认了最新远端 |
| X | `/tmp/codex-payment-repos/x402` | `faa1c694c66a3732aaabbce93bdae140e521a67d` | `2026-10-15T09:58:51-07:00` | 不等于确认了最新远端 |

SHA 和提交时间来自各克隆的 Git HEAD 提交对象（`git log -1 --format='%H%n%cI%n%s'`），不是规范发布日期。离线来源分别为 `/tmp/payment-repos/AP2`、`/tmp/payment-repos/ACP`、`/tmp/payment-repos/x402`；目标克隆的 `origin` 因本地传输指向这些目录，不能伪装为直接从 GitHub 成功获取。

### 阅读摘要

- AP2 快照的 V2 规范使用 open/closed Checkout / Payment Mandate，源码同时保留旧 Intent / Cart / Payment 和 W3C 风格模型；不能混成同一代消息。详见 §4.1。[`CHANGELOG.md@AP2`；`docs/ap2/specification.md@AP2`；`code/sdk/python/ap2/models/mandate.py@AP2`。]
- ACP 快照标注的 latest stable 是 2026-04-17，但日期目录存在后续追补；checkout、vault、webhook 的 schema 与示例也有差异。详见 §4.2。[`README.md@ACP`；`changelog/unreleased/unwrap-error-envelope-released-delegate-payment.md@ACP`；`spec/2026-04-17/openapi/openapi.agentic_checkout.yaml@ACP`。]
- x402 v2 使用 PAYMENT-REQUIRED / PAYMENT-SIGNATURE / PAYMENT-RESPONSE；X-PAYMENT / X-PAYMENT-RESPONSE 是 v1。核心类型、scheme / network、transport 应分层理解，不能把所有scheme描述成同一种链上付款。详见 §4.3。[`specs/transports-v1/http.md@X`；`specs/transports-v2/http.md@X`；`specs/x402-specification-v2.md@X`。]
- 独立 TAP、IETF、MCP 与 A2A扩展仓库仍未成功获取；仅保留定位与未确认事项。x402 / ACP 自己的 MCP 适配另有本地证据，但不替代独立 MCP SEP。详见 §4.4–4.5；获取证据为 E2、E5–E8。

### 1.1 本应满足的证据标准

- 每个协议结论应附仓库内文件路径和实际克隆提交的完整 SHA。
- 默认分支名称应从实际远端和克隆元数据确认，不能凭惯例认定为 `main`。
- 协议版本、SDK 包版本、示例版本和发布标签应分别记录，不能互相替代。
- 规范与实现如不一致，应分别标明其证据。
- JSON 应从仓库实际示例裁剪，不应自行构造后标为“原文示例”。
- 开放问题和发布历史属于远端元数据，应与固定提交的源码快照分别注明采集时间。
- 单次 `--depth 1` 克隆不能自动提供完整提交或发布历史；还需读取仓库内 changelog 和第一方发布元数据。

以上是本任务的研究规则与核查方法，不是关于任何协议的事实断言。

## 2. 获取源码的实际尝试

### 2.1 第一阶段：直接 HTTPS 克隆目标与结果

以下目标目录在调用克隆前均不存在；没有覆盖已有目录。各命令使用 Git 克隆工具，并禁用 Git hooks；没有执行仓库中的任何程序。

| 证据编号 | 区域 | 请求的远端 | 请求的目标目录 | 结果 |
| --- | --- | --- | --- | --- |
| E1 | AP2 | `https://github.com/google-agentic-commerce/AP2.git` | `/tmp/codex-payment-repos/ap2` | DNS 解析失败；未克隆 |
| E2 | A2A x402 扩展 | `https://github.com/google-agentic-commerce/a2a-x402.git` | `/tmp/codex-payment-repos/a2a-x402` | DNS 解析失败；未克隆 |
| E3 | ACP | `https://github.com/agentic-commerce-protocol/agentic-commerce-protocol.git` | `/tmp/codex-payment-repos/acp` | DNS 解析失败；未克隆 |
| E4 | x402 | `https://github.com/coinbase/x402.git` | `/tmp/codex-payment-repos/x402` | DNS 解析失败；未克隆 |
| E5 | Visa TAP | `https://github.com/visa/trusted-agent-protocol.git` | `/tmp/codex-payment-repos/tap` | DNS 解析失败；未克隆 |
| E6 | IETF Web Bot Auth | `https://github.com/webbotauth/draft-ietf-webbotauth-httpsig-protocol.git` | `/tmp/codex-payment-repos/ietf-webbotauth` | DNS 解析失败；未克隆 |
| E7 | IETF directory 旧入口 | `https://github.com/thibmeu/http-message-signatures-directory.git` | `/tmp/codex-payment-repos/ietf-directory-legacy` | DNS 解析失败；未克隆 |
| E8 | MCP 规范与 SEP | `https://github.com/modelcontextprotocol/modelcontextprotocol.git` | `/tmp/codex-payment-repos/mcp` | DNS 解析失败；未克隆 |

实际使用的命令模板如下，所有调用都包含 `--depth 1`，没有指定或猜测分支：

```sh
git -c core.hooksPath=/dev/null clone --depth 1 <远端> <目标目录>
```

E1 的输出原文如下；E2–E8 的错误与之相同，仅 URL 和目录不同：

```text
Cloning into '/tmp/codex-payment-repos/ap2'...
fatal: unable to access 'https://github.com/google-agentic-commerce/AP2.git/': Could not resolve host: github.com
```

### 2.2 安全诊断与未采取的操作

`getent hosts github.com` 没有返回地址；只读查看了系统 DNS 配置，没有修改该配置。上述诊断不能进一步证明根因究竟是 DNS 服务、网络隔离还是其他环境限制，故这里只记录观察到的解析失败，不推断根因。

网页检索通道可读取 GitHub 第一方页面，但它不提供本任务要求的终端 Git 浅克隆。插件发现结果显示 GitHub 插件 `installed: false`；没有安装插件、改变连接状态或扩张权限。没有使用固定 IP、改写 hosts、修改代理或其他方式尝试绕过网络限制。

本报告中的环境观察均来自本次工具输出，不冒充仓库规范事实。第二阶段以下三条本地传输命令成功；它们满足浅克隆和独立新目录要求，但不满足“直接确认最新上游”：

```sh
git -c core.hooksPath=/dev/null clone --depth 1 file:///tmp/payment-repos/AP2 /tmp/codex-payment-repos/ap2
git -c core.hooksPath=/dev/null clone --depth 1 file:///tmp/payment-repos/ACP /tmp/codex-payment-repos/acp
git -c core.hooksPath=/dev/null clone --depth 1 file:///tmp/payment-repos/x402 /tmp/codex-payment-repos/x402
```

## 3. 主源定位线索：网页观察，未经本地克隆核查

这一节只记录网页定位入口，不将网页当作克隆源码。网页自身没有固定 SHA；即使同名仓库后来取得离线快照，也不能把网页内容自动绑定到该快照。网页可能来自不同抓取时间，不能合成为同一个 2026-10-19 源码快照。

### 3.1 AP2、A2A x402、ACP、x402

网页通道能够打开任务提供的四个 GitHub 仓库入口；本次没有据此确认默认分支的实际远端 HEAD、是否发生过历史迁移，或是否存在归档状态。定位证据如下，**提交哈希均未取得**：

| 编号 | 仓库入口 | 网页内的路径定位 | 证据性质 |
| --- | --- | --- | --- |
| W1 | `https://github.com/google-agentic-commerce/AP2` | 仓库根页渲染的 `README.md` | 第一方网页入口；不是本地文件证据 |
| W2 | `https://github.com/google-agentic-commerce/a2a-x402` | 仓库根页 | 第一方仓库入口；没有提取源码结论 |
| W3 | `https://github.com/agentic-commerce-protocol/agentic-commerce-protocol` | 仓库根页 | 第一方仓库入口；没有提取源码结论 |
| W4 | `https://github.com/coinbase/x402` | 仓库根页 | 第一方仓库入口；没有提取源码结论 |

### 3.2 Visa TAP

搜索定位到候选主仓库 `visa/trusted-agent-protocol`，并打开了该仓库的 `README.md` 网页。未取得克隆，因此没有将 README 的安全性描述视为代码实际验证了这些性质的证据。[W5：`visa/trusted-agent-protocol/README.md@main`，网页读取，SHA 未取得。]

W5 入口：

```text
https://github.com/visa/trusted-agent-protocol/blob/main/README.md
```

### 3.3 IETF Web Bot Auth 与 directory 旧入口

网页读取 `thibmeu/http-message-signatures-directory` 时，工具报告重定向到 `webbotauth/draft-ietf-webbotauth-httpsig-protocol`。这是 URL 重定向的观察，不是从本地 Git 元数据证明的仓库迁移历史，也不说明各版本草案之间的全部关系。[W6：GitHub 仓库页面重定向元数据；非仓库文件证据；SHA 未取得。]

新入口的 `README.md` 将其标为 IETF WEBBOTAUTH 工作组草案工作区；仓库网页提供 `draft-ietf-webbotauth-httpsig-protocol.md` 文件入口。这里只保留待克隆位置，不确认正式发布的最新草案编号和日期。[W7：`webbotauth/draft-ietf-webbotauth-httpsig-protocol/README.md@main`；W8：同仓库 `draft-ietf-webbotauth-httpsig-protocol.md@main`；均为网页读取，SHA 未取得。]

W6–W8 入口：

```text
https://github.com/thibmeu/http-message-signatures-directory
https://github.com/webbotauth/draft-ietf-webbotauth-httpsig-protocol/blob/main/README.md
https://github.com/webbotauth/draft-ietf-webbotauth-httpsig-protocol/blob/main/draft-ietf-webbotauth-httpsig-protocol.md
```

不据旧仓库名称推断当前仍存在独立的 directory 工作组草案；这需要克隆后的文档内容及其指向的 IETF 第一方版本记录核查。

### 3.4 MCP 支付相关 SEP 与 proposal

第一方仓库检索定位到以下候选入口，尚未验证它们与克隆中已合并规范的关系。Issue 正文、网页标题或关闭状态不能替代已合并规范证据，也不能单独证明提案已采纳。

| 编号 | 定位项 | 路径或对象 | SHA |
| --- | --- | --- | --- |
| W9 | 支付支持提案 | `modelcontextprotocol/modelcontextprotocol` 的 Issue `#2009` | Issue 不属于克隆文件；未固定正文版本 |
| W10 | 另一支付提案入口 | 同仓库 Issue `#2008` | Issue 不属于克隆文件；未固定正文版本 |
| W11 | URL mode elicitation SEP | `docs/seps/1036-url-mode-elicitation-for-secure-out-of-band-intera.mdx@main` | 未取得 |
| W12 | MCP 主仓库说明 | `README.md@main` | 未取得 |

W9–W12 入口：

```text
https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2009
https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2008
https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/seps/1036-url-mode-elicitation-for-secure-out-of-band-intera.mdx
https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/README.md
```

以上均为候选定位记录，不宣称 MCP 已有某种正式支付协议、支付 capability、方法名或错误码。

## 4. 五个研究区域的核查结果

本节的源码结论限定到第 1 节的固定 SHA。所有“无法确认”均表示**本次未取得所需证据**，不是断言仓库没有相应内容。第 4.1–4.3 节在离线源码检查后逐段补充；第 4.4 的独立 TAP 仓库仍缺克隆。

### 4.1 Google AP2 与 A2A x402 扩展

直接 HTTPS 获取 E1 失败，后成功取得 AP2 离线浅克隆；独立 A2A x402 扩展 E2 仍失败。x402 仓库内可读的 A2A transport 另见 §4.5.3，不冒充独立扩展源码。

#### 4.1.1 版本：V2 规范与旧模型并存

本节引用全部固定为 `@AP2 = e1ea56db72a6385bce3e5c1112b3a56ce60acb43`。

| 层级 | 快照中的实际记录 | 来源 |
| --- | --- | --- |
| 发布日志 | 0.1.0，2025-09-16，创建 AP2 | `CHANGELOG.md@AP2` |
| 发布日志 | 0.2.0，2026-04-28，标记 Release of V2 | `CHANGELOG.md@AP2` |
| 规范标题 | Agentic Payment Protocol (v0.2)；Checkout / Payment 两类 Mandate，各有 open / closed | `docs/ap2/specification.md@AP2` |
| 旧模型 | IntentMandate、CartMandate、旧 PaymentMandate 仍存在 | `code/sdk/python/ap2/models/mandate.py@AP2`；`code/samples/go/pkg/ap2/types/mandate.go@AP2` |
| 包版本 | 根 Python 项目仍为 `version="0.1"` | `pyproject.toml@AP2` |
| V2 credential 类型 | `mandate.checkout.1`、`mandate.checkout.open.1`、`mandate.payment.1`、`mandate.payment.open.1` | `docs/ap2/specification.md@AP2`；`docs/ap2/checkout_mandate.md@AP2`；`docs/ap2/payment_mandate.md@AP2` |

不能把旧三模型流程与 V2 两类 open/closed 授权拼成同一规范。“旧 / 遗留”是本研究按规范结构对照所作分类，不表示源码给这些类加了正式 deprecated 注解。[上述规范与 `code/sdk/python/ap2/models/mandate.py@AP2`。]

#### 4.1.2 旧 Intent / Cart / Payment Mandate：逐字段

引用 `AM = code/sdk/python/ap2/models/mandate.py@AP2`。必填性按 Python 声明；文字安全要求不等于模型实施了签名或状态验证。

| IntentMandate 字段 | 类型 / 默认 / 必填性 | 含义 | 来源 |
| --- | --- | --- | --- |
| `user_cart_confirmation_required` | bool，默认 true | false 可代购；文字要求未签名 Intent 必须为 true，但类中没有对应验证器 | AM 第 32–46 行 |
| `natural_language_description` | string，必填 | 用户确认的自然语言购买意图 | AM 第 47–55 行 |
| `merchants` | string[] / null，默认 null | 商户限定；缺省不限定适合商户 | AM 第 56–62 行 |
| `skus` | string[] / null，默认 null | SKU 限定；缺省不限定 SKU | AM 第 63–68 行 |
| `requires_refundability` | bool / null，默认 false | 是否要求可退款 | AM 第 69–72 行 |
| `intent_expiry` | string，必填 | ISO 8601 是描述要求，没有声明日期类型或格式校验 | AM 第 73–76 行 |

| CartMandate / CartContents 字段 | 类型 / 默认 / 必填性 | 含义 | 来源 |
| --- | --- | --- | --- |
| `contents` | CartContents，必填 | 商户购物车内容 | AM 第 107–113 行 |
| `merchant_authorization` | string / null，默认 null | 描述为商户 JWT；字段本身不验证 JWT | AM 第 114–136 行 |
| `contents.id` | string，必填 | Cart 唯一 ID | AM 第 85 行 |
| `contents.user_cart_confirmation_required` | bool，必填，无默认 | 商户是否要求用户确认 | AM 第 86–92 行 |
| `contents.payment_request` | PaymentRequest，必填 | 商品 / 价格 / 支付方式 | AM 第 93–100 行 |
| `contents.cart_expiry` | string，必填 | 文字要求 ISO 8601，无格式验证器 | AM 第 101–103 行 |
| `contents.merchant_name` | string，必填 | 商户名称 | AM 第 104 行 |

`merchant_authorization` 的文字说明列 `iss/sub/aud/iat/exp/jti/cart_hash`，hash 应覆盖 CartContents 的 canonical JSON；没有指定或执行统一 JCS 算法。不能把该说明等同于实际验签实现。[AM 第 114–136 行。]

| 旧 PaymentMandate / Contents 字段 | 类型 / 默认 / 必填性 | 含义 | 来源 |
| --- | --- | --- | --- |
| `payment_mandate_contents` | PaymentMandateContents，必填 | 授权内容 | AM 第 167–183 行 |
| `user_authorization` | string / null，默认 null | VC presentation 描述；模型不自动验签 | AM 第 184–203 行 |
| `contents.payment_mandate_id` | string，必填 | Payment Mandate ID | AM 第 139–144 行 |
| `contents.payment_details_id` | string，必填 | 支付请求 ID | AM 第 145–147 行 |
| `contents.payment_details_total` | PaymentItem，必填 | 总额 | AM 第 148–150 行 |
| `contents.payment_response` | PaymentResponse，必填 | 选定支付方式与响应 | AM 第 151–157 行 |
| `contents.merchant_agent` | string，必填 | 商户标识 | AM 第 158 行 |
| `contents.timestamp` | string，默认当前 UTC ISO 字符串 | 创建时间 | AM 第 159–164 行 |

`user_authorization` 说明包含 issuer JWT 的 `cnf`，以及 KB-JWT 的 `aud/nonce/sd_hash/transaction_data`；transaction_data 描述为 CartMandate 与 PaymentMandateContents 的 hash 数组。该字符串字段没有实现这些校验。[AM 第 184–203 行。]

旧 DataPart key 为 `ap2.mandates.IntentMandate` / `CartMandate` / `PaymentMandate`；V2 示例改用 `ap2.mandates.CheckoutMandateSdJwt`、`ap2.mandates.PaymentMandateSdJwt`，不是同种 payload 的别名。[AM 第 27–29 行；`code/samples/python/src/common/constants.py@AP2`。]

#### 4.1.3 W3C PaymentRequest 复用：仓库实际模型

引用 `APR = code/sdk/python/ap2/models/payment_request.py@AP2`。文件说明这是仓库自行定义的 W3C 风格对象；本报告不将其称为完整、经过认证的 W3C 标准实现。以下是类字段，不是外部标准逐项对照。[APR 第 15–23 行。]

记号：R 为必填，O 为非必填，`/ null` 表示声明允许 null；默认值按 Python Field / 类型声明。

| 类型 | 全部字段 | 类型约束与注意点 | 来源 |
| --- | --- | --- | --- |
| PaymentRequest | `method_data: PaymentMethodData[] R`；`details: PaymentDetailsInit R`；`options: PaymentOptions/null O=null`；`shipping_address: ContactAddress/null O=null` | 支付方式列表未设置最小长度 | APR 第 184–201 行 |
| PaymentMethodData | `supported_methods: string R`；`data: dict/null O={}`，default_factory | data 内部字段取决于支付方式，未封闭定义 | APR 第 117–133 行 |
| PaymentDetailsInit | `id: string R`；`display_items: PaymentItem[] R`；`shipping_options: PaymentShippingOption[]/null O=null`；`modifiers: PaymentDetailsModifier[]/null O=null`；`total: PaymentItem R` | 无列表最小长度或总额求和校验声明 | APR 第 160–181 行 |
| PaymentCurrencyAmount | `currency: string R`；`value: float R` | 币种 ISO 4217 仅为说明；没有精度 / 非负声明 | APR 第 34–44 行 |
| PaymentItem | `label: string R`；`amount: PaymentCurrencyAmount R`；`pending: bool/null O=null`；`refund_period: int O=30` | 退款期单位为天，无非负约束 | APR 第 47–65 行 |
| PaymentShippingOption | `id: string R`；`label: string R`；`amount: PaymentCurrencyAmount R`；`selected: bool/null O=false` | 未声明 ID 唯一或仅一项 selected | APR 第 68–86 行 |
| PaymentOptions | `request_payer_name/email/phone: bool/null O=false`；`request_shipping: bool/null O=true`；`shipping_type: string/null O=null` | shipping/delivery/pickup 仅在说明中，非枚举 | APR 第 89–114 行 |
| PaymentDetailsModifier | `supported_methods: string R`；`total: PaymentItem/null O=null`；`additional_display_items: PaymentItem[]/null O=null`；`data: dict/null O=null` | 支付方式对应的金额和展示修改 | APR 第 136–157 行 |
| PaymentResponse | `request_id: string R`；`method_name: string R`；`details: dict/null O=null`；`shipping_address: ContactAddress/null O=null`；`shipping_option: PaymentShippingOption/null O=null`；`payer_name/email/phone: string/null O=null` | shipping_option 是完整对象；details 没有通用内部 schema | APR 第 204–229 行 |
| ContactAddress | `city/country/dependent_locality/organization/phone_number/postal_code/recipient/region/sorting_code: string/null O=null`；`address_line: string[]/null O=null` | 全部可省略，未声明地区 / 联系方式格式 | `code/sdk/python/ap2/models/contact_picker.py@AP2` 第 32–48 行 |

Go 也将旧金额声明为 float64，并把 refund_period=0 通过 getter 解释为默认 30；Python 的 int 字段没有相同 getter 语义。[`code/samples/go/pkg/ap2/types/payment_request.go@AP2` 第 19–42 行；APR 第 63–65 行。]

未找到可直接原样摘录的旧 IntentMandate / CartMandate / PaymentMandate 完整固定 JSON 实例；Cart 与 PaymentRequest 在 Go 中动态构造，Intent 只有字段级 example。为遵守“原文 JSON”要求，不将构造代码自行序列化为伪称仓库原例的 JSON。V2 实际 JSON 另见后文。[AM；APR；`code/samples/go/pkg/roles/merchant_agent/storage.go@AP2` 第 163–215 行；`code/samples/go/pkg/roles/merchant_payment_processor_agent/executor.go@AP2` 第 56–69 行。]

#### 4.1.4 V2 角色与实际消息序列

| 角色 | 规范责任 | 来源 |
| --- | --- | --- |
| Shopping Agent | 发现商品、构造 Checkout、执行购买；视作潜在攻击者 | `docs/ap2/specification.md@AP2` Roles、Agentic vs Non-Agentic |
| Credential Provider | 验证访问凭据的授权并限定支付凭据作用域 | 同上 |
| Merchant | 完成 Checkout，验证商品购买授权，维护库存 / 价格 / 优惠完整性 | 同上 |
| Merchant Payment Processor | 验证支付凭据有权支付当前 Checkout，并处理支付 | 同上 |
| Trusted Surface | 确定性展示并获取知情同意；必须 non-agentic | 同上 |

同一实体可兼任角色；无论参与者是否 agentic，授权验证必须通过确定性代码。AP2 不定义完整 catalog / checkout 更新 API，而是在 commerce protocol 内提供安全授权。[`docs/ap2/specification.md@AP2` Scope、Roles、Agentic vs Non-Agentic。]

Human Present，以下为仓库明确标识的非规范性 flow：

1. 用户发起购物，SA 与商户形成 Checkout，商户返回签名 Checkout；SA 获取并选择 instrument。[`docs/ap2/flows.md@AP2` 第 37–42 行。]
2. SA 构造 closed Checkout / Payment Mandate 内容，交 Trusted Surface 展示、认证和取得同意。[同文件第 51–56 行。]
3. Trusted Surface 签署两类 Mandate，checkout JWT hash 绑定二者，返回 SA。[同文件第 57–62 行。]
4. SA 将 Payment Mandate 给 CP，CP 验证并释放限定支付 token，可能涉及 Network。[同文件第 63–67 行。]
5. SA 将 token 与 Checkout Mandate 发商户；商户验证当前购物车，向 MPP 发起支付。[同文件第 68–72 行。]
6. MPP 验证 Payment 授权及 checkout hash 绑定，返回签名 Payment Receipt；商户返回签名 Checkout Receipt。[同文件第 73–77 行。]

Human Not Present：

1. 用户尚在场时批准 open Checkout / Payment Mandate，Trusted Surface 签署，内含 Agent public key；open Payment 关联 open Checkout hash。[`docs/ap2/flows.md@AP2` 第 97–117 行。]
2. 用户离开后 SA 自主选择商品，取得商户签名 Checkout。[同文件第 118–128 行。]
3. SA 用 Agent key 签 closed Mandate，并通过 `sd_hash` 将 closed 绑定到 open。[同文件第 139–150 行。]
4. CP 验证 open+closed Payment 链后释放 token；商户取得 token 与 open+closed Checkout 链。[同文件第 151–157 行。]
5. 商户 / MPP 分别检查内容、约束及 hash 关联并签收据；遇 `unresolved_constraint` 可要求用户返回直接授权 closed Mandate。[同文件第 158–167、11–13 行；`docs/ap2/agent_authorization.md@AP2` 第 524–535 行。]

旧 human-present 样例 README 仍叙述 Intent→Cart→Payment、DPAN 和 mock OTP；这不应替代上述 V2 规范流程。[`code/samples/python/scenarios/a2a/human-present/cards/README.md@AP2`。]

#### 4.1.5 V2 Mandate 与嵌套字段

以下表格区分 JSON Schema 的 required 与生成模型的 default；schema 的 `default` 不等于任何验证器都会补字段。

| V2 对象 | 必填字段 | 可选字段 / 限制 | 来源 |
| --- | --- | --- | --- |
| Closed Checkout | `vct:string="mandate.checkout.1"`；`checkout_jwt:string`；`checkout_hash:string` | `iat/exp:integer`；checkout_jwt 可选择性披露；hash 跟随 `_sd_alg`，缺省 SHA-256 | `code/sdk/schemas/ap2/checkout_mandate.json@AP2` |
| Open Checkout | `vct:string="mandate.checkout.open.1"`；`constraints:array`；`cnf:object` | `iat/exp:integer`；constraints schema 要求至少一个 checkout.line_items | `code/sdk/schemas/ap2/open_checkout_mandate.json@AP2` |
| Closed Payment | `vct:string="mandate.payment.1"`；`transaction_id:string`；`payee:Merchant`；`payment_amount:Amount`；`payment_instrument:PaymentInstrument` | `pisp:PISP`；`execution_date:string`；`risk_data:object`；`iat/exp:integer`；transaction_id 为 checkout JWT hash，缺执行日期为立即执行 | `code/sdk/schemas/ap2/payment_mandate.json@AP2` |
| Open Payment | `vct:string="mandate.payment.open.1"`；`constraints:array`；`cnf:object` | 可选 payee、payment_amount、payment_instrument、pisp、execution_date、risk_data、iat、exp；constraints schema 要求 payment.reference | `code/sdk/schemas/ap2/open_payment_mandate.json@AP2` |

生成的 Python 模型给 vct 提供默认值，其他可选字段默认 None；不要把 schema 的 required 和生成模型的可省略性当成同一种规则。[`code/sdk/python/ap2/sdk/generated/checkout_mandate.py@AP2`；`code/sdk/python/ap2/sdk/generated/payment_mandate.py@AP2`；`code/sdk/python/ap2/sdk/generated/open_checkout_mandate.py@AP2`；`code/sdk/python/ap2/sdk/generated/open_payment_mandate.py@AP2`。]

| 共享对象 | 全部字段 | 来源 |
| --- | --- | --- |
| Amount | 必需 `amount:integer`，minor units；`currency:string`；未规定 amount 非负或 currency pattern | `code/sdk/schemas/ap2/types/amount.json@AP2` |
| Merchant | 必需 `id/name:string`；可选 `website:string` | `code/sdk/schemas/ap2/types/merchant.json@AP2` |
| PaymentInstrument | 必需 `id/type:string`；可选 `description:string`；type 没有封闭枚举 | `code/sdk/schemas/ap2/types/payment_instrument.json@AP2` |
| PISP | 必需 `legal_name/brand_name/domain_name:string`；QWAC 说明不是证书验证代码 | `code/sdk/schemas/ap2/types/pisp.json@AP2` |
| 候选 acceptable item | 必需 id/title；没有 price | `code/sdk/schemas/ap2/open_checkout_mandate.json@AP2` |
| 另一个共享 Item | 必需 id/title、price（integer≥0）；可选 image_url（URI） | `code/sdk/schemas/ap2/types/item.json@AP2` |
| cnf | Open schema 仅为 object；SDK 链验证读取 cnf.jwk | `code/sdk/schemas/ap2/open_payment_mandate.json@AP2`；`code/sdk/python/ap2/sdk/sdjwt/common.py@AP2` |
| JsonWebKey | 必需 kty=EC；可选 crv=P-256、x/y（43 字符 base64url pattern）、use（sig/enc）、key_ops（sign/verify/encrypt/decrypt/wrapKey/unwrapKey/deriveKey/deriveBits）、alg=ES256、kid、x5u URI、x5c string[]、x5t、x5t#S256；禁止额外属性 | `code/sdk/schemas/ap2/types/jwk.json@AP2`；`code/sdk/python/ap2/sdk/generated/types/jwk.py@AP2` |

内置 constraint 字段：各 type 为 required string 且有 const/default；Python 对 type 提供默认值。

| type | 全部其余字段 | 规范的评估要求 | 来源 |
| --- | --- | --- | --- |
| checkout.allowed_merchants | 必需 `allowed:Merchant[]`；元素可选择性披露 | 商户在已披露列表中；没有披露则失败 | `code/sdk/schemas/ap2/open_checkout_mandate.json@AP2`；`docs/ap2/checkout_mandate.md@AP2` |
| checkout.line_items | 必需 items，至少1项；每项 id:string、acceptable_items:{id,title}[]、quantity:integer>0 | 匹配商品 ID、数量，不能重复占用 | 同上 |
| payment.allowed_payees | 必需 `allowed:Merchant[]` | 收款方在 allowed 中 | `code/sdk/schemas/ap2/open_payment_mandate.json@AP2`；`docs/ap2/payment_mandate.md@AP2` |
| payment.allowed_payment_instruments | 必需 `allowed:PaymentInstrument[]` | Instrument 在 allowed 中 | 同上 |
| payment.allowed_pisps | 必需 `allowed:PISP[]` | PISP 在 allowed 中 | 同上 |
| payment.amount_range | 必需 currency:string、max:integer；可选 min:integer | 币种相同，minor-unit 金额在范围内 | 同上 |
| payment.budget | 必需 max:number、currency:string | 本次金额加累计金额≤max；依赖历史状态 | 同上 |
| payment.reference | 必需 conditional_transaction_id:string | 关联授权链包含 hash 匹配的 open Checkout | 同上 |
| payment.agent_recurrence | 必需 frequency：ON_DEMAND/DAILY/WEEKLY/BIWEEKLY/MONTHLY/QUARTERLY/ANNUALLY；可选 max_occurrences:integer | 满足时间间隔、次数并跟踪历史 | 同上 |
| payment.execution_date | 可选 not_before/not_after:string | 执行日期在包含端点的区间内 | 同上 |

Receipt schema：

| 对象 | 字段和状态条件 | 来源 |
| --- | --- | --- |
| Checkout Receipt | 必需 status（Success/Error）、iss:string、iat:integer、reference:string；Success 要求 order_id:string；Error 要求 error/error_description:string | `code/sdk/schemas/ap2/checkout_receipt.json@AP2` |
| Payment Receipt | 必需 status、iss、iat、reference、payment_id；Success 另需 psp_confirmation_id/network_confirmation_id:string；Error 另需 error/error_description:string | `code/sdk/schemas/ap2/payment_receipt.json@AP2` |

生成模型并不完全禁止另一状态的附加字段：例如 Success 仍可含可选 error。不能把文档“if and only if”自动视作完整互斥验证。[`code/sdk/python/ap2/sdk/generated/checkout_receipt.py@AP2`；`code/sdk/python/ap2/sdk/generated/payment_receipt.py@AP2`。]

#### 4.1.6 签名、哈希、挑战与重放

可确认的商户 JWT helper：生成 P-256 (`SECP256R1`) key，header 使用 ES256 / JWT / merchant-key-1；对 compact JSON header 与 payload 分别无 padding base64url，签 `ASCII(h+"."+p)`，算法 ECDSA(SHA256)，DER 签名转 32-byte r + 32-byte s 后 base64url。没有 `sort_keys=True` 或 JCS 调用。[`code/samples/python/src/roles/merchant_agent/tools.py@AP2` 第 84–131、202–207 行。]

Checkout hash 对**完整 compact checkout JWT 字符串**做 SHA-256 后无 padding base64url，而不是重排 JSON 后 hash。普通 JWT / receipt helper 则用 jwcrypto JWS、ES256 与普通 `json.dumps`；不能声称所有 helper 使用同一种 JSON 序列化。[同上；`code/sdk/python/ap2/sdk/jwt_helper.py@AP2`。]

Android helper 对传入 headerJson / payloadJson 的 UTF-8 原始字节做 base64url，再签拼接结果，使用 ES256，返回 r+s；也没有在该函数重新规范化 JSON。[`code/samples/android/shopping_assistant/app/src/main/java/com/example/a2achatassistant/agent/CryptoUtils.kt@AP2` 第 73–79 行。]

SD-JWT issuance 调用外部 `SDJWTIssuer` 并传 `sign_alg=None`。依赖实现不在已克隆的 AP2 文件中；不能猜默认签名选择、salt 长度、RNG 或最终发行 token 的内部序列化。[`code/sdk/python/ap2/sdk/sdjwt/common.py@AP2` 第 272–289 行；`pyproject.toml@AP2`。]

| 哈希 / 链操作 | 精确输入与规则 | 来源 |
| --- | --- | --- |
| `_sd_alg` | sha-256 / sha-384 / sha-512；缺省 SHA-256，其他值拒绝 | `code/sdk/python/ap2/sdk/sdjwt/common.py@AP2` 第 24–28、156–162 行 |
| `sd_hash` | issuer JWT + `~` + 原顺序 disclosures + 末尾 `~` 的 ASCII；不含 trailing KB-JWT | 同文件第 53–63、165–172 行 |
| `issuer_jwt_hash` | 仅 compact issuer JWT ASCII，不含 disclosures | 同文件第 175–177 行 |
| disclosure digest | 已编码 disclosure 字符串 ASCII，不是 decoded JSON 重序列化 | 同文件第 180–182 行 |
| hop 绑定 | sd_hash / issuer_jwt_hash 必须恰有一个且匹配上一 token | 同文件第 185–222 行 |
| chain 编码 | hop 用 `~~` 分隔，验证恢复相应末尾 `~` | `code/sdk/python/ap2/sdk/mandate.py@AP2` 第 68–75、390–400 行 |
| typed issuance 数据 | `model_dump(by_alias=True,exclude_none=True)` | `code/sdk/python/ap2/sdk/sdjwt/common.py@AP2` 第 225–229 行 |

挑战和状态边界：

- 规范要求 open Mandate 的 cnf 绑定 Agent key，建议短 exp，并要求收到 rejection receipt 前不呈现重叠授权；Checkout JWT 签名不应是可猜测的确定性签名，以避免 hash 字典攻击。[`docs/ap2/specification.md@AP2` Autonomous、Payment Mandate；`docs/ap2/security_and_privacy_considerations.md@AP2`。]
- 创建 KB-SD-JWT hop 要求 aud / nonce 非空并注入 iat，下一 hop 用上一 cnf.jwk 验签；终端 hop 的 aud / nonce 只在调用者提供 expected 值时比较。带独立尾部 KB-JWT 的 presentation 有额外要求 expected_aud / expected_nonce 同时提供。[`code/sdk/python/ap2/sdk/sdjwt/kb_sd_jwt.py@AP2`；`code/sdk/python/ap2/sdk/sdjwt/common.py@AP2` 第 292–311 行；`code/sdk/python/ap2/sdk/mandate.py@AP2` 第 225–234 行。]
- 默认 clock skew 为 300 秒；存在 exp 才检查截止，存在 iat 检查未来偏差，没有统一强制 exp 或 iat 最大年龄。次数、累计金额依赖调用者提供的 MandateContext，不是纯验签保证。[`code/sdk/python/ap2/sdk/sdjwt/chain.py@AP2`；`code/sdk/python/ap2/sdk/constraints.py@AP2`。]
- 商户演示 nonce 是固定 `merchant-nonce-xyz`；不能从该演示证明有部署级一次性 challenge 消费和去重。[`code/samples/python/src/roles/merchant_agent/tools.py@AP2` 第 73–77、415–422 行。]

#### 4.1.7 V2 的原文 JSON 裁剪

以下是 decoded payload / delegation 内容，不是包含可验证签名的完整 wire token。只删除字段或包装，保留原值；checkout_jwt 的截断字符串是原例已有。

两个 closed Mandate 的 delegate 请求原例裁剪。[`docs/ap2/agent_authorization.md@AP2` 第 191–229 行。]

```json
{
  "type": "delegate",
  "format": "dc+sd-jwt",
  "credential_ids": ["dpc_credential"],
  "transaction_data_hashes_alg": ["sha-256"],
  "delegate_payload": [
    {
      "vct": "mandate.checkout.1",
      "checkout_jwt": "eyJhbGciOiJFUzI1NiIs...<Merchant_Checkout_JWT>",
      "checkout_hash": "3WiKMabE8NRYJgveUbyAZ3pBqRfPrWwGDbOyvbO1eYA"
    },
    {
      "vct": "mandate.payment.1",
      "transaction_id": "3WiKMabE8NRYJgveUbyAZ3pBqRfPrWwGDbOyvbO1eYA",
      "payment_amount": {"amount": 15000, "currency": "USD"},
      "payee": {"id": "merchant_1", "name": "Generic Merchant"},
      "payment_instrument": {
        "id": "b3f1c8a2-6d4e-4f9a-9e3d-8a7c2f1b9d34",
        "type": "dpc",
        "description": "DPC ···· 4444"
      }
    }
  ]
}
```

Open Checkout，省略另一可选 constraint、iat / exp；`"..."` 是原例的 SD-JWT array digest key，不是本报告的省略符。[`docs/ap2/checkout_mandate.md@AP2` 第 195–234 行。]

```json
{
  "vct": "mandate.checkout.open.1",
  "constraints": [
    {
      "type": "checkout.line_items",
      "items": [
        {
          "id": "line_1",
          "acceptable_items": [
            {"...": "y3aocAD2rhYpJQOUMN016faDFGkTBGEDVl1R1TRHdbw"}
          ],
          "quantity": 1
        }
      ]
    }
  ],
  "cnf": {
    "jwk": {
      "crv": "P-256",
      "kty": "EC",
      "x": "QpSyxPQHy38xckypDr54gZ3T42zj9iLtV4koyb5U27c",
      "y": "37HLd7JJinxjJIn8J7HijssoeclbfhdW-gUL7feI9lw"
    }
  }
}
```

Open Payment，裁剪 allowed_payees / 时间字段，保留 amount 与 reference。[`docs/ap2/payment_mandate.md@AP2` 第 306–340 行。]

```json
{
  "vct": "mandate.payment.open.1",
  "constraints": [
    {"type": "payment.amount_range", "currency": "USD", "max": 20000, "min": 0},
    {
      "type": "payment.reference",
      "conditional_transaction_id": "FzLoxbbtgQGYZxoSM2NJYJtkFTSsdfUBoVEQ12k7JN8"
    }
  ],
  "cnf": {
    "jwk": {
      "crv": "P-256",
      "kty": "EC",
      "x": "QpSyxPQHy38xckypDr54gZ3T42zj9iLtV4koyb5U27c",
      "y": "37HLd7JJinxjJIn8J7HijssoeclbfhdW-gUL7feI9lw"
    }
  }
}
```

#### 4.1.8 SDK、参考实现、治理与具体缺口

正式 runtime SDK 只观察到 **Python 一种语言**；参考工件分为 Python 多角色示例、Go 多角色示例、Android Kotlin 客户端、TypeScript/React Web demo 四类，不是四个同等完备的 SDK。AP2 schemas 有 6 个顶层 Mandate / Receipt 和 7 个共享类型；Go / Kotlin / Web 的存在不证明 V2 互通已完成。[`code/README.md@AP2`；`code/samples/python/README.md@AP2`；`code/samples/go/README.md@AP2`；`code/samples/android/README.md@AP2`；`code/web-client/package.json@AP2`；`code/sdk/schemas/README.md@AP2`；AP2 提交的 `code/sdk/schemas/ap2/` 树。]

根 LICENSE / Python manifest 声明 Apache-2.0；CONTRIBUTING 声明 core specification 已捐赠 FIDO，后续本仓库贡献范围为 samples / SDK。这里转述快照声明，未确认当前外部 FIDO 工作区。贡献需要 Google CLA 和 PR review，CODEOWNERS 指向 `@google-agentic-commerce/google-payments-swe`。[`LICENSE@AP2`；`pyproject.toml@AP2`；`CONTRIBUTING.md@AP2`；`.github/CODEOWNERS@AP2`。]

以下是静态证据，不是已经实测的漏洞或仍开放 issue；尤其不能把“依赖调用者完成”误说成“协议完全没有要求”：

| 差异 / 缺口 | 已观察证据 | 来源 |
| --- | --- | --- |
| Receipt reference 输入不同 | 规范描述按 sd_hash；SDK 推荐 SHA-256(last compact JWT)，不含 disclosures | `docs/ap2/agent_authorization.md@AP2` 第 503–513 行；`code/sdk/python/ap2/sdk/mandate.py@AP2` 第 402–417 行 |
| Receipt 状态字段不同 | 授权文档为 result:success/error，AP2 schema 为 status:Success/Error | 同文档第 503–516 行；`code/sdk/schemas/ap2/types/receipt_status.json@AP2` |
| 必需 constraint 未同层落实 | Schema contains 要求 line_items / reference；生成模型 constraints 为普通 list，检查器仅遍历已有项，测试可构造空列表 | `code/sdk/schemas/ap2/open_checkout_mandate.json@AP2`；`code/sdk/schemas/ap2/open_payment_mandate.json@AP2`；`code/sdk/python/ap2/sdk/constraints.py@AP2`；`code/sdk/python/ap2/tests/agent_recurrence_tests.py@AP2` |
| 数量匹配语义不同 | 文档要求总量匹配；代码按容量分配，测试要求3件 / cart2件仍期待通过 | `docs/ap2/checkout_mandate.md@AP2` 第 83–105 行；`code/sdk/python/ap2/sdk/max_flow_helper.py@AP2`；`code/sdk/python/ap2/tests/constraints_line_items_tests.py@AP2` |
| 空 acceptable_items 作 wildcard | Helper 将空列表视为任意 SKU，而文档要求商品位于披露列表 | 同文档第 85–89 行；`code/sdk/python/ap2/sdk/max_flow_helper.py@AP2` 第 36–44、74–82 行 |
| Recurrence 间隔检查 | Evaluator 检查 total_uses/max_occurrences，未用 frequency/last_used_date 检查间隔 | `docs/ap2/payment_mandate.md@AP2` 第 57–62 行；`code/sdk/python/ap2/sdk/constraints.py@AP2` 第 197–229 行 |
| 金额单位 / 精度差异 | V2 Amount 为 minor-unit int；budget 为 number、SDK 固定 int(max*100)；旧 PaymentRequest 为 float | `code/sdk/schemas/ap2/types/amount.json@AP2`；`code/sdk/schemas/ap2/open_payment_mandate.json@AP2`；`code/sdk/python/ap2/sdk/constraints.py@AP2` 第 304–311 行；APR |
| 执行日期比较 | Evaluator 按 string 比较，缺 execution_date 无违规；不能确认任意 ISO 表示和立即执行语义均正确 | `code/sdk/python/ap2/sdk/constraints.py@AP2` 第 316–345 行 |
| Open preset claim 范围 | 专门检查 payee、amount、instrument、execution_date；该函数未查 pisp / risk_data | `docs/ap2/agent_authorization.md@AP2` 第 460–465 行；`code/sdk/python/ap2/sdk/constraints.py@AP2` 第 433–486 行 |
| Checkout 链不是完整验商户 JWT | 本类解析 payload / constraints，只在有 expected_checkout_hash 时比较；不验 checkout JWT 签名或自算 hash | `code/sdk/python/ap2/sdk/checkout_mandate_chain.py@AP2` 第 44–137 行 |
| 商户样例验证降级 | initiate_payment 仅有 Checkout Mandate 时验证；链模型校验失败后 helper 返回 raw dict | `code/samples/python/src/roles/merchant_agent/tools.py@AP2` 第 256–264、429–449 行；对照 `docs/ap2/specification.md@AP2` Merchant |
| Receipt callback 可选却调用 | 参数默认 None，文档称 optional；实现直接调用，未见 None guard；本次未执行验证结果 | `code/sdk/python/ap2/sdk/receipt_wrapper.py@AP2` 第 100–105、144–158 行；`code/sdk/python/ap2/sdk/README.md@AP2` |
| x5c 依赖信任配置 | trusted_roots 可为 None，仅有值时检查根锚；该函数未覆盖完整有效期 / 用途验证 | `code/sdk/python/ap2/sdk/sdjwt/chain.py@AP2` 第 44–108 行 |
| 多 payload / hop 分层 | create(payloads) 只用 payloads[0]；底层可多 hop，两个 AP2 chain wrapper 要求恰好2 payload | `code/sdk/python/ap2/sdk/mandate.py@AP2`；`code/sdk/python/ap2/sdk/sdjwt/chain.py@AP2`；`code/sdk/python/ap2/sdk/checkout_mandate_chain.py@AP2`；`code/sdk/python/ap2/sdk/payment_mandate_chain.py@AP2` |
| Go 是 mock | merchant_authorization=FakeJWT，MPP 返回固定 SUCCESS / AUTH-123456 | `code/samples/go/pkg/roles/merchant_agent/tools.go@AP2`；`code/samples/go/pkg/roles/merchant_payment_processor_agent/executor.go@AP2` |
| Trusted Surface 演示 stub | Web biometric auth 直接 true；Android key 函数注释将来再用 keystore | `code/web-client/src/trustedSurface.ts@AP2`；`code/samples/android/shopping_assistant/app/src/main/java/com/example/a2achatassistant/agent/CryptoUtils.kt@AP2` |
| 文档维护 TODO | Code of Conduct 保留 steward/name/email 占位；Go README1.21+ vs go.mod1.24.0 | `CODE_OF_CONDUCT.md@AP2`；`code/samples/go/scenarios/a2a/human-present/cards/README.md@AP2`；`code/samples/go/go.mod@AP2` |

外部 SD-JWT 库的默认行为、当前上游修复状态和开放 issue 均无法从本快照确认。

### 4.2 ACP

直接 HTTPS 克隆 E3 失败，后取得 ACP 离线浅克隆。以下“已发布 / 当前”仅按固定 SHA 的 README 和日期目录标记，不表示已验证线上或最新上游。

#### 4.2.1 证据索引与版本历史

本节全部 `@ACP = 7fdd78df677a94dce04c770644b0fbbb1401272b`。短引用如下；R 为 schema 必需，O 为可选，不自行补默认值。

| 短引用 | 文件 |
| --- | --- |
| AC | `spec/2026-04-17/openapi/openapi.agentic_checkout.yaml@ACP` |
| AJ | `spec/2026-04-17/json-schema/schema.agentic_checkout.json@ACP` |
| AD | `spec/2026-04-17/openapi/openapi.delegate_payment.yaml@ACP` |
| ADJ | `spec/2026-04-17/json-schema/schema.delegate_payment.json@ACP` |
| AW | `spec/2026-04-17/openapi/openapi.agentic_checkout_webhook.yaml@ACP` |
| AA | `spec/2026-04-17/openapi/openapi.delegate_authentication.yaml@ACP` |
| ARC | `rfcs/rfc.agentic_checkout.md@ACP` |
| ARD | `rfcs/rfc.delegate_payment.md@ACP` |
| ARH | `rfcs/rfc.payment_handlers.md@ACP` |
| AEX | `examples/2026-04-17/examples.agentic_checkout.json@ACP` |
| ADEX | `examples/2026-04-17/examples.delegate_payment.json@ACP` |

README 标记 beta，同时 Latest Stable 链接指向 **2026-04-17**；开发版为 `spec/unreleased`，OpenAPI info.version 为 unreleased。npm manifest 的 specification `0.1.0` 不是日期 API 版本。[`README.md@ACP` 第 6–10、71–98 行；`package.json@ACP`；`spec/unreleased/openapi/openapi.agentic_checkout.yaml@ACP`；`spec/unreleased/openapi/openapi.delegate_payment.yaml@ACP`。]

| 标注发布日 | 日志内容 | 来源 |
| --- | --- | --- |
| 2025-09-29 | 初始 checkout、delegate payment、OpenAPI 与示例 | `changelog/2025-09-29.md@ACP` |
| 2025-12-12 | 嵌套 fulfillment_details、多履约、展示、intent traces、affiliate；弃用初始版本 | `changelog/2025-12-12.md@ACP` |
| 2026-01-16 | Authentication provider、merchant_id、Adyen | `changelog/2026-01-16.md@ACP` |
| 2026-01-30 | Capability negotiation、payment handlers、extensions、discounts；handler 为 breaking change | `changelog/2026-01-30.md@ACP` |
| 2026-04-17 | Cart/feed/auth/discovery/MCP、订单跟踪、强制幂等；弃用前版 | `changelog/2026-04-17.md@ACP` |

**日期目录不是不可变发布原件**：unreleased 日志明确将 delegate-payment 示例修复追补至五个已发布目录；订单 RFC 的 2026-04-23 / 30 变化也反映在 2026-04-17 文件中。因此本报告引用“该 SHA 下的标注版本”，不声称是 4 月17日当日原件。[`changelog/unreleased/unwrap-error-envelope-released-delegate-payment.md@ACP`；`changelog/unreleased/order-schema-alignment.md@ACP`；`rfcs/rfc.orders.md@ACP` 第 528–534 行。]

开发版新增 Item.url、suggested_price（amount、source、observed_at，source 含 feed_id 及可选 channel/feed_update_id/updated_at），complete.fulfillment_details，以及 feed_update_id / last_update_id 等；不能混入已发布字段。complete 地址变更的 409 保护在变更说明与 OpenAPI 描述间仍未完整对齐。[`changelog/unreleased/product-url-on-checkout-item.md@ACP`；`changelog/unreleased/suggested-pricing.md@ACP`；`changelog/unreleased/fulfillment-details-on-complete.md@ACP`；`changelog/unreleased/add-feed-update-id.md@ACP`；`spec/unreleased/openapi/openapi.agentic_checkout.yaml@ACP`。]

#### 4.2.2 角色与消息序列

Agent 代表买方，但不必成为 merchant of record；卖方是订单、税、付款和合规的 system of record，授权与结算经其 PSP。[`README.md@ACP`；ARC Overview。]

1. Agent 可公开读取 `/.well-known/acp.json`，取得 API base、支持版本、transport / 稳定能力；不需 Bearer，session payment handlers 仍在创建时协商。[`rfcs/rfc.discovery.md@ACP`。]
2. Agent `POST /checkout_sessions`，卖方返回权威 session / cart 状态及协商后的 handlers。[AC Create；ARH Session Negotiation。]
3. Agent 更新或 GET session，选择商品、履约和身份信息；不能以 agent 自算总价取代卖方权威报价。[AC Update / Get；`rfcs/rfc.product_feeds.md@ACP`。]
4. 若 handler.requires_delegate_payment，Agent 向指定 PSP 提交 credential + allowance，取得绑定 merchant / session 的 vault token。[ARH；`docs/mcp-binding.md@ACP`。]
5. Session 为 authentication_required 时完成认证，在 complete 返回结果，包括失败结果；认证本身有独立 provider API。[ARC Authentication；AA。]
6. Agent complete，提交 handler_id 与 instrument / token；成功建立 order 并返回 completed session。[AC Complete。]
7. 非 completed/canceled 的 session 可以 cancel，携可选 intent trace。[AC Cancel。]
8. 商户向 webhook 接收端发送完整 Order 更新售后状态；webhook 不是执行资金结算的 API。[AW；ARC。]

#### 4.2.3 Checkout endpoint 与 headers

| 方法 / 路径 | Body | 成功结果 | 来源 |
| --- | --- | --- | --- |
| POST `/checkout_sessions` | R CreateRequest | 201 CheckoutSession | AC 第 16–120 行 |
| POST `/checkout_sessions/{checkout_session_id}` | R UpdateRequest | 200 CheckoutSession | AC 第 121–197 行 |
| GET 同一路径 | 无 JSON body | 200 CheckoutSession；可404 | AC 第 198–216 行 |
| POST `…/complete` | R CompleteRequest | 200 CheckoutSessionWithOrder | AC 第 217–312 行 |
| POST `…/cancel` | Body 可省略 | 200 CheckoutSession；已完成 / 取消可405 | AC 第 313–373 行 |

Session path ID 必需 string；所有 checkout POST 要求 Idempotency-Key，GET 不要求。[AC Parameters；ARC Idempotency。]

| Header | 合同 | 来源 |
| --- | --- | --- |
| Authorization | R Bearer API key | AC 第 375–388 行 |
| API-Version | R 日期版本；示例仍是旧 2026-01-16，不能当 info.version | AC 第 444–450 行；ARC |
| Content-Type | 有 body 时 application/json；RFC 要求，OpenAPI 参数列表覆盖不统一 | ARC 第 59–71 行；AC |
| Idempotency-Key | POST R，1–255 字符，建议 UUIDv4，identity+endpoint scope | AC 第 410–419 行 |
| Accept-Language、User-Agent、Request-Id | O string，RFC 建议 | AC 第 396–426 行；ARC |
| Signature、Timestamp | OpenAPI O；RFC 推荐 canonical JSON 签名 / RFC3339；Signature 描述误写 webhook HMAC | AC 第 427–443 行；ARC |
| 响应 Idempotency-Key / Idempotent-Replayed | 成功 POST 返回 key，可附 `"true"` replay 标识 | AC 对应成功响应 |

#### 4.2.4 请求和 Session 字段全集

| 请求 | R 字段 | O 字段 | 来源 |
| --- | --- | --- | --- |
| Create | line_items:Item[]（至少1）、currency:string、capabilities | buyer、fulfillment_details、fulfillment_groups[]、affiliate_attribution、coupons:string[]（deprecated）、discounts、locale、timezone、quote_id、metadata；AJ 另有 order_notes≤5000字符 | AC 第 2969–3027 行；AJ 第 3435–3513 行 |
| Update | 无 R 属性，但 body 必需 | buyer、line_items:Item[]、fulfillment_details、fulfillment_groups[]、selected_fulfillment_options[]、coupons（deprecated）、discounts；AJ 另有 order_notes≤5000 | AC 第 3028–3063 行；AJ |
| Complete | payment_data | buyer、authentication_result、affiliate_attribution、risk_signals、marketing_consents[]；AJ 另有 order_notes≤5000；已发布版没有 fulfillment_details | AC 第 3087–3112 行；AJ；开发版差异见前文 |
| Cancel | 无，body 可省略 | intent_trace | AC CancelSessionRequest |

Create / Update / Complete 顶层 `additionalProperties:false`，旧 RFC 的 items 或旧 token/provider 结构不能自动视为当前合法字段。[AC 对应 Request schemas。]

| Session 类别 | 全部字段 / 条件 | 来源 |
| --- | --- | --- |
| 必需 | id、status、currency、line_items[]、totals[]、fulfillment_options[]、messages[]、links[]、capabilities | AC CheckoutSessionBase 第 2768–2905 行 |
| status enum | incomplete、not_ready_for_payment、requires_escalation、authentication_required、ready_for_payment、pending_approval、complete_in_progress、completed、canceled、in_progress、expired | 同上 |
| 可选协议 / 买方 | protocol:{version}（version 日期 string 必需，protocol 本身 O）、buyer | AC ProtocolVersion、CheckoutSessionBase |
| 可选币种 / 本地化 | presentment_currency、exchange_rate:number、exchange_rate_timestamp:date-time、locale、timezone | AC 第 2796–2814 行 |
| 可选履约 | fulfillment_details、selected_fulfillment_options[]、fulfillment_groups[] | AC 第 2820–2840 行 |
| 可选时效 / 恢复 | created_at、updated_at、expires_at、quote_expires_at:date-time；continue_url:URI、quote_id | AC 第 2861–2887 行 |
| 可选其他 | authentication_metadata、metadata:object、marketing_consent_options[]、discounts | AC 第 2859–2895 行 |
| order | complete WithOrder 必需；AJ Base 另声明可选 order，AC Base 没有该属性 | AC CheckoutSessionWithOrder；AJ |

核心嵌套类型，字段未写 R 的为 O；无特别说明的标识 / 描述为 string：

| 对象 | 全部字段 / 类型 / 约束 | 来源 |
| --- | --- | --- |
| Item | R id；O name、unit_amount:integer；**当前 AC 没有 quantity** | AC 第 991–1010 行 |
| LineItem | R id、item、quantity:integer≥1、totals[]；O name、description、images:URI[]、unit_amount:int、disclosures[]、custom_attributes[]、marketplace_seller_details、product_id、sku、variant_id、category、tags:string[]、weight、dimensions、availability_status、available_quantity:int≥0、max_quantity_per_order:int≥1、fulfillable_on:date-time、variant_options[]、discount_details[]、tax_exempt:bool、tax_exemption_reason、parent_id | AC 第 1261–1373 行 |
| 商品补充 | VariantOption R name/value；Weight R value:number、unit:g/kg/oz/lb；Dimensions R length/width/height:number、unit:cm/in；CustomAttribute R display_name/value；MarketplaceSellerDetails R name；availability enum 为 in_stock/low_stock/out_of_stock/backorder/pre_order | AC 第 1046–1074、1154–1222、1261–1373 行 |
| Disclosure | R type=disclaimer、content_type:plain/markdown、content；CommonMark 0.31.2，禁止 raw HTML | AC 第 1011–1045 行 |
| DiscountDetail | R type:percentage/fixed/bogo/volume、amount:int；O code、description、source:coupon/automatic/loyalty | AC 第 1223–1260 行 |
| Total | R type、display_text、amount:int；O presentment_amount:int、description、breakdown[]；type 为 items_base_amount/items_discount/subtotal/discount/fulfillment/tax/fee/gift_wrap/tip/store_credit/total/amount_refunded；breakdown 项 R jurisdiction、rate:number、amount:int | AC 第 1387–1449 行 |
| Address | R name、line_one、city、state、country、postal_code；O line_two；AJ 另有 company | AC 第 495–535 行；AJ 第 146–190 行 |
| FulfillmentDetails | 全 O：name、phone_number、email、address | AC 第 639–668 行 |
| Buyer | R email；O first_name、last_name、full_name、phone_number、customer_id、account_type:guest/registered/business、authentication_status:authenticated/guest/requires_signin、company、loyalty、tax_exemption | AC 第 748–802 行 |
| Company / Loyalty / TaxExemption | Company R name，O tax_id/department/cost_center；Loyalty 全 O tier、points_balance:int、member_since:date-time；TaxExemption R certificate_id、certificate_type:resale/exempt_organization/government，O exempt_regions:string[]、expires_at | AC 第 669–747 行 |
| FulfillmentOptionShipping | R type=shipping、id、title、totals[]；O description、carrier、earliest_delivery_time、latest_delivery_time | AC 第 1599–1636 行 |
| FulfillmentOptionDigital | R type=digital、id、title、totals[]；O description | AC 第 1646–1672 行 |
| FulfillmentOptionPickup | R type=pickup、id、title、location、totals[]；location R name/address，O phone/instructions；O description、pickup_type:in_store/curbside/locker、ready_by、pickup_by | AC 第 1458–1518 行 |
| FulfillmentOptionLocalDelivery | R type=local_delivery、id、title、totals[]；O description、delivery_window:{start/end:date-time R}、service_area:{radius_miles:number,center_postal_code:string} | AC 第 1533–1584 行 |
| SelectedFulfillmentOption | R type:shipping/digital/pickup/local_delivery、option_id、item_ids:string[]；扁平对象 | AC 第 1680–1704 行 |
| FulfillmentGroup | R id、item_ids:string[]、destination_type:shipping/pickup/local_delivery/digital；O fulfillment_details、location_id、instructions | AC 第 2059–2091 行 |
| Link | R type、url；O title；type 为 terms_of_use/privacy_policy/return_policy/shipping_policy/contact_us/about_us/faq/support | AC 第 1952–1977 行 |
| RiskSignals（checkout） | 全 O：ip_address、user_agent、accept_language、session_id、device_fingerprint；不同于 delegate 的 RiskSignal[] | AC 第 3064–3086 行 |
| IntentTrace | R reason_code；O trace_summary≤500、metadata（平面 string/number/bool）；reason enum：price_sensitivity/shipping_cost/shipping_speed/product_fit/trust_security/returns_policy/payment_options/comparison/timing_deferred/other | AC 第 3278–3316 行 |

能力与支付 handler：

| 对象 | 全部字段 | 来源 |
| --- | --- | --- |
| Capabilities | 全 O payment、interventions、extensions；请求 extensions 为唯一 string[]，响应为 ExtensionDeclaration[] | AC 第 870–898 行 |
| InterventionCapabilities | 全 O supported[]:3ds/biometric/address_verification、required[]:3ds/biometric、enforcement:always/conditional/optional、display_context:native/webview/modal/redirect、redirect_context:in_app/external_browser/none、max_redirects:int≥0、max_interaction_depth:int≥1；请求 / 响应各有适用范围 | AC 第 803–861 行 |
| ExtensionDeclaration | R name（标识符 / 可带日期）；O extends:string[]（受限 JSONPath）、schema:URI、spec:URI | AC 第 907–935 行 |
| Payment | R handlers:PaymentHandler[]，未限定非空 | AC 第 1140–1153 行 |
| PaymentHandler | R id、name、version:YYYY-MM-DD、spec:URI、requires_delegate_payment:bool、requires_pci_compliance:bool、psp、config_schema:URI、instrument_schemas:URI[]、config:object；O display_name；AJ 另有 display_order:int，AC 缩进有误 | AC 第 1075–1133 行；AJ 第 889–961 行 |
| PaymentData | anyOf 要求 handler_id+instrument，或 purchase_order_number；可同时满足；O billing_address、payment_terms:immediate/net_15/net_30/net_60/net_90、due_date、approval_required:bool；instrument R type、credential，credential R type/token | AC 第 1998–2058 行 |

两个 requires flag 在 machine schema 必需且无 default；ARH 建议 delegation=true、PCI=false，不等于 schema 自动补值。Card handler 的 config 是 handler-specific：ARH 内嵌 schema 要求 merchant_id、psp、accepted_brands，后者枚举 visa/mastercard/amex/discover/diners/jcb/unionpay；可选 funding types、supports_3ds（default true）、3ds_versions（default ["2.2"]）、environment。[ARH 第 169–181、830–902 行；AC。]

Message / 认证与可选扩展：

| 对象 | 全部字段 / 重要枚举 | 来源 |
| --- | --- | --- |
| MessageInfo | R type=info、content_type:plain/markdown、content；O severity:info/low/medium/high/critical、resolution:recoverable/requires_buyer_input/requires_buyer_review、param | AC 第 1712–1759 行 |
| MessageWarning | 同上但 type=warning，另 R code；code：low_stock/high_demand/shipping_delay/price_change/expiring_promotion/limited_availability，及下述8个 discount error | AC 第 1765–1832 行 |
| MessageError | 同上但 type=error，另 R code：missing/invalid/out_of_stock/payment_declined/requires_sign_in/requires_3ds/low_stock/quantity_exceeded/coupon_invalid/coupon_expired/minimum_not_met/maximum_exceeded/region_restricted/age_verification_required/approval_required/unsupported/not_found/conflict/rate_limited/expired/intervention_required | AC 第 1839–1921 行 |
| HTTP Error | R type:invalid_request/processing_error/service_unavailable、code、message；O param；AJ 另有 supported_versions:日期string[]；无有效 session 时用 flat error | AC 第 3328–3365 行；AJ |
| AuthenticationMetadata | R acquirer_details、directory_server:american_express/mastercard/visa；acquirer R acquirer_bin≤11、acquirer_country长度2、acquirer_merchant_id≤35、merchant_name≤40；O requestor_id≤35；O flow_preference，内 R type=challenge/frictionless，可选对应 subtype | AC 第 3113–3199 行 |
| Checkout AuthenticationResult | R outcome:abandoned/attempt_acknowledged/authenticated/canceled/denied/informational/internal_error/not_supported/processing_error/rejected；authenticated/informational/attempt_acknowledged 要求 outcome_details；其内 R three_ds_cryptogram、electronic_commerce_indicator:01/02/05/06/07、transaction_id、version | AC 第 3203–3273 行 |
| Marketing option / decision | Option R channel、display_text、privacy_policy_url；O is_subscribed，描述省略=false；decision R channel、opted_in:bool；未展示的 channel 应省略，不改原订阅状态 | AC 第 2714–2767、3101–3105 行 |
| AffiliateAttribution | R provider；token/publisher_id 至少之一；O campaign_id、creative_id、sub_id、source、issued_at、expires_at、metadata、touchpoint:first/last；source R type:url/platform/unknown，O url；write-only，token 视为 secret | AC 第 536–638 行 |
| Discounts input / output | Input O codes:string[]，大小写不敏感、替换旧值、[]清除；output O codes、applied、rejected | AC 第 2648–2688 行 |
| AppliedDiscount / Allocation | Applied R id/coupon/amount:int≥0；O code、automatic:bool=false、start/end、method:each/across、priority:int≥1、allocations[]；allocation R path、amount:int≥0 | AC 第 2489–2507、2563–2614 行 |
| Coupon | R id/name；O percent_off:0–100、amount_off:int≥0、currency小写3字母、duration:once/repeating/forever、duration_in_months≥1、max_redemptions≥1、times_redeemed≥0、metadata:string map | AC 第 2508–2562 行 |
| RejectedDiscount | R code/reason；O message；reason=`discount_code_` + expired/invalid/already_applied/combination_disallowed/minimum_not_met/user_not_logged_in/user_ineligible/usage_limit_reached | AC 第 2615–2647 行 |

3DS cryptogram 的文字称20字节 / Base64 28字符，但类型只声明 string；Coupon 互斥 / 币种条件也主要在 description，不是完整 machine-level if/then 合同。[AC 第 3230–3234、2522–2559 行。]

#### 4.2.5 Delegated payment / vault token：完整端点与字段

唯一端点为 `POST /agentic_commerce/delegate_payment`，当前合同只接受 card credential；它是 tokenization，不定义 PSP 的 authorization / capture API。[AD Paths；ARD Scope。]

请求的 payment_method、allowance、risk_signals[]、metadata 均 R，billing_address O，顶层禁止其他属性。成功 201 返回 id:string、created:date-time、metadata:string map，三者 R；没有标准的 response.object / type / allowance 属性。[AD 第 295–323、588–616 行。]

Authorization、Content-Type、API-Version、Idempotency-Key R；Signature / Timestamp / Accept-Language / User-Agent / Request-Id O。错误响应列400/401/409/422/429/500/503；in-flight 409 包含 Retry-After。[AD 第 110–293 行。]

| 对象 | R 字段 | O 字段 / 约束 | 来源 |
| --- | --- | --- | --- |
| PaymentMethodCard | type=card、card_number_type:fpan/network_token、number:string、display_card_funding_type:credit/debit/prepaid、metadata:string map | exp_month≤2、exp_year≤4、name、cvc≤4、cryptogram、eci_value≤2、checks_performed[]:avs/cvv/ani/auth0、iin≤8、display_wallet_type、display_brand、display_last4恰4数字、virtual:bool | AD 第 364–451 行 |
| Allowance | reason=one_time、max_amount:int minor units、currency小写3字母、checkout_session_id、merchant_id≤256、expires_at:date-time | 没有定义 max_amount 最小值、默认 expiry 或最长 TTL | AD 第 518–559 行 |
| RiskSignal | type=card_testing、score:int、action:blocked/manual_review/authorized | score 无范围；risk_signals 数组允许空 | AD 第 560–587、308–312 行 |
| Billing Address | name≤256、line_one≤60、city≤60、state、country长度2、postal_code≤20 | line_two≤60 | AD 第 470–517 行 |
| Error | type:invalid_request/rate_limit_exceeded/processing_error/service_unavailable、code、message | param；ADJ 另有 supported_versions；code enum 为 invalid_card/duplicate_request/idempotency_conflict/too_many_requests/idempotency_key_required/idempotency_in_flight | AD 第 617–654 行；ADJ |

Token 只能在 allowance 内使用，到达 expires_at 必须失效；handler RFC 还描述单次使用、merchant / session 绑定。所读 API 未定义标准 redemption / revoke / retrieve / refund endpoint，不能补出生命周期 API。[ARD 第 62–65 行；ARH 第 309–313、450–477 行；AD Paths。]

SPT credential 的 RFC 内嵌 schema 要求 token 匹配 `^vt_[a-zA-Z0-9]+$`，但 checkout 示例给 `spt_123`，通用 PaymentData 又没限制前缀；应保留此差异。[ARH 第 731–775 行；AC PaymentData；AEX complete 示例。]

3DS 辅助接口与 vault token API 分开：

| 接口 / 对象 | 可确认合同 | 来源 |
| --- | --- | --- |
| 认证接口 | POST `/delegate_authentication`、POST `…/{authentication_session_id}/authenticate`、GET `…/{authentication_session_id}` | AA Paths |
| Create | R merchant_id、payment_method、amount；O acquirer_details、channel、checkout_session_id、flow_preference、challenge_notification_url、shopper_details；card R type/number/exp_month/exp_year/name；amount R value:int、currency大写3字母 | AA 第 375–444、542–588 行 |
| Authenticate | R fingerprint_completion:Y/N/U；O channel、checkout_session_id、challenge_notification_url、shopper_details | AA 第 446–472 行 |
| Session | R authentication_session_id/status；O action；status 为 action_required/pending/not_supported/authenticated/attempted/not_authenticated/rejected/unavailable/expired/challenge_abandoned | AA 第 475–539 行 |
| Action | type=fingerprint/challenge，对应对象 R；fingerprint R three_ds_method_url/three_ds_server_trans_id；challenge R acs_url/acs_trans_id/three_ds_server_trans_id/message_version | AA 第 761–825 行 |
| 认证 provider 结果 | R trans_status、transaction_id、three_ds_server_trans_id、version；O electronic_commerce_indicator、three_ds_cryptogram、authentication_value、trans_status_reason、cardholder_info；**不同于 checkout outcome/outcome_details** | AA 第 827–865 行；AC AuthenticationResult |

#### 4.2.6 Webhook：完整 Order、状态与嵌套对象

商户向 `POST /agentic_checkout/webhooks/order_events` 发送订单事件；规格描述为 OpenAI-hosted receiver，但 servers 中的 example URL 不是本次核实的真实部署。[AW 第 5–6、24–41 行。]

| 外层字段 | 合同 | 来源 |
| --- | --- | --- |
| Merchant-Signature header | R，HMAC 签名；不是 checkout 的 Signature | AW 第 42–50 行 |
| Content-Type header | R application/json | AW |
| Request-Id / Timestamp header | O；Timestamp 为 ISO date-time，独立于签名内 Unix t | AW 第 51–65 行 |
| type | R string，定义 order_create/order_update，允许未知事件值 | AW 第 221–231 行 |
| data | R EventDataOrder：完整 Order，不是 delta；额外要求 type=order、status | AW 第 213–231 行 |
| ACK | 200，R received:bool、O request_id:string；文件只给 schema，无现成 JSON 字面实例 | AW 第 154–164 行 |

订单字段的主体在 AC；webhook 应发送可用的 line_items / fulfillments / adjustments / totals。付款状态变化本身仍以商户 / PSP 为准，webhook 承载同步。[AW Description；ARC。]

| 对象 | 字段与状态 | 来源 |
| --- | --- | --- |
| Order | R id、checkout_session_id、permalink_url:URI；O type=order、order_number、status、estimated_delivery、confirmation、support、line_items[]、fulfillments[]、adjustments[]、totals[]；AJ 另有 client_reference_id。status 开放 string，定义 created/confirmed/manual_review/processing/shipped/completed/canceled | AC 第 2163–2219 行；AJ |
| OrderLineItem | R id/title/quantity；O product_id、description、image_url:URI、url:URI、unit_price:int、subtotal:int、totals[]、status（processing/partial/fulfilled/removed，开放） | AC 第 2228–2272 行 |
| Quantity | R ordered:int≥1、current:int≥0；O fulfilled:int≥0 default0；current=0→removed，fulfilled=current→fulfilled，0<fulfilled<current→partial，否则processing | AC 第 2266–2305 行 |
| Fulfillment | R id、type:shipping/pickup/digital；O status、line_items[]、carrier、tracking_number、tracking_url、destination、estimated_delivery、digital_delivery、description、events[]。status 开放，定义 pending/processing/shipped/in_transit/out_for_delivery/ready_for_pickup/delivered/failed/canceled | AC 第 2324–2390 行 |
| DigitalDelivery / EstimatedDelivery | Digital 全 O access_url、license_key、expires_at；EstimatedDelivery R earliest/latest:date-time | AC 第 2099–2113、2365–2379 行 |
| FulfillmentEvent | R id/type/occurred_at；O description/location；type 开放，定义 processing/shipped/in_transit/out_for_delivery/ready_for_pickup/delivered/failed_attempt/returned_to_sender/canceled/undeliverable | AC 第 2398–2422 行 |
| Adjustment | R id/type/occurred_at/status；O line_items[]、amount:int、currency小写3字母、description、reason；type 开放：refund/credit/return/exchange/price_adjustment/cancellation/dispute；status 开放：pending/completed/failed | AC 第 2427–2467 行 |
| LineItemReference | R id、quantity:int≥1 | AC 第 2306–2320 行 |
| Confirmation | 全 O confirmation_number、confirmation_email_sent:bool、receipt_url、invoice_number；AJ 另有 order_notes | AC 第 2118–2142 行；AJ |
| Support | 全 O email、phone、hours、help_center_url | AC 第 2143–2162 行 |

Order total 始终指原始扣款，amount_refunded 为累计退款；旧 refunds[] 被移除，售后调整应放 adjustments[]，但 JSON examples 仍有旧字段，见下文。[AC 第 2211–2215 行；AW Migration Notes；AEX。]

#### 4.2.7 密码学、签名字节、重放与幂等

Webhook 是快照内最明确的签名字节合同：

```text
Merchant-Signature: t=<unix_seconds>,v1=<64_hex>
signed_payload = timestamp + "." + raw_body
signature = HMAC-SHA256(signed_payload, shared_secret)
```

Header pattern 为 `^t=\d+,v1=[a-fA-F0-9]{64}$`；签收到的**原始 body**，不是 JSON parse / 排序 / 重序列化结果。缺失、格式错误、过期、验签失败均401；建议时间容差300秒。[AW 第 8–13、43–50、170–199 行。]

合同没有完整定义字符 / 字节编码边界、恒定时间比较、secret 发放 / 轮换、多签名兼容、nonce / event 去重。外层也没有独立 event ID；时间窗不能单独证明窗口内重复投递只处理一次。这是对所列全部字段和签名规则的静态分析，不是断言某个部署没有额外控制。[AW Parameters、WebhookEvent。]

普通请求不同于 webhook：

| 项目 | 已定义内容与边界 | 来源 |
| --- | --- | --- |
| Checkout 签名 | SHOULD canonical JSON 签名 / RFC3339 Timestamp；算法带外发布；无统一 suite / 固定 JCS | ARC 第 36–38、69–70、322–327 行 |
| Delegate payment 签名 | 初始化举 Ed25519 / ES256；准备阶段 MUST canonical serialize、detached private-key sign、base64url Signature 与 Timestamp；header / security 又写 RECOMMENDED / SHOULD，OpenAPI O | ARD 第 29–55、272–278 行；AD |
| 覆盖范围 | 明确描述 request JSON，但未统一规定 method/path/API-Version/Idempotency-Key/Timestamp 如何进入签名字节，也缺统一 key ID / grammar / 验证向量 | ARC；ARD |
| Freshness | 要求有界 clock skew，但 checkout/payment 没统一数值窗口或 nonce cache；不能套用 webhook 的300秒 | ARC；ARD |
| Transport / PCI | checkout/payment RFC 写 HTTPS / TLS1.3、禁止记录完整 PAN/CVC；authentication RFC 为 TLS1.2+ | ARC；ARD；`rfcs/rfc.delegate_authentication.md@ACP` |

幂等是重复副作用控制，不等于统一请求签名协议：

| 规则 | 合同 | 来源 |
| --- | --- | --- |
| 适用与 scope | checkout 四类 POST / delegate payment R key；opaque≤255字符、建议UUIDv4；identity+endpoint | ARC 第 232–249 行；ARD 第 187–204 行 |
| Body 比较 | 语义 JSON，仅 body；忽略对象key顺序，null≠absent、1.0=1、数组顺序有意义，active extension参与 | ARC 第 251–262、309–311 行；ARD 第 206–217 行 |
| JCS / SHA-256 | MAY 用于幂等 body fingerprint，**不是固定请求签名算法** | ARC 第 262 行；ARD 第 217 行 |
| 同 key 同 body | 返回原响应 / HTTP status，不重复付款、库存或vault创建；SHOULD附Idempotent-Replayed:true | ARC 第 264–269 行；ARD 第 219–223 行 |
| 缺 key | 400 idempotency_key_required | ARC；ARD |
| 同 key 不同 body | 422 idempotency_conflict，永久不可重试 | ARC；ARD |
| 原请求进行中 | 409 idempotency_in_flight，可按Retry-After重试；RFC说SHOULD，OpenAPI标R | ARC 第 271–297 行；AC 第 464–481 行 |
| 5xx与保存期 | 5xx不得缓存，原key重试作fresh request；映射至少24小时，过期key作新请求 | ARC 第 299–307 行；ARD 第 253–261 行 |
| 崩溃一致性 | SHOULD与业务同ACID事务；外部PSP副作用SHOULD recovery points；本仓库无运行实现保证 | ARC 第 313–318 行；ARD 第 263–268 行 |

#### 4.2.8 核心 JSON 原文裁剪

只删除示例包装 / 非核心字段并压缩空白，保留原字段和值。**响应多为片段，不满足完整响应所有 required 字段；原例本身也可能与 schema 不一致。压缩后的 JSON 不能当作 webhook 的原始签名字节。**

Create 请求与响应片段。[AEX create 示例。]

```json
{
  "currency": "usd",
  "line_items": [{"id": "item_123"}],
  "capabilities": {
    "interventions": {
      "supported": ["3ds", "address_verification"],
      "display_context": "webview",
      "redirect_context": "in_app",
      "max_redirects": 1,
      "max_interaction_depth": 1
    }
  }
}
```

```json
{
  "id": "checkout_session_123",
  "protocol": {"version": "2026-04-17"},
  "status": "ready_for_payment",
  "currency": "usd",
  "line_items": [{
    "id": "line_item_123",
    "item": {"id": "item_123"},
    "quantity": 1,
    "totals": [{"type": "total", "display_text": "Total", "amount": 330}]
  }]
}
```

PaymentHandler 原例裁剪；config 是 handler-specific，而非通用固定内部字段。[AEX 第 65–87 行。]

```json
{
  "id": "card_tokenized",
  "name": "dev.acp.tokenized.card",
  "version": "2026-01-22",
  "spec": "https://acp.dev/handlers/tokenized.card",
  "requires_delegate_payment": true,
  "requires_pci_compliance": false,
  "psp": "stripe",
  "config_schema": "https://acp.dev/schemas/handlers/tokenized.card/config.json",
  "instrument_schemas": [
    "https://acp.dev/schemas/handlers/tokenized.card/instrument.json"
  ],
  "config": {
    "merchant_id": "acct_1234567890",
    "psp": "stripe",
    "accepted_brands": ["visa", "mastercard", "amex", "discover"]
  },
  "display_order": 1
}
```

Update 请求和响应片段；order_notes 在 AJ 有定义、AC 没有，保留原例而不偷偷修正。[AEX 第 285–318 行；AJ；AC。]

```json
{
  "selected_fulfillment_options": [{
    "type": "shipping",
    "option_id": "fulfillment_option_456",
    "item_ids": ["item_123"]
  }],
  "order_notes": "Please ring doorbell twice. Leave with neighbor at #12 if no answer."
}
```

```json
{
  "id": "checkout_session_123",
  "protocol": {"version": "2026-04-17"},
  "capabilities": {
    "payment": {"handlers": []},
    "interventions": {"supported": [], "required": [], "enforcement": "conditional"}
  },
  "status": "ready_for_payment",
  "currency": "usd"
}
```

GET 无 JSON 请求 body，原响应裁剪如下。[AEX 第 678–770 行；AC Get。]

```json
{
  "id": "checkout_session_123",
  "protocol": {"version": "2026-04-17"},
  "status": "ready_for_payment",
  "currency": "usd",
  "totals": [{"type": "total", "display_text": "Total", "amount": 430}]
}
```

Complete 请求 / 响应，`spt_123` 是原例，不符合 ARH 内嵌 vt_ pattern；order.client_reference_id 仅 AJ 定义。[AEX 第 428–443、518–534、665–675 行；ARH；AJ。]

```json
{
  "payment_data": {
    "handler_id": "card_tokenized",
    "instrument": {
      "type": "card",
      "credential": {"type": "spt", "token": "spt_123"}
    }
  }
}
```

```json
{
  "id": "checkout_session_123",
  "protocol": {"version": "2026-04-17"},
  "status": "completed",
  "currency": "usd",
  "order": {
    "id": "ord_abc123",
    "checkout_session_id": "checkout_session_123",
    "order_number": "#1001",
    "client_reference_id": "chatgpt_txn_abc789",
    "permalink_url": "https://www.testshop.com/orders/ord_abc123"
  }
}
```

Cancel 请求 / 响应。[AEX 第 811–840 行。]

```json
{
  "intent_trace": {
    "reason_code": "shipping_cost",
    "trace_summary": "User loves the item but isn't willing to pay $10 for standard shipping.",
    "metadata": {"target_shipping_cost": 0, "competitor_reference": "amazon_prime"}
  }
}
```

```json
{
  "id": "checkout_session_123",
  "protocol": {"version": "2026-04-17"},
  "status": "canceled",
  "currency": "usd"
}
```

Delegate-payment 请求：卡号是仓库测试样例；原例过往 expires_at 原样保留，不代表观察日可使用。[ADEX 第 2–29 行。]

```json
{
  "payment_method": {
    "type": "card",
    "card_number_type": "fpan",
    "virtual": false,
    "number": "4242424242424242",
    "exp_month": "11",
    "exp_year": "2026",
    "name": "Jane Doe",
    "cvc": "223",
    "display_card_funding_type": "credit",
    "metadata": {"issuing_bank": "temp"}
  },
  "allowance": {
    "reason": "one_time",
    "max_amount": 2000,
    "currency": "usd",
    "checkout_session_id": "csn_01HV3P3XYZ9ABC",
    "merchant_id": "acme_store",
    "expires_at": "2025-10-09T07:20:50.52Z"
  },
  "risk_signals": [{"type": "card_testing", "score": 10, "action": "manual_review"}],
  "metadata": {"campaign": "q4", "source": "chatgpt_checkout"}
}
```

成功 vault token 与错误原例。[ADEX 第 39–74 行。]

```json
{
  "id": "vt_01J8Z3WXYZ9ABC",
  "created": "2025-09-29T11:00:00Z",
  "metadata": {
    "source": "agent_checkout",
    "merchant_id": "acme_store",
    "idempotency_key": "idem_abc123"
  }
}
```

```json
{
  "type": "invalid_request",
  "code": "invalid_card",
  "message": "Invalid card number provided",
  "param": "payment_method.number"
}
```

Checkout 3DS 结果原例；不是 authentication provider 的 trans_status 结构。[AEX 第 1180–1194 行。]

```json
{
  "outcome": "authenticated",
  "outcome_details": {
    "three_ds_cryptogram": "ABCDEFGHIJKLMNOPQRSTUVWX==",
    "electronic_commerce_indicator": "05",
    "transaction_id": "dsTransId_12345",
    "version": "2.2.0"
  }
}
```

**Webhook JSON 原例已过时，不是当前正确模板**：缺 Order.id，并保留被删除的 refunds；当前 YAML 示例采用完整 Order / adjustments。按“原文 JSON”要求保留错误而不重新构造正确例。[AEX 第 958–981 行；AW 第 17–23、73–153、213–219 行。]

```json
{
  "type": "order_create",
  "data": {
    "type": "order",
    "checkout_session_id": "checkout_session_123",
    "permalink_url": "https://www.testshop.com/orders/checkout_session_123",
    "status": "created",
    "refunds": []
  }
}
```

```json
{
  "type": "order_update",
  "data": {
    "type": "order",
    "checkout_session_id": "checkout_session_123",
    "permalink_url": "https://www.testshop.com/orders/checkout_session_123",
    "status": "shipped",
    "refunds": [{"type": "original_payment", "amount": 100}]
  }
}
```

#### 4.2.9 SDK / 实现计数、治理及其他已发布表面

本仓库没有观察到 checkout / vault / webhook runtime SDK 或服务端源码包；package.json 是 specification + devDependencies，实际工具代码为 JavaScript validator 和 shell 辅助脚本。README 声明 OpenAI、Stripe 最先实现，故只能记录**仓库宣称两家外部参考实现，本地可核查 runtime SDK 为0**；外部 SDK 的包数、语言与当前状态未知。[`package.json@ACP`；`scripts/README.md@ACP`；`scripts/validate-consistency.js@ACP`；`README.md@ACP` 第 102–116 行；ACP 提交的跟踪文件树。]

许可为 Apache-2.0，NOTICE 版权主体为2025 OpenAI / Stripe。MAINTAINERS 列 OpenAI、Stripe、Meta；governance 描述3个当前 TSC 席位、上限7，OpenAI/Stripe 为 founders，并有任免及有限 veto 权限；转向中立 foundation 仍是未来目标。[`LICENSE@ACP`；`NOTICE@ACP`；`MAINTAINERS.md@ACP`；`docs/governance.md@ACP` 第 44–100、225–278 行。]

普通 PR 要求至少2个 approval；重大改变经 SEP、TSC sponsor、7工作日公开 review、50%+1多数。CLA 必需；但 CONTRIBUTING 仍写至少1 maintainer review，与 governance 不同步。[`docs/governance.md@ACP` 第 151–214 行；`CONTRIBUTING.md@ACP`；`.github/CODEOWNERS@ACP`。]

| 额外表面 | 已发布资产 / 边界 | 来源 |
| --- | --- | --- |
| Discovery | R protocol/api_base_url/transports/capabilities；protocol R name=acp/version/supported_versions（旧→新），O documentation_url；transports rest/mcp；capabilities R services:checkout/orders/delegate_payment/carts，O extensions/intervention_types/supported_currencies/supported_locales | AJ Discovery schema；`rfcs/rfc.discovery.md@ACP` |
| Cart | POST /carts、GET /carts/{id}、PUT /carts/{id}、POST /carts/{id}/cancel；PUT全量替换line_items，totals非权威，无付款或自动转checkout | `spec/2026-04-17/openapi/openapi.cart.yaml@ACP`；`rfcs/rfc.cart.md@ACP` |
| Feed | POST /feeds、GET /feeds/{id}、GET /feeds/{id}/products、PATCH /feeds/{id}/products；merchant推送至Agent-hosted服务，文件全替换、PATCH partial upsert | `rfcs/rfc.product_feeds.md@ACP`；`changelog/2026-04-17.md@ACP` |
| MCP | 五个 checkout tool：create_checkout_session、get_checkout_session、update_checkout_session、complete_checkout_session、cancel_checkout_session；OpenRPC是接口描述，不是5个SDK | `spec/2026-04-17/openrpc/openrpc.agentic_checkout.json@ACP` |
| MCP字段 | meta R api_version，O idempotency_key/request_id/user_agent/accept_language/signature/timestamp；get/update/complete/cancel R id；create/update/complete R payload，cancel payload O；auth带外；ACP错误在JSON-RPC -32000的error.data | `docs/mcp-binding.md@ACP`；同 OpenRPC 文件 |
| Spec资产计数 | 发布清单列7个JSON Schema、6个OpenAPI、1个OpenRPC，不是runtime实现数 | `changelog/2026-04-17.md@ACP` 第 633–649 行 |

MCP 的 delegate-payment binding、discovery JWK 广告、cart→checkout 桥接、feed促销/增量cursor/partial删除、seller-backed split payments 等仍明确延期；不是已经实现的能力。[`docs/mcp-binding.md@ACP` 第 243–253 行；`rfcs/rfc.discovery.md@ACP` 第 358–366 行；`rfcs/rfc.cart.md@ACP`；`rfcs/rfc.product_feeds.md@ACP`；`rfcs/rfc.seller_backed_payment_handler.md@ACP` 第 931–933 行。]

#### 4.2.10 具体契约不一致与静态 TODO

以下结论仅来自固定 SHA；不是当前开放 issue 清单，也不宣称运行验证器得到这些失败。

| 项目 | 差异与影响 | 来源 |
| --- | --- | --- |
| API版本头不同步 | Checkout / payment / auth RFC页首旧日期，不等于最新AC/AD/AA标记 | ARC；ARD；`rfcs/rfc.delegate_authentication.md@ACP`；AC/AD/AA |
| Create数量入口缺失 | 请求line_items引用Item，Item只有id/name/unit_amount；quantity仅在响应LineItem。此份OpenAPI不能正常表达多件购买 | AC Item、Request、LineItem |
| 发布日志数量承诺不同 | 日志称Item quantity改number，现文件Item无quantity、LineItem为integer | `changelog/2026-04-17.md@ACP` 第 164–170 行；AC |
| JSON Schema与OpenAPI字段差异 | company、order_notes、client_reference_id、supported_versions等仅AJ/ADJ有；可能使生成客户端和校验规则分裂 | AJ/ADJ；对应AC/AD对象 |
| Complete组合schema矛盾 | AC Base additionalProperties:false且没有order，WithOrder以allOf要求order；按组合语义互相冲突；AJ Base已补order | AC 第 2768–2905、2937–2946 行；AJ |
| Auth WithResult类似问题 | Base禁止未知属性却没authentication_result，allOf另一分支再加该字段；本次未跑校验器 | AA 第 475–539 行；`spec/2026-04-17/json-schema/schema.delegate_authentication.json@ACP` |
| Handler display_order缩进错误 | AC放在properties之外，AJ正常在properties内 | AC 第 1117–1123 行；AJ 第 942–949 行 |
| 内嵌例多为旧形状 | items代替line_items；旧token/provider；LineItem缺item/totals；Total为金额map；handler缺必需字段 | AC 第 45–50、253–260、1134–1139、1374–1385、1450–1456、3021–3026 行 |
| Webhook JSON旧例 | 缺id并仍含refunds，开发版JSON也残留；与当前Order / webhook不符 | AEX；`examples/unreleased/examples.agentic_checkout.json@ACP`；AW |
| Vault RFC与schema不同 | RFC要求virtual / 至少1 risk signal，AD的virtual O且数组可空；发布日志已移除minItems | ARD 第 100–134 行；AD；`changelog/2026-04-17.md@ACP` 第 103–109 行 |
| Error enum与HTTP例不符 | 401 unauthorized、429 rate_limit_exceeded、500 internal_server_error、503 service_unavailable未完全落入AD Error enums；版本错误code也有缺项 | AD 第 128–211、617–654 行；ADJ；ARD |
| SPT/vault响应不统一 | RFC中vt_ pattern与AEX spt_不同；handler流程响应有created数字和object/type/allowance，AD要求created date-time且仅id/created/metadata | ARH 第 378–397、749–753 行；AEX；AD |
| Seller-backed合同缺口 | RFC声称无需API改变，但例带顶层handler_id、非card type并省略risk_signals/metadata，与AD限制冲突 | `rfcs/rfc.seller_backed_payment_handler.md@ACP` 第 336–395、937–939 行；AD |
| MCP幂等不统一 | meta只要求api_version，idempotency_key O；REST全POST要求key，binding安全节又称语义相同 | `docs/mcp-binding.md@ACP` 第 107–115、261–263 行；OpenRPC；ARC |
| Complete开发版409模糊 | changelog说地址改变409保护，字段描述仅写付款前应用履约更新，409仍指幂等in-flight | `changelog/unreleased/fulfillment-details-on-complete.md@ACP`；`spec/unreleased/openapi/openapi.agentic_checkout.yaml@ACP` |
| 校验工具盲点 | OpenAPI example检查仅确认是否有例，不验证例的schema；JSON实例模型推断只覆盖部分，不明模型/缺schema等可跳过。本次只读代码，未跑工具/CI | `scripts/validate-consistency.js@ACP` 第 487–635 行 |
| 治理文件不同步 | README强调两家、MAINTAINERS已有Meta；贡献文档1review vs governance2approvals | `README.md@ACP`；`MAINTAINERS.md@ACP`；`CONTRIBUTING.md@ACP`；`docs/governance.md@ACP` |

未确认外部 issue / PR 当前状态。源码变更说明提到 #161 的 Error enum gap，只能作为该文件自述的问题线索，不能写成“#161今天仍open”。[源线索：`changelog/unreleased/unwrap-error-envelope-released-delegate-payment.md@ACP`。]

### 4.3 Coinbase x402

直接网络克隆 E4 失败，后成功取得离线浅克隆 X。快照的规范链接使用 `x402-foundation/x402`，既有来源仓库的 `.git/config` 也记录该远端；因此报告采用该命名定位源码，但不能从这些证据推断仓库迁移发生的具体日期。[`specs/x402-specification-v2.md@X` §1；来源快照 `.git/config` 的 `remote.origin.url`。]

#### 4.3.1 源码引用索引与版本

本节短引用展开如下；**每个 `@X` 均为 SHA `faa1c694c66a3732aaabbce93bdae140e521a67d`**，不是网页分支。表格中的同一短引用可用于定位该表全部字段；小节号对应原文。

| 引用 | 仓库文件路径 |
| --- | --- |
| X1 | `specs/x402-specification-v1.md@X` |
| X2 | `specs/x402-specification-v2.md@X` |
| XH1 | `specs/transports-v1/http.md@X` |
| XH2 | `specs/transports-v2/http.md@X` |
| XMigration | `docs/guides/migration-v1-to-v2.mdx@X` |
| XTypes | `typescript/packages/core/src/types/payments.ts@X` |
| XFacTypes | `typescript/packages/core/src/types/facilitator.ts@X` |
| XSchema | `typescript/packages/core/src/schemas/index.ts@X` |
| XPyTypes | `python/x402/schemas/payments.py@X` |
| XGoTypes | `go/types/v2.go@X` |
| XHTTP | `typescript/packages/core/src/http/index.ts@X` |
| XHTTPClient | `typescript/packages/core/src/http/x402HTTPClient.ts@X` |
| XFacHTTP | `typescript/packages/core/src/http/httpFacilitatorClient.ts@X` |
| XFlow | `typescript/packages/core/src/server/paymentFlow.ts@X` |
| XEVM | `specs/schemes/exact/scheme_exact_evm.md@X` |
| XSVM | `specs/schemes/exact/scheme_exact_svm.md@X` |
| XHTTPSig | `specs/extensions/http-message-signatures.md@X` |

快照同时保留协议版本 1 和 2；v2 主规范明确标记 `Protocol Version: 2`。以下是文件内历史，不等同于 GitHub Release 或 SDK 包的发布日期。[X1、X2 文首及 Version History。]

| 文档历史标识 | 原文日期 | 变化 | 来源 |
| --- | --- | --- | --- |
| `v0.1` | `2025-8-29` | 初稿 | X1、X2 Version History |
| `v0.2` | `2025-10-3` | 传输无关重构 | X1、X2 Version History |
| `v2.0` | `2025-12-9` | CAIP-2、对象重组、资源信息分离、扩展 | X2 Version History |

这里的 `v0.1`、`v0.2` 是文档历史表条目，不能改写成消息中的 `x402Version: 0.1/0.2`；消息版本来自各自 schema。[X1 §5；X2 §5。]

#### 4.3.2 角色与消息顺序

三种核心角色是 client、resource server、facilitator；facilitator API 使用 HTTP，即使 client 与 resource server 之间使用其他传输。[X2 §3、§7。]

默认 `authorization` 流的逐步序列：

1. Client 请求受保护资源。[X2 §2；XH2 Payment Required Signaling。]
2. 无有效支付时，server 返回 HTTP 402，`PAYMENT-REQUIRED` 携带 Base64 编码的 `PaymentRequired`。[XH2 Payment Required Signaling。]
3. Client 选择 `accepts[]` 中一项，按 scheme / network 形成支付授权，并把所选条件放进 `PaymentPayload.accepted`。[X2 §5.2；XEVM Phase 1。]
4. Client 重试请求，通过 `PAYMENT-SIGNATURE` 发送 Base64 编码的完整 `PaymentPayload`。[XH2 Payment Payload Transmission。]
5. Server 调用 facilitator 的 `POST /verify`，提交支付载荷和 server 自己要验证的支付条件；`/verify` 不得提交支付状态或写链。[X2 §7.1；XFacHTTP `verify()`。]
6. 验证通过后执行资源处理，再调用 `POST /settle` 提交支付状态；不同 scheme 的提交未必是同步链上转账。[X2 §6.1、§7.2。]
7. Server 返回业务响应，通过 `PAYMENT-RESPONSE` 发送结算结果；该结果编码不包含 server 内部的 `extensionResponses`。[XH2 Settlement Response Delivery；XHTTP `encodePaymentResponseHeader()`。]

不能把上述顺序泛化成全部 x402 支付模式：

| flow | 顺序 | 含义 | 来源 |
| --- | --- | --- | --- |
| `authorization` | verify → resource → settle → respond | 执行业务前只验证，业务后提交 | X2 §6.1；XFlow `PAYMENT_FLOWS` |
| `upfront` | settle → resource → respond | 先持久提交，再执行资源；不调用 verify | X2 §6.1；XFlow |
| `escrow` | settle → resource → settle → respond | 先提交押金 / 上限，再最终计费 | X2 §6.1；XFlow |

`extra.assetTransferMethod` 和 `extra.paymentFlow` 是保留键。省略时按 mechanism 的默认值解析；非 `authorization` 的实际 flow 必须在挑战中显式广告。SDK 里的 ATM 哨兵 `"default"` 不得直接上 wire。[X2 §6.1；XFlow `resolvePaymentFlow()`、`applyPaymentFlowWireExtra()`。]

#### 4.3.3 核心结构：逐字段

`PaymentRequired`：

| 字段 | 类型与要求 | 含义 | 来源 |
| --- | --- | --- | --- |
| `x402Version` | number，必需且为 2 | 协议版本 | X2 §5.1.2；XSchema |
| `error` | string，可选 | 需支付的说明 | X2 §5.1.2 |
| `resource` | ResourceInfo，规范必需 | 受保护资源 | X2 §5.1.2；实现差异见下文 |
| `accepts` | PaymentRequirements[]，必需 | 可接受的支付选项；TS schema 至少一项 | X2 §5.1.2；XSchema |
| `extensions` | object，可选 | 扩展键值映射 | X2 §5.1.2 |

`PaymentRequirements`：

| 字段 | 类型与要求 | 含义 | 来源 |
| --- | --- | --- | --- |
| `scheme` | string，必需 | Scheme 标识 | X2 §5.1.2 |
| `network` | string，必需 | CAIP-2 `namespace:reference` | X2 §5.1.2、§11.1 |
| `amount` | string，必需 | 原子单位金额，不是浮点价格 | X2 §5.1.2 |
| `asset` | string，必需 | Token 地址，或规范允许的法币 ISO 4217 代码 | X2 §5.1.2 |
| `payTo` | string，必需 | 收款地址，或规范允许的角色常量 | X2 §5.1.2 |
| `maxTimeoutSeconds` | number，必需 | 支付完成允许时限；TS schema 要求正数 | X2 §5.1.2；XSchema |
| `extra` | object，规范可选 | 保留 flow / ATM 键及 scheme 参数 | X2 §5.1.2、§6.1 |

`asset` 可表示法币代码是 schema 的可扩展性，不证明任意 facilitator 已支持法币结算。[X2 §5.1.2、§7.3、§11.2；这是对规范边界的分析。]

`ResourceInfo`：

| 字段 | 类型与要求 | 含义 / 限制 | 来源 |
| --- | --- | --- | --- |
| `url` | string，必需 | 资源 URL | X2 §5.1.2 |
| `description` | string，可选 | 描述 | X2 §5.1.2 |
| `mimeType` | string，可选 | 响应 MIME 类型 | X2 §5.1.2 |
| `serviceName` | string，可选 | 可打印 ASCII，最多 32 字符 | X2 §5.1.2；XSchema |
| `tags` | string[]，可选 | 最多 5 项，每项可打印 ASCII，最多 32 字符 | X2 §5.1.2；XSchema |
| `iconUrl` | string，可选 | 绝对 HTTP(S) URL，最多 2048 字符；TS schema 本身只检查长度 | X2 §5.1.2；XSchema |

`PaymentPayload`：

| 字段 | 类型与要求 | 含义 | 来源 |
| --- | --- | --- | --- |
| `x402Version` | number，必需；v2 TS schema 固定 2 | 版本 | X2 §5.2.2；XSchema |
| `resource` | ResourceInfo，可选 | 被访问资源 | X2 §5.2.2 |
| `accepted` | PaymentRequirements，必需 | Client 选择的完整条件 | X2 §5.2.2 |
| `payload` | object，必需 | Scheme-specific 付款证明 / 授权 | X2 §5.2.2 |
| `extensions` | object，可选 | 协议扩展 | X2 §5.2.2 |

扩展通常为 `extension-name → {info, schema}`；主规范要求二者，并要求 client 保留 server 已给出的 `info`，可追加但不能删改。个别扩展允许省略 schema，存在例外，见缺口小节。[X2 §5.1.2；XHTTPSig Schema Omission。]

EVM exact / EIP-3009 的 `payload` 为 `{signature, authorization}`；签名是十六进制字符串，authorization 内的六个字段全部必需。[X2 §5.2.2；XEVM EIP-3009。]

| `authorization` 字段 | wire 类型 | 签名中的类型 / 含义 | 来源 |
| --- | --- | --- | --- |
| `from` | string | address，付款方 | X2 §5.2.2；`typescript/packages/mechanisms/evm/src/constants.ts@X` |
| `to` | string | address，收款方 | 同上 |
| `value` | string | uint256，原子单位金额 | 同上 |
| `validAfter` | string | uint256，Unix 起始秒数 | 同上 |
| `validBefore` | string | uint256，Unix 截止秒数 | 同上 |
| `nonce` | string | bytes32，随机 32 字节防重放值 | 同上 |

Permit2 exact 的 payload 不是上述授权对象：它含 `signature`、`permit2Authorization`，后者含 `permitted.token`、`permitted.amount`、`from`、`spender`、`nonce`、`deadline`、`witness.to`、`witness.validAfter`。地址均为 hex address；amount / nonce / deadline / validAfter 为十进制字符串。`spender` 应为 exact proxy，不是 facilitator。[`typescript/packages/mechanisms/evm/src/types.ts@X` `ExactPermit2Payload`、`Permit2Authorization`；XEVM Permit2。]

SVM exact 的 payload 为 `{transaction: string}`，值是 Base64 编码的部分签名 versioned transaction；`extra.feePayer` 必需，`memo`、`recentBlockhash`、`lastValidBlockHeight` 可选。memo 为最多 256 字节 UTF-8；block height 为十进制字符串，两个 blockhash 提示不强制将实际交易绑定到该提示值。[XSVM PaymentRequirements、PaymentPayload。]

Facilitator 请求、响应：

| 对象 / 字段 | 类型与要求 | 语义 | 来源 |
| --- | --- | --- | --- |
| `/verify`、`/settle`: `x402Version` | number，必需 | 外层版本 | X2 §7.1–7.2；XFacTypes |
| 两个请求的 `paymentPayload` | PaymentPayload，必需 | Client 提交的证明 | 同上 |
| 两个请求的 `paymentRequirements` | PaymentRequirements，必需 | Server 要求，不能只信任 client 选项 | 同上 |
| VerifyResponse.`isValid` | boolean，必需 | 授权是否有效 | X2 §5.4.2 |
| VerifyResponse.`invalidReason` | string，可选 | 无效原因 | X2 §5.4.2 |
| VerifyResponse.`payer` | string，可选 | 付款人 | X2 §5.4.2 |
| VerifyResponse.`extensions` | object，可选 | 扩展结果 | X2 §5.4.2 |
| VerifyResponse.`extra` | object，可选 | Scheme 附加数据 | X2 §5.4.2 |
| SettleResponse.`success` | boolean，必需 | 提交是否成功 | X2 §5.3.2 |
| SettleResponse.`errorReason` | string，可选 | 失败原因 | X2 §5.3.2 |
| SettleResponse.`payer` | string，可选 | 付款人 | X2 §5.3.2 |
| SettleResponse.`transaction` | string，必需 | 交易 hash；未广播可为空；`settlement_pending` 必须非空 | X2 §5.3.2、§9 |
| SettleResponse.`network` | string，必需 | CAIP-2 网络 | X2 §5.3.2 |
| SettleResponse.`amount` | string，可选 | 实际结算原子金额 | X2 §5.3.2 |
| SettleResponse.`extensions` | object，可选 | 扩展 | X2 §5.3.2 |

TS 类型额外含 `invalidMessage`、`errorMessage`，以及 SettleResponse.`extra`；`extensionResponses` 是内部 sidechannel，不发给买方。HTTP facilitator 可以把扩展结果放进 `EXTENSION-RESPONSES` Base64 JSON header，而不是买方响应 body。[XFacTypes；X2 §7.2.1；XHTTP。]

`GET /supported` 返回 `kinds[]`、`extensions[]`、`signers`，三者必需。每个 kind 含 `x402Version`、`scheme`、`network`，可选 `extra`；signers 为 `eip155:*` / `solana:*` 等 family pattern 到 signer address 数组的映射。[X2 §7.3.1；XFacTypes。]

#### 4.3.4 v1 与 v2：不能混用的线格式

| 项目 | v1 | v2 | 来源 |
| --- | --- | --- | --- |
| 挑战位置 | 402 JSON body | 402 `PAYMENT-REQUIRED` header，Base64 JSON | XH1；XH2 |
| Client 支付头 | `X-PAYMENT`，Base64 JSON | `PAYMENT-SIGNATURE`，Base64 JSON | XH1；XH2；XMigration |
| Server 结算头 | `X-PAYMENT-RESPONSE` | `PAYMENT-RESPONSE` | 同上 |
| 网络 | 如 `base-sepolia` | 如 `eip155:84532` | X1 §5；X2 §11.1 |
| 金额字段 | `maxAmountRequired` | `amount` | X1 §5；X2 §5；XMigration |
| 资源元数据 | 嵌于每个 requirements | 独立 `resource: ResourceInfo` | X1 §5；X2 §5 |
| Payload 选项 | 顶层 `scheme`、`network` | `accepted: PaymentRequirements` | X1 §5.2；X2 §5.2 |
| 可选扩展 | v1 核心对象没有同样的顶层扩展结构 | 顶层 `extensions` | X1 §5；X2 §5 |
| TS 包布局 | `x402`、`x402-express`、`x402-axios` | `@x402/core`、`@x402/express`、`@x402/axios`、机制包 | XMigration |

快照中的 TS HTTP client 按版本生成 `PAYMENT-SIGNATURE` 或 `X-PAYMENT`，并支持读取新旧结算头。因此“v2 改名”和“SDK 保留兼容读取”应分别陈述。[XHTTPClient `encodePaymentSignatureHeader()`、`getPaymentSettleResponse()`。]

#### 4.3.5 签名、规范化与重放边界

EVM EIP-3009 的 client 构造 EIP-712 domain `{name, version, chainId, verifyingContract}`，primary type 为 `TransferWithAuthorization`，签名 message 仅包含上表六个 authorization 字段。整数从 wire string 转 BigInt，地址规范化；不是对原始 JSON 排序或 Base64 字节签名。[`typescript/packages/mechanisms/evm/src/exact/client/eip3009.ts@X`；`typescript/packages/mechanisms/evm/src/constants.ts@X`。]

EOA 实现使用 65-byte ECDSA / secp256k1 恢复；Python 的 typed-data digest 明确为 `keccak256(0x19 || 0x01 || domainSeparator || hashStruct(message))`。合约账户不是简单套用同一种 EOA 算法：TS 按是否有代码选择 ECDSA 或严格 EIP-1271，合约验证失败时不回退 ECDSA；ERC-6492 未部署账户另有 factory allowlist 和 simulation / settle 路径。[`python/x402/mechanisms/evm/verify.py@X` `verify_eoa_signature()`、`_hash_typed_data()`；`typescript/packages/mechanisms/evm/src/shared/verifySignature.ts@X`；`typescript/packages/mechanisms/evm/src/exact/facilitator/eip3009.ts@X`。]

防重放与有效期：

- Nonce 由 `crypto.getRandomValues(new Uint8Array(32))` 生成；EIP-3009 合约的 nonce 消耗和有效时间窗用于阻止再次转账。[`typescript/packages/mechanisms/evm/src/utils.ts@X` `createNonce()`；X2 §10.1。]
- v2 TS client 设置 `validAfter: "0"`、`validBefore = now + maxTimeoutSeconds`；facilitator 检查收款人、金额、network、有效期，截止时间保留至少 6 秒缓冲。[`typescript/packages/mechanisms/evm/src/exact/client/eip3009.ts@X`；`typescript/packages/mechanisms/evm/src/exact/facilitator/eip3009.ts@X`。]
- Settle 会再次 verify；已广播但确认不明时，可返回 `settlement_pending` 并用 pending-settlement store 对相同 signature 关联旧交易，避免简单重广播。默认 store 是内存级，源码要求多副本场景注入共享实现。[`typescript/packages/mechanisms/evm/src/exact/facilitator/eip3009.ts@X` `settleEIP3009()`；`typescript/packages/mechanisms/evm/src/exact/facilitator/scheme.ts@X`；X2 §9。]
- Permit2 使用 nonce / deadline、绑定 receiver 的 witness 和 proxy spender；`upto` witness 还含 facilitator。`upto` settle 重验时应使用签名最大金额，而不是实际扣款额。[`typescript/packages/mechanisms/evm/src/types.ts@X`；`specs/schemes/upto/scheme_upto.md@X`。]
- SVM signer 用 Ed25519 签实际 transaction message bytes，client 与 fee payer 各签自己的位置；交易 lifetime 使用 recent blockhash。SVM 规范另建议短期重复结算缓存，并约定重复错误 `duplicate_settlement`。[`go/signers/svm/client.go@X`；XSVM Protocol Flow、Replay / Duplicate Settlement 部分；`typescript/packages/mechanisms/svm/src/exact/client/scheme.ts@X`。]

安全边界分析：上述 EVM 授权对象没有 HTTP method、URL、body 或 agent 身份字段，故不能把付款授权等同于 HTTP 请求身份签名；HTTP header 的 JSON/Base64 只是传输编码。可选 `http-message-signatures` 扩展才另行描述 RFC 9421 身份签名；它不是 Visa TAP 仓库或 IETF 草案本身。[签名对象：`typescript/packages/mechanisms/evm/src/exact/client/eip3009.ts@X`；编码：XHTTP；扩展：XHTTPSig。]

#### 4.3.6 已观察的规范 / 实现差异

这些是静态契约检查结果，不是经过攻击验证的漏洞，也不是声明仍有某个开放 issue：

1. 主规范允许省略 `PaymentRequirements.extra`，TS 手写 type 却将它声明为必需；TS Zod 接受可选 / null，Python 默认空 dict，Go 标为 `omitempty`。[X2 §5.1.2；XTypes；XSchema；XPyTypes；XGoTypes。]
2. `PaymentRequired.resource` 在主规范与 TS schema 中必需，Python 和 Go 类型却可省略。Python 的 `x402_version` 是 int 默认 2，而 TS schema 使用 literal 2。这是跨 SDK 类型 / 验证契约差异，不证明调用端一定接受错误数据。[X2 §5.1.2；XSchema；XPyTypes；XGoTypes。]
3. TS network schema 只检查最小长度和是否包含冒号；不能把这称为完整 CAIP-2 语法验证。`ResourceInfo.iconUrl` 同样未在这层检查 URL scheme。[XSchema `NetworkSchemaV2`、`ResourceInfoSchema`；对照 X2 §5.1.2、§11.1。]
4. EVM exact 规范列出 EIP-3009、Permit2、ERC-7710 三种 ATM；本次读取的 TS `AssetTransferMethod` union、client dispatcher 与 facilitator router 只实现 EIP-3009 / Permit2。不能据文档中的第三项宣称该 TS dispatcher 已支持 ERC-7710。[XEVM Summary；`typescript/packages/mechanisms/evm/src/types.ts@X`；`typescript/packages/mechanisms/evm/src/exact/client/scheme.ts@X`、`typescript/packages/mechanisms/evm/src/exact/facilitator/scheme.ts@X`。]
5. Permit2 示例标为 JSON，但 `accepted` 后缺逗号，含 `//` 注释及末尾 trailing comma；不能原样当合法 JSON 执行。验证说明引用 `permit2Authorization.amount`，实际类型字段位于 `permit2Authorization.permitted.amount`。[XEVM 原文件第 132–165、189 行；`typescript/packages/mechanisms/evm/src/types.ts@X`。]
6. `http-message-signatures` 的字段说明把 `tags` 标为 required，但同文件 JSON Schema 的 required 数组没有 `tags`；主规范要求扩展 `schema`，该扩展又允许省略。这两项应在实现互通前统一。[XHTTPSig 第 40、55、57 行；X2 §5.1.2。]
7. SVM exact 名称强调 exact，但规范的必需支付结果是至少 required amount，并另讨论 overpayment；不能用 EVM amount equality 概括所有网络。[XSVM §1.1、§1.4；EVM 比较：`typescript/packages/mechanisms/evm/src/exact/facilitator/eip3009.ts@X`。]

远端开放 issue、GitHub release 和正式迁移日期仍未与克隆 SHA 关联确认；不把文件中已有修复说明误记成今天仍未解决的问题。

#### 4.3.7 核心 JSON：仓库原文裁剪

以下仅删除原例中的可选字段或其他元素，不改写字段值。支付地址、签名和时间戳是仓库示例值，不是可用付款凭据；含 `"0x..."` 的原文占位符不可直接提交。

`PaymentRequirements`，原例 `accepts[0]`。[X2 §5.1.1。]

```json
{
  "scheme": "exact",
  "network": "eip155:84532",
  "amount": "10000",
  "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
  "payTo": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C",
  "maxTimeoutSeconds": 60,
  "extra": {"name": "USDC", "version": "2"}
}
```

`PaymentRequired`，省略原例的 error、额外资源描述和空 extensions；该对象经 Base64 编码放入 v2 `PAYMENT-REQUIRED`。[X2 §5.1.1；XH2。]

```json
{
  "x402Version": 2,
  "resource": {"url": "https://api.example.com/premium-data"},
  "accepts": [
    {
      "scheme": "exact",
      "network": "eip155:84532",
      "amount": "10000",
      "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
      "payTo": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C",
      "maxTimeoutSeconds": 60,
      "extra": {"name": "USDC", "version": "2"}
    }
  ]
}
```

`PaymentPayload`，省略可选 resource / extensions；同一 payload 用 v1 包装时顶层是 scheme / network，而非 accepted，不能只替换 header。[X2 §5.2.1；X1 §5.2.1。]

```json
{
  "x402Version": 2,
  "accepted": {
    "scheme": "exact",
    "network": "eip155:84532",
    "amount": "10000",
    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "payTo": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C",
    "maxTimeoutSeconds": 60,
    "extra": {"name": "USDC", "version": "2"}
  },
  "payload": {
    "signature": "0x2d6a7588d6acca505cbf0d9a4a227e0c52c6c34008c8e8986a1283259764173608a2ce6496642e377d6da8dbbf5836e9bd15092f9ecab05ded3d6293af148b571c",
    "authorization": {
      "from": "0x857b06519E91e3A54538791bDbb0E22373e36b66",
      "to": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C",
      "value": "10000",
      "validAfter": "1740672089",
      "validBefore": "1740672154",
      "nonce": "0xf3746613c2d920b5fdabc0856f2aeb2d4f88ee6037b8cc5d04a71a4462f13480"
    }
  }
}
```

Facilitator `/verify` 的原例裁剪，仅省略 paymentPayload.resource；`/settle` 在规范中复用同一种请求结构。示例中的 `"0x..."` 本来就是原文占位符，不是本报告新增的有效地址。[X2 §7.1、§7.2。]

```json
{
  "x402Version": 2,
  "paymentPayload": {
    "x402Version": 2,
    "accepted": {
      "scheme": "exact",
      "network": "eip155:84532",
      "amount": "10000",
      "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
      "payTo": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C",
      "maxTimeoutSeconds": 60,
      "extra": {"name": "USDC", "version": "2"}
    },
    "payload": {
      "signature": "0x...",
      "authorization": {
        "from": "0x...",
        "to": "0x...",
        "value": "10000",
        "validAfter": "1740672089",
        "validBefore": "1740672154",
        "nonce": "0x..."
      }
    }
  },
  "paymentRequirements": {
    "scheme": "exact",
    "network": "eip155:84532",
    "amount": "10000",
    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "payTo": "0x209693Bc6afc0C5328bA36FaF03C514EF312287C",
    "maxTimeoutSeconds": 60,
    "extra": {"name": "USDC", "version": "2"}
  }
}
```

`VerifyResponse` 原例；没有 transaction，因为 verify 不提交。[X2 §7.1。]

```json
{
  "isValid": true,
  "payer": "0x857b06519E91e3A54538791bDbb0E22373e36b66"
}
```

`SettleResponse` 原例，亦为 v2 `PAYMENT-RESPONSE` 编码前的对象。[X2 §7.2；XH2。]

```json
{
  "success": true,
  "payer": "0x857b06519E91e3A54538791bDbb0E22373e36b66",
  "transaction": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
  "network": "eip155:84532"
}
```

`SupportedResponse` 原例，裁剪为一个 kind 和一个 signer family。[X2 §7.3。]

```json
{
  "kinds": [
    {"x402Version": 2, "scheme": "exact", "network": "eip155:84532"}
  ],
  "extensions": [],
  "signers": {
    "eip155:*": ["0x1234567890abcdef1234567890abcdef12345678"]
  }
}
```

SVM payload 核心原例，与 EVM authorization 结构不同。[XSVM PaymentPayload。]

```json
{
  "transaction": "AAAAAAAAAAAAA...AAAAAAAAAAAAA="
}
```

`http-message-signatures` 扩展中 info 的原例；这仅是 x402 自身文件里的原例，不是 IETF 或 TAP 的克隆证据。[XHTTPSig Extension Definition。]

```json
{
  "registrationUrl": "https://network.example.com/signature-agents",
  "signatureSchemes": ["ed25519", "ecdsa-p256-sha256", "rsa-pss-sha512"],
  "tags": ["web-bot-auth", "agent-browser-auth"]
}
```

#### 4.3.8 SDK、参考实现、许可与治理

计数按语言 SDK 家族而非 npm 模块或示例场景：快照有 **4 个家族：TypeScript、Python、Go、Java；前三者有 v2，Java 是 v1 实现**。这是仓库内计数，不是生态总 SDK 数；没有把 examples 的每种框架包装重复计为独立协议实现。[`CONTRIBUTING.md@X`；`docs/sdk-features.md@X`；`java/src/main/java/org/x402/client/X402HttpClient.java@X`。]

| 家族 | 快照中的版本和组织方式 | 来源 |
| --- | --- | --- |
| TypeScript | 23 个 `@x402/*` 模块，均为 `2.28.0`；另有 6 个 legacy 模块 | `typescript/.changeset/config.json@X`；`typescript/packages/core/package.json@X`；`typescript/packages/` 下各 manifest |
| Python | v2 发行项目 `x402==2.25.0`，功能按 extras / 子模块组织；legacy 为 `1.0.0` | `python/x402/pyproject.toml@X`；`python/legacy/pyproject.toml@X` |
| Go | v2 module `github.com/x402-foundation/x402/go/v2`；日志顶部 `v2.28.0`；另有 legacy module | `go/go.mod@X`；`go/CHANGELOG.md@X`；`go/legacy/go.mod@X` |
| Java | `org.x402:x402:1.0.0-SNAPSHOT`，Java 17；client 固定 v1 / exact / base-sepolia | `java/pom.xml@X`；`java/src/main/java/org/x402/client/X402HttpClient.java@X` |

23 个 TS scoped 模块包括 13 个机制、7 个 HTTP / paywall 包，以及 core、extensions、mcp；6 个 legacy 包分别为 `x402`、`x402-axios`（1.2.1），`x402-express`、`x402-fetch`、`x402-hono`、`x402-next`（1.2.0）。TS / Python / Go legacy README 标记 deprecated 且只接安全修复；不能据 Java 同样是 v1 自动推断相同承诺。[`typescript/.changeset/config.json@X`；`typescript/packages/legacy/x402/package.json@X`、`typescript/packages/legacy/x402-axios/package.json@X`、`typescript/packages/legacy/x402-express/package.json@X`、`typescript/packages/legacy/x402-fetch/package.json@X`、`typescript/packages/legacy/x402-hono/package.json@X`、`typescript/packages/legacy/x402-next/package.json@X`；`typescript/packages/legacy/x402/README.md@X`；`python/legacy/README.md@X`；`go/legacy/README.md@X`。]

许可声明存在差异，应原样记录而不作法律效力判断：

| 位置 | 观察到的声明 | 来源 |
| --- | --- | --- |
| 根目录 | Apache-2.0，NOTICE 标注 Copyright 2026 x402 Foundation | `LICENSE@X`；`NOTICE@X` |
| TS core / MCP 等发行模块 | Apache-2.0 | `typescript/packages/core/package.json@X`；`typescript/packages/mcp/package.json@X` |
| Python v2 manifest | MIT | `python/x402/pyproject.toml@X` |
| Python legacy manifest | Apache-2.0 | `python/legacy/pyproject.toml@X` |
| TS 私有聚合项目 | ISC | `typescript/package.json@X` |
| EVM 模块 | Apache-2.0；NOTICE 另记录部分代码的 MIT 来源 | `typescript/packages/mechanisms/evm/package.json@X`；`typescript/packages/mechanisms/evm/NOTICE@X` |

TSC 文件列 Coinbase / Erik Reppel、Cloudflare / Rohin Lohe、Stripe / Steve Kaliski 三组参与组织和代表。CONTRIBUTING 将合并裁量交给 x402 Foundation 团队，新链流程采用“规范 → 单 SDK 参考实现 → 其他 SDK”；因此 spec-only 网络并不矛盾。[`TSC.md@X`；`CONTRIBUTING.md@X` New Schemes、New Chains。]

#### 4.3.9 SDK 日志历史与日期

此节仅提取本地日志：`XGoLog = go/CHANGELOG.md@X`，`XPyLog = python/x402/CHANGELOG.md@X`，`XTSLog = typescript/packages/core/CHANGELOG.md@X`。**日期不是独立核验的发布平台记录。**

Go / Python 有日期的全部版本标题按时间合并如下；同日期两列不代表相同 SemVer 或能力。[XGoLog、XPyLog 各版本标题。]

| 日期 | Go 日志版本 | Python 日志版本 | 来源 |
| --- | --- | --- | --- |
| 2025-09-12 | 1.0.0 | — | XGoLog |
| 2025-10-12 | 2.0.0 | — | XGoLog |
| 2026-01-09 | 2.1.0 | — | XGoLog |
| 2026-02-11 | 2.2.0 | 2.1.0 | XGoLog；XPyLog |
| 2026-02-20 | 2.3.0 | 2.2.0 | 同上 |
| 2026-02-25 | 2.4.0、2.4.1 | — | XGoLog |
| 2026-03-06 | 2.5.0 | 2.3.0 | XGoLog；XPyLog |
| 2026-03-16 | — | 2.4.0 | XPyLog |
| 2026-03-17 | 2.6.0 | — | XGoLog |
| 2026-03-19 | — | 2.5.0 | XPyLog |
| 2026-03-23 | 2.7.0 | — | XGoLog |
| 2026-04-02 | 2.8.0 | 2.6.0 | XGoLog；XPyLog |
| 2026-04-13 | 2.9.0 | 2.7.0 | 同上 |
| 2026-04-17 | — | 2.8.0 | XPyLog |
| 2026-04-27 | 2.10.0 | 2.9.0 | XGoLog；XPyLog |
| 2026-05-11 | 2.11.0 | — | XGoLog |
| 2026-05-13 | — | 2.10.0 | XPyLog |
| 2026-05-22 | 2.12.0 | 2.11.0 | XGoLog；XPyLog |
| 2026-05-29 | 2.13.0、2.14.0 | 2.12.0 | 同上 |
| 2026-06-12 | 2.15.0 | 2.13.0 | 同上 |
| 2026-06-19 | 2.16.0 | 2.13.1 | 同上 |
| 2026-06-26 | 2.17.0 | 2.14.0 | 同上 |
| 2026-07-10 | 2.18.0 | 2.15.0 | 同上 |
| 2026-07-17 | 2.19.0 | 2.16.0 | 同上 |
| 2026-07-27 | 2.20.0 | 2.17.0 | 同上 |
| 2026-08-04 | 2.21.0 | 2.18.0 | 同上 |
| 2026-08-11 | 2.22.0 | 2.19.0 | 同上 |
| 2026-08-18 | 2.23.0 | 2.20.0 | 同上 |
| 2026-08-27 | 2.24.0 | 2.21.0 | 同上 |
| 2026-09-04 | 2.25.0 | 2.22.0 | 同上 |
| 2026-09-15 | 2.26.0 | 2.23.0 | 同上 |
| 2026-09-22 | 2.27.0 | 2.24.0 | 同上 |
| 2026-09-29 | 2.28.0 | 2.25.0 | 同上 |

Python `1.0.0`、`2.0.0` 标题没有日期。TS core 日志保留 1.0.0、2.0.0、2.3.0、2.3.1、2.4.0–2.28.0，但标题没有发布日期；不能给同号 TS 版本套 Go 日期。[XPyLog 尾部；XTSLog。]

能力里程碑：Go MCP 集成记在 2.2.0 / 2026-02-11、EVM upto 在 2.8.0 / 2026-04-02、EVM batch-settlement 在 2.11.0 / 2026-05-11、SVM upto 在 2.23.0 / 2026-08-18、auth-capture client 在 2.24.0 / 2026-08-27、SVM batch-settlement 在 2.28.0 / 2026-09-29。Python MCP 集成记在 2.1.0 / 2026-02-11、EVM upto 在 2.8.0 / 2026-04-17、TVM exact 在 2.10.0 / 2026-05-13、EVM batch-settlement 在 2.11.0 / 2026-05-22。[XGoLog、XPyLog 对应 Added 条目。]

版本陷阱与残留记录：

- Go auth-capture server / facilitator 源码已经存在，但其变更片段仍在 `unreleased`，时间为 2026-09-29；delegated-auth 片段为 2026-10-05。不能宣称完整能力随 2.28.0 已发布。[`go/.changes/unreleased/added-20260929-auth-capture-server-facilitator.yaml@X`；`go/.changes/unreleased/added-20261005-auth-capture-delegated-auth.yaml@X`。]
- Go / Python 的 MCP 子目录日志均保留 `0.1.0—2025-02-05` alpha 和 Unreleased，但 SDK 顶层日志的 MCP 集成日期是 2026-02-11。应保留这种记录差异，不从子目录日期推出确定的首次对外发行。[`go/mcp/CHANGELOG.md@X`；`python/x402/mcp/CHANGELOG.md@X`；XGoLog；XPyLog。]
- 协议 v2、SDK 2.28.0 / 2.25.0 与 scheme v1.1 是不同层级；auth-capture 概述 v1.1 日期 2026-08-18，EVM binding v1.1 日期 2026-08-26。[X2；`specs/schemes/auth-capture/scheme_auth_capture.md@X`；`specs/schemes/auth-capture/scheme_auth_capture_evm.md@X`。]

#### 4.3.10 Schemes、网络与语言覆盖

快照实际 scheme 目录共有四类：exact、upto、batch-settlement、auth-capture。`GET /supported` 是某个 facilitator 的能力广告；规范或源码存在不等于该服务部署了全部能力。[`specs/schemes/` 在 X 提交的树；X2 §6、§7.3。]

| Scheme | 语义 | 规范来源 |
| --- | --- | --- |
| exact | 单次付款，具体金额约束由绑定规定 | `specs/schemes/exact/scheme_exact.md@X`；XSVM |
| upto | 单次授权上限；实际费用不超过上限；不支持同授权多次 settle | `specs/schemes/upto/scheme_upto.md@X` |
| batch-settlement | 请求时承诺 / 存储，价值转移在后续 redemption；支持 capital-backed / credit-backed | `specs/schemes/batch-settlement/scheme_batch_settlement.md@X` |
| auth-capture | authorize / charge / capture / void / refund / reclaim 生命周期；默认 escrow | `specs/schemes/auth-capture/scheme_auth_capture.md@X` |

exact 的网络家族规范共 17 个，其中 13 个有 TS mechanism manifest。下列是**家族 / 文件覆盖**，不是具体主网、测试网或线上 facilitator 数量；“仅规范”限定为本快照的三类 v2 SDK 树。[`specs/schemes/exact/`、`typescript/packages/mechanisms/`、`python/x402/mechanisms/`、`go/mechanisms/` 在 X 提交的树；`typescript/.changeset/config.json@X`。]

| 家族 | exact 规范路径（均 @X） | v2 SDK 覆盖及证据（均 @X） |
| --- | --- | --- |
| EVM | `specs/schemes/exact/scheme_exact_evm.md` | TS / Go / Python：`typescript/packages/mechanisms/evm/package.json`；`go/mechanisms/evm/exact/README.md`；`python/x402/mechanisms/evm/exact/__init__.py` |
| SVM / Solana | `specs/schemes/exact/scheme_exact_svm.md` | TS / Go / Python：`typescript/packages/mechanisms/svm/package.json`；`go/mechanisms/svm/exact/facilitator/scheme.go`；`python/x402/mechanisms/svm/exact/__init__.py` |
| TON / TVM | `specs/schemes/exact/scheme_exact_ton.md` | TS / Python：`typescript/packages/mechanisms/tvm/package.json`；`python/x402/mechanisms/tvm/exact/__init__.py` |
| Algorand / AVM | `specs/schemes/exact/scheme_exact_algo.md` | TS：`typescript/packages/mechanisms/avm/package.json` |
| Aptos | `specs/schemes/exact/scheme_exact_aptos.md` | TS：`typescript/packages/mechanisms/aptos/package.json` |
| Cardano | `specs/schemes/exact/scheme_exact_cardano.md` | TS：`typescript/packages/mechanisms/cardano/package.json` |
| Casper | `specs/schemes/exact/scheme_exact_casper.md` | TS：`typescript/packages/mechanisms/casper/package.json` |
| Concordium | `specs/schemes/exact/scheme_exact_concordium.md` | TS：`typescript/packages/mechanisms/concordium/package.json` |
| Hedera | `specs/schemes/exact/scheme_exact_hedera.md` | TS：`typescript/packages/mechanisms/hedera/package.json` |
| Keeta | `specs/schemes/exact/scheme_exact_keeta.md` | TS：`typescript/packages/mechanisms/keeta/package.json` |
| NEAR | `specs/schemes/exact/scheme_exact_near.md` | TS：`typescript/packages/mechanisms/near/package.json` |
| Stellar | `specs/schemes/exact/scheme_exact_stellar.md` | TS：`typescript/packages/mechanisms/stellar/package.json` |
| XRPL | `specs/schemes/exact/scheme_exact_xrpl.md` | TS：`typescript/packages/mechanisms/xrpl/package.json` |
| Canton | `specs/schemes/exact/scheme_exact_canton.md` | 仅规范；上述 mechanism 树未见对应实现 |
| Bitcoin Lightning | `specs/schemes/exact/scheme_exact_lnbtc.md` | 同上 |
| Starknet | `specs/schemes/exact/scheme_exact_starknet.md` | 同上 |
| Sui | `specs/schemes/exact/scheme_exact_sui.md` | 同上 |

非 exact 覆盖：EVM upto、EVM batch-settlement 有 TS / Go / Python；SVM upto、SVM batch-settlement 有 TS / Go；EVM auth-capture 有 TS / Go client/server/facilitator 源码，Go 部分能力须保留 unreleased 限定；Cloudflare batch-settlement 是规范覆盖，不在这三类 mechanism 树中找到独立参考模块。[`specs/schemes/upto/scheme_upto_evm.md@X`、`specs/schemes/upto/scheme_upto_svm.md@X`；`go/mechanisms/evm/upto/README.md@X`；`go/mechanisms/svm/upto/README.md@X`；`python/x402/mechanisms/evm/upto/__init__.py@X`；`typescript/packages/mechanisms/evm/src/batch-settlement/README.md@X`、`typescript/packages/mechanisms/svm/src/batch-settlement/README.md@X`；`go/mechanisms/evm/batch-settlement/README.md@X`、`go/mechanisms/svm/batch-settlement/README.md@X`；`python/x402/mechanisms/evm/batch_settlement/__init__.py@X`；`go/mechanisms/evm/auth-capture/README.md@X`；`specs/schemes/batch-settlement/scheme_batch_settlement_cloudflare.md@X`。]

核心规范实际列举的网络标识含 Base Sepolia `eip155:84532`、Base `eip155:8453`、Avalanche Fuji `eip155:43113`、Avalanche `eip155:43114`，Solana mainnet `solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp`、devnet `solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1`；它们是原文示例，不是完整网络列表。信用结算绑定另用 `cloudflare:402`。[X2 §11.1；`specs/schemes/batch-settlement/scheme_batch_settlement_cloudflare.md@X`。]

#### 4.3.11 非 exact schemes：签名与重放不能套用同一规则

本节补充六种绑定；其他 exact 家族本次没有进行与 EVM / SVM 同等深度的密码学源码审计，不能据机制包存在声称全部签名实现已核查。

| 绑定 | 实际签名 / 承诺对象 | 状态、期限与重放控制 | 来源 |
| --- | --- | --- | --- |
| upto / EVM | EIP-712 PermitWitnessTransferFrom；domain 为 name=Permit2、chainId、verifyingContract=Permit2；签 permitted.token/amount、proxy spender、nonce、deadline、witness.to/facilitator/validAfter | Random uint256 nonce，deadline / validAfter，绑定特定facilitator；单次结算actual≤signed max，settle验签仍使用max | `typescript/packages/mechanisms/evm/src/upto/client/permit2.ts@X`；`specs/schemes/upto/scheme_upto_evm.md@X` |
| upto / SVM | Client签open交易，不在初始payload签voucher；meter后receiverAuthorizer用Ed25519签voucher bytes | 一请求一channel；PDA绑定payer/feePayer/token/authorizer/nonce/openSlot，deposit=ceiling，deadline约束voucher；最终settle_and_seal消费channel状态 | `specs/schemes/upto/scheme_upto_svm.md@X` §4.2、§5、§8；`typescript/packages/mechanisms/svm/src/payment-channels/voucher.ts@X` |
| batch / EVM | EIP-712 ChannelConfig、Voucher、Refund、ClaimBatch；domain x402 Batch Settlement / version1，绑定chain和部署地址；payerAuthorizer非零走ECDSA，零走payer的SignatureChecker | Cumulative watermark不回退；退款另有refundNonce，先增nonce再转账；不同操作typehash分离；voucher无expiry，押金授权期限不是voucher期限 | `specs/schemes/batch-settlement/scheme_batch_settlement_evm.md@X`；`typescript/packages/mechanisms/evm/src/batch-settlement/constants.ts@X` |
| batch / SVM | Client/server模式均使用Ed25519 voucher；server模式另有payer签的单请求bearer proof；close签独立domain的SHA-256 digest | Voucher watermark单调，expiresAt固定0；server-mode按(channelId,requestId)单次操作记录拒绝重复；client-mode按锁/charged watermark防重复；closeAuthorization有validBefore | `specs/schemes/batch-settlement/scheme_batch_settlement_svm.md@X` §4.2、§5.5 |
| batch / cloudflare:402 | RFC9421 HTTP Message Signatures；规范的扩展示例广告ed25519 / web-bot-auth；支付payload只有amount/asset，签名在HTTP headers | 绑定authority、signature-agent、payment-signature；时间freshness通常30秒；binding称stateless，没有定义独立一次性nonce消费合同 | `specs/schemes/batch-settlement/scheme_batch_settlement_cloudflare.md@X` Verification、Security；XHTTPSig |
| auth-capture / EVM | Client签ERC-3009/Permit2 collector授权；receiverAuthorizer另签Charge/Capture/Void/Refund EIP-712（ECDSA或ERC1271）；authorize不用server签名 | Client token nonce绑定PaymentInfo/chain/escrow；capture/refund签expected balances作状态重放key；delegated类型只在facilitator查余额，文档明确best-effort，非链上原子保证 | `specs/schemes/auth-capture/scheme_auth_capture_evm.md@X`；`typescript/packages/mechanisms/evm/src/auth-capture/authorizerSigner.ts@X` |

关键字节级细节：

SVM upto / batch voucher 的签名消息精确为下面50字节；upto 的 cumulativeAmount 是实际本次计费，batch 是累积授权 ceiling，batch 的 expiresAt 必须0。[`specs/schemes/upto/scheme_upto_svm.md@X` §4.2；`specs/schemes/batch-settlement/scheme_batch_settlement_svm.md@X` §4.2；`typescript/packages/mechanisms/svm/src/payment-channels/voucher.ts@X`。]

```text
0x56 0x01 || channelId[32] || u64(cumulativeAmount).le || i64(expiresAt).le
```

SVM batch server-mode 的单请求 payer proof 签以下原始拼接字节；requestId 是1–256 UTF-8字节，地址解码必须各32字节；这不同于累积voucher。[`specs/schemes/batch-settlement/scheme_batch_settlement_svm.md@X` §4.2；`typescript/packages/mechanisms/svm/src/batch-settlement/authorization.ts@X`。]

```text
UTF8("x402-batch-authorization-v2") ||
channelId[32] || payer[32] || operator[32] ||
u16(len(UTF8(requestId))).le || UTF8(requestId) ||
u64(authorizedAmount).le || i64(expiresAt).le
```

SVM close 用Ed25519签以下串的SHA-256 digest，绑定network/program/feePayer/channel/watermark/expiry；不是把JSON key排序后签名。[`specs/schemes/batch-settlement/scheme_batch_settlement_svm.md@X` CloseAuthorization；`typescript/packages/mechanisms/svm/src/batch-settlement/closeAuthorization.ts@X`。]

```text
UTF8("x402:batch-settlement:svm:close:v1") || 0x00 ||
u16(len(UTF8(network))).le || UTF8(network) ||
programId[32] || feePayer[32] || channelId[32] ||
u64(maxClaimableAmount).le || i64(voucherExpiresAt).le || i64(validBefore).le
```

EVM batch 的签名 types：

| Type | 所签字段 | 来源 |
| --- | --- | --- |
| ChannelConfig | payer、payerAuthorizer、receiver、receiverAuthorizer、token、withdrawDelay:uint40、salt:bytes32 | `typescript/packages/mechanisms/evm/src/batch-settlement/constants.ts@X` |
| Voucher | channelId:bytes32、maxClaimableAmount:uint128 | 同上 |
| Refund | channelId:bytes32、nonce:uint256、amount:uint128 | 同上 |
| ClaimBatch | claims:ClaimEntry[]；每项channelId、maxClaimableAmount、totalClaimed | 同上 |
| Deposit | ReceiveWithAuthorization或带DepositWitness.channelId的Permit2；ERC3009 nonce=keccak256(abi.encode(channelId,salt)) | 同上；`typescript/packages/mechanisms/evm/src/batch-settlement/encoding.ts@X` |

Auth-capture receiver签名domain：

```text
{name:"x402 Auth Capture Operator",version:"1",
 chainId,verifyingContract:extra.captureAuthorizer}
```

v1.1 所签字段如下；v1.0 的 charge / capture 使用 feeBps:uint16，而不是 feeAmount:uint256，不能换算后冒充原digest。[`specs/schemes/auth-capture/scheme_auth_capture_evm.md@X` Authorizer signatures；`typescript/packages/mechanisms/evm/src/auth-capture/authorizerSigner.ts@X`。]

| 操作 | 字段 | 来源 |
| --- | --- | --- |
| Charge | paymentInfoHash、amount、tokenCollector、collectorDataHash、feeAmount、feeReceiver | 同上 |
| Void | paymentInfoHash | 同上 |
| Capture | paymentInfoHash、amount、feeAmount、feeReceiver、expectedCapturableAmount、expectedRefundableAmount | 同上 |
| Refund | paymentInfoHash、amount、tokenCollector、expectedCapturableAmount、expectedRefundableAmount | 同上 |

PaymentInfo 包含 operator、payer、receiver、token、maxAmount、preApprovalExpiry、authorizationExpiry、refundExpiry、minFeeBps、maxFeeBps、feeReceiver、salt。Client nonce 是对payer清零后的PaymentInfo hash，再与chainId / escrow地址做ABI编码哈希；有authorizer/policy绑定时salt另承诺domain-tag、两个地址、saltNonce。三个期限按preApproval≤authorization≤refund排序分别约束collect、capture/reclaim、refund。[`specs/schemes/auth-capture/scheme_auth_capture_evm.md@X` Payment identity、PaymentInfo、Expiry ordering；`typescript/packages/mechanisms/evm/src/auth-capture/nonce.ts@X`。]

Cloudflare payment payload 的原例，与链上授权结构明显不同。[`specs/schemes/batch-settlement/scheme_batch_settlement_cloudflare.md@X` Decoded PAYMENT-SIGNATURE。]

```json
{
  "x402Version": 2,
  "payload": {"amount": "5", "asset": "USD"},
  "accepted": {
    "scheme": "batch-settlement",
    "network": "cloudflare:402",
    "amount": "5",
    "asset": "USD",
    "payTo": "merchant",
    "maxTimeoutSeconds": 30
  }
}
```

该binding的三个必需签名component不包含HTTP method/path；freshness要求也不能单独证明窗口内同承诺严格单次消费。这是对绑定范围的分析，不是经过运行验证的漏洞；尤其不能将这份x402绑定文档当成独立IETF/TAP规范的全部要求。[`specs/schemes/batch-settlement/scheme_batch_settlement_cloudflare.md@X` Verification、Security Considerations。]

### 4.4 Visa Trusted Agent Protocol

获取证据：E5 失败；网页定位：W5。

| 请求项 | 本次结果 |
| --- | --- |
| 当前规范版本与带日期的发布历史 | 无法从克隆确认 |
| Agent、merchant、registry / directory 等角色及消息顺序 | 无法从克隆确认 |
| `Signature`、`Signature-Input` 的实际组件与参数 | 无法从克隆确认 |
| 浏览、支付等 tag 的精确字符串与验证逻辑 | 无法从克隆确认 |
| Key directory 的路径、对象、字段与密钥生命周期 | 无法从克隆确认 |
| 签名与目录消息的裁剪原文，包括是否实际存在 JSON 示例 | 未取得；不提供自造示例 |
| 允许算法、实际实现算法及规范化签名 base | 无法从克隆确认 |
| 时间窗、nonce、重放缓存、域名 / 方法 / 路径 / body 绑定 | 无法从克隆确认；README 安全声明不能代替代码检查 |
| 许可证、治理、参考实现数量及语言 | 无法从克隆确认 |
| TODO、开放问题及样例与安全声明的不一致 | 未做源码核查 |

### 4.5 IETF Web Bot Auth / directory 与 MCP 支付提案

#### 4.5.1 独立 IETF / MCP 主仓库的证据缺口

E6、E7、E8 失败，网页定位为 W6–W12。下面的“无法确认”只针对独立主仓库；后续小节另提供 **x402 仓库自己的 MCP / A2A transport 证据**，两者不能替代。

| 请求项 | 本次结果 |
| --- | --- |
| IETF 草案的新旧名称、最新正式编号、日期及历史关系 | 仅观察到 W6 的网页重定向；源码和正式版本历史均未核查 |
| HTTP Message Signatures profile 的字段、tag 与签名组件 | 无法从克隆确认 |
| Directory / JWKS 的字段、媒体类型、well-known 路径 | 无法从克隆确认 |
| 算法、目录响应签名、key discovery、轮换及防重放 | 无法从克隆确认 |
| IETF 流程、贡献和许可规定 | 无法从克隆确认 |
| MCP 支付 SEP 的规范文本、当前状态与是否合并 | 仅定位候选；不能确认已成为规范 |
| MCP 支付 proposal 的 discovery、挑战、响应、验证等具体方法与字段 | 无法从 MCP 主仓库克隆确认 |
| URL mode elicitation 与支付支持 proposal 的关系 | 无法从克隆确认；不将两者视为同一种支付协议 |
| IETF / 独立 MCP proposal 的核心消息 JSON 原文 | 未取得；不提供自造示例 |
| IETF / MCP 主仓库的参考实现数量、语言、TODO 和开放问题 | 未做源码核查 |

#### 4.5.2 x402 仓库内的 MCP transport：可确认部分

引用 `XMCP = specs/transports-v2/mcp.md@X`。规范的付费工具流程如下；它是 x402 定义的 MCP 适配，不是对 MCP 支付 SEP 正式采纳状态的证明。[XMCP Payment Flow Overview。]

1. Client 无支付调用工具。[XMCP。]
2. Server 返回 `isError: true` 的工具结果，`structuredContent` 直接为 PaymentRequired，`content[0].text` 是同一对象的 JSON 字符串；规范要求两种表示同时存在。[XMCP Payment Required Signaling。]
3. Client 优先读取 structuredContent，必要时解析 text，形成 PaymentPayload。[XMCP Client Requirements；`typescript/packages/mcp/src/client/x402MCPClient.ts@X`。]
4. Client 重试 `tools/call`，支付载荷位于 `params._meta["x402/payment"]`，不是 HTTP `PAYMENT-SIGNATURE` header。[XMCP Payment Payload Transmission。]
5. Server 按配置支付 flow 验证 / 执行 / 结算。[XMCP；`typescript/packages/mcp/src/server/paymentWrapper.ts@X`。]
6. 成功结果在 `result._meta["x402/payment-response"]` 提供 SettlementResponse。[XMCP Settlement Response Delivery。]

| 字段路径 | 类型 / 意义 | 来源 |
| --- | --- | --- |
| `result.isError` | boolean，挑战时 true | XMCP |
| `result.structuredContent` | PaymentRequired，挑战中必需 | XMCP |
| `result.content[0].text` | string，重复编码同一挑战对象 | XMCP |
| `params._meta["x402/payment"]` | PaymentPayload，付费重试 | XMCP |
| `result._meta["x402/payment-response"]` | SettlementResponse，结算信息 | XMCP |

三语言有实际 MCP client / server 包装：TS `@x402/mcp`、Go module 内 `mcp` 子包、Python `x402.mcp`。TS wrapper 明确增强 `callTool()`，其他 MCP 方法透传；不能泛化为全部 resource / prompt 方法自动付费。[`typescript/packages/mcp/package.json@X`；`typescript/packages/mcp/src/client/x402MCPClient.ts@X`；`typescript/packages/mcp/src/server/paymentWrapper.ts@X`；`go/mcp/client.go@X`；`go/mcp/server.go@X`；`python/x402/mcp/__init__.py@X`。]

TS 类型和客户端还识别 JSON-RPC `-32042`，接受 `data` 或 `data.x402`，并兼容 402；注释引用 SEP-1036。这只证明适配器的错误格式兼容，不证明 x402 支付已成为独立 MCP SEP 的最终规范。[`typescript/packages/mcp/src/types/mcp.ts@X`；`typescript/packages/mcp/src/client/x402MCPClient.ts@X` `extractPaymentRequiredFromError()`。]

实现差异：TS / Go 结算失败返回 PaymentRequired 风格工具错误；Python 某些包装路径将 `x402/payment-response` 嵌入错误数据内部，位置不同于成功结果的顶层 `_meta`。集成时应按失败类型解析，不能只看成功路径。[`typescript/packages/mcp/src/server/paymentWrapper.ts@X`；`go/mcp/server.go@X`；`python/x402/mcp/server.py@X`；`python/x402/mcp/server_async.py@X`。]

MCP 三条原文裁剪如下，**都是结构片段，不是完整可提交消息**：第一条省略规范要求的重复 content 文本，requirements 的部分必需字段也被删除；第二条省略 signature / payload；第三条省略完整 receipt。[XMCP 三个相应 Example。]

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "isError": true,
    "structuredContent": {
      "x402Version": 2,
      "resource": {"url": "mcp://tool/financial_analysis"},
      "accepts": [
        {"scheme": "exact", "network": "eip155:84532", "amount": "10000"}
      ]
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "financial_analysis",
    "_meta": {
      "x402/payment": {
        "x402Version": 2,
        "accepted": {
          "scheme": "exact",
          "network": "eip155:84532",
          "amount": "10000"
        }
      }
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "_meta": {
      "x402/payment-response": {"success": true, "network": "eip155:84532"}
    }
  }
}
```

#### 4.5.3 x402 仓库内的 A2A transport；独立扩展未克隆

引用 `XA2A = specs/transports-v2/a2a.md@X`。独立 `google-agentic-commerce/a2a-x402` 仓库的 E2 克隆仍失败；以下只能确认 x402 内部 transport 文档，不能证明独立扩展最新版本与其一致。

| 阶段 | A2A 位置 / 字段 | 来源 |
| --- | --- | --- |
| 要求支付 | task state 为 `input-required`；`status.message.metadata["x402.payment.status"]="payment-required"`；要求在 `x402.payment.required` | XA2A Payment Required Signaling |
| 提交授权 | `message/send`；`params.message.taskId` 关联原 task；metadata 内 `x402.payment.status="payment-submitted"`、`x402.payment.payload=PaymentPayload` | XA2A Payment Payload Transmission |
| 完成 | task state `completed`；`payment-completed`；`x402.payment.receipts` 是 SettlementResponse 数组 | XA2A Settlement Response Delivery |
| 失败 | task state `failed`；`payment-failed`；错误在 `x402.payment.error` | XA2A Error Handling |
| 其他支付状态 | `payment-rejected`、`payment-verified` | XA2A Payment Status Lifecycle |
| 声明与激活 | AgentCard.`capabilities.extensions[]`；client `X-A2A-Extensions` | XA2A Extension Declaration and Activation |

文档中的 URI 仍是 `https://github.com/google-a2a/a2a-x402/v0.1`；该字符串是协议扩展标识，不应根据新的仓库名称自行改写，也不能从 URI 的 `v0.1` 推断 x402 core 是 v1。[XA2A Extension Declaration and Activation；X2 文首。]

本快照有 A2A 规范；已检索的三类 v2 SDK、Java、examples、e2e 树中没有找到专用 A2A payment adapter / package。这个负面结果仅限该树，不否定仓库外的独立实现。[XA2A；X 提交中的 `typescript/packages/`、`python/x402/`、`go/`、`java/src/`、`examples/`、`e2e/` 文件树。]

三条原文裁剪，只展示 task / metadata 路径；省略签名及部分必需字段，**不是可提交的完整 A2A 消息**。[XA2A 相应三个 Example。]

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "kind": "task",
    "id": "task-123",
    "status": {
      "state": "input-required",
      "message": {
        "metadata": {
          "x402.payment.status": "payment-required",
          "x402.payment.required": {
            "x402Version": 2,
            "accepts": [
              {"scheme": "exact", "network": "eip155:8453", "amount": "48240000"}
            ]
          }
        }
      }
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "method": "message/send",
  "id": "req-003",
  "params": {
    "message": {
      "taskId": "task-123",
      "role": "user",
      "metadata": {
        "x402.payment.status": "payment-submitted",
        "x402.payment.payload": {"x402Version": 2}
      }
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": "req-003",
  "result": {
    "kind": "task",
    "id": "task-123",
    "status": {
      "state": "completed",
      "message": {
        "metadata": {
          "x402.payment.status": "payment-completed",
          "x402.payment.receipts": [
            {"success": true, "network": "eip155:8453"}
          ]
        }
      }
    }
  }
}
```

## 5. 继续研究的必要条件与剩余工作

完成原任务需要让当前终端能够在现有授权范围内访问 GitHub 的 Git HTTPS 服务，从而成功执行所要求的 `git clone --depth 1`。网页检索通道恢复或单独提供摘要不足以满足这个条件。

离线快照的源码分析与原文裁剪已经在第 4 节提供。以下要求仍未满足：

1. 直接确认 AP2、ACP、x402 的最新远端默认分支、当前 SHA、重命名 / 归档元数据，并将离线结果与新快照复核。
2. 对独立 A2A x402、Visa TAP、IETF / directory、MCP 主仓库执行成功的浅克隆，补齐各自字段、流程、JSON、密码学、治理与实现清单。
3. 从第一方发布和 issue 元数据确认 release 日期、开放问题状态；不能把旧 changelog 的修复条目视作仍开放问题。
4. 若当前 SDK 依赖实现不在主仓库中，需要另行取得其源码后才能确认默认签名、salt 和规范化细节；本次没有用猜测填补。

本报告只确认所列固定 SHA 内的源码、规范和示例，不确认代码运行结果、真实资金交易、互操作或上游修复状态。原任务的全部仓库和最新上游要求未完成，不能据报告长度视作已完成。

### 产物检查

报告分段写入；使用系统文本工具检查引用路径，并仅对**本报告**的35个 JSON 代码块做语法解析，全部通过。没有执行仓库 validator、测试、脚本或构建；JSON 语法通过不等于满足协议 schema，文中已明确标记裁剪片段和原例过时问题。最终 Git 状态检查确认三个克隆没有工作树修改，本次会话只写入本报告。检查时同目录另有 `2026-10-19-agentic-payment-protocols-survey.md`，没有读取或改动，也未纳入本报告证据。
