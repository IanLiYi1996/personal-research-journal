# PSP 沙箱调研报告

> 调研日期：2026-10-20  
> 目标：AWS Agentic Payment 演示 Demo（面向出海跨境支付机构）  
> 方法：仅采信官方一手文档；拿不准的标「未验证」

---

## 1. Airwallex

### 1.1 沙箱注册与 KYB

- **能否自助注册**：**可以**。访问 https://www.airwallex.com/docs（右上角"Get started in sandbox"）即可创建沙箱账号，**即时激活，无需等待**。
  - 来源：https://www.airwallex.com/docs/developer-tools/sandbox-environment/sandbox-environment-overview
- **SG / HK 主体**：支持任何国家。创建账号时填写商业邮箱、姓名、**Business Country**（SG / HK 均可）即可。
- **KYB**：沙箱**不需要**真实 KYB。创建额外账户时，Entity Name 必须使用 `"New Business Sandbox"` 或 `"Sandbox Business"`，系统会自动通过 onboarding 验证。
- **例外**：Connected Account APIs（嵌入式金融 embedded finance 场景）在沙箱中**不可用**，须联系 Airwallex Sales 单独开通。
  - 来源：同上

### 1.2 沙箱能力

#### 1.2.1 Payouts / Transfers（本地实时付款）

- **有沙箱支持**：可用 Transfers Simulation API 推进转账状态（即模拟清算网络推进）。
  - 来源：https://www.airwallex.com/docs/developer-tools/sandbox-environment/payouts
- **本地清算网络覆盖**（生产 API 列出的轨道，沙箱中可测试）：
  | 国家/地区 | 本地轨道 |
  |---|---|
  | 新加坡 (SGD) | FAST, GIRO, RTGS |
  | 香港 (HKD/CNY/USD) | FPS, ACH, RTGS |
  | 印度尼西亚 | BI-FAST, SKN, iACH |
  | 越南 | NAPAS |
  | 马来西亚 | DuitNow, GIRO, iACH |
  | 泰国 | PromptPay, Smart Credit, BAHTNET |
  | 菲律宾 | InstaPay, PesoNet (PHP); PDDTS, GSRT (USD) |
  | 欧元区 | SEPA Instant, SEPA |
  | 英国 | Faster Payments (UK FPS), CHAPS |
  | 美国 | ACH, Next-Day ACH, FedNow, Fedwire, Paper Check |
  | 印度 | NEFT, IMPS, RTGS, iACH |
  | 澳大利亚 | NPP, BPAY, Bank Transfer |
  | 巴西 | CIP-SILOC |
  | 墨西哥 | SPEI |
  | …（100+ 国家）| … |
  
  注：PayNow（SG）**未在 payouts 文档中单独列出**，SG 轨道显示为 FAST/GIRO/RTGS。
  - 来源：https://www.airwallex.com/docs/payouts/payout-network/bank-accounts

#### 1.2.2 FX 报价与锁汇（Transactional FX）

- **有沙箱支持**：沙箱 Web App 中可直接做 Wallet Conversions（Wallet → Conversions）。
- 相关 API：`POST /v1/conversions` (创建换汇), `POST /v1/fx/quotes` (获取报价), `GET /v1/fx/rates` (获取汇率)。
- API 版本 2024-06-30+ 后有全新 Conversion 状态定义。
  - 来源：https://www.airwallex.com/docs/transactional-fx/conversions

#### 1.2.3 Issuing（虚拟卡）

- **有沙箱支持**：可以创建虚拟卡，并模拟卡授权（一般授权、失败授权、增量授权、单次消息授权+扣款）。
- 沙箱 Simulation APIs：
  - `POST /v1/issuing/simulations/transactions` — 模拟一次卡授权，hold 余额
  - `POST /v1/issuing/simulations/transactions/{id}/capture` — 模拟 capture（全额或部分）
- 可测试 issuing webhook 事件。
  - 来源：https://www.airwallex.com/docs/developer-tools/sandbox-environment/issuing

#### 1.2.4 Webhooks

- **有沙箱支持**：在 Web App Account → Developer → Webhooks 配置，建议将沙箱 IP 加白名单。
- 可在 Web App 中查看和**重触发**历史 webhook 事件（"View and re-trigger webhook events"）。
  - 来源：https://www.airwallex.com/docs/developer-tools/sandbox-environment/sandbox-environment-overview

#### 1.2.5 Payment Acceptance（Payment Intents / 收单）

- **有沙箱支持**：可测试 Payment Intents、Payment Attempts、Refunds、Customers 等。
  - 来源：API Reference → /docs/api/payments/payment_intents

### 1.3 鉴权方式与速率限制

```
# 获取 access token（沙箱）
POST https://api.sandbox.airwallex.com/api/v1/authentication/login
Headers:
  x-api-key: {{YOUR_API_KEY}}
  x-client-id: {{YOUR_CLIENT_ID}}
  Content-Type: application/json

# 后续请求
GET https://api.sandbox.airwallex.com/api/v1/balances/current
Headers:
  Authorization: Bearer {{ACCESS_TOKEN}}
```

- Token 有效期：**30 分钟**（`expires_at` 字段为 ISO8601）；勿在每次请求前重新获取。
- 速率限制：返回 HTTP 429，具体阈值**未公开**。
- 来源：https://www.airwallex.com/docs/api/authentication/api_access_token

### 1.4 SDKs

| 语言 | 包名 | 状态 |
|---|---|---|
| Node.js | `@airwallex/node-sdk` (npm) | Beta，支持 API >= 2025-02-14 |
| Python | — | **无官方 Python SDK** |

- 来源：https://www.airwallex.com/docs/developer-tools/sdks/server-side-sdks-(beta)

### 1.5 MCP Server / Agent Toolkit

Airwallex 提供两个 MCP Server 和一个 CLI：

| 工具 | 环境 | 用途 | 鉴权 |
|---|---|---|---|
| **Developer MCP** | 沙箱 | 编码/集成测试；调用沙箱 API、模拟事件 | OAuth |
| **Docs MCP** | 只读文档 | 只需查文档的 Agent | 无需 |
| **AgentOS MCP** | 生产 | 生产账号操作（余额查询、创建受益人等） | OAuth |

```bash
# Claude Code 安装 Developer MCP（沙箱）
claude mcp add-json airwallex-dev '{ "type": "http", "url": "https://mcp.sandbox.airwallex.com/developer" }'
```

AgentOS 预置 Skills：
- `contract-to-billing` — 从合同/PO 提取账单信息
- `beneficiary-creation` — 从供应商文档提取银行信息并验证
- `card-provisioning` — 创建持卡人和虚拟/实体企业卡
- `manage-cashflow` — 多货币余额聚合、FX 敞口分析
- `awx-best-practices` — 通用守护 guardrail

- 来源：https://www.airwallex.com/docs/developer-tools/ai/developer-mcp  
  https://www.airwallex.com/docs/developer-tools/ai/agentos

### 1.6 Agentic Commerce Suite / Agentic Wallet

- **未找到独立 API 文档或 waitlist 页面**。官方 docs.airwallex.com 和 API Reference 中均未见名为 "Agentic Commerce Suite" 或 "Agentic Wallet" 的产品。  
- 官网（airwallex.com/en-us）使用 "Scale faster with agentic finance" 和 "One agentic business account" 等营销语言，但并无独立的 Agentic Commerce API。  
- 最接近的能力是 **AgentOS**（生产 MCP + Skills）和 **Payment Intents + Payouts + Issuing** 组合。  
- **结论**：截至 2026-10-20，不存在可以独立接入的 Agentic Commerce Suite 沙箱；如有私测 waitlist，需联系 Sales。

---

## 2. 连连国际（LianLian Global）

### 2.1 开发者平台与沙箱

- **官方 API 文档（Stoplight 托管）**：https://developer.lianlianglobal.com  
  （重 JavaScript 渲染；列出以下 API 产品模块：B2B-Cross-Border、Cards-Open API、Global Collection API、global-payout-fx、e-wallet openapi、Standard Remittance 等）
- **开发者注册入口**：https://global.lianlianpay.com/developer/api
- **能否自助注册沙箱**：**不能自助**。注册需先完成账号注册 → **企业认证（KYB）** → 获取 API KEY → 开发 → 切换生产环境。**没有公开可见的免 KYB 沙箱**。
- 来源：https://global.lianlianpay.com/developer/api

### 2.2 沙箱中的能力

| 能力 | 状态 |
|---|---|
| 跨境收款 | 有 API（Global Collection API），沙箱是否单独隔离**未验证** |
| 跨境付款 / 本地清算 | 有 API（global-payout、B2B-Cross-Border），支持 SWIFT，具体本地轨道**未验证** |
| 换汇（FX）| 有 API（global-payout-fx），50+ 货币对，支持锁汇和限价订单 |
| 虚拟卡（越达卡 VCC）| 有 API（Cards-Open API），**仅限机构级 Open API**，需要单独接入资质 |
| 状态模拟 / 沙箱推进 | **未验证**——文档使用 Stoplight 托管，具体沙箱能力页面未能加载 |

- 认证方式（从官方 Python 示例推断）：

```python
# https://global.lianlianpay.com/developer/api 示例
POST https://api.lianlianpay.com/v1/payments
Headers:
  Authorization: Bearer {API_KEY}
Body:
  { "country_code": "SG", "currency_list": ["SGD"], "payment_methods": ["SWIFT"] }
```

### 2.3 与银联国际的 AI Agent 跨境 B2B 支付合作

- **未验证**：在 developer.lianlianglobal.com、global.lianlianpay.com/news 等一手来源中未找到相关公开 API 文档或开发者公告。  
- 若该合作属实，API 可能通过连连机构级 Open API 接入，需单独洽谈。

### 2.4 申请评估

- 连连国际面向**机构/平台**客户，自助沙箱**不存在**；需联系连连商务 (service@lianlianpay.com / 400-091-0999)，预计等待 **1～4 周**（企业认证 + 商务评估）。

---

## 3. 备选对照：Nium、Rapyd、Wise Platform

### 3.1 Nium

- **开发者文档**：https://docs.nium.com
- **自助沙箱**：**可以**——访问 https://app.nium.com/sign-up 注册，邮箱验证 + MFA，即可在 Nium Portal → Configuration → API Keys 获取 `clientHashId` 和 `x-api-key`；**无需等待**。
- **沙箱=生产**：沙箱与生产环境行为一致，上线只需换密钥，无需改代码。
- **本地实时轨道**：声称覆盖"40+ local payment rails"，具体清单在 Nium Atlas（未在首页逐一列出）；美国 ACH routing code 在文档中有示例，IBAN 校验有支持，Confirmation of Payee 支持 50+ 国。
- **虚拟卡**：`cardType: "VIR"` → 状态 `"VIRTUAL_ACTIVE"`，沙箱可用。
- **FX**：FX quotes、rate locking、2000+ 货币对。
- **Simulation APIs**：
  - 模拟 onboarding 状态推进（under_review → clear）
  - Magic values（账号末尾 `987654321` = PAID；`897654321` = 银行拒绝）
  - 模拟入账、RFI、合规 review
- **SDKs**：Node, Python, Go, Java
- **MCP Server**：官方 Nium MCP Server，支持 Claude Code、Cursor、Claude Desktop、Codex、ChatGPT、VS Code。
  - 来源：https://docs.nium.com（Developers → Building with AI → Nium MCP Server）
- **适合场景**：开箱即用沙箱、虚拟卡、多轨道付款；对 APAC 本地轨道覆盖情况需在 Nium Atlas 单独确认。

### 3.2 Rapyd

- **文档**：https://docs.rapyd.net  
  （重 JavaScript 渲染，部分页面在本次研究中未能正常加载）
- **自助沙箱**：**可以**——在 dashboard.rapyd.net 注册后，在 API Control 页面获取 access key 和 secret key。
- **本地支付方式**：Rapyd Collect 支持 100+ 国家本地支付方式；Rapyd Disburse 支持 190+ 国家付款。具体轨道清单在 Docs → Countries & Payment Methods 页面。
- **虚拟卡**：有 Issuing 产品（Virtual Cards）；Virtual Accounts 支持 40+ 国、25+ 货币收款。
- **鉴权**：HMAC-SHA256 请求签名（access_key + secret_key + salt + timestamp + body）——**未验证**（文档页面未能加载，此为业界通识）。
- **SDK**：Python SDK（beta，需申请早期访问）；OpenAPI spec 在 GitHub 上公开。
- **Wallet**：Rapyd Wallet 支持 70+ 货币。
- **注意**：本次调研中 docs.rapyd.net 所有子页面均返回 "Rapyd Docs | Error"，建议在本地浏览器或允许 JS 的环境重新验证。

### 3.3 Wise Platform

- **文档**：https://docs.wise.com/api-docs
- **自助沙箱**：**不完全自助**——需要"Received Valid API credentials from Wise"（来自 tech support）。部分能力（webhook clientKey、bank details）须联系 Wise 支持开通；预计等待 **数天至 1 周**。
- **沙箱 URL**：`https://api.wise-sandbox.com/2026Q4`（生产：`https://api.wise.com/2026Q4`）
- **本地收款**：支持 GB、US、BE 账户详情接收；PayNow（SGD）仅作为**入金**方式，**不作为付款轨道**。
- **虚拟卡**：支持虚拟卡和实体卡；沙箱可模拟 auth、clearing、reversal、kiosk 制卡；Apple Pay / Google Pay 在沙箱中**不可用**。
- **沙箱限制**：Rates API 的 `from`/`to`/`group` 参数在沙箱中**不可测试**。
- **FX**：Quotes API 可获取报价；rate locking 需 API 接入。
- **批量转账**：Batch Groups（最多 1000 笔/批），支持直接借记或余额划拨。
- **SCA**：JOSE 签名、设备指纹、OTP。
- **适合场景**：UK/EU 为主的B2B跨境收付；如需覆盖 APAC 本地轨道，建议以 Airwallex / Nium 为主。

---

## 4. 对 Demo 多轨设计的建议

### 4.1 各轨道推荐沙箱

| 场景 | 推荐沙箱 | 理由 |
|---|---|---|
| **本地实时付款（APAC 重点）** | **Airwallex**（主）+ Nium（补充） | Airwallex 覆盖 SG FAST、HK FPS、ID BI-FAST、VN NAPAS、MY DuitNow、TH PromptPay；自助沙箱、状态模拟完善。Nium 作为 APAC 备选，自助沙箱同等方便。 |
| **欧美本地轨道（SEPA Instant / UK FPS / FedNow）** | **Airwallex** | 完整覆盖 SEPA Instant、UK Faster Payments、US FedNow，均已在本地清算网络文档中列出。 |
| **FX 报价与锁汇** | **Airwallex**（主）+ Nium（备） | Airwallex Quotes/Conversions API 可在沙箱直接测试；Nium 支持 2000+ 货币对、rate locking。 |
| **虚拟卡（Issuing）+ 卡授权模拟** | **Airwallex**（主）+ Nium（备） | Airwallex 沙箱有完整 Issuing Simulation（失败授权、增量授权、capture）；Nium 有 VIR 类型卡，模拟 API 也可测试。 |
| **收单（Payment Intents）** | **Airwallex** | Payment Intents 完整的在沙箱可用。 |
| **连连国际轨道（中国相关通道）** | 需向连连商务申请 | 连连无公开自助沙箱；如 demo 需展示中国/亚太特殊通道，需提前 2-4 周联系连连。 |

### 4.2 申请什么、预计等待多久

| PSP | 需申请 | 预计等待 |
|---|---|---|
| Airwallex | 自助注册 sandbox.airwallex.com；Connected Account API 须联系 Sales | **0 天**（沙箱立即开通）；Connected Account：1-2 周 |
| Nium | 自助注册 app.nium.com/sign-up；邮箱验证 + MFA | **0 天**（即时激活） |
| Rapyd | 自助注册 dashboard（未验证具体流程） | **0 天**（预计，需本地浏览器验证） |
| Wise | 联系 Wise tech support 申请 API credentials | **3-7 天** |
| 连连国际 | 联系 service@lianlianpay.com；需完成企业 KYB | **1-4 周** |

### 4.3 通用适配层应抽象的共同操作与状态

以下为各家 API 共性，建议在多轨适配层（multi-rail adapter）中统一抽象：

#### 操作层（Operations）

```
1. authenticate()          → 获取 access token / API key（Airwallex: Bearer token 30min; Nium: API key 常驻; Rapyd: HMAC signing）
2. getQuote(srcCcy, dstCcy, amount) → FX 报价
3. lockRate(quoteId)        → 锁汇（Airwallex: conversion; Nium: FX lock; Wise: quote lock）
4. createPayout(beneficiary, amount, rail) → 发起付款，指定本地轨道
5. getPayoutStatus(txId)    → 查询转账状态
6. simulateStatusTransition(txId, targetStatus) → 沙箱专用：推进状态
7. createVirtualCard(cardholderInfo, spendConfig) → 创建虚拟卡
8. simulateCardAuth(cardId, merchant, amount) → 模拟卡授权
9. registerWebhook(url, events) → 注册 webhook
10. getBalance(currency)    → 查询余额
```

#### 状态机（共性状态枚举）

| 通用状态 | Airwallex | Nium | Wise |
|---|---|---|---|
| 待处理 | `PENDING` | `SUBMITTED` | `processing` |
| 清算中 | `PROCESSING` | `PROCESSING` | `processing` |
| 已完成 | `SUCCESS` | `PAID` | `outgoing_payment_sent` |
| 被拒 / 退回 | `FAILED` | `RETURN` | `cancelled` |
| 合规审查 | `REQUIRES_ACTION` | RFI (`UNDER_REVIEW`) | `waiting_recipient` |

#### 轨道映射表（Rail Enum）

```
RailType:
  SG_FAST, SG_PAYNOW(未验证/Airwallex 不含),
  HK_FPS, HK_ACH,
  ID_BIFASST, ID_SKN,
  VN_NAPAS,
  MY_DUITNOW,
  TH_PROMPTPAY,
  EU_SEPA_INSTANT, EU_SEPA,
  UK_FPS, UK_CHAPS,
  US_ACH, US_FEDNOW, US_FEDWIRE,
  SWIFT
```

#### 共性 Header / Auth 封装

```python
class PSPAdapter(ABC):
    def __init__(self, env: Literal["sandbox", "production"]): ...
    def authenticate(self) -> str: ...          # 返回 Bearer token 或 API key
    def payout(self, req: PayoutRequest) -> PayoutResponse: ...
    def fx_quote(self, req: FXRequest) -> FXQuote: ...
    def issue_card(self, req: CardRequest) -> VirtualCard: ...
    def simulate(self, txId: str, state: str) -> None: ...  # sandbox only
```

---

## 附录：URL 速查

| PSP | 文档 | 沙箱 API | 注册/申请 |
|---|---|---|---|
| Airwallex | https://www.airwallex.com/docs | https://api.sandbox.airwallex.com | https://www.airwallex.com/docs（右上角） |
| Airwallex MCP（dev） | https://www.airwallex.com/docs/developer-tools/ai/developer-mcp | https://mcp.sandbox.airwallex.com/developer | 同沙箱账号 |
| Airwallex AgentOS | https://www.airwallex.com/docs/developer-tools/ai/agentos | 生产 | OAuth |
| 连连国际 | https://developer.lianlianglobal.com | 未公开 | https://global.lianlianpay.com/developer/api |
| Nium | https://docs.nium.com | https://app.nium.com（沙箱） | https://app.nium.com/sign-up |
| Nium MCP | https://docs.nium.com（Developers → MCP Server） | 沙箱 API key | 同 Nium 账号 |
| Rapyd | https://docs.rapyd.net | dashboard（API Control） | https://dashboard.rapyd.net |
| Wise Platform | https://docs.wise.com/api-docs | https://api.wise-sandbox.com/2026Q4 | 联系 tech support |
