---
title: 多轨沙箱背景材料——AWS 内部 BD / SA
date: 2026-10-20
status: 内部使用
tags: [agentic-payment, multi-rail, sandbox, bd-sa-enablement]
---

# 多轨沙箱背景材料
**AWS FSI · 内部 BD / SA 赋能 · 2026-10-20**

---

## 验证状态（2026-10-20 更新）

> 以下为 2026-10-20 在真实 AWS 环境（账号 <AWS_ACCOUNT_ID>，ap-southeast-1）完成的沙箱验证结果。验证详情见 `demo/_spike/` 目录。**不含任何密钥或生产凭证。**

| 能力 | 验证结论 | 变化说明 |
|------|---------|---------|
| WAF Web Bot Auth（ap-southeast-1）| ✅ 已验证 | 新加坡区域可用，三路径（无签名/伪签名/AgentCore Browser）均正确分类 |
| WAF Monetize → x402 v2（402）| ✅ 已验证 | 格式为 x402Version:2，Base Sepolia，0.01 USDC |
| AgentCore Payments 链上付款 | ✅ 已验证（两条路径）| x402 直付 + AgentCore Payments + Stripe Privy；预算超限基础设施层硬拒绝 |
| Airwallex 本地付款（VN/HK/SG）| ✅ 已验证 | **不需要 Admin Key**（2026-08-24 后新账号不再发放）；状态模拟改用 `/simulation/transfers/{id}/transition` |
| Airwallex 换汇 | ✅ 已验证 | 4 货币对 SETTLED；标准 API Key 可用 |
| Airwallex 发卡 + 授权模拟 | ✅ 已验证 | VISA 虚拟卡 ACTIVE；49→APPROVED，80→DECLINED；无需 Admin Key |
| Nium FX 换汇 | ✅ 已验证 | SGD→USD，status=processing，实时可跑 |
| Nium 本地付款 / 虚拟卡 | ⚠️ 需开通 | 默认沙箱 fundingInstrumentType=RESTRICTED，cardTxnProductCode=null；需 Nium 账号团队配置 |
| AgentCore Payments 钱包 | ✅ 改用 Stripe Privy | Coinbase CDP 连接器需 Marketplace 订阅（订阅失败）；Stripe Privy 无需订阅，用户会前通过 Privy 前端授权即可（约 5 分钟）|

**关键踩坑（影响 SA 上手）：**
1. WAF 规则须同时匹配 `wba:failed` 与 `wba:invalid`，只匹配 `invalid` 会放行伪造签名。
2. AgentCore Payments 服务角色权限须手动补齐（`TagResource`、`GetResourcePaymentToken`、`secretsmanager:GetSecretValue`）。
3. `ProcessPayment` 只返回 x402 payload 部分，调用方须自行组装完整 `PaymentPayload` 并 base64 编码；建议直接用 Strands SDK。

---

> 阅读约定：  
> - **【已定】** = 用户在讨论中明确决定，不再变更  
> - **「未验证」** = 调研中未能通过一手文档确认  
> - **「推断」** = 逻辑推断，有待确认  
> - 所有事实均引用一手来源 URL；无 URL 的为公开常识或推断

---

## 0. 一页摘要

### 核心结论

四条轨道全部做**真实 API 调用、不动真钱**【已定】：

| 轨道 | 方案 | 就绪状态 |
|------|------|---------|
| 稳定币 | AgentCore Payments（x402 / MPP），Base Sepolia 测试网 + Tempo testnet | **今日可用** |
| 本地实时 + 换汇 | Airwallex 沙箱（主）+ Nium 沙箱（补充）| **自助注册，即时开通** |
| 卡（发卡 + 授权模拟）| 同一 PSP 沙箱（Airwallex / Nium）的 Issuing 功能 | **自助注册，即时开通** |
| 卡（Stripe 路径）| Stripe Issuing → SPT → MPP；需美国主体或合作伙伴账号 | **可选，通过合作团队申请** |
| 连连国际 | POC 阶段由客户用自己的沙箱凭证接入 | **需提前说明** |
| 卡组织（Visa TAP）| 本地开源实现演示商户验签 | **今日可用** |
| 卡组织（Visa VIC）| 真实沙箱通过合作渠道并行申请 | **数周到数月** |
| 卡组织（Mastercard AP）| 无公开沙箱，不纳入 | **不纳入** |
| 商业层协议（AP2）| 本地可运行，仅需 Google API Key | **今日可用** |

### 沙箱按客户切换原则【已定】
- **Airwallex 场景**：用 Airwallex 沙箱；不在 Airwallex 客户面前展示 Nium。
- **连连及通用场景**：用 Nium 沙箱。
- 两家沙箱均支持自助开通。Nium 发卡与本地付款需 Nium 账号团队开通（默认 RESTRICTED 配置）；Airwallex 发卡与状态模拟**不需要 Admin Key**（✅ 2026-10-20 已验证）。

---

## 1. 为什么多轨对出海支付机构重要

### 1.1 场景驱动：一张支付轨道不够

出海支付机构的企业客户往往在同一天需要：

| 场景 | 需要什么轨道 | 为什么不能只用一条 |
|------|------------|----------------|
| AI agent 按次付费调用 API / MCP 工具 | 稳定币（x402 / MPP）| 金额极小（0.01 USD），卡网络单笔成本不合算；需要无人值守自动执行 |
| 向东南亚供应商批量付款 | VN NAPAS / SG FAST / ID BI-FAST / MY DuitNow | 收款方无法收链上 USDC；本地 T+0 到账是合同要求 |
| 企业差旅与 SaaS 订阅 | 虚拟卡（Visa / MC 网络）| 全球商户只接卡，不接稳定币；需要支出限额与 MCC 控制 |
| 大额跨境 B2B 货款 | SWIFT / 本地大额清算（RTGS）| 金额大、需要汇率锁定、可撤销 |

### 1.2 出海机构的痛点

1. **协议碎片化**：agent 发起的支付请求可能带 x402、MPP、ACP、AP2 中的任意一种头部，商户侧需要全部兼容。
2. **轨道碎片化**：同一个企业的供应商分布在 10+ 个国家，每个国家有自己的本地最优轨道。
3. **合规负担**：不同轨道有不同的 KYC/AML 要求、汇率管制、报告义务。
4. **可撤销性差异**：稳定币链上不可撤销；卡有争议期；本地实时支付各异。

### 1.3 PSP 的多轨价值

PSP（如 Airwallex、Nium）的核心价值是：**一套 API 接多条轨道、一份牌照覆盖多个监管辖区**。Airwallex 持有 85+ 张牌照，Nium 声称覆盖 40+ 条本地支付轨道。这使得 demo 中的"轨道切换"不是技术玩具，而是真实业务需求。

---

## 2. 轨道 × 沙箱总表

### 2.1 总表

| 轨道 | 提供方 | 沙箱能力 | 申请门槛 | 地区限制 | 就绪时间 | demo 中的角色 | 按客户切换 |
|------|--------|---------|---------|---------|---------|------------|----------|
| x402（Base Sepolia）稳定币 | Coinbase CDP + AgentCore Payments | x402 协议全流程；WAF Monetize TEST 模式 | Coinbase CDP 账号（30 分钟自助）| 无 | **今日** | 稳定币轨主线 | 通用，不需切换 |
| MPP（Tempo testnet）稳定币 | Stripe + AgentCore Payments | Stripe sandbox 自动切 testnet；`mppx validate` | Stripe Privy 账号（自助；AgentCore Payments 经 StripePrivy 连接器使用 MPP / Tempo testnet）| 无（testnet）| **今日** | 稳定币轨对比展示 | 通用 |
| 本地实时付款（APAC）| Airwallex 沙箱 | SG FAST、HK FPS、VN NAPAS、ID BI-FAST、MY DuitNow、TH PromptPay 等；Simulation API | 自助注册（0 天）| 无 | **今日** | Airwallex 场景主轨 | **Airwallex 场景专用** |
| 本地实时付款（APAC 备选）| Nium 沙箱 | 40+ 条本地轨道（具体清单「需开通后核实」）；Magic value 状态模拟 | 自助注册（0 天）| 无 | **今日** | 连连 / 通用场景 | **非 Airwallex 场景** |
| FX 报价与锁汇 | Airwallex 沙箱 | Quotes/Conversions API；沙箱 Web App 可直接操作 | 同上 | 无 | **今日** | Airwallex 场景 | Airwallex 专用 |
| FX 报价与锁汇（备选）| Nium 沙箱 | 2000+ 货币对；rate locking | 同上 | 无 | **今日** | 通用场景 | 非 Airwallex 场景 |
| 虚拟卡发卡 + 授权模拟 | Airwallex 沙箱 Issuing | ✅ VISA 虚拟卡 ACTIVE；49→APPROVED / 80→DECLINED；无需 Admin Key（2026-10-20 验证）| 同上 | 无 | **今日** | 卡轨（Airwallex 场景）| Airwallex 专用 |
| 虚拟卡发卡 + 授权模拟（备选）| Nium 沙箱 | ⚠️ `cardTxnProductCode=null`；`fundingInstrumentType=RESTRICTED`；本地付款和卡均需 Nium 账号团队开通（2026-10-20 验证）| 同上 | 无 | **需开通** | 卡轨（通用场景）| 非 Airwallex 场景 |
| 虚拟卡（Stripe 路径）| Stripe Issuing | 完整 test helpers；MCC 控制；支出限额 | 需美国主体；Issuing 审批 1–3 天 | **SG/HK 不支持直接注册** | **3–7 天** | **可选**，通过合作团队申请 | — |
| 连连国际 | 连连沙箱（客户凭证）| 「未验证」，等客户自测 | 需 KYB（1–4 周）| 无公开自助沙箱 | **POC 阶段** | 讲点："换上你们的接口就能跑" | 连连场景由客户提供 |
| 商户验签（Visa TAP）| `visa/trusted-agent-protocol`（开源）| 5 个服务本地运行，无 Visa 凭证要求 | 无 | 无 | **今日** | 卡组织层验签演示 | 通用 |
| Visa Intelligent Commerce | Visa VIC 沙箱（`sandbox.mcp.visa.com`）| `enroll-card`、`initiate-purchase-instruction` 等 | 需通过 Visa VIC onboarding，不自助 | 无 | **数周到数月** | 并行申请，第一版不依赖 | 通用 |
| Mastercard Agent Pay | 无公开沙箱 | 「未验证」| 需合作关系 | — | **不纳入** | 架构图中标注 | — |
| AP2 商业层 | Google AP2（开源）| 本地可运行，含卡支付 mock 场景 | Google API Key（5 分钟）| 无 | **今日** | 商业层协议演示 | 通用 |

### 2.2 关键说明

- **稳定币轨是 AWS 原生**：AgentCore Payments 原生支持 x402 和 MPP，买方和卖方（WAF Monetize）都在 AWS 内闭环，无需 PSP 配合。
- **本地实时轨依赖 PSP**：AgentCore Payments 没有本地清算能力，必须通过 Airwallex / Nium 的 API 调用。
- **卡轨用 PSP Issuing**：Airwallex 和 Nium 在自己的沙箱里已经有发卡功能，不需要 Stripe。Stripe 只作为可选项，通过合作渠道申请。
- **连连无法自助**：调研确认连连国际没有公开免 KYB 沙箱，只能在 POC 阶段由客户提供凭证，demo 架构预留接口。

---

## 3. 分项详解

### 3.1 稳定币轨：x402（Base Sepolia）+ MPP（Tempo testnet）

#### x402 沙箱

- **Base Sepolia**（EIP-155 链 ID `84532`）：Stripe sandbox 账号调 `POST /v1/crypto/deposit_addresses -d network=base` 可获取测试网地址（`sk_test_...` 自动切 testnet）。
- **公共 facilitator**：`https://x402.org/facilitator`，无需 CDP 账号即可测试。
  - 来源：https://docs.stripe.com/payments/machine/x402
- **Circle 水龙头**：`https://faucet.circle.com/` 发放 Base Sepolia 测试 USDC。
- **AgentCore Payments 连接**：`paymentType: CRYPTO_X402`，`network: "eip155:84532"`，Coinbase CDP 或 Stripe Privy 连接器均可。
  - 来源：https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments-process-payment.html

#### MPP / Tempo testnet

- Stripe sandbox 密钥（`sk_test_...`）下调 `POST /v1/crypto/deposit_addresses -d network=tempo`，`mppx` 自动切 Tempo testnet。
- 测试工具：`npx mppx@latest validate http://localhost:4242`（跑完整 roundtrip）。
  - 来源：https://docs.stripe.com/payments/machine/mpp
- AgentCore Payments 通过 `paymentType: MPP` 支持，转发 `WWW-Authenticate: Payment` challenge。

#### AgentCore Payments 与 MPP 的关系澄清

- AgentCore Payments 支持 MPP，但**在 AgentCore 层面 MPP 是纯稳定币轨**（EVM/Tempo/Solana USDC），不含 Stripe 的卡（SPT）路径。
- MPP 的卡支付是 Stripe 服务端特性（SPT → MPP merchant），需要独立 Stripe 账号，地区受限（US/CA/EU），SG/HK 不支持直接注册。
  - 来源：https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments-how-it-works.html

---

### 3.2 本地实时支付与换汇：Airwallex + Nium（附 Rapyd、Wise 对照）

#### Airwallex 沙箱

- **注册**：访问 https://www.airwallex.com/docs 右上角"Get started in sandbox"，即时激活，无 KYB。创建额外账户用 Entity Name `"New Business Sandbox"` 或 `"Sandbox Business"`。
  - 来源：https://www.airwallex.com/docs/developer-tools/sandbox-environment/sandbox-environment-overview
- **本地轨道**：SG（FAST / GIRO / RTGS）、HK（FPS / ACH / RTGS）、VN（NAPAS）、ID（BI-FAST / SKN）、MY（DuitNow）、TH（PromptPay）、EU（SEPA Instant）、UK（Faster Payments）、US（FedNow / ACH）等 100+ 国家。
  - 注：PayNow（SG）**未在 Airwallex payouts 文档单独列出**；SG 轨道文档显示为 FAST/GIRO/RTGS「未验证 PayNow」。
  - 来源：https://www.airwallex.com/docs/payouts/payout-network/bank-accounts
- **Simulation API**：`POST /v1/transfers/simulate` 推进转账状态，支持 `PROCESSING → SUCCESS / FAILED`。
- **FX 锁汇**：`POST /v1/fx/quotes` → `POST /v1/conversions`（创建锁汇）。
- **Issuing（虚拟卡）**（✅ 2026-10-20 实测验证，无需 Admin Key）：
  - 发卡：`POST /api/v1/issuing/cards/create`，`program: {"purpose":"COMMERCIAL","type":"DEBIT"}`（不需要 program ID），`form_factor: VIRTUAL`，持卡人类型 DELEGATE。
  - 模拟授权：`POST /api/v1/simulation/issuing/create`，`transaction_amount: 49`→APPROVED，`80`→DECLINED（单笔限额 50 USD）。
  - 状态模拟（转账）：`POST /api/v1/simulation/transfers/{id}/transition`，body `{"next_status":"SENT"}` → `{"next_status":"PAID"}`（旧路径 `/simulations/transfers/{id}/succeed` 需 Admin Key，已弃用）。
  - 来源：airwallex-sandbox-verification.md 补核节（2026-10-20）
- **MCP Server**：Airwallex Developer MCP（沙箱）可挂到 AgentCore Gateway 上。
  - `claude mcp add-json airwallex-dev '{"type":"http","url":"https://mcp.sandbox.airwallex.com/developer"}'`
  - 来源：https://www.airwallex.com/docs/developer-tools/ai/developer-mcp
- **Python SDK**：无官方 Python SDK；Node.js SDK（`@airwallex/node-sdk`，Beta）可用。

#### Nium 沙箱

- **注册**：https://app.nium.com/sign-up，邮箱 + MFA，即时激活，无 KYB。
  - 来源：https://docs.nium.com
- **本地轨道**：声称覆盖"40+ local payment rails"；APAC 具体覆盖在 Nium Atlas 中「需注册后确认清单」。
- **Magic values**：账号末尾 `987654321` = PAID；`897654321` = 银行拒绝，便于状态测试。
- **虚拟卡**：⚠️ **2026-10-20 实测**：`cardTxnProductCode: null`，`fundingInstrumentType: "RESTRICTED"`，`plasticId` / `cardProductId` 未配置；虚拟卡和本地付款均被阻止。**需 Nium 账号团队开通后方可使用。** 换汇（FX）不受影响，实时可用。
- **FX**：2000+ 货币对，支持 rate locking。
- **MCP Server**：官方 Nium MCP Server，支持 Claude Code / Cursor 等。
  - 来源：https://docs.nium.com（Developers → Building with AI → Nium MCP Server）
- **SDK**：Node, Python, Go, Java 均有。

#### Rapyd（对照参考）

- 自助注册 dashboard.rapyd.net；100+ 国本地收款，190+ 国付款。
- 本次调研时 docs.rapyd.net 页面**无法加载（JS 渲染失败）**，文档内容「未验证」。
- 不作为主选，备用参考。

#### Wise Platform（对照参考）

- 沙箱需联系 tech support 申请 API credentials，等待 3–7 天；PayNow（SGD）仅作为**入金方式**，不作付款轨道。
- 覆盖以 UK/EU 为主，APAC 本地轨道不及 Airwallex / Nium，不作为 demo 主选。
  - 来源：https://docs.wise.com/api-docs

---

### 3.3 卡轨：PSP 发卡（主选）与 Stripe（可选）

#### PSP 发卡（主选：Airwallex / Nium Issuing）

- **原则**：Airwallex 场景用 Airwallex Issuing，通用 / 连连场景用 Nium Issuing。不需要 Stripe。
- **能演示什么**：为 agent 创建虚拟卡、设置 MCC 支出限额、模拟授权成功 / 拒绝 / 增量 / 清算。
- **Airwallex 已确认**：Simulation APIs 在沙箱文档中明确列出（见 3.2）。
- **Nium 需确认**：`cardType: VIR` 在文档中列出，发卡 Simulation API 细节「需注册后核实」。

#### Stripe（可选路径，通过合作团队申请）

**为什么不由用户个人申请**：Stripe Issuing 的本地发卡支持不含 SG/HK（官方 22 国列表不含两地）；SPT（Shared Payment Token）同样不支持 SG/HK。以 SG/HK 注册主体直接申请会被拒或功能受限。

**正确路径**：通过 **AWS 与 Stripe 的合作团队申请合作伙伴测试环境**，合作团队可以安排一个 US 实体的 partner 沙箱账号。

**Stripe Issuing 沙箱能力**（已确认，限 US/CA/EU 主体）：
- 创建 Cardholder → 发行虚拟卡 → `POST /v1/test_helpers/issuing/authorizations` 模拟授权 → Capture / 拒绝。
- 所有 Spending Controls（MCC、金额限制、每月限额）在 sandbox 全部可测。
- 来源：https://docs.stripe.com/issuing/testing?testing-method=with-code

**Stripe SPT（Shared Payment Token）**：
- 测试 helper：`POST /v1/test_helpers/shared_payment/granted_tokens`。
- Preview 阶段（API version `2026-09-30.preview`）；需同意 preview 条款。
- 测试卖家 profile 固定值：`profile_test_61TU90nIeGjU7NNVXA6TU90m7ISQWsBxpcx9lASWWXTk`。
- 来源：https://docs.stripe.com/agentic-commerce/concepts/shared-payment-tokens

---

### 3.4 连连国际：POC 阶段由客户接入

**现状**：连连国际（LianLian Global）**没有公开免 KYB 沙箱**。注册需完成企业认证（KYB），等待 1–4 周。开发者文档托管在 Stoplight（https://developer.lianlianglobal.com），页面加载依赖 JavaScript，沙箱能力页面「未能验证」。

**决定**【已定】：
- 连连不由 SA 自行申请沙箱。
- 在 POC 阶段，由连连客户（或连连方）提供自己的沙箱凭证，SA 负责接口适配。
- Demo 架构预留 `PSPAdapter` 接口，适配连连的认证方式（Bearer token，见 psp-sandboxes.md 第 2.2 节代码示例）。
- **讲点**：面向连连演示时明确说"换上你们的接口就能跑"，这既是务实口径也是商业价值点。

---

### 3.5 卡组织协议

#### Visa Trusted Agent Protocol（TAP）——今日可运行

- 仓库：https://github.com/visa/trusted-agent-protocol（213 stars，2026-10-14 更新）
- **无需 Visa 凭证**，完全本地可运行，`pip install -r requirements.txt` 启动 5 个服务：
  - agent-registry (8001)、merchant-backend (8000)、cdn-proxy (3002)、merchant-frontend (3001)、tap-agent (8501, Streamlit)
- 技术：RFC 9421 HTTP 签名；演示商户验签（区分 bot vs 合法 agent）、防重放。
- **demo 中的角色**：展示"商户如何验证 agent 身份"的卡组织层信任框架。

#### Visa Intelligent Commerce（VIC）——需申请，并行推进

- 沙箱端点：`https://sandbox.mcp.visa.com`（已确认）。
- **不自助**：developer.visa.com 的 VIC 页面显示 "No access — request access to view"，需通过 Visa VIC onboarding 获得 API 凭证（`VISA_VIC_API_KEY`、证书、外部 App ID 等）。
  - 来源：https://raw.githubusercontent.com/visa/ai/main/apps/vic-mcp-examples/.env.example
- **建议**：并行申请，但第一版 demo 不依赖 VIC。申请入口：https://developer.visa.com/capabilities/visa-intelligent-commerce
- **申请周期**：「未验证」，业界通常为数周到数月（partner onboarding）。

#### Mastercard Agent Pay——不纳入

- `developer.mastercard.com/product/agent-pay` 无法访问，无公开自助沙箱「未验证」。
- **AP4M**（Agent Pay for Machines）2026-06-10 推出，面向机器间微支付，无公开 API。
- **决定**【已定】：本次 demo 不纳入 Mastercard Agent Pay；架构图中作为"卡轨道 B"标注，说明需合作关系。

---

### 3.6 商业层协议：ACP、AP2、UCP 可运行性

#### AP2（首选演示协议）

| 项目 | 详情 |
|------|------|
| 仓库 | https://github.com/google-agentic-commerce/AP2（3237 stars）|
| 是否本地可运行 | **是**，`bash code/samples/python/scenarios/a2a/human-present/cards/run.sh` |
| 所需凭证 | `GOOGLE_API_KEY`（免费，5 分钟申请：https://aistudio.google.com/apikey）|
| 卡支付场景 | 模拟 DPAN + mock OTP `123`，**无真实卡网络调用** |
| 是否可部署到 AgentCore Runtime | 是，标准 Python 服务可容器化 |
| 注意 | AP2 已捐赠给 FIDO Alliance，是中立标准，不属于"Google 的私有协议" |

**5 分钟快速验证**：
```bash
git clone https://github.com/google-agentic-commerce/AP2
cd AP2
export GOOGLE_API_KEY=your_key
bash code/samples/python/scenarios/a2a/human-present/cards/run.sh
# 打开 http://0.0.0.0:8000/dev-ui，选 shopping_agent
```

#### ACP（Stripe 路径，按需使用）

- 规范仓库：https://github.com/agentic-commerce-protocol/agentic-commerce-protocol（OpenAI + Stripe，beta）
- **纯规范**，无独立可运行服务器；必须依赖 Stripe 或 OpenAI 账号测试真实 checkout。
- **不推荐作为独立 demo 起点**；若已有 Stripe 沙箱账号（合作团队申请后），可附带演示。

#### UCP（NVIDIA 参考实现）

- 仓库：https://github.com/NVIDIA-AI-Blueprints/Retail-Agentic-Commerce（ACP + UCP 双协议）
- 所需：NVIDIA API Key（免费，https://build.nvidia.com/settings/api-keys）+ Docker。
- `docker compose up` 启动 8 个服务；内置模拟 PSP，可替换真实 PSP。
- **可选补充**：作为"商户侧接 UCP 的参考实现"展示。

---

## 4. 支付路由与通用适配层

### 4.1 共同操作抽象

依据 psp-sandboxes.md 第 4.3 节，多轨适配层应统一以下操作：

```python
class PSPAdapter(ABC):
    def authenticate(self) -> str: ...       # Bearer token / API key
    def fx_quote(self, src, dst, amount) -> FXQuote: ...
    def lock_rate(self, quote_id) -> str: ...
    def payout(self, req: PayoutRequest) -> PayoutResponse: ...
    def get_payout_status(self, tx_id) -> PayoutStatus: ...
    def simulate(self, tx_id, target_state) -> None: ...  # sandbox only
    def issue_card(self, req: CardRequest) -> VirtualCard: ...
    def simulate_card_auth(self, card_id, merchant, amount) -> AuthResult: ...
    def get_balance(self, currency) -> Decimal: ...
```

### 4.2 状态机枚举（跨 PSP 通用）

| 通用状态 | Airwallex | Nium | Wise |
|---------|-----------|------|------|
| 待处理 | `PENDING` | `SUBMITTED` | `processing` |
| 清算中 | `PROCESSING` | `PROCESSING` | `processing` |
| 已完成 | `SUCCESS` | `PAID` | `outgoing_payment_sent` |
| 被拒 / 退回 | `FAILED` | `RETURN` | `cancelled` |
| 合规审查 | `REQUIRES_ACTION` | `UNDER_REVIEW` (RFI) | `waiting_recipient` |

### 4.3 轨道枚举

来源：psp-sandboxes.md 第 4.3 节

```
RailType:
  # 东南亚
  SG_FAST, HK_FPS, ID_BIFASST, VN_NAPAS, MY_DUITNOW, TH_PROMPTPAY,
  # 欧洲
  EU_SEPA_INSTANT, EU_SEPA, UK_FPS, UK_CHAPS,
  # 美洲
  US_ACH, US_FEDNOW, US_FEDWIRE,
  # 稳定币
  STABLECOIN_X402_BASE_SEPOLIA, STABLECOIN_MPP_TEMPO,
  # 卡
  CARD_VIRTUAL,
  # 通用
  SWIFT
```

注：`SG_PAYNOW` 在 Airwallex payouts 文档中**未单独列出**「未验证」；`ID_BIFASST` 来自文档字段名「需确认拼写」。

### 4.4 路由决策维度与示例

路由选择应考虑以下维度（「推断」，具体权重按业务需求调整）：

| 维度 | 说明 |
|------|------|
| 成本 | 稳定币链上成本极低；卡交换费 1.5–2.9%、SWIFT 单笔 15–30 USD（行业示意值，demo 中以 PSP 沙箱报价 / 配置为准，推断） |
| 速度 | 稳定币 <1 分钟；本地实时 T+0；ACH T+1；SWIFT T+2–3 |
| 最小金额 | x402 最低 0.01 USDC；本地轨道通常无下限；Stripe SPT 最低 0.50 USD |
| 可撤销性 | 链上不可撤；卡有争议期（120 天）；本地实时支付视轨道而定 |
| 收款方接受度 | 供应商可能只接当地法币；agent 微服务只接链上 USDC |
| 通道可用性 | 取决于收款方所在国家和 PSP 的轨道覆盖 |

**示例路由决策**（来自 demo 设计草案）：

| 场景 | 金额 | 路由 | 理由 |
|------|------|------|------|
| agent 按次付费 API 调用 | $0.01 | `STABLECOIN_X402_BASE_SEPOLIA` | 成本最低，无人值守，速度快 |
| 越南供应商货款 | $300 | `VN_NAPAS` + FX 锁汇 | 收款方在越南，法币到账；锁汇规避汇率风险 |
| 美国 SaaS 订阅 | $49 | `CARD_VIRTUAL` | 商户只接卡；需要支出限额 |
| 欧盟大额 B2B | €5,000 | `EU_SEPA_INSTANT` 或 `SWIFT` | 金额大，可撤销性重要 |

### 4.5 路由通过 Cedar 策略与人工审批

所有路由决策**必须经过 Cedar 策略检查**，并在超过审批阈值时进入 Step Functions 人工审批流，再执行——这是 demo 安全主线的核心。路由层只负责"选哪条轨道最优"，不负责授权。

---

## 5. 对 BD 的讲法

### 5.1 客户会问什么、怎么答

| 客户问题 | 推荐口径 |
|---------|---------|
| "你们的 agent 支付支持我们的本地轨道吗？" | "AgentCore Payments 原生支持稳定币轨道，本地实时支付轨道通过 PSP 适配层接入，演示中用的是 Airwallex 沙箱，支持 [列出客户关心的轨道]。" |
| "Stripe 已经都支持了，AWS 有什么优势？" | "Stripe 是支付服务商，AWS 是基础设施。AgentCore Payments 本身可以接 Stripe 的钱包（Privy），两者互补。AWS 的价值在于：限额在基础设施层强制执行，不靠 prompt；整个支付链路有完整的审计、策略和安全控制。" |
| "这个 demo 是真实的还是模拟的？" | "稳定币轨道是真实的测试网调用，链上有交易哈希可查。本地实时支付用 PSP 沙箱，API 是真实的，但走的是测试环境，不动真钱。卡组织层用 Visa 的开源实现做验签演示。" |
| "连连的接口支持吗？" | "我们的适配层预留了连连接口，连连用你们自己的沙箱凭证接入就能跑——这是这套方案的设计原则：换接口，不换架构。" |
| "为什么 Airwallex 场景不用 Nium？" | "我们在 [客户名] 的场景里展示的是 [客户名] 自己的沙箱和产品名称，不会在同一场合展示竞争对手的系统。" |

### 5.2 哪些不能说成已有合作

- **Stripe 卡 + 发卡**：Stripe Issuing / SPT 的 APAC 限制是真实的，不要声称"Stripe 已支持"。通过合作团队申请的 partner 测试环境是内部过程，不是公开合作。
- **Visa VIC 集成**：VIC 沙箱需要通过 Visa 的 onboarding，未完成前不能声称"已接 Visa VIC"。
- **Airwallex / Nium 的 MCP 挂接**：两家都有 MCP Server，但挂到 AgentCore Gateway 上受 Cedar 管控的方案「需核实」，不能说"已经做好"。
- **连连合作**：连连无自助沙箱，目前是"接口预留"，不是"已集成"。

### 5.3 按客户切换沙箱的原因及怎么讲

**原因**：不要在一个客户面前演示其竞争对手的系统。PSP 之间互相视对方为竞争，在 Airwallex 的会议室里出现 Nium 的品牌会立即破坏信任。

**讲法**：
- 面向 Airwallex："这套 demo 接的是 Airwallex 的沙箱，所有支付轨道名称和品牌都是 Airwallex 的体系。"
- 面向连连："这套框架设计上就是可插拔的，你们用自己的 API Key 接进来，其他的不用改。"
- 面向通用场景："这里用的是 Nium 沙箱，因为它完全自助开通，演示起来最方便。换成任何支持本地实时支付的 PSP 都是一样的流程。"

---

## 6. 给 SA 的行动清单

### 6.1 SA 可自助完成（今日起可做，无需等待）

| 任务 | 步骤 | 预计时间 |
|------|------|---------|
| 开通 Airwallex 沙箱 | 访问 https://www.airwallex.com/docs → "Get started in sandbox" | **即时**（<5 分钟）|
| 开通 Nium 沙箱 | 访问 https://app.nium.com/sign-up | **即时**（<5 分钟）|
| 获取 Google API Key（AP2）| 访问 https://aistudio.google.com/apikey | **即时**（<5 分钟）|
| 获取 NVIDIA API Key（UCP）| 访问 https://build.nvidia.com/settings/api-keys | **即时**（<5 分钟）|
| 开通 Coinbase CDP（x402 / AgentCore Payments）| 访问 https://portal.cdp.coinbase.com | **<30 分钟**（自助 OAuth）|
| 领取 Base Sepolia 测试 USDC | 访问 https://faucet.circle.com/ | **即时** |
| 运行 Visa TAP 本地 demo | `git clone https://github.com/visa/trusted-agent-protocol` + `pip install -r requirements.txt` | **<30 分钟** |
| 运行 AP2 本地 demo | `git clone https://github.com/google-agentic-commerce/AP2` + 设置 `GOOGLE_API_KEY` | **<15 分钟** |
| 注册 Stripe 沙箱账号（测试 MPP） | 访问 https://dashboard.stripe.com/register | **即时**（<10 分钟）|

### 6.2 需要走合作团队或等待批准

| 任务 | 负责方 | 步骤 | 预计时间 |
|------|--------|------|---------|
| Stripe 卡轨（Issuing + SPT + MPP 卡路径）| SA + 合作团队 | 联系 AWS 与 Stripe 的合作团队，申请 partner 测试环境（US 主体）| **1–2 周** |
| Nium Issuing 能力核实 | SA | 注册后测试 `cardType: VIR` API，检查 Simulation 文档 | **注册后 1 天** |
| Visa VIC onboarding | SA 发起 + 等待 Visa 审批 | 登录 https://developer.visa.com，点击 "request access" 申请 | **数周到数月** |
| Airwallex Connected Account API | SA + Airwallex Sales | 联系 Airwallex Sales，说明嵌入式金融场景 | **1–2 周** |
| 连连国际沙箱 | 客户提供 | 在 POC 阶段由客户方提供凭证，SA 负责接口适配 | **客户安排** |

### 6.3 申请优先级与顺序

**第一步（今日）**：Airwallex 沙箱 + Nium 沙箱 + Coinbase CDP + Google API Key。

**第二步（本周）**：运行 Visa TAP 本地 demo 确认能跑通；运行 AP2 本地 demo 确认能跑通；用 Airwallex Simulation API 跑一笔模拟 NAPAS 付款。

**第三步（本周到下周）**：通过合作团队联系 Stripe 申请 partner 测试环境；发起 Visa VIC access request（并行，不阻塞主线）。

**第四步（POC 阶段）**：对接连连凭证；集成 PSP MCP Server 到 AgentCore Gateway。

---

## 7. 风险与未验证项

| # | 项目 | 类型 | 影响 | 当前状态 | 建议行动 |
|---|------|------|------|---------|---------|
| R1 | Nium Issuing Simulation API 细节（发卡 + 模拟授权）| ✅ **已验证（结论：需开通）** | 通用/连连场景卡轨 | 2026-10-20 实测：`cardTxnProductCode=null`，`fundingInstrumentType=RESTRICTED`；本地付款和卡均被阻止，FX 可用 | 联系 Nium 账号团队开通 |
| R2 | Airwallex PayNow（SG）payouts 支持 | **未验证** | SG 本地轨道完整性 | payouts 文档显示 FAST/GIRO/RTGS，PayNow 未单独列出 | 注册后查 API Reference |
| R3 | Nium APAC 本地轨道具体清单 | **未验证** | 通用场景的轨道覆盖 | "40+ rails"为营销说法，Atlas 内容未查看 | 注册后查 Nium Atlas |
| R4 | Airwallex / Nium MCP 挂接 AgentCore Gateway 的 Cedar 管控细节 | **未验证** | S6 场景（工具分发）| 两家都有 MCP Server，但与 Gateway + Cedar 的集成「需核实」| 在 Gateway 接入时验证 |
| R5 | Stripe SPT preview 条款是否允许演示用途 | **未验证** | Stripe 卡路径（可选）| `2026-09-30.preview` 仍为 preview，条款需合规确认 | 合作团队申请时一并确认 |
| R6 | Visa VIC onboarding 实际周期 | **未验证** | VIC 集成时间规划 | "数周到数月"为业界通常估计，Visa 未公开具体 SLA | 发起申请后跟踪 |
| R7 | AP2 部署到 AgentCore Runtime 的兼容性 | **推断** | 商业层协议集成 | AP2 为标准 Python 服务，理论上可容器化，但未实际测试 | 第二周 spike 时验证 |
| R8 | Airwallex `ID_BIFASST` 字段拼写 | **未验证** | 轨道枚举准确性 | 调研中基于文档截图重建，实际字段名需查 API Reference | 调用 API 时确认 |
| R9 | 连连国际 Stoplight 文档沙箱能力页面 | **未验证** | 连连接口适配工作量 | 页面 JS 渲染失败，未能加载具体 API | POC 阶段由客户提供文档 |
| R10 | Rapyd 沙箱能力（docs 无法加载）| **未验证** | 备选 PSP 可行性 | 本次调研时页面报错，全部内容未验证 | 在本地浏览器重新验证 |

---

*文档路径：`topics/agentic-payment/demo/2026-10-20-multi-rail-sandbox-background.md`*  
*来源调研文件：`_recon/stripe-mpp.md`、`_recon/psp-sandboxes.md`、`_recon/cards-protocols.md`*  
*关联文档：`2026-10-20-agentic-payment-demo-design.md`（demo 设计草案，只读）*

---

## 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2026-10-20（初版）| 文档创建，基于调研报告汇总多轨沙箱信息 |
| 2026-10-20（验证更新）| 同步 demo 沙箱验证结果（spike 报告）：① 新增文首"验证状态"节；② Airwallex 发卡 / 授权模拟标记"已验证，无需 Admin Key"；③ Nium 本地付款 / 虚拟卡标记"需 Nium 账号团队开通（RESTRICTED 配置）"，FX 换汇已验证；④ Airwallex Issuing 端点更新为实测路径（`/simulation/transfers/{id}/transition`，`/simulation/issuing/create`）；⑤ AgentCore Payments 钱包改为 Stripe Privy（Coinbase CDP 需 Marketplace 订阅，失败）；⑥ 沙箱申请清单补 Privy 会前授权说明；⑦ 更新 R1 风险项为已验证结论；⑧ 更新 0.一页摘要中 Nium Admin Key 表述。|
