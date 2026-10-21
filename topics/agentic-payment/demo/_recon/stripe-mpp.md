# Stripe · MPP · x402 生态沙箱调研
**调研日期**: 2026-10-20  
**目标**: 验证各能力能否在测试模式下真实跑通，供跨境支付演示 demo 使用

---

## 1. Stripe Shared Payment Token (SPT) / ACP Delegated Payment

**来源**: https://docs.stripe.com/agentic-commerce/concepts/shared-payment-tokens?agent-seller=agent  
https://docs.stripe.com/agentic-commerce/concepts/shared-payment-tokens?agent-seller=seller

### 测试模式支持
**完整支持**。Stripe 为 SPT 提供专门的 test helpers：

- **卖家侧（Seller）** 无需真实 agent，用 sandbox secret key 调用：
  ```
  POST /v1/test_helpers/shared_payment/granted_tokens
  -d payment_method=pm_card_visa
  -d "usage_limits[currency]=usd"
  -d "usage_limits[max_amount]=1000"
  -d "usage_limits[expires_at]=1798761600"
  Header: Stripe-Version: 2026-09-30.preview
  ```
- **Agent 侧** 发行 SPT 时，测试卖家 profile 固定值：`profile_test_61TU90nIeGjU7NNVXA6TU90m7ISQWsBxpcx9lASWWXTk`
- **`link-cli`** 可从个人 Link 账号发行真实模式 SPT 用于集成测试：
  ```
  npx @stripe/link-cli auth login
  npx @stripe/link-cli spend-request create --payment-method-id csmrpd_xxx ...
  ```

### 申请门槛
- 自助注册 Stripe 账号即可开始
- 需同意 preview 条款：https://stripe.com/legal/ssa-service-terms#stripe-agentic-commerce-agent-services-preview
- 需在 Dashboard 创建 Stripe Profile（`profile_xxx`）
- 目前为 **preview 阶段**（API version `2026-09-30.preview`）

### 地区限制（重要）
**SPT 支持的国家/地区**（官方列表）：AT BE BG CA CH CY CZ DE DK EE ES FI FR GB GI GR HR HU IE IT LI LT LU LV MT NL NO PL PT RO SE SI SK **US**  
**新加坡（SG）和香港（HK）均不在支持列表中。**  
如使用 SG/HK 主体，需以 US 或 EU 主体注册 Stripe 账号，或通过 Stripe Connect 平台间接使用。

### 关键 API 对象
| 对象 | 说明 |
|------|------|
| `shared_payment.issued_token` | Agent 侧发行 SPT |
| `shared_payment.granted_token` | Seller 侧接收 SPT |
| `POST /v1/shared_payment/issued_tokens` | Agent 发行 |
| `POST /v1/payment_intents` with `payment_method_data[shared_payment_granted_token]` | Seller 收款 |
| `POST /v1/shared_payment/granted_tokens/{id}/resolve` | 第三方处理器场景，取出网络 token（dpan/agentic_token/fpan） |

### 能演示什么
- Agent 持有用户卡（通过 Payment Element 收集），向商户发行一次性限额 SPT
- 商户用 SPT 创建 PaymentIntent 完成收款
- 可在同一个 Stripe 测试账号中同时扮演 Agent 方（issued_token）和 Seller 方（granted_token）
- 整个流程**无需真实资金**，仅需 sandbox API key

### 限制
- SG/HK 主体不能直接开户使用（地区限制，见上）
- Preview 阶段，接口和行为可能变化
- 最低金额：SPT 卡支付 0.50 USD/次
- 3DS 认证场景需额外处理 `requires_action` 状态

---

## 2. Stripe Machine Payments (MPP)

**来源**: https://docs.stripe.com/payments/machine/mpp  
https://docs.stripe.com/payments/machine  
https://mpp.dev

### 测试模式说明
**完整支持**。关键配置：

1. 使用 `sk_test_...` 密钥；`livemode` 通过 key 前缀自动检测
2. 在 sandbox 账号创建 Stripe Profile，获取 `profile_test_xxx` 作为 `networkId`
3. Tempo 测试网地址：在 sandbox 账号调用：
   ```
   POST /v1/crypto/deposit_addresses
   -d network=tempo
   Header: Stripe-Version: 2026-07-29.preview
   ```
   返回的 address 即 Tempo **testnet**（因为 API key 是 test 模式，`mppx` 的 `stripe.create` 自动切换到 Tempo testnet）
4. 测试工具：
   - `npx mppx@latest validate http://localhost:4242`（自动跑完整 roundtrip）
   - SPT 测试：`npx @stripe/link-cli mpp pay http://localhost:4242/paid`
   - Stablecoin 测试：`tempo wallet fund && tempo request ...`（Tempo testnet CLI）

### 卡轨（SPT）在 MPP 中是否可走 WWW-Authenticate 流程
**是**。MPP 默认就支持 SPT（卡支付）和稳定币两种支付方式：
- `WWW-Authenticate: Payment` challenge 中会包含 SPT 和 Tempo 两种 method
- Agent 选择 SPT 方式，在 `Authorization: Payment` 中附上 SPT credential
- 这就是标准的 MPP HTTP 402 → retry 流程

### SDK 可用性
| SDK | 语言 | 说明 |
|-----|------|------|
| `mppx` | TypeScript | 官方，服务端 + 客户端，中间件（Express/Hono/Next.js/Elysia） |
| Python SDK | Python | FastAPI 集成 |
| `mpp-rs` | Rust | Axum 集成，Tempo Labs 维护 |
| `mpp-go` | Go | Tempo Labs 维护 |
| `mpp-rb` | Ruby | Rack-based，Stripe 维护 |
| `mppx` client | TypeScript | `Fetch.polyfill`, `McpClient.wrap` 可集成 MCP |

### Tempo Testnet 情况
- Tempo 是 Stripe 和 Paradigm 孵化的 Layer 1 区块链，专为稳定币支付设计
- Tempo 主网使用 USDC.e（TIP-20 标准）
- **当 Stripe API key 为 test 模式时，`mppx` 自动切换到 Tempo testnet**（无需手动配置）
- Tempo testnet 地址通过 `POST /v1/crypto/deposit_addresses` 在 sandbox 账号创建
- Tempo CLI 文档：https://tempo.xyz/developers/docs/cli

### 地区限制
- **SPT（卡）**：同 SPT 章节，US/CA/EU，不含 SG/HK
- **稳定币**：美国全州（纽约州除外）。美国以外需发邮件至 machine-payments@stripe.com 申请（30+ 国家可申请）

### 能演示什么
- HTTP 402 Challenge-Credential 完整流程
- 同一 endpoint 同时接受 SPT（卡）和 Tempo（USDC.e）
- 使用 `mppx validate` 自动验证全流程

---

## 3. Stripe Issuing for Agents

**来源**: https://docs.stripe.com/issuing/agents  
https://docs.stripe.com/issuing/testing?testing-method=with-code  
https://docs.stripe.com/issuing/global

### 测试模式下虚拟卡能力
**完整支持**，无需真实资金：

1. **创建 Cardholder**：`POST /v1/issuing/cardholders`（sandbox key）
2. **发行虚拟卡**：`POST /v1/issuing/cards -d type=virtual`
3. **模拟授权（test helper）**：
   ```
   POST /v1/test_helpers/issuing/authorizations
   -d card=ic_xxx
   -d amount=1000
   -d authorization_method=chip
   -d "merchant_data[category]=taxicabs_limousines"
   ```
4. **模拟 Capture**：`POST /v1/test_helpers/issuing/authorizations/{id}/capture`
5. 触发 `issuing_authorization.request` webhook，可测试实时授权逻辑
6. 需先在 sandbox 充入测试资金：US 账号用 Top-ups API；UK/EU 用 Funding Instructions API

### 支出控制（Spending Controls）测试
全部在 sandbox 可用：
- 商户类别控制（MCC）：`spending_controls[allowed_categories][]`
- 金额限制：`spending_controls[spending_limits]`
- 每次/每月限额
- 卡冻结：`status=inactive`

### 与 SPT/MPP 的连接
- Issuing 虚拟卡 → 创建 SPT → 用于 MPP 或 UCP 流程
- `docs.stripe.com/issuing/agents/programmatic-checkout` 提供完整链路指引

### 新加坡 / 香港可用性
**Stripe Issuing 本地发卡支持的 22 个国家**（官方列表）：  
AT, BE, BG, CA, CY, DE, EE, ES, FI, FR, GB, GR, HR, IE, IT, LT, LU, LV, MT, NL, PT, SI, SK  
**新加坡（SG）和香港（HK）均不在列表中。**

- 商业发卡（Commercial Issuing）：US, UK, EEA
- 稳定币支持卡（Stablecoin-backed cards）：LatAm/Caribbean/Africa 30+ 国，**不含 APAC**，且处于 **private preview**（需联系 Stripe 销售）
- 跨境发卡（Multinational Program）：Visa/Mastercard 跨境项目，面向美国跨国公司子公司，不适用于 SG/HK 独立注册主体

### 申请门槛
- 自身用途（Cards for your own business）：在 Dashboard 自助申请 → https://dashboard.stripe.com/issuing/overview
- 平台用途（Cards for your platform）：需联系 Stripe → https://stripe.com/contact/embedded-finance
- 审批时间：通常数日（地区合规审查）

---

## 4. Stripe 对 x402 的支持（Base 上的 USDC）

**来源**: https://docs.stripe.com/payments/machine/x402  
https://docs.stripe.com/payments/machine

### Stripe x402 集成方式
- x402 mainnet：Base 网络（`eip155:8453`），USDC，通过 Coinbase CDP facilitator 结算
- Stripe 接收 Base USDC 并记录为 PaymentIntent（`transaction_verification` 模式）
- USDC 合约地址（Base mainnet）：`0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`

### 测试网（Base Sepolia）支持
**完整支持**，官方文档明确说明：

1. Stripe sandbox 创建 Base 存款地址：
   ```
   POST /v1/crypto/deposit_addresses -d network=base
   -u sk_test_...
   Header: Stripe-Version: 2026-07-29.preview
   ```
2. 使用公共测试 facilitator（无需 CDP 账号）：`https://x402.org/facilitator`
3. 网络：`eip155:84532`（Base Sepolia）
4. USDC 测试币：Circle faucet → https://faucet.circle.com/
5. 测试客户端：`purl http://localhost:4242/paid`（Stripe 官方 CLI）

### 关键代码片段
```typescript
// facilitator for test
const facilitatorClient = new HTTPFacilitatorClient({ url: "https://x402.org/facilitator" });

// Base Sepolia
const resourceServer = new x402ResourceServer(facilitatorClient).register(
  "eip155:84532", new ExactEvmScheme()
);
```

### 地区限制
- 美国全州（纽约州除外）
- 美国以外：邮件申请 machine-payments@stripe.com（30+ 国可开通）
- **无需 waitlist**，标准流程申请开通 "Stablecoins and Crypto" 支付方式即可

### 能演示什么
- HTTP 402 → x402 payment → Base Sepolia → Stripe PaymentIntent 全流程
- 与 AgentCore Payments `CRYPTO_X402` 类型完全对接（Base Sepolia `eip155:84532` 已确认）

---

## 5. AgentCore Payments 与 MPP 的关系

**来源**: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments-concepts.html  
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments-process-payment.html  
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments-how-it-works.html

### AgentCore Payments 架构
- 顶层资源：**PaymentManager**（管理认证和 IAM）
- 连接器：**PaymentConnector**，支持两种类型：
  - **CoinbaseCDP**：Coinbase Developer Platform，crypto 钱包
  - **StripePrivy**：Stripe + Privy 嵌入式钱包基础设施
- 支付工具：**PaymentInstrument**（`EMBEDDED_CRYPTO_WALLET` 类型）
- 支付会话：**PaymentSession**（含支出限额 `maxSpendAmount`）

### MPP 是否走 Tempo
**是**，MPP 支持 `tempo` method（Tempo 链上任意 chainId，使用 USDC-equivalent token）。  
AgentCore Payments 对 MPP 的 method 支持：

| Method | Coinbase CDP | Stripe (Privy) |
|--------|-------------|----------------|
| `evm`（USDC） | 支持 | 支持 |
| `tempo`（USDC.e） | 支持 | 支持 |
| `solana`（mainnet/devnet） | 不支持 | 支持 |

**MPP 不走卡（card）**。AgentCore Payments 的 PaymentInstrument 仅为 `EMBEDDED_CRYPTO_WALLET`，无法直接处理卡/SPT。  
卡（SPT）是 Stripe 侧 MPP 的付款方式，不是 AgentCore Payments 支持的付款工具。

### 两种支付类型
`ProcessPayment` API 的 `paymentType` 参数：
- `CRYPTO_X402`：x402 协议，`X-PAYMENT` header 携带签名证明
- `MPP`：转发 `WWW-Authenticate: Payment` challenge，返回 `Authorization: Payment <token>`

### Testnet 配置
- **x402 (Base Sepolia)**：`network: "eip155:84532"`，Coinbase CDP 或 Stripe Privy 均可
- **MPP via Solana devnet**：Stripe (Privy) 连接器 + `solana` method + devnet
- **MPP via Tempo testnet**：使用 Stripe (Privy) 或 Coinbase CDP，`tempo` method

### SDK 支持
```python
pip install 'bedrock-agentcore[strands-agents]'   # Strands Agents
pip install 'bedrock-agentcore[langgraph]'          # LangGraph
```
- AgentCore payments plugin 自动拦截 HTTP 402 响应，调用 ProcessPayment，重试原始请求

### 关键结论
- AgentCore Payments 官方文档确认支持 MPP，通过 `WWW-Authenticate: Payment` / `Authorization: Payment` 头部流转
- **MPP 在 AgentCore 层面是纯稳定币轨道**（EVM/Tempo/Solana USDC），不含卡
- Tempo testnet 通过配置 Stripe (Privy) 或 Coinbase CDP 连接器，在 test mode 下自动切换

---

## 6. x402 / MPP 生态的法币或卡 Scheme

**来源**: https://paymentauth.org（MPP spec index）  
https://github.com/coinbase/x402  
https://mpp.dev

### 现状
| 方案 | 状态 | 说明 |
|------|------|------|
| `draft-card-charge-00` (paymentauth.org) | **草案，未实现** | MPP Card method 规范草案，仅存在于 paymentauth.org spec index，无实现 |
| `draft-stripe-charge-00` (paymentauth.org) | **草案，未实现** | MPP Stripe method 规范草案（这里的 Stripe 指 SPT 流），对应 mpp.dev 的 `stripe.spt` |
| x402 card/fiat scheme | **无** | x402 协议官方表示"可能扩展到 fiat 网络，但不会以牺牲链上支付为代价"，目前无 card scheme |
| MPP `stripe` method（SPT）| **可用**（通过 Stripe 服务端集成） | mpp.dev 和 docs.stripe.com/payments/machine/mpp 均已支持，但需要 Stripe 账号，且 SPT 地区限制同上 |

### 结论
- **在沙箱中能跑通的卡轨**：只有 Stripe MPP 的 SPT 路径（Issuing 卡 → SPT → MPP merchant），且仅限 US/CA/EU 主体
- x402 本身没有、也没有计划短期内支持卡/法币 scheme
- paymentauth.org 的 Card draft 处于早期规范阶段，无可用实现
- 「未验证」：paymentauth.org 的 `draft-card-charge-00` 具体内容（页面仅显示标题，未展示草案正文）

---

## 对 Demo 多轨设计的建议

### 卡轨：最稳路径

**推荐路径**：Stripe Issuing → SPT → Stripe MPP（HTTP 402 challenge-credential flow）

**原因**：
1. 测试模式完整覆盖：Issuing test helpers + SPT test helpers + mppx validate，三层均可全自动跑通
2. 无需链上钱包、无 gas 费用问题，纯 API 调用
3. 与 AgentCore Payments 的稳定币轨（x402/MPP）并列演示，形成双轨对比
4. Stripe 官方文档明确支持，SDK 成熟（mppx 0.9.x）

**需要申请的内容**：
| 步骤 | 方式 | 预计时间 |
|------|------|---------|
| 注册 Stripe 账号（US 主体或通过 Connect 平台） | 自助 | 即时 |
| 申请 Stripe Issuing | Dashboard 自助申请 | 1–3 个工作日（合规审查） |
| 同意 SPT preview 条款 | 自助，Dashboard ToS | 即时 |
| 申请 Stablecoins & Crypto 支付方式（x402 轨） | Dashboard 申请 → Stripe 审核 | 1–5 个工作日 |

**地区绕过方案**（SG/HK 主体）：
- 通过美国 Stripe Connect 平台账号（如已有合作伙伴），以 connected account 方式使用 Issuing
- 或用 US 实体注册专门用于演示的 Stripe 账号
- 稳定币轨（x402/MPP Tempo）：邮件联系 machine-payments@stripe.com，说明账号 ID，30+ 国可开通

**预计从零到跑通演示总时间**：3–7 个工作日（主要等待 Issuing 审批和 Stablecoins 审批）

### 稳定币轨（已确定）
- Base Sepolia（x402）：公共 facilitator `https://x402.org/facilitator` + Circle faucet，**零审批，即时可用**
- Tempo testnet（MPP）：Stripe sandbox 自动切换，**即时可用**
