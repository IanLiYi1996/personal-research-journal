# Spike：AgentCore Browser 签名 × AWS WAF Web Bot Auth（2026-10-20）

> 目的：验证 demo 第一幕（可信 agent 接入）的头号风险 R1：AgentCore Browser 的 Web Bot Auth 签名在 ap-southeast-1 是否可用，AWS WAF 能否把它判为 `verified`，并能拦截无签名 / 伪造签名的请求。
> 结论：**✅ 全部验证通过**。

## 环境（账号 <AWS_ACCOUNT_ID>，均打标签 `project=agentic-payment-demo-spike`）
| 资源 | 标识 | 说明 |
|---|---|---|
| IAM 角色 | `apdemo-spike-browser-role` | 仅信任 bedrock-agentcore.amazonaws.com（SourceAccount/SourceArn 条件），无附加权限 |
| AgentCore Browser | `apdemo_spike_signed_browser-RVbuoucKaw`（ap-southeast-1） | `browserSigning.enabled = true` |
| S3 源站 | `apdemo-spike-origin-<AWS_ACCOUNT_ID>`（ap-southeast-1） | Block Public Access 全开；桶策略仅允许指定 CloudFront 分发经 OAC 读取 |
| CloudFront | `E1NDNYJBSIIHZI` / `d1sn29qdua50gp.cloudfront.net` | OAC 回源；绑定下方 Web ACL |
| WAF Web ACL | `apdemo-spike-acl`（CLOUDFRONT 作用域，us-east-1） | Bot Control（Version_7.0，COMMON，Count 覆盖）+ 3 条自定义规则；日志 → CloudWatch Logs `aws-waf-logs-apdemo-spike` |

自定义规则：① 标签 `web_bot_auth:invalid` 或 `web_bot_auth:failed` → Block；② `web_bot_auth:verified` → Allow；③ `bot:category:http_library` → Block。

## 结果
| # | 请求 | HTTP | WAF 动作 / 规则 | 关键标签 |
|---|---|---|---|---|
| 1 | AgentCore Browser 签名头回显（httpbin.org/headers） | — | — | `Signature`、`Signature-Input`（ed25519，`tag="web-bot-auth"`，含 created / expires / nonce / keyid）、`Signature-Agent: "https://slf696jfe4gp0.keydirectory.signer.ap-southeast-1.on.aws"`；UA 含 `Amazon-Bedrock-AgentCore-Browser/1.0` |
| 2 | 普通 curl（无签名） | **403** | BLOCK `block-http-library` | `bot:category:http_library`、`bot:name:curl`、`bot:unverified` |
| 3 | 伪造签名（复用真实 Signature-Agent / keyid，签名值伪造） | **403** | BLOCK `block-invalid-wba` | **`bot:web_bot_auth:failed`** |
| 4 | AgentCore Browser（签名开启）访问 `/` 与 `/api/quote` | **200** | ALLOW `allow-verified-wba` | `bot:web_bot_auth:verified`、`bot:vendor:agentcore`、`bot:name:aws_agentcore_ap_southeast_1`、`bot:category:ai`、`bot:intent:ai_assistant`、`bot:user_triggered:verified`、`bot:organization:amazon` |

## 发现与坑
1. **伪造签名的标签是 `web_bot_auth:failed`**，而 WAF 文档列出的是 `invalid / expired / unknown_bot`。规则需同时匹配 `failed` 与 `invalid`，否则伪造签名会被默认放行（本次首轮即如此）。
2. 本次**未出现**文档所述的 `bot:account:<hash>` 标签（按 AWS 账号区分 agent）。是否需要额外配置或仅特定条件下出现，待查；若要演示"按账号白名单"，需先确认。
3. Bot Control 以 Count 模式观察标签、以自定义规则决定放行 / 拦截，便于在控制台展示"每条请求被打了什么标签、命中哪条规则"。
4. WAF 日志约 30–60 秒可在 CloudWatch Logs 查到，现场演示可接受；也可用 Sampled requests。

## 对 demo 的影响
- 第一幕 1-1、1-2、1-3 **可现场真实演示**；1-4（Monetize 402）待下一步验证。
- 设计文档 R1 风险可降级；需更新规则写法（`failed`）与"账号标签"的表述。

## 补充：第一幕 1-4 —— WAF Monetize（测试模式）返回 HTTP 402 ✅
- 配置：Web ACL `MonetizationConfig = {CurrencyMode: TEST, CryptoConfig.PaymentNetworks: [{Chain: BASE_SEPOLIA, WalletAddress: 0x12F5…08ef, Prices: [{Amount: "0.01", Currency: USDC}]}]}`；新增规则 `monetize-api`（`web_bot_auth:verified` **且** URI 以 `/api/` 开头 → `Monetize`），优先级在"放行已验证 agent"之前。
- 卖方收款钱包：新生成的 **Base Sepolia 测试网**地址 `0x12F5B616d36D8E8BC91CB3fcEC0240b3435008ef`，私钥存于 Secrets Manager `apdemo/seller-wallet-base-sepolia`（仅测试网）。
- 结果（AgentCore Browser 签名请求）：
  - `/` → **200**（未收费路径照常放行）
  - `/api/quote` → **402**，响应头 `PAYMENT-REQUIRED`（base64 JSON）解码为：
    - `x402Version: 2`，`error: "PAYMENT-SIGNATURE header is required"`
    - `accepts[0]`: `scheme: exact`，`network: eip155:84532`（Base Sepolia），`amount: "10000"`（= 0.01 USDC，6 位小数），`payTo: 0x12F5…08ef`，`asset: 0x036CbD53842c5426634e7929541eC2318f3dCF7e`（Base Sepolia USDC），`maxTimeoutSeconds: 300`，`extra: {name: USDC, version: "2"}`
  - 响应体还包含给"人类用户"的说明与 AWS 官方 agent skill 的安装指引。
- 结论：WAF Monetize 使用 **x402 v2** 线上格式；AgentCore Payments 文档声明支持 v1 与 v2，R4（版本不匹配）风险预期较低，待买方实付验证。

## 下一步
- 买方实付：创建 AgentCore Payments `PaymentManager` + Coinbase CDP 连接器（Quick create 需用户在约 10 分钟内完成 OAuth 授权）→ 创建钱包 → 用户从 Circle 水龙头领取测试 USDC 并授予 agent 签名权限 → agent 支付 `/api/quote` → 核对链上交易与卖方钱包到账。

## 补充：买方真实付款（测试网）

### 方案 3：x402 官方客户端直接付款 ✅（端到端走通）
- 买方：测试钱包 `0xc299732a015f789de6d643dCa69a964911290D22`（私钥在 Secrets Manager `apdemo/buyer-wallet-base-sepolia`），Circle 水龙头领取 20 测试 USDC。
- 流程：AgentCore Browser（签名）请求 `/api/quote` → **402** → 用 `x402[evm]` 2.25.0（`x402ClientSync` + `ExactEvmScheme` + `EthAccountSigner`）生成 EIP-3009 授权签名 → 以 `PAYMENT-SIGNATURE` 头重试 → **200**，响应头 `PAYMENT-RESPONSE`：`success=True`，`network=eip155:84532`，`transaction=0xc7661efcca388a45192823d787c612ce7ac7438d1c286cca19b4cbd95323d013`。
- 链上核对（Base Sepolia RPC）：交易 `status=0x1`，区块 48359022，提交方（facilitator）`0x64cc…1f0a`；买方余额 20.00 → **19.99**，卖方 `0x12F5…08ef` 0 → **0.01** USDC。
- 结论：**"可信 agent → WAF 验签 → 402 → 签名付款 → 链上结算 → 返回数据"整条链路在 AWS 上真实跑通**；买方 gas 由 facilitator 承担。

### 方案 2：AgentCore Payments + Stripe Privy ⚠️（差最后一步：用户授权）
- 已完成：Payment Manager `apdemospikepm-<redacted>`（AWS_IAM）；Privy 凭证提供方 `apdemo-privy`（authorization key 由脚本生成 P-256 密钥对，经 Privy `POST /v1/key_quorums` 注册，key quorum `<privy-signer-id>`）；StripePrivy 连接器 `privy-iknkqbdeep`（READY，**不需要** Marketplace 订阅）；钱包工具 `payment-instrument-Wgzv6kpll0npud6`（ACTIVE，地址 `0x03ff…64A5`，绑定用户邮箱，已领 20 测试 USDC）；PaymentSession（1.00 USD / 30 分钟）创建成功。
- 卡点：`ProcessPayment` 返回 `AccessDeniedException: Privy credentials are invalid`。Privy API 显示该钱包 `owner_id` 为用户、`additional_signers = []` —— **用户尚未把 agent 的 key quorum 授权为签名方**。需用户经 Privy 前端（privy-io/aws-agentcore-sdk）登录并授权后重试。
- 坑：① 服务角色的"基础权限"文档称会自动附加，实测需手动授予（含 workload identity 的 `TagResource`）；② 每个连接器还需 `GetResourcePaymentToken`（凭证提供方）+ `secretsmanager:GetSecretValue`（托管密钥）权限；③ Coinbase CDP 连接器需 AWS Marketplace 订阅，本账号订阅失败。

### 方案 2 更新：AgentCore Payments + Stripe Privy ✅（端到端走通）
- 用户经 Privy 授权前端（privy-io/aws-agentcore-sdk，仅监听 127.0.0.1，经 SSH 端口转发访问）登录并授权后，Privy API 显示钱包 `additional_signers = [{signer_id: <privy-signer-id>}]`。
- 流程：AgentCore Browser（签名）请求 `/api/quote` → 402 → `CreatePaymentSession`（上限 1.00 USD / 30 分钟）→ `ProcessPayment`（`paymentType: CRYPTO_X402`，`version: "2"`，payload = 402 中的 `accepts[0]`）→ `status: PROOF_GENERATED`，`paymentOutput.cryptoX402.payload` 含 EIP-3009 `authorization` 与 `signature`（Privy 托管钱包签名，AWS 不持有私钥）→ 组装完整 x402 v2 PaymentPayload `{x402Version: 2, resource, accepted: accepts[0], payload}`，base64 后放入 `PAYMENT-SIGNATURE` 头重试 → **200** + `PAYMENT-RESPONSE`。
- 链上：交易 `0x7831ab5f32d4ac99ec444f31f0cd21cdfe628ff78b087b15c961914b8677f186`（区块 48359681）；agent 钱包 `0x03ff…64A5` 20.00 → **19.99**，卖方 0.01 → **0.02** USDC。
- **预算边界**：上限 0.005 USD 的会话再付 0.01 → `ValidationException: Insufficient budget for session … Pending amount: 0.005 USD, Transaction amount: 0.010000 USD` —— 基础设施层硬拒绝，不签名、不上链 ✅。
- 坑：`ProcessPayment` 只返回 x402 的 `payload` 部分，调用方需自行拼成完整 PaymentPayload 并编码（官方 SDK 的 payment tool 应已封装，demo 建议直接用 SDK）；`expiryTimeInMinutes` 最小 15。
- 授权前端已停止，本地 `.env.local` 已删除。
