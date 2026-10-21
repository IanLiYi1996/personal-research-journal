# 卡组织相关内容核实报告

**核实日期**：2026-10-19  
**核实者**：AI 研究 agent  
**目标文档**：2026-10-19-agentic-payment-protocols-survey.md  
**核实范围**：B.4（Visa TAP）、B.5（Mastercard Agent Pay）、C.2 对比表、C.3 责任划分、C.6 Visa/Mastercard 部分、参考文献 41/42

---

## 1. Visa Trusted Agent Protocol（TAP）

### 1.1 核实结论

| 条目 | 状态 | 一手来源 |
|------|------|----------|
| GitHub 仓库存在 | ✅ 已验证 | https://github.com/visa/trusted-agent-protocol |
| 仓库创建日期 | ✅ **2025-10-10** | GitHub API: `created_at: 2025-10-10T02:05:59Z` |
| 最近更新 | ✅ 2026-10-14 | GitHub API: `updated_at: 2026-10-14T00:53:18Z` |
| 许可证 | ✅ **非开源 OSS**，Visa Developer Center Terms of Use | LICENSE.md: 指向 developer.visa.com/terms；代码文件本身使用 MIT 风格文本但受 VDC Terms 约束 |
| 签名算法 | ✅ Ed25519（推荐）和 RSA-PSS-SHA256 | tap-agent/README.md；agent_app.py |
| 签名组件 | ✅ `@authority`、`@path`、`created`、`expires`、`nonce`、`keyId`、`tag` | tap-agent/agent_app.py（代码直接证实） |
| Tag：`agent-browser-auth` | ✅ 用于 Product Details（浏览）阶段 | agent_app.py: `tag = "agent-browser-auth"` for "Product Details" |
| Tag：`agent-payer-auth` | ✅ 用于 Checkout（支付）阶段 | agent_app.py: `tag = "agent-payer-auth"` for "Complete Checkout" |
| 密钥目录（Agent Registry） | ✅ 通过 `/keys/{key_id}` endpoint 检索公钥 | agent-registry/README.md；cdn-proxy 调用 `AGENT_REGISTRY_URL` |
| 消费者识别与 PAR | ✅ 官方文档描述 | developer.visa.com: "securely receive key information needed for checkout via query parameters...verifiable consumer identifiers, Payment Account References (PARs) for cards on file, or other identifiers like loyalty numbers, emails and phone numbers" |
| 支付容器对象 | ⚠️ **未在 GitHub 参考实现中直接体现**；官方文档提及但细节在 developer.visa.com 登录区域（需申请访问权限）| developer.visa.com 登录墙："No access — request access to view" |
| 与 IETF Web Bot Auth 关系 | ✅ TAP 以 Web Bot Auth 为基础层 | Cloudflare 博客（2025-10-24）: "Both Trusted Agent Protocol and Agent Pay leverage Web Bot Auth as the agent authentication layer" |
| Cloudflare 合作 | ✅ Cloudflare 参与 TAP 和 Agent Pay 的验证服务 | blog.cloudflare.com/secure-agentic-commerce（2025-10-24） |
| 当前状态 | ✅ 开发者预览/beta；Visa Intelligent Commerce 页面注: "Product is currently in the process of deployment" | visa.com/en-us/solutions/intelligent-commerce |
| "Signature-Agent" 头部（IETF Web Bot Auth 风格） | ⚠️ **TAP 参考实现不使用 `Signature-Agent` 头**；使用的是 `Signature-Input` 中的 `keyId` + Agent Registry 查询；与 IETF draft 所述 `Signature-Agent` Structured Field 不同 | merchant-backend/app/security/signature_verification.py |

**关键发现**：综述 B.4 节引用的"Signature-Agent 头部（Structured Fields Dictionary，含 `type` 参数 `directory`/`jwks_uri`/`cimd`）"是 IETF Web Bot Auth 草案的机制，但 Visa TAP 参考实现（github.com/visa/trusted-agent-protocol）使用的是自己的 Agent Registry 服务，通过 `keyId` 查找公钥，不完全对应 IETF 草案中的 `Signature-Agent` 头。两者关系是：TAP 以 Web Bot Auth 为设计灵感和基础协议（RFC 9421 HTTP Message Signatures），但在密钥发现上采用了中心化的 Agent Registry 方案。

### 1.2 建议替换文本

**替换 B.4 节"TAP 架构（未验证具体技术细节）"部分（原文约 398–401 行，原文开头"TAP 在 Web Bot Auth 基础上"）**：

---

#### Visa TAP 架构（已从 GitHub 参考实现核实）

TAP 仓库（`github.com/visa/trusted-agent-protocol`，创建于 2025-10-10，截至 2026-10-14 仍在更新，213 ★）包含完整参考实现，许可证为 Visa Developer Center Terms of Use（非标准 OSS 许可，需同意 developer.visa.com/terms）。

**已核实的技术细节**：

- **签名算法**：Ed25519（推荐）和 RSA-PSS-SHA256，均符合 RFC 9421
- **签名组件**：`@authority`、`@path`、`created`、`expires`、`nonce`、`keyId`、`tag`
- **动态 Tag**：
  - `agent-browser-auth`：agent 处于浏览/产品详情阶段
  - `agent-payer-auth`：agent 处于结账/支付阶段
- **密钥发现**：通过 Agent Registry 服务（`GET /keys/{key_id}`），而非 IETF Web Bot Auth 草案所述的 `Signature-Agent` Structured Fields 头部；Visa 和 Mastercard 分别托管各自的密钥目录
- **消费者识别**：agent 通过 query parameters 向商家传递消费者信息，包括 Payment Account References（PARs，卡档案引用）、loyalty numbers、email、phone 等（需消费者同意）
- **nonce 防重放**：签名必须包含 `nonce` 字段，由 Cloudflare 等中间层检验唯一性

**与 IETF Web Bot Auth 的关系**：TAP 以 Web Bot Auth 为基础协议层（HTTP Message Signatures + 公钥加密），Cloudflare 参与验证服务；但 TAP 的密钥发现机制采用中心化 Agent Registry（而非 IETF 草案的 `Signature-Agent` Structured Fields + `/.well-known/` 端点）。

**Cloudflare 合作**（来源：blog.cloudflare.com/secure-agentic-commerce，2025-10-24）：Cloudflare 作为网络层验证者，负责：①确认签名头部存在；②从目录拉取并缓存公钥；③检验 `created`/`expires` 时间窗口；④检验 nonce 唯一性；⑤校验 tag 类型；⑥重建 signature base；⑦执行 Ed25519 验签。

**当前状态**：预览阶段（developer.visa.com 需申请访问权限；visa.com 页面注"Product is currently in the process of deployment"）。

---

## 2. Visa Intelligent Commerce

### 2.1 核实结论

| 条目 | 状态 | 来源 |
|------|------|------|
| 官方发布日期 | ✅ **2025-04-30**（"Visa Global Product Drop"活动） | investor.visa.com press release: "APRIL 30, 2025" |
| 发布标题 | ✅ "Find and Buy with AI: Visa Unveils New Era of Commerce" | investor.visa.com |
| 合作伙伴（发布时） | ✅ Anthropic、IBM、Microsoft、Mistral AI、OpenAI、Perplexity、Samsung、Stripe | investor.visa.com |
| AI-Ready Cards（代币化凭证） | ✅ 正式组件 | 官方新闻稿: "Replaces card details with tokenized digital credentials, enhancing security" |
| AI-Powered Personalization | ✅ 正式组件 | 官方新闻稿 |
| Simple and Secure AI Payments | ✅ 正式组件（限额、条件、实时信号） | 官方新闻稿 |
| Visa Accept | ⚠️ **官方材料中未见该名称** | 官方 visa.com/en-us/solutions/intelligent-commerce 中的组件名为"Intelligent Commerce Connect"，非"Visa Accept" |
| Visa Intelligent Commerce MCP Server | ✅ **已确认** | visa.com Intelligent Commerce 页面: "Visa introduces the Model Context Protocol (MCP) Server, a secure integration layer for AI agents and large language models to connect directly to Visa Intelligent Commerce APIs" |
| Intelligent Commerce Connect | ✅ 正式组件 | "single integration for secure, permissioned transactions across payment schemes, token providers and agent ecosystems" |
| Visa CLI（Command Line Interface） | ✅ 新组件 | "Give your AI agents real payment capability as you code. With Visa CLI, complete card-based transactions programmatically" |
| developer.visa.com 文档 | ✅ 存在但需申请访问 | developer.visa.com/capabilities/visa-intelligent-commerce: "No access — request access to view" |
| 当前状态 | ✅ **部署中（非 GA）** | visa.com 底部注脚: "Product is currently in the process of deployment and should be understood as a representation of the potential features of the fully-deployed product" |
| 稳定币结算 | ⚠️ **官方 Intelligent Commerce 发布材料中未提及** | 官方新闻稿和产品页无相关内容；稳定币仅在 Agent Pay for Machines（Mastercard）中明确提及 |
| Visa Direct 在 agent 场景的正式表述 | ⚠️ **未在 Intelligent Commerce 官方材料中找到明确表述** | 官方页面中未见 agent + Visa Direct 的联合文档 |
| OpenAI 合作 | ✅ 已在 visa.com 主页突出展示 | "Visa and OpenAI Partner to Help Shape the Future of Agentic Commerce" |

### 2.2 建议替换文本

**替换 B.4 节背景段（原文约 373–375 行，原文开头"Visa 于 2025 年发布 Intelligent Commerce 平台"）**：

---

#### 背景

Visa 于 **2025-04-30**（"Visa Global Product Drop"活动）正式宣布 **Intelligent Commerce** 平台，标题"Find and Buy with AI: Visa Unveils New Era of Commerce"。[来源：investor.visa.com，2025-04-30]

初始合作伙伴包括：Anthropic、IBM、Microsoft、Mistral AI、OpenAI、Perplexity、Samsung、Stripe。

**Visa Intelligent Commerce 主要组件**（截至 2026-10，来源：visa.com/en-us/solutions/intelligent-commerce）：

1. **AI-Ready Cards**：以网络令牌取代卡片明细，提升安全性，确认特定 agent 有权代表消费者操作
2. **Intelligent Commerce Connect**：单一集成入口，跨支付 scheme、令牌提供方和 agent 生态系统
3. **Trusted Agent Protocol（TAP）**：商家识别和互动可信 agent 的框架
4. **MCP Server**：Visa 推出的 Model Context Protocol 服务器，作为 AI agent 和 LLM 连接 Visa Intelligent Commerce API 的集成层
5. **Command Line Interface（CLI）**：供开发者以编程方式在代码环境中完成卡支付
6. **Visa Commercial Solutions**：B2B 场景 agentic commerce 能力

**当前状态**：部署中（非 GA）。官方注："Product is currently in the process of deployment and should be understood as a representation of the potential features of the fully-deployed product."

**稳定币结算**：官方 Intelligent Commerce 材料中未提及稳定币；稳定币支持在 Mastercard Agent Pay for Machines（2026-06）中有明确表述。

注意：综述中提及的"Visa Accept"未出现在官方组件命名中，对应的正式组件名为"Intelligent Commerce Connect"。

---

**替换 C.6 中"Visa Direct 部分"（原文约 840–842 行，原文开头"Visa TAP 目前关注卡轨道"）**：

---

**Visa Intelligent Commerce（已核实）**

- Visa 于 2025-04-30 正式推出 Intelligent Commerce，组件包括 AI-Ready Cards（令牌化凭证）、Intelligent Commerce Connect、TAP、MCP Server 和 CLI（来源：investor.visa.com）。
- Visa 和 OpenAI 已宣布合作，在 visa.com 主页重点展示。
- Visa Direct（实时推送付款，覆盖 200+ 国家，支持 Faster Payments、RTP、FedNow 等）是 Visa 进入 A2A 实时支付的核心产品。
- Visa Direct + agent 场景的正式协议集成：**未在官方 Intelligent Commerce 文档中找到明确表述**，仍标注"未验证"。

---

## 3. Mastercard Agent Pay

### 3.1 核实结论

| 条目 | 状态 | 来源 |
|------|------|------|
| 发布日期 | ✅ **2025-04-29** | bankingnewsai.com（综合 mastercard.com 官方发布，URL 路径含 /2025/april/）；DuckDuckGo 搜索结果 URL |
| 发布标题 | ✅ "Mastercard unveils Agent Pay, pioneering agentic payments technology to power commerce" | DuckDuckGo 搜索结果 |
| 初始合作伙伴（发布时） | ✅ **Microsoft**（Azure OpenAI Service、Copilot Studio）、IBM、Braintree、Checkout.com | bankingnewsai.com timeline |
| Agentic Tokens 机制 | ✅ 已核实，专为 AI agent 签发的令牌，绑定消费者授权的条件和范围 | bankingnewsai.com: "Agentic Tokens and agent registration let verified AI agents pay within consumer limits" |
| Agent Pay Acceptance Framework | ✅ 已演进为 **Agent Pay Trust Framework**（2026-09-30 发布），五支柱：identity、intent、controls、trusted execution、intelligence | bankingnewsai.com（Sep 30, 2026 entry） |
| 与 Web Bot Auth / Cloudflare 合作 | ✅ Cloudflare 参与 TAP 和 Agent Pay 验证；2026-09-30 新增 privacy-preserving data partnership | blog.cloudflare.com/secure-agentic-commerce（2025-10-24）；bankingnewsai.com |
| 与 Microsoft 合作 | ✅ 发布时即有 Microsoft（Azure OpenAI Service、Copilot Studio）为合作方 | bankingnewsai.com |
| Verifiable Intent | ✅ **与 Google 共同开发，2026-03-05 开源**，"aligned with Google's AP2 and UCP protocols" | bankingnewsai.com |
| Verifiable Intent 捐赠 FIDO | ✅ **2026-04-28**，Google 将 AP2 捐赠给 FIDO Alliance，Mastercard 将 Verifiable Intent 贡献给 FIDO Alliance Payments Technical Working Group | fidoalliance.org（PYMNTS 引用，2026-05-04） |
| 所有美国持卡人是否已开通 | ✅ **"all U.S. Mastercard cardholders due by mid-November 2025"（节日季前覆盖）** | bankingnewsai.com: "all U.S. Mastercard cardholders due by mid-November 2025" |
| OpenAI ChatGPT Instant Checkout | ✅ 2025-09-30 可通过 ChatGPT Instant Checkout 使用；Citi 和 U.S. Bank 持卡人已开通 | bankingnewsai.com |
| 欧洲发行行 | ✅ 2026-06-02 全部欧洲发行行在网络层启用 Agent Pay | bankingnewsai.com |
| 开发者文档（developer.mastercard.com） | ⚠️ developer.mastercard.com/product/agent-pay/ 返回 404；文档可能存在于需登录区域 | WebFetch 测试 |
| Agent Pay for Machines | ✅ 2026-06-10 推出，支持 M2M 支付"across cards, accounts and stablecoins"，30+ 初始参与方含 Adyen、Coinbase、Stripe、Cloudflare | bankingnewsai.com |
| Multi-Token Network (MTN) 在 Agent Pay 中 | ✅ **已通过 Agent Pay for Machines（2026-06）实现**：稳定币支持明确列入；MTN 是底层基础设施 | bankingnewsai.com |
| 当前状态 | ✅ **Rolling out**（非 GA 但已上线，持续扩展） | bankingnewsai.com AI systems table |
| Skyfire 合作 | ✅ 2026-09-30 Agent Pay Trust Framework 发布时新增 Skyfire KYA 合作 | bankingnewsai.com |
| 美国已启用 issuer | ✅ 2026-01-29 Mastercard 在 Q4 2025 财报时说明"已启用美国 issuer 支持 agentic commerce" | bankingnewsai.com |

### 3.2 建议替换文本

**替换 B.5 节全文（原文约 414–430 行，原文开头"Mastercard 于 2025 年（具体日期未验证）"）**：

---

### B.5 Mastercard Agent Pay

#### 概览（已从多方一手来源核实）

Mastercard 于 **2025-04-29** 正式推出 **Agent Pay** 计划，标题"Mastercard unveils Agent Pay, pioneering agentic payments technology to power commerce"。[来源：mastercard.com/us/en/news-and-trends/press/2025/april/...]

**发布时核心组件**：

1. **Agentic Tokens（Agentic 令牌）**：专为 AI agent 签发的 Mastercard 令牌，绑定消费者设定的授权条件（金额、商家范围等），每笔交易绑定特定授权 agent。
2. **Agent 注册与验证**：要求 AI agent 注册并通过验证才能使用网络令牌发起交易。

**发布时合作伙伴**：Microsoft（Azure OpenAI Service、Copilot Studio）、IBM、Braintree、Checkout.com。

**主要里程碑（时间线）**：

| 日期 | 事件 |
|------|------|
| 2025-04-29 | Agent Pay 正式推出，Agentic Tokens + agent 注册 |
| 2025-09-30 | 通过 OpenAI ChatGPT Instant Checkout 可用；Citi、U.S. Bank 持卡人已开通；宣布节日季前（mid-Nov 2025）覆盖全美 Mastercard 持卡人 |
| 2026-01-29 | 宣布已启用美国 issuer 支持 agentic commerce |
| 2026-03-05 | **Verifiable Intent** 开源（与 Google 共同开发，与 AP2/UCP 协议对齐） |
| 2026-04-28 | Google 将 AP2 捐赠 FIDO Alliance；Mastercard 将 Verifiable Intent 贡献给 FIDO Payments TWG |
| 2026-06-02 | 全部欧洲发行行在网络层启用 Agent Pay |
| 2026-06-10 | **Agent Pay for Machines** 推出：M2M 支付，支持卡、账户、**稳定币**，30+ 参与方含 Adyen、Coinbase、Stripe、Cloudflare |
| 2026-09-30 | **Agent Pay Trust Framework**（五支柱）；新增 Cloudflare privacy-preserving data partnership；Skyfire KYA 合作 |

**Verifiable Intent 与 FIDO Alliance**：Mastercard 与 Google 共同开发 Verifiable Intent——为 agent 购买行为提供用户授权的密码学证明；与 AP2（Google）共同于 2026-04-28 捐赠/贡献给 FIDO Alliance Payments Technical Working Group 开展标准化。[来源：fidoalliance.org，2026-05-04]

**稳定币与 Multi-Token Network**：Agent Pay for Machines（2026-06-10）明确支持稳定币结算；Mastercard Multi-Token Network（MTN，2023 年推出）是底层多代币互通基础设施。

**全美持卡人覆盖**：已于 2025 年节日季前（mid-November 2025）实现，通过 ChatGPT Instant Checkout 渠道。

**当前状态**：Rolling out（正在扩展中），已有多家银行（Citi、U.S. Bank、Santander、ING 等）完成试点或正式部署。

---

**替换 C.6 中"Mastercard Agent Pay 部分"（原文约 844–848 行，原文开头"Mastercard Multi-Token Network (MTN)"）**：

---

**Mastercard Agent Pay（已核实）**

- Mastercard **Multi-Token Network (MTN)**（2023 年发布）是稳定币/CBDC 互通网络；Agent Pay for Machines（2026-06-10）已明确将稳定币列为结算轨道之一，30+ 初始参与方含 Coinbase、Adyen、Stripe。
- Mastercard 与 AP2 的关系：Mastercard 独立开发 Verifiable Intent，与 Google AP2 协议对齐，两者于 2026-04-28 同时捐赠/贡献给 FIDO Alliance（而非 Mastercard 参与 AP2 的捐赠）。Verifiable Intent 提供密码学证明，与 AP2 Mandate 互补。
- Agent Pay 的稳定币支持：**已通过 Agent Pay for Machines（2026-06-10）验证**，官方确认支持"cards, accounts and stablecoins"。
- 当前状态：Rolling out（多个发行行已上线，持续扩展）。

---

## 4. 卡组织规则：Agent 场景责任/Chargeback 更新

### 4.1 核实结论

| 条目 | 状态 | 来源 |
|------|------|------|
| 是否有 agent 专属争议/chargeback 规则 | ✅ **截至 2026-08，无任何卡组织已发布 agent 专属 binding chargeback 规则** | BeastInsights 分析文章（"As of August 2026, no card network has published a binding chargeback rule for disputes on agent-initiated transactions"）|
| Visa Core Rules 更新 | ✅ **Visa 已更新 Core Rules 正式承认（recognize）agentic transactions**，但该承认以遵守 Visa Intelligent Commerce 规范并使用其令牌为条件 | BeastInsights: "Visa updated its Core Rules to formally recognize agentic transactions. The rules condition that recognition on identity verification under Visa Intelligent Commerce specs and use of the provisioned token." |
| Mastercard Transaction Processing Rules | ✅ **Mastercard 的公开 Transaction Processing Rules 中仍无 agentic 术语**；其框架在产品程序（Agent Pay）中，而非约束商家的运营规则中 | BeastInsights: "Mastercard's public Transaction Processing Rules, by contrast, still contain no agentic terminology. Its framework lives in product programs like Agent Pay rather than in the operating rules that bind merchants." |
| American Express | ✅ **Amex 于 2026-04 宣布 Agent Purchase Protection**：已注册 agent + 验证购买意图场景下，Amex 吸收 agent 错误 | BeastInsights |
| 新交易标识 | ⚠️ Visa 在 Intelligent Commerce 规范中有 agent identity verification 要求，但具体的新交易标识代码未公开发布 | 间接信息 |
| 责任归属（争议场景） | ✅ 商家默认承担责任；Mastercard agentic token 模式下发卡行承担 fraud 责任（令牌合规发行且交易符合消费者策略） | BeastInsights: "Under Mastercard's agentic token model, the issuer carries fraud liability when the token is validly issued and the transaction honors the consumer's policy at authorization" |
| 现有 reason code 适用性 | ✅ agent 争议仍被纳入现有 CNP reason codes（如 Visa 10.4 unauthorized fraud、13.2、13.5），无独立 agentic dispute 类别 | BeastInsights |

### 4.2 建议替换文本

**替换 C.3 "当前状态"最后一条（原文约 641 行，原文开头"现行卡组织规则（Visa Core Rules、Mastercard Rules）"）**：

---

- **Visa**：已更新 Core Rules 正式承认（formally recognize）agentic transactions，但该承认以使用 Visa Intelligent Commerce 规范下的身份验证和令牌为条件；尚未发布独立的 agent 争议 reason code 体系。[来源：BeastInsights 分析，2026-08]
- **Mastercard**：公开的 Transaction Processing Rules 截至 2026-08 仍未包含 agentic 术语；Agent Pay 框架存在于产品程序层而非约束商家的运营规则层。在 Agentic Token 模型下，令牌合规发行且交易符合消费者授权策略时，发卡行承担 fraud 责任。
- **American Express**：2026-04 宣布 Agent Purchase Protection，对已注册 agent 的验证交易，Amex 吸收 agent 错误损失，而非推给商家（条件性豁免，非全面新制度）。
- **整体状态**：截至 2026-08，无任何卡组织发布针对 agent 发起交易争议的专属 binding rule；agent 争议仍适用现有 card-not-present 规则和 reason codes，无独立 agentic dispute 类别。

---

## 5. 参考文献 41/42 核实

### 5.1 参考文献 41：Mastercard MTN

原文：`https://www.mastercard.com/news/press/2023/（⚠️ 具体 agentic 集成未验证）`

**核实结论**：
- Mastercard MTN 是真实产品，发布于 2023 年
- 其在 Agent Pay 场景中的角色：已通过 Agent Pay for Machines（2026-06-10）验证——明确支持"cards, accounts and stablecoins"
- URL 精确化：原文 URL 不完整（缺少具体路径）

**建议更新**（将"具体 agentic 集成未验证"改为）：
`https://www.mastercard.com/news/press/2023/...（MTN）；Agent Pay for Machines（2026-06-10）已将稳定币明确列为支付轨道，30+ 参与方含 Coinbase，bankingnewsai.com 核实）`

### 5.2 参考文献 42：Visa Direct

原文：`https://developer.visa.com/capabilities/visa_direct（⚠️ agentic 集成未验证）`

**核实结论**：
- Visa Direct 确为实时推送付款服务，覆盖 FedNow/RTP/SEPA 等轨道
- Visa Direct + agent 场景的正式协议文档：**仍未找到**，维持"未验证"标注
- Visa 官方 Intelligent Commerce 材料未直接提及 Visa Direct 在 agent 场景中的集成规范

---

## 6. B.4 节末尾"补充"节关于 visa/trusted-agent-protocol 的说明（原文约 1153 行）

原文："深挖文件（W5）确认 GitHub 上存在 `visa/trusted-agent-protocol` 仓库，并能访问其 `README.md`（网页入口），但因 DNS 解析失败未能完成本地克隆，所有技术细节仍属'无法从克隆确认'。本文 B.4 节相关描述的'未验证'标注维持不变。"

**核实结论**：

本次核实通过 GitHub API 和 raw.githubusercontent.com 直接下载了仓库内容，成功访问并分析了：
- `README.md`
- `tap-agent/README.md`
- `tap-agent/agent_app.py`（完整源代码）
- `cdn-proxy/README.md`
- `agent-registry/README.md`
- `merchant-backend/app/security/signature_verification.py`

技术细节已经过直接代码核实（见第 1 节），B.4 节相关"未验证"标注可以**大幅降级**。

**建议替换原补充节内容**：

> **Visa TAP GitHub 仓库技术细节已核实**
>
> 通过 GitHub API（api.github.com/repos/visa/trusted-agent-protocol）和 raw.githubusercontent.com 直接获取了参考实现的完整源代码（包括 `agent_app.py`、签名验证模块），主要技术细节已在本文 B.4 节更新。仓库创建于 2025-10-10，截至 2026-10-14 仍在维护。许可证为 Visa Developer Center Terms of Use（非标准 OSS 许可），签名算法为 Ed25519/RSA-PSS-SHA256，Tag 值 `agent-browser-auth`（浏览）和 `agent-payer-auth`（支付）均从代码直接核实。

---

## 7. 仍未验证的内容

| 条目 | 尝试过的途径 | 说明 |
|------|------------|------|
| Visa Intelligent Commerce API 技术规范细节 | developer.visa.com（需登录）、visa.com 页面（产品描述层） | 需注册 Visa Developer 账户申请访问 |
| Mastercard 官方 Agent Pay 技术规范 | mastercard.com（403 Forbidden）、developer.mastercard.com（404）、Web Archive | Mastercard 网站对云 IP 全面封锁，无法从一手来源提取 |
| Visa Direct + agent 场景的正式 API 文档 | developer.visa.com、visa.com Intelligent Commerce 页、Bing/DuckDuckGo 搜索 | 未找到官方文档；结论：未集成或内部路线图，不公开 |
| Mastercard Agent Pay Acceptance Framework 原始文本 | mastercard.com（403）、Web Archive（无快照）| 参照 Agent Pay Trust Framework（Sep 2026）作为最新描述 |
| Visa Core Rules 具体 agent 相关条款 | BeastInsights 二手分析 | 一手 Visa Core Rules 文档需持卡组织成员资格访问 |
| Mastercard Transaction Processing Rules agent 条款 | BeastInsights 二手分析 | 同上 |

---

*核实时间：2026-10-19。一手来源：github.com/visa/trusted-agent-protocol（代码直接分析）、investor.visa.com（官方新闻稿）、visa.com/en-us/solutions/intelligent-commerce（官方产品页）、blog.cloudflare.com/secure-agentic-commerce（Cloudflare 官方博客，2025-10-24）、fidoalliance.org（FIDO Alliance 官方网站）、bankingnewsai.com（综合 Mastercard 官方时间线，含原始 mastercard.com URL 参考）。二手来源：beastinsights.com（2026-08 分析文章，用于卡组织规则部分）。*
