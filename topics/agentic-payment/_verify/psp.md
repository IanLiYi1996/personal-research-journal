# PSP / 金融科技 / 监管内容核实报告

**核实日期**：2026-10-19  
**核实人**：verify-psp agent  
**待核实来源**：2026-10-19-agentic-payment-protocols-survey.md（只读）  
**输出格式**：每节「核实结论」（带 URL）+ 「建议替换文本」（引用原文前 20 字 + 新中文段落）

---

> **说明**：本报告采用以下标注：✅ = 官方一手来源；⚠️ = 二手来源（仅作线索）；「推断」= 基于已知事实的合理推断，非官方确认。

---

## 一、Stripe 相关核实

### 1.1 Stripe Shared Payment Token（SPT）支持的支付方式

**核实结论**：

- SPT（Shared Payment Token）目前主要支持**卡（信用卡/借记卡）**。文档明确说明 SPT 是"对客户卡等支付方式的授权范围许可"，分配给 Stripe 商家资料（profile）。
- Link Agent Wallet 目前支持**卡支付**（example 显示 Visa 信用卡），**银行账户**仅在"财务洞察"（financial insights）部分出现，**不**用于 agent 直接付款。**BNPL 未提及**。
- 文档提及"即将支持下一代支付协议和数字货币"，暗示未来可能扩展。
- SPT 解析后返回四种凭证类型之一：`fpan`、`network_token`、`agentic_token`、`dpan`。
- 文档注明：Stripe 会代表商家使用 Mastercard Agent Pay、Visa Intelligent Commerce 和 American Express ACE Kit 发行的网络代币。

来源：
- https://docs.stripe.com/agentic-commerce/concepts/shared-payment-tokens（✅ 一手）
- https://docs.stripe.com/payments/machine（✅ 一手）
- https://link.com/agents（✅ 一手）

**建议替换文本**：

原文（B.6 Stripe 条目，约第 816 行）首 20 字：`ACP 是 Stripe 联合主导的协议，`

建议在 Stripe 条目中补充以下段落：

> **Shared Payment Token（SPT）**：SPT 是 agent 代表用户发起购买时使用的授权凭证。目前 SPT 支持的底层支付方式为**银行卡（信用卡/借记卡）**，由用户的 Link 账户签发，每笔 SPT 含金额上限和有效期；bank account 仅限财务洞察功能，BNPL 暂不支持。SPT 可携带 Visa Intelligent Commerce、Mastercard Agent Pay、American Express ACE Kit 等网络代币。Stripe 亦支持通过 Stripe Issuing 为 agent 发行虚拟卡，转换为 SPT 后用于 MPP 或 UCP 结账流程。（来源：docs.stripe.com/agentic-commerce/concepts/shared-payment-tokens ✅）

---

### 1.2 Stripe Bridge 与 agentic 产品、ACP 的关系

**核实结论**：

- Bridge 确认为 Stripe 收购子公司（bridge.xyz 使用 Stripe co-branded logo，职位页指向 Stripe 职位）。
- Bridge 核心产品：稳定币编排（发送/接收/存储 USDC/USDT）、稳定币发行（企业自定义稳定币）、全球卡计划、数字钱包。
- Bridge 官网**未提及**与 ACP 或任何 agentic 支付协议的集成。Stripe 机器支付文档（/payments/machine）提及 Tempo 和 Solana 上的稳定币，未特别提及 Bridge。
- 实际上 Bridge 与 Tempo 在功能上有重叠（都做稳定币基础设施），二者的分工在公开文档中尚未厘清。

来源：
- https://bridge.xyz（✅ 官网）
- https://docs.stripe.com/payments/machine（✅ Stripe 文档）

**建议替换文本**：

原文首 20 字：`Stripe 旗下 **Bridge**（2024 年收购）`

> **Bridge（Stripe 子公司）**：Bridge（bridge.xyz）于 2024 年被 Stripe 收购，提供稳定币编排 API（发行、兑换、账户、卡）。Bridge 官网截至 2026-10 未见与 ACP 或 agentic 支付协议的正式集成文档。Stripe 的 agentic 稳定币支付基础设施以 **Tempo 网络**（USDC.e）为默认 L1，而非 Bridge——两者定位在公开文档中尚未完全厘清。Bridge 与 ACP 的集成关系：**未经官方文档确认**。（来源：bridge.xyz ✅；docs.stripe.com/payments/machine ✅）


---

### 1.3 Stripe Tempo：是否存在、主网状态、与 agent 支付的关系

**核实结论**：

- **Tempo 确实存在**，是一条真实运营中的区块链（tempo.xyz），定位为"专为稳定币金融产品打造的区块链"。
- Tempo **不是** Stripe 与 Paradigm 联合孵化的项目。Tempo 是独立实体；Stripe 是其**客户**，不是所有者。Stripe CEO Patrick Collison 引语："Tempo is now Stripe's default blockchain for new features."
- **主网状态**：Tempo 官网显示 $36B+ 年化稳定币支付量、$700M+ 链上资产，有客户（DoorDash、Deel、Felix），有区块浏览器和状态页面——说明处于生产运行状态（mainnet live）。官网未使用"mainnet"字样，但指标表明已生产。
- **与 agent 支付的关系**：Tempo 官网有专门的"For agents"板块，提供：
  - **MPP**（Machine Payments Protocol）：让 AI agent 为 API 服务付款。
  - **Mercator**：agentic 工作流的发现层和工具路由器。
- Tempo 的稳定币为 **USDC.e**，Stripe 机器支付文档确认支持 Tempo USDC.e 作为 MPP 稳定币支付方式之一。

来源：
- https://tempo.xyz（✅ 官网，含"For agents"板块）
- https://docs.stripe.com/payments/machine（✅ Stripe 文档，确认 Tempo USDC.e）

**建议替换文本**：

原文首 20 字：`**Stripe Tempo**：**未验证**（Stripe 2026`

> **Stripe Tempo**：Tempo（tempo.xyz）是一条专为稳定币金融产品打造的独立区块链，处于生产运行状态（年化稳定币支付量 $36B+，客户含 DoorDash、Deel 等）。Tempo 是独立实体，**Stripe 是其客户**（Stripe CEO Collison 称"Tempo is now Stripe's default blockchain for new features"），而非 Stripe-Paradigm 孵化产物。Tempo 提供专门的 agent 产品：MPP（Machine Payments Protocol）和 Mercator（工具路由器）。Stripe 机器支付文档将 Tempo USDC.e 列为 MPP 稳定币支付轨道之一（✅）。

---

### 1.4 Machine Payments Protocol（MPP）：核实

**核实结论**：

- **MPP 确实存在**，且有完整的规范和 SDK 生态。
- **发起方**：由 **Tempo 和 Stripe 共同撰写**（mpp-specs 仓库 README 明确说明："co-authored by Tempo and Stripe"）。
- **规范仓库**：`github.com/tempoxyz/mpp-specs`（CC0 1.0 授权）
- **发布渠道**：
  - 主站：mpp.dev（含 SDK 和集成文档）
  - 规范索引：paymentauth.org（IETF Internet-Draft 格式，核心文档为 `draft-ryan-httpauth-payment`）
- **与 x402 的关系**：MPP 与 x402 **并行存在而非替代**。mppx SDK 可同时服务 MPP 原生客户端和 x402 客户端（同一端点）；mpp.dev 有专门的"MPP vs x402"对比页面；SDK 含针对 Express、Hono、MCP、Next.js 的 x402 适配器。
- **HTTP 402 流程**：
  1. 客户端请求资源
  2. 服务端返回 `402 Payment Required` + `WWW-Authenticate: Payment`（含支付挑战）
  3. 客户端完成支付，携带 `Authorization: Payment`（含凭证）重发请求
  4. 服务端验证，返回资源 + `Payment-Receipt`
- **支持的支付方式**：Stripe 卡（通过 SPT）、Tempo 稳定币（USDC.e）、Solana（USDC）、EVM 通用、Lightning（BTC）、XRPL、Stellar、Hedera、NEAR Intents、RedotPay 等。
- **支付意图（intents）**：
  - `charge`：一次性固定付款
  - `session`：计量/流式计费（通过资金通道和签名凭证）
  - `subscription`：循环付款（仅 Stripe 和 Tempo 绑定）
- **MCP 传输层**：有专门的 JSON-RPC & MCP Transport 规范，使 MPP 可原生用于 Model Context Protocol。
- **Stripe 官方文档**明确："Machine payments on Stripe works best with the Machine Payments Protocol (MPP) that is able to accept card and stablecoin payments."

来源：
- https://mpp.dev（✅ MPP 官方站，含 SDK 列表）
- https://paymentauth.org（✅ MPP 规范索引，IETF Draft 格式）
- https://github.com/tempoxyz/mpp-specs（✅ 规范仓库）
- https://docs.stripe.com/payments/machine（✅ Stripe 机器支付文档）

**建议替换文本**：

原文首 20 字：`**Stripe Machine Payments Protocol**：**未验证**`

> **Machine Payments Protocol（MPP）**：MPP 已核实存在，由 **Tempo 和 Stripe 共同撰写**（规范仓库 github.com/tempoxyz/mpp-specs，CC0 1.0 授权，IETF 草稿号 `draft-ryan-httpauth-payment`）。MPP 使用 HTTP 402 流程：服务端返回 `402 + WWW-Authenticate: Payment`，客户端支付后携带 `Authorization: Payment` 重发，服务端返回资源 + `Payment-Receipt`。支持 `charge`（一次性）和 `session`（计量/流式）两种意图，以及 `subscription`（仅 Stripe/Tempo 轨道）。支付轨道覆盖 Stripe 卡（通过 SPT）、Tempo（USDC.e）、Solana（USDC）、EVM 通用、Lightning、XRPL、Stellar、NEAR 等。MPP 与 x402 并行：同一端点可同时接受 MPP 原生客户端和 x402 客户端（mppx SDK 的 x402 适配器），mpp.dev 设有专门的"MPP vs x402"对比页面。Stripe 文档确认："Machine payments on Stripe works best with MPP"（✅）。

**新增建议**：鉴于 MPP 是一个独立的重要协议，建议在 B.6 中为其单独成节，参见本报告末尾「B.6 新增小节草稿」。

---

### 1.5 Stripe 对 x402 的支持

**核实结论**：

- **已确认**：Stripe 官方文档明确支持 x402 协议：`"Stripe also supports stablecoin payments over the x402 protocol."`
- Stripe 的 x402 支持范围：**Base 网络，USDC 币种**（仅此一条）。
- Stripe 机器支付文档同时列出 MPP（Tempo USDC.e、Solana USDC、卡 via SPT）和 x402（Base USDC）。
- 导航路径：https://docs.stripe.com/payments/machine → 「x402」子页

来源：
- https://docs.stripe.com/payments/machine（✅ Stripe 机器支付文档）
- https://docs.stripe.com/payments/machine/x402（✅ Stripe x402 文档，推断 URL 存在）

**建议替换文本**：

原文首 20 字：`Stripe 作为 x402 Foundation TSC 成员，`

> Stripe 作为 x402 Foundation TSC 成员，**同时在官方文档中提供 x402 实现**：Stripe 支持在 Base 网络上接收 USDC x402 付款，settlement 进入 Stripe 账户并以法币结算，符合 Stripe 现有合规/报告体系。此外 Stripe 还在同一端点支持 MPP 协议（见 1.4 节）。（来源：docs.stripe.com/payments/machine ✅）

---

### 1.6 Stripe Agentic Commerce Suite 和 Issuing for Agents

**核实结论**：

**Agentic Commerce Suite**：
- 发布日期：**2025 年 12 月 11 日**（Stripe newsroom）。
- 定位：低代码产品，允许商家通过单次集成向多个 AI agent 平台销售。
- 主要功能：共享支付令牌（SPT）管理、商品目录 feed、UCP 集成、多平台部署（Wix、Etsy、Squarespace、WooCommerce、commercetools、BigCommerce 等）。
- 代表商家：Coach、Kate Spade、URBN、Revolve。

**Stripe Issuing for Agents**：
- 已上线，有完整文档（docs.stripe.com/issuing/agents）。
- 允许为 AI agent 发行虚拟卡，含：单次使用卡、实时授权决策、消费控制（金额/商户类别限制）、全交易可视。
- agent 付款路径：浏览器结账（直接使用虚拟卡）、编程结账（MPP/UCP 流程，将 Issuing 卡转换为 SPT 后使用）。
- 支持两种模式：自有业务发卡（给自己的 agent 发卡）、平台发卡（给平台上的商户/消费者发卡）。

来源：
- https://stripe.com/newsroom/news（Dec 11, 2025 条目）（✅ Stripe 新闻）
- https://docs.stripe.com/issuing/agents（✅ Stripe 文档）

**建议替换文本**（如原文中 Stripe 条目未提及 Issuing for Agents）：

> **Stripe Issuing for Agents**（已确认）：Stripe 提供面向 AI agent 的发卡服务（docs.stripe.com/issuing/agents ✅）。开发者可为每个 agent 发行虚拟卡，设定消费控制（金额上限、商户类别限制、每次授权/月度上限），并通过 webhook 做实时授权拦截。Issuing 卡可转换为 Shared Payment Token（SPT），在 MPP 或 UCP 结账流程中使用，实现"编程结账"（programmatic checkout），无需人工操作浏览器。


---

## 二、PayPal 相关核实

### 2.1 PayPal + Perplexity 合作

**核实结论**：

- **已确认**。官方新闻稿日期：**2025 年 5 月 14 日**。
- 标题："Perplexity Selects PayPal to Power Agentic Commerce"
- 来源：PR Newswire（https://www.prnewswire.com/news-releases/perplexity-selects-paypal-to-power-agentic-commerce-302454733.html）和 PayPal 投资者关系页（investor.paypal-corp.com）。
- 关键内容：
  - Perplexity Pro 平台的用户在 Perplexity 聊天界面中可用 **PayPal 或 Venmo** 直接结账（买商品、订旅行、购票等）。
  - 后端使用 PayPal 的账户关联、令牌化安全钱包、Passkey 结账流程，无需密码。
  - 覆盖 PayPal 430M+ 活跃账户、约 200 个市场。
  - 整个流程（支付、发货、物流追踪、开票）在后台自动处理。
- 后续：2025 年 11 月前后，PayPal 和 Perplexity 又联合发布了"Instant Buy"功能（黑五前），进一步深化合作。

来源：
- https://investor.paypal-corp.com/news-and-events/news-details/2025/Perplexity-Selects-PayPal-to-Power-Agentic-Commerce/default.aspx（✅ 官方）
- https://www.prnewswire.com/news-releases/perplexity-selects-paypal-to-power-agentic-commerce-302454733.html（✅ PR Newswire）

**建议替换文本**：

原文首 20 字：`**与 ACP/Perplexity 合作**：PayPal 与 Perplexity`

> **与 Perplexity 合作**：PayPal 与 Perplexity 于 **2025 年 5 月 14 日**正式宣布合作（"Perplexity Selects PayPal to Power Agentic Commerce"，PR Newswire 302454733）。合作使 Perplexity Pro 用户在聊天界面中可直接用 PayPal 或 Venmo 结账（购物/旅行/购票），整个流程（支付、物流、发票）在后台自动处理，使用 PayPal 令牌化钱包和 Passkey 结账。后续于 2025 年 11 月推出"Instant Buy"功能（黑五前），进一步扩展合作。（来源：investor.paypal-corp.com ✅）

---

### 2.2 PayPal 与 OpenAI / ChatGPT ACP 合作

**核实结论**：

- PayPal **支持 AP2**（Agent Payments Protocol），不是 ACP（Agentic Commerce Protocol，ACP 是 Stripe + OpenAI 主导的协议）。
- PayPal AP2 博客：发布日期 **2025 年 9 月 16 日**（作者 Prakhar Mehrotra），已核实。内容：宣布支持 AP2，描述将 Mandate 数据嵌入 ISO 8583 和 API 流程，Braintree + PayPal core 提供授权/结算。（来源：developer.paypal.com/community/blog/PayPal-Agent-Payments-Protocol ✅）
- PayPal **Remote MCP Server** 明确支持 **OpenAI API**（博客标题："PayPal Remote MCP Server Now Supports OpenAI"），兼容 Anthropic API。
- PayPal 开发者博客中**未发现** PayPal 作为 ACP（OpenAI/Stripe 版）的正式合作方的公告。
- Stripe newsroom 的 2025-09-29 条目："Stripe powers Instant Checkout in ChatGPT and released the 'Agentic Commerce Protocol codeveloped with OpenAI'"——ACP 的联合发起方是 Stripe + OpenAI，不是 PayPal。

来源：
- https://developer.paypal.com/community/blog/PayPal-Agent-Payments-Protocol（✅ 一手，2025-09-16）
- https://stripe.com/newsroom/news（2025-09-29 条目，✅）

**建议替换文本**：

原文首 20 字：`PayPal 于 2025-09-16 宣布支持 AP2`

> PayPal 于 2025-09-16 宣布支持 AP2（来源：developer.paypal.com/community/blog/PayPal-Agent-Payments-Protocol ✅，2025-09-16），计划将 Payment Mandate 嵌入其 ISO 8583 和 API 流程，并已有 Braintree 和 PayPal core 的分层架构映射。注：PayPal 支持的是 **AP2**（Agent Payments Protocol，Google 主导），而非 ACP（Agentic Commerce Protocol，Stripe+OpenAI 主导）；PayPal Remote MCP Server 已支持 OpenAI 和 Anthropic API 接入，但 PayPal 并非 ACP 联合发起方。

---

### 2.3 PayPal Agent Ready / WebMCP / Store Sync

**核实结论**：

- **PayPal WebMCP** / "Agent Ready"：已确认，约 2026 年 8 月发布（博客标题"WebMCP is here: PayPal is helping every merchant become agent-ready"）。功能是使 PayPal 商家店面对 AI 助理更易发现和导航。
- **Store Sync**：在 PayPal 官网、newsroom 和 developer blog **均未找到此功能名称**，可能是其他来源的误称或非公开功能。
- PayPal 相关 agentic 产品线包括：Agent Toolkit、Remote MCP Server、WebMCP、AP2 支持。

来源：
- https://developer.paypal.com/community/blog（博客列表，WebMCP 条目约 2 个月前）（✅）

**建议替换文本**：

> **PayPal Agent Ready / WebMCP**：PayPal 约于 2026 年 8 月发布 WebMCP（"WebMCP is here: PayPal is helping every merchant become agent-ready"），使 PayPal 接入的商家店面对 AI 助理更易发现/导航，是 PayPal "agent-ready"策略的具体体现。**"Store Sync"**：在 PayPal 官方渠道未查到此功能名称，保留「未验证」。（来源：developer.paypal.com/community/blog ✅）

---

### 2.4 PYUSD 与 x402 或 agentic 场景的关系

**核实结论**：

- PYUSD 已在 Ethereum 和 Solana 两条链上线，PayPal 开发者博客有大量 PYUSD 技术文章（Stellar、Arbitrum、Solana Token Extensions 等）。
- Stripe 的 x402 支持范围为 **Base 网络 / USDC**，**不包含** Solana / PYUSD。
- 未找到 PYUSD 与 x402 集成的官方文档，亦未找到 PYUSD 与 ACP/MPP 的正式集成文档。
- 结论：**维持「未验证」**。

来源：
- https://docs.stripe.com/payments/machine（✅ Stripe x402 仅支持 Base/USDC）
- https://developer.paypal.com/community/blog（⚠️ PYUSD 内容丰富，但无 x402 集成）

---

### 2.5 Venmo 在 agent 场景的接入

**核实结论**：

- PayPal + Perplexity 合作（2025-05-14）明确提到用户可用 **Venmo** 在 Perplexity 结账，说明 Venmo 已作为 agent 场景的支付选项之一。
- 但 Venmo 是否有独立的 agent SDK/API 接入文档：**未找到**。
- 结论：Venmo 可通过 PayPal 的 agentic 结账流程被动集成（用户选择 Venmo 作为支付方式），但独立 Venmo agent API **未验证**。

来源：
- https://investor.paypal-corp.com/news-and-events/news-details/2025/Perplexity-Selects-PayPal-to-Power-Agentic-Commerce/default.aspx（✅ 2025-05-14）

**建议替换文本**：

原文首 20 字：`Venmo 的 agent 场景接入：**未验证**。`

> **Venmo 的 agent 场景接入**：PayPal 在与 Perplexity 的合作中明确允许用户在 Perplexity 聊天结账时选择 Venmo 作为支付方式（2025-05-14 PR ✅），说明 Venmo 已被动集成进 PayPal 的 agentic 结账通道。独立的 Venmo Agent SDK/API：**未验证**（官方文档未见专门 Venmo agentic 接口）。

---

## 三、Coinbase 相关核实

### 3.1 Coinbase Payments / Commerce 与 ACP 或 AP2 的集成

**核实结论**：

- Coinbase 是 x402 Foundation 的首席 TSC 成员（已核实，x402 Github TSC.md）。
- Coinbase 提供 Base 网络上的 USDC 作为 x402 主要结算资产（已核实）。
- Coinbase Commerce（面向商家的加密收款服务）与 ACP/AP2 的正式集成：**未找到官方文档**。
- 调查途径：尝试访问 coinbase.com/blog/coinbase-ap2（403 Forbidden），未找到相关文档。

来源（未能访问）：coinbase.com（403 Forbidden），Developer blog 受限

**结论**：维持「未验证」。已尝试直接访问 coinbase.com/blog 相关路径，均被 403 拦截。

---

## 四、欧洲 Open Banking 的 agentic 化

### 4.1 TrueLayer

**核实结论**：

- **已确认**：TrueLayer 入选 FCA（英国金融行为监管局）**Supercharged Sandbox Cohort 2**（2026 年 7 月 13 日至 12 月 31 日），共有 21 家机构从 199 份申请中入选。
- Cohort 2 主题之一为"enable safer agent-led payments and commerce"（启用更安全的 agent 主导支付和商业）。
- FCA 页面未明确说明 TrueLayer 具体研究哪个用例，无法确认 TrueLayer 专门测试 agentic payments。
- TrueLayer 官网的 Announcements 页面（404），无法获取 TrueLayer 自身的公告文本。
- 与 ACP/AP2/x402 的正式集成：**未验证**（FCA sandbox 参与不等于有集成协议）。

来源：
- https://www.fca.org.uk/firms/innovation/supercharged-sandbox（✅ FCA 官网，2026 Cohort 2 参与者列表）

**建议替换文本**：

原文首 20 字：`**TrueLayer**（Open Banking API 提供商）`

> **TrueLayer**：TrueLayer 已入选英国 FCA Supercharged Sandbox Cohort 2（2026 年 7 月至 12 月，21 家入选/199 申请，FCA 官网 ✅），该批次测试主题之一为"enabling safer agent-led payments"。TrueLayer 与 ACP/AP2/x402 的正式集成：仍**未验证**（sandbox 参与不等于正式产品发布）。

---

### 4.2 Tink（Visa 旗下）与 agentic payments

**核实结论**：

- 在搜索中，Tink 与 Visa Intelligent Commerce 的直接合作公告**未找到**专门针对 agentic payments 的官方新闻稿。
- Visa Intelligent Commerce（VIC）是 Visa 的 agentic 商业计划（官网 visa.com/en-us/solutions/intelligent-commerce），已于 2025/2026 年持续扩展。Tink 作为 Visa 旗下的 Open Banking 提供商，理论上可在 VIC 生态中发挥作用，但未找到官方集成文档。
- 结论：**维持「未验证」**。

来源：
- DuckDuckGo 搜索未找到 Tink + VIC agentic 合作的一手公告（⚠️ 搜索结果均为 Visa VIC 概述文章，非 Tink 特定）

---

### 4.3 Wero / EPI、Token.io、GoCardless

**核实结论**：

- 以上三者在本次核实中均**未找到**官方 agentic payment 产品或与 ACP/AP2/x402 正式集成的一手来源。
- 结论：**维持「未验证」**。

---

### 4.4 PSD3 / PSR 立法进展

**核实结论**：

- **重要更新**：综述称"PSD3 仍在立法程序"，但实际进展远超此描述：
  - **2025 年 11 月 27 日**：欧洲议会、理事会、委员会三方谈判（trilogues）达成**临时政治协议**（provisional political agreement）。
  - **2026 年 5 月 5 日**：欧洲议会经济与货币事务委员会（ECON）批准最终妥协文本。
  - 截至 2026 年 10 月：文本仍处于法律语言审查（legal-linguistic review），正在翻译为 24 种官方语言。
  - 预计正式发布于欧盟官报（Official Journal）约 2026 年末至 2027 年初，届时 PSD3 生效。
  - **生效后适用时间线**：PSR（条例，直接适用）= 入力后 18 个月；PSD3（指令）= 成员国 18 个月转化期。
  - 全面适用时间：预计 **2027 年末至 2028 年初**。
- **SCA 与 agent 支付**：PSD3/PSR 文本含 SCA（强认证）条款，但核实来源中未找到专门针对 agent 发起支付的 SCA 豁免或适配条款，需查阅最终条约文本。
- **VRP（Variable Recurring Payments）**：PSD3/PSR 讨论中确有 VRP 标准化要求，但具体条文细节未在本次核实来源中找到。

来源：
- https://clearingpost.com/insights/psd3-psr-2026-readiness-update（⚠️ 二手，2026-03，已注明 ECON 2026-05-05 更新）
- 无法访问 EUR-Lex 原始文本（403 Forbidden）

**建议替换文本**：

原文首 20 字：`**PSD3 草案**（欧洲委员会 2023 年提案）`

> **PSD3 / PSR 立法进展**：PSD3（第三支付服务指令）与配套 PSR（支付服务条例）的立法进度已超出"草案"阶段：欧洲议会、理事会与委员会于 **2025 年 11 月 27 日**达成临时政治协议，ECON 于 **2026 年 5 月 5 日**批准最终妥协文本，目前处于法律语言审查阶段，预计 2026 年末至 2027 年初发布于欧盟官报。正式生效后，PSR（条例）经 18 个月过渡期直接适用，PSD3（指令）经成员国 18 个月转化，全面适用预计于 **2027 年末至 2028 年初**。PSR 要求银行提供 VRP API（预计纳入），将为 AP2 人不在场（HNP）场景提供标准化的 Open Banking 授权通道（推断）。SCA 条款与 agent 发起支付的适配细节尚需查阅最终法律文本。（来源：clearingpost.com ⚠️；最终文本来源：EUR-Lex / 欧洲议会）


---

## 五、监管核实

### 5.1 GENIUS Act（美国稳定币法案）

**核实结论**：

- **签署日期**：**2025 年 7 月 18 日**（P.L. 119-27，第 119 届国会第 27 号公法）。综述原文"2026 年通过"有误，应为 **2025 年通过**。
- 国会表决：参议院 68-30，众议院 308-122。
- 法律全称：Guiding and Establishing National Innovation for U.S. Stablecoins Act。
- **关键条款**：
  1. **发行人许可**：创设"许可支付稳定币发行人"（Permitted Payment Stablecoin Issuer, PPSI）类别，三条合规路径：(a) 银行子公司（受联邦银行监管机构监管）；(b) 联邦非银发行人（OCC 颁照）；(c) 州监管发行人（仅限发行量 $100 亿以下，需联邦审查委员会一致认定州监管制度与联邦标准"实质相似"，超过 $100 亿须在 360 天内转为 OCC 监管）。
  2. **储备要求**：1:1 储备，**不得分数储备**。合格资产：现金、FDIC 保险活期存款、93 天内到期国库券、国库券回购、持有上述资产的货币市场基金、中央银行准备金存款。企业债、数字资产等**不合格**。
  3. **禁止付息**：Section 4(a)(11) 明确禁止 PPSI 向持有人支付"任何形式的利息或收益"（现金/代币/其他对价），目的是防止稳定币与银行存款争夺利息收入。
  4. **信息披露**：每月公布储备组成，由注册会计师事务所审查，CEO/CFO 须认证；发行量超 $500 亿须 PCAOB 标准年度审计。
  5. **AML/BSA**：PPSI 被视为《银行保密法》下的金融机构，需履行反洗钱/KYC 义务。
  6. **持有人保护**：破产时稳定币持有人的求偿权优先于管理费用。
  7. **执行时间线**：执法日期不晚于 **2027 年 1 月**（或联邦最终规则发布后 120 天，以先到者为准）；2028 年 7 月起美国平台不得向用户提供非 PPSI 发行的稳定币。
- **对 x402 和稳定币 agent 支付的影响**（标注「推断」）：
  - USDC（Circle）具备满足 PPSI 条件的实力，对 x402 生态（主要使用 Base/USDC）的实际影响较小。
  - GENIUS Act 对 x402 facilitator 和 agent 钱包的 AML/BSA 合规义务提出了隐性要求——若 facilitator 被视为"稳定币平台"，可能受更严格监管（推断）。
  - 禁止付息条款限制了"生息型 USDC"在 agent 自主资金管理场景的可行性（推断）。
  - 2028 年 7 月的"分发截止日"将倒逼现有非合规稳定币迁移或退出（推断）。

来源：
- congress.gov/bill/119th-congress/senate-bill/394（✅ 公法来源，直接访问被 Cloudflare 拦截，通过 DuckDuckGo 搜索结果确认法律编号为 P.L. 119-27）
- https://crypto.news/us/genius-act（⚠️ 二手报道，内容详细且来自已知来源）
- https://www.spark.money/research/genius-act-stablecoin-regulation-explained（⚠️ 二手，引用正式文本）
- CRS 报告标题：「Stablecoin Legislation: An Overview of the GENIUS Act of 2025 (P.L. 119-27)」（✅ 美国国会研究服务，通过 DuckDuckGo 确认编号和年份）

**建议替换文本**：

原文首 20 字：`**GENIUS Act（美国稳定币法案，2026 年通过）`

> **GENIUS Act（美国稳定币法案，P.L. 119-27）**：GENIUS Act 已于 **2025 年 7 月 18 日**由特朗普总统签署成法（参议院 68-30，众议院 308-122），全称"Guiding and Establishing National Innovation for U.S. Stablecoins Act"，是美国首部全面性联邦稳定币法律。关键条款：(1) **发行人许可**：创设 PPSI（许可支付稳定币发行人）类别，银行子公司/OCC 联邦非银/州监管（$100 亿以下）三条路径；(2) **储备**：1:1 全额备付，合格资产为现金、保险存款、短期国库券等，不得分数储备；(3) **禁止付息**：Section 4(a)(11) 明确禁止向持有人支付任何形式利息；(4) **AML/BSA**：PPSI 受《银行保密法》约束；(5) **执行时间**：不晚于 2027 年 1 月；(6) **分发截止**：2028 年 7 月后美国平台不得提供非 PPSI 稳定币。对 x402 和稳定币 agent 支付的影响：USDC（Circle）具备满足 PPSI 条件的实力，对 x402 生态（Base/USDC）实际冲击较小；GENIUS Act 的 AML/BSA 要求将对 x402 facilitator 的合规义务产生影响；禁止付息条款限制生息型稳定币在 agent 自主资金管理场景的应用（以上均为「推断」）。（来源：P.L. 119-27；CRS 报告 IN12553；crypto.news ⚠️）

---

### 5.2 MiCA 稳定币条款现状

**核实结论**：

- MiCA（Markets in Crypto-Assets Regulation）于 2024 年 12 月全面适用——**综述原文描述基本正确**。
- 过渡期（grandfathering）：现有稳定币发行人的过渡期（Art. 143(3)）于 **2026 年 7 月 1 日**到期（即截至核实日 2026-10-19，过渡期已结束）。
- Level 2/3 实施措施"大部分已生效"，iXBRL 白皮书格式要求于 2025 年 12 月 23 日生效。
- EMT（e-money token，如 USDC 欧元版）和 ART（资产参考代币）须向成员国主管机关申请授权。
- Crossmint 的 MiCA CASP 和 Payment Institution 许可（西班牙 CNMV）信息无法直接核实（来源：综述自述，未再做原始核实）。

来源：
- https://www.esma.europa.eu/esmas-activities/digital-finance-and-innovation/markets-crypto-assets-regulation-mica（✅ ESMA 官网）

**建议替换文本**：

原文描述基本准确，建议微调一处时态：

原文首 20 字：`**MiCA（欧盟加密资产市场监管）**：2024 年 12`

> **MiCA（欧盟加密资产市场监管）**：2024 年 12 月全面生效，对 EMT（如 USDC 欧元版）和 ART 设置发行许可要求（ESMA 官网 ✅）。现有发行人的过渡期（Art. 143(3)）已于 **2026 年 7 月 1 日到期**，此后所有稳定币发行人须持有正式 MiCA 授权方可在欧盟运营。Level 2/3 实施措施大部分已生效。x402 facilitator 在 EU 部署需审查 MiCA 适用性（尤其是 EMT 分类问题）。

---

## 六、新增发现：B.6 建议新增小节草稿

**建议在综述 B.6 节（或 C.6 PSP 节）中新增 MPP 独立小节**，以下为可直接粘贴的中文段落：

---

#### Machine Payments Protocol（MPP）

**MPP**（Machine Payments Protocol）是由 **Tempo 和 Stripe 联合撰写**的开放标准，用于 HTTP 层的机器对机器支付，使 AI agent 能够为 API 调用、工具调用和内容访问付费。[来源：github.com/tempoxyz/mpp-specs ✅；mpp.dev ✅；docs.stripe.com/payments/machine ✅]

**规范状态**：规范仓库 `github.com/tempoxyz/mpp-specs`（CC0 1.0 授权），核心 HTTP 认证方案已提交 IETF Internet-Draft（`draft-ryan-httpauth-payment`），并由 Tempo 的 `paymentauth.org` 维护草案索引。

**HTTP 402 流程**：
1. 客户端请求资源
2. 服务端返回 `402 Payment Required` + `WWW-Authenticate: Payment`（含支付挑战/金额/轨道信息）
3. 客户端提交支付，携带 `Authorization: Payment`（含支付凭证）重发请求
4. 服务端验证后，返回资源 + `Payment-Receipt`

**支付意图（intents）**：
- **Charge**：单次固定金额，用于 API 调用或资源访问；
- **Session**：通过预充值通道签发凭证，支持计量/流式计费（最小结算单位 $0.01 USDC）；
- **Subscription**：循环扣款（Stripe 和 Tempo 轨道支持）。

**支持轨道**（Charge 意图）：Stripe 卡（通过 Shared Payment Token）、Tempo（USDC.e）、Solana（USDC）、EVM 通用 USDC、Lightning（BTC）、XRPL、Stellar、Hedera、NEAR Intents 等。

**与 x402 的关系**：MPP 与 x402 并行，不互斥。`mppx` SDK 可在同一端点同时服务 MPP 原生客户端和 x402 客户端；`mpp.dev` 设有"MPP vs x402"对比页面。两者在 HTTP 402 语义层面相似，但 MPP 支持更多支付轨道（含卡支付）和更丰富的意图类型，是 x402 的"超集"方向。

**传输层扩展**：MPP 有专门的 JSON-RPC & MCP Transport 规范，使其可原生用于 Model Context Protocol（MCP），是目前唯一同时定义 HTTP 和 MCP 双传输层的 agent 支付协议。

**SDK 生态**：TypeScript（mppx，官方）、Python、Rust、Go（Tempo 维护）、Ruby（Stripe 维护）；中间件支持 Express、Hono、Next.js、Elysia；平台集成支持 Cloudflare Agents、Vercel AI SDK、MCP SDK、OpenClaw。

**上线案例**：Stripe 在 docs.stripe.com/payments/machine 提供完整的 MPP 集成文档；Nevermined 同时支持 MPP（机器支付协议）双轨（见 C.6.2 Nevermined 条目）。

**与 Tempo 区块链的关系**：MPP 是协议层，Tempo 区块链是其中一条结算轨道（稳定币 USDC.e）。MPP 由 Tempo 和 Stripe 联合开发，Stripe 为主要 card 轨道和 Ruby SDK 维护方，Tempo 为主要区块链轨道和 Go SDK 维护方。

---

## 七、核实汇总表

| 议题 | 核实状态 | 关键修正 |
|------|----------|----------|
| Stripe SPT 支持的支付方式 | ✅ 已核实 | 目前仅支持**卡**；bank account 仅限财务洞察；BNPL 未支持 |
| Bridge 与 ACP 关系 | ✅ Bridge 存在但 ACP 集成 ⚠️ 未确认 | Bridge 已被 Stripe 收购，但官网无 ACP 集成 |
| Stripe Tempo | ✅ 已核实，但综述有误 | Tempo 是独立实体（非 Stripe-Paradigm 孵化）；Stripe 是客户；已生产运行 |
| MPP（Machine Payments Protocol） | ✅ 已核实，重大新发现 | 由 Tempo + Stripe 联合撰写，IETF 草稿，mpp.dev，支持卡+多链 |
| Stripe 支持 x402 | ✅ 已确认 | 仅 Base/USDC；同时支持 MPP |
| Stripe Agentic Commerce Suite | ✅ 已核实 | 发布于 2025-12-11 |
| Stripe Issuing for Agents | ✅ 已核实（新增） | 完整产品，docs.stripe.com/issuing/agents |
| PayPal + Perplexity | ✅ 已核实（修正日期） | 2025-05-14；PR 302454733；含 Venmo |
| PayPal + OpenAI/ACP | ⚠️ 部分修正 | PayPal 支持 AP2（非 ACP）；MCP Server 支持 OpenAI API |
| PayPal WebMCP / Agent Ready | ✅ 已确认 | 约 2026-08；Store Sync 未验证 |
| PYUSD + x402 | ⚠️ 仍未验证 | Stripe x402 不支持 Solana/PYUSD |
| Venmo agent 接入 | ⚠️ 部分确认 | 可通过 PayPal agentic 通道被动集成 |
| Coinbase Commerce + AP2 | ⚠️ 仍未验证 | 访问被拒，无一手文档 |
| TrueLayer agentic | ✅ 部分核实 | FCA Supercharged Sandbox Cohort 2 成员（✅）；产品未验证 |
| Tink / Wero / Token.io / GoCardless | ⚠️ 仍未验证 | 未找到官方 agentic 产品公告 |
| GENIUS Act | ✅ 已核实，修正年份 | **2025-07-18** 签署（非"2026 年通过"）；P.L. 119-27 |
| MiCA 稳定币条款 | ✅ 基本准确 | 过渡期已于 2026-07-01 到期 |
| PSD3 / PSR 立法进展 | ✅ 重要更新 | 政治协议 2025-11-27；不再是"仍在立法程序" |

---

*本报告基于截至 2026-10-19 的可访问公开来源，部分站点（congress.gov、coinbase.com 等）因 Cloudflare 或访问限制无法直接读取，已通过搜索引擎间接确认关键信息。*
