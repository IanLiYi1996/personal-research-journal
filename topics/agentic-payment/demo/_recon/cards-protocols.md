# 卡组织 Agent 沙箱 & 商业层协议调研报告

> 采信标准：一手来源（官方 GitHub / 官方开发者门户 / raw.githubusercontent.com）  
> 日期：2026-10-20  
> 未验证：标注「未验证」

---

## 1. Visa Intelligent Commerce (VIC)

### 1.1 沙箱访问

| 项目 | 详情 |
|------|------|
| 是否有沙箱 | **是**，沙箱 MCP Server 端点：`https://sandbox.mcp.visa.com` |
| 是否自助申请 | **否**。developer.visa.com/docs/visa-intelligent-commerce 需要登录并 "request access"，页面显示 "No access — request access to view"，未提供自助试用路径 |
| 访问入口 | [developer.visa.com/capabilities/visa-intelligent-commerce](https://developer.visa.com/capabilities/visa-intelligent-commerce)（需登录）+ Visa Intelligent Commerce onboarding 流程 |
| 参考资料 | [github.com/visa/ai](https://github.com/visa/ai) — 官方集成工具包（45 stars，2026-10-04 更新） |

### 1.2 沙箱所需凭证

来源：[visa/ai apps/vic-mcp-examples/.env.example](https://raw.githubusercontent.com/visa/ai/main/apps/vic-mcp-examples/.env.example)

```
VISA_MCP_BASE_URL=https://sandbox.mcp.visa.com   # 沙箱地址已确认
VISA_VIC_API_KEY           # VIC onboarding 获得
VISA_VIC_API_KEY_SS        # VIC onboarding 获得
VISA_EXTERNAL_CLIENT_ID    # VIC onboarding 获得
VISA_EXTERNAL_APP_ID       # VIC onboarding 获得
VISA_VTS_API_KEY / SS      # VTS onboarding 获得
VISA_MLE_SERVER_CERT       # 自行生成（按 developer.visa.com/pages/encryption_guide）
VISA_MLE_PRIVATE_KEY       # 自行生成
USER_SIGNING_PRIVATE_KEY   # 自行生成（PKCS#8 RSA）
```

### 1.3 沙箱可调用 API

VIC MCP Server（`sandbox.mcp.visa.com`）提供以下工具：

| Tool | 说明 |
|------|------|
| `enroll-card` | 将支付卡注册为 VIC 购买指令凭证 |
| `initiate-purchase-instruction` | 创建 mandate（指令），含商户、金额、时间约束 |
| `update-purchase-instruction` | 修改 mandate |
| `cancel-purchase-instruction` | 取消 mandate |
| `retrieve-payment-credentials` | 获取结账凭证（PAR、loyalty、email 等） |
| `confirm-transaction-events` | 确认交易完成、发送商业信号 |

来源：[visa/ai apps/vic-mcp-examples/README.md](https://github.com/visa/ai/tree/main/apps/vic-mcp-examples)

### 1.4 Visa MCP Server / CLI 沙箱模式

- **Visa Documentation MCP Server** 端点：`https://sandbox.mcp.visa.com/mcp/doc`（提供 `get-docs` tool，读取集成文档）
- **Visa MCP Hub**：[mcp.visa.com](https://mcp.visa.com)（wiki 页面引用，未直接验证内容）
- **Visa CLI**：未发现独立 CLI 工具或其沙箱模式的官方仓库。[未验证]

### 1.5 Trusted Agent Protocol (TAP) — 本地可运行

| 项目 | 详情 |
|------|------|
| 仓库 | [github.com/visa/trusted-agent-protocol](https://github.com/visa/trusted-agent-protocol)（213 stars，2026-10-14 更新）|
| 是否需要 Visa 凭证 | **否**，本地 demo 完全自包含 |
| 运行方式 | `pip install -r requirements.txt`，启动 5 个服务：agent-registry (8001)、merchant-backend (8000)、cdn-proxy (3002)、merchant-frontend (3001)、tap-agent (8501, Streamlit) |
| 技术 | RFC 9421 HTTP 签名；agent-registry 维护公钥；cdn-proxy 在签名层验签；merchant-backend 处理已验证请求 |
| 演示内容 | 商户侧验签（区分 bot vs 合法 agent）；签名绑定域名+操作；防重放；PAR/loyalty 信息传递 |
| 适合 demo | **是**，可直接展示"merchant 如何验证 agent 身份"这一层 |

### 1.6 awslabs/agentcore-samples 中的 Visa 相关内容

经检索 [awslabs/agentcore-samples](https://github.com/awslabs/agentcore-samples) 全树，**未发现 visa/ 或 visa-b2b-account-payable-agent 路径**（搜索关键词：visa, mastercard, shopping-concierge-agent）。

`01-features/08-agents-that-transact/` 目录下的支付样例均基于**稳定币（x402 协议 + Coinbase CDP / Stripe Privy 嵌入式钱包）**，不涉及卡组织 API：
- Coinbase CDP（Base Sepolia testnet / 免费测试 USDC via faucet.circle.com）
- Stripe/Privy（嵌入式 MPC 钱包）
- x402 协议（HTTP 402 + on-chain USDC 支付）

---

## 2. Mastercard Agent Pay

### 2.1 沙箱访问

- `developer.mastercard.com/product/agent-pay`：访问时返回错误（SPA 渲染失败或页面已迁移）
- `developer.mastercard.com/documentation/agent-pay`：页面加载中，无法获取内容

**结论：未发现公开自助沙箱。** [未验证]

Mastercard Developer Center 的通常模式是注册账户后在项目中获取测试密钥，但 Agent Pay 是否已开放自助申请尚未确认。建议直接登录 [developer.mastercard.com](https://developer.mastercard.com) 搜索 "Agent Pay"。

### 2.2 Agentic Token 机制（从公开资料重建）

来源：[agenticcommerce.report/ai-agent-payment-apis](https://agenticcommerce.report/ai-agent-payment-apis/)（第三方分析报告，部分细节未验证）

| 特性 | 说明 |
|------|------|
| Token 类型 | Agentic Token：绑定用户、商户、交易上下文 |
| 生命周期 | 4 步：request → provision → authorize → confirm |
| 约束 | 可配置花费上限，用户/发卡行可撤销 |
| 集成路径 | 通过认证中间商（Stripe、Google Pay 等）或直连 MDES |
| AP4M | **Mastercard Agent Pay for Machines**，2026-06-10 正式推出，面向机器间支付（sub-cent 经济学）|
| 争议处理 | 单独争议轨道（MC 4849 reason code） |
| 地理覆盖 | 几乎所有 Mastercard 发卡行（理论上全球） |

### 2.3 Verifiable Intent

未找到开源参考实现。[未验证]

---

## 3. 商业层协议可运行参考实现

### 3.1 ACP（Agentic Commerce Protocol）

| 项目 | 详情 |
|------|------|
| 官方仓库 | [github.com/agentic-commerce-protocol/agentic-commerce-protocol](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol) |
| 维护方 | OpenAI + Stripe（beta 状态） |
| 最新稳定版 | 2026-04-17 |
| 仓库性质 | **纯规范**（OpenAPI + JSON Schema + examples JSON）；无独立可运行服务器 |
| 参考实现 | [docs.stripe.com/agentic-commerce](https://docs.stripe.com/agentic-commerce)（需 Stripe 账户）；[developers.openai.com/commerce](https://developers.openai.com/commerce)（需 OpenAI 账户）|
| 本地跑通 | **不能独立运行**：需要 Stripe 测试密钥或 OpenAI 商户集成才能测试真实 checkout session |
| 可运行替代 | NVIDIA-AI-Blueprints/Retail-Agentic-Commerce（同时实现 ACP + UCP，仅需 NVIDIA API Key）|
| 对接支付轨道 | 卡（通过 Stripe 处理；标准 Stripe 费率 ~1.9-2.9% + $0.30） |
| ACP 协议端点 | `POST /checkout_sessions/`，`POST /checkout_sessions/{id}/fulfill`，webhook 回调 |

### 3.2 AP2（Agent Payments Protocol，Google）

| 项目 | 详情 |
|------|------|
| 官方仓库 | [github.com/google-agentic-commerce/AP2](https://github.com/google-agentic-commerce/AP2)（3237 stars，2026-10-19 更新） |
| 是否本地可运行 | **是**，`uv` + `bash run.sh` |
| 必要凭证 | `GOOGLE_API_KEY`（从 [aistudio.google.com/apikey](https://aistudio.google.com/apikey) 免费获取）；或 Vertex AI 凭证（可选） |
| 是否依赖 GCP | **否**，Google API Key 模式无需 GCP 项目 |
| 是否可部署到 AgentCore Runtime | **可以**，AP2 samples 是标准 Python 服务，可容器化部署 |
| 卡支付场景 | `code/samples/python/scenarios/a2a/human-present/cards/`：DPAN（模拟令牌化卡）+ OTP 挑战（mock OTP `123`）。**无真实卡网络 API 调用**，完全模拟 |
| 人不在场场景 | `a2a/human-not-present/` 目录（暂未详查） |
| 框架 | Google ADK（Agent Development Kit），但协议本身与框架无关 |
| 传输层 | A2A（agent-to-agent）JSON-RPC；同时兼容 MCP 和 UCP |
| 对接支付轨道 | 协议无关；card sample 用 mock；实际部署可接 Google Pay 卡处理（~1.9-2.9% + $0.30） |
| AP2 概要 | AP2 overview.md 明确：本协议设计为 A2A / MCP / UCP 的扩展层，解决 agent 支付授权与问责问题 |

**运行步骤（5 分钟快速验证）：**
```bash
git clone https://github.com/google-agentic-commerce/AP2
cd AP2
export GOOGLE_API_KEY=your_key
bash code/samples/python/scenarios/a2a/human-present/cards/run.sh
# 打开 http://0.0.0.0:8000/dev-ui，选 shopping_agent
```

### 3.3 UCP（Universal Commerce Protocol）

| 项目 | 详情 |
|------|------|
| 官方规范 | 无独立 GitHub org；规范散见于 AP2 docs 和 NVIDIA 实现 |
| 最佳可运行参考 | [github.com/NVIDIA-AI-Blueprints/Retail-Agentic-Commerce](https://github.com/NVIDIA-AI-Blueprints/Retail-Agentic-Commerce)（81 stars，ACP + UCP 双协议实现，2026-10-15 更新）|
| 所需凭证 | NVIDIA API Key（[build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys) 免费注册）；**不需要 GPU**（默认使用公共 NVIDIA API 端点）|
| 协议端点 | `/.well-known/ucp`、`/.well-known/agent-card.json`、`/a2a`（A2A message/send） |
| 架构 | Merchant API (8000) + PSP Service (8001) + 4 个 NAT Agents (8002–8005) + Apps SDK MCP Server (2091) + Next.js Demo UI |
| 运行方式 | `docker compose up` 或 `./install.sh`（8 个服务全启动） |
| 非官方实现 | `nymrel/open-ucp`（npm/pip 中间件，0 stars，非官方）；`financedistrict-platform/prestashop-agentic-commerce`（PrestaShop + Prism payments）|
| 对接支付轨道 | 协议无关；NVIDIA 实现内置 PSP 服务（模拟支付，可接真实 PSP）|

---

## 4. 各协议与支付轨道对应关系

| 协议/标准 | 主要支付轨道 | 备注 |
|-----------|------------|------|
| Visa TAP | Visa 卡网络（标准交换经济学）| 商户层验签；TAP Agent 生成 RFC 9421 签名 |
| Visa VIC | Visa 卡网络（tokenized DPAN / PAR）| 购买指令 + 凭证取回 |
| Mastercard Agent Pay | Mastercard 卡网络 | Agentic Token；AP4M 面向机器间微支付 |
| ACP | 卡（via Stripe 或 OpenAI commerce）| 也可接其他支付 handler |
| AP2 | 协议无关；card sample 用模拟 DPAN | 实际可接 Google Pay 或任意 PSP |
| UCP（NVIDIA 实现）| 协议无关；内置模拟 PSP | 可替换为真实 PSP |
| x402（agentcore-samples）| 稳定币（USDC on Base Sepolia / Solana Devnet）| 链上 USDC；无 MV；faucet 可获测试币 |

---

## 5. 对 demo 多轨设计的建议

### 5.1 卡组织 Agent Token 能否进入 demo，在哪一层

**可以进入 demo，但分两个层次：**

**层次 A — 商户验签层（可立即使用，无需申请）：**  
使用 `github.com/visa/trusted-agent-protocol`。该仓库完全本地可运行，5 个服务，无 Visa 凭证要求。在 demo 中展示 "agent 出示 RFC 9421 签名 → CDN proxy 验签 → merchant backend 确认合法 agent" 这一流程。这是向商户演示"卡组织级信任框架"的最快路径，**今日即可运行**。

**层次 B — VIC 真实沙箱（需要申请，周期未知）：**  
真正调用 Visa MCP Server（`sandbox.mcp.visa.com`）进行 enroll-card / initiate-purchase-instruction。需要通过 Visa Intelligent Commerce onboarding 获得 API 凭证，不是自助流程。建议并行申请，但不作为 demo 关键路径依赖。

**Mastercard Agent Pay**：开发者沙箱未确认公开开放，暂不纳入 demo 关键路径；可作为架构图中的"卡轨道 B"存在，真实集成依赖合作关系。

### 5.2 商业层协议选哪一个最适合演示"agent 下单 → 多轨支付"

**推荐：AP2（首选）+ NVIDIA Retail-Agentic-Commerce（补充 UCP 层）**

理由：
1. AP2 可在 10 分钟内本地跑通 end-to-end（Google API Key 即可），有完整的卡支付 mock 场景和 A2A 协议日志，演示效果最直观
2. AP2 协议层与 Visa TAP 不冲突：AP2 定义支付委托流程，TAP 定义 agent 身份验证；两者组合可以展示"协议协商 + 身份验证 + 支付执行"完整链路
3. NVIDIA 蓝图同时实现 ACP + UCP，可作为商户侧展示（需 NVIDIA API Key + Docker）
4. ACP（Stripe 实现）适合已有 Stripe 账户时直接用沙箱演示；如果演示中已有 Stripe 集成，ACP 是最低摩擦路径

**不推荐 ACP 作为独立 demo 起点**：spec-only，无独立可运行服务器，必须依赖 Stripe 或 OpenAI。

### 5.3 需要申请什么、预计多久

| 需申请项 | 入口 | 预计周期 | 优先级 |
|---------|------|---------|-------|
| Google API Key（AP2 运行用）| [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | **5 分钟**（自助）| 立即办理 |
| NVIDIA API Key（NVIDIA blueprint 用）| [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys) | **5 分钟**（自助）| 立即办理 |
| Visa Intelligent Commerce 访问权限 | [developer.visa.com](https://developer.visa.com) 注册后 request access | **数周到数月**（partner onboarding）| 并行申请 |
| Stripe 测试账户（ACP 用）| [dashboard.stripe.com/register](https://dashboard.stripe.com/register) | **10 分钟**（自助）| 按需 |
| Mastercard Agent Pay 沙箱 | [developer.mastercard.com](https://developer.mastercard.com) 注册，搜 Agent Pay | **未知**（可能需 partner 关系）| 并行申请后评估 |
| Coinbase CDP 账户（x402/稳定币轨道）| [portal.cdp.coinbase.com](https://portal.cdp.coinbase.com) | **30 分钟**（自助）| 稳定币轨道需要 |

### 5.4 建议的 demo 多轨架构

```
用户 / 商户
    │
    ├── 轨道 A：Visa TAP（商户验签层）
    │   └── visa/trusted-agent-protocol 本地运行 ← 今日即可
    │
    ├── 轨道 B：AP2 卡支付（agent 委托下单）
    │   └── AP2 human-present/cards scenario ← 今日即可（mock DPAN）
    │
    ├── 轨道 C：x402 稳定币（AgentCore 原生）
    │   └── agentcore-samples pay-for-x402-secure-data ← 需 Coinbase CDP
    │
    └── 轨道 D（未来）：Visa VIC 真实沙箱 / Mastercard AP4M
        └── 待 onboarding 完成后接入
```

---

## 参考链接汇总

| 资源 | URL |
|------|-----|
| Visa AI 工具包 | https://github.com/visa/ai |
| Visa Trusted Agent Protocol | https://github.com/visa/trusted-agent-protocol |
| Visa Developer Center | https://developer.visa.com |
| Visa MCP Hub | https://mcp.visa.com |
| VIC .env.example（凭证清单）| https://raw.githubusercontent.com/visa/ai/main/apps/vic-mcp-examples/.env.example |
| ACP 规范仓库 | https://github.com/agentic-commerce-protocol/agentic-commerce-protocol |
| ACP Stripe 实现 | https://docs.stripe.com/agentic-commerce |
| AP2 仓库 | https://github.com/google-agentic-commerce/AP2 |
| AP2 human-present/cards README | https://github.com/google-agentic-commerce/AP2/blob/main/code/samples/python/scenarios/a2a/human-present/cards/README.md |
| NVIDIA ACP+UCP 蓝图 | https://github.com/NVIDIA-AI-Blueprints/Retail-Agentic-Commerce |
| agentcore-samples 支付教程 | https://github.com/awslabs/agentcore-samples/tree/main/01-features/08-agents-that-transact |
| Mastercard Agent Pay（新闻）| https://www.mastercard.com/news/press/2025/april/mastercard-unveils-agent-pay |
| 第三方 API 对比分析 | https://agenticcommerce.report/ai-agent-payment-apis/ |
