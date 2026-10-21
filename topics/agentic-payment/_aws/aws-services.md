---
title: "AWS 服务能力事实底稿：Agentic Payment 方案"
date: 2026-10-20
author: research-journal
scope: AWS FSI SA — 跨境支付/金融科技（Airwallex 等 PSP 客户）
---

# AWS 服务能力事实底稿：Agentic Payment 方案

**撰写日期：** 2026-10-20  
**性质：** 一手来源事实底稿（每条带 URL）  
**受众：** AWS 金融行业 SA  
**⚠️ 区域风险：** 香港（ap-east-1）：Amazon Bedrock、AgentCore、Payment Cryptography 均**不可用**，且 Bedrock/AgentCore **仅计划中**（isBeingPlannedIn），Payment Cryptography **明确不扩展**（isNotExpandingIn）。

---

## 目录

1. [AgentCore Payments（核心）](#一agentcore-payments核心)
2. [AgentCore 其他组件](#二agentcore-其他组件)
3. [支付与安全基础设施](#三支付与安全基础设施)
4. [风控与数据](#四风控与数据)
5. [生成式 AI 与治理](#五生成式-ai-与治理)
6. [区块链与稳定币](#六区块链与稳定币)
7. [合规](#七合规)
8. [样例与参考架构](#八样例与参考架构)
9. [竞争对比（Google / Azure）](#九竞争对比google--azure)

---

## 一、AgentCore Payments（核心）

### 1.1 发布时间线

| 里程碑 | 日期 | URL |
|--------|------|-----|
| Amazon Bedrock AgentCore 平台 **preview** | 2025-07 | https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-bedrock-agentcore-preview/ |
| Amazon Bedrock AgentCore 平台 **GA**（9 个区域，含 VPC 支持、A2A 协议）| 2025-10 | https://aws.amazon.com/about-aws/whats-new/2025/10/amazon-bedrock-agentcore-available/ |
| AgentCore Payments **preview**（x402、MPP；与 Coinbase/Stripe 合作）| 2026-04 | https://aws.amazon.com/about-aws/whats-new/2026/04/amazon-bedrock-agentcore-payments-preview/ |
| Solv Labs 合作案例（可验证、可审计的 agent 支付）| 2026-08-13 | https://aws.amazon.com/blogs/machine-learning/pay-with-confidence-how-solv-labs-built-verifiable-auditable-agent-payments-on-amazon-bedrock-agentcore-payments/ |
| OpenClaw agents 官方教程（Pay with AgentCore payments）| 2026-08-18 | https://aws.amazon.com/blogs/machine-learning/build-openclaw-agents-that-transact-with-amazon-bedrock-agentcore-payments/ |
| AgentCore Payments **GA** | **2026-08-19** | https://aws.amazon.com/about-aws/whats-new/2026/08/bedrock-agentcore-payments-ga/ |

### 1.2 架构与概念

**来源：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments.html  
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments-how-it-works.html

#### 核心资源层次

```
PaymentManager（账户级，IAM/JWT 授权类型）
  └── PaymentConnector（外部钱包提供商集成）
        ├── CoinbaseCDP（Coinbase Developer Platform，Quick Create 或 Manual）
        └── StripePrivy（Stripe + Privy 嵌入式钱包，仅 Manual）
PaymentSession（会话级预算控制：maxSpendAmount + currency + expiry）
  └── PaymentInstrument（链上钱包地址，用户的链上钱包实例）
        └── ProcessPayment（单次支付执行 API）
```

#### 支付协议

| 协议 | 版本支持 | 挑战格式 | 凭证格式 |
|------|----------|----------|----------|
| **x402** | v1 + v2 | 响应体含 `cryptoX402` 对象 | `X-PAYMENT` 请求头 |
| **MPP（Machine Payments Protocol）** | GA | `WWW-Authenticate: Payment` 头 | `Authorization: Payment` 头 |

**x402 `upto` scheme（动态定价）**：用于 pay-per-inference 和动态定价场景。Merchant 不指定确切金额，而是声明一个上界（`upto`），Agent 在上界范围内签署支付。这是 GA 时新增特性，专为价格在执行前不确定的场景（如 LLM token 计量）设计。

#### 钱包所有权与托管

- **钱包私钥**：AgentCore **不持有**私钥——私钥在 Coinbase CDP 或 Stripe Privy 的钱包基础设施中托管
- **凭证存储**：Coinbase API Key 和 Stripe Privy 凭证通过 AgentCore Identity 存储在 **AWS Secrets Manager**，以 PaymentCredentialProvider ARN 引用，不暴露给 agent 推理层
- **用户所有权**：Payment Instrument 代表**终端用户**的钱包地址，用户通过 Coinbase-powered frontend 授权 agent 代其交易
- **资金结算**：以 USDC 稳定币在 **Base 网络**上结算（on-chain，可独立审计）

#### Coinbase Bazaar MCP Server

- 通过 AgentCore Gateway 接入，包含 **10,000+** pay-per-use x402 端点
- Agent 可在 Console 或 CLI 中浏览/搜索 x402-enabled 服务并添加为 Gateway Target
- 订阅需要：Coinbase Wallets for AgentCore Payments AWS Marketplace listing（免费订阅，用于合并账单）

### 1.3 API 与 SDK

**控制面 API（创建/管理资源）：**
- `CreatePaymentManager` / `UpdatePaymentManager` / `DeletePaymentManager`
- `CreatePaymentConnector` / `UpdatePaymentConnector` / `DeletePaymentConnector`
- API 参考：https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_CreatePaymentManager.html

**数据面 API（运行时支付执行）：**
- `CreatePaymentSession` — 创建预算会话
- `CreatePaymentInstrument` — 为终端用户创建钱包地址
- `ProcessPayment` — 执行单次支付（检查预算 → 签名 → 返回支付凭证）
- API 参考：https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_ProcessPayment.html

**SDK 与框架集成：**
- AWS CLI：`aws bedrock-agentcore process-payment`
- AWS SDK（Boto3/Python）
- AgentCore SDK（高层封装）
- AgentCore CLI（开发者体验）
- **Strands Agents**：`AgentCorePaymentsPlugin`——拦截 x402 请求、自动处理支付、带证明重试（参见快速入门）
- **LangGraph**：框架集成支持
- **AgentCore Payments Skill**（AWS agent toolkit `aws-agents` 插件，ClawHub 上发布）

**快速入门：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments-getting-started.html

### 1.4 支付限额与策略

- **Session-level 预算**：`maxSpendAmount`（最大金额）+ `currency` + `expiry`（过期时间）
- **基础设施层强制执行**：spending limit **在 agent 推理层之外**强制执行（deterministically，不可绕过）
- **并发保护**：多 agent 并发对同一 session 支付时，预算不会超额（原子性检查）
- **失败回滚**：若签名失败，预算不扣减，transaction 记为 FAILED
- **限额粒度**：可按用户（user_id 参数传入 ProcessPayment）和 session 两级控制
- **⚠️ Human approval gap**：AgentCore Payments 当前**没有内置事中人工审批**——预防靠上限，证据靠 Observability；但可通过 Strands hooks（`BeforeToolCallEvent` + `interrupt()`）在 agent runtime 层实现支付前人工批准（参见 Healthcare HITL 博客）

**Policy in AgentCore（Cedar）**：可在 Gateway 层拦截工具调用，实现更细粒度的访问控制（例如"仅允许支付给白名单地址"），见第二章 2.4 节。

### 1.5 与 AgentCore 其他组件的关系

| 组件 | 在支付场景中的作用 |
|------|------------------|
| **Identity** | 存储 PaymentCredentialProvider（Coinbase/Stripe 凭证），`GetResourcePaymentToken` 在运行时检索令牌；Consent Portal 让用户授权 agent 代其交易 |
| **Gateway** | 连接 Coinbase Bazaar MCP server（10,000+ x402 端点目录）；可在 Gateway 拦截器 Lambda 中插入支付前验证逻辑 |
| **Observability** | 每次 ProcessPayment 调用自动产生 vended logs（CloudWatch）和 vended spans（X-Ray）；支持交易成功率监控、花费追踪、错误诊断 |
| **Browser** | 配合 AgentCore Payments 使 agent 可访问设有 x402 paywall 的网站（Browser Tool + Playwright 拦截 402 响应后调用 ProcessPayment） |
| **Policy（Cedar）** | 可配置"仅允许访问已知商户"、"单次最大金额"等细粒度规则，在 Gateway 层拦截 |
| **Runtime** | 提供 8 小时执行窗口和会话隔离，保证长时间 agent 支付循环的完整性 |

### 1.6 Human-in-the-Loop

AgentCore Payments **本身**不内置人工审批步骤（这是当前设计取舍：事前上限 + 事后可观测，中间批准需自行实现）。

**实现路径：**
1. **Strands hooks**：`BeforeToolCallEvent` + `interrupt()` 可在 agent 调用支付工具前暂停，等待人工确认（支持 approve/deny/trust-session）
2. **AgentCore Runtime 暂停/恢复**：Runtime 支持外部 callback 恢复执行（适合异步邮件/Slack 审批工作流）
3. **Lambda Durable Functions**：用于长时等待人工批准，暂停期间不计费

**参考：** https://aws.amazon.com/blogs/machine-learning/human-in-the-loop-constructs-for-agentic-workflows-in-healthcare-and-life-sciences/

### 1.7 定价

**来源：** https://aws.amazon.com/bedrock/agentcore/pricing/

- **AWS 不额外收费**：AgentCore Payments 仅透传底层第三方钱包操作费用
- **Coinbase CDP**：每次 `CreateInstrument` = 1 wallet operation；每次 `ProcessPayment` = 1 wallet operation（按 Coinbase CDP 官网定价）
- **Stripe Privy**：`CreateInstrument` 免费；每次 `ProcessPayment` = 1 wallet operation（按 Privy 官网定价）
- **无最低消费**，按量付费

### 1.8 区域可用性

**来源（官方区域表）：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html

| 区域 | AgentCore Payments GA |
|------|----------------------|
| us-east-1（弗吉尼亚北部） | ✅ 可用 |
| us-east-2（俄亥俄） | ✅ 可用 |
| us-west-2（俄勒冈） | ✅ 可用 |
| eu-central-1（法兰克福） | ✅ 可用 |
| eu-west-1（爱尔兰） | ✅ 可用 |
| eu-west-2（伦敦） | ✅ 可用 |
| eu-west-3（巴黎）、eu-south-1（米兰）、eu-south-2（西班牙）、eu-north-1（斯德哥尔摩）| ✅ 可用 |
| ap-southeast-1（新加坡） | ✅ 可用 |
| ap-southeast-2（悉尼） | ✅ 可用 |
| **ap-northeast-1（东京）** | ❌ **不可用** |
| **ap-northeast-2（首尔）** | ❌ **不可用** |
| **ap-south-1（孟买）** | ❌ **不可用** |
| **ap-east-1（香港）** | ❌ **不可用**（整个 AgentCore 计划中，Payment Cryptography 明确不扩展） |
| Canada（Central）、São Paulo | ❌ 不可用 |

**⚠️ 关键风险点**：
- **香港（ap-east-1）**：AgentCore 整体标记为 `isBeingPlannedIn`，无 ETA；银行监管要求数据驻留的客户无法使用
- **东京（ap-northeast-1）**：AgentCore Payments 不可用，但 AgentCore Runtime/Gateway/Identity 等核心组件可用
- **Preview 时区域**（us-east-1, us-west-2, eu-central-1, ap-southeast-2）→ GA 后已扩展到 12+ 区域

### 1.9 收款方（商户/卖方）能否用 AWS 基础设施收款

**对 PSP 客户（例如 Airwallex）来说，这是关键问题：如何将 API 或 MCP 工具变成 x402 付费端点来收款。**

答案是：**可以，但需要用 CloudFront + Lambda@Edge（或 AWS WAF AI Traffic Monetization）实现，不是通过 AgentCore Payments（AgentCore Payments 是买方侧服务）。**

**卖方侧参考架构：**
1. **CloudFront + Lambda@Edge x402**：在边缘拦截请求，检查是否携带有效 `X-PAYMENT` 头；若无则返回 HTTP 402（含 x402 payment payload）；收到支付证明后验证链上交易，放行原始请求
2. **AWS WAF AI Traffic Monetization**：在 CloudFront 分发上配置 WAF "Monetize" 规则，自动生成 402 响应和 x402 payment specification；验证支付后返回内容。**仅支持 CloudFront 分发**（Regional ALB/API GW 支持待发布）
3. **API Gateway + Lambda**：自定义 Authorizer 验证 x402 支付凭证，适用于非 CloudFront 场景

**官方参考实现（AWS 已发布）：**
- `x402 Content Monetization with Amazon CloudFront and AWS WAF`（含访问控制 + x402 验证）
- `Monetize Any HTTP Application with x402 and CloudFront + Lambda@Edge`

**来源：** https://aws.amazon.com/blogs/industries/x402-and-agentic-commerce-redefining-autonomous-payments-in-financial-services/  
https://docs.aws.amazon.com/waf/latest/developerguide/cloudfront-features.html

**对 PSP 客户的含义**：PSP 可以：
1. **作为发卡行/钱包提供商**：接入 AgentCore Payments 的 Connector（当前仅 Coinbase/Stripe；未来 AWS 计划开放更多支付提供商）
2. **作为收单行/API 提供商**：用 CloudFront + WAF AI Traffic Monetization 将自己的 API 暴露为 x402 付费端点，让 AgentCore Payments 用户支付
3. **将 MCP Server 变成付费端点**：通过 x402 协议为 MCP 工具收费，加入 Coinbase Bazaar MCP 目录

### 1.10 Solv Labs 案例要点

**来源：** https://aws.amazon.com/blogs/machine-learning/pay-with-confidence-how-solv-labs-built-verifiable-auditable-agent-payments-on-amazon-bedrock-agentcore-payments/

- **目标**：构建可被监管机构、审计方、交易对手独立验证的 agent 支付记录（不依赖算子的声誉）
- **四层技术栈**：
  1. **AgentCore Payments**：支付编排层（x402 执行）
  2. **Automated Reasoning Checks（ARc）**：形式化策略评估（正式逻辑验证支付决策合规性）
  3. **Nitro Enclaves**：硬件可信度证明（防止签名密钥被篡改）
  4. **x402**：已广泛采用的 agent 支付协议
- **核心架构特点**：Nitro Enclaves 提供 PCR 度量值，嵌入 KMS 密钥策略；任何不来自 Enclave 内的解密请求被 KMS 拒绝 → 签名证明链可被第三方独立验证
- **结论**：Solv Labs 认为这是"有治理的 agent 支付"（governed agent payments）从研究项目变为生产选择的第一个完整实现

---

## 二、AgentCore 其他组件

### 2.1 Runtime（运行时）

**来源：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/  
https://aws.amazon.com/about-aws/whats-new/2026/09/new-agentcore-runtime-generally-available/

| 能力 | 详情 |
|------|------|
| 执行窗口 | **8 小时**（行业领先，支持长时间 agent 工作流） |
| 会话隔离 | Firecracker microVM 级隔离（每个会话独立沙箱） |
| VPC 支持 | ✅ 支持，可连接私有网络资源 |
| A2A 协议 | ✅ 支持 Agent-to-Agent 通信（GA 时新增） |
| 部署方式 | 容器部署（Docker）+ 直接代码上传（快速原型，22 个区域） |
| 新版 Runtime V2（2026-09 GA）| Elastic memory（用多少分多少，reclaim 未用内存）；P75 冷启动 1.9-2.0 秒（vs V1 的 5.4-30 秒）；镜像 200MB-2GB 均适用；platformVersion 设为 `V2` 启用 |
| 双向流 | ✅ 支持（适合自然语言对话类 agent）|
| 框架支持 | 框架无关（Strands、LangGraph、LangChain、LlamaIndex、自定义等） |

**区域（Runtime Instances，全功能）**：us-east-1, us-east-2, us-west-2, eu-central-1, eu-west-1, ap-southeast-1, ap-southeast-2, ap-northeast-1, ap-south-1  
Runtime microVMs（全功能 microVM 隔离）：22 个区域（含以上所有）

### 2.2 Gateway（网关）

**来源：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html

| 能力 | 详情 |
|------|------|
| 协议转换 | MCP server → agent tool；OpenAPI spec → agent tool；Lambda → agent tool |
| 工具目录 | ✅ 支持（含 Coinbase Bazaar MCP 预置集成） |
| 拦截器 Lambda | ✅ 支持（在工具调用前后插入自定义逻辑：验证、审批、记录）|
| Cedar 策略 | ✅ 配合 Policy in AgentCore 实现实时工具调用控制 |
| Runtime 目标 | ✅ 支持将 AgentCore Runtime 作为 Gateway target（含 API schema 和 source enforcement）|
| 出站 OAuth | ✅ 通过 Identity 服务检索 OAuth token 访问外部受保护服务 |
| 入站认证 | ✅ SigV4（AWS IAM）或 JWT |
| 区域 | 所有 22 个 AgentCore 区域 |

### 2.3 Identity（身份）

**来源：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity.html

| 能力 | 详情 |
|------|------|
| Token Vault | 存储 OAuth refresh token，运行时自动刷新，凭证不暴露给模型 |
| 入站 OAuth | 验证进入 AgentCore Gateway 的 JWT；支持 Cognito 和自定义 IdP |
| 出站 OAuth | 代 agent 发起 OAuth 授权码流程，获取第三方服务（Google、Salesforce 等）的访问令牌 |
| Consent Portal | 托管授权页面（hosted portal）：用户通过 portalUrl 授权 agent 代其访问资源，完成后 Identity 自动存储访问令牌 |
| Workload Identity | 非用户代理场景下为 agent 负载颁发工作负载访问令牌（`GetWorkloadAccessToken`）|
| Custom Claims | ✅ 支持（2025-12 GA 新增）|
| 支付场景 | PaymentCredentialProvider 存储 Coinbase/Stripe 凭证；`GetResourcePaymentToken` 运行时检索；私钥/API Key 不暴露给 agent |
| 区域 | 所有 22 个 AgentCore 区域 |

### 2.4 Policy in AgentCore（Cedar 策略）

**来源：** https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-bedrock-agentcore-policy-evaluations-preview/

| 能力 | 详情 |
|------|------|
| 策略语言 | **Cedar**（Amazon 开源策略语言，用于 Verified Permissions 和 IAM Roles Anywhere）|
| 自然语言转 Cedar | ✅ 支持（团队用自然语言描述策略，系统转换为 Cedar）|
| 实时拦截 | ✅ 在 Gateway 层拦截每次工具调用，不降低延迟（策略评估在 Gateway 内同步执行）|
| 时间规则 | ✅ 可基于时间条件（交易窗口、工作时间等）控制工具访问 |
| Session 累计值 | ✅ 可基于 session 内累计消费额、调用次数等状态设置阈值 |
| 支付场景应用 | 限制可支付商户白名单、单次最大金额、每日累计上限、仅在特定时间窗口内允许支付 |
| Strands Box | Strands 开源库，底层使用相同 Cedar/Dogwood 技术，支持 per-tool 语义策略（"allow payments API but only up to $100/day"）|
| 区域 | 所有 22 个 AgentCore 区域 |

### 2.5 Memory（记忆）

**来源：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html

- **Episodic Memory（2025-12 新增）**：agent 跨会话学习和适应
- **Session Persistence（preview 2026-04）**：文件系统持久化，agent 可暂停/恢复
- **支付场景**：记录历史交易偏好、可信商户列表、用户预算模式
- **区域**：大多数 AgentCore 区域（参见官方区域表）

### 2.6 Observability（可观测）

**来源：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html

- OTEL 兼容（OpenTelemetry），CloudWatch 仪表板 + X-Ray traces
- **支付场景**：每次 ProcessPayment 自动产生 vended logs + vended spans；可监控交易成功率、花费分布、异常支付
- Vended logs：直接推送到客户 CloudWatch log group（无需额外配置）
- Vended spans：在 X-Ray 中可视化支付调用链（PaymentSession → ProcessPayment → 链上结算）
- 区域：所有 22 个区域

### 2.7 Evaluations（评估）

**来源：** https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-bedrock-agentcore-policy-evaluations-preview/

- 13 个内置评估器（覆盖质量维度）+ 自定义模型评分
- **区域**：us-east-1, us-west-2, ap-southeast-2, eu-central-1（初始）→ 2026 年扩展到所有区域
- 支付场景：评估 agent 是否正确解析支付指令、是否在预算内、是否触发了不必要的支付

### 2.8 Browser（云浏览器）

**来源：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/browser-tool.html  
https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments-browser.html

| 能力 | 详情 |
|------|------|
| 托管云浏览器 | VPC 连接的 Playwright 环境；内置会话录制 |
| x402 支付集成 | ✅ 通过 Playwright 响应拦截实现：agent 访问 x402 paywall 网站 → 拦截 402 响应 → 调用 ProcessPayment → 带支付证明重试 |
| Web Bot Auth 签名 | ❓ **文档未明确说明 AgentCore Browser 是否内置 Web Bot Auth（WBA）请求签名**；WBA 在 AWS WAF 侧（November 2025 GA，用于 CloudFront 验证来自 AgentCore microVM 共享 IP 的 agent 流量）；建议向 AWS 团队确认 Browser 工具是否自动签署 WBA 签名 |
| VPC 隔离 | ✅ 在 VPC 内运行，可访问内网资源 |

**Web Bot Authentication（WBA）— 来源：** https://aws.amazon.com/about-aws/whats-new/2025/11/aws-waf-web-bot-auth-support/  
https://aws.amazon.com/blogs/security/authenticate-legitimate-ai-agent-traffic-with-aws-waf-bot-control/

- WBA 在 **AWS WAF Bot Control** 中于 **2025-11 GA**（Rule Group Version_4.0+）
- 使用 IETF 草案（directory draft + protocol draft）实现非对称签名验证
- **当前仅支持 CloudFront 分发**（Regional ALB/API GW 支持 coming soon）
- 允许已验证 AI agent 流量通过，替代 IP/User-Agent 等可伪造方法
- 与 AgentCore microVM 场景相关：多租户 shared IP 空间下，WBA 是区分合法 agent 流量的机制

### 2.9 Code Interpreter

- 沙箱代码执行环境，支持 Python（适合数据处理、计算型 agent 工具）
- 区域：参见 AgentCore 区域表（Built-in Tools 类别，22 个区域）

### 2.10 Agent Registry

**来源：** https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html

- 区域：**us-east-1, us-west-2, eu-west-1, ap-southeast-2, ap-northeast-1** 仅以上 5 个区域
- 用于 agent 发现和 A2A 多 agent 协作
- 支付场景：注册支付专用 agent，让其他 agent 发现并委托支付操作


---

## 三、支付与安全基础设施

### 3.1 AWS Payment Cryptography

**来源：** https://aws.amazon.com/payment-cryptography/  
https://docs.aws.amazon.com/payment-cryptography/latest/userguide/  
https://aws.amazon.com/blogs/security/pci-pin-and-p2pe-compliance-packages-for-aws-payment-cryptography-are-now-available/

| 能力 | 详情 |
|------|------|
| 服务性质 | 全托管 Payment HSM 服务（PCI PTS 认证 HSM，AWS 负责全生命周期）|
| PCI PIN | ✅ KMCP（密钥管理）、KLCP（密钥装载）、DMCP（解密管理）认证 |
| PCI P2PE | ✅ DMCP、KMCP、KLCP 认证（含物理密钥交换 Physical Key Exchange 功能）|
| PCI DSS | ✅ PCI DSS 合规 |
| PCI 3DS | ✅ 支持 |
| 发卡场景 | ARQC 生成/验证（EMV 芯片卡交易授权）；PIN 验证值（PVV）生成/验证；PIN 翻译 |
| 密钥管理 | ANSI TR-31 属性强制执行；自动轮转/备份/恢复；DUKPT 支持 |
| 区域证明 | AS2805（澳大利亚 AusPay）、Cartes Bancaires（法国）已认证 |

**区域可用性（来源：aws-mcp get_regional_availability）：**

| 区域 | 状态 |
|------|------|
| us-east-1 | ✅ 可用 |
| ap-southeast-1（新加坡） | ✅ 可用 |
| ap-southeast-2（悉尼） | ✅ 可用 |
| ap-northeast-1（东京） | ✅ 可用 |
| eu-central-1（法兰克福） | ✅ 可用 |
| eu-west-2（伦敦） | ✅ 可用 |
| **ap-east-1（香港）** | ❌ **isNotExpandingIn**（明确不扩展，可能因地区 HSM 监管要求） |

**对 PSP 的意义**：可替代本地 HSM 基础设施；在澳洲（AusPay 认证）、法国（CB 认证）等市场无需自建 HSM；用于在 AWS 上实现完整发卡处理流程。

### 3.2 AWS KMS 和 CloudHSM

**KMS**：
- 用于加密 agent 凭证、支付会话数据、存储在 Secrets Manager 中的钱包 API Key
- 支持非对称密钥（RSA/ECC）用于签名验证
- 区域：99.9%+ 区域可用，包括香港（ap-east-1）

**CloudHSM**：
- 客户独占 FIPS 140-2 Level 3 HSM
- 比 KMS 更强的密钥隔离（客户完全控制）
- 适用于需要完全控制 HSM 的合规场景（如 MAS 对关键密码设备的要求）
- 区域：ap-east-1 ✅, ap-southeast-1 ✅, ap-southeast-2 ✅, ap-northeast-1 ✅

### 3.3 AWS Nitro Enclaves（钱包私钥与签名密钥保护）

**来源：** https://aws.amazon.com/blogs/web3/aws-nitro-enclaves-for-secure-blockchain-key-management-part-1/  
https://aws.amazon.com/blogs/web3/building-secure-verifiable-blockchain-key-management-on-aws-nitro-enclaves-at-turnkey/

| 能力 | 详情 |
|------|------|
| 隔离性 | 独立 CPU 和内存，无网络（仅 vsock 通信），无持久存储，无 SSH |
| 可信度证明 | 签名 EIF 产生 PCR 度量值（SHA384），嵌入 KMS 密钥策略条件键 |
| 私钥用法 | KMS 加密私钥存入 Secrets Manager → 传入 Enclave → `kmstool-enclave-cli` 解密（仅 Enclave 内成功）→ 签名 → 立即销毁明文 |
| 链上签名 | ✅ 支持 Ethereum/EVM 交易签名（go-ethereum，Geth）|
| 关键保证 | KMS 拒绝所有非 Enclave 来源的解密请求（PCR 条件键）；签名证明链可被第三方独立验证 |
| 合规场景 | Solv Labs 案例：ARc（形式化策略验证）+ Nitro Enclaves（硬件证明）构建可审计 agent 支付 |
| 生产案例 | Turnkey（MPC 密钥管理）、Solv Labs（agent 支付治理） |

**对 PSP 的意义**：可用于保护 agent 的支付私钥（wallet signing key）和内部密码操作密钥；满足 HKMA/MAS 对"关键密码系统"访问控制的要求。

### 3.4 AWS Secrets Manager 和 Private CA

**Secrets Manager**：
- AgentCore Identity 的底层存储层（支付凭证 PaymentCredentialProvider 存储在此）
- 自动轮转、细粒度访问控制、CloudTrail 审计
- 区域：全球可用，含香港

**AWS Private CA（Private Certificate Authority）**：
- 构建内部 PKI，用于 agent 间 mTLS 通信
- 支持短生命周期证书（适合 Workload Identity 场景）
- 区域：主要商业区域可用，含香港

### 3.5 AWS WAF Bot Control 与 Web Bot Auth（WBA）

**来源：** https://docs.aws.amazon.com/waf/latest/developerguide/waf-bot-control.html  
https://aws.amazon.com/about-aws/whats-new/2025/11/aws-waf-web-bot-auth-support/  
https://aws.amazon.com/blogs/security/authenticate-legitimate-ai-agent-traffic-with-aws-waf-bot-control/

| 能力 | 详情 |
|------|------|
| WBA GA 日期 | **2025-11** |
| 验证机制 | 非对称加密签名（IETF 草案：directory draft + protocol draft），防伪 |
| 当前作用域 | **仅 CloudFront 分发**（Regional ALB/API GW 支持 coming soon） |
| 规则版本要求 | `AWSManagedRulesBotControlRuleSet` Version_4.0+ |
| 已验证 agent 行为 | 自动允许（`Allowed: WBA-Verified`），替代传统 IP/User-Agent 过滤 |
| 费用 | 无额外费用（标准 WAF 收费） |
| 与 AgentCore 关系 | AgentCore microVM 共享 IP 空间，WBA 可区分同一 IP 下不同 agent 实体 |
| 商户侧价值 | 接受已验证 AI agent 流量，拒绝伪冒 agent；对 Visa TAP/Web Bot Auth 商户侧验证有直接对应 |

**AWS WAF AI Traffic Monetization**：
- 在 CloudFront 分发上配置 WAF "Monetize" 规则
- 自动为未付款请求返回 **HTTP 402** + x402 payment specification
- 支付后放行内容（200 响应经正常 CloudFront 管道）
- **仅 CloudFront**（不支持 ALB、API GW）

### 3.6 CloudFront + API Gateway（x402 商户端）

**PSP 客户将 API 变成 x402 付费端点的两种模式：**

**模式 A：CloudFront + Lambda@Edge**
```
Agent Request
   → CloudFront Distribution
      → Lambda@Edge Viewer Request: 检查 X-PAYMENT 头
         → 无支付：返回 402 + cryptoX402 payload
         → 有支付：验证链上交易（Base/Ethereum JSON-RPC）→ 放行到 Origin
```

**模式 B：API Gateway + Lambda Authorizer**
```
Agent Request
   → API Gateway
      → Lambda Authorizer: 验证 Authorization: Payment 头（MPP）或 X-PAYMENT 头（x402）
         → 验证通过：调用后端 Lambda/ECS
```

**来源：** https://aws.amazon.com/blogs/industries/x402-and-agentic-commerce-redefining-autonomous-payments-in-financial-services/

### 3.7 AWS Shield、Verified Access、Verified Permissions、IAM Roles Anywhere

**AWS Verified Permissions（Cedar）**：
- 与 Policy in AgentCore 使用相同底层 Cedar 语言
- 可用于应用层细粒度授权（如"用户 X 的 agent 每日最多支付 $Y"）
- 区域：所有商业区域

**AWS IAM Roles Anywhere**：
- 允许本地/混合负载使用短期 AWS 凭证（基于 X.509 证书）
- 适合跨境支付 PSP 在 on-premises 或非 AWS 基础设施上的 agent 安全认证

**AWS Shield（Standard/Advanced）**：
- 针对支付 API 端点的 DDoS 保护（Shield Advanced 24/7 响应团队 + WAF 集成）

### 3.8 Amazon Cognito（消费者授权）

- 托管用户池，支持 OAuth2 授权码流程
- 与 AgentCore Identity Consent Portal 集成（Identity Provider 配置）
- 适用于"消费者授权 agent 代其支付"场景的用户认证和同意管理

---

## 四、风控与数据

### 4.1 Amazon Fraud Detector 现状

**⚠️ 重要：Amazon Fraud Detector 自 2025-11-07 起不再接受新客户。**

**来源：** https://docs.aws.amazon.com/frauddetector/latest/ug/doc-history.html  
https://aws.amazon.com/fraud-detector/

> "Amazon Fraud Detector is no longer open to new customers as of November 7, 2025. For capabilities similar to Amazon Fraud Detector, explore Amazon SageMaker, AutoGluon, and AWS WAF."

**现有客户可继续使用**，但新 PSP 项目不可基于 Fraud Detector。

### 4.2 风控替代方案

**Amazon SageMaker**：
- 自训练 XGBoost/GNN 欺诈检测模型，部署为实时 Endpoint
- SageMaker Feature Store：实时特征存储，毫秒级特征读取（适合实时 agent 支付授权）
- SageMaker Model Monitor：检测模型漂移

**AutoGluon**（AWS 官方推荐）：
- AutoML 框架，自动训练欺诈检测模型
- 支持表格数据、时序数据、文本特征

**Amazon Neptune（图风控）**：
- 图数据库，用于账户关系网络分析（识别资金环流、共谋欺诈）
- Neptune Analytics（向量图分析）：用于实时图算法
- 适合 agent 支付场景：分析 agent 支付网络是否存在异常拓扑

**Amazon Bedrock（LLM 风控）**：
- 用于异常 agent 行为自然语言描述分析
- Bedrock Agents 可构建自解释的风控推理链（可审计）

**AWS WAF**（应用层保护）：
- 可识别和阻止针对支付 API 的攻击模式
- 与 Bot Control 协同：区分合法 AI agent 流量和恶意爬虫

### 4.3 实时流处理

| 服务 | 适用场景 | 区域 |
|------|----------|------|
| **Amazon Kinesis Data Streams** | 实时 agent 交易流入 + 风控规则引擎 | 全部主要区域含 ap-east-1 |
| **Amazon MSK（Managed Kafka）** | 高吞吐量支付事件流（适合跨境大规模 PSP）| 全部主要区域含 ap-east-1 |
| **Amazon Managed Service for Apache Flink** | 实时流式 SQL 风控（CEP 规则，毫秒级检测）| 主要区域 |
| **EventBridge Pipes** | 低代码事件路由（支付事件 → 风控 Lambda）| 主要区域 |

### 4.4 数据库与存储

**Aurora DSQL（账本/分布式交易）**：
- **来源：** https://aws.amazon.com/blogs/database/amazon-aurora-dsql-for-global-scale-financial-transactions/
- 全托管多区域 active-active，**强一致性**，PostgreSQL 兼容
- 无 Schema 迁移锁，IAM Token 认证（无静态密码）
- **适合**：跨区域账户账本、余额查询、资金转账
- **限制**：不支持引用完整性（应用层管理）；不支持自增主键（需 UUID）；不支持序列（`gen_random_uuid()`）
- 当前区域：us-east-1, us-east-2, us-west-2, eu-central-1（GA 扩展中）

**Amazon QLDB 现状**：
- AWS SDK v2 JavaScript 标注 deprecated；AWS CDK v1 Java package 标注 deprecated
- **官方替代**：Aurora DSQL（分布式强一致性账本场景）+ 应用层不可变日志模式（S3 + CloudTrail Lake）
- 现有 QLDB 用户可继续使用，但新项目应选 Aurora DSQL

**DynamoDB Global Tables**：
- 多区域活跃-活跃，Eventually Consistent（最终一致）
- 适用：Session 状态、会话锁定、支付去重（Idempotency Key 存储）
- **支付去重模式**：PaymentInstrument ID + timestamp → DynamoDB 原子写入（防重复支付）

**Amazon Aurora（PostgreSQL/MySQL）**：
- RDS Proxy + Aurora Serverless v2 → 弹性伸缩
- 适用：交易记录、对账数据（结构化，低并发）

**Amazon S3 + Athena/Iceberg**：
- 历史交易归档 + 合规报告
- Iceberg 格式支持 ACID 事务 + time-travel 查询（适合监管报告）

### 4.5 其他数据服务

**AWS Clean Rooms**：
- 与支付网络合作伙伴（如 Visa、Mastercard）在不共享原始数据的情况下进行联合风控分析
- 适合 PSP 与银行/卡网络合作，分析欺诈模式

**AWS Entity Resolution**：
- 跨数据集的实体解析（如将 agent 身份与已知欺诈者数据库匹配）
- 无需共享明文 PII

**AWS Security Lake**：
- 聚合 CloudTrail + VPC Flow Logs + GuardDuty + WAF 日志
- OCSF 格式，适合合规审计和安全分析（SIEM 集成）

**Amazon CloudTrail Lake**：
- 不可变 API 调用历史（包括所有 AgentCore Payments API 调用）
- SQL 查询接口，用于合规审计和支付争议解决
- 数据保留期：最长 7 年

---

## 五、生成式 AI 与治理

### 5.1 Amazon Bedrock 模型与区域可用性

**来源（区域）：** aws-mcp get_regional_availability → Amazon Bedrock

| 区域 | 状态 |
|------|------|
| us-east-1（弗吉尼亚北部） | ✅ 可用 |
| ap-southeast-1（新加坡） | ✅ 可用 |
| ap-southeast-2（悉尼） | ✅ 可用 |
| ap-northeast-1（东京） | ✅ 可用 |
| eu-central-1（法兰克福） | ✅ 可用 |
| eu-west-2（伦敦） | ✅ 可用 |
| **ap-east-1（香港）** | ❌ **isBeingPlannedIn**（无 ETA） |

**跨区域推理（Cross-region Inference）**：
- Asia Pacific 地理区域：推理路由覆盖 ap-northeast-1, ap-northeast-2, ap-northeast-3, ap-south-1, ap-south-2, ap-southeast-1, ap-southeast-2, ap-southeast-4
- 来源：https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/cross-region-inference.html

**主要模型（2026-10 可用）**：
- **Claude 系列**（Anthropic）：Claude 3.7/4.x 系列，支持 extended thinking
- **Nova 系列**（Amazon）：Nova Pro/Lite/Micro，低延迟，适合支付 agent 工具调用
- **Grok 4.6**（xAI）：500K context，configurable reasoning effort（low/medium/high/xhigh），适合长时 agent；2026-08 进入 Bedrock
- **Llama 3.x**（Meta）：开源模型
- **其他**：Mistral、Command R+ (Cohere)、Stable Diffusion 等

### 5.2 Bedrock Guardrails（LLM 安全防护）

**来源：** https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html

| 组件 | 详情 |
|------|------|
| **Content Filters** | 屏蔽有害内容、prompt injection 检测（已 GA）|
| **Topic Policies** | 禁止 agent 讨论 off-scope 话题（如禁止理财 agent 讨论竞争对手产品）|
| **Sensitive Information Redaction** | 自动脱敏 PII（卡号、SSN、账户号）|
| **Contextual Grounding** | 检查响应是否与知识库内容一致（防幻觉，适合合规 Q&A agent）|
| **Automated Reasoning Checks (ARc)** | 形式化逻辑验证（Solv Labs 案例核心组件）；**⚠️ 目前 Preview，仅 us-west-2（Oregon）；English only** |
| **Prompt Attack 检测** | Content Filters 内置 prompt injection 检测（GA，全部 Bedrock 区域）|

**支付场景应用**：ARc 可验证 agent 的支付决策是否符合形式化定义的合规规则（如"不得向制裁名单实体支付"）；这是 Solv Labs 案例中实现可验证合规的核心。

### 5.3 Amazon Q / Kiro（开发者体验）

**Amazon Q Developer（开发者助手）**：
- 支付代码生成（调用 AWS Payment Cryptography SDK、AgentCore SDK 等）
- 安全代码扫描（SCA）、IaC 生成

**Kiro（AI IDE）**：
- 2026 年发布，AgentCore Skills 在 Kiro 上首先支持（Claude Code、Codex、Cursor 跟随）
- 可用 AgentCore Payments Skill 脚手架支付资源

### 5.4 Strands Agents SDK

**来源：** https://strandsagents.com/  
https://aws.amazon.com/blogs/machine-learning/introducing-strands-agents-sdk/

- AWS 开源 agent 编排框架（Apache-2.0）
- `AgentCorePaymentsPlugin`：内置 x402 支付拦截器，3 行代码集成 AgentCore Payments
- Strands Box：语义 per-tool 策略执行（Cedar/Dogwood），支持"payments API 每日 $100 上限"规则
- A2A 协议支持（即将）
- 支持 Python（1.0 GA）和 TypeScript（1.0 GA）
- 与 LangGraph 等框架并列，均由 AgentCore Runtime 托管

---

## 六、区块链与稳定币

### 6.1 Amazon Managed Blockchain（AMB）

**来源：** https://docs.aws.amazon.com/managed-blockchain/latest/ethereum-dev/managed-blockchain-ethereum-overview.html

| 网络 | 功能 |
|------|------|
| **Ethereum Mainnet** | 单租户专属节点（dedicated），JSON-RPC API（Geth + Lighthouse）；用于 USDC/x402 交易验证 |
| **Ethereum Testnets** | Sepolia 等测试网；用于 x402 开发测试（`base-sepolia` 是 AgentCore Payments 快速入门默认网络）|
| **Bitcoin** | AMB Access Bitcoin（多租户 API）|
| **Hyperledger Fabric** | 私有联盟链；适合 PSP 内部账本或联盟网络 |
| **AMB Ordering Service** | 基于 QLDB 技术，有完整不可变事务历史 |

**注意**：Base 网络（Coinbase L2，AgentCore Payments USDC 结算网络）不是 AMB 直接支持的网络；Base 基于 Ethereum OP Stack，可通过标准 Ethereum JSON-RPC 接入，但 AMB 节点是以太坊主网节点，不是 Base 专用节点。

### 6.2 Coinbase CDP / x402 AWS 集成

**来源：** https://github.com/coinbase/cdp-agentcore-template  
GA 公告：https://aws.amazon.com/about-aws/whats-new/2026/08/bedrock-agentcore-payments-ga/

- Coinbase Developer Platform（CDP）是 AgentCore Payments 两个支持的钱包提供商之一
- `cdp-agentcore-template`（GitHub）：Coinbase 提供的前端模板，供 agent 开发者让用户 login + connect agent + onramp funds
- Coinbase x402 Bazaar MCP Server：10,000+ x402 端点目录，通过 AgentCore Gateway 接入
- AWS Marketplace：`Coinbase Wallets for AgentCore Payments`（免费订阅，合并账单）

### 6.3 QLDB 已停用，替代方案

- QLDB 不是区块链（官方明确），是集中化不可变账本
- AWS CDK v1 QLDB package 标注 `@Deprecated`；文档指向 Aurora DSQL 作为多区域强一致账本替代
- 对于不可变审计日志需求：CloudTrail Lake（不可变 API 日志）+ S3 Object Lock（防篡改存储）

---

## 七、合规

### 7.1 PCI DSS Level 1

**来源：** https://aws.amazon.com/compliance/pci-faqs/  
https://aws.amazon.com/compliance/services-in-scope/

- AWS 是 PCI DSS Level 1 服务提供商（最高级别，由 Coalfire QSA 评估，Visa/Mastercard 双方认可）
- 所有 PCI 范围内服务（全球所有区域）均处于合规环境
- **范围内服务清单**：https://aws.amazon.com/compliance/services-in-scope/（定期更新，可从 AWS Artifact 下载 AOC）
- **关键 PCI 服务**：EC2、S3、KMS、CloudHSM、Payment Cryptography、Lambda、API Gateway、RDS、DynamoDB、CloudTrail 等（完整列表见官网）
- **AgentCore Payments PCI 状态**：服务较新，建议确认当前 PCI 范围内状态

**AWS Payment Cryptography PCI 认证**：
- PCI PIN（KMCP + KLCP + DMCP）：已扩展到 São Paulo + Sydney
- PCI P2PE（DMCP + KMCP + KLCP）：已扩展
- AOC 报告通过 AWS Artifact 获取

### 7.2 亚太区域监管合规文件

**MAS TRM（新加坡金融管理局）**：
- AWS 发布了 MAS TRM 用户指南/Workbook，帮助金融机构满足 MAS Technology Risk Management 要求
- 来源：https://aws.amazon.com/compliance/mas-trm/

**HKMA（香港金融管理局）**：
- AWS 发布了 HKMA 合规 Workbook（包含 HKMA Supervisory Policy Manual TM-G-1 对应的控制措施）
- **⚠️ 重要**：Bedrock 和 AgentCore 在香港不可用，需要评估数据主权和推理在港需求是否可通过跨区域推理（新加坡/东京）满足
- 来源：https://aws.amazon.com/compliance/hkma/

**APRA CPS 234（澳大利亚审慎监管局）**：
- AWS 发布了 APRA CPS 234 Workbook（信息安全要求）
- Payment Cryptography 在 Sydney（ap-southeast-2）已覆盖 PCI + AusPay 认证

### 7.3 FSI Landing Zone

**Landing Zone Accelerator for Financial Services（LZA-FSI）**：
- 基于 AWS Control Tower 的多账户基础架构加速器
- 内置 FSI 相关安全控制和服务控制策略（SCP）
- 来源：https://aws.amazon.com/solutions/implementations/landing-zone-accelerator-on-aws/

**数据驻留**：
- **ap-east-1（香港）**：Bedrock/AgentCore 不可用，推理需路由到其他区域（可能违反数据驻留要求）
- **新加坡**：MAS 对数据本地化无强制要求（基于风险评估），但部分银行有内部政策
- **全区域 VPC Service Controls**：可用于限制数据不离开特定 VPC/区域

### 7.4 AWS Artifact 合规文件获取

- 所有 AOC（合规证明）、PCI 责任摘要、HKMA/MAS Workbook 均可通过 **AWS Artifact** 免费下载
- 地址：https://aws.amazon.com/artifact/


---

## 八、样例与参考架构

### 8.1 AWS GitHub 仓库

| 仓库 / 来源 | 内容 | URL |
|-------------|------|-----|
| `coinbase/cdp-agentcore-template` | Coinbase × AgentCore 前端模板：用户登录 + agent 连接 + 资金 onramp | https://github.com/coinbase/cdp-agentcore-template |
| AgentCore Payments 官方文档样例 | 快速入门（Coinbase/Stripe）、x402 Browser 集成、x402 exact/upto scheme 代码示例 | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/payments-getting-started.html |
| OpenClaw agents 博客 | ClawHub 插件安装 + 端到端 x402 支付实现（含 PaymentManager 配置代码）| https://aws.amazon.com/blogs/machine-learning/build-openclaw-agents-that-transact-with-amazon-bedrock-agentcore-payments/ |
| Solv Labs 可审计 agent 支付 | AgentCore + ARc + Nitro Enclaves 架构，FSI 监管合规参考 | https://aws.amazon.com/blogs/machine-learning/pay-with-confidence-how-solv-labs-built-verifiable-auditable-agent-payments-on-amazon-bedrock-agentcore-payments/ |
| x402 + CloudFront FSI 博客 | 买方侧（AgentCore）+ 卖方侧（CloudFront + Lambda@Edge）完整参考架构，针对金融数据提供商 | https://aws.amazon.com/blogs/industries/x402-and-agentic-commerce-redefining-autonomous-payments-in-financial-services/ |
| BlockRun + Incarna pay-per-inference | AgentCore Payments 用于 AI 模型推理按次计费案例 | https://aws.amazon.com/blogs/machine-learning/pay-per-inference-for-ai-agents-how-blockrun-and-incarna-use-amazon-bedrock-agentcore-payments/ |
| Ampersend pay-per-intelligence | 多模型路由统一支付渠道 | https://aws.amazon.com/blogs/machine-learning/building-pay-per-intelligence-for-ai-agents-how-ampersend-uses-amazon-bedrock-agentcore-payments/ |
| Technical deep dive: AgentCore Payments | 技术深挖博客（ProcessPayment API、支付流程、MPP 与 x402 对比） | https://aws.amazon.com/blogs/machine-learning/technical-deep-dive-agentcore-payments-and-innovation-in-agentic-commerce/ |
| Nitro Enclaves 区块链密钥管理（Part 1-3） | Ethereum 私钥在 Enclave 中签名完整指南 | https://aws.amazon.com/blogs/web3/aws-nitro-enclaves-for-secure-blockchain-key-management-part-1/ |
| Turnkey × Nitro Enclaves | 生产级 MPC 密钥管理实现 | https://aws.amazon.com/blogs/web3/building-secure-verifiable-blockchain-key-management-on-aws-nitro-enclaves-at-turnkey/ |
| AWS Payment Cryptography 样例（Python SDK）| ECDH PIN 保护流程、ARQC 验证、PIN Reset/SET 示例 | https://aws.amazon.com/blogs/industries/how-to-use-aws-payment-cryptography-service-to-simplify-payment-flows/ |
| Auth0 for AI Agents × AgentCore | OAuth2 + Identity token vault 身份安全架构 | https://aws.amazon.com/blogs/apn/securing-enterprise-ready-ai-agents-with-auth0-for-ai-agents-and-amazon-bedrock-agentcore/ |
| Aurora DSQL 全球金融交易博客 | 核心银行账本、跨区域 ACID 交易 SQL 示例 | https://aws.amazon.com/blogs/database/amazon-aurora-dsql-for-global-scale-financial-transactions/ |
| Human-in-the-loop（医疗/支付模式参考） | Strands hook HITL 实现、Runtime interrupt/resume 模式 | https://aws.amazon.com/blogs/machine-learning/human-in-the-loop-constructs-for-agentic-workflows-in-healthcare-and-life-sciences/ |
| WAF Bot Control + WBA | AI agent 流量 WBA 加密验证技术指南 | https://aws.amazon.com/blogs/security/authenticate-legitimate-ai-agent-traffic-with-aws-waf-bot-control/ |

### 8.2 合作伙伴公告

| 合作方 | 内容 | 状态 |
|--------|------|------|
| **Coinbase** | AgentCore Payments 核心钱包提供商；Coinbase CDP + Bazaar MCP（10,000+ 端点）；AWS Marketplace 上架 | ✅ GA（2026-08-19）|
| **Stripe（Privy）** | AgentCore Payments 第二个支持的钱包提供商（嵌入式稳定币钱包）| ✅ GA |
| **Solv Labs** | 可验证/可审计 agent 支付合作案例（AgentCore + ARc + Nitro Enclaves）| ✅ GA 博客（2026-08-13）|
| **BlockRun / Incarna** | Pay-per-inference 案例（AI 推理按次计费）| ✅ 博客（2026-10-05）|
| **Ampersend** | Pay-per-intelligence 多模型路由案例 | ✅ 博客 |
| **Auth0（Okta）** | AgentCore Identity + Auth0 for AI Agents 联合身份方案 | ✅ APN 博客 |
| **Visa Intelligent Commerce / TAP** | AWS 博客提及，Amazon 参与 Visa TAP 生态（商户侧 CloudFront 模式） | ✅ 生态关联 |
| **Mastercard Agent Pay** | 与 AWS 无直接官方 GA 合作公告；Microsoft Azure 是初始合作方 | ⚠️ 仅生态关联 |
| **x402 Foundation（Coinbase + Linux Foundation）** | x402 协议是 AgentCore Payments 核心协议之一 | ✅ 协议层集成 |

---

## 九、竞争对比（Google / Azure）

> **来源说明**：本节信息来自 Google Cloud 和 Microsoft 官方文档（部分通过 WebFetch 抓取），以及本 research journal 已核实的信息（见主综述文件）。标注了每条来源和可信度。

### 9.1 Google Cloud

**来源：** https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview（2026-10-15 更新）  
综述文件已核实：AP2（2025-09-16）、Verifiable Intent 捐赠 FIDO Alliance（2026-04-28）

| 能力 | 详情 | 与 AWS 对比 |
|------|------|------------|
| **Vertex AI Agent Engine（托管 agent 运行时）** | 全托管 agent 运行时，框架无关（ADK、A2A、LangChain、LangGraph、AG2、LlamaIndex、自定义）；Sessions + Memory Bank；内置 Code Execution + Computer Use；HIPAA、VPC SC、CMEK；私有服务接入（Private Service Connect） | ≈ AgentCore Runtime，但无内置 Payment 能力；Google 无 AgentCore Payments 等价物 |
| **AP2（Agent Payments Protocol）** | 开源支付协议规范（GitHub: google-agentic-commerce/ap2），2025-09-16 发布；Mandate 机制（Intent Mandate + Cart Mandate），JWT/SD-JWT 格式可验证凭证；60+ 合作方（Mastercard、PayPal、Coinbase、Adyen 等） | AP2 是协议规范，不是云托管服务；无 Google 托管的等价于 AgentCore Payments 的托管支付执行服务 |
| **AP2 支付轨道** | 卡（Mastercard、Amex、PayPal）、A2A 实时转账、x402 稳定币（通过 A2A x402 扩展，与 Coinbase 合作）；`PaymentInstrument.type` 是开放字符串（含 UPI 示例），路线图含 UPI/Pix | 协议层覆盖比 AgentCore Payments 广（含卡轨），但无托管执行基础设施 |
| **Verifiable Intent → FIDO Alliance** | Google + Mastercard 联合开发的可验证意图机制，2026-04-28 开源捐赠给 FIDO Alliance；成为行业标准候选 | AWS 未参与此联合开发，但 AgentCore Gateway 支持 JWT/OIDC 验证（可接入 FIDO 标准）|
| **UCP（Universal Commerce Protocol）** | Google、Shopify、Amazon、Microsoft 等 20+ 联合，2026 年发布；覆盖目录/购物车/结账/订单；AP2 作为安全支付子层 | Amazon 是 UCP 联合方之一 |
| **Google Wallet Agent 集成** | Google Pay 支持 agent 通过 AP2 发起支付 | Google Pay 作为支付通道，而非 AWS 那样的基础设施层 |
| **无托管支付基础设施** | Google 没有等价 AgentCore Payments 的云托管服务（钱包托管、spending limits、x402/MPP 执行）；需要开发者自行集成钱包 SDK | **AWS 的差异点：AgentCore Payments 是唯一同时提供协议执行 + 基础设施层 spending control + 可观测的托管服务** |
| **数据驻留** | Vertex AI Agent Engine：Memory Bank ML 处理在模型 endpoint 配置的区域内；大多数服务支持 VPC SC、CMEK；Example Store 不支持数据驻留 | AgentCore：香港不可用（isBeingPlannedIn）；类似挑战 |

**Google 在 agentic payment 上的核心定位**：协议制定者（AP2）+ 平台提供者（Vertex AI Agent Engine）；Google Pay 作为支付通道；**没有 AWS 级别的支付基础设施托管服务**。

---

### 9.2 Microsoft Azure

**来源：** https://azure.microsoft.com/en-us/products/ai-foundry/（2026-10 抓取）  
主综述文件已核实：Mastercard Agent Pay 合作（2025-04-29），Microsoft 是初始合作方之一  
Verifiable Intent 与 FIDO Alliance 捐赠（2026-04-28，Mastercard + Google 联合，Microsoft 参与）

| 能力 | 详情 | 与 AWS 对比 |
|------|------|------------|
| **Azure AI Foundry（前 Azure AI Studio）** | 全托管 AI agent 开发平台；11,000+ 模型；Foundry Agent Service（自主 agent）；Microsoft Agent Framework（开源 SDK，多 agent）；serverless 容器 + event-driven 部署；80,000+ 企业用户，Fortune 500 80% | ≈ AgentCore + Bedrock 合并；无内置 payment 能力 |
| **Mastercard Agent Pay 集成（2025-04-29）** | Microsoft 是 Mastercard Agent Pay 初始合作方之一；Azure AI agent 可通过 Mastercard 的 token 化机制发起卡支付授权；这是**卡网络轨道**（卡支付委托授权），不是稳定币/x402 | 与 AgentCore Payments（稳定币/x402）路线不同；卡支付委托授权在合规场景更有吸引力；AWS 尚无等价的 Visa/Mastercard 官方卡轨 agentic payment 托管集成 |
| **OpenAI + Stripe ACP** | Azure AI Foundry 深度集成 OpenAI（GPT-5 等）；ACP（Agentic Commerce Protocol）是 OpenAI + Stripe 合作的 agent 商业协议；Azure 是 ACP 的重要部署平台 | ACP 在 AWS 上也可部署（不绑定 Azure）；AWS 同样支持 Strands + ACP 集成 |
| **1400+ 业务系统连接器** | SAP、Salesforce、Dynamics 365 等；MCP 工具扩展；Permission-aware grounding（Microsoft Graph）| AgentCore Gateway 支持 MCP/OpenAPI/Lambda，但预置连接器生态不如 Azure 丰富 |
| **治理套件** | Entra（身份）+ Purview（数据治理）+ Defender（安全）统一控制平面；OpenTelemetry 可观测 | AgentCore Identity + CloudTrail + SecurityHub + GuardDuty 类似体系，但非 Azure 那样的单一控制台 |
| **无托管支付基础设施** | Azure 没有等价 AgentCore Payments 的托管 x402/MPP 执行服务；Mastercard Agent Pay 集成通过 Mastercard 的 SDK 实现，Azure 提供运行平台 | **AWS 差异点**：AgentCore Payments 是目前唯一云原生托管的 agent 支付执行服务（含钱包、spending limits、协议处理、可观测）|
| **香港/亚太数据驻留** | Azure 香港区域（East Asia）可用；Azure AI Foundry 香港可用性需单独核实 | AWS Bedrock + AgentCore 香港均不可用（isBeingPlannedIn）；**Azure 可能在香港场景有先发优势** |
| **Verifiable Intent 参与** | Microsoft 参与 Mastercard + Google 联合开发的 Verifiable Intent 机制，捐赠给 FIDO Alliance | AWS 未直接参与该工作组 |

**Azure 在 agentic payment 上的核心定位**：企业 AI 平台（Foundry）+ Mastercard 卡轨 agentic payment（Token 化委托）+ OpenAI 深度集成；**卡轨优势明显，稳定币/x402 生态弱于 AWS**。

---

### 9.3 三方竞争定位总结（BD 用）

| 维度 | AWS | Google Cloud | Microsoft Azure |
|------|-----|-------------|-----------------|
| **托管 agent 支付执行** | ✅ AgentCore Payments（GA 2026-08，x402+MPP，stablecoin）| ❌ 无（AP2 是协议规范，非云服务）| ❌ 无（Mastercard 集成在 SDK 层）|
| **稳定币/x402 生态** | ✅ 最深（Coinbase CDP + Bazaar 10,000+ 端点）| ✅ 协议层（AP2 x402 扩展，Coinbase 合作）| ⚠️ 有限 |
| **卡网络 agentic 集成** | ⚠️ 无官方 Visa/MC agentic card 托管集成 | ⚠️ AP2 合作方含 Mastercard（协议层）| ✅ Mastercard Agent Pay 初始合作方（卡轨）|
| **基础设施层 spending limits** | ✅ 确定性强制执行，model 层之外 | ❌ 无等价机制 | ❌ 无等价机制 |
| **PCI HSM（Payment Cryptography）** | ✅ GA，PCI PIN/P2PE，多区域 | ❌ 无等价服务 | ❌ 无等价服务 |
| **Nitro Enclaves（TEE 密钥保护）** | ✅ 生产案例（Solv Labs、Turnkey）| Google Confidential Computing（类似）| Azure Confidential Computing（类似）|
| **香港可用性** | ❌ Bedrock/AgentCore 不可用 | ⚠️ 需核实 | ✅ East Asia 区域可用 |
| **企业业务系统连接器** | ⚠️ 通过 Gateway MCP，预置连接器较少 | ⚠️ 类似 | ✅ 1400+ 预置连接器（Azure Foundry）|
| **合规文件**（MAS/HKMA/PCI）| ✅ 全套 Workbook + Artifact | ✅ 有但覆盖不及 AWS 全面 | ✅ 有 |
| **开源 agent 框架** | ✅ Strands Agents（Apache-2.0）| ✅ ADK、LangChain 等 | ✅ Microsoft Agent Framework |

**AWS 独特竞争优势**（对 PSP/跨境支付客户）：
1. **唯一云原生托管 agentic payment 执行服务**（AgentCore Payments GA，基础设施层 spending control）
2. **PCI PIN/P2PE 认证的托管 Payment HSM**（AWS Payment Cryptography）
3. **Nitro Enclaves 签名密钥保护**（生产案例 + 合规参考架构）
4. **10,000+ x402 端点目录**（Coinbase Bazaar MCP）
5. **完整的商户侧参考架构**（CloudFront + WAF AI Traffic Monetization）

**AWS 风险点**（对 PSP/跨境支付客户）：
1. **香港不可用**（ap-east-1 Bedrock/AgentCore isBeingPlannedIn；Payment Cryptography isNotExpandingIn）
2. **东京 AgentCore Payments 不可用**（ap-northeast-1 Payments: No）
3. **仅支持稳定币（USDC/Base）**，无原生卡轨 agentic payment（需通过 Stripe/Coinbase 桥接）
4. **ARc（Automated Reasoning Checks）仍 Preview，仅 us-west-2，English only**
5. **WBA（Web Bot Auth）仅 CloudFront**，Regional ALB/API GW 待支持
6. **Coinbase Marketplace 订阅要求**（仅影响 Coinbase 连接器）

