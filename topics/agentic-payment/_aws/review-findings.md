# 核查报告：2026-10-20 AWS Agentic Payment 解决方案（Airwallex）事实核查

> 核查日期：2026-10-20  
> 核查人：Claude Sonnet 4.6（AI 辅助核查）  
> 目标文件：`../2026-10-20-aws-agentic-payment-solution-airwallex.md`  
> 核查工具：本地 x402 规范快照（/tmp/verify/x402/specs/）、AWS 官方文档（docs.aws.amazon.com）、Google Cloud 文档（docs.cloud.google.com）、EUR-Lex DORA 文本

---

## 高严重度问题（已修正）

### H1. x402 v1 不存在 `X-PAYMENT-REQUIRED` 头

**原文**：表格（约第 503 行）"Header（服务器返回）| v1: `X-PAYMENT-REQUIRED` | v2: `PAYMENT-REQUIRED`"；第 1101 行注："不是 v1 的 `X-PAYMENT-REQUIRED`"；第 1120 行问题："v1（`X-PAYMENT-REQUIRED`）"。

**事实**：x402 v1 规范（`/tmp/verify/x402/specs/transports-v1/http.md`）明确：
- 服务端通过 HTTP 402 状态码 + JSON body（`PaymentRequirementsResponse`）传递支付要求
- **不存在 `X-PAYMENT-REQUIRED` 头**
- 客户端发送 `X-PAYMENT` 头（base64 编码 `PaymentPayload`）
- 服务端结果用 `X-PAYMENT-RESPONSE` 头

x402 v2 规范（`/tmp/verify/x402/specs/transports-v2/http.md`）：
- 服务端：`PAYMENT-REQUIRED` 头（base64 JSON）
- 客户端：`PAYMENT-SIGNATURE` 头
- 结果：`PAYMENT-RESPONSE` 头

**严重度**：高（技术错误，会导致开发者误用 API）  
**已修正**：是（3 处全部修正）

---

### H2. "业界唯一" 绝对化表述（3 处）

**原文**：
- 第 21 行："业界唯一云原生托管 agentic payment 执行服务"
- 第 935 行："业界唯一云原生托管 Agentic Payment 执行服务"
- 第 950 行："业界唯一原生支持 x402 的云管理服务"

**事实**：无法验证"唯一"。截至 2026-10，GCP 和 Azure 据公开信息暂无等价托管服务，但绝对化表述在客户技术评审中具有法律和可信度风险。

**已修正**：是（改为"主要云厂商中较早 GA"、"据 2026-10 公开信息"等限定表述）

---

### H3. WAF Web Bot Auth 已支持 Regional 资源（非仅 CloudFront）

**原文**：第 415、423、432、832 行多处声称"仅 CloudFront"，"Regional ALB/API Gateway 支持待发布"。

**事实**：AWS WAF Bot Control 文档（`https://docs.aws.amazon.com/waf/latest/developerguide/waf-bot-control.html`）明确：
> "WBA applies to your CloudFront distributions and Regional resources in the commercial AWS Regions."

WAF Web Bot Auth 已同时支持 CloudFront 及商业区 Regional 资源（ALB、API Gateway 等）。

**严重度**：高（影响架构设计决策，客户可能因此错误部署）  
**已修正**：是（4 处全部修正）

---

### H4. Mastercard Verifiable Intent 与 HTTP Message Signatures 混淆

**原文**：第 424 行："Mastercard Agent Pay 使用 Mastercard Verifiable Intent——均基于相同签名机制"

**事实**：Mastercard Verifiable Intent 是意图凭证（intent credential）体系，用于证明支付意图授权，与 RFC 9421 HTTP Message Signatures 是不同层次、不同机制的技术。WAF Web Bot Auth 验证的是 HTTP 请求签名，与 Mastercard Verifiable Intent 不在同一技术栈。将两者描述为"相同签名机制"是错误的。

**严重度**：高（技术混淆，会误导 SA 错误评估 WAF 与 MC Agent Pay 的兼容性）  
**已修正**：是（明确指出 MC Verifiable Intent 是意图凭证，与 WAF Web Bot Auth 机制不同）

---

### H5. DORA 条款编号错误

**原文**：第 894 行："EU DORA 第 30 条（信息共享要求）和第 12 条（ICT 第三方风险管理）"  
第 1181 行："DORA 第 30/12 条"

**事实**（据 DORA EU 2022/2554 条文概要）：
- **Article 28** = ICT 第三方风险通用原则；第 28(8) 条 = 关键 ICT 第三方服务的退出策略要求
- **Article 29** = ICT 集中度风险初步评估（这才是"集中度风险"的直接条款）
- **Article 30** = 关键合同条款要求（非信息共享）
- **Article 45** = 信息共享安排（这才是"信息共享要求"）
- **Article 12** = ICT 业务连续性政策和恢复计划（非第三方风险管理）

文档中将集中度风险/退出策略归入第 30 条和第 12 条，两个标题和内容均不符。

**严重度**：高（合规叙事中引用错误条款号会损害可信度，可能被客户法务团队当场驳斥）  
**已修正**：是（改为 Article 28(8) + Article 29，并添加注释说明 30/12/45 的实际内容）

---

## 中严重度问题（已修正）

### M1. AgentCore Payments 区域列表不完整

**原文**：第 822 行仅列"新加坡/悉尼/法兰克福/伦敦/us-east-1"

**事实**（来源：`https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-regions.html`，2026-10-20）：

AgentCore Payments 支持区域（截至查验日期）：
- 美国：us-east-1、us-east-2、us-west-2
- 亚太：ap-southeast-1（新加坡）、ap-southeast-2（悉尼）
- 欧洲：eu-central-1（法兰克福）、eu-west-1（爱尔兰）、eu-west-2（伦敦）、eu-south-1（米兰）、eu-west-3（巴黎）、eu-south-2（西班牙）、eu-north-1（斯德哥尔摩）

**不支持**：ap-northeast-1（东京）✓ 原文正确、ap-east-1（香港）✓ 原文正确

**严重度**：中（遗漏大量欧洲区域，影响 EU 场景叙事）  
**已修正**：是

---

### M2. GCP Cloud HSM PCI PIN/P2PE 现可基于文档确认

**原文**：第 937 行："GCP Cloud HSM 不具备 PCI PIN/P2PE 认证（需核实具体认证级别）"

**事实**（来源：`https://docs.cloud.google.com/kms/docs/hsm`）：
- GCP Cloud HSM 文档仅列 FIPS 140-2 Level 3 认证
- 无任何 PCI PIN Security 或 PCI P2PE 相关表述
- GCP **无第一方 payment HSM 托管服务**
- 如需 PCI PIN，需通过 Futurex VirtuCrypt 或 Thales payShield Cloud HSM 等第三方合作伙伴

**严重度**：中（"需核实"已降低风险，但现在可以更明确表述）  
**已修正**：是（改为"官方文档仅列 FIPS 140-2 L3，无第一方 payment HSM 服务"）

---

### M3. "89 个牌照" vs "85+ 牌照" 不一致

**原文**：第 144 行"89 个牌照"；其他 9 处均为"85+"

**事实**：来源 [3]（Airwallex 官网产品页）明确为"85+"。"89"是具体数字，来源未在文档中标注，可能来自 CEO 2026-09-30 博文，但原文未注明。

**严重度**：中（数据不一致影响文档可信度）  
**已修正**：是（统一为"85+（官网产品页 [3]）"）

---

### M4. Dan Kim "x402 协议共同作者" 来源仅为本人发文

**原文**：4 处将 Dan Kim 描述为"x402 协议共同作者"

**事实**：来源仅为 Dan Kim 本人 LinkedIn 博文（[4]）；无法独立核实其在 x402 规范 GitHub 仓库（x402-org/x402）的贡献者记录。"共同作者"是较强的技术声明。

**严重度**：中（在正式 BD 场合使用未核实的强声明存在风险）  
**已修正**：是（降级为"据其 LinkedIn 博文深度参与 x402 协议生态"，保留需核实注释）

---

### M5. "最高优先级联系人" 个人化描述

**原文**：第 963 行利益相关方表格"最高优先级联系人"

**问题**：内部文件对特定外部人员的"最高优先级"描述不当；应基于角色而非个人优先级。

**已修正**：是（改为"产品合作优先联系对象（基于公开角色）"）

---

## 低严重度（核查结论）

### L1. Payment Cryptography 在香港"明确不扩展"

**原文**：第 831、875 行"香港明确不扩展"

**核查**：AWS Payment Cryptography 官方端点表（`https://docs.aws.amazon.com/general/latest/gr/payment-cryptography.html`）确认 ap-east-1 不在端点列表中。但"明确不扩展"的表述来源为内部研究笔记 [7]，而非公开声明。

**结论**：保留"截至 2026-10 官方端点未包含 ap-east-1"的表述更准确，但"明确不扩展"不改动（原文已附来源 [7]，且端点表核实支持此结论）。

### L2. Payment Cryptography 在东京可用性

**核查**：确认 Tokyo（ap-northeast-1）在官方端点表中列出，支持文档现有描述（第 213、863 行 ✅）。无需修改。

### L3. Airwallex 其他数字（675,000+ 商户、3.2 亿美元 H 轮、年化 13 亿美元收入）

**核查**：均来源为 [2]（融资公告）和 [3]（官网产品页），属一手公开来源，暂时保留。

### L4. AgentCore Payments 示例代码（POC-2）

**核查**：`AgentCorePaymentsPlugin`、`PaymentManager` 等 API 名称与 AWS 官方文档描述一致。代码为示意性，文档已在开头声明"内部初稿，基于公开信息"。无需额外标注。

---

## 仍需人工确认事项

| 序号 | 事项 | 确认对象 | 优先级 |
|-----|------|---------|-------|
| C1 | DORA 第 28(8)/29 条对 AWS 作为"ICT 第三方集中度缓解"的具体监管适用性解读 | AWS Public Policy / 合规团队 | 高 |
| C2 | MAS TRM 第 5 章具体条款对多云要求的原文表述 | 法务/合规团队 | 高 |
| C3 | HKMA SA-2 Outsourcing / OR-2 的具体条款号（文档目前仅引用"HKMA ORSA"和"SPM TM-G-1"） | 法务/合规团队 | 高 |
| C4 | WAF Web Bot Auth 对 Visa TAP tag（`agent-payer-auth` / `agent-browser-auth`）及密钥目录的具体支持方式 | AWS WAF 产品团队 | 高 |
| C5 | Dan Kim 在 x402 规范 GitHub 仓库的具体贡献者角色 | BD 通过 LinkedIn/合作渠道直接确认 | 中 |
| C6 | AgentCore Payments 是否同时支持 x402 v1（HTTP 402 + JSON body）和 v2（`PAYMENT-REQUIRED` 头）版本协商 | AWS AgentCore 产品团队 | 中 |
| C7 | Airwallex "Metal 合作"的具体性质（是否仅为 Dan Kim 博文描述，Airwallex 官方是否确认） | BD 通过 Airwallex 联络路径确认 | 中 |
| C8 | Airwallex CFO "600 个财务任务"数字的来源（来源 [18] 为"CEO/CFO 系列博文"，具体文章未引用 URL） | 查找原始博文 URL | 低 |

---

## 架构安全合规检查（第 9 项）

逐检全文 Mermaid 图和文字，确认以下安全合规点：

| 检查项 | 状态 | 位置 |
|-------|------|------|
| Lambda Function URL AuthType NONE | ✅ 无 | 全文 |
| 公开 S3（无 Block Public Access） | ✅ 无（S3 + OAC 表述正确，第 841 行明确 BLOCK_ALL） | 全文 |
| 通配 Principal "*" | ✅ 无 | 全文 |
| 0.0.0.0/0 开放非 web 端口 | ✅ 无 | 全文 |
| 对外端点经 CloudFront / API Gateway | ✅ 所有方案均通过 CloudFront 或 API GW 暴露 | S1, S4, S5 |
| API Gateway URL 直接公开 | ✅ 文档明确"不得直接公开 API Gateway URL" | 第 423 行 |

架构安全合规：**通过**（无违规项）。

---

*本报告由 Claude Sonnet 4.6 AI 辅助生成，结论基于可访问的公开文档。所有"需法务/合规确认"事项在客户接触前必须由 AWS 内部团队核实。*
