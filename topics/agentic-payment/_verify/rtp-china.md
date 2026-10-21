# 核实报告：实时支付轨道与中国生态（agentic payment 综述）

**核实日期：2026-10-19**
**对应综述文件：** `2026-10-19-agentic-payment-protocols-survey.md`（只读）
**核实范围：** B.6（中国生态，约第515–522行）、C.6.3（实时支付轨道，约第858–895行）

---

## 研究方法说明

- 使用 WebFetch 直接访问官方来源（npci.org.in、bcb.gov.br、frbservices.org、federalreserve.gov、theclearinghouse.org、docs.antom.com、mas.gov.sg、github.com/alipay 等）
- 5 个并行研究 agent 同时工作
- 中文官方来源（alipay.com、antgroup.com、pay.weixin.qq.com、cn.unionpay.com、ecny.cn、pcipss.org.cn、caict.ac.cn）多数因 JS 渲染或地理封锁无法直接抓取
- NPCI 官网 (npci.org.in) 全部页面均为 JS 渲染，WebFetch 只返回 shell HTML
- 媒体报道（TechCrunch、Wikipedia）作为线索，回溯官方时有标注

---

## 一、印度 UPI

### 核实结论

#### 1.1 NPCI + OpenAI/ChatGPT agentic payments 试点

**结论：部分核实（媒体来源充分，官方网页不可直接抓取）**

- **发布日期：2025年10月9日**
- **官方合作方：** NPCI + OpenAI + Razorpay（三方联合）；银行层：Axis Bank、Airtel Payments Bank；首批商户：BigBasket（生鲜杂货）、Vi（移动充值）
- **底层机制：** UPI Reserve Pay（资金预留）+ UPI Circle（代理授权）
- **来源：**
  - TechCrunch（2025-10-09）：https://techcrunch.com/2025/10/09/india-pilots-ai-chatbot-led-e-commerce-with-chatgpt-gemini-claude-in-the-mix/
  - Wikipedia UPI 词条（引用 CNBCTV18 "Razorpay, NPCI, OpenAI launch Agentic Payments"）
  - NPCI X 帖子 https://x.com/NPCI_NPCI/status/1976146268373024807（已引用，需登录）
  - TechCrunch（2026-06-27，NPCI CEO Dilip Asbe 访谈）：https://techcrunch.com/2026/06/27/indian-payments-chief-thinks-ai-will-be-heavily-involved-in-next-era-of-digital-payment-growth/
- **局限：** npci.org.in 所有页面均为 JS 渲染，无法直接抓取官方新闻稿正文；NPCI X 帖子需登录

#### 1.2 UPI Reserve Pay / single-block-multiple-debits

**结论：部分核实（机制已确认；RBI 官方名称为"single-block-and-multiple-debit"；NPCI 产品名"UPI Reserve Pay"在媒体中已确认）**

- **RBI 公告日期：2022年12月7日**（RBI 货币政策委员会第五次会议，发展与监管政策声明）
- **官方名称：** "single-block-and-multiple-debit functionality in UPI"（RBI 表述）
- **NPCI 产品品牌名：** "UPI Reserve Pay"（TechCrunch 2025-10-09 使用此名称，描述为"set aside funds for future debits by designated merchants"）
- **agentic 场景用途：** 资金预先锁定，后续由指定商户多次扣划，与 AP2 `agent_recurrence` 约束语义高度一致
- **来源：**
  - Wikipedia UPI 词条，引用 [46]（RBI Statement on Developmental and Regulatory Policies, MPC meeting December 5–7, 2022）
  - TechCrunch 2025-10-09（同上）
- **局限：** RBI 原始新闻稿未能直接抓取；npci.org.in 产品页不可访问

#### 1.3 Hello UPI / 对话式支付

**结论：未验证（仅见于媒体 URL，官方页面不可访问）**

- 证据：TechCrunch（2026-06-27）提及 NPCI CEO 称"2023年推出了语音助手式互动系统，但采用率未起来"；未使用"Hello UPI"名称。一个 News18 文章 URL 含"npci-launches-hello-upi-the-voice-assistant"字样
- **NPCI CoRover 对话式语音支付（不同产品）：** 2024年8月30日在全球金融科技节上发布，NPCI + IRCTC + CoRover/BharatGPT，用于铁路订票，这是一个已确认的独立产品
- 尝试途径：npci.org.in（JS 渲染，失败）；PIB（HTTP 403）；News18 原文链接未能获取

#### 1.4 UPI 在 Google Gemini / AP2 中的情况

**结论：AP2 协议本身已核实；Gemini-UPI 具体集成未验证**

- **Google AP2 宣布日期：2025年9月16日**（Google Cloud 博客，作者 Stavan Parikh VP/GM, Payments + Rao Surapaneni VP/GM, Business Applications）
  - 来源：https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol
  - 合作方 60+ 家，包括 Juspay（印度支付平台）；Juspay 在引用语中提到"our work with UPI"
  - AP2 文档（payment_mandate.md）已为 `"type": "UPI"` 预留 schema
- **Gemini-UPI PoC（未验证）：** Razorpay CEO Harshil Mathur（TechCrunch 2025-10-09）称"与 Google Gemini 和 Anthropic Claude 的 PoC 已完成，将在数周内落地"。截至 2026-06，NPCI CEO 称"尚未大规模推出"

#### 1.5 NPCI International agentic payment

**结论：未验证**

- npciinternational.com 无法连接；相关报道未提及 NIPL 的 agentic payment 动向

---

## 二、巴西 Pix

### 核实结论

#### 2.1 Pix Automático 上线日期与规则

**结论：已核实**（来源：BCB 官方页面 + 官方管理报告 PDF）

- **上线月份：2025年6月**（BCB 官方管理报告 PDF 原文："Até chegar à maturidade requerida para seu lançamento, em junho de 2025..."）
  - 来源 1：https://www.bcb.gov.br/estabilidadefinanceira/pix-automatico（BCB 产品页）
  - 来源 2：BCB "Relatório de Gestão do Pix 2023-2025" PDF，最后修改 2026-08-18
    URL：https://www.bcb.gov.br/content/estabilidadefinanceira/pix/relatorio_de_gestao_pix/relatorio_gestao_pix_2026.pdf
  - 监管依据：Instrução Normativa BCB nº 508/2024
- **主要规则：**
  - 用户一次性授权，后续无需每次操作；商户在到期日前数天发起扣款通知
  - 支持固定和可变金额；可设置单笔最高金额和使用信用额度规则
  - 适用场景：电费/电话费/学费/健身房/物业/订阅/保险等周期性账单
  - 对付款方免费；收款方可能被其 PSP 收费；跨行无转账费
  - 论坛征询：2023年6月、2024年8月
- **agentic 关联性：** 功能上接近 AP2 "HNP 场景"（预授权+自动触发），但 BCB 官方文件未使用"agentic"术语

#### 2.2 Pix por Aproximação 现状

**结论：已核实（已上线，2025年正式可用）**

- **技术标准发布：2024年8月**（BCB）
- **官方来源：**
  - https://www.bcb.gov.br/estabilidadefinanceira/pix-aproximacao
  - BCB "Faça mais com o Pix" 页面：https://www.bcb.gov.br/estabilidadefinanceira/mais-com-pix
- **关键规则：** NFC 近场通信；单笔默认最高 R$500（用户可下调）；需打开银行 App + 生物认证，或通过 Open Finance 预绑定的数字钱包

#### 2.3 Pix Garantido

**结论：未验证——此名称在 BCB 官方文件中不存在**

- BCB 官方管理报告 PDF（2023-2025）中搜索"Pix Garantido"：0 次出现
- BCB 网站搜索返回 404（"Conteúdo não existente"）
- BCB 官方产品列表（"Faça mais com o Pix"）中无此名称
- 结论：可能是媒体非正式用语、行业提案名称，或错误术语；不是 BCB 官方命名产品

#### 2.4 BCB/Mercado Pago/Nubank 的 agentic payment 官方公告

**结论：未验证**

- BCB 官方文件未出现"pagamento agêntico"等词汇；最接近的概念是"transferências inteligentes"（智能转账，基于 Open Finance API 的自动余额调配），但未以 AI/agentic 框架定义
- Nubank 和 Mercado Pago 未找到相关官方公告

#### 2.5 Pix 与 AP2/UCP 的联系

**结论：未验证**

- AP2 官方路线图（已在综述 C.6.1 中核实）确实提到 Pix，但 Google 未发布 Pix 参考实现
- 无 Google、Apple 或其他科技公司关于 Pix + agent payment 协议正式集成的官方公告

---

## 三、美国 FedNow/RTP

### 核实结论

#### 3.1 FedNow 交易限额

**结论：已核实**

- **当前限额：$10,000,000（1000万美元）**，2025年11月生效
- **历史沿革：**
  - 上线时（2022年1月公告）：最高 $500,000；参与机构默认 $100,000
  - 2025年6月24日：上调至 $1,000,000
    来源：https://www.frbservices.org/news/press-releases/062425-fednow-service-launches-new-risk-mitigation-features/
  - 2025年9月5日宣布，2025年11月生效：上调至 $10,000,000
    来源：https://www.frbservices.org/news/press-releases/090525-fednow-transaction-limit-increase/
    标题："FedNow® Service will raise transaction limit to $10 million to meet increased demand, unlocking higher-value use cases"
- **ISO 20022：** FedNow 从上线起即使用 ISO 20022；2025年9月公告中特别强调数据丰富有利于企业财务对账

#### 3.2 RTP 交易限额

**结论：已核实**

- **当前限额：$10,000,000（1000万美元）**，2025年2月9日生效
- **历史沿革：**
  - 2022年4月：从 $100,000 上调至 $1,000,000
  - 2024年12月4日宣布，2025年2月9日生效：上调至 $10,000,000
    来源：https://www.theclearinghouse.org/payment-systems/Articles/2024/12/Higher_10_Million_RTP_Network_Transaction_Limit_Empowers_New_Uses_12-04-2024
  - 2025年2月10日：BNY 完成首笔 $1000 万即时支付
    来源：https://www.theclearinghouse.org/payment-systems/Articles/2025/02/BNY_$10M_Payment_RTP-Network_02-10-2025
- **规模（2026 Q3）：** 1.5 亿笔、$6210 亿，日均 $69 亿

#### 3.3 美联储关于 AI Agent 的官方表述

**结论：已核实（多篇演讲，无 FedNow 专项政策）**

- 美联储理事 Christopher Waller 系列演讲（均已访问官方 URL）：
  - 2025-08-20，怀俄明区块链研讨会："Technological Advancements in Payments"，称 agentic AI 是"支付领域 AI 发展的下一波浪潮"，为"能自主规划和执行多步骤任务的系统"。
    来源：https://www.federalreserve.gov/newsevents/speech/waller20250820a.htm
  - 2025-09-29，Sibos 2025："The Next Frontier of Payments Innovation"，描述 AI agent"代表个人或企业快速廉价地执行任务"。
    来源：https://www.federalreserve.gov/newsevents/speech/waller20250929a.htm
  - 2025-10-15，DC Fintech Week："Innovation at the Speed of AI"，提到 agentic AI 代人缴账单。
    来源：https://www.federalreserve.gov/newsevents/speech/waller20251015a.htm
  - 2025-10-21，Fed 首届支付创新大会（Payments Innovation Conference）：将 AI 快速应用列为重塑支付格局的技术之一。
    来源：https://www.federalreserve.gov/newsevents/speech/waller20251021a.htm
- **未找到：** 专门针对 AI Agent 使用 FedNow 的官方政策文件或技术指南
- **注意：** frbservices.org 的"FedNow payment agents"文档指的是代理金融机构（On Behalf Of 中介），与 AI agent 无关

#### 3.4 TCH 关于 AI agent 使用 RTP 的官方表述

**结论：未找到**

- theclearinghouse.org 近期新闻稿聚焦代币化存款、稳定币监管、"链上货币倡议"（2026年6月、9月），未涉及 AI agent

---

## 四、中国生态

### 核实结论

#### 4.1 支付宝/蚂蚁：Antom MCP Server（国际品牌）

**结论：已核实（国际市场产品，非中国境内）**

- **Antom** 是蚂蚁集团旗下服务跨境/海外市场的支付品牌（原 Alipay+），与面向中国境内的"支付宝"分开运营
- **官方 MCP 文档：** https://docs.antom.com/ac/mcp/overview（页面时间戳：2026-01-06）
  - 官方声明：**"Antom is the first payment institution in China to support the MCP protocol"**（Antom 是中国首家支持 MCP 协议的支付机构）
  - 支持平台：Claude Code、ChatGPT、Cursor、Gemini CLI、Dify、Vercel、Wix
  - OAuth 端点：`https://mcp.antom.com`
- **官方 GitHub 仓库：** https://github.com/alipay/global-alipayplus-mcp（alipay 官方组织下）
  - 包名：`ant-intl-alipayplus-mcp`；支持 uvx 或 pip 安装
  - 工具：`create_payment`、`query_payment`、`cancel_payment`、`create_refund`、`customs_declare`、`query_customs_declare`
  - API 网关：`https://open-sea-global.alipay.com`；MIT 许可证；最后更新 2025年7月16日
- **MCP 工具 API 参考：** https://docs.antom.com/ac/ref/mcp（更新时间：2026-08-14）
  - 工具包括：`antom_inquire_merchant_profile`、`antom_query_transaction_detail`、`antom_diagnose_error`、`antom_inquire_diagnosis_result`、`antom_inquire_integration_logs`、`antom_fetch_bill_download_url`
  - 访问受白名单管控，需商户管理员开通；文档含提示注入风险警告
- **Antom AI 能力矩阵（同 docs.antom.com）：**
  1. Antom MCP Server（标准化 AI Agent 连接）
  2. Antom CLI（命令行工具，支持沙箱调用、诊断、报表查询）
  3. Antom Integration Skill（自然语言转集成代码）
  4. Antom Reconciliation Expert Skill（AI 驱动对账单检查）

#### 4.2 国内支付宝（opendocs.alipay.com）MCP 或 AI 支付

**结论：未验证**

- opendocs.alipay.com 为 JS 渲染页面，WebFetch 无法抓取有效内容
- GitHub alipay 组织的 98 个公开仓库中可见 10 个，无境内市场 MCP 仓库（注：其余 88 个仓库未全部可见）
- GitHub 搜索"alipay mcp"返回 16 个第三方社区仓库，官方仓库仅 global-alipayplus-mcp（国际）
- 尝试途径：opendocs.alipay.com（JS 渲染）；alipay.com（无有效内容）；antgroup.com（无有效内容）

#### 4.3 微信支付 MCP 或 agent 支付接口

**结论：未验证**

- pay.weixin.qq.com 页面内容为标准支付产品，无 MCP、AI 支付或 agentic payment 相关内容
- open.weixin.qq.com 有"小程序接入微信AI生态"测试版公告，但不含支付组件，无 MCP Server
- 未在 GitHub 或其他公开来源找到微信支付 MCP 仓库

#### 4.4 银联"智能体支付"或 AI 支付标准

**结论：未验证**

- cn.unionpay.com 主页无"智能体支付"、"AI 支付"或 agentic payment 相关内容
- 仅有面向商户和银行的"智能决策平台"（数据分析工具，非支付协议）
- 无官方公告

#### 4.5 数字人民币（e-CNY）智能合约与 agent 场景

**结论：未验证**

- ecny.cn 域名 DNS 解析失败（ENOTFOUND），无法访问
- pbc.gov.cn 主页加载但无 e-CNY 智能合约或 agent 相关公告
- 数字货币研究所网站未能抓取有效内容

#### 4.6 中国行业标准（中国支付清算协会、信通院）

**结论：未验证（网站不可访问）**

- pcipss.org.cn（中国支付清算协会）：DNS 解析失败
- caict.ac.cn（中国信息通信研究院）：返回 HTTP 412（地理封锁或机器人拦截）
- 未找到涉及"智能体支付"的团体标准或行业标准文件

---

## 五、其他亚洲

### 核实结论

#### 5.1 新加坡 MAS Project Orchid（目的绑定货币 PBM）

**结论：已核实（试点阶段，未正式上线；与 agent payments 的联系未验证）**

- **官方来源：** https://www.mas.gov.sg/schemes-and-initiatives/project-orchid（页面最后更新：2023年11月30日）
- **Project Orchid 性质：** MAS 多阶段探索项目，研究数字新加坡元（数字 SGD）的设计和技术可行性。MAS 明确表示"目前没有迫切的零售 CBDC 发行需求"
- **PBM（Purpose-Bound Money）：** 一种附带使用条件编程逻辑的数字货币协议，持有人可在无中间人情况下转移；2022年（第一阶段）和后续阶段已完成多个试点（政府代金券、商业代金券、CPF 养老金拨付、SkillsFuture 补贴等）
- **是否已正式上线：** 否，仍为试点/研究阶段
- **与 agent payments 的联系：** 官方页面无任何 AI agent 或 agentic payments 提及；规则执行逻辑（条件释放、托管）在技术上与 agent 场景兼容，但无官方表述

#### 5.2 PayNow Generation 2 与 agentic commerce

**结论：已核实（2026年6月报告，明确提及 agentic commerce）**

- **官方来源：** MAS + ABS（新加坡银行公会）联合发布
  - 报告 PDF：https://www.mas.gov.sg/-/media/PNG2-Phase-1-Collaterals_PDDC/PayNow-Gen2-Phase-1-Report_25-June-2026.pdf
  - 媒体通稿日期：2026年6月25日
- **报告明确引文（verbatim）：**
  > "AI agents will increasingly shop, schedule, and pay on behalf of consumers, driving demand for delegated, policy-based execution. Required functionality includes preauthorised spend rules covering merchant whitelists, transaction caps, daily and monthly limits, validity periods, and revocation."
  （AI 代理将越来越多地代表消费者购物、调度和支付，推动对委托式、策略执行的需求。所需功能包括：预授权消费规则，涵盖商户白名单、交易上限、每日/每月限额、有效期和撤销权）
- **行动承诺（Enhancement #4）：** MAS 和 ABS 将于 2026 年启动 agentic commerce 基础工作（Phase 2 评估可行性）
- **立场：** "agentic commerce 对新加坡而言是较长期的选项，将采取积极风险管理姿态"，确保创新伴随健全保障、治理和监管

#### 5.3 MAS SAFR（金融领域 AI Agent 安全保障框架）

**结论：已核实（2026年7月白皮书）**

- **官方来源：** https://www.mas.gov.sg/news/media-releases/2026/mas-partners-industry-to-develop-safeguards-for-ai-agents-in-finance
- **发布日期：2026年7月3日**
- **内容：** MAS 与金融机构和金融科技公司合作发布行业白皮书，提出 AI agent 在金融服务领域安全可靠运作的框架（BuildFin.ai 计划下，延伸自 Project Mindforge 的 AI 风险管理工具包）
- **与支付的关联：** 明确测试用例之一为"agent-assisted payments and treasury operations"（agent 辅助支付和财资操作）
- **关键保障机制：** 治理检查点、策略绑定执行、实时校验、可审计性、互操作性
- **后续：** Future of Finance Institute (FFI) 将通过试点和沙盒测试支持采纳

#### 5.4 日本

**结论：未验证**

- 日本银行（BOJ）CBDC 试点计划（2023年2月启动）：进展报告至 2026年6月，均未提及 AI agent 支付
- 日本金融厅（FSA）：2026年3月3日发布"AI Discussion Paper v1.1"，聚焦 AI 网络安全威胁，无 agentic payment 内容
- 未找到 JCB 或主要日本金融科技公司关于 AI agent 支付的官方公告

#### 5.5 韩国

**结论：未验证**

- 韩国银行（BOK）：尚未决定是否发行 CBDC，研究阶段，无 agentic payment 内容
- Naver：2026年9月发布"AI Shopping Agent"（推荐商品，生成订单），但无支付授权组件
- KakaoPay、Samsung Pay：未找到 agentic payment 官方公告

---

## 六、建议替换文本

### 6.1 替换 B.6「中国生态」（综述第 515–522 行）

**原文定位前 20 字：**「**支付宝/蚂蚁 AI 支付**：蚂蚁集团于 2025-2026 年在支付宝 App 中推出」

**建议替换段落（可直接粘贴）：**

```
#### 中国生态

**支付宝/蚂蚁——Antom MCP Server（国际品牌）（✅ 国际市场已核实）**

蚂蚁集团旗下国际支付品牌 **Antom**（原 Alipay+，面向跨境/海外市场）于 2026年1月正式发布 **Antom MCP Server**，是官方自称"中国首家支持 MCP 协议的支付机构"（来源：[docs.antom.com/ac/mcp/overview](https://docs.antom.com/ac/mcp/overview)，页面时间戳 2026-01-06）。

核心信息：
- **GitHub 官方仓库**：[alipay/global-alipayplus-mcp](https://github.com/alipay/global-alipayplus-mcp)，MIT 许可证，包名 `ant-intl-alipayplus-mcp`，最后更新 2025年7月16日
- **OAuth 端点**：`https://mcp.antom.com`；支持 Claude Code、ChatGPT、Cursor、Gemini CLI、Dify、Vercel、Wix
- **MCP 工具**：`create_payment`、`query_payment`、`cancel_payment`、`create_refund`、`customs_declare`、`query_customs_declare`（来源：[docs.antom.com/ac/ref/mcp](https://docs.antom.com/ac/ref/mcp)，更新时间 2026-08-14）
- **扩展 AI 能力**：Antom CLI（命令行）、Integration Skill（自然语言转集成代码）、Reconciliation Expert Skill（AI 对账）
- **访问控制**：白名单制，需商户管理员开通；文档包含提示注入风险提示

**注意**：Antom MCP Server 服务跨境商户，对应 Alipay+ 生态，而非中国境内的"支付宝"App。境内支付宝（opendocs.alipay.com）是否有等价的 MCP 能力，因官网全站 JS 渲染无法直接核实，**标注"未验证"**。

**微信支付（⚠️ 未验证）**

截至 2026-10，pay.weixin.qq.com 文档中未发现 MCP Server 或专用 agentic payment 接口；open.weixin.qq.com 有"小程序接入微信AI生态"测试版，但不含支付组件。尚无微信支付 agentic payment 协议的公开规范。

**银联（⚠️ 未验证）**

cn.unionpay.com 未发现"智能体支付"或 AI 支付标准；银联已有"智能决策平台"（商户数据分析），但这属于数据分析工具，非支付协议。无专项 agentic payment 公告。

**数字人民币 e-CNY（⚠️ 未验证）**

ecny.cn 域名无法访问（DNS 解析失败）；pbc.gov.cn 未发布 e-CNY 智能合约或 agent 场景说明。

**行业标准（⚠️ 未验证）**

中国支付清算协会（pcipss.org.cn，DNS 失败）和信通院（caict.ac.cn，HTTP 412）均无法访问，未能核实"智能体支付"相关团体标准。
```

---

### 6.2 替换/扩充 C.6.3「印度 UPI」段落

**原文定位前 20 字：**「**印度 UPI（⚠️ 部分未验证）**NPCI（国家支付公司）UPI 是全球最成熟」

**建议替换段落：**

```
**印度 UPI（⚠️ 部分核实）**

NPCI（国家支付公司）UPI 是全球最成熟的实时 A2A 支付系统（2023–24 年月均 120 亿笔以上）。

- **NPCI + OpenAI/ChatGPT agentic 试点（⚠️ 媒体来源充分，npci.org.in 官网 JS 渲染不可抓取）**：2025年10月9日，NPCI、OpenAI、Razorpay 联合启动试点，支持用户在 ChatGPT 内通过 UPI 完成购物支付；合作银行为 Axis Bank 和 Airtel Payments Bank；首批商户为 BigBasket（生鲜）和 Vi（移动充值）。底层机制为 UPI Reserve Pay（资金预留）+ UPI Circle（代理授权）。截至 2026年6月，NPCI CEO Dilip Asbe 确认试点存在，但称"尚未大规模推出"（来源：TechCrunch 2025-10-09；TechCrunch 2026-06-27）。
- **UPI Reserve Pay（single-block-and-multiple-debit）（⚠️ 机制已核实，品牌名媒体确认）**：RBI 于 2022年12月7日（MPC 会议）宣布"single-block-and-multiple-debit functionality in UPI"，允许用户将资金预锁定供指定商户分批划扣。NPCI 产品品牌名"UPI Reserve Pay"已在 2025-10-09 试点发布材料中使用，与 AP2 `agent_recurrence` 约束语义高度一致。
- **Gemini-UPI（⚠️ 未验证）**：Razorpay CEO（2025-10-09）称与 Google Gemini 的 PoC 已完成；无 Google 官方公告。
- **AP2 协议（✅ 已核实，见 C.6.1）**：AP2 于 2025-09-16 发布，合作方含 Juspay（印度），AP2 schema 已为 `"type": "UPI"` 预留。
- **Hello UPI（⚠️ 未验证）**：媒体 URL 可见"npci-launches-hello-upi"字样（约 2023 年），官方页面不可访问。不应与 2024年8月的 NPCI+IRCTC+CoRover/BharatGPT 对话式语音支付（独立产品）混淆。
```

---

### 6.3 替换/扩充 C.6.3「巴西 Pix」段落

**原文定位前 20 字：**「**巴西 Pix（⚠️ 部分未验证）**Banco Central do Brasil 的 Pix 是另一个成熟的即时支付系统。」

**建议替换段落：**

```
**巴西 Pix（✅ 主体已核实；⚠️ agentic 集成未验证）**

Banco Central do Brasil 的 Pix 是另一个成熟的即时支付系统（2026年5月月均超70亿笔，用户逾1.7亿）。

- **Pix Automático（✅ 已核实）**：BCB 于 2025年6月正式上线 Pix Automático，监管依据为 Instrução Normativa BCB nº 508/2024（来源：[bcb.gov.br/estabilidadefinanceira/pix-automatico](https://www.bcb.gov.br/estabilidadefinanceira/pix-automatico)；BCB 管理报告 PDF，最后修改 2026-08-18）。允许用户一次性预授权后，商户按周期自动发起扣款（固定或可变金额），适用于账单、订阅、保险等场景。功能上接近 AP2 HNP 场景所需的"预授权+自动触发"，但 BCB 官方文件未使用"agentic"框架。
- **Pix por Aproximação（✅ 已核实）**：BCB 于 2024年8月发布技术标准，2025年正式可用；基于 NFC，单笔默认最高 R$500（来源：[bcb.gov.br/estabilidadefinanceira/pix-aproximacao](https://www.bcb.gov.br/estabilidadefinanceira/pix-aproximacao)）。
- **Pix Garantido（⚠️ 未验证——此名称在 BCB 官方文件中不存在）**：BCB 官方管理报告（2023–2025）中无此名称；BCB 网站搜索返回 404。可能为媒体非正式用语，不应作为已核实术语引用。
- **AP2 路线图明确包含 Pix（✅ 见 C.6.1）**，但当前无 Pix 参考实现；无 BCB、Mercado Pago 或 Nubank 关于 agentic payments 的官方公告。
```

---

### 6.4 替换/扩充 C.6.3「美国 FedNow/RTP」段落

**原文定位前 20 字：**「**美国 FedNow/RTP（⚠️ agentic 集成未验证）**- FedNow（美联储，2023 年上线）」

**建议替换段落：**

```
**美国 FedNow/RTP（✅ 基础设施参数已核实；⚠️ agentic 集成无官方政策）**

- **FedNow 当前交易上限：$10,000,000**（2025年11月生效）。历史：上线时 $500K，2025年6月24日升至 $1M，2025年9月5日宣布升至 $10M（有效日期 2025年11月）。来源：[frbservices.org/news/press-releases/090525-fednow-transaction-limit-increase](https://www.frbservices.org/news/press-releases/090525-fednow-transaction-limit-increase/)。
- **RTP 当前交易上限：$10,000,000**（2025年2月9日生效）。2024年12月4日宣布，2025年2月10日 BNY 完成首笔 $1000 万即时支付验证。来源：[theclearinghouse.org](https://www.theclearinghouse.org/payment-systems/Articles/2024/12/Higher_10_Million_RTP_Network_Transaction_Limit_Empowers_New_Uses_12-04-2024)。
- FedNow 和 RTP 均使用 **ISO 20022** 报文（FedNow 从上线起，Fedwire Funds 于 2025年7月15日完成迁移）。x402 规范提及 `ach:us` 作为 CAIP-2 格式，但目前无 AP2×FedNow 参考实现。
- **美联储关于 AI agent 的官方表述（✅ 演讲层面）**：理事 Christopher Waller 在 2025年8–10月多次演讲中称 agentic AI 是支付领域"下一波浪潮"，指其"能自主规划和执行多步骤任务"（来源：federalreserve.gov/newsevents/speech/，2025-08-20、2025-09-29、2025-10-15、2025-10-21）。**但：** 无针对 AI agent 使用 FedNow/RTP 的官方政策文件或技术指南。
- **注意**：frbservices.org 的"FedNow payment agents"文档指的是"代理金融机构"（On Behalf Of 中介安排），与 AI agent 含义不同，不应混淆。
```

---

### 6.5 补充段落：新加坡（建议在 C.6.3 欧洲段落之后、中国段落之前新增）

**原文定位：** 建议插入在「**中国生态（⚠️ 均未验证）**」前一行

**建议新增段落：**

```
**新加坡（✅ 官方框架已核实；agent 集成为中期规划）**

- **MAS Project Orchid / PBM**：MAS 探索性项目，研究数字新加坡元（数字 SGD）设计，PBM（Purpose-Bound Money）是附加使用条件的数字货币协议。已完成多阶段试点（政府代金券、CPF 养老金、SkillsFuture 补贴等），但 MAS 明确表示"目前无迫切的零售 CBDC 发行需求"（来源：[mas.gov.sg/schemes-and-initiatives/project-orchid](https://www.mas.gov.sg/schemes-and-initiatives/project-orchid)，最后更新 2023-11-30）。PBM 与 agent 场景的官方连接：未见表述。
- **PayNow Generation 2 Phase 1 报告（✅ 明确提及 agentic commerce）**：MAS 与新加坡银行公会（ABS）于 2026年6月25日发布报告，明确将 agentic commerce 列为五大结构性变化之一，定义所需功能为"预授权消费规则，含商户白名单、交易上限、每日/月限额、有效期和撤销权"。MAS/ABS 承诺 2026 年内启动基础工作，Phase 2 评估可行性（来源：[PayNow Gen2 Phase 1 Report PDF](https://www.mas.gov.sg/-/media/PNG2-Phase-1-Collaterals_PDDC/PayNow-Gen2-Phase-1-Report_25-June-2026.pdf)）。当前立场：视为"较长期选项"，优先建立保障机制。
- **MAS SAFR（2026-07-03）**：MAS 与金融机构联合发布行业白皮书，提出金融领域 AI agent 安全框架，测试用例明确包含"agent-assisted payments and treasury operations"（来源：[mas.gov.sg/news/media-releases/2026/mas-partners-industry-to-develop-safeguards-for-ai-agents-in-finance](https://www.mas.gov.sg/news/media-releases/2026/mas-partners-industry-to-develop-safeguards-for-ai-agents-in-finance)）。
```

---

## 七、未验证事项汇总与尝试路径

| 事项 | 尝试途径 | 结果 |
|------|----------|------|
| NPCI 官方新闻稿（npci.org.in） | WebFetch，多个 URL | JS 渲染，只返回 shell HTML |
| NPCI X 帖子 | WebFetch | 需登录，无法访问 |
| RBI UPI 单次授权冻结原始新闻稿 | WebFetch（pib.gov.in，rbi.org.in） | HTTP 403 / 内容不匹配 |
| Hello UPI 官方页面 | WebFetch（npci.org.in/hello-upi） | JS 渲染，无内容 |
| Gemini-UPI 官方公告 | WebFetch（Google、NPCI、Razorpay 官网） | 未找到 |
| Pix Garantido BCB 官方文件 | WebFetch（bcb.gov.br），PDF 全文搜索 | 不存在 |
| 境内支付宝 MCP（opendocs.alipay.com） | WebFetch | JS 渲染，无内容 |
| WeChat Pay MCP | WebFetch（pay.weixin.qq.com） | 无相关内容 |
| 银联智能体支付 | WebFetch（cn.unionpay.com） | 无相关内容 |
| e-CNY 智能合约 | WebFetch（ecny.cn，pbc.gov.cn） | DNS 失败 / 无内容 |
| 中国支付清算协会标准 | WebFetch（pcipss.org.cn） | DNS 解析失败 |
| 信通院智能体支付标准 | WebFetch（caict.ac.cn） | HTTP 412 |
| NPCI International agentic | WebFetch（npciinternational.com） | 连接拒绝 |
| 日本 BOJ CBDC + AI agent | WebFetch（boj.or.jp） | 无相关内容 |
| 韩国 BOK + AI agent | WebFetch（bok.or.kr） | 无相关内容 |

---

*报告完成时间：2026-10-19*
*核实执行者：Claude Sonnet 4.6（automated research agents + direct WebFetch）*

---

## 补核（云浏览器）

**补核日期：2026-10-19**
**工具：Amazon Bedrock AgentCore 云浏览器（JS 渲染，可绕过部分简单反爬）**
**搜索引擎：Bing（www.bing.com/news/search，英文新闻模式有效）**

---

### A. 支付宝/蚂蚁——境内市场 MCP 与 agentic commerce

#### A.1 境内支付宝 MCP Server（opendocs.alipay.com，2025年4月前后）

**结论：⚠️ 2025年4月无独立 MCP 产品；2026年8月有大规模平台发布（见 A.2）**

- **opendocs.alipay.com 搜索"mcp"**：云浏览器渲染后可见，搜索结果页无任何 MCP 相关内容，仍为通用开发文档
- **GitHub alipay 组织搜索"mcp"**：只返回 1 个仓库——`global-alipayplus-mcp`（国际品牌 Antom），无境内 MCP 仓库
- **Bing 新闻"Alipay MCP server China 2025 April ModelScope"**：无结果
- **npm 搜索 `@alipay/mcp-server-alipay`**：Cloudflare 阻止访问
- **`mcp.alipay.com`**：超时（域名可能存在但对境外 IP 屏蔽）
- **结论**：综述中提到的"2025年4月前后支付宝 MCP Server 在魔搭/百炼上线"指向的是 **Antom（国际品牌）**的 MCP Server（docs.antom.com，2026-01-06），而非境内支付宝独立产品。境内版在此之前无公开 MCP 规范。

#### A.2 支付宝全栈 agentic commerce 平台（2026年8月）—— 新发现

**结论：✅ 已核实（媒体+官方公告，非官方主页直接抓取）**

- **发布日期：2026年8月17日**，支付宝在杭州举办首届 AI 生态大会
- **产品：** "中国首个全栈 Agent 商业基础设施" + **AHA（Agent Hub Access）协议套件**（多智能体跨端互联互通协议）
- **核心技术：** 支付宝 AI 开放平台可将商户现有页面、商品目录和服务流程转化为 **Skills 和 MCP 协议**，供各类 agent 调用
- **"阿宝"（Abao）agent**：2026年6月开始邀请测试；已有超过 1 万个服务完成 AI 适配，覆盖交通、餐饮、文旅、政务民生等 8 大类
- **AHA 协议三组件：** 智能交互标准 + agent 互联互通协议 + 设备感知与执行框架
- **合作方（20+ 家）：** Qwen（通义）、华为、OPPO、vivo、小米、荣耀、比亚迪、吉利、理想汽车、蔚来、阶跃星辰（StepFun）、Rokid
- **商户：** 麦当劳、蜜雪冰城、瑞幸咖啡、高德地图、滴滴、德邦快递、圆通速递、丰巢快递柜
- **Ant Group CEO 韩歆毅（Cyril Han）：** 称 agentic commerce 将在 6-12 个月内快速扩展
- **支付激励：** 推出"智能运营激励计划"，提供免费计算 token、降低支付费率、交易补贴，以及57项原子业务能力（会员体系、商家优惠券、芝麻信用、融资工具）免费开放
- **来源（媒体报道，原始官方公告内容来自 PR Newswire）：**
  - Crowdfund Insider（2026-08-17）：https://www.crowdfundinsider.com/2026/08/298068-alipay-unveils-full-stack-agentic-commerce-infrastructure-in-china/
  - Finextra（Cloudflare 阻止访问，但存在）：https://www.finextra.com/pressarticle/110650/alipay-launches-full-stack-agentic-commerce-platform-in-china

**说明：** 这是境内支付宝（非 Antom）正式进入 agentic commerce 领域的里程碑公告，产品中明确包含 MCP 协议支持，为综述中 B.6 节提供了关键更新依据。

---

### B. 微信支付——Skills 开发者工具（非支付 agent 协议）

**结论：✅ 确认存在"微信支付接入 Skill"，但性质为开发者助手，非 agentic payment 协议**

- **官方文档页：** https://pay.weixin.qq.com/doc/v3/merchant/4019638116（更新 2026-09-17）
- **GitHub 仓库：** https://github.com/wechatpay-apiv3/wechatpay-skills（367 ⭐，v1.3 更新于 2026-10-13）
- **安装命令：** `npx skills add https://github.com/wechatpay-apiv3/wechatpay-skills --yes`
- **功能：** 产品选型建议、示例代码检索（curl/Java/Go）、接入质量评估（安全/合规/资金链路）、答疑排障（含 CLI 工具查支付单/退款单）
- **适用场景：** 在支持 Skill 的 AI IDE（Cursor、Windsurf 等）中帮助**工程师集成微信支付**
- **本质区分：** 这是**开发者工具**，帮助开发者接入支付 API，而非让 AI agent 直接代用户发起/授权支付。与 Antom MCP Server（允许 agent 调用 `create_payment` 等工具）性质不同。
- **无 agentic payment 协议：** 微信支付文档中无 MCP 端点、无预授权 agent 支付接口。微信文档提供 `llms.txt` 和"复制给 LLM""问 AI"按钮，是文档 AI 友好化设计，非支付 agent 协议。

---

### C. 银联——最新进展

#### C.1 银联 × 华为：战略合作（已核实）

**结论：✅ 已核实（cn.unionpay.com 官网直接访问）**

- **日期：2026年5月7日**（深圳签署）
- **内容：** 联合发布《自主创新行动计划》与《人工智能行动计划》，聚焦 ICT 基础设施升级、AI 创新应用、终端支付及鸿蒙生态、国际化发展
- **原文引用：** "探索人工智能与金融科技的融合""以人工智能联合创新赋能金融业务智能化升级"
- **来源：** https://cn.unionpay.com/upowhtml/cn/templates/newInfo-nosub/7885004da382485e8bde5a0ba000fdd3/20260508181113.html
- **注意：** 文章**未出现"智能体支付"**词汇，属于 AI 应用整体战略，非专项 agentic payment 标准或产品公告

#### C.2 银联国际 × 连连数字：AI-agent 跨境支付合作（新发现）

**结论：✅ 已核实（PR Newswire 官方新闻稿，Yahoo Finance 转载）**

- **日期：2026年7月18日**签署合作协议，**2026年7月27日**对外公告
- **主体：** 连连数字（Lianlian DigiTech）+ **银联国际**（UnionPay International，非境内银联）
- **产品：** AI-agent 跨境支付，初期聚焦两个场景：
  1. **全球采购**（B2B 跨境支付，Human-in-the-Loop 模型）
  2. **AI Token 充值**（为 AI 服务补充 token 的自动化支付）
- **技术架构：** Human-in-the-Loop——AI agent 自主完成供应商匹配、商品筛选、生成支付订单，用户保留最终审批权
- **来源（官方 PR Newswire）：** https://finance.yahoo.com/technology/ai/articles/lianlian-digitech-unionpay-international-partner-003000728.html
- **重要说明：** 这是**银联国际**（跨境/海外）的合作，非境内银联的 agentic payment 标准公告

#### C.3 境内银联"智能体支付"标准——仍未验证

**结论：⚠️ 仍未验证**

- cn.unionpay.com 首页及新闻中心无"智能体支付"词汇
- pcipss.org.cn（中国支付清算协会）云浏览器仍 DNS 解析失败

---

### D. NPCI 印度——补核与新发现

#### D.1 NPCI + OpenAI + Razorpay 官方来源（Razorpay 博客）

**结论：✅ 官方合作方来源已核实（Razorpay 为联合发起方，非媒体报道）**

- **Razorpay 官方博客**（合作方官方发布）：
  - URL：https://razorpay.com/blog/razorpay-unveils-agentic-payments-on-chatgpt-with-npci-indias-first-ai-powered-conversational-payment-experience/
  - 日期：**2025年10月9日**
  - 内容：private beta 启动；底层机制为 NPCI 的 **UPI Circle + UPI Reserve Pay**；合作方 OpenAI、NPCI、Razorpay；银行 Axis Bank + Airtel Payments Bank；商户 BigBasket（Tata 旗下）+ Vi（Vodafone Idea 移动充值）
  - 原文关键引句："By leveraging new capabilities from NPCI—specifically UPI Circle and UPI Reserve Pay—the system allows users to securely pre-authorize their trusted AI agent to make purchases within pre-set, user-defined spending limits."
- **NPCI 官网（media/press-release）** 依然无法加载（服务器返回 500 错误）；NPCI 官方新闻稿正文仍未核实

#### D.2 UPI Reserve Pay 官方说明——更新（Razorpay 博客 2026-03-12）

**结论：✅ 更新确认：底层为 NPCI "Single Block Multi Debit (SBMD)" 框架**

- Razorpay 于 2026年3月12日正式发布 UPI Reserve Pay 产品，官方说明书：
  - URL：https://razorpay.com/blog/upi-reserve-pay/
  - 原文："Razorpay's launch of UPI Reserve Pay, built on the National Payments Corporation of India's **Single Block Multi Debit (SBMD) framework**."
  - 已在 Global Fintech Fest 和 India AI Impact Summit 上演示，接入了 **ChatGPT、Gemini 和 Claude**
  - 在 Zepto、Zomato、Swiggy 等高频平台试点

#### D.3 印度 agentic payment 生态（截至 2026年9月，媒体综合）

**结论：来自 Forkast/Yahoo Finance（2026-09-09）和 Reuters（2026-09-01、2026-09-10）**

- **NPCI "Unified Agent Protocol"（统一智能体协议）**：Reuters 引 3 名知情人士称 NPCI 正在筹备，允许 AI agent 无需逐笔用户授权完成小额数字支付；**NPCI 尚未官方确认**
- **Pine Labs P3P**：2026年6月，Pine Labs 发布"印度首个基于 UPI 的 agentic payment 协议"P3P，已上生产
- **Mastercard Agent Pay**：2026年2月于 India AI Impact Summit 演示"印度首笔经认证的 agentic 交易"，合作方 Axis Bank、RBL Bank，商户 Swiggy、Zepto
- 来源：https://finance.yahoo.com/technology/ai/articles/india-eyes-ai-driven-upi-084800065.html

#### D.4 Hello UPI

**结论：⚠️ 仍未验证——npci.org.in/hello-upi 返回 404**

---

### E. 中国行业标准与监管机构——仍未验证

| 机构 | 云浏览器结果 |
|------|-------------|
| 中国支付清算协会（pcipss.org.cn） | DNS 解析失败（与前次一致） |
| 中国信通院（caict.ac.cn） | HTTP 412（地理封锁，与前次一致） |
| 人民银行（pbc.gov.cn） | 超时，部分加载但内容不可搜索 |
| e-CNY（ecny.cn） | DNS 解析失败（与前次一致） |

---

### F. 可替换中文段落——基于补核更新的「中国生态」节（完整替换版）

以下内容可完整替换综述 B.6 节。相比上次版本，新增了 **支付宝 AHA/全栈 agentic commerce 平台（2026-08）** 和 **银联国际 × 连连数字（2026-07）** 两项新内容。

```
#### 中国生态

##### 支付宝/蚂蚁

**Antom MCP Server（国际品牌）（✅ 国际市场已核实）**

蚂蚁集团旗下国际支付品牌 Antom（原 Alipay+，面向跨境/海外市场）于 2026年1月正式发布 Antom MCP Server，是官方自称"中国首家支持 MCP 协议的支付机构"（来源：docs.antom.com/ac/mcp/overview，时间戳 2026-01-06）。核心信息：GitHub 仓库 `alipay/global-alipayplus-mcp`，MIT 许可证，OAuth 端点 `https://mcp.antom.com`，支持 Claude Code、ChatGPT、Cursor、Gemini CLI、Dify、Vercel、Wix；工具包括 `create_payment`、`query_payment`、`cancel_payment`、`create_refund` 等（文档更新 2026-08-14）。

**境内支付宝——全栈 agentic commerce 平台（✅ 2026年8月已核实）**

2026年8月17日，支付宝在杭州举办首届 AI 生态大会，发布"中国首个全栈 Agent 商业基础设施"及 AHA（Agent Hub Access）多智能体跨端互联互通协议。平台核心能力：将商户现有页面、商品目录和服务流程转化为标准 **Skills 和 MCP 协议**，供各类 AI agent 调用，实现"单次指令→多 agent 协作→安全履约→交易结算"的完整闭环。

- "阿宝"（Abao）agent：2026年6月开始邀请测试，已适配超 1 万项服务（交通、餐饮、文旅、政务等 8 大类）
- 终端覆盖：5个手机品牌（占比超70%市场份额）+ 16家主要车企（跨设备生态：手机、车机、AI 眼镜、IoT 设备、大模型应用）
- 生态合作方（20+家）：Qwen（通义）、华为、OPPO、vivo、小米、荣耀、比亚迪、吉利、理想汽车、蔚来、阶跃星辰、Rokid 等
- 已接入商户：麦当劳、蜜雪冰城、瑞幸咖啡、高德地图、滴滴、德邦快递、圆通速递、丰巢快递柜
- Ant Group CEO 韩歆毅：预计 agentic commerce 将在 6-12 个月内快速扩展
- 来源：PR Newswire/Crowdfund Insider（2026-08-17）；Finextra（2026-08-17）

**微信支付（⚠️ 有开发者 Skill，无 agentic payment 协议）**

微信支付于 2026年发布"微信支付接入 Skill"（GitHub: `wechatpay-apiv3/wechatpay-skills`，v1.3，2026-10-13），这是一个帮助开发者集成微信支付 API 的 AI 开发助手（产品选型建议、示例代码检索、接入质量评估），在 AI IDE 内运行。该 Skill **不是** agentic payment 协议——不包含让 AI agent 代用户发起或授权支付的能力。微信支付开发文档 pay.weixin.qq.com 无任何 MCP 端点或预授权 agent 支付接口。

**银联**

- **银联 × 华为战略合作（✅ 2026-05-07，境内）**：联合发布《AI 行动计划》，聚焦 AI 与金融科技融合，未涉及"智能体支付"具体产品（来源：cn.unionpay.com 官网）
- **银联国际 × 连连数字：AI-agent 跨境支付（✅ 2026-07-18，跨境）**：两方签署合作协议，基于连连 AI-agent 平台与银联国际全球支付网络，初期聚焦 B2B 全球采购和 AI Token 充值的 Human-in-the-Loop agentic payment（来源：PR Newswire/Yahoo Finance，2026-07-27）
- **境内"智能体支付"标准（⚠️ 仍未验证）**：cn.unionpay.com 无相关公告；pcipss.org.cn（支付清算协会）DNS 故障，无法访问

**数字人民币 e-CNY（⚠️ 仍未验证）**

ecny.cn 域名仍 DNS 解析失败；pbc.gov.cn 超时，无 agent 相关内容。

**行业标准（⚠️ 仍未验证）**

中国支付清算协会（pcipss.org.cn，DNS 失败）和信通院（caict.ac.cn，HTTP 412）均无法访问，"智能体支付"团体标准立项情况无法核实。
```

---

### G. 补核总结

| 核实项 | 上次状态 | 补核结果 |
|--------|---------|---------|
| 境内支付宝 MCP Server（2025-04） | 未验证 | **更新**：原始 MCP 为 Antom（国际）；境内支付宝于 2026-08 发布 AHA 协议含 MCP 支持 |
| 支付宝 AI 付/全栈 agentic commerce | 未验证 | **新增核实** ✅：2026-08-17 发布（Crowdfund Insider/PR Newswire） |
| 微信支付 MCP/agent 支付接口 | 未验证 | **更新**：有开发者 Skill，非 agentic payment 协议 |
| 银联"智能体支付"境内公告 | 未验证 | **仍未验证**；新增确认银联国际×连连数字 AI-agent 跨境（2026-07） |
| 银联 × 华为合作 | 未验证 | **新增核实** ✅：2026-05-07，AI 整体战略，未涉及专项智能体支付 |
| 数字人民币 agent/智能合约 | 未验证 | **仍未验证**（DNS 失败） |
| 行业标准（支付清算协会/信通院） | 未验证 | **仍未验证**（DNS/HTTP 412） |
| NPCI 官方新闻稿（2025-10 试点） | 未验证 | **仍未验证**（npci.org.in/media/press-release 服务器 500）；Razorpay 官方博客已确认为合作方官方来源 |
| UPI Reserve Pay 官方说明 | 部分核实 | **强化确认** ✅：NPCI SBMD 框架；Razorpay 2026-03-12 官方发布 |
| Hello UPI | 未验证 | **仍未验证**（npci.org.in/hello-upi 返回 404） |
| Gemini/AP2 支持 UPI 官方公告 | 未验证 | **部分补充**：Razorpay 博客确认 Gemini 已在演示中接入 UPI Reserve Pay；无 Google 官方独立公告 |

---

*补核完成时间：2026-10-19*
*工具：Amazon Bedrock AgentCore 云浏览器（session 01M59SM019CZANNTAVKA8VSWWZ）*
