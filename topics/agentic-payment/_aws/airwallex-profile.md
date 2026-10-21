# Airwallex 客户画像调研底稿

> 调研日期：2026-10-20  
> 用途：AWS 金融行业 SA 面向 Airwallex 等跨境支付 / FinTech 客户设计 Agentic Payment 解决方案  
> 说明：每条事实均标注来源 URL + 日期；无法核实的标「未验证」；推断标「推断」。

---

## 1. 公司概况

### 1.1 基本信息

| 维度 | 数据 | 来源 |
|------|------|------|
| 成立时间 | 2015 年，澳大利亚墨尔本 | [airwallex.com/about](https://www.airwallex.com/about)，访问 2026-10-20 |
| 总部 | 新加坡 + 旧金山（2025 年起双总部） | 同上 |
| 办公室 | 全球 27 个 | 同上 |
| 员工规模 | 2,300+ 人 | 同上 |
| 估值 | 110 亿美元（2026 年 6 月） | [系列 H 公告](https://www.airwallex.com/global/newsroom/airwallex-secures-320-million-in-series-h-funding-valuation-hits-11-billion)，2026-06-25 |
| 累计融资 | 18 亿美元 | [airwallex.com/about](https://www.airwallex.com/about)，2026-10-20 |

### 1.2 最新融资

- **系列 H（2026-06-25）**：3.2 亿美元，估值 110 亿美元，较 2025 年 12 月系列 G 的 80 亿美元提升 37.5%
- **领投方**：Addition（回归领投）
- **跟投方**：Baillie Gifford、Hummingbird、QED Investors、T. Rowe Price（代理账户）、Hedosophia、Haun Ventures、Washington University in St. Louis、Amex Ventures
- **资金用途**：加速「自主财务（autonomous finance）与 Agentic 商务（agentic commerce）」产品开发；扩大监管覆盖；扩充 AI 原生金融软件团队
- 来源：[系列 H 新闻稿](https://www.airwallex.com/global/newsroom/airwallex-secures-320-million-in-series-h-funding-valuation-hits-11-billion)，2026-06-25

### 1.3 营收与规模

| 指标 | 数据 | 时间节点 | 来源 |
|------|------|------|------|
| ARR（年化营收） | 13 亿美元，同比 +74% | 2026 年 3 月 | 系列 H 新闻稿，2026-06-25 |
| 年化交易额 | 2,870 亿美元，同比 +120%+ | 2026 年 3 月 | 同上 |
| 客户数 | 675,000+ 家企业（产品页）/ 200,000+ 家（About 页） | 2026 年 | 产品页 + About 页，2026-10-20 |
| 覆盖国家/地区 | 200+ 国家（收付款）；120+ 国家（本地转账） | 2026 年 | About 页，2026-10-20 |
| 已处理交易额 | 累计 2,870 亿美元+ | 截至 2026 年 | About 页，2026-10-20 |
| 当日到账率 | 93% 的转账当日结算，50% 即时到账 | 2026 年 | About 页，2026-10-20 |

> **注**：About 页显示 200,000+ 家企业；产品页显示 675,000+ 家。两个数字可能指不同统计口径（前者可能为注册付费客户，后者含免费层），推断仅供参考。

### 1.4 主要市场

APAC（澳大利亚、新加坡、香港、中国大陆、日本、马来西亚、印尼、越南、韩国）、欧洲（英国、欧盟—荷兰 + 立陶宛双实体）、美国、拉丁美洲（巴西、墨西哥）、中东（以色列）、北美（加拿大）、大洋洲（新西兰）

EMEA 扩张重点：2026 年 3 月宣布投资超过 10 亿美元用于 EMEA 扩张；英国五年投资 5.9 亿英镑（2025-12-11 公告）。

### 1.5 近期并购（2025–2026）

| 标的 | 战略意义 | 日期 | 来源 |
|------|------|------|------|
| **PT Skye Sab Indonesia** | 印尼 Bank Sentral BI Category 1 支付服务牌照；东南亚扩张 | 2025-12-08 | [新闻稿](https://www.airwallex.com/global/newsroom/) |
| **Paynuri（韩国）** | 韩国电子支付业务运营资质 + 外汇业务许可；进入韩国市场 | 2026-01-21 | [新闻稿](https://www.airwallex.com/global/newsroom/) |
| **Leapfin** | 收入确认、收入核对自动化 SaaS；强化 CFO Office 产品线 | 2026-06-04 | [新闻稿](https://www.airwallex.com/global/newsroom/airwallex-acquires-leapfin-expanding-financial-lifecycle-capabilities) |

---

## 2. 产品线

### 2.1 产品全景与 Agentic 意义

#### Business Accounts（商业账户）

| 产品 | 简介 | Agentic 意义 |
|------|------|------|
| **Borderless Cash** | 20+ 种货币账户，全球收款 | Agent 可操作多币种账户，在框架内跨币种调拨 |
| **FX & Transfers** | 200+ 国家，银行同业汇率，93% 当日到账 | Agent 可自主执行跨境汇款，减少人工审批 |
| **Yield** | 美元/新币余额理财，资产管理规模超 10 亿美元（2026-03 突破） | Agent 可基于规则自动分配余额到 Yield，优化资金效率 |

#### Spend（支出管理）

| 产品 | 简介 | Agentic 意义 |
|------|------|------|
| **Corporate Cards** | 多币种实体/虚拟卡，自定义限额 | Agent 专用虚拟卡（受控支出）的基础设施 |
| **Expense Management** | 报销、AI 收据匹配、费用 AI 审核 | Expense Policy Agent 已上线：自动执行策略 |
| **Bill Pay** | AP 自动化，从上传账单到审批支付 | AI 自动提取发票字段、路由审批 |
| **Purchase Orders** | 采购流程控制 | 与 Bill Pay 联动，Agent 可管理完整 P2P 流程 |

#### Payments（收单）

| 产品 | 简介 | Agentic 意义 |
|------|------|------|
| **Checkout** | 低代码支付表单，优化转化率 | — |
| **Airi** | 一键结账钱包，3x 速度提升，转化率 +14% | **Airi for Agents**（即将推出）：Shopper 授权 Agent 使用其储存支付凭证付款 |
| **Agentic Commerce Suite** | 待候补名单开放；含 Agentic Checkout SDK、Shopping Assistant SDK、Agent Trust & Risk SDK、Agentic Token Vault、Agentic Payment Rails | **核心 Agentic 产品**；支持 ACP、UCP 协议；Agentic Wallet 提供"协议兼容虚拟卡（protocol-compatible virtual cards）" |
| **Optimize 360** | AI 欺诈拦截 + 下降订单挽回 | Agent 发起的交易需要 Bot/Agent 风险识别能力 |
| **Sentinel** | 风险管理 | — |

> 来源：[airwallex.com/global/payments/agentic-commerce](https://www.airwallex.com/global/payments/agentic-commerce)，2026-10-20；[airwallex.com/global/payments/airi](https://www.airwallex.com/global/payments/airi)，2026-10-20

#### Billing（计费）

| 产品 | 简介 |
|------|------|
| Invoicing | 全球发票，160+ 本地支付方式 |
| Subscription Management | 订阅与混合定价，多频率计费 |
| Usage-Based Billing | 按量计费，适合 AI API 公司 |

**Agentic 意义**：Airwallex 与 Stigg 合作（2026-10-15 公告），将全球支付与权益管理打通，专注 AI 公司计费场景。Usage-Based Billing 天然适配 AI API / Agent 微支付场景（尽管 Stablecoin 更适合极低单价）。

#### Platform APIs & Embedded Finance

| 产品 | 简介 | Agentic 意义 |
|------|------|------|
| **Connected Accounts** | 程序化开户、商户 KYB 入网 | 平台型 AI 公司可为其用户批量开户 |
| **Issuing API** | 大规模发卡，国际交易零手续费 | Agent 专用虚拟卡基础；Agentic Wallet 的底层 |
| Transactional FX | 银行同业汇率 + 货币风险管理 | Agent 可实时 FX 套汇，优化跨境支出 |
| Payouts | 全球快速低成本汇款 | Agent 可自主触发跨境付款 |

#### AI 产品线（内部 + 对外）

| 产品 | 上线状态 | 简介 |
|------|------|------|
| **T:0** | 2026-09-01 上线（私测） | AI 原生会计平台，面向初创公司；覆盖账务、预测、税务、合规、报告 |
| **Expense Policy Agent** | 已上线 | "全天候 AI Agent，跨实体/货币/语言执行费用政策" |
| **Kai** | 已上线 | AI 助理，实时答疑 + 全球现金流可视化 |
| Receipt Matching AI | 已上线 | 自动匹配收据与卡交易 |
| AP AI | 已上线 | 自动提取发票字段、PO 匹配 |
| **Agentic Business Account** | 产品公告 2026-09-30 | 整合 AI Agent 操作全球财务的单一账户枢纽 |

---

## 3. 牌照与合规

### 3.1 全球牌照清单

以下来源：[help.airwallex.com/hc/en-gb/articles/900001757106](https://help.airwallex.com/hc/en-gb/articles/900001757106-How-is-Airwallex-licensed-and-regulated)，最后访问 2026-10-20

| 地区 | 主体 | 牌照类型 | 监管机构 | 牌照号 |
|------|------|------|------|------|
| **澳大利亚** | Airwallex Pty Ltd | AFSL | ASIC | 487221 |
| **澳大利亚** | Airwallex Capital Pty Ltd | AFSL | ASIC | 549026 |
| **巴西** | AWX Brasil Instituição de Pagamento Ltda. | 支付机构（发行、采购、FX） | Banco Central do Brasil | — |
| **加拿大** | Airwallex (Canada) International Payments Ltd | MSB（联邦） | FINTRAC | M19395067 |
| **加拿大** | 同上 | MSB（魁北克省） | Revenu Québec | 14460 |
| **中国大陆** | **云汇支付有限公司（Yunhui Pay Co. Ltd）** | 支付牌照（预付卡 I 类） | **中国人民银行（PBOC）** | Z2025844000016 |
| **香港** | Airwallex (Hong Kong) Ltd | MSO（货币服务） | 香港海关 | 16-09-01929 |
| **香港** | UniCard Solution Ltd（集团成员） | **SVF（储值支付工具）** | **HKMA** | SVF0009 |
| **香港** | Airwallex Capital Hong Kong Limited | SFC Type 4（证券顾问）+ Type 9（资产管理） | SFC | BUL570 |
| **印度尼西亚** | PT Skye Sab Indonesia | 支付服务提供商 Category 1 | Bank Sentral RI | 23/583/DKSP/Srt/B |
| **以色列** | Airwallex (Israel) Ltd | 支付服务 | Israel Securities Authority | H.P. 516851961 |
| **日本** | Airwallex Japan K.K. | 电子支付代理业（電子決済等代行業） | 金融厅（FSA） | 119（2023-09-04） |
| **日本** | 同上 | 资金转移业者 Type 2（第二種資金移動業） | 関東財務局 | 00100（2025-09-08） |
| **立陶宛** | AWX Lithuania, UAB | 电子货币机构（EMI）+ **EEA 护照** | 立陶宛银行 | 88（2021-10-06） |
| **马来西亚** | Airwallex (Malaysia) Sdn Bhd | MSB Class A（汇款 + 换汇）+ 电子货币 + 收单 | Bank Negara Malaysia | 00767 |
| **墨西哥** | MexPago Transacciones, S.A.P.I. de C.V. | IFPE（电子支付基金机构） | CNBV | — |
| **荷兰（EEA）** | Airwallex (Netherlands) B.V. | EMI + **EEA 护照** | De Nederlandsche Bank | R179622 |
| **荷兰（EEA）** | Airwallex Capital (Netherlands) B.V. | 投资服务（货币市场基金申购/赎回）+ EEA 护照 | AFM | — |
| **新西兰** | Airwallex (New Zealand) Ltd | FSP | FSPR | FSP1001602 |
| **新加坡** | Airwallex (Singapore) Pte. Ltd. | **主要支付机构（MPI）** | **MAS** | PS20200541 |
| **新加坡** | Airwallex Capital (Singapore) Pte. Ltd. | CMS 资本市场服务牌照 | MAS | CMS101830 |
| **韩国** | Paynuri Co., Ltd.（集团成员） | 电子金融业（PG + 预付电子支付工具）+ 外汇 | FSC / MOEF | — |
| **英国** | Airwallex (UK) Ltd | **FCA EMI**（电子货币机构） | FCA | 900876 |
| **美国** | Airwallex US, LLC | MTL（货币传输许可证，45+ 州/领地） | 各州监管机构 | NMLS 1928093 |
| **美国** | AWX Capital US | FINRA 会员 + SEC 注册 | FINRA / SEC | — |
| **越南** | Payment Telecom Informatics Post A Member of Company Limited | IPS（中间支付服务） | State Bank of Vietnam | — |

**合计：85+ 个牌照和注册**（官方口径，含未单独列出的细项）

### 3.2 EPI 主要成员

- Airwallex 于 2025–2026 年加入欧洲支付倡议（EPI）成为**主要成员（Principal Member）**，引入 Wero 支付方式至欧洲商户
- 来源：官方 newsroom 页面（DuckDuckGo 搜索发现的官方 URL：`airwallex.com/global/newsroom/airwallex-becomes-a-principal-member-of-epi-...`）；EPI Company 网站

### 3.3 Visa / Mastercard 网络地位

- 官网及 help 中心均未明确声称 Visa 或 Mastercard "主要成员（Principal Member）"身份
- 官网仅称"与顶级支付网络合作（partnered with top-tier payment networks）"（vague）
- Agentic Commerce Suite 页面明示 Agentic Payment Rails"基于 Visa 和 Mastercard 最新标准构建"
- 2026-07-23 与 Visa Commercial Solutions 签署合作，联合开发货运平台嵌入式金融解决方案
- **推断**：Airwallex 发卡业务（UniCard SVF + Issuing API）很可能通过某种 Visa/Mastercard sponsoring 或 principal membership 实现，但未找到官方公开披露，标「未验证」

---

## 4. 技术栈与云

### 4.1 主力云：GCP（Google Cloud Platform）

**关键证据（最高可信度）**：

> "In 2022, it migrated its infrastructure to Google Cloud to reduce service latency and meet global compliance requirements."  
> ——Timothy Wong, VP of Data and AI, Airwallex  
> 来源：[Google Cloud 官方案例研究 cloud.google.com/customers/airwallex-ai](https://cloud.google.com/customers/airwallex-ai)，访问 2026-10-20

**已确认使用的 GCP 服务**：

| GCP 服务 | 用途 |
|------|------|
| Google Cloud Storage | Delta Lake 底层存储（Databricks on GCP） |
| Vertex AI | 客户服务语音分析、争议解决自动化、测试代码生成 |
| **Gemini**（on Vertex AI） | 工程师 coding 辅助、NotebookLM 知识管理 |
| **NotebookLM** | 内部知识问答 |
| **Agent2Agent（A2A）** | Agent 之间互联，用于 agentic finance 构建 |
| Looker | 实时 BI，支撑全公司决策 |
| BigQuery | 历史 BI 仓库（已迁移部分到 Databricks Delta Lake） |

**GCP 量化效果**（来自 Google Cloud 案例研究）：
- 网络延迟降低 50%
- 客户问题处理速度提升 10x
- 争议解决节省 40% 时间
- 产品上市速度加快 95%+
- 90%+ 工程师日常集成 AI 提升生产力
- 测试代码编写时间：3 天 → 15 分钟

### 4.2 数据与 AI 平台：Databricks on GCP

来源：[databricks.com/customers/airwallex/gcp](https://www.databricks.com/customers/airwallex/gcp)，访问 2026-10-20

- Delta Lake（Google Cloud Storage 上），Bronze/Silver/Gold 三层架构
- Unity Catalog 数据治理（角色访问控制、审计）
- Lakeflow Spark 声明式管道
- **整合的 AI 模型**：Gemini + **Anthropic** + **OpenAI**
- 量化效果：批量管道从 1,600 条合并为 60 条（-96%）；数据延迟 1 小时 → 15 分钟；洞察速度 3–4x

### 4.3 其他技术栈（来自 GitHub）

来源：[github.com/airwallex](https://github.com/airwallex)（34 个公开仓库），访问 2026-10-20

| 组件 | 证据 |
|------|------|
| Kubernetes | `k8s-pod-restart-info-collector`（387 stars），`opentelemetry-helm-charts` |
| Cassandra | `java-driver`（fork of DataStax driver） |
| Apache APISIX | 云原生微服务 API 网关（fork） |
| HashiCorp Vault + Consul | `terraform-google-vault`、`terraform-google-consul`（均为 Google Cloud Terraform 模块） |
| Splunk | `splunk-pipeline-oss` |
| OpenTelemetry | `opentelemetry-helm-charts` |
| Spinnaker | `gate`（Spinnaker API Gateway fork）—CI/CD |
| Apache Spark | `AirSkiff`（实时流处理库，Java） |
| ML Feature Store | `chronon`（Scala，类似 Twitter/Stripe 的 Chronon） |
| **阿里云** | `saml2alibabacloud`（通过 SAML IdP 获取阿里云临时凭证的 CLI 工具）—**可能用于中国区** |

**结论**：Airwallex 是深度 GCP 用户，迁移始于 2022 年，当前无任何公开的 AWS 合作案例、AWS Marketplace 上架或 AWS 战略合作公告。中国业务可能通过阿里云运营（推断）。

### 4.4 AWS 关系

- **现有证据强度：弱**。搜索 AWS 官方案例库（`aws.amazon.com/solutions/case-studies/airwallex/`）返回 404；未发现 AWS Marketplace 上架；未发现 AWS 战略合作或 re:Invent/AWS Summit 提及 Airwallex 的公开演讲。
- **对 AWS 竞争策略的含义**：Airwallex 目前是 GCP 的旗舰 FinTech 案例之一，在 GCP 上构建 AI/Agentic 能力。AWS 如需切入，需要从 AWS 独特优势切入（如 AWS 合规框架在特定市场的优势、Amazon Bedrock 的 Claude/Nova 差异化、AWS Payment Cryptography、AWS 在中东/LatAm 的基础设施覆盖），而非直接竞争云基础设施。

---

## 5. AI 与 Agentic 动作

### 5.1 官方 MCP Server 与 Agent Toolkit

- **GitHub 扫描**（2026-10-20）：在 34 个公开仓库中**未发现名为 MCP Server、AI SDK 或 Agent Toolkit 的仓库**。`airi-cli`（Shell）是唯一可能与 AI 相关的 CLI 工具，但描述为"Airi command-line interface"，用途未明确。
- **结论**：Airwallex **尚无公开 MCP Server**（标「未验证是否有私有/未公开仓库」）。

### 5.2 Agentic Commerce Suite（最重要的 Agentic 产品）

来源：[airwallex.com/global/payments/agentic-commerce](https://www.airwallex.com/global/payments/agentic-commerce)，2026-10-20

**目前状态**：候补名单（waitlist）开放，未正式 GA

**面向商户**：
- **Agentic Checkout SDK**：与商务系统实时集成，让产品可被 AI Agent 发现并购买
- **Shopping Assistant SDK**：内嵌 AI 购物助手，在同一对话中完成推荐 + 结账
- **Agent Trust & Risk SDK**：区分合规 Agent 与恶意 Bot；记录每个 Agent 的身份、意图与操作；支持争议解决
- **Agentic Token Vault**：对用户支付凭证 Tokenize，供 Agent 重复使用
- **Agentic Payment Rails**：提高授权率的支付通道，"基于 Visa 和 Mastercard 最新标准构建"

**面向 AI 公司（Agent 侧）**：
- **Native Checkout SDK**：用户无需离开 AI 平台即可完成结账
- **Agentic Wallet**：为 AI Agent 提供"协议兼容虚拟卡（protocol-compatible virtual cards）"，实现自主支付

**已支持协议**：
- **ACP**（OpenAI Agentic Commerce Protocol）✓ 明确命名
- **UCP**（Universal Commerce Protocol，Google/Adyen 等共创）✓ 明确命名
- "and more"（未明确列出其他协议）

**未提及**：x402、AP2（Agent Payments Protocol）、MCP（Model Context Protocol）、MPP（Machine Payments Protocol）、Visa Intelligent Commerce / Visa TAP、Mastercard Agent Pay

### 5.3 Airi（消费者侧 Agentic 钱包）

来源：[airwallex.com/global/payments/airi](https://www.airwallex.com/global/payments/airi)，2026-10-20；系列 H 新闻稿 2026-06-25

- **现状（已上线）**：一键结账钱包，存储支付凭证，跨商户使用；声称转化率提升 14%，覆盖 170+ 国家
- **Airi for Agents（即将推出）**：消费者可将购物委托给 AI Agent，使用其存储的支付凭证"安全授权结账"
- 系列 H 公告将 Airi 定位为"agentic 消费者钱包"，计划增加：委托 Agent 支付、支出限额、权限控制、多币种余额

### 5.4 CEO Jack Zhang 的 AI / Agentic 战略

来源：[airwallex.com/global/blog/the-agentic-banking-era](https://www.airwallex.com/global/blog/the-agentic-banking-era)（Jack Zhang，2026-09-30）

核心论点：
1. **三件事**：每家公司都在收钱、花钱、管钱；谁先把这三件事交给 Agent，谁就获得复利竞争优势
2. **两个层次**：运营层（Agent 持续观察、在参数内决策、无需人工审批）+ 基础设施层（资金以秒而非天为单位跨实体/货币/市场流动）
3. **"Agentic 速度"**：互联网速度下人类仍做每个决策；Agent 时代"工作单元从交易转向结果"，财务从周期性检查转为持续运营
4. **Agent 操作框架**："Observe → Decide → Act → Reconcile → Repeat"
5. **Agentic Business Account** 定位为统一栈的中心节点（89 个牌照 + 直连本地清算网络）

### 5.5 稳定币战略

来源：[airwallex.com/global/blog/why-airwallex-customers-didnt-need-stablecoins-and-why-were-building-for-them-now](https://www.airwallex.com/global/blog/why-airwallex-customers-didnt-need-stablecoins-and-why-were-building-for-them-now)（Dan Kim，VP Product Partnerships，2026-08-25）

**作者背景**：Dan Kim 曾任 Coinbase 五年，**共同起草了 x402 协议**（互联网原生支付标准，面向 API、应用与 AI Agent）。

**关键结论**：
1. Airwallex 老客户不需要稳定币：已有高效的本地货币池 + 本地清算网络，93% 当日到账
2. **为什么现在建**：AI Agent 开始生产部署；稳定币适合（a）Agent 微支付（固定手续费吃掉价值，稳定币无此问题）（b）慢速廊道（等待银行清算窗口的受益方）
3. **Metal 合作**："Airwallex invested to incubate Metal and become its first design partner. Metal is settlement infrastructure for regulated financial institutions, built so they can move money at the speed agents transact."
4. **推进中**：为允许的市场客户添加稳定币支持；与 Metal 合作推进监管机构认可的结算基础设施

**香港稳定币牌照**：未在官方渠道找到申请证据，标「未验证」

**USDC 支持**：官网/新闻稿未明确提及 USDC，标「未验证」

### 5.6 与 AI 公司的合作关系

- **Anthropic / OpenAI**：Databricks 案例研究显示 Airwallex 在数据平台上集成了 Anthropic 和 OpenAI 的模型（用于内部 AI 工作）；未发现 Airwallex 与 Anthropic / OpenAI 签署公开合作协议或 Airwallex 成为其官方支付合作伙伴
- **Latitude 37**：Airwallex 设立每年 100 万澳元的基金，支持澳大利亚年轻 AI 创业者（2026-04-30 公告）
- **Stigg**：与 Stigg 合作，将全球支付与权益管理打通，专注 AI 公司计费场景（2026-10-15 公告）
- AI 公司用 Airwallex 收款：产品页面以"SaaS 及科技公司"为目标市场；未找到具体 AI 公司（如 Anthropic、OpenAI、Cohere）使用 Airwallex 收款的官方案例

### 5.7 内部 AI 应用（已落地）

来源：Google Cloud 案例研究 + Databricks 案例研究 + 官网 Spend AI 页面

| 场景 | 能力 | 量化效果 |
|------|------|------|
| 客户服务（Voice of Customer 分析） | Vertex AI 实时分析客户反馈 | 客户问题处理速度 10x |
| 争议解决自动化 | Vertex AI 自动处理争议案件 | 节省 40% 时间 |
| 工程效率 | Gemini 生成测试代码 | 测试编写 3 天 → 15 分钟；上市速度 +95% |
| KYB / KYC（推断） | 连接 Accounts API + Connected Accounts | 相关招聘 JD 提及 ML 风控，但未找到公开披露 |
| 费用政策执行 | Expense Policy Agent | 自动化合规执行，减少人工审批 |
| 发票处理 | AP AI | 自动提取字段、路由审批 |
| 收据匹配 | Receipt Matching AI | 处理分期、拆分、时间错位场景 |
| 财务智能 | T:0（2026-09 私测上线） | AI 原生全财务功能（账务、预测、税务、合规） |

---

## 6. 主要竞争对手的 Agentic 动作

### Stripe

- **Agentic Commerce Suite**（美国，2025–2026 年推出）
- **已支持协议**：UCP、MPP（Machine Payments Protocol，开放互联网原生标准）、x402、SPT（Shared Payment Tokens）
- **Agent 支付产品**：Link Agent Wallet（代表用户付款）+ Issuing for Agents（Agent 专用虚拟卡 + 支出控制）
- **市场地位**：78% Forbes AI 50 名单企业用 Stripe；90% 美国成年人曾在使用 Stripe 的商户购物
- 来源：[stripe.com/use-cases/agentic-commerce](https://stripe.com/use-cases/agentic-commerce)，2026-10-20

### Adyen

- **Adyen Agentic**（2026-06-16 纽约发布，限量美国企业商户，全球扩张中）
- **三层产品**：Agentic Feed（目录分发）+ Agentic Cart（购物车 + 结账集成）+ Agentic Payments（支付 + 风控）
- **已支持协议**：UCP（早期背书并参与共创）、AP2（Agent Payments Protocol）、ACP（OpenAI）
- **已有 MCP Server**：[docs.adyen.com/development-resources/mcp-server](https://docs.adyen.com/development-resources/mcp-server)
- **战略合作伙伴**：Amex、Mastercard、Salesforce、Visa；早期商户 SharkNinja、Scheels、ESW、Sézane
- 来源：[adyen.com/press-and-media/adyen-agentic](https://www.adyen.com/press-and-media/adyen-agentic)，2026-10-20

### Checkout.com

- 设有专门的 Agentic Commerce 资源中心（checkout.com/resources/agentic-commerce）
- Mastercard Agent Pay 合作（产品页 URL 含"agent-pay"关键词，指向推断）
- 对 ACP / UCP 有深度跟踪（分析文章），但未发现具体 GA 产品
- 来源：[checkout.com/resources/agentic-commerce](https://www.checkout.com/resources/agentic-commerce)，2026-10-20

### Wise

- 主打 B2B / 个人跨境汇款，公开资料中未发现专项 Agentic 支付产品
- 未验证是否有 MCP Server 或 Agent Toolkit

### Payoneer

- 专注自由职业者、电商平台收款；公开资料未发现专项 Agentic 产品
- 未验证

### Nium

- B2B 嵌入式金融基础设施；公开资料未发现专项 Agentic 产品
- 未验证

### Rapyd

- 支付即服务（PaaS）；公开资料未发现专项 Agentic 产品
- 未验证

### Ant International（Antom / Alipay+）

- APAC + 新兴市场最强竞争对手；Alipay+ 的 AI 能力主要集中在消费端智能推荐；未发现专项 B2B Agentic 支付产品
- 未验证；在中国市场，Airwallex 的 Yunhui Pay 与蚂蚁金服的境内支付直接竞争

### PingPong / 连连

- 主要服务中国跨境电商卖家（Amazon 等）；AI 功能集中在风控 + 汇率预测；未发现公开 Agentic 支付产品
- 未验证

**对 Airwallex 的压力小结**：Stripe（协议最广 + AI 生态最深）和 Adyen（多协议 + 企业商户 + 有 MCP Server）是最直接的 Agentic 压力来源。Airwallex 在 ACP + UCP 上已起步，但 MCP Server 缺失、AP2 未支持、部分产品（Agentic Commerce Suite、Airi for Agents）仍在候补名单阶段，存在明显产品差距。

---

## 7. Airwallex 此类客户最可能关心的 8–12 个问题

以下每条问题均注明依据来源：产品线现状、公开表态或竞争压力。

### Q1. 如何让其商户安全接收 AI Agent 发起的支付？

**依据**：Agentic Commerce Suite 仍在候补名单阶段；竞争对手 Stripe/Adyen 已 GA；商户客户要求明确。Agentic Payment Rails 需要区分 Agent 与人类买家的欺诈信号（Agent Trust & Risk SDK 在建中）。

**AWS 切入**：Amazon EventBridge + Lambda 实时事件处理 Agent 支付流；Amazon Fraud Detector 提供 Agent 行为特征模型。

### Q2. 如何给 AI Agent 发卡并精细化控制支出（Know Your Agent）？

**依据**：Agentic Wallet 声称提供"协议兼容虚拟卡"，但目前产品细节模糊；Issuing API 已存在但未公开对 Agent 场景的优化；竞争对手 Stripe Issuing for Agents 已有虚拟卡 + 支出控制 + 实时可见性。

**AWS 切入**：Amazon DynamoDB 实时权限/支出规则存储；AWS Payment Cryptography 安全管理卡密钥；AWS Step Functions 编排 Agent 支付审批流。

### Q3. 多协议适配成本如何降低（ACP + UCP + x402 + AP2 + MPP 同时支持）？

**依据**：Airwallex 产品页面仅明确支持 ACP + UCP；Dan Kim 是 x402 共同作者但 x402 官网未列入；Adyen 已支持 ACP + UCP + AP2；Stripe 支持 UCP + MPP + x402。跨境支付客户面向的 AI Agent 来自不同生态，协议碎片化是真实痛点。

**AWS 切入**：Amazon API Gateway + AWS AppSync 统一多协议适配层；Amazon Bedrock Agents 内置对多协议的原生调用支持。

### Q4. 跨境 Agent 支付的合规边界在哪里（KYA / 责任主体认定）？

**依据**：Airwallex 拥有 85+ 个牌照（覆盖 20+ 司法管辖区），但 Agent 发起交易的监管属性（谁是"Payer"）在各市场尚无统一定论；Airwallex 博客"Agentic banking starts with governance"（Dan Kim，2026-10-07）专门论述治理先行。Airwallex 在美国有 FINRA 成员（AWX Capital US）用于证券类业务，但 MTL 不一定覆盖 Agent 作为授权委托的新模式。

**AWS 切入**：Amazon Macie / AWS Audit Manager 可帮助建立 Agent 交易审计日志；AWS IAM Identity Center + Organizations 实现跨组织的 Agent 权限治理。

### Q5. Agent 高频微支付的清算效率与成本如何优化？

**依据**：稳定币博文明确指出：卡支付的固定手续费在微支付场景不经济（"the seller pays more in card fees than it collects"）；Metal 合作是回应。但 Metal 目前仍是内部孵化阶段，何时开放给客户未知。

**AWS 切入**：Amazon Aurora + ElastiCache 实现高频记账的低延迟；Amazon Managed Blockchain / QLDB 不可篡改账本记录 Agent 微支付流水。

### Q6. AI 驱动的实时风控如何区分合规 Agent 与恶意 Bot？

**依据**：Optimize 360 已上线，但是通用欺诈 AI；Agentic Commerce Suite 中的 Agent Trust & Risk SDK 专门应对此场景（身份、意图、操作记录）。随着 Agentic 商务规模扩大，Bot 冒充合规 Agent 攻击的风险上升。

**AWS 切入**：Amazon SageMaker 训练 Agent 行为基线模型；Amazon Bedrock Guardrails 对 Agent 输出进行内容 + 行为审查；AWS WAF 过滤非授权 Agent 请求。

### Q7. 如何为 AI 原生平台（SaaS、AI API 公司）提供嵌入式金融（Embedded Finance）？

**依据**：Stigg 合作（2026-10-15）标志 Airwallex 正在向 AI 公司的计费 + 支付一体化切入；Connected Accounts + Issuing API 是技术基础；Usage-Based Billing 适配 AI API 按量计费。AI 公司（如 Anthropic 的商业 API 客户）需要全球收款能力。

**AWS 切入**：AWS Marketplace SaaS 订阅 + Airwallex 全球收款联合解决方案；Amazon Bedrock 的 AI 公司作为共同客户目标。

### Q8. 多币种智能资金池与实时财库管理（24/7 Treasury）？

**依据**：Airwallex Yield 产品（$1B+ AUM）；博文"24/7 Treasury: From static thresholds to programmatic rules at Airwallex"（Bart Verweij，2026-10-13）描述内部实践；CEO 博文中的"Manage"场景（Agent 自动调拨多国余额）。客户需要 Agent 能实时执行跨实体、跨货币的资金调拨，而不等人工审批。

**AWS 切入**：Amazon Bedrock Agents 编排财库 Agent 决策链；Amazon QuickSight 可视化多币种现金流；AWS Step Functions 实现分级审批 + 自动执行。

### Q9. 数据驻留与区域部署策略如何满足监管要求？

**依据**：Airwallex 在 GCP 上运营，覆盖 20+ 司法管辖区。不同地区的数据本地化要求（GDPR、中国数据安全法、印度 PDPB 等）要求支付数据不得出境。Airwallex 已在澳大利亚、新加坡、欧盟等地设立本地实体和账户。

**AWS 切入**：AWS Local Zones、AWS GovCloud（合规金融场景）、Amazon S3 对象锁可满足特定市场数据驻留；AWS Control Tower 实现跨区域合规管理。

### Q10. AI 带来的内部运营效率（Finance AI Transformation）如何衡量与推进？

**依据**：CFO Pranav Sood 系列博文（2026-08 ~ 2026-10）描述"600 个财务任务"的 AI 转型路线图；T:0 收购 Leapfin 后推出；Airwallex 自身先做"金融工程师"的内部实践，再向客户输出。

**AWS 切入**：Amazon Q Business 企业智能助理；Amazon Bedrock Knowledge Bases 构建财务 RAG；AWS Glue + Amazon Athena 数据湖 + AI 分析。

### Q11. Agent 支付如何适配稳定币（USDC/USDT）与法币混合结算？

**依据**：Dan Kim（VP，x402 共同作者）的博文明确 Agent 微支付是稳定币的核心场景；Metal 合作提供监管机构认可的稳定币结算基础设施；香港监管机构已开放稳定币牌照申请（2024–2025 年），UniCard 持有 SVF 牌照可为稳定币进入支付系统铺路。

**AWS 切入**：Amazon Managed Blockchain 为 USDC 链上清算提供基础设施；AWS Key Management Service 管理稳定币钱包密钥；Amazon Bedrock 的 AI Agent 可读取链上状态并触发法币出金。

### Q12. 如何扩展至新兴市场（东南亚、拉美、中东）同时保持合规？

**依据**：Airwallex 2026 年密集在 EMEA + SEA 扩张（印尼、马来西亚、巴西、墨西哥、以色列全牌照）；Bank Negara Malaysia 2026 年 4 月批准；PT Skye Sab 2025 年 12 月收购；EPI/Wero 进入欧洲。新兴市场 Agent 商务落地需要本地 FPS（Pix、PayNow、PromptPay 等）直接集成。

**AWS 切入**：AWS Region 在巴西（São Paulo）、新加坡、中东（UAE）、以色列的覆盖；AWS FinSpace 合规数据管理；Amazon SES / SNS 全球通知基础设施。

---

## 附录 A：数据来源汇总

| 来源 | 类型 | 置信度 |
|------|------|------|
| [cloud.google.com/customers/airwallex-ai](https://cloud.google.com/customers/airwallex-ai) | 一手（GCP 官方案例） | 极高 |
| [help.airwallex.com/hc/en-gb/articles/900001757106](https://help.airwallex.com/hc/en-gb/articles/900001757106) | 一手（官方 help center） | 极高 |
| [airwallex.com/us/state-licenses](https://www.airwallex.com/us/state-licenses) | 一手（官方牌照页） | 极高 |
| [databricks.com/customers/airwallex/gcp](https://www.databricks.com/customers/airwallex/gcp) | 接近一手（Databricks 官方案例） | 高 |
| [airwallex.com/global/newsroom/...series-h...](https://www.airwallex.com/global/newsroom/airwallex-secures-320-million-in-series-h-funding-valuation-hits-11-billion) | 一手（官方新闻稿） | 极高 |
| [airwallex.com/global/payments/agentic-commerce](https://www.airwallex.com/global/payments/agentic-commerce) | 一手（官方产品页） | 极高 |
| [airwallex.com/global/blog/...stablecoins...](https://www.airwallex.com/global/blog/why-airwallex-customers-didnt-need-stablecoins-and-why-were-building-for-them-now) | 一手（官方博客，VP 署名） | 高 |
| [airwallex.com/global/blog/the-agentic-banking-era](https://www.airwallex.com/global/blog/the-agentic-banking-era) | 一手（CEO 博客） | 高 |
| [github.com/airwallex](https://github.com/airwallex) | 一手（官方 GitHub） | 高 |
| [stripe.com/use-cases/agentic-commerce](https://stripe.com/use-cases/agentic-commerce) | 一手（Stripe 官方，用于竞品对比） | 极高 |
| [adyen.com/press-and-media/adyen-agentic](https://www.adyen.com/press-and-media/adyen-agentic) | 一手（Adyen 官方，用于竞品对比） | 极高 |
| DuckDuckGo 搜索发现 Airwallex EPI 新闻稿 URL | 间接（搜索结果指向官方 newsroom URL） | 中（URL 发现，内容未完整验证） |

---

*底稿撰写：AI 调研助手，2026-10-20。本文件仅为内部参考，所有事实需结合最新官方来源二次核实后用于正式 SA 材料。*
