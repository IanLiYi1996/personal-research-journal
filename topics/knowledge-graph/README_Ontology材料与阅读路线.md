# Ontology 文档入口

整理日期：2026-10-23，按 Asia/Shanghai 日期口径。

## 给管理层阅读

新版完整调研 PDF：企业知识怎样进入 AI 与业务系统

[新版可编辑正文](管理层深度调研版/Ontology_企业知识与AI应用_深度调研报告.md)

这份是给零基础管理者的完整深度调研，包含决策摘要、16 章正文、4 个附录及 8 张图。从一个产品答疑问题出发，展开传统与当前构建方法、RDF／OWL 与属性图、虚拟图、数据库产品、版本治理、查询与推理、RAG／GraphRAG／Agent、公开案例、成本和试点验收。研究部分核查了六篇核心论文全文，并补核一篇 2026 年论文。

配套资料：

- [产品能力核实](管理层深度调研版/研究底稿/01_产品能力核实.md)
- [公开案例与证据](管理层深度调研版/研究底稿/02_公开案例与证据.md)
- [LLM 与本体研究证据](管理层深度调研版/研究底稿/03_LLM与本体研究证据.md)
- 论文结构化元数据
- [文件说明与排版复现](管理层深度调研版/README.md)

上一份 六页简明版 PDF 和 [简明版正文](管理层阅读版/Ontology_管理层说明.md) 保留，适合会前快速浏览，不替代新版深度报告。

## 技术说明

[Ontology 怎么建立、怎么存储、怎么用](Ontology_构建管理与应用_完整指南.md)

这是重新组织后的统一主稿，按建立模型、选择数据库、管理知识和接入应用展开。包含传统技术与 LLM 辅助、RDF／OWL，以及 Neo4j、Neptune、NebulaGraph、JanusGraph、ArangoDB、GraphDB、Stardog 和 Jena 的能力差异。

正文中直接解释代码与边界，无需来回阅读旧报告。

## 代码与验证

[参考代码和执行记录](technical-example/README.md)

RDF／SHACL／OWL-RL 的 13 项本地参考测试已执行通过；Neo4j、Neptune、NebulaGraph、Fuseki 和 Ontop 等数据库配置没有部署。示例目录明确列出两种状态。

## 核实记录

- [标准状态与构建方法](2026-10-23_Ontology标准与方法_核实附录.md)
- [传统构建与当前工具](2026-10-23_Ontology建立方法_补充核实.md)
- [存储与查询细节](2026-10-23_Ontology存储与查询_技术核实.md)
- [NebulaGraph 3.8.0 能力](2026-10-23_NebulaGraph本体管理_核实.md)
- [GraphRAG 查询、更新与实验边界](2026-10-22_GraphRAG_一手资料核实.md)

旧报告未删除；主稿归并其主要结论。需要查来源或历史组织方式时，可展开下面的旧导航。旧导航的核实范围和测试声明按对应早期资料理解，当前测试结果以上面示例目录为准。

<details>
<summary>早期报告与资料目录</summary>

这份导航把此前讨论、两份旧报告和新整理的一手资料放到同一阅读路径。覆盖构建到应用的主要环节，不声称穷尽所有行业本体或论文。

需要技术细节时，先读 [构建、图数据库管理与应用的技术实现版](2026-10-23_Ontology技术实现_构建_图数据库管理_应用.md)。它并列讲 Neo4j 原生元模型管理和 RDF／OWL 路线，包含代码、约束边界、存储配置、版本流程与应用接口；[示例目录](technical-example/README.md) 附有本地已执行的 13 项测试和明确未执行的数据库脚本。

只想先理解全过程，可读 [建立、存储管理与应用的易读版](2026-10-23_Ontology建立_存储管理_应用_易读版.md)。其他材料保留作参考。

## 1. 先读哪份

| 你现在要解决的问题 | 推荐入口 |
|---|---|
| 想看实际模型、约束、查询、存储配置和应用调用 | [技术实现版](2026-10-23_Ontology技术实现_构建_图数据库管理_应用.md) |
| 想顺着读懂建立、存储管理与应用的全过程 | [易读版](2026-10-23_Ontology建立_存储管理_应用_易读版.md) |
| 想一次梳理 Ontology 的发展、构建和应用 | [全景主报告](2026-10-23_Ontology从构建到应用_全景梳理.md) |
| 想看同一业务问题如何落到各技术层 | [贯穿案例：服务发布与依赖](2026-10-23_Ontology构建到应用_贯穿示例.md) |
| 想确认标准是否已经正式发布 | [标准与方法核实附录](2026-10-23_Ontology标准与方法_核实附录.md) |
| 想比较分类、本体、知识图谱和检索路线 | [前稿：知识图谱与 GraphRAG 调研](2026-10-22_Ontology_Taxonomy_知识图谱与GraphRAG调研报告.md) |
| 想检查微软 GraphRAG 的更新能力与研究实验 | [前稿：GraphRAG 核实附录](2026-10-22_GraphRAG_一手资料核实.md) |

旧稿保留历史核实日期和固定源码提交；新稿以 Ontology 生命周期为主线，没有覆盖或删除旧稿。

## 2. 三条阅读路线

### 路线 A：建立完整理解

全景报告第 0—4 节 → 第 5 节流程 → 贯穿案例 → 全景报告第 11—16 节应用与治理。

先看“对象与关系的共同定义”，再看语言、引擎和存储的职责，不从某个数据库产品开始。

### 路线 B：准备实施

先写真实 CQ → Stanford 方法 → 词表与模型 → 复用来源 → OWL 片段／数据形状 → 映射与身份 → 存储／访问 → 测试 → 查询服务 → 应用／动作。

每一步都用贯穿案例核对“输入是什么、谁负责、怎么验收”。不要求所有项目引入全部技术。

### 路线 C：研究当前发展

先读逐文档状态表 → LinkML／Schema Automator／OntoGPT → Ontop 的虚拟图 → GraphRAG／Graphiti → Palantir 对象与动作 → 安全与治理。

注意：语言标准进展、项目能力描述、厂商产品定义和本报告的工程建议，是四种不同证据。

## 3. 主题一：核心概念与入门方法

| 一手材料 | 读它解决什么 | 阅读时注意 |
|---|---|---|
| [Stanford：Ontology Development 101](https://protege.stanford.edu/publications/ontology_development/ontology101.pdf) | 范围、用途、能力问题 CQ、术语、复用与迭代 | 示例属于框架系统；槽约束不能直接等同 OWL 语义 |
| [W3C：OWL 2 Primer](https://www.w3.org/TR/owl2-primer/) | 类、属性、实例、公理、开放世界 | 本体不是数据库必填校验器 |
| [W3C：OWL 2 Overview](https://www.w3.org/TR/owl2-overview/) | OWL 文档和语义／语法的整体关系 | 先选所需表达和语义，不只选文件后缀 |
| [W3C：SKOS Primer](https://www.w3.org/TR/skos-primer/) | 词表、标签、概念层次与映射 | broader、类继承与对象身份分别处理 |
| [W3C：RDF Schema](https://www.w3.org/TR/rdf-schema/) | 类、属性、子类、子属性与 domain／range | domain／range 可产生类型蕴含 |

## 4. 主题二：表达、查询与正式推理

| 一手材料 | 作用 | 阅读时注意 |
|---|---|---|
| [RDF 1.1 Concepts](https://www.w3.org/TR/rdf11-concepts/) | 已发布数据模型基线、IRI、三元组与数据集 | 表示模型不是存储产品 |
| [SPARQL 1.1 Query](https://www.w3.org/TR/sparql11-query/) | 图查询、过滤、聚合和路径语法 | 路径匹配不等于完整 OWL 推理 |
| [OWL 2 Direct Semantics](https://www.w3.org/TR/owl2-direct-semantics/) | 精确理解公理、身份与蕴含 | 用于核对形式语义，而不是业务动作授权 |
| [OWL 2 Profiles](https://www.w3.org/TR/owl2-profiles/) | 选择 EL／QL／RL 片段 | 不是通用“高、中、低档”推理等级 |
| [Apache Jena：Inference](https://jena.apache.org/documentation/inference/) | 推理模型与查询结合的实现例子 | 内置 OWL 规则引擎不是完整 OWL DL |

Recommendation 说明标准地位，不说明某个引擎已经完整实现全部能力；语言片段与实现能力必须分别验证。

## 5. 主题三：构建、复用、模型生成与治理

| 一手材料 | 作用 | 阅读时注意 |
|---|---|---|
| [OBO Foundry Principles](https://obofoundry.org/principles/fp-000-summary.html) | 标识、范围、定义、关系、责任与维护 | 是 OBO 社区原则，不是全球强制标准 |
| [OBO：Versioning](https://obofoundry.org/principles/fp-004-versioning.html) | 可取回的版本与版本 IRI | 不悄悄改变旧标识所指含义 |
| [W3C：Publishing RDF Vocabularies](https://www.w3.org/TR/swbp-vocab-pub/) | 发布 URI、人读／机读表示和内容协商 | 2008 年 Working Group Note，不是 Recommendation |
| [W3C：Data on the Web Best Practices](https://www.w3.org/TR/dwbp/) | 词表复用、来源、标识和版本实践 | 面向数据发布，不是完整本体建模教程 |
| [Protégé](https://protege.stanford.edu/) | 编辑与检查本体的入口 | 编辑器不等于线上知识服务 |
| [ROBOT](https://robot.obolibrary.org/) | 自动化本体处理工作流 | CI 与治理制度仍需项目设计 |
| [LinkML](https://linkml.io/) | 声明式模型及多种模式／代码生成 | YAML Schema 不自动等价于 OWL |
| [Schema Automator](https://linkml.io/schema-automator/introduction.html) | 从已有模式／数据辅助构建模型 | 自动生成结果要领域复核 |
| [OntoGPT](https://github.com/monarch-initiative/ontogpt) | LLM 辅助结构化提取与本体 grounding | 输出候选不等于事实或公理已被证明 |

## 6. 主题四：数据映射、存储与访问

| 一手材料 | 作用 | 阅读时注意 |
|---|---|---|
| [W3C：R2RML](https://www.w3.org/TR/r2rml/) | 关系数据库到 RDF 的映射 | 支持物化或虚拟访问的实现思路，不要求先搬库 |
| [Ontop](https://ontop-vkg.org/guide/) | 虚拟知识图、SPARQL 转 SQL、轻量本体 | 数据仍可保留在原系统，具体源和查询需测试 |
| [Apache Jena：Fuseki](https://jena.apache.org/documentation/fuseki2/) | SPARQL 服务 | 查询端点不自动配置所需完整推理 |
| [Neo4j：Graph concepts](https://neo4j.com/docs/getting-started/appendix/graphdb-concepts/) | 属性图、节点、关系和 schema | 属性图 schema 与正式本体语义不同 |
| [Neo4j n10s 4.3：Inference](https://neo4j.com/labs/neosemantics/4.3/inference/) | 导入本体后做层次推断的示例 | 特定旧版文档示例，不当作最新版本部署推荐 |

关键判断：**Ontology 可以通过映射和查询作用于现有数据库，不需要只能存在于专门图数据库中。**

## 7. 主题五：质量、来源和安全

| 一手材料 | 作用 | 阅读时注意 |
|---|---|---|
| [W3C：SHACL](https://www.w3.org/TR/shacl/) | 指定数据图与 shapes 的符合性检查 | 通过验证不等于现实陈述正确 |
| [W3C：PROV-O](https://www.w3.org/TR/prov-o/) | 表达实体、活动、代理和来源关系 | 记录来源不自动证明来源可信 |
| [OWASP：Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) | 外部内容注入、最小权限与人工审批 | RAG 不能完全消除提示注入 |
| [Palantir：Ontology permissions](https://palantir.com/docs/foundry/ontologies/ontology-permissions/) | 类型定义、对象数据和动作权限的产品实例 | 资源可见与具体对象可读不是同一权限 |

## 8. 主题六：从查询到 AI 和业务运营

| 一手材料 | 作用 | 阅读时注意 |
|---|---|---|
| [Microsoft GraphRAG：Query overview](https://microsoft.github.io/graphrag/query/overview/) | Local／Global／DRIFT／Basic 等检索策略 | 社区摘要不是 OWL 推理或权威事实 |
| [Microsoft GraphRAG：固定提交 README](https://github.com/microsoft/graphrag/blob/c600ff80b77b9b62c47d539a3ac9569e95d8949b/README.md) | 项目维护和支持声明 | 声明针对具体项目，不针对所有图增强检索 |
| [Graphiti](https://github.com/getzep/graphiti) | 时序上下文图与类型化知识处理 | 框架及参考服务不替代生产认证、授权和防护 |
| [Palantir：Ontology overview](https://www.palantir.com/docs/foundry/ontology/overview) | 对象、关系、动作和运营层定义 | 厂商产品语境不等于 OWL 新版本 |
| [Palantir：Action Types](https://www.palantir.com/docs/foundry/action-types/overview/) | 对象模型如何连接业务修改 | 写动作需单独授权、审批与审计 |
| [W3C：SSN／SOSA](https://www.w3.org/TR/vocab-ssn/) | 观测、传感器、过程、结果与时间 | 领域本体可采用轻量核心＋扩展 |

不采用 README 中对竞争产品的宣传排名作为独立证据，也不把不同语料、模型和预算下的实验结果合成一个“最佳框架”排行榜。

## 9. 主题七：当前标准进展，按文档看

以下是本轮正文及带日期版本的核实状态；正文与出版历史存在差异的细节见核实附录。

| 文档 | 正文日期 | 正文标注状态 |
|---|---|---|
| [RDF 1.2 Concepts](https://www.w3.org/TR/2026/CR-rdf12-concepts-20260407/) | 2026-04-07 | Candidate Recommendation Snapshot |
| [RDF 1.2 Semantics](https://www.w3.org/TR/2026/CRD-rdf12-semantics-20260924/) | 2026-09-24 | Candidate Recommendation Draft |
| [RDF 1.2 Turtle](https://www.w3.org/TR/2026/WD-rdf12-turtle-20261007/) | 2026-10-07 | Working Draft |
| [SPARQL 1.2 Query](https://www.w3.org/TR/2026/WD-sparql12-query-20261011/) | 2026-10-11 | Working Draft |
| [SHACL 1.2 Core](https://www.w3.org/TR/2026/WD-shacl12-core-20261014/) | 2026-10-14 | Working Draft |

CR Snapshot、CR Draft 和 WD 均不等于最终 Recommendation。SHACL 的规则／节点表达式扩展和 SPARQL 的 Query／Update／Protocol 等也需要分别看状态，不根据一个系列名做统一判断。

当前材料里，本体基线仍使用已发布 OWL 2；模型或平台的“Ontology”宣传不是 OWL 版本号。

## 10. 最后记住的五条分界

1. 概念定义 ≠ 数据记录。
2. 图查询 ≠ 正式逻辑推理。
3. 形状通过 ≠ 事实真实。
4. 语义理解 ≠ 行动授权。
5. 新文档发布 ≠ 所有工具已经兼容。

## 11. 本材料包的边界

本次未进行产品基准复现、云成本测算、生产部署、完整许可审计或示例引擎执行。厂商／项目文档说明能力定位，不证明目标场景效果。

旧稿的经验计划和新稿的流程、模型、接口、测试门槛均是工程建议；没有把建议伪装成标准条款。

</details>
