# Ontology 怎么建立、怎么存储、怎么用

更新日期：2026-10-23。下面用一个产品知识库讲解，产品和数据均为虚构。代码、资料来源和实际测试结果都保留在文中及文末。

## 1. 从一句产品说明开始

假设手册里写着：“产品 A 支持批量推理。”

把这句话存下来很容易。但用户问“2.0 版在 Linux 上能不能用”时，这句话就不够了。它没有说版本，也没有说 Linux 下采用哪种部署方式。如果另一份资料写“仅云服务支持”，还要弄清两份资料是否谈论同一版本。

我们需要先约定几件事：产品和产品版本分开记录；“支持某功能”要附带环境条件；每条说法都能回到资料中的原位置。

把这些概念、关系和规则写清楚，就是建立领域本体，也就是这里讨论的 Ontology。若进一步采用 OWL，还能给某些规则规定严格的逻辑含义，让程序按它们推导结论。[R01]

这与选数据库是两件事。数据库负责存储和查询；本体说明存进去的对象和关系是什么意思。同一套业务定义可以放进 Neo4j、Neptune、NebulaGraph，也可以写成 RDF 文件，或通过映射查询现有 SQL 数据。[R06] [R19] [R20] [R21]

我们先为产品资料定五种对象：

```text
Product       产品，例如产品 A
Release       一次发布，例如产品 A 的 2.1 版
Capability    一项能力，例如批量推理
SupportClaim  关于支持情况的一条陈述
Evidence      支持这条陈述的资料
```

这里多出来的 `SupportClaim` 值得解释。它相当于一条带出处的记录：“A 的 2.1 版，在 Linux 自托管环境，支持批量推理。”版本、环境、有效时间和出处都跟着这条记录走。以后资料被纠正，能找到该修改哪条记录。

如果只在产品 A 上挂一条 `SUPPORTS` 边，2.0 和 2.1 的差别很容易丢失。“文档提到了某功能”和“产品确实支持该功能”也会混在一起。这是我们在这个知识库里采用陈述节点的原因。

正式本体不一定只含概念，OWL 也允许实例。这里把定义和具体资料分开管理，是为了让它们各自更新，方便追踪。[R01]

### 分类目录又放在哪里

如果还要给产品分“数据库”“推理引擎”等类别，可以另外维护分类词表。Taxonomy 主要做这种分类；SKOS 可以记录名称、别名和上下位概念。[R07]

但“产品属于数据库类别”“版本属于产品”“组件是产品的一部分”是三种关系。名字都可以叫“属于”，程序却不能把它们当成同一回事。设计本体时，关系的定义要比名称更具体。

## 2. 这套定义怎么做出来：过去的方法和现在的方法

### 先拿资料试着回答问题

我会先找几份真实手册和发布记录，人工回答几个问题：哪个版本支持批量推理？在什么环境？依据在哪？两份资料中的同名产品是不是同一个？

这样很快能看出模型缺什么。比如，手册中的“企业版”可能是一个产品系列，也可能是某版本的发行类型。此时要先查清业务含义，而不是看到这个词就创建一个类。

Stanford 的本体构建方法也是从用途、范围和要回答的问题开始，再检查能否复用已有本体，整理术语、类、属性和实例。类层次可以从大类往下细分，也可以从实际对象往上归纳。[R08]

问题要带预期答案，留作测试。假如系统把 2.1 的能力答成“所有版本都支持”，就能定位到版本定义、数据接入或查询出了问题。

### 传统人工建模做什么

领域专家负责解释概念，工程人员将它们写成程序能处理的形式。Protégé 是这类编辑工具的一个例子。[R09]

专家讨论的往往很具体：什么叫“最新版本”？版本号最大、发布日期最近，还是现在允许部署的版本？三种定义可能得到不同答案。先达成一致，程序才有确定的做法。

接着写关系。一个 Release 对应哪个 Product，一条 SupportClaim 指向哪个 Release 和 Capability。版本号是什么类型，哪些字段必填，一条陈述能挂几份证据，也在这个阶段确定。

下面的 JSON 展示一种项目内部的模型描述：

```json
{
  "model_version": "0.1.0",
  "types": ["Product", "Release", "Capability", "SupportClaim", "Evidence"],
  "relations": [
    {"name": "FOR_RELEASE", "source": "SupportClaim",
     "target": "Release", "min_targets": 1, "max_targets": 1},
    {"name": "ASSERTS", "source": "SupportClaim",
     "target": "Capability", "min_targets": 1, "max_targets": 1},
    {"name": "SUPPORTED_BY", "source": "SupportClaim",
     "target": "Evidence", "min_targets": 1}
  ],
  "properties": [
    {"owner": "Release", "name": "version",
     "datatype": "string", "required": true}
  ]
}
```

这不是标准本体文件，只是一种便于生成和检查的中间格式。程序读到它后，可以生成数据库支持的约束，也可以生成应用检查代码。团队使用 LinkML 等声明式模型工具时，同样需要说清生成结果承担什么工作。[R02]

### 大模型之前，已经有自动化

以前并非全部靠人逐句录入。GATE 的 JAPE 规则能按指定句式抽取关系；TermRaider 会找名词短语、做评分，提出术语和层次候选。[R10] [R11]

例如，一类规则可以专门识别“某版本支持某能力”的句式。识别到了，还要检查它是否处在否定句里，是否限定某种环境，以及两个名称到底指谁。规则减少重复阅读，但这些判断不会凭空消失。

结构化数据库更适合另一种办法。产品表和版本表已经有可靠主键，就把它们映射到产品、版本和关联关系。R2RML 定义了关系数据库到 RDF 的映射方法。[R12]

这些技术今天仍有用。对于已经明确记录的数据，让程序按映射读取，比让大模型重新猜一遍更容易检查。

### 现在，大模型可以帮到哪一步

大模型可以先读资料，整理术语、别名和候选关系，也可以按已经发布的模型提取具体记录。两种工作要分开：提出“应该增加一个发行类型”，是在改模型；提取“A 2.1 属于企业版”，是在填事实。

OntoGPT 主要按既定模式提取信息并对齐本体标识。它也能在预先配置的模板下输出 OWL 公理，模板先规定允许的逻辑结构。Schema Automator 则能辅助从已有数据或模式生成、补充模型。[R13] [R14]

在这个知识库里，我会让模型交出如下记录：

```json
{
  "release_uri": "https://example.org/onto-tech/Release21",
  "capability_uri": "https://example.org/onto-tech/BatchInference",
  "environment": "linux-selfhosted",
  "quote": "A 2.1 在 Linux 自托管环境支持批量推理。",
  "document_version_id": "demo-doc-v3",
  "locator": "部署支持，第2段"
}
```

审核状态由后端填写，模型不能自行把它写成“已批准”。后端先检查字段和允许的关系类型，再检查引文是否存在、对象是否识别正确、版本和条件是否保留。

还要读上下文。模型引用的那句话可能确实存在，但前一段写着“以下仅适用于预览版”。仅验证引文字符串，就会漏掉这种问题。

新提议的概念先留在候选区，经过评审后发布成新模型版本。否则今天一份手册叫它“批处理”，明天另一份叫“批量推理”，数据库里就会长出两种未经确认的关系。

## 3. 定义和事实都可以存成图

### 把“类型定义”当成数据保存

图数据库里可以存产品，也可以存“什么是产品”这份定义。通常会建立一组专门的节点：

```text
OntologyVersion     这套定义的版本
EntityType          对象类型，例如 Release
PropertyDefinition  属性要求，例如 version 必填
RelationType        关系含义及允许的端点
```

用关系连接它们：

```text
OntologyVersion → CONTAINS_TYPE → EntityType
EntityType → HAS_PROPERTY → PropertyDefinition
EntityType → SUBCLASS_OF → EntityType
RelationType → FROM_TYPE / TO_TYPE → EntityType
```

这通常叫“元模型”：它保存的是模型的定义。业务对象可以通过 `INSTANCE_OF` 指向相应类型。于是，查询程序既能找出某产品的发布记录，也能找到 Release 的定义和属性要求。

实际业务数据则按前面的含义连接：

```text
Release → RELEASE_OF → Product
SupportClaim → FOR_RELEASE → Release
SupportClaim → ASSERTS → Capability
SupportClaim → SUPPORTED_BY → Evidence
```

这套组织方式可用于不同属性图数据库。不过，具体建库语句和约束要针对产品编写。

### 保存了一条规则，为什么还不能挡住坏数据

假如你在 PropertyDefinition 节点上写：

```text
name = version
datatype = STRING
required = true
```

对数据库来说，这只是三个属性值。它还不知道你希望所有 Release 节点必须填写版本号。

要让这条定义生效，模型管理程序得继续做一件事：如果数据库支持相应约束，就生成约束；如果不支持，就让写入服务检查。关系数量、合法端点、证据完整性也这样处理。

这个转换过程最好有明确记录。例如，模型的 `version.required=true` 被转换成了哪条 DDL，或哪段验证代码。以后改模型，才知道哪些地方要一起改。

### 名称和标识为什么分开

“产品 A”是显示名称。用它做主键，改名就可能影响所有关联记录；两个不同产品恰好同名时，还可能被合并。

因此示例使用稳定 URI 或数据库元素 ID 定位对象，名称另存。类型定义还带模型版本：0.1.0 版 Release 的定义，与后来的定义可以分别查到。

这不意味着见到两个标识就必然是两个真实对象。合并身份需要证据。尤其采用 OWL 时，`sameAs` 是强身份声明，不能拿来表示“名字很像”。[R01]

## 4. 换了数据库，哪些实现会跟着变

同样要求“版本号必填”，一种数据库能直接拒绝缺字段的记录，另一种可能要由应用检查。同样查询“所有 Release”，一种实现走类型层次，另一种可能事先给实例加上父类标签。

所以，业务定义可以共用，数据库实现要分别处理。下面沿用产品知识库，看几个具体例子。代码的执行情况集中列在最后，避免把说明样例误认为已部署的系统。

### Neo4j：把定义变成约束

Neo4j 用节点、标签、关系和属性保存图，通过 Cypher 查询。[R19]

可以先把 Release 的定义存进去：

```cypher
MERGE (v:OntologyVersion {uri:'urn:onto-tech:model:0.1.0'})
MERGE (t:EntityType {
  modelVersion:'0.1.0',
  iri:'https://example.org/onto-tech/Release'
})
SET t.name='Release', t.definition='A particular product release.'
MERGE (v)-[:CONTAINS_TYPE]->(t);
```

这里建立的是定义节点。它与业务图中的 `(:Release)` 节点有关联，但不会凭这几行就检查所有发布记录。

例如，防止 Release 的 URI 重复，要再创建：

```cypher
CREATE CONSTRAINT release_uri IF NOT EXISTS
FOR (r:Release) REQUIRE r.uri IS UNIQUE;
```

社区版支持属性唯一性；属性存在性、类型和键约束有企业版要求。上面的唯一性约束也没有要求每个节点必须填写 URI，缺失值要另外处理。[R03]

Neo4j 企业版的 Graph Types 提供了更集中的 schema 声明，使用 Cypher 25，从 2026.06 起正式可用。例如：

```cypher
CYPHER 25
ALTER CURRENT GRAPH TYPE ADD {
  (:Release => {
    uri :: STRING NOT NULL,
    version :: STRING NOT NULL
  })
}
```

这时 `NOT NULL` 才是数据库要执行的要求，而不只是定义节点里的文字。Graph Types 当前是开放型的：它约束声明覆盖的数据，未声明的节点、关系和属性并不会全部被拒绝。[R27]

改 schema 时尤其要留意命令含义。`ADD` 增加新 element，不直接修改已有 element；`SET` 会替换原来的 graph type 或 constraints。同步模型前，要先读当前定义，确认迁移会影响哪些数据。[R28] [R29] [R30]

另一些要求可以由查询检测。假设一条 SupportClaim 必须只连接一个 Release：

```cypher
MATCH (c:SupportClaim)
OPTIONAL MATCH (c)-[link:FOR_RELEASE]->(r:Release)
WITH c, count(link) AS targets
WHERE targets <> 1
RETURN c.uri, targets;
```

返回的就是违规记录。写入服务应将检查与写入组织在合适的事务里，失败时回滚；有并发写入时，还要防止两个请求分别检查通过、最后却一起写出两条关联。锁、修订检查或串行化需要按实际负载测试。

完整脚本在 [neo4j-model.cypher](technical-example/neo4j-model.cypher)、[查询脚本](technical-example/neo4j-query.cypher)和[检查脚本](technical-example/neo4j-validate.cypher)。其中时间、环境、审核状态、租户和证据都参与查询。

### Neptune：同一个模型，更多检查留在应用里

Amazon Neptune Database 支持属性图与 RDF。属性图可以用 Gremlin 或 openCypher，RDF 则用 SPARQL。[R20]

采用属性图时，仍然可以存 EntityType、RelationType 和业务实例。但官方迁移文档说明，可用 schema 约束是节点和边的 ID 唯一性，没有额外的属性唯一性、值或 schema 约束。[R05]

这会改变实现方式。`uri` 属性不是数据库元素 ID；给它填一个业务 URI，不会自动防止重复。导入程序需要确定物理 ID，并检查重复、必填字段、合法端点和关系数量。Neo4j 的约束 DDL 也不能原样搬过来。

查询可以按相同业务条件写成 openCypher。下面把时间先转换成 UTC 毫秒整数，减少对特定日期函数的依赖：

```cypher
MATCH (c:SupportClaim)-[:FOR_RELEASE]->(r:Release),
      (c)-[:ASSERTS]->(k:Capability),
      (c)-[:SUPPORTED_BY]->(e:Evidence)
WHERE c.model_version=$modelVersion
  AND c.environment=$environment
  AND c.status='approved'
  AND c.valid_from_ms <= $atMs
  AND (c.valid_to_ms IS NULL OR $atMs < c.valid_to_ms)
  AND k.uri=$capabilityUri
RETURN r.uri, r.version, c.uri, e.uri, e.quote;
```

`$atMs`、`$environment` 等值由后端绑定。日期类型和函数也可以按所用 engine 的支持选择；这里使用整数，不是在断言 Neptune 没有时间支持。其他 openCypher 功能同样要查兼容说明。[R33]

这段代码只突出查询条件，没有展示完整授权过滤。实际接口还要确定用户能看哪些对象和证据，配置网络、连接、凭据，以及启用时的 IAM 签名。

还有一个容易漏的区别：Neptune 的 SPARQL 默认图是全部 named graphs 的并集。不指定 GRAPH 等范围时，查询会考虑数据库里的全部 triples。[R34]

如果把不同租户的数据仅放在不同图名下，却让应用随意查默认图，就可能取回超出预期的数据。图名可以组织数据，权限仍要明确实施，摘要和缓存也一样。

Neptune Analytics 是另一个服务。它面向图分析，使用 openCypher；它能导入 RDF N-Triples，并将其映射成属性图。[R31] [R32] [R64] 数据导入成功后，仍需检查原来的 RDF 语义和图上下文保留了多少，而不是将它与 Database 的 RDF 路线视为相同实现。

### NebulaGraph：先定义 Tag 和 Edge，再写实例

以下按公开 3.8.0 文档说明，其他发行版或企业版需要另查。[R21]

NebulaGraph 用 Space 组织图数据。创建 Space 时指定顶点 ID，即 VID 的类型，也设置分区和副本。一个 VID 在 Space 内唯一，不是每个 Tag 各有一套主键。顶点可以有多个 Tag；Tag 定义属性，Edge type 定义边和属性。[R36]

下面是一组最小示例：

```ngql
CREATE SPACE ontology_demo(
  partition_num=1,
  replica_factor=1,
  vid_type=FIXED_STRING(128)
);
```

这里的单分区、单副本只为演示。确认 Space 生效后，再定义字段：

```ngql
USE ontology_demo;
CREATE TAG Product(name string NOT NULL);
CREATE TAG Release(version string NOT NULL, model_version string NOT NULL);
CREATE EDGE RELEASE_OF();
```

Tag 的 `NOT NULL` 是 schema 要求。等 schema 生效后，才写产品和版本：

```ngql
INSERT VERTEX Product(name)
VALUES "product:A":("产品A");
INSERT VERTEX Release(version, model_version)
VALUES "release:A:2.1":("2.1", "0.1.0");
INSERT EDGE RELEASE_OF()
VALUES "release:A:2.1"->"product:A"@0:();

GO FROM "release:A:2.1" OVER RELEASE_OF
YIELD dst(edge) AS product_vid;
```

最后一条查询沿 `RELEASE_OF` 找产品。这里的 CREATE 创建 schema，INSERT 写实例；nGQL 只兼容部分 openCypher 查询语法，不能照着 Neo4j 的写法直接执行。[R38] [R39]

同样可以定义 EntityType、OntologyVersion 等 Tag，保存本体定义。但 Tag／Edge 的字段类型、NULL 和 DEFAULT，并没有替应用解释概念含义、检查业务基数或证据完整性。[R04] [R37]

该版本还允许悬空边。同一 VID＋Tag、或同一边身份的写入可能覆盖已有值，所以导入器要检查端点和更新意图。[R40] [R41] 边身份中的 rank 可以区分多条边，不自动表示版本或证据。条件多时，仍可保留独立 Claim 节点。

运维上，graphd 处理连接、认证和查询，metad 管 Space、schema 和分区等元数据，storaged 管数据与副本。[R42] 分布式组件带来了不同的运维工作，不能仅据此判断它一定比另一产品快。

公开 3.8.0 文档不支持 RDF／SPARQL；这节用的是属性图加 nGQL 的做法，不外推未来版本的能力。[R38]

### 其他产品也各有自己的规则

JanusGraph 的 schema 可以给边定义 multiplicity。例如 MANY2ONE 能限制某标签的出边数量。端点约束还受 `schema.constraints`、`schema.default` 配置影响，存储和索引后端也需要一起考虑。[R22] [R43]

所以，不能一概说“关系数量只能在应用层检查”。应查清选定产品能执行哪些规则，以及需要哪些配置。

ArangoDB 用文档集合和边集合组织图。边文档通过 `_from`、`_to` 引用其他文档，查询使用 AQL。[R23] 产品、版本和证据可以按这个方式保存，接入程序则要适应文档式表示。

GraphDB 和 Stardog 更适合放到 RDF 与推理这一侧讨论。GraphDB 有可配置规则集，Stardog 的推理引擎支持 OWL 2 profiles 的表达及相应模式设置。[R24] [R25] 真要使用时，列出要处理的公理和数据，测试所选配置，别仅凭产品名称判断推理覆盖。

选型时可以先用下面这张表找方向，再用自己的数据测试：

| 路线 | 数据和查询 | 需要特别确认的事 |
|---|---|---|
| Neo4j | 属性图、Cypher | edition、约束、Graph Types 与应用校验 |
| Neptune Database | 属性图／RDF；Gremlin、openCypher、SPARQL | ID、查询兼容、默认图和授权范围 |
| NebulaGraph 3.8.0 | Tag／Edge、nGQL | VID、schema 生效、端点检查和运维 |
| JanusGraph | 属性图、Gremlin | multiplicity、配置和存储／索引后端 |
| ArangoDB 3.12 文档 | 文档和图、AQL | 集合、边引用及相应查询 |
| GraphDB、Stardog | RDF、SPARQL、推理 | 实际规则或公理覆盖及更新方式 |
| Jena Fuseki＋TDB2 | RDF dataset、SPARQL | 持久存储和推理分别配置 |

这里没有性能或价格排名。跨产品比较时，连同应用补做的验证、推理和运维工作一起计算，结果才有意义。

## 5. RDF、OWL、SHACL 分别帮你做什么

### RDF 是一种表达数据的办法

前面用节点和边存产品记录。RDF 则将一条陈述写成“主体、关系、客体”三个部分。例如，产品 A 的 2.1 版属于产品 A，可以写成：

```turtle
@prefix ex: <https://example.org/onto-tech/> .

ex:Release21 ex:releaseOf ex:ProductA .
```

`ex:` 是一个缩写前缀。`ex:Release21` 是版本的标识，`ex:releaseOf` 是“属于哪个产品”的关系，`ex:ProductA` 是产品标识。这三个标识写清楚后，不同来源可以围绕同一对象连接数据。[R15]

版本号则是一个值：

```turtle
ex:Release21 ex:version "2.1" .
```

上面这种文本写法叫 Turtle。RDF 是数据模型，Turtle 是表达它的一种文件格式。用 TriG 还可以给图命名，把多份图放进同一个数据集。[R16]

完整产品记录仍保留之前定义的支持声明：

```turtle
@prefix ex:  <https://example.org/onto-tech/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:Release21 a ex:EnterpriseRelease ;
    ex:releaseOf ex:ProductA ;
    ex:version "2.1" .
ex:Claim21 a ex:SupportClaim ;
    ex:forRelease ex:Release21 ;
    ex:capability ex:BatchInference ;
    ex:environment "linux-selfhosted" ;
    ex:status "approved" ;
    ex:validFrom "2026-10-20T00:00:00+08:00"^^xsd:dateTime ;
    ex:hasEvidence ex:Evidence21 .
```

`a` 表示“是这种类型的实例”。`xsd:dateTime` 告诉程序，这个值是日期时间。后面的查询才有办法按时间比较，而不是按普通字符串猜含义。

完整数据在 [data.trig](technical-example/data.trig)。模型定义在 [ontology.ttl](technical-example/ontology.ttl)。这样，一批新资料进来时可以只更新事实；修改类和关系时，则单独发布模型。

### OWL 让某些定义能够用于推理

如果数据写着“Release21 是 EnterpriseRelease”，用户却查询“所有 Release”，程序怎样知道要把它算进去？

可以在 OWL 中声明 EnterpriseRelease 是 Release 的子类：

```turtle
@prefix ex:   <https://example.org/onto-tech/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .

ex:EnterpriseRelease rdfs:subClassOf ex:Release .
ex:releaseOf owl:inverseOf ex:hasRelease .
```

第一条允许相应推理器从 EnterpriseRelease 类型推出 Release 类型。第二条说明两种关系是逆关系：版本属于产品，反过来就是产品拥有这个版本。[R17]

这些结论从已有定义和数据推导而来。它们与读取一条存好的边不同，也与模型根据名称猜测不同。

OWL 还有一个与普通数据库检查差别很大的地方：未记录的信息保留为未知。手册没提批量推理，并不直接说明产品不支持；两个不同标识也不默认指两个不同对象。[R01]

所以，需要区分“有支持证据”“有反对证据”“还不知道”。我不会仅因为搜索没找到，就往图里写一条“不支持”。

OWL 2 的 EL、QL、RL 是不同的语言片段，各自限制可用表达，以适应不同推理任务。例如 QL 适合一类查询改写，RL 适合规则式实现。先列出需要的公理，再选实际支持它们的引擎。[R46]

### SHACL 检查的是提交的数据

但我们还需要回答另一个问题：某条 Release 记录没有版本号，能不能入库？

这类要求用数据校验处理。采用 RDF 时，可以写 SHACL：

```turtle
@prefix ex:  <https://example.org/onto-tech/> .
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:ReleaseShape a sh:NodeShape ;
  sh:targetClass ex:Release ;
  sh:targetSubjectsOf ex:releaseOf ;
  sh:property [
    sh:path ex:version ;
    sh:minCount 1 ; sh:maxCount 1 ;
    sh:datatype xsd:string ; sh:minLength 1
  ] .
```

这段要求 version 只有一个、必须存在、是非空字符串。`targetSubjectsOf` 还会找出有 releaseOf 关系的记录，避免一条数据因为漏了类型，连检查名单都没进入。[R18]

完整 [shapes.ttl](technical-example/shapes.ttl) 会继续检查端点、证据、环境、状态和时间区间。SHACL 通过后，我们只知道数据满足了这些结构要求，还要单独确认资料确实支持它。

校验顺序也要定下来。本地示例先校验原始数据，再推理。如果先用 domain／range 推导补出类型，原来的缺失可能被盖住。你的业务若允许推导类型，就选相应流程并记录配置。[R18] [R49]

参考脚本实际采用的是下面的顺序：

```python
ok, _, _ = validate(
    data_graph=authorized_view,
    shacl_graph=shapes,
    inference="none",
    meta_shacl=True,
    inplace=False,
)
if not ok:
    raise ValueError("数据没有通过形状检查")

view = clone(authorized_view)
DeductiveClosure(
    OWLRL_Semantics,
    axiomatic_triples=False,
    datatype_axioms=False,
).expand(view)
```

`authorized_view` 是模型和用户获准事实组成的图。`clone` 是参考脚本中复制图的函数：先留住来源数据，再在副本里增加推导出来的三元组。这里使用 owlrl 的 OWL-RL 实现；需要其他逻辑表达时，还要检查引擎是否支持。[R47]

### 到了查询时，差别能实际看出来

本地样例中，Release21 没有直接写成 Release，只写了 EnterpriseRelease。直接查 `?release a ex:Release`，没有当前符合条件的记录；执行相应推理后，会找到一条。

也可以把查询写得更明确，不先运行推理：

```sparql
?release rdf:type/rdfs:subClassOf* ex:Release .
```

它沿类型和子类关系寻找 Release。不过，这条路径没有执行逆关系或等价类等其他公理，只是解决了当前的类型查询。[R48]

属性图也有对应做法：导入时给对象补上父类标签，或者查询定义图里的类层次。选择哪种做法，要写进程序和测试。

查询支持情况时，完整条件还包括：

```sparql
PREFIX ex: <https://example.org/onto-tech/>
SELECT DISTINCT ?claim ?release ?version ?evidence ?quote
WHERE {
  ?claim a ex:SupportClaim ;
      ex:forRelease ?release ;
      ex:capability ?capability ;
      ex:environment ?environment ;
      ex:status "approved" ;
      ex:validFrom ?from ;
      ex:hasEvidence ?evidence .
  ?release a ex:Release ; ex:version ?version .
  ?evidence ex:quote ?quote .
  OPTIONAL { ?claim ex:validTo ?to . }
  FILTER (?from <= ?at)
  FILTER (!BOUND(?to) || ?at < ?to)
}
```

后端绑定能力、环境和查询时刻。在这个系统的时间约定里，区间是“起点包含、终点不包含”，validTo 为空表示尚未记录结束时间。这样的时间处理是应用作出的约定，OWL 不会替业务选择。

查询只在用户可见的图上执行。不能先用全部租户数据推理，再从答案中删几条引用。

## 6. 文件、持久存储和现有数据库怎样连接

### 定义先存成文件也可以

模型还小的时候，可以把类、关系和公理保存为版本化文件，把校验规则和映射分开维护：

```text
ontology.ttl   类型、关系和公理
shapes.ttl     录入与发布检查
mappings/     原系统怎样接入
tests/        样例和预期答案
```

便于编辑不等于能在线服务。业务事实持续增长后，还要选择持久存储、查询接口、备份和访问控制。定义文件和测试样例可以随版本保存，原始资料和大量业务数据则按各自的规模管理。

### 用 Fuseki 和 TDB2 提供 RDF 服务

Jena Fuseki 提供 SPARQL 服务，TDB2 可以将数据保存到磁盘。示例配置是：

```turtle
@prefix fuseki: <http://jena.apache.org/fuseki#> .
@prefix tdb2: <http://jena.apache.org/2016/tdb#> .

<#service> a fuseki:Service ;
  fuseki:name "onto-demo" ;
  fuseki:serviceQuery "sparql" ;
  fuseki:serviceUpdate "update" ;
  fuseki:serviceReadWriteGraphStore "data" ;
  fuseki:dataset <#store> .
<#store> a tdb2:DatasetTDB2 ;
  tdb2:location "/var/lib/fuseki/onto-demo-data" .
```

service 名决定 URL 路径；dataset 指向持久存储。完整 [fuseki-config.ttl](technical-example/fuseki-config.ttl) 还包含服务列表。目录是示例路径，部署时须设置所有权、备份和读写权限。[R26]

按此配置，查询在 `/onto-demo/sparql`，更新在 `/onto-demo/update`，图文件接入在 `/onto-demo/data`。Graph Store POST 合并目标图，PUT 替换目标图，使用 PUT 前必须确认目标和内容。[R44]

保存到磁盘以后，推理仍要另外配置。Jena 内置 OWL 规则引擎不等于完整 OWL DL；包装默认图的推理，也不会自动证明所有 named graphs 都有相同配置。[R45]

### 数据已经在 SQL 里，可以不搬走

如果产品和版本都由原系统管理，还可以让本体通过映射作用于它们。Ontop 的虚拟知识图会将 SPARQL 改写成 SQL，到原数据库查询。[R06]

映射说明“这张表怎样成为图中的对象”：

```text
mappingId ReleaseMapping
target <https://example.org/release/{release_id}> a ex:Release ;
       ex:version {version} ;
       ex:releaseOf <https://example.org/product/{product_id}> .
source SELECT release_id, product_id, version FROM product_release
```

source 是 SQL，target 是 RDF 模板。表里的主键生成版本标识，product_id 生成产品关联。还需要完整配置文件、数据库连接和标识规则。

Ontop 使用轻量 RDFS／OWL 2 QL 路线。这种做法减少了数据复制，但源库负载、改写后的查询和权限传播都需要测试。[R06]

因此，“采用本体”不必直接变成“增加一个图数据库并搬入全量数据”。是否需要物化，取决于现有系统和实际查询。

## 7. 数据变了以后，这套定义怎么维护

### 产品升级和模型修改，分别记版本

产品 A 发布 2.2 版，是新增一份业务数据。给它一个新的 Release 标识，记录能力和资料即可，不必因此修改“什么是 Release”。

如果团队决定以后每条支持声明还要记录最低内存要求，这才是模型变化。要发布新的定义，决定旧资料怎样补齐，哪些接口需要更新，哪些检查需要重新跑。

这里至少有两种版本：产品版本，例如 2.1；模型版本，例如 0.1.0。把它们放在同一个 version 字段，后面会很难查。

模型发布时，我会一起保存定义、校验、映射、查询模板和推理配置，附上测试及迁移说明。OBO 的版本原则要求正式版本可以标识并取回，值得借鉴。[R50]

图里可以同时留下 0.1 和 0.2 的定义，但数据库当前执行的 schema 仍是具体配置的那一套。新约束如何处理旧记录，必须在迁移里决定，存两个版本节点不会自动解决它。

### 资料出处要能找回原件

证据不宜只留一个网页链接。网页可能改，解析器也可能把段落重新切分。我会保存文档快照、内容哈希、解析器版本、引文和原位置。

如果资料是 PDF，还要说明定位指的是页码，还是清洗后文本的字符偏移。二者往往不是同一坐标。用户点击引用时，需要能回到对应内容。

PROV-O 可以记录资料、处理活动及处理者的来源关系。[R51] 这些记录帮助追查“这条数据怎么来的”，至于资料有没有说对，仍要核对原文。

类型定义也应配有人能阅读的文档，并保留机器能处理的表示。W3C 的词汇发布方法讨论了稳定标识、版本和这两种表示，可用于检查发布过程。[R52]

### 一份资料被纠正，后面哪些东西跟着改

假设原手册后来纠正了“A 2.1 支持批量推理”的说法。先找到由这份资料支持的 Claim，修订或撤销它，再检查查询视图、推论、便捷边、向量索引、摘要和缓存。

如果另一份独立资料仍支持同一结论，就不能简单删除整个事实。系统要知道每条陈述分别依赖哪些证据。

这个记录不完整时，最容易出现的问题是：原文已经改了，机器人还从旧摘要里回答。小系统可以重算受影响视图；大系统再实现细粒度的依赖追踪。

微软 GraphRAG 的旧核实记录也提醒了这一点：固定提交中的默认新增文档流程，不等于同名正文更新和删除传播都已完成。使用某个框架前，要实际测试这些操作。[R54]

历史版本、资料纠错、内容删除和权限撤销还应分开。2.1 版的历史记录可能需要保留；被要求删除的正文则按保留政策处理；用户失去权限后，已有缓存也要重新授权。

### 写入流程要有一个负责的入口

我会让写入服务先绑定登录身份和模型版本，再检查字段、对象标识、证据、端点和关系数量，最后提交并记录审计。候选记录由审核流程改为 approved，模型不能自己完成这一步。

并发也要考虑。两个请求可能同时看见同一份旧记录，分别检查通过后又一起写入。修订条件、锁或串行化该用哪一种，取决于数据库和业务，不能只靠一次检查查询。

如果正文、图数据和工单保存在不同系统里，还需要处理部分成功、部分失败的情况。图数据库的事务只负责它覆盖的范围。

## 8. 应用怎样使用这些数据

### 先做一个能确定回答范围的查询接口

不管底下选了哪种数据库，应用可以先暴露同一个业务接口：

```text
POST /v1/support-claims/search
```

```json
{
  "capability_uri": "https://example.org/onto-tech/BatchInference",
  "environment": "linux-selfhosted",
  "at": "2026-10-23T10:00:00+08:00",
  "model_version": "0.1.0"
}
```

后端确认身份和参数，选择用户能看的数据，再执行相应查询。这一层适配代码负责 ID、时间表示、查询语言和范围。换数据库时，它要跟着调整，而不是只换连接地址。

返回结果应带产品版本、条件、Claim、证据和模型版本。若结论来自推理，也说明所用定义及输入版本。这样，页面、报表和模型都能使用同一份结果。

返回为空时，也要说清含义。可能是没找到支持证据，也可能是资料未覆盖或用户无权查看，不能一律显示“不支持”。收集了部分手册时，查到五个对象也只能说找到了五个。

这个接口可以用来做版本对比、跨系统整合、依赖查询或条件过滤。图路径能帮助列出关联对象，但真实影响还取决于环境和业务行为；仅“走得到”不足以证明故障必然传播。

### 接 RAG 时，让模型解释已经查清的内容

用户问“哪些版本在 Linux 自托管环境支持批量推理”，先解析功能和环境，调用查询接口，再取回对应原文。把已知事实、冲突和缺失说明一起交给模型，让它写答案。

如果直接搜“支持批量推理”几个字，可能取到云服务版的段落。把环境作为明确条件交给查询，至少能让系统按定义筛选，而不是等生成答案时再碰运气。

对于简单查原文的任务，先测试关键词和向量检索即可；确实需要对象整合、关系查询时，再加对应能力。全库主题总结则可以另测社区报告等方法。Microsoft GraphRAG 的 Local、Global、DRIFT、Basic 是不同检索方式，不是 OWL 推理。[R53]

Graphiti 等框架可帮助处理时序事实、来源和类型化上下文，但认证、权限、删除和防护仍要结合自己的系统实现。[R55] 要验证的是机器人在资料更新后能否改口，而不只是它是否多了一个 memory 功能。

### 业务动作放到另一类接口

假如用户接着要求“创建版本升级评审单”，本体帮助明确涉及哪个产品、版本、依赖和证据。真正创建评审单，应交给业务动作服务。

Palantir 的 Ontology 和 Action Types 展示了对象模型连接业务修改的一种产品路线。[R56] [R57] 自己实现也可以采用这个分工。

动作请求可包含：

```json
{
  "action": "propose_release_upgrade",
  "release_uri": "https://example.org/onto-tech/Release21",
  "expected_revision": "revision-demo-7",
  "idempotency_key": "proposal-demo-001",
  "evidence_uris": ["https://example.org/onto-tech/Evidence21"]
}
```

`expected_revision` 用来确认对象没在计划期间被别人改过；`idempotency_key` 用来处理重复提交。服务还要检查权限、参数和审批要求，记录执行结果，失败时恢复或补偿。

模型可以提议做事，不能通过写“我是管理员”获得权限。外部文档里的指令也不应成为工具授权。OWASP 提醒，RAG 不会完全解决提示注入，高风险操作仍需最小权限和必要审批。[R58]

实现时可以先只提供只读查询、静态模板和参数绑定。需要模型生成查询，再增加解析、读写分类、白名单和资源限制。权限要覆盖返回的关系、引文、摘要和缓存，不能只过滤原始文件。

## 9. 做到什么程度，才值得继续投入

先选一个小范围，用真实资料回答一组固定问题。我会把文档检索作为对照，再逐步加术语统一、身份解析、关系查询或推理，看具体哪一步解决了错误。

比如，同名产品误合并了，先修身份规则；环境条件丢了，修抽取和查询；类查询漏结果了，检查是否需要路径或推理。不要遇到每个问题都直接换大模型。

跨数据库比较时，固定数据、权限、问题和预期答案，还要计入外部校验、推理、审核、索引和运维工作。某产品原生完成的规则，换一个产品可能由应用补上，比较时要把这部分算进去。

下面是现有参考代码实际做过的验证，而不是数据库性能测试。

### 已执行的参考测试

本地环境使用 Python 3.14.5、RDFLib 7.6.0、pySHACL 0.40.1 和 owlrl 7.6.2，在临时环境跑过 13 项测试，全部通过，没有修改系统 Python。

```text
获准数据视图：86 triples
OWL-RL 扩展后：330 triples
SHACL conforms：true
当前条件结果：Claim21 / Release21 / Evidence21
13 tests，0 failures，0 errors
```

它们检查缺证据、缺版本、错误类型和时间区间，也检查类继承、逆关系、条件过滤、撤销和受限图排除。Release21 的直接类型查询、类路径查询和推理后查询，结果差异也经过测试。

代码在 [参考目录](technical-example/README.md)，结果在 verification-results.json。其中的图白名单是固定测试数据，不是完整的多租户权限系统。

Neo4j、Neptune、NebulaGraph、Fuseki／TDB2、Ontop 和 HTTP API 未部署或运行。正文中的数据库脚本供实施时对照，只有明确列出的本地部分经过执行。

### 标准和产品版本怎样确认

OWL 2、RDF、SPARQL 和 SHACL 有已发布标准可用作基线。1.2 系列则要逐份看状态：

| 文档 | 所引正文日期 | 状态 |
|---|---|---|
| RDF 1.2 Concepts | 2026-04-07 | Candidate Recommendation Snapshot |
| RDF 1.2 Semantics | 2026-09-24 | Candidate Recommendation Draft |
| RDF 1.2 Turtle | 2026-10-07 | Working Draft |
| SPARQL 1.2 Query | 2026-10-11 | Working Draft |
| SHACL 1.2 Core | 2026-10-14 | Working Draft |

CR 和 WD 还不是最终 Recommendation。部分正文与出版历史日期也有差异，详情在 [标准核实记录](2026-10-23_Ontology标准与方法_核实附录.md)。使用新特性前，要在目标引擎验证导入、查询、导出和迁移。[R59] [R60] [R61] [R62] [R63]

NebulaGraph 用的是公开 3.8.0 文档，ArangoDB 用 3.12 文档，GraphDB 来源为 11.2 文档。其他发行版、edition 或部署方式应再核对。这里没有做完整许可证审计、云成本报价或跨产品性能复现。

开始实施时，可以先把问题、定义、资料、查询和测试连起来。等它能稳定回答实际问题，再决定是否需要更多推理、另一种存储或业务动作。旧报告和核实记录保留在 [材料导航](README_Ontology材料与阅读路线.md)，完整代码单独查阅即可。

## 资料出处

正文编号指向标准或工具维护方的一手材料。实现步骤和示例模型是这份指南给出的做法，厂商文档说明的能力不等于目标场景已测试通过。

建模与语言可查 [R01] [R07] [R08] [R15] [R17] [R18] [R46]；传统和当前辅助工具可查 [R10] [R11] [R12] [R13] [R14]。产品文档及版本约束在对应章节标注，来源、更新和应用资料可查 [R50] [R51] [R52] [R53] [R56] [R58]。

[R01]: https://www.w3.org/TR/owl2-primer/ "OWL 2 Primer"
[R02]: https://linkml.io/ "LinkML"
[R03]: https://neo4j.com/docs/cypher-manual/current/schema/constraints/ "Neo4j constraints"
[R04]: https://docs.nebula-graph.io/3.8.0/3.ngql-guide/10.tag-statements/1.create-tag/ "NebulaGraph 3.8 Tag schema"
[R05]: https://docs.aws.amazon.com/neptune/latest/userguide/migration-compatibility.html "Neptune schema constraints and migration"
[R06]: https://ontop-vkg.org/guide/concepts.html "Ontop virtual query"
[R07]: https://www.w3.org/TR/skos-primer/ "SKOS Primer"
[R08]: https://protege.stanford.edu/publications/ontology_development/ontology101-noy-mcguinness.html "Ontology Development 101"
[R09]: https://protege.stanford.edu/ "Protégé"
[R10]: https://gate.ac.uk/wiki/quick-start/GATE-Tutorial-2010-Gate4.0.pdf "GATE rule extraction"
[R11]: https://gate.ac.uk/sale/talks/gate-course-jun14/module-6-applications/termraider.pdf "TermRaider"
[R12]: https://www.w3.org/TR/r2rml/ "R2RML"
[R13]: https://monarch-initiative.github.io/ontogpt/custom/ "OntoGPT schemas and OWL templates"
[R14]: https://linkml.io/schema-automator/introduction.html "Schema Automator"
[R15]: https://www.w3.org/TR/rdf11-concepts/ "RDF 1.1 Concepts"
[R16]: https://www.w3.org/TR/trig/ "TriG"
[R17]: https://www.w3.org/TR/owl2-direct-semantics/ "OWL 2 Direct Semantics"
[R18]: https://www.w3.org/TR/shacl/ "SHACL"
[R19]: https://neo4j.com/docs/getting-started/appendix/graphdb-concepts/ "Neo4j property graph"
[R20]: https://aws.amazon.com/neptune/faqs/ "Amazon Neptune FAQ"
[R21]: https://docs.nebula-graph.io/3.8.0/1.introduction/1.what-is-nebula-graph/ "NebulaGraph 3.8 scope"
[R22]: https://docs.janusgraph.org/schema/ "JanusGraph schema"
[R23]: https://docs.arangodb.com/3.12/graphs/ "ArangoDB graph model"
[R24]: https://graphdb.ontotext.com/documentation/11.2/reasoning.html "GraphDB reasoning"
[R25]: https://docs.stardog.com/inference-engine/ "Stardog reasoning"
[R26]: https://jena.apache.org/documentation/tdb2/tdb2_fuseki.html "Fuseki with TDB2"
[R27]: https://neo4j.com/docs/cypher-manual/current/schema/graph-types/ "Neo4j Graph Types"
[R28]: https://neo4j.com/docs/cypher-manual/current/schema/graph-types/extend-graph-types/ "Graph type ADD"
[R29]: https://neo4j.com/docs/cypher-manual/current/schema/graph-types/set-graph-types/ "Graph type SET"
[R30]: https://neo4j.com/docs/cypher-manual/current/schema/graph-types/list-graph-types/ "Show Graph Types"
[R31]: https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html "Neptune Analytics"
[R32]: https://docs.aws.amazon.com/neptune-analytics/latest/userguide/using-rdf-data.html "RDF mapping for Neptune Analytics"
[R33]: https://docs.aws.amazon.com/neptune/latest/userguide/feature-opencypher-compliance.html "Neptune openCypher compliance"
[R34]: https://docs.aws.amazon.com/neptune/latest/userguide/feature-sparql-compliance.html "Neptune SPARQL default graph"
[R36]: https://docs.nebula-graph.io/3.8.0/1.introduction/2.data-model/ "NebulaGraph IDs and edge ranks"
[R37]: https://docs.nebula-graph.io/3.8.0/3.ngql-guide/11.edge-type-statements/1.create-edge/ "NebulaGraph edge schema"
[R38]: https://docs.nebula-graph.io/3.8.0/3.ngql-guide/1.nGQL-overview/1.overview/ "nGQL compatibility"
[R39]: https://docs.nebula-graph.io/3.8.0/3.ngql-guide/7.general-query-statements/3.go/ "nGQL GO"
[R40]: https://docs.nebula-graph.io/3.8.0/3.ngql-guide/12.vertex-statements/1.insert-vertex/ "NebulaGraph vertex insert"
[R41]: https://docs.nebula-graph.io/3.8.0/3.ngql-guide/13.edge-statements/1.insert-edge/ "NebulaGraph edge insert"
[R42]: https://docs.nebula-graph.io/3.8.0/1.introduction/3.nebula-graph-architecture/1.architecture-overview/ "NebulaGraph architecture"
[R43]: https://docs.janusgraph.org/storage-backend/ "JanusGraph storage backends"
[R44]: https://jena.apache.org/documentation/fuseki2/fuseki-config-endpoint "Fuseki endpoints"
[R45]: https://jena.apache.org/documentation/inference/ "Jena inference"
[R46]: https://www.w3.org/TR/owl2-profiles/ "OWL 2 Profiles"
[R47]: https://owl-rl.readthedocs.io/en/latest/ "OWL-RL Python implementation"
[R48]: https://www.w3.org/TR/sparql11-query/ "SPARQL 1.1 Query"
[R49]: https://github.com/RDFLib/pySHACL "pySHACL"
[R50]: https://obofoundry.org/principles/fp-004-versioning.html "OBO versioning"
[R51]: https://www.w3.org/TR/prov-o/ "PROV-O"
[R52]: https://www.w3.org/TR/swbp-vocab-pub/ "Publishing RDF vocabularies"
[R53]: https://microsoft.github.io/graphrag/query/overview/ "GraphRAG query"
[R54]: https://github.com/microsoft/graphrag/blob/c600ff80b77b9b62c47d539a3ac9569e95d8949b/packages/graphrag/graphrag/index/workflows/load_update_documents.py "GraphRAG update at verified commit"
[R55]: https://github.com/getzep/graphiti "Graphiti"
[R56]: https://www.palantir.com/docs/foundry/ontology/overview "Palantir Ontology"
[R57]: https://www.palantir.com/docs/foundry/action-types/overview/ "Palantir Action Types"
[R58]: https://genai.owasp.org/llmrisk/llm01-prompt-injection/ "Prompt Injection"
[R59]: https://www.w3.org/TR/2026/CR-rdf12-concepts-20260407/ "RDF 1.2 Concepts CR Snapshot"
[R60]: https://www.w3.org/TR/2026/CRD-rdf12-semantics-20260924/ "RDF 1.2 Semantics CR Draft"
[R61]: https://www.w3.org/TR/2026/WD-rdf12-turtle-20261007/ "RDF 1.2 Turtle WD"
[R62]: https://www.w3.org/TR/2026/WD-sparql12-query-20261011/ "SPARQL 1.2 Query WD"
[R63]: https://www.w3.org/TR/2026/WD-shacl12-core-20261014/ "SHACL 1.2 Core WD"
[R64]: https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query.html "Neptune Analytics query language"
