# NebulaGraph 本体管理核实

核实日：2026-10-23。范围仅为 NebulaGraph；采用公开 **3.8.0 文档**及官方仓库 **v3.8.0** 发布标签，不声称这是最新或全部产品版本。仅核对一手资料，未部署或执行示例。[版本文档](https://docs.nebula-graph.io/3.8.0/)、[仓库发布](https://github.com/vesoft-inc/nebula/releases/tag/v3.8.0)。

## 模型、身份与语言

Space 隔离图数据，创建时确定 VID 类型、分区数和副本数。VID 为 INT64 或 FIXED_STRING(N)，在同一 Space 内唯一，不是每个 Tag 各有主键。顶点可带多个 Tag；Tag 定义顶点属性，Edge type 定义有向边及属性。边身份为“源 VID、边类型、rank、目标 VID”，rank 默认 0，允许同端点同类型有多条不同 rank 的边。[Space](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/9.space-statements/1.create-space/)、[数据模型](https://docs.nebula-graph.io/3.8.0/1.introduction/2.data-model/)。

字段支持整数、浮点、布尔、STRING/FIXED_STRING、日期时间及 GEOGRAPHY；可声明 NULL/NOT NULL 和 DEFAULT，不写空值约束时允许 NULL。LIST/SET/MAP 可用于查询，但不能作为存储属性。schema 不等同本体语义约束。[Tag 字段](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/10.tag-statements/1.create-tag/)、[Edge 字段](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/11.edge-type-statements/1.create-edge/)、[容器类型边界](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/3.data-types/6.list/)。

nGQL 仅兼容部分 openCypher 查询语法，不是完整兼容。3.8.0 的原生 nGQL 支持 CREATE SPACE/TAG/EDGE、INSERT 和 GO；这里的 CREATE 创建 schema，不能照搬图案式 CREATE/MERGE 来写实例。MATCH 属性访问须留意 `变量.Tag.属性`；普通属性也没有 UNIQUE 约束。[兼容边界](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/1.nGQL-overview/1.overview/)、[MATCH](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/7.general-query-statements/2.match/)。

## 本体管理设计

以下是应用建模建议，不是产品内置本体模块：用三个元模型 Tag，EntityType 保存名称、定义；RelationType 保存关系含义、domain/range 与基数规则；OntologyVersion 保存版本、发布状态，并连接该版本的类型定义。类型定义按版本保存，发布后保持不可变。

实例用 Product、Release、SupportClaim、Evidence 四类 Tag：Product 经 has_release 连接 Release；SupportClaim 实体化“某版本被支持”的陈述，连接 Release、Evidence 和 OntologyVersion。实例可经 instance_of 连接 EntityType；边类型与 RelationType 按名称、版本映射。涉及证据和版本的关系用陈述节点承载，避免假设边能直接连接元模型节点。[属性图建模基础](https://docs.nebula-graph.io/3.8.0/1.introduction/2.data-model/)。

schema 负责属性类型、空值及默认值，不自动解释文字定义、限制边端点 Tag、检查业务基数或进行语义推理。domain/range、证据完整性、版本一致性及推理规则应由导入校验、查询审计或应用实现。该版本允许悬空边；同 VID＋Tag 或同边身份写入可覆盖原值，因此版本不可变性也要应用保障。[顶点写入](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/12.vertex-statements/1.insert-vertex/)、[边写入](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/13.edge-statements/1.insert-edge/)。

发布流程建议先检查类型引用、端点与基数，再写入版本定义和实例，最后更新应用的当前版本指针。迁移记录保留源版本、目标版本、映射规则与证据出处；数据库 schema 变更不代表语义版本已经发布。这些是应用治理约定，不能仅靠 Tag 名称或 rank 实现。

## 最小示意：未执行

仅演示 Product—Release 片段，非完整本体实现。语法对照上述文档及 [GO](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/7.general-query-statements/3.go/) 核对；单分区单副本仅为示意，不是生产配置。新建 Space/schema 后须等待生效，勿把整段当作无需等待的脚本。

```ngql
CREATE SPACE ontology_demo(
  partition_num=1, replica_factor=1, vid_type=FIXED_STRING(64)
);
-- 等待 Space 生效后再继续
USE ontology_demo;
CREATE TAG Product(name string NOT NULL);
CREATE TAG Release(version string NOT NULL);
CREATE EDGE has_release();
-- 等待 Tag/Edge schema 生效后再写入
INSERT VERTEX Product(name) VALUES "p1":("示例产品");
INSERT VERTEX Release(version) VALUES "r1":("1.0");
INSERT EDGE has_release() VALUES "p1"->"r1"@0:();
GO FROM "p1" OVER has_release YIELD dst(edge) AS release_vid;
```

## 分布式职责与运维范围

graphd 处理查询入口、解析、校验、执行计划与查询执行；metad 维护 Space、schema、分区位置、用户和权限等元数据，采用 Raft；storaged 管理分区数据、读写及副本，采用 Raft 与 RocksDB。[Graph](https://docs.nebula-graph.io/3.8.0/1.introduction/3.nebula-graph-architecture/3.graph-service/)、[Meta](https://docs.nebula-graph.io/3.8.0/1.introduction/3.nebula-graph-architecture/2.meta-service/)、[Storage](https://docs.nebula-graph.io/3.8.0/1.introduction/3.nebula-graph-architecture/4.storage-service/)。

自运维应覆盖副本与故障域、磁盘和服务监控、备份恢复演练、账户权限、升级与 schema 变更生效检查。认证默认关闭，须显式启用；该版本 BR 仅支持全量备份，备份期间阻断相关 DDL/DML，恢复还有限制，副本不能代替备份。不作性能排名或预算推算。[监控](https://docs.nebula-graph.io/3.8.0/6.monitor-and-metrics/1.query-performance-metrics/)、[备份恢复](https://docs.nebula-graph.io/3.8.0/backup-and-restore/nebula-br/1.what-is-br/)、[认证](https://docs.nebula-graph.io/3.8.0/7.data-security/1.authentication/1.authentication/)、[升级](https://docs.nebula-graph.io/3.8.0/4.deployment-and-installation/3.upgrade-nebula-graph/upgrade-nebula-comm/)。

本次按“属性图＋nGQL”路线核实。3.8.0 文档明确该版本未支持 RDF、SPARQL；这不等于“永远不支持”。本次未取得可引用的官方 OWL 推理能力说明，不将其列为已核实能力。[官方 RDF/SPARQL 说明](https://docs.nebula-graph.io/3.8.0/3.ngql-guide/1.nGQL-overview/1.overview/#does-nebulagraph-support-w3c-rdf-sparql-or-graphql)。
