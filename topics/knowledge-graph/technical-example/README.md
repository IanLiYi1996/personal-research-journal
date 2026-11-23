# 两条 Ontology 实现路线的技术示例

日期：2026-10-23。所有对象和陈述为虚构演示数据。

统一主文：[Ontology 构建、管理与应用指南](../Ontology_构建管理与应用_完整指南.md)。

早期的 [双路线技术说明](../2026-10-23_Ontology技术实现_构建_图数据库管理_应用.md) 保留作代码说明参考。

## 1. 原生图数据库路线

不要求先有 RDF／OWL 文件，也不要求 APOC 或 n10s。

- [neo4j-model.cypher](neo4j-model.cypher)：唯一性约束、带版本的 EntityType／RelationType／PropertyDefinition 元模型和业务实例。
- [neo4j-query.cypher](neo4j-query.cypher)：条件、时间、状态、租户和证据查询；元模型类层次查询。
- [neo4j-validate.cypher](neo4j-validate.cypher)：必填、关系数量、模型版本、状态和时间违规检测。

这些脚本未在 Neo4j 执行。仅用于隔离 demo 库，不是生产同步器或完整通用元模型编译器。Neo4j 约束、Graph Types 的 edition／版本边界见主文。

示例查询参数：

```json
{
  "modelVersion": "0.1.0",
  "tenantId": "tenant-demo",
  "capabilityUri": "https://example.org/onto-tech/BatchInference",
  "environment": "linux-selfhosted",
  "at": "2026-10-23T10:00:00+08:00",
  "targetType": "https://example.org/onto-tech/Release"
}
```

`tenantId` 由可信后端绑定。本例不包含完整对象级权限、并发写入控制或实际事务 API。

## 2. RDF／OWL 路线

- [ontology.ttl](ontology.ttl)：类、属性、继承与逆关系。
- [data.trig](data.trig)：公共与受限 named graph；当前、过期、候选和不同环境陈述。
- [shapes.ttl](shapes.ttl)：结构与时间区间检查。
- [query.rq](query.rq)：在获准视图中进行参数化条件查询。
- [verify.py](verify.py)：13 项本地参考测试。
- [requirements.txt](requirements.txt)：本次测试的直接依赖版本。
- [fuseki-config.ttl](fuseki-config.ttl)：TDB2 持久化配置示例，只有 Turtle 语法经过本地解析。

在此目录下，可自行创建隔离环境后复测：

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python verify.py
```

上述命令需要网络安装依赖。本次实际运行用的是 `/tmp/ontology-verify-Ic2g0Z/venv`，没有在工作区创建 `.venv` 或修改系统 Python。

本次环境为 Python 3.14.5、RDFLib 7.6.0、pySHACL 0.40.1、owlrl 7.6.2。requirements 固定直接依赖，不是带所有平台哈希的生产锁文件。

## 3. 已执行的结果

verification-results.json 保存本次结果摘要。

13 项测试全部通过。获准视图包含 86 条三元组，OWL-RL 扩展后 330 条，SHACL 符合，当前条件查询返回 Claim21／Release21／Evidence21。

原始图没有显式 `Release21 type Release`。直接类型查询无当前结果；类路径查询或 OWL-RL 扩展后各返回一条。这验证的是特定公理和查询，不是 OWL DL／Full 的完备性。

测试的 graph allowlist 是 fixture，不是生产多租户权限实现。旧的贯穿案例文件仍保留其“未执行”状态，本目录是独立新示例。

## 4. 未执行的部分

未启动 Neo4j、Fuseki、TDB2、Ontop 或 HTTP API；未运行 Graph Types DDL；未测数据库性能、备份恢复、事务并发或完整权限系统。不能由 13 项本地测试推断这些部分已验收。
