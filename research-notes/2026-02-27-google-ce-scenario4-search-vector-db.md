# Google Cloud Practice CE Presentation - Scenario 4: Retail Search & Vector Database

- **Date:** 2026-02-27
- **Tags:** Google Cloud, CE Interview, Search, Vector Database, E-Commerce, Vertex AI

## Context

Google Cloud Practice Customer Engineer (AI) 面试准备。选择 Scenario 4: Retail / E-Commerce - Search & Vector Database。本文档包含原题完整内容、要点拆解、架构设计及演讲准备。

---

# Part 0: Google Cloud 最新组件知识 (2026-02 更新)

> 以下信息从 Google Cloud 官方文档 (cloud.google.com) 实时获取并交叉验证，确保演讲中使用最新产品名称和能力描述。
>
> **文档来源**:
> - https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings
> - https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings
> - https://cloud.google.com/vertex-ai/docs/vector-search/overview
> - https://cloud.google.com/vertex-ai/docs/vector-search/configuring-indexes
> - https://cloud.google.com/retail/docs/overview
> - https://cloud.google.com/alloydb/docs/ai/work-with-embeddings

## 0.1 Embedding 模型 — 最新选型

### gemini-embedding-001 (最新旗舰模型)

| 属性 | 详情 |
|---|---|
| **维度** | 最高 **3,072 维** (可通过 `output_dimensionality` 降维至 256/512/768 等) |
| **序列长度** | 最大 2,048 tokens |
| **语言支持** | 英文 + 多语言 + 代码 (统一模型，不再需要区分) |
| **定位** | 统一了之前的 `text-embedding-005` 和 `text-multilingual-embedding-002`，**所有领域表现更优** |
| **Task Types** | RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, 等 |
| **距离度量** | 支持 cosine similarity, dot product, Euclidean |

**关键更新**: `gemini-embedding-001` 是当前最新最强的文本 embedding 模型，取代了之前的 `text-embedding-005`。**演讲中应优先推荐此模型**。

### text-embedding-005 (前代模型，仍可用)

| 属性 | 详情 |
|---|---|
| 维度 | 最高 768 维 |
| 语言 | 仅英文 + 代码 |
| 状态 | 仍在支持，但 gemini-embedding-001 性能更优 |

### text-multilingual-embedding-002

| 属性 | 详情 |
|---|---|
| 维度 | 最高 768 维 |
| 语言 | 多语言 |
| 状态 | 已被 gemini-embedding-001 统一取代 |

### 开源模型 (MaaS 方式提供)

- **multilingual-e5-small**: 384 维, 512 tokens, 12 层
- **multilingual-e5-large**: 1,024 维, 512 tokens, 24 层

### API 限制

- 单次请求最多 250 条文本
- 单次请求最多 20,000 tokens
- 单条文本最多 2,048 tokens (超出默认静默截断，可设 `autoTruncate=false` 报错)

---

## 0.2 Multimodal Embedding — multimodalembedding@001

| 属性 | 详情 |
|---|---|
| **模型名** | `multimodalembedding@001` |
| **默认维度** | **1,408 维** |
| **可选维度** | 128, 256, 512 (用于优化延迟/存储) |
| **文本输入** | 最大 **32 tokens** (~32 词), **仅英文** |
| **图片输入** | BMP/GIF/JPG/PNG, 最大 20MB, 内部调整为 512x512 |
| **视频输入** | AVI/FLV/MKV/MOV/MP4 等, 最长分析 **2 分钟** |
| **QPS 限制** | 120-600 requests/min (取决于 region) |

**重要限制**:
- 文本输入仅 32 tokens — 对于长商品描述不够用！
- **建议策略**: 文本搜索用 `gemini-embedding-001` (2048 tokens)，图片搜索用 `multimodalembedding@001`，**两套索引但可以做 cross-modal bridge**
- 或者: 对商品同时生成两种 embedding，查询时根据输入类型选择对应索引

**演讲中的架构调整**: 鉴于 multimodal embedding 的文本限制 (32 tokens)，更好的方案是:
1. **文本搜索**: 用 `gemini-embedding-001` (3072 维，2048 tokens)
2. **图片搜索**: 用 `multimodalembedding@001` (1408 维)
3. **跨模态能力**: multimodal embedding 仍可将 text 和 image 映射到同一空间，但仅用于短查询 (Shop the Look 场景)

---

## 0.3 Vertex AI Vector Search (Matching Engine)

### 核心能力

| 特性 | 详情 |
|---|---|
| **算法** | 基于 Google Research 的 **ScaNN** |
| **Embedding 类型** | Dense embeddings (语义) + **Sparse embeddings (关键词)** + **混合搜索** |
| **索引类型** | TreeAH (approximate), Brute-force (exact) |
| **过滤** | Token restricts (分类) + Numeric restricts (数值范围) |
| **实时更新** | **Streaming ingestion** — 实时添加/删除/更新向量 |
| **部署方式** | Public endpoint / Private Service Connect (PSC) / VPC Peering |
| **Auto-scaling** | 支持按 QPS 自动扩缩容 |

### Shard 大小

| Size | 容量 |
|---|---|
| SMALL | 2 GiB |
| MEDIUM | 20 GiB (默认) |
| LARGE | 50 GiB |

Sparse embeddings 容量为 shard 字节大小的 20%。

### 过滤机制 (Filtering / Restricts)

**Token Restricts (分类过滤)**:
- 按命名空间 (namespace) 组织
- 跨命名空间: AND 逻辑
- 同命名空间内: OR 逻辑
- 示例: `{color: [red, blue], shape: [square]}` → `(red OR blue) AND square`
- 支持 **Denylist** (排除匹配)

**Numeric Restricts (数值过滤)**:
- 支持运算符: `LESS`, `LESS_EQUAL`, `EQUAL`, `GREATER_EQUAL`, `GREATER`
- 支持 int, float, double 类型
- 适用于价格、库存量等数值过滤

**关键发现 — 原生混合搜索**:
Vector Search 现在 **同时支持 dense + sparse embeddings**，意味着可以在同一索引中实现语义搜索 + 关键词搜索的混合，不需要单独的 BM25 引擎！

### 索引配置 (官方文档验证)

- **算法选择**: TreeAH (推荐, approximate) — `approximateNeighborsCount` 通常设 150
- **TreeAH 参数**: `leafNodeEmbeddingCount` (默认 1000), `fractionLeafNodesToSearch` (默认 0.05, 范围 0.0-1.0)
- **距离度量**: DOT_PRODUCT_DISTANCE (默认, **推荐**), COSINE_DISTANCE, SQUARED_L2_DISTANCE, L1_DISTANCE
- **官方建议**: 使用 `DOT_PRODUCT_DISTANCE` + `UNIT_L2_NORM` 替代 COSINE_DISTANCE，因为 DOT_PRODUCT 经过了更多优化
- **特征归一化**: `featureNormType` 支持 `UNIT_L2_NORM` 和 `NONE`
- **索引更新方式**: `BATCH_UPDATE` (批量) 或 `STREAM_UPDATE` (流式实时)
- **索引构建时间**: 最长约 2 小时 (Terraform timeout 设置)
- **机器类型**:

| 机器类型 | SMALL | MEDIUM | LARGE |
|---|---|---|---|
| e2-standard-2 | **默认** | - | - |
| e2-standard-16 | 支持 | **默认** | - |
| e2-highmem-16 | - | - | **默认** |
| n1-standard-16 | 支持 | 支持 | 支持 |
| n1-standard-32 | 支持 | 支持 | 支持 |
| n2d-standard-32 | 支持 | 支持 | 支持 |

### 真实客户案例 (官方文档提及)

- **eBay**: 使用 Vector Search 做推荐系统，在大规模商品目录中发现相似产品
- **Mercado Libre**: 使用 Vector Search 改善市场平台推荐
- **Bloomreach**: 利用 Vector Search 实现性能、可扩展性和成本效益

> "Customers like Bloomreach, eBay, and Mercado Libre use Vertex AI for its performance, scalability, and cost-effectiveness, achieving benefits like faster search and increased conversions."

### 相关产品集成

- **Vertex AI Ranking API**: 基于预训练语言模型对搜索结果进行重排序 (re-ranking)，适合与 Vector Search 配合使用
- **Vertex AI Feature Store**: 基于 BigQuery 的特征服务，可用于存储和即时检索用户画像特征
- **Vertex AI Pipelines**: 自动化 embedding 生成、索引更新的 MLOps 管道

---

## 0.4 Vertex AI Search for Commerce (零售专用)

**这是本场景最重要的"开箱即用"选项！**

| 特性 | 详情 |
|---|---|
| **产品名** | Vertex AI Search for Commerce (前身: Retail Search / Discovery Engine) |
| **核心能力** | AI 驱动的商品搜索 + 推荐 + 个性化 |
| **搜索功能** | 动态搜索、查询扩展、Autocomplete、Faceted Navigation |
| **个性化** | 基于用户行为和意图的 AI 排序、CRM 个性化 |
| **业务规则** | Serving controls (boost/bury/pin/filter) |
| **推荐** | 个性化商品推荐、邮件推荐、浏览类目推荐 |
| **数据要求** | Product catalog + User events (实时 + 历史导入) |
| **集成时间** | 官方说 "平均集成时间为数周级别" |

### 关键引用

> "Vertex AI Search for commerce performance (relevancy, ranking, and revenue optimization) is extremely sensitive to the uploaded data, including catalogs, product info, and user events."

> "Never cache personalized results from an end user, and never return personalized results to a different end user."

### 四阶段迁移方法 (官方文档原文验证)

官方描述: "By diligently following this four-phased approach, a typical migration to A/B testing can be achieved in about **two to three months**, depending on the current search system complexity and speed of execution."

1. **Keep merchant teams informed** — 主动沟通变化原因，解释 AI-first 方法
2. **Educate teams on the new paradigm** — 系统基于用户行为和意图检测，产品排序更个性化，搜索结果看起来会不同
3. **Set clear guidelines for business rules** — 业务规则仅用于有数据支撑的特定原因 (如合同义务或明确的收入驱动策略)，**让 AI 做它的工作**
4. **A/B test new rules** — 每条新规则都应通过 A/B 测试验证，一组有规则，一组无规则，**让数据决定**

### 数据质量是关键

> "Vertex AI Search for commerce has multiple dashboards and data quality checks in place to ensure that any issues or potential flaws in the data or data schema are flagged."

> "If data flaws are overlooked from the start, the model won't train accurately and an initial A/B test does not produce the expected results, the root cause being more often than not the catalog or user data rather than Vertex AI Search for commerce itself."

**最佳实践维度**:
- Product catalog best practices (商品目录)
- User events best practices (用户事件)
- Integration and configuration best practices (集成配置)
- A/B experiments best practices (实验)

---

## 0.5 AlloyDB AI — 向量搜索 + SQL (官方文档验证)

| 特性 | 详情 |
|---|---|
| **向量索引** | 支持 **ScaNN, HNSW, IVF, IVFFLAT** 四种索引算法 |
| **Embedding 集成** | 原生调用 Vertex AI 模型 (`google_ml.embedding()`) |
| **支持模型** | gemini-embedding-001, text-embedding-005, OpenAI text-embedding-ada-002/3-small/3-large |
| **扩展名** | `google_ml_integration` 扩展 |
| **两种 Schema** | `public.embedding()` (无需注册) / `google_ml.embedding()` (支持 task type 等高级参数) |

### 官方代码示例 (已验证)

```sql
-- 基础 embedding 生成
SELECT google_ml.embedding(
  model_id => 'gemini-embedding-001',
  content => 'AlloyDB is a managed, cloud-hosted SQL database service'
);

-- 跨项目调用需先注册模型端点
CALL google_ml.create_model(
  model_id => 'gemini-embedding-001',
  model_request_url => 'https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT/locations/REGION/publishers/google/models/gemini-embedding-001:predict',
  model_provider => 'google',
  model_type => 'text_embedding',
  model_auth_type => 'alloydb_service_agent_iam',
  model_in_transform_fn => 'google_ml.vertexai_text_embedding_input_transform',
  model_out_transform_fn => 'google_ml.vertexai_text_embedding_output_transform'
);
```

### 语义搜索场景 (官方文档原文)

> "SQL queries using LLM-powered embeddings can help return semantically similar responses. By applying embeddings, you can query the table for items whose complaints have semantic similarity to a given text prompt."
>
> 例如: `SELECT * FROM item WHERE complaints LIKE '%wrong color%'` 无法匹配 "The picture shows a blue one, but the one I received was red"，但语义搜索可以。

**关键优势**: 如果客户团队习惯 SQL，AlloyDB 提供最低学习曲线的向量搜索方案 — 在 SQL 查询中直接做 `WHERE` 过滤 + 向量近邻搜索。官方提供了 "build a smart shopping assistant with AlloyDB" 教程。

---

## 0.6 演讲中的产品选型更新

基于最新信息，**更新选型建议**:

### 方案 A: 全托管方案 (推荐给本场景)

```
Vertex AI Search for Commerce
  ├── 开箱即用的电商语义搜索
  ├── 内置个性化 + 推荐
  ├── 业务规则 (boost/bury/pin)
  ├── Autocomplete + 查询扩展
  └── 短板: 图片搜索 (Shop the Look) 需要额外构建
```

**补充 Shop the Look**:
```
Image Upload → Gemini Vision (object detection)
            → multimodalembedding@001 (图片 embedding)
            → Vertex AI Vector Search (相似商品检索)
```

### 方案 B: 自建方案 (更灵活，更适合演示技术深度)

```
Text Search:  gemini-embedding-001 (3072 维) → Vector Search (dense + sparse 混合)
Image Search: multimodalembedding@001 (1408 维) → Vector Search (单独 index)
Filters:      Vector Search Token/Numeric Restricts 或 AlloyDB SQL
Biz Rules:    Cloud Run 自定义 re-ranking 层
```

### 演讲策略建议

**呈现两个方案，对比 trade-offs — 这恰好是 "Architectural Decisions & Trade-offs" 环节的最佳素材:**

| 维度 | 方案 A (Commerce Search) | 方案 B (自建) |
|---|---|---|
| 上线速度 | 快 (数周) | 慢 (数月) |
| 自定义程度 | 中 | 高 |
| 图片搜索 | 需额外建设 | 完全自控 |
| 维护成本 | 低 (全托管) | 高 |
| 技术深度展示 | 低 | **高 (面试加分)** |
| 推荐给客户 | Phase 1 快速见效 | Phase 2 差异化能力 |

**最佳策略: 先用方案 A 快速上线 → 再用方案 B 构建 Shop the Look 等差异化能力。**

---

## 0.7 补充组件 — 演讲中可引用的高级能力

### Vertex AI Ranking API (Re-ranker)

> "The ranking API reranks documents based on relevance to a query using a pre-trained language model, providing precise scores. It's ideal for improving search results from various sources including Vector Search."

用途: 在 Vector Search 返回 top-100 候选后，用 Ranking API 做精排 (re-ranking)，显著提升排序质量。

### Two-Tower Retrieval Model (个性化检索)

官方提供了完整的参考架构:
> "The two-tower modeling framework is a powerful retrieval technique for personalization use cases because it learns the semantic similarity between two different entities, such as web queries and candidate items."

- 一侧 tower 编码 user (查询 + 用户画像)
- 一侧 tower 编码 item (商品属性 + 描述)
- 训练后两侧输出在同一向量空间 → 用 Vector Search 做快速检索
- 官方 notebook: "Implement two-tower retrieval for large-scale candidate generation"

### Hybrid Search (混合搜索)

官方提供了专门的 tutorial notebook:
> "Combining Semantic & Keyword Search: A Hybrid Search Tutorial with Vertex AI Vector Search"

- 支持 dense + sparse embeddings 在同一索引
- 可以配合 Vertex AI Ranking API 做 ensemble re-ranking

### Multimodal Search (多模态搜索)

官方描述的方法:
> "Building a multimodal search engine with Vertex AI that combines text and image search using a weighted **Rank-Biased Reciprocal Rank** ensemble method."

这与我们架构中的 RRF (Reciprocal Rank Fusion) 方案一致！可以在演讲中引用 Google 官方推荐的 ensemble 方法。

---

# Part 1: 原题完整内容

## Problem Statement

A fashion retailer with 2 million SKUs and $2 billion in annual online revenue is experiencing significant conversion issues attributed to poor search quality. Analytics show that 35% of cart abandonments are correlated with search-related friction.

### Current Search Infrastructure

- Elasticsearch cluster with keyword-based matching
- Average search latency: 800ms (P50), 2.5 seconds (P99)
- Bounce rate when latency > 200ms: 40% of users leave immediately
- Search result click-through rate: 12% (industry benchmark: 25%)
- Zero-result rate: 15% of queries return no results

### Example Problem Queries

| User Query | Current Result | Expected Result |
|---|---|---|
| "floral dress for winter wedding" | Summer sundresses (matched "floral dress") | Long-sleeve or velvet floral dresses |
| "something blue for a baby shower" | Random blue items | Blue dresses, accessories, gift items |
| "shoes like the ones Taylor Swift wore at Eras Tour" | No results | Silver boots, sparkly heels |
| "comfortable heels for standing all day" | All heels sorted by popularity | Block heels, kitten heels, platform heels |

### Product Catalog Characteristics

- 2 million active SKUs across clothing, shoes, accessories, home goods
- Average product has 15 attributes (color, size, material, occasion, style, brand, etc.)
- 50,000 new products added monthly; 30,000 discontinued
- Product descriptions vary wildly in quality (some are 5 words, some are 500 words)
- 40% of products have multiple images; 10% have only a stock photo

### "Shop the Look" Feature Request

Users want to upload a photo (from Instagram, Pinterest, or a magazine) and find similar items in the catalog. This requires:
- Image understanding (identify clothing items in the photo)
- Style matching (find products with similar aesthetic, not just similar color)
- Multi-item results (if photo shows complete outfit, suggest matching items)

### Traffic Patterns

- Normal daily traffic: 1 million searches/day (~12 QPS average, 50 QPS peak)
- Black Friday: 10 million searches in 8 hours (350 QPS sustained, 1000 QPS spike)
- Mobile: 70% of searches (latency even more critical on mobile networks)

### Business Constraints

- Search must incorporate business rules: promoted products, inventory levels, margin optimization
- Personalization: Repeat customers should see results influenced by purchase history
- Filters must work with semantic search (search "summer dress" + filter "under $50")
- Cannot afford downtime during Black Friday — this is 20% of annual revenue

### The Customer Wants

1. Natural language search that understands intent, not just keywords
2. Image-based search for "Shop the Look" functionality
3. Sub-200ms latency at P99, including during Black Friday
4. Seamless hybrid search: semantic understanding + structured filters + business rules
5. Offline evaluation framework to test ranking changes before production

### Questions to Address

- How do you handle the 10x traffic spike on Black Friday? What's your scaling strategy?
- How do you combine semantic search with structured filtering (color, size, price)?
- For image search, do you use the same vector space as text search, or separate?
- How do you measure search quality? What metrics would you track?
- How do you handle cold-start for new products with no engagement data?

### Deliverable

> Design a search architecture that addresses these requirements. Walk me through the text search pipeline, image search pipeline, and how you would handle the Black Friday scaling challenge.

---

# Part 2: 要点拆解与分析

## 2.1 核心挑战解构

本场景的本质问题是：**从关键词匹配到语义理解的范式跃迁**，同时在极端流量下保持亚 200ms 延迟。

### 挑战 1: 语义鸿沟 (Semantic Gap)

传统 Elasticsearch 的 BM25 是 **词频-逆文档频率** 匹配，无法理解：
- **意图** — "for winter wedding" 隐含了"长袖/保暖/正式"
- **文化语境** — "Taylor Swift Eras Tour" 需要流行文化知识
- **功能需求** — "comfortable for standing all day" 是对鞋类结构的要求

解法：用 **embedding model** 将查询和商品映射到同一语义空间，使得语义相近的内容在向量空间中距离更近。

### 挑战 2: 多模态搜索 (Multimodal Search)

"Shop the Look" 要求图片和文本共享理解能力：
- 一张穿搭照片中可能包含 3-5 件单品
- 需要区分"风格相似"和"颜色相同"
- 用户可能用文字补充说明 ("like this but in black")

解法：使用 **多模态 Embedding 模型** 将图片和文本映射到统一向量空间。

### 挑战 3: 延迟与规模 (Latency at Scale)

- 正常: 50 QPS → Black Friday: 1000 QPS (20x spike)
- P99 < 200ms 包含完整链路：网络 + 向量检索 + 过滤 + 排序 + 业务规则
- 200 万向量的 ANN 搜索 + 结构化过滤 需要在 <50ms 内完成

解法：ANN 索引 (ScaNN) + 预过滤 + 缓存 + 自动扩缩容

### 挑战 4: 混合搜索 (Hybrid Search)

用户同时需要：
- 语义理解 ("summer dress") — 向量搜索
- 精确过滤 ("under $50", "size M") — 结构化查询
- 业务规则 (促销商品置顶, 低库存降权) — 后处理排序

这三者必须在单次查询中无缝融合。

### 挑战 5: 搜索质量的可度量性

上线前必须有离线评估框架，上线后需要持续监控：
- 不能只靠 A/B 测试 — 需要 offline evaluation
- 标注数据成本高 — 需要利用隐式反馈 (点击、购买)

---

## 2.2 为什么现有方案不够 — "Unblocker" 论述

这是演讲中 "Specialized Challenge" 环节的核心逻辑：

```
Off-the-shelf Elasticsearch
  ├── ❌ BM25 无语义理解 → 零结果率 15%
  ├── ❌ 无多模态能力 → 图片搜索不可能
  ├── ❌ kNN 插件延迟高 → P99 无法 <200ms @ 1000 QPS
  └── ❌ 无统一 embedding 管线 → 维护成本高

需要的 "Unblocker":
  ├── ✅ 语义 Embedding (text + image → 统一向量空间)
  ├── ✅ 专用向量搜索引擎 (ScaNN / HNSW, 非 ES kNN)
  ├── ✅ 混合检索融合层 (semantic + keyword + filters + biz rules)
  └── ✅ 自动扩缩容基础设施 (应对 Black Friday)
```

---

# Part 3: 架构设计

## 3.1 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Layer                                │
│   Web / Mobile App  ──→  Cloud CDN  ──→  Cloud Load Balancer       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                      API Gateway (Cloud Run)                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ Text     │  │ Image    │  │ Hybrid    │  │ Business Rules   │  │
│  │ Search   │  │ Search   │  │ Merger &  │  │ Engine           │  │
│  │ Handler  │  │ Handler  │  │ Re-Ranker │  │ (boost/bury/pin) │  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └────────┬─────────┘  │
└───────┼──────────────┼──────────────┼─────────────────┼────────────┘
        │              │              │                  │
┌───────▼──────────────▼──────────────▼──────────────────▼────────────┐
│                     Search & Retrieval Layer                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         Vertex AI Vector Search (Matching Engine)            │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  │   │
│  │  │ Text Embedding│  │ Image Embedding│  │ Combined Index │  │   │
│  │  │ Index         │  │ Index          │  │ (Multimodal)   │  │   │
│  │  └───────────────┘  └───────────────┘  └────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         Structured Search (AlloyDB / Cloud SQL)              │   │
│  │  price, color, size, brand, category, inventory, promotions  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         Cache Layer (Memorystore - Redis)                    │   │
│  │  Hot queries cache │ Embedding cache │ Result cache          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Embedding & Indexing Pipeline                     │
│                                                                     │
│  Cloud Storage  ──→  Dataflow / Cloud Functions  ──→  Embedding    │
│  (product feed)      (ETL, validation)               Generation    │
│                                                      (Vertex AI)   │
│                                                          │         │
│                                          ┌───────────────▼──────┐  │
│                                          │ Vector Search Index  │  │
│                                          │ (streaming update)   │  │
│                                          └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   Monitoring & Evaluation Layer                      │
│  Cloud Monitoring │ BigQuery (search logs) │ Vertex AI Experiments  │
│  Looker Dashboard │ Offline Eval Pipeline  │ A/B Testing Framework  │
└─────────────────────────────────────────────────────────────────────┘
```

## 3.2 Text Search Pipeline (文本搜索管线)

### Step 1: Query Understanding

```
User Query: "floral dress for winter wedding"
                    │
                    ▼
        ┌───────────────────┐
        │ Query Preprocessor│
        │ - spell correction│
        │ - query expansion │
        │ - intent detection│
        └────────┬──────────┘
                 │
     ┌───────────▼───────────┐
     │ Vertex AI Embeddings  │
     │ text-embedding-005    │
     │ (768-dim vector)      │
     └───────────┬───────────┘
                 │
   ┌─────────────▼─────────────┐
   │   Parallel Retrieval       │
   │                            │
   │  ┌──────────┐ ┌─────────┐ │
   │  │ Vector   │ │Keyword  │ │
   │  │ Search   │ │Search   │ │
   │  │ (ANN,    │ │(BM25 on │ │
   │  │ top-100) │ │AlloyDB) │ │
   │  └────┬─────┘ └────┬────┘ │
   └───────┼─────────────┼─────┘
           │             │
     ┌─────▼─────────────▼─────┐
     │   Reciprocal Rank Fusion │
     │   (RRF) Score Merging    │
     │   score = Σ 1/(k + rank) │
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │   Structured Filtering   │
     │   (price, size, color,   │
     │    availability)         │
     └────────────┬────────────┘
                  │
     ┌────────────▼────────────┐
     │   Business Rules Layer   │
     │   - Promoted items boost │
     │   - Low inventory bury   │
     │   - Margin optimization  │
     │   - Personalization      │
     └────────────┬────────────┘
                  │
                  ▼
           Final Results
           (top 20-50)
```

### 关键设计决策

**Embedding 模型选择:**
- 使用 Vertex AI **`gemini-embedding-001`** (最新旗舰，最高 3,072 维，可降维)
- 推荐使用 **768 维** (通过 `output_dimensionality=768` 降维) — 精度与延迟最佳平衡
- 商品端: 将 title + description + attributes 拼接后生成 embedding (最长 2,048 tokens)
- 查询端: 直接对用户查询生成 embedding
- 为什么不用 3,072 全维度: 200 万向量 × 3072 维 = 索引太大，延迟增加；768 维已足够

**混合检索 — 为什么需要 keyword + semantic:**
- 纯语义搜索对精确匹配弱 (搜 "Nike Air Max 90" 应精确匹配)
- 纯关键词搜索无法理解意图 (如 "winter wedding" 场景)
- **最新能力**: Vertex AI Vector Search 现在 **原生支持 dense + sparse embeddings 混合搜索**，可在同一索引中完成
- 也可用 **Reciprocal Rank Fusion (RRF)** 融合两路独立结果，取各自优势

**Pre-filter vs Post-filter:**
- 选择 **Pre-filter** 策略: 先根据结构化条件缩小候选集，再做向量搜索
- 原因: 200 万 SKU 中，"size M + under $50" 可能只有 10 万条，ANN 搜索量大幅减少
- Vertex AI Vector Search 原生支持 restricts/crowding 机制实现 pre-filter

## 3.3 Image Search Pipeline (图片搜索管线)

```
User uploads photo (e.g., Instagram outfit)
                    │
                    ▼
        ┌───────────────────────┐
        │  Image Preprocessing   │
        │  - resize to 512x512  │
        │  - normalize           │
        └────────┬──────────────┘
                 │
     ┌───────────▼───────────────┐
     │  Object Detection (opt.)  │
     │  Identify individual items│
     │  - dress, shoes, bag      │
     │  (Vertex AI Vision / Gemini)│
     └───────────┬───────────────┘
                 │
     ┌───────────▼───────────────┐
     │ Multimodal Embedding      │
     │ multimodalembedding@001   │
     │ (1408-dim vector)         │
     │ Same space as text!       │
     └───────────┬───────────────┘
                 │
     ┌───────────▼───────────────┐
     │ Vector Search             │
     │ (same index as text)      │
     │ - ANN search, top-100     │
     └───────────┬───────────────┘
                 │
     ┌───────────▼───────────────┐
     │ Category Grouping         │
     │ "Complete the Look"       │
     │ - top shoes, top dress,   │
     │   top accessories         │
     └───────────┬───────────────┘
                 │
                 ▼
          Grouped Results
```

### 关键设计决策

**统一向量空间 vs 分离向量空间:**

| 方案 | 优点 | 缺点 |
|---|---|---|
| **统一空间** (推荐) | 图片查文字、文字查图片天然互通；索引只维护一份 | 需要多模态模型；维度可能更高 |
| 分离空间 | 各自模型可独立优化 | 需要跨空间检索的 bridge；维护两套索引 |

**推荐: 统一多模态向量空间**
- Google 的 `multimodalembedding@001` 已将 text 和 image 映射到同一 1408 维空间
- 商品入库时: 用 text + image 一起生成 embedding (multimodal)
- 文本查询: 用 text-only 输入生成同空间 embedding
- 图片查询: 用 image 输入生成同空间 embedding
- 好处: 用户可以 "图片 + 文字" 混合查询 ("like this but in black")

**Object Detection 的作用:**
- 一张穿搭照中可能有 5 件单品
- 用 Gemini Vision 先识别各个单品的 bounding box
- 对每个单品区域单独生成 embedding → 分别检索
- 将结果按品类分组呈现："Similar dresses | Similar shoes | Similar bags"

## 3.4 Black Friday 扩缩容策略

```
                  Normal Day              Black Friday
                  ──────────              ────────────
QPS               12 avg / 50 peak       350 avg / 1000 peak
Vector Search     2 nodes                 20 nodes (auto-scale)
Cloud Run         2 instances             50 instances (auto-scale)
Redis Cache       1 node, 13GB            3 nodes, cluster mode
Hit Rate Target   60% cache hit           85% cache hit (pre-warm)
```

### 分层策略

**Layer 1 — CDN & Edge Caching**
- 静态搜索建议 (autocomplete) 缓存在 Cloud CDN
- 热门类目页面预渲染并缓存

**Layer 2 — Application Cache (Redis)**
- Black Friday 前 2 天: 预热 top 10,000 查询的结果到 Redis
- Cache key: `hash(query_embedding + filters + biz_rules_version)`
- TTL: 5 分钟 (平衡新鲜度和命中率)
- 目标: Black Friday 85% cache hit rate → 实际打到 Vector Search 的 QPS 降为 150

**Layer 3 — Vector Search Auto-scaling**
- Vertex AI Vector Search 支持按 QPS 自动扩缩
- 预配置: min 5 nodes, max 25 nodes
- Black Friday 前 1 小时: 手动 warm-up 到 15 nodes
- 监控: QPS / latency / error rate → Cloud Monitoring alerts

**Layer 4 — Graceful Degradation**
- 如果 Vector Search 延迟超标: fallback 到纯 keyword search + cache
- Circuit breaker 模式: 10 秒内 >5% 错误率 → 自动切换 fallback
- 用户感知: 搜索质量略降，但服务不中断

### 延迟预算分配 (P99 < 200ms)

```
Total budget: 200ms P99
├── Network (CDN → LB → Cloud Run):  20ms
├── Query preprocessing:              10ms
├── Cache lookup:                     5ms (hit) / skip (miss)
├── Embedding generation:             30ms
├── Vector Search (ANN):              40ms
├── Structured filter (AlloyDB):      30ms (parallel with vector)
├── RRF merge + re-rank:              15ms
├── Business rules application:       10ms
├── Serialization + response:         10ms
└── Buffer:                           30ms
```

## 3.5 Personalization (个性化)

```
Repeat Customer Query: "dress"
         │
         ├── Base: semantic search for "dress"
         │
         └── Personalization signals:
             ├── Purchase history embedding (avg of past buys)
             ├── Browse history (recent 30 days)
             ├── Price affinity (avg order value)
             └── Size/brand preferences
                      │
                      ▼
              Re-ranking boost:
              final_score = 0.7 × relevance_score
                          + 0.2 × personalization_score
                          + 0.1 × business_rule_score
```

- 用户画像存储在 **Bigtable** (低延迟 key-value)
- 用户历史购买 embedding = 历史购买商品 embedding 的加权平均
- Query time: 拉取用户画像 → 调整 re-ranking 权重 (不影响召回阶段)

## 3.6 Cold Start 处理 (新品冷启动)

新品没有点击/购买数据，但有:
1. **商品描述 + 图片** → 直接生成 embedding → 进入向量索引
2. **属性元数据** (color, material, occasion) → 结构化过滤可用
3. **类似商品映射** → 找到同品牌/同品类的历史热销品，作为 proxy signal
4. **Exploration boost** → 新品 7 天内给予小幅曝光加权，收集初始信号

关键点: **基于 content 的 embedding 不依赖行为数据**，因此新品在入库后立即可被语义搜索发现。这是向量搜索相对于协同过滤的天然优势。

## 3.7 搜索质量评估框架

### Offline Evaluation (上线前)

| 指标 | 定义 | 目标 |
|---|---|---|
| **NDCG@10** | 前 10 结果的排序质量 | > 0.65 |
| **MRR** | 第一个相关结果的排名倒数 | > 0.45 |
| **Zero-result Rate** | 返回空结果的查询比例 | < 3% |
| **Recall@100** | 前 100 候选中包含相关商品的比例 | > 0.90 |

**评估数据构建:**
- 标注集: 从搜索日志中采样 2,000 条查询，人工标注相关商品 (0-3 分)
- 隐式标注: 点击且购买 = 3, 点击 = 2, 展现未点击 = 0
- 定期更新标注集 (季度)

### Online Evaluation (上线后)

| 指标 | 定义 | 目标 |
|---|---|---|
| **CTR** | 搜索结果点击率 | 12% → 25% |
| **Conversion Rate** | 搜索 → 购买 | 提升 30% |
| **Zero-result Rate** | 实时监控 | < 3% |
| **P99 Latency** | 端到端延迟 | < 200ms |
| **Revenue per Search** | 每次搜索带来的收入 | 提升 20% |

**A/B 测试框架:**
- 用 Vertex AI Experiments 管理实验
- 流量分割: 5% → 20% → 50% → 100% 灰度放量
- 最小检测效应: CTR 提升 2 个百分点，置信度 95%
- Black Friday 期间冻结实验 (全量使用验证过的最优配置)

---

# Part 4: 业务价值量化 (ROI)

## 直接收入影响

```
Current state:
  Annual online revenue:              $2,000,000,000
  Search-driven revenue (est. 60%):   $1,200,000,000
  Cart abandonment from search:       35%

Projected improvement:
  CTR: 12% → 22% (+83%)
  Zero-result rate: 15% → 3%
  Bounce rate (latency): 40% → 10%

Conservative estimate:
  Search conversion improvement:       +15-20%
  Revenue impact:                      $180M - $240M / year
  First-year investment:               ~$2-3M (infra + implementation)
  ROI:                                 60-80x
```

## 运营效率

- 搜索团队从手动调 synonym/rules → 自动语义理解
- 新品上架即可被搜索发现 (无需手动标签)
- 搜索质量实验周期从 weeks → days

## 竞争优势

- "Shop the Look" 是差异化功能 (竞品多数没有)
- 语义搜索能力使长尾查询转化率大幅提升
- 数据飞轮: 搜索数据 → 改进 embedding → 更好的搜索 → 更多数据

---

# Part 5: 实施路线图

## Phase 1: Text Semantic Search MVP (4 weeks)

- [ ] 搭建 Vertex AI Embeddings 管线 (商品 embedding 生成)
- [ ] 部署 Vertex AI Vector Search 索引 (200 万 SKU)
- [ ] 实现 RRF 混合检索 (semantic + keyword)
- [ ] 基础结构化过滤集成
- [ ] Offline evaluation: NDCG > 0.55
- **Demo 目标: 并排对比语义搜索 vs 关键词搜索**

## Phase 2: Image Search & Shop the Look (4 weeks)

- [ ] Multimodal embedding 集成 (统一向量空间)
- [ ] 图片上传 → object detection → 多商品检索
- [ ] "Complete the Look" 品类分组呈现
- [ ] Offline evaluation: image-to-product recall > 0.70

## Phase 3: Personalization & Business Rules (4 weeks)

- [ ] 用户画像构建 (Bigtable)
- [ ] Personalized re-ranking 上线
- [ ] Business rules engine (促销/库存/利润率)
- [ ] A/B 测试框架集成

## Phase 4: Production Hardening & Black Friday Prep (4 weeks)

- [ ] 负载测试 (模拟 1000 QPS)
- [ ] Cache 预热策略
- [ ] 自动扩缩容配置 & 验证
- [ ] Graceful degradation / circuit breaker
- [ ] Monitoring dashboard (Looker)
- [ ] Runbook for Black Friday operations

---

# Part 6: 预期 Q&A 准备

## Q1: 为什么选 Vertex AI Vector Search 而不是继续用 Elasticsearch kNN?

**A:** 四个原因:
1. **延迟**: ES kNN 在百万级向量上 P99 通常 >500ms; Vertex AI Vector Search 基于 Google 的 ScaNN 算法，百万级 P99 <50ms
2. **扩缩容**: ES 需要手动管理 shard/replica; Vector Search 支持按 QPS 自动扩缩，Black Friday 场景关键
3. **原生混合搜索**: Vector Search 现在同时支持 dense + sparse embeddings，一个索引搞定语义+关键词混合搜索
4. **过滤能力**: Token restricts (分类过滤 AND/OR) + Numeric restricts (价格范围等)，原生 pre-filter

如果客户想降低迁移风险，**AlloyDB AI** 是渐进式方案 — 支持 ScaNN/HNSW/IVF 索引，SQL 接口可直接调 Vertex AI embedding 函数，团队学习曲线最低。

## Q2: 如何保证 embedding 质量? 时尚领域是否需要 fine-tune?

**A:**
- 第一步: 用最新的 `gemini-embedding-001` (3072 维，统一多语言) 做 baseline — 比前代 text-embedding-005 性能更强
- 第二步: 用搜索日志 (query-click pairs) 评估是否满足需求
- 第三步: 如果垂直领域效果不够，用 Vertex AI 的 embedding tuning 功能做 fine-tune
- 时尚领域特有挑战: "cottagecore", "coastal grandmother" 等风格术语 → fine-tune 收益明显

## Q3: 成本预估?

**A:**
- Vertex AI Vector Search: ~$0.50/node-hour, 正常 5 nodes ≈ $1,800/月, Black Friday 峰值 25 nodes (临时)
- Embeddings API: ~$0.025/1000 queries → 1M queries/day ≈ $750/月
- AlloyDB: ~$1,500/月 (结构化数据)
- Memorystore Redis: ~$500/月
- **月度总成本: ~$5,000-8,000** (vs 搜索改善带来的 $180M+ 年收入增量)

## Q4: 如何处理商品描述质量参差不齐的问题?

**A:**
- **Enrichment pipeline**: 用 Gemini 对低质量描述 (< 20 words) 自动补充
  - 输入: 商品标题 + 图片 + 属性
  - 输出: 丰富的商品描述 (100-200 words)
- Embedding 生成时拼接: `title + enriched_description + attributes`
- 定期评估 enriched vs original 的搜索质量差异

## Q5: 个性化搜索是否会造成 "filter bubble"?

**A:**
- Personalization 只影响 re-ranking (排序)，不影响 retrieval (召回)
- 权重设计: 70% 相关性 + 20% 个性化 + 10% 业务规则
- 每个搜索结果页保留 "exploration slots" (位置 8-10 放非个性化推荐)
- 提供 "See all results" 开关，让用户关闭个性化

## Q6: 数据隐私和合规?

**A:**
- 所有数据在客户的 GCP project 内处理
- Vertex AI Embeddings API 不会用客户数据训练模型 (data processing terms)
- 搜索日志存储在客户的 BigQuery, 受 IAM + VPC Service Controls 保护
- 个性化数据 (用户画像) 有 TTL，30 天无活动自动清除

---

# Part 7: Demo 设计思路

## 推荐 Demo 方案: Colab / Jupyter Notebook

**演示流程 (5-7 分钟):**

1. **数据准备** — 展示 100 件时尚商品 (预先准备好 JSON)
2. **Embedding 生成** — 调用 Vertex AI Embeddings API
3. **索引构建** — 创建 Vector Search index (或用 AlloyDB pgvector 简化版)
4. **关键词 vs 语义对比** — 并排展示 "floral dress for winter wedding" 的两种结果
5. **图片搜索** — 上传一张穿搭图，展示相似商品检索
6. **混合搜索** — 语义查询 + 价格过滤，展示无缝融合

### 准备要点
- 预先跑通全流程，录屏作为 backup (避免现场 API 超时)
- 数据用真实感的时尚商品 (不用 confidential 数据)
- 代码简洁，关键步骤有注释
- 准备 2-3 个"wow moment"查询 (如 Taylor Swift 那个例子)

---

## References

### Google Cloud 官方文档 (已验证)
- [Vertex AI Vector Search Overview](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- [Vector Search Index Configuration](https://cloud.google.com/vertex-ai/docs/vector-search/configuring-indexes)
- [Vector Search Index Management](https://cloud.google.com/vertex-ai/docs/vector-search/create-manage-index)
- [Vertex AI Text Embeddings (gemini-embedding-001)](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)
- [Multimodal Embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings)
- [Vertex AI Search for Commerce](https://cloud.google.com/retail/docs/overview)
- [AlloyDB AI Embeddings](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings)

### Google 官方 Notebooks & Tutorials
- Vertex AI Vector Search Quickstart
- Getting Started with Text Embeddings and Vector Search
- Combining Semantic & Keyword Search: Hybrid Search Tutorial
- Implement Two-Tower Retrieval for Large-Scale Candidate Generation
- Multimodal Search with Rank-Biased Reciprocal Rank Ensemble

### 学术 & 开源
- [ScaNN: Efficient Vector Similarity Search](https://github.com/google-research/google-research/tree/master/scann)
- [Reciprocal Rank Fusion (Cormack et al.)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Gecko: Text Embeddings Distilled from LLMs (text-embedding-005 research paper)](https://arxiv.org/abs/2403.20327)
