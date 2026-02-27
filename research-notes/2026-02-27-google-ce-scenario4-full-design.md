# Fashion E-Commerce Semantic Search — Technical Design Document

- **Date:** 2026-02-27
- **Author:** Practice Customer Engineer, AI — Google Cloud
- **Scenario:** Retail / E-Commerce — Search & Vector Database
- **Status:** Design Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State & Problem Analysis](#2-current-state--problem-analysis)
3. [Solution Overview](#3-solution-overview)
4. [Technology Selection](#4-technology-selection)
5. [System Architecture](#5-system-architecture)
6. [Text Search Pipeline](#6-text-search-pipeline)
7. [Image Search Pipeline](#7-image-search-pipeline)
8. [Hybrid Search & Filtering](#8-hybrid-search--filtering)
9. [Infrastructure & Scaling](#9-infrastructure--scaling)
10. [Data Ingestion Pipeline](#10-data-ingestion-pipeline)
11. [Personalization](#11-personalization)
12. [Search Quality & Evaluation](#12-search-quality--evaluation)
13. [Business Value](#13-business-value)
14. [Architecture Decisions & Trade-offs](#14-architecture-decisions--trade-offs)
15. [Implementation Roadmap](#15-implementation-roadmap)
16. [Risk & Mitigation](#16-risk--mitigation)
17. [Appendix A: Component Specifications](#appendix-a-component-specifications)
18. [Appendix B: Q&A Reference](#appendix-b-qa-reference)
19. [Appendix C: Demo Design](#appendix-c-demo-design)
20. [References](#references)

---

## 1. Executive Summary

### 客户

一家时尚零售电商，200 万 SKU，年线上营收 $20 亿。

### 问题

当前 Elasticsearch 关键词搜索导致 **35% 购物车放弃**：语义理解缺失（CTR 仅 12%，行业基准 25%）、零结果率 15%、延迟 P99 = 2.5s。

### 方案

基于 Google Cloud 构建**语义向量搜索平台**，核心组件：

- **gemini-embedding-001** — 文本语义向量化
- **multimodalembedding@001** — 图片搜索向量化
- **Vertex AI Vector Search** — 高性能近邻检索（ScaNN 算法）
- **Vertex AI Ranking API** — 精排 re-ranking

### 预期成果

| 指标 | 当前 | 目标 |
|---|---|---|
| 搜索 CTR | 12% | 25% |
| 零结果率 | 15% | < 3% |
| P99 延迟 | 2.5s | < 200ms |
| 搜索转化率 | baseline | +30% |
| 年营收增量 | — | $180M–$240M |

---

## 2. Current State & Problem Analysis

### 2.1 当前基础设施

| 项目 | 现状 |
|---|---|
| 搜索引擎 | Elasticsearch，关键词 BM25 匹配 |
| 延迟 | P50 = 800ms，P99 = 2.5s |
| CTR | 12%（行业基准 25%） |
| 零结果率 | 15% 查询返回空 |
| 跳出率 | 延迟 > 200ms 时 40% 用户离开 |

### 2.2 典型失败案例

| 用户查询 | 当前返回 | 期望返回 | 失败原因 |
|---|---|---|---|
| "floral dress for winter wedding" | 夏季碎花裙 | 长袖/丝绒花裙 | BM25 匹配 "floral dress"，忽略 "winter wedding" 意图 |
| "something blue for a baby shower" | 随机蓝色商品 | 蓝色裙装/配饰/礼品 | 无法理解场合语境 |
| "shoes like Taylor Swift at Eras Tour" | 无结果 | 银色靴子/亮片高跟 | 无流行文化知识 |
| "comfortable heels for standing all day" | 按热度排序的所有高跟 | 粗跟/猫跟/坡跟 | 无法理解功能需求 |

### 2.3 根因分析

```
根因: BM25 词频匹配 ≠ 语义理解

BM25 的工作方式:
  "comfortable heels for standing all day"
  → 分词: [comfortable] [heels] [standing] [all] [day]
  → 在商品文本中匹配这些词的出现频率
  → 不理解 "comfortable for standing" = 粗跟/坡跟/厚底

缺失的三种能力:
  ① 意图理解  — "for winter wedding" 隐含长袖、保暖、正式
  ② 文化语境  — "Taylor Swift Eras Tour" 需要流行文化知识
  ③ 功能推理  — "comfortable for standing" 是对鞋类结构的要求
```

### 2.4 为什么 off-the-shelf 不够

| 方案 | 问题 |
|---|---|
| ES + kNN 插件 | 百万向量 P99 > 500ms；手动 shard 管理；无多模态 |
| 开源向量数据库 (Milvus, Pinecone) | 缺少与 GCP 生态集成；无 Google embedding 模型原生支持 |
| 纯 LLM 方案 | 延迟太高（秒级）；成本不可控；不适合检索场景 |

**需要的 Unblocker:**

```
① Embedding Models    → 文本/图片 → 高维语义向量
② Vector Search       → 百万级 ANN 检索 < 50ms
③ Hybrid Retrieval    → 语义 + 关键词 + 过滤 + 业务规则 融合
④ Auto-scaling Infra  → 应对 Black Friday 20x 流量
```

---

## 3. Solution Overview

### 3.1 核心思路

将商品和用户查询映射到**同一语义向量空间**，使得语义相近的内容在向量空间中距离更近。

```
传统方式: query → 关键词匹配 → 按词频排序
新方式:   query → embedding → 向量近邻搜索 → 按语义相关性排序
```

### 3.2 解决方案映射

| 客户需求 | 解决方案 |
|---|---|
| 自然语言搜索 | gemini-embedding-001 语义向量 + Vector Search ANN |
| 图片搜索 | multimodalembedding@001 + Gemini Vision 物体检测 |
| P99 < 200ms | Vector Search ScaNN + Redis 缓存 + CDN |
| 混合搜索 | Dense + Sparse 双路检索 + Token/Numeric Restricts |
| 离线评估 | NDCG/MRR 离线指标 + A/B 在线实验 |

### 3.3 系统边界

```
IN SCOPE:
  ✓ 文本语义搜索
  ✓ 图片搜索 (Shop the Look)
  ✓ 混合检索 (语义 + 关键词 + 结构化过滤)
  ✓ 业务规则引擎 (boost/bury/pin)
  ✓ 基础个性化 (用户历史加权)
  ✓ Black Friday 扩缩容
  ✓ 搜索质量评估框架

OUT OF SCOPE:
  ✗ 推荐系统 (可后续集成)
  ✗ 自然语言对话式搜索
  ✗ 商品描述自动生成 (仅做 enrichment)
```

---

## 4. Technology Selection

### 4.1 组件清单

| 层 | 组件 | Google Cloud 产品 | 用途 |
|---|---|---|---|
| Embedding | 文本向量化 | **gemini-embedding-001** | 查询/商品 → 语义向量 |
| Embedding | 图片向量化 | **multimodalembedding@001** | 商品图片/用户上传照片 → 向量 |
| Retrieval | 向量检索 | **Vertex AI Vector Search** | ANN 近邻搜索 (ScaNN) |
| Retrieval | 精排 | **Vertex AI Ranking API** | Cross-encoder 重排序 |
| Storage | 结构化数据 | **AlloyDB** | 商品属性/价格/库存 |
| Storage | 用户画像 | **Bigtable** | 低延迟 K-V 读取 |
| Storage | 日志分析 | **BigQuery** | 搜索日志/离线评估 |
| Cache | 查询缓存 | **Memorystore (Redis)** | 热门查询结果缓存 |
| Compute | 应用服务 | **Cloud Run** | 无服务器，自动扩缩容 |
| Pipeline | 数据管道 | **Dataflow** | 实时 embedding 生成 |
| Network | 边缘加速 | **Cloud CDN + Load Balancer** | 静态缓存 + 流量分发 |
| Monitor | 可观测性 | **Cloud Monitoring + Looker** | 延迟/错误率/搜索质量 |

### 4.2 关键选型理由

**为什么 gemini-embedding-001 而不是 text-embedding-005？**

| 维度 | gemini-embedding-001 | text-embedding-005 |
|---|---|---|
| 维度 | 最高 3072（可降维） | 最高 768 |
| 语言 | 多语言 + 代码 | 仅英文 + 代码 |
| 性能 | 全面超越前代 | 前代 |
| 序列长度 | 2048 tokens | 2048 tokens |

> 官方原文: "gemini-embedding-001 unifies the previously specialized models and achieves better performance in their respective domains."

**为什么 Vector Search 而不是继续 Elasticsearch kNN？**

| 维度 | Vertex AI Vector Search | Elasticsearch kNN |
|---|---|---|
| 算法 | ScaNN (Google Research) | HNSW |
| 百万级 P99 | < 50ms | > 500ms |
| 扩缩容 | 自动按 QPS | 手动管理 shard |
| 混合搜索 | 原生 dense + sparse | 需额外配置 |
| 过滤 | 原生 Token/Numeric Restricts | Script scoring |
| 生态集成 | Vertex AI 原生 | 需自建管道 |

---

## 5. System Architecture

### 5.1 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                            │
│  ┌───────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ Mobile App│  │ Web Browser  │  │ Partner API           │ │
│  └─────┬─────┘  └──────┬───────┘  └───────────┬───────────┘ │
└────────┼───────────────┼──────────────────────┼─────────────┘
         └───────────────┼──────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                 Edge Layer                                    │
│  Cloud CDN (autocomplete 缓存) → Cloud Load Balancer         │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Application Layer (Cloud Run)                  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Text Search  │  │ Image Search │  │ Autocomplete      │ │
│  │ Service      │  │ Service      │  │ Service           │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────┘ │
└─────────┼─────────────────┼─────────────────────────────────┘
          │                 │
┌─────────▼─────────────────▼─────────────────────────────────┐
│               Retrieval Layer                                │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Memorystore (Redis) — Query Result Cache               │ │
│  │ key: hash(query_vec + filters + rules_ver)  TTL: 5min  │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │ cache miss                           │
│  ┌────────────────────▼───────────────────────────────────┐ │
│  │ Vertex AI Vector Search                                │ │
│  │ ┌───────────────────┐  ┌─────────────────────────────┐│ │
│  │ │ Text Index         │  │ Image Index                 ││ │
│  │ │ 768-dim            │  │ 1408-dim                    ││ │
│  │ │ dense + sparse     │  │ multimodal                  ││ │
│  │ │ Token/Numeric      │  │ Token restricts             ││ │
│  │ │ Restricts          │  │                             ││ │
│  │ └───────────────────┘  └─────────────────────────────┘│ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Vertex AI Ranking API — Cross-encoder Re-ranking       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Business Rules Engine — boost / bury / pin / filter    │ │
│  │ + Personalization Layer                                │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│               Data Layer                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ AlloyDB      │  │ Bigtable     │  │ BigQuery           │ │
│  │ 商品属性      │  │ 用户画像      │  │ 搜索日志           │ │
│  │ 价格/库存     │  │ 行为历史      │  │ 离线评估           │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│               Ingestion Pipeline (Dataflow)                   │
│  Product Feed → Enrichment → Embedding → Index Update         │
│                                        → AlloyDB Write        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│               Observability                                   │
│  Cloud Monitoring → Looker Dashboard → Alerting               │
│  Search Logs → BigQuery → Vertex AI Experiments               │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 请求流 (Text Search)

```
① User types "floral dress for winter wedding"
② Cloud CDN / LB routes to Cloud Run
③ Cloud Run checks Redis cache → miss
④ gemini-embedding-001 generates 768-dim query vector
⑤ Vector Search: dense ANN + sparse BM25 parallel retrieval
⑥ Token/Numeric restricts applied (category, price, inventory)
⑦ RRF fuses dense + sparse results → top 100
⑧ Ranking API re-ranks → top 50
⑨ Business rules (boost promoted, bury low-stock)
⑩ Personalization re-weight → final top 20
⑪ Cache result in Redis (TTL 5min)
⑫ Return to client
```

### 5.3 请求流 (Image Search)

```
① User uploads outfit photo
② Cloud Run receives image
③ Gemini Vision API detects objects: dress, shoes, bag
④ Each crop → multimodalembedding@001 → 1408-dim vector
⑤ Vector Search: per-item ANN on Image Index → top 20 each
⑥ Results grouped by category: "Similar dresses | shoes | bags"
⑦ Return to client
```

---

## 6. Text Search Pipeline

### 6.1 Query Understanding

| 步骤 | 功能 | 实现 |
|---|---|---|
| 拼写纠正 | "florall" → "floral" | Cloud Natural Language API 或自定义字典 |
| 查询扩展 | "dress" → "dress, gown, frock" | Synonym mapping + LLM-based expansion |
| 意图识别 | 判断是浏览/精确查找/比较 | Lightweight classifier |

### 6.2 Embedding 生成

```python
# Query side
from google import genai
from google.genai.types import EmbedContentConfig

client = genai.Client()
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["floral dress for winter wedding"],
    config=EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768,
    ),
)
query_vector = response.embeddings[0].values  # 768-dim
```

```python
# Product side (ingestion time)
response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[f"{product.title}. {product.description}. {product.attributes}"],
    config=EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    ),
)
product_vector = response.embeddings[0].values  # 768-dim
```

**关键参数选择:**
- `output_dimensionality=768`：200 万向量 × 768 维 ≈ 6 GB，MEDIUM shard 足够容纳
- `task_type` 区分 query/document：官方推荐的最佳实践，提升检索质量

### 6.3 双路检索 + RRF 融合

```
           query
             │
     ┌───────┴───────┐
     ▼               ▼
┌─────────┐   ┌─────────┐
│ Dense   │   │ Sparse  │
│ ANN     │   │ BM25    │
│ top-100 │   │ top-100 │
└────┬────┘   └────┬────┘
     │             │
     └──────┬──────┘
            ▼
   RRF Score Fusion
   score(d) = Σ 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))
   k = 60 (standard)
            │
            ▼
      merged top-100
```

**为什么需要两路：**

| 场景 | Dense 表现 | Sparse 表现 | 谁赢 |
|---|---|---|---|
| "comfortable heels for standing" | 理解功能需求 → 粗跟 | 仅匹配 "heels" | Dense |
| "Nike Air Max 90" | 返回各种运动鞋 | 精确命中型号 | Sparse |
| "blue summer dress" | 理解蓝色+夏季 | 匹配 "blue"+"dress" | 互补 |

Vector Search **原生支持 dense + sparse 在同一索引**，一次请求完成两路检索。

### 6.4 Re-ranking

```
RRF top-100 → Vertex AI Ranking API → top-50

Ranking API 使用 cross-encoder 模型:
  输入: (query, candidate_text) pair
  输出: relevance score (比 embedding 相似度更精确)
  延迟: ~30ms for 100 candidates
```

### 6.5 Business Rules 层

```python
for item in ranked_results:
    score = item.ranking_score

    # Boost promoted items
    if item.is_promoted:
        score *= 1.2

    # Bury low-stock items
    if item.inventory < 10:
        score *= 0.7

    # Pin contractual items to specific positions
    if item.is_pinned:
        pin_to_position(item, item.pin_slot)

    # Margin optimization
    score += item.margin_score * 0.05

    item.final_score = score
```

---

## 7. Image Search Pipeline

### 7.1 物体检测

```
用户上传图片 (Instagram 穿搭)
         │
         ▼
┌──────────────────────────────┐
│ Gemini Vision API            │
│ Prompt: "Identify each       │
│ clothing item and accessory. │
│ Return bounding boxes and    │
│ category labels."            │
│                              │
│ Output:                      │
│ ┌──────┐ ┌──────┐ ┌──────┐ │
│ │Dress │ │Shoes │ │ Bag  │ │
│ │bbox1 │ │bbox2 │ │bbox3 │ │
│ └──────┘ └──────┘ └──────┘ │
└──────────────────────────────┘
```

### 7.2 Per-item Embedding

```python
# For each detected item crop
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel

model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

# Image embedding (per crop)
embedding = model.get_embeddings(
    image=crop_image,
    dimension=1408,  # or 128/256/512 for faster search
)
image_vector = embedding.image_embedding  # 1408-dim
```

### 7.3 向量空间设计: 两套索引

| 索引 | 用途 | 模型 | 维度 | 商品入库数据 |
|---|---|---|---|---|
| **Text Index** | 文本搜索 | gemini-embedding-001 | 768 | title + description + attributes |
| **Image Index** | 图片搜索 | multimodalembedding@001 | 1408 | 主商品图片 |

**为什么分离而非统一:**

- `multimodalembedding@001` 文本仅支持 **32 tokens**（约 32 词），无法编码完整商品描述
- `gemini-embedding-001` 支持 **2048 tokens** 但不支持图片输入
- 文本搜索是 **80% 的流量**，不能因迁就多模态而牺牲文本搜索质量
- 通过 **应用层融合**（RRF / weighted ensemble）实现跨模态统一体验

### 7.4 混合查询: "like this but in black"

```
用户上传图片 + 输入文字 "but in black"
  │
  ├── 图片 → multimodalembedding@001 → image_vector
  ├── 文字 → multimodalembedding@001 → text_vector (短文本，32 tokens 够用)
  │
  └── combined_vector = 0.7 * image_vector + 0.3 * text_vector
      → Image Index ANN search
      → + Token restrict: color = "black"
```

---

## 8. Hybrid Search & Filtering

### 8.1 过滤策略: Pre-filter

```
查询: "summer dress" + filter: size=M, price < $50

策略: Pre-filter (先过滤，再向量搜索)

原因:
  200 万 SKU
  → size=M 过滤后约 40 万
  → price < $50 再过滤后约 10 万
  → 对 10 万条做 ANN 比对 200 万条快得多
```

### 8.2 Vector Search 过滤配置

```json
// 商品数据格式 (入库时)
{
  "id": "SKU-123456",
  "embedding": [0.12, -0.34, ...],  // 768-dim dense
  "sparse_embedding": {"values": [0.8, 0.2], "dimensions": [1024, 5678]},
  "restricts": [
    {"namespace": "category", "allow": ["dress", "women"]},
    {"namespace": "color", "allow": ["blue", "navy"]},
    {"namespace": "size", "allow": ["S", "M", "L"]},
    {"namespace": "brand", "allow": ["zara"]}
  ],
  "numeric_restricts": [
    {"namespace": "price", "value_float": 49.99},
    {"namespace": "inventory", "value_int": 156}
  ]
}
```

```json
// 查询时 (过滤条件)
{
  "deployed_index_id": "fashion_index",
  "queries": [{
    "datapoint": {
      "feature_vector": [0.23, -0.11, ...],
      "restricts": [
        {"namespace": "category", "allow": ["dress"]},
        {"namespace": "size", "allow": ["M"]}
      ],
      "numeric_restricts": [
        {"namespace": "price", "value_float": 50.0, "op": "LESS"}
      ]
    },
    "neighbor_count": 100
  }]
}
```

**过滤逻辑:**
- 跨 namespace: **AND**
- 同 namespace 内: **OR**
- 示例: `category=dress AND (size=S OR size=M) AND price < 50`

---

## 9. Infrastructure & Scaling

### 9.1 容量规划

| 指标 | 日常 | Black Friday | 倍数 |
|---|---|---|---|
| QPS (avg) | 12 | 350 | 29x |
| QPS (peak) | 50 | 1,000 | 20x |
| 持续时间 | 全天均匀 | 8 小时集中 | — |
| 年营收占比 | — | **20%** | — |

### 9.2 四层防御

```
┌─────────────────────────────────────────────┐
│ Layer 1: Cloud CDN                          │
│ 缓存 autocomplete、类目页                    │
│ 削峰: ~20% 请求在边缘返回                    │
├─────────────────────────────────────────────┤
│ Layer 2: Redis Cache                         │
│ 热门查询结果 (TTL 5min)                      │
│ Black Friday 前 2 天预热 top 10K 查询        │
│ 目标: 85% hit rate → 到 Vector Search 仅 150 QPS │
├─────────────────────────────────────────────┤
│ Layer 3: Auto-scaling                        │
│ Vector Search: min 5 → max 25 nodes          │
│ Cloud Run: min 2 → max 50 instances          │
│ 提前 1 小时 warm-up 到 15 nodes              │
├─────────────────────────────────────────────┤
│ Layer 4: Graceful Degradation                │
│ 熔断: 10s 内 >5% 错误 → fallback keyword     │
│ 效果: 搜索质量略降，服务不中断                 │
└─────────────────────────────────────────────┘
```

### 9.3 延迟预算 (P99 < 200ms)

| 步骤 | 耗时 | 备注 |
|---|---|---|
| 网络 (CDN → LB → Cloud Run) | 20ms | |
| Redis 查找 | 5ms | hit 直接返回 |
| Embedding 生成 | 30ms | gemini-embedding-001 |
| Vector Search ANN | 40ms | ScaNN on 200 万向量 |
| 结构化过滤 | — | 与 ANN 并行 (restricts) |
| RRF 融合 | 10ms | |
| Ranking API re-rank | 30ms | 100 candidates |
| 业务规则 + 个性化 | 15ms | |
| 序列化 + 响应 | 10ms | |
| **缓冲** | **40ms** | |
| **总计** | **200ms** | |

### 9.4 Vector Search 节点规划

| 场景 | 节点数 | 机器类型 | Shard | 月费 (估) |
|---|---|---|---|---|
| 日常 | 5 | e2-standard-16 | MEDIUM | ~$1,800 |
| Black Friday peak | 25 | e2-standard-16 | MEDIUM | ~$9,000 (临时) |
| 年均 | ~6 | — | — | ~$2,200 |

---

## 10. Data Ingestion Pipeline

### 10.1 商品入库流程

```
┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌──────────────┐
│ Product  │───▶│ Dataflow  │───▶│ Embedding    │───▶│ Vector Search│
│ Feed     │    │ ETL       │    │ Generation   │    │ Index Update │
│ (GCS)    │    │           │    │              │    │ (streaming)  │
└──────────┘    └─────┬─────┘    └──────────────┘    └──────────────┘
                      │
                      ├──────────────────────────▶ AlloyDB (结构化数据)
                      │
                      └──────────────────────────▶ BigQuery (日志/分析)
```

### 10.2 Enrichment — 低质量描述补全

```
商品描述 < 20 words ?
  ├── Yes → Gemini 生成丰富描述
  │         输入: title + image + attributes
  │         输出: 100-200 word description
  │
  └── No → 直接使用原始描述

Embedding 输入 = title + " | " + enriched_description + " | " + attributes_string
```

### 10.3 更新频率

| 数据类型 | 更新方式 | 频率 |
|---|---|---|
| 新品上架 (50K/月) | Streaming index update | 实时 |
| 下架 (30K/月) | Streaming delete | 实时 |
| 价格变动 | AlloyDB update + 过滤条件更新 | 实时 |
| 库存变动 | Numeric restrict update | 近实时 (5min) |
| 全量重建 | Batch index rebuild | 周级别 |

---

## 11. Personalization

### 11.1 信号

| 信号 | 存储 | 用途 |
|---|---|---|
| 购买历史 | Bigtable | 用户偏好向量 (历史购买商品 embedding 加权平均) |
| 浏览历史 (30 天) | Bigtable | 短期兴趣 |
| 价格偏好 | Bigtable | avg order value → 价格区间偏好 |
| 尺码/品牌偏好 | Bigtable | 高频选项 |

### 11.2 融合公式

```
final_score = 0.7 × relevance_score
            + 0.2 × personalization_score
            + 0.1 × business_rule_score
```

- 个性化**仅影响排序** (re-ranking)，不影响召回 (retrieval)
- 查询时拉取用户画像 (Bigtable 延迟 < 5ms)

### 11.3 防 Filter Bubble

- 每页第 8–10 位保留 **exploration slots**（非个性化推荐）
- 提供 "See all results" 开关让用户关闭个性化
- 新用户无画像时 fallback 到纯相关性排序

---

## 12. Search Quality & Evaluation

### 12.1 离线指标 (上线前验证)

| 指标 | 定义 | 目标 |
|---|---|---|
| **NDCG@10** | 前 10 结果排序质量 | > 0.65 |
| **MRR** | 首个相关结果排名倒数 | > 0.45 |
| **Recall@100** | top-100 候选含相关商品比例 | > 0.90 |
| **Zero-result Rate** | 无结果查询占比 | < 3% |

**评估集构建:**
- 采样 2,000 条搜索日志查询
- 标注规则: 点击且购买=3, 点击=2, 展现未点击=0
- 季度更新评估集

### 12.2 在线指标 (持续监控)

| 指标 | 当前 | 目标 | 监控频率 |
|---|---|---|---|
| CTR | 12% | 25% | 实时 |
| Zero-result Rate | 15% | < 3% | 实时 |
| P99 Latency | 2.5s | < 200ms | 实时 |
| Conversion Rate | baseline | +30% | 日级 |
| Revenue per Search | baseline | +20% | 日级 |

### 12.3 A/B 测试框架

```
灰度策略: 5% → 20% → 50% → 100%
最小检测效应: CTR 提升 2pp, 置信度 95%
实验管理: Vertex AI Experiments

重要规则:
  - Black Friday 冻结所有实验
  - 全量使用已验证的最优配置
  - 事后分析 Black Friday 数据
```

---

## 13. Business Value

### 13.1 营收影响

```
Current:
  年线上营收:              $2,000,000,000
  搜索驱动营收 (est. 60%): $1,200,000,000
  搜索相关购物车放弃:       35%

Projected:
  CTR:        12% → 22%  (+83%)
  零结果率:    15% → 3%
  跳出率:      40% → 10%

Conservative estimate:
  搜索转化提升:   +15~20%
  年营收增量:     $180M – $240M
```

### 13.2 成本分析

| 组件 | 月费用 |
|---|---|
| Vector Search (5 nodes) | $1,800 |
| Embeddings API (1M/day) | $750 |
| Ranking API | $500 |
| AlloyDB | $1,500 |
| Memorystore Redis | $500 |
| Cloud Run | $800 |
| 其他 (监控/日志/存储) | $650 |
| **月度总计** | **~$6,500** |

**ROI: $6.5K/月 投入 vs $15M+/月营收增量 → 2300x**

### 13.3 非量化收益

- **竞争差异化**: "Shop the Look" 图片搜索是行业领先功能
- **运营效率**: 新品上架即可被搜索发现，无需手动打标签
- **数据飞轮**: 搜索数据 → 改进 embedding → 更好搜索 → 更多数据

---

## 14. Architecture Decisions & Trade-offs

### 14.1 全托管 vs 自建

| 维度 | Vertex AI Search for Commerce | 自建 (Vector Search + Embeddings) |
|---|---|---|
| 上线时间 | 2-3 个月 | 4-6 个月 |
| 自定义程度 | 中 | 高 |
| 图片搜索 | 不原生支持 | 自由实现 |
| 维护成本 | 低 | 高 |
| 个性化 | 内置 | 自建更灵活 |
| 成本模式 | 按查询 | 按节点 |

**决策**: 自建方案。原因:
1. 客户需要 Shop the Look（Commerce Search 不原生支持）
2. 自建方案可完全控制向量空间和排序逻辑
3. 后续可评估是否将文本搜索迁移至 Commerce Search

### 14.2 向量空间: 统一 vs 分离

| 维度 | 统一空间 | 分离空间 (Text + Image) |
|---|---|---|
| 架构复杂度 | 低 | 中 |
| 文本质量 | 受限 (multimodal 仅 32 tokens) | **高 (2048 tokens)** |
| 图片质量 | 高 | 高 |
| 跨模态查询 | 天然支持 | 需应用层融合 |

**决策**: 分离空间。原因: 文本搜索占 80% 流量，不能因迁就多模态而牺牲文本质量。

### 14.3 Pre-filter vs Post-filter

| 维度 | Pre-filter | Post-filter |
|---|---|---|
| 流程 | 先过滤再搜索 | 先搜索再过滤 |
| 延迟 | 更低 | 更高 |
| 精确度 | 可能漏掉边缘结果 | 更完整 |

**决策**: Pre-filter。原因: 200 万 SKU + 强过滤条件下可缩小搜索空间 5-20 倍。Vector Search 原生 restricts 天然支持 pre-filter。

### 14.4 Embedding 维度: 768 vs 3072

| 维度 | 768 | 3072 |
|---|---|---|
| 索引大小 | ~6 GB | ~24 GB |
| 搜索延迟 | 更低 | 更高 |
| 质量 | 足够 | 略优 |
| 成本 | 更低 | 更高 |

**决策**: 768 维。原因: 精度差异小，但延迟和成本差异大。通过 `output_dimensionality=768` 降维。

---

## 15. Implementation Roadmap

```
Phase 1: Text Semantic Search MVP                     Week 1-4
  ├── Embedding pipeline (gemini-embedding-001)
  ├── Vector Search index (2M SKU, streaming)
  ├── Dense + Sparse hybrid retrieval (RRF)
  ├── Basic structured filtering (restricts)
  ├── Offline eval: NDCG > 0.55
  └── ✓ Deliverable: 可演示的 Demo

Phase 2: Image Search & Shop the Look                 Week 5-8
  ├── Multimodal embedding integration
  ├── Gemini Vision object detection
  ├── Image Index (1408-dim)
  ├── Category-grouped results
  ├── Offline eval: image recall > 0.70
  └── ✓ Deliverable: 图片搜索端到端可用

Phase 3: Ranking + Personalization + Biz Rules         Week 9-12
  ├── Ranking API integration (re-ranking)
  ├── User profile pipeline (Bigtable)
  ├── Business rules engine (boost/bury/pin)
  ├── A/B test framework (Vertex AI Experiments)
  └── ✓ Deliverable: 个性化搜索 + A/B 测试上线

Phase 4: Production Hardening & Black Friday Prep      Week 13-16
  ├── Load test: 1000 QPS sustained
  ├── Cache warm-up strategy
  ├── Auto-scaling validation
  ├── Circuit breaker + graceful degradation
  ├── Monitoring dashboard (Looker)
  ├── Runbook for Black Friday ops
  └── ✓ Deliverable: 生产就绪
```

---

## 16. Risk & Mitigation

| 风险 | 影响 | 可能性 | 缓解措施 |
|---|---|---|---|
| Embedding 模型效果不足 (时尚垂域) | 搜索质量未达标 | 中 | Fine-tune embedding；Two-Tower 自训练模型 |
| Black Friday 流量超预期 | 服务降级/中断 | 低 | 四层防御 + 提前压测 + graceful degradation |
| 商品描述质量差 | Embedding 质量低 | 高 | Enrichment pipeline (Gemini 补全低质量描述) |
| Vector Search 索引构建超时 | 上线延迟 | 低 | 提前规划；streaming update 做增量 |
| 个性化导致 filter bubble | 用户体验下降 | 中 | Exploration slots + 用户控制开关 |
| Redis 缓存击穿 (Black Friday) | 延迟飙升 | 中 | 预热 + 布隆过滤器 + 限流 |
| API 配额不足 | Embedding 生成阻塞 | 低 | 提前申请配额；批量预计算 |

---

## Appendix A: Component Specifications

### A.1 gemini-embedding-001

| 参数 | 值 | 来源 |
|---|---|---|
| 最大维度 | 3072 | 官方文档 |
| 可选维度 | 任意 ≤ 3072 (via `output_dimensionality`) | 官方文档 |
| 序列长度 | 2048 tokens | 官方文档 |
| 语言 | 英文 + 多语言 + 代码 | 官方文档 |
| Task types | RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, etc. | 官方文档 |
| API 限制 | 250 texts/request, 20K tokens/request | 官方文档 |
| 截断行为 | 默认静默截断，`autoTruncate=false` 可报错 | 官方文档 |

### A.2 multimodalembedding@001

| 参数 | 值 | 来源 |
|---|---|---|
| 默认维度 | 1408 | 官方文档 |
| 可选维度 | 128, 256, 512 | 官方文档 |
| 文本限制 | 32 tokens (~32 words), 仅英文 | 官方文档 |
| 图片格式 | BMP, GIF, JPG, PNG | 官方文档 |
| 图片大小 | 最大 20MB，内部缩放到 512x512 | 官方文档 |
| 视频 | 最长分析 2 分钟 | 官方文档 |
| QPS 限制 | 120-600 req/min (按 region) | 官方文档 |

### A.3 Vertex AI Vector Search

| 参数 | 值 | 来源 |
|---|---|---|
| 算法 | TreeAH (ScaNN), Brute-force | 官方文档 |
| 距离 | DOT_PRODUCT (推荐), COSINE, SQUARED_L2, L1 | 官方文档 |
| 归一化 | UNIT_L2_NORM, NONE | 官方文档 |
| Shard 大小 | SMALL 2G, MEDIUM 20G, LARGE 50G | 官方文档 |
| 过滤 | Token restricts + Numeric restricts | 官方文档 |
| 更新方式 | BATCH_UPDATE, STREAM_UPDATE | 官方文档 |
| 官方建议 | DOT_PRODUCT + UNIT_L2_NORM > COSINE | 官方文档 |

### A.4 AlloyDB AI (备选方案)

| 参数 | 值 | 来源 |
|---|---|---|
| 向量索引 | ScaNN, HNSW, IVF, IVFFLAT | 官方文档 |
| Embedding 函数 | `google_ml.embedding(model_id, content)` | 官方文档 |
| 支持模型 | gemini-embedding-001, text-embedding-005, OpenAI | 官方文档 |
| 架构 | 计算/存储分离，PostgreSQL 兼容 | 官方文档 |
| 适用场景 | SQL 团队、中小规模、混合 OLTP+向量搜索 | 官方文档 |

---

## Appendix B: Q&A Reference

### Q1: 为什么不继续用 ES + kNN？

ES kNN 在百万向量 P99 > 500ms；需手动 shard 管理；无原生多模态。Vector Search 基于 ScaNN，P99 < 50ms，自动扩缩，原生 dense+sparse 混合。备选: AlloyDB AI (SQL 接口 + ScaNN 索引)。

### Q2: Embedding 要不要 fine-tune？

三步走: ① gemini-embedding-001 直接用做 baseline → ② 搜索日志评估 → ③ 不够再 fine-tune。Vertex AI 提供 embedding tuning 功能。时尚术语 ("cottagecore", "coastal grandmother") 可能需要 fine-tune。

### Q3: 新品冷启动？

向量搜索基于内容 (content-based)，不依赖行为数据。新品入库即有 embedding，立即可被搜索发现。额外措施: 7 天 exploration boost + 同品类 proxy signal。

### Q4: 成本？

月度约 $6.5K。对比年营收增量 $180M+，ROI > 2000x。

### Q5: 数据隐私？

全部在客户 GCP project 内。Vertex AI Embedding API 不用客户数据训练。搜索日志受 IAM + VPC Service Controls 保护。

### Q6: 个性化的 filter bubble？

个性化仅影响排序不影响召回。权重 70/20/10。Exploration slots + 用户开关。

### Q7: Embedding 效果不够怎么办？

三条路: ① 增加维度 (768→3072) ② Fine-tune ③ Two-Tower 自训练模型（Google 官方参考架构）。

---

## Appendix C: Demo Design

### 环境

Colab / Jupyter Notebook，预先跑通全流程，录屏作为 backup。

### 流程 (5-7 分钟)

```
Step 1: 展示 100 件时尚商品数据 (JSON)
Step 2: 调用 gemini-embedding-001 生成 embedding
Step 3: 创建 Vector Search index (或 AlloyDB pgvector 简化版)
Step 4: 关键词 vs 语义搜索并排对比
        → "floral dress for winter wedding" 的两种结果差异
Step 5: 图片上传搜索 demo
Step 6: 混合搜索: "summer dress" + filter "under $50"
```

### Wow Moments

1. "shoes like Taylor Swift at Eras Tour" — 从零结果到精准推荐
2. 上传一张穿搭照 → 立即找到相似单品
3. "comfortable heels" → 粗跟/坡跟 vs 以前的全部高跟鞋

---

## Appendix D: 演讲节奏 (30 分钟)

| 环节 | 时长 | 内容 | 本文对应 |
|---|---|---|---|
| Title & Context | 2 min | 自我介绍 + 场景 | §1 |
| Specialized Challenge | 5 min | ES 为什么不够 + Unblocker | §2 |
| **Tech Deep-Dive + Demo** | **12 min** | 架构 + 管线 + Live Demo | §5-8, App C |
| Architecture Decisions | 5 min | 选型对比 + Trade-offs | §14 |
| Business Value | 3 min | ROI 测算 | §13 |
| Implementation | 3 min | 路线图 | §15 |

---

## References

### 官方文档 (已验证)
- [Vector Search Overview](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- [Vector Search Index Configuration](https://cloud.google.com/vertex-ai/docs/vector-search/configuring-indexes)
- [Text Embeddings API (gemini-embedding-001)](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)
- [Multimodal Embeddings API](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings)
- [Vertex AI Search for Commerce](https://cloud.google.com/retail/docs/overview)
- [AlloyDB AI Embeddings](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings)
- [AlloyDB Overview](https://cloud.google.com/alloydb/docs/overview)

### 官方 Tutorials & Notebooks
- Vector Search Quickstart
- Hybrid Search Tutorial (Semantic + Keyword with Vector Search)
- Two-Tower Retrieval for Large-Scale Candidate Generation
- Multimodal Search with Rank-Biased Reciprocal Rank Ensemble
- Smart Shopping Assistant with AlloyDB

### 客户案例 (官方引用)
- **eBay** — Vector Search 商品推荐
- **Mercado Libre** — 市场平台搜索改善
- **Bloomreach** — 电商搜索性能和转化率提升

### 学术
- [ScaNN: Efficient Vector Similarity Search (Google Research)](https://github.com/google-research/google-research/tree/master/scann)
- [Reciprocal Rank Fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Gecko: Text Embeddings Distilled from LLMs (Google, 2024)](https://arxiv.org/abs/2403.20327)
