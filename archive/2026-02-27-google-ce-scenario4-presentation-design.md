# Scenario 4: Retail Search & Vector Database — Presentation Design

- **Date:** 2026-02-27
- **Tags:** Google Cloud, CE Interview, Search, Vector Database, Retail

---

# 1. 场景概要

## 1.1 客户画像

| 项目 | 数据 |
|---|---|
| 行业 | 时尚零售电商 |
| SKU 规模 | 200 万活跃商品 |
| 年线上营收 | $20 亿 |
| 日常搜索量 | 100 万次/天 (峰值 50 QPS) |
| Black Friday | 1000 万次/8 小时 (峰值 1000 QPS) |
| 移动端占比 | 70% |

## 1.2 核心痛点

```
当前 Elasticsearch 关键词搜索的三个致命问题:

1. 语义失败 → 搜 "floral dress for winter wedding" 返回夏季碎花裙
2. 零结果率 15% → 搜 "shoes like Taylor Swift at Eras Tour" 无结果
3. 延迟过高 → P99 = 2.5s，超 200ms 时 40% 用户直接离开
```

## 1.3 客户诉求 (5 条)

1. **自然语言搜索** — 理解意图，不只匹配关键词
2. **图片搜索 (Shop the Look)** — 上传照片找相似商品
3. **P99 < 200ms** — 包括 Black Friday
4. **混合搜索** — 语义 + 结构化过滤 + 业务规则 融为一体
5. **离线评估框架** — 上线前验证排序质量

## 1.4 必须回答的 5 个问题

- [ ] Black Friday 10x 流量怎么扩缩容？
- [ ] 语义搜索如何与结构化过滤结合？
- [ ] 图片搜索和文本搜索用同一向量空间还是分开？
- [ ] 搜索质量怎么度量？
- [ ] 新品没有行为数据，冷启动怎么解决？

---

# 2. 为什么现有方案失败 — "Unblocker" 分析

> 演讲第 2 环节: "The Specialized Challenge"，约 5 分钟。

## 2.1 Elasticsearch BM25 的根本局限

```
用户查询: "comfortable heels for standing all day"

BM25 做了什么:
  → 分词: [comfortable] [heels] [standing] [all] [day]
  → 在商品标题/描述中匹配这些词
  → 按词频相关度排序

结果: 返回所有含 "heels" 的商品，按热度排序
问题: BM25 不理解 "comfortable for standing" = 粗跟/坡跟/厚底
```

**根因: 关键词匹配 ≠ 语义理解**

## 2.2 三个维度的差距

| 维度 | Elasticsearch 现状 | 客户需要 |
|---|---|---|
| **语义理解** | 词频匹配，无法理解意图 | 理解 "winter wedding" 隐含长袖/保暖 |
| **多模态** | 仅文本，无图片能力 | 上传穿搭照找同款 |
| **延迟/扩缩容** | P99 = 2.5s，手动扩容 | P99 < 200ms，自动应对 10x 流量 |

## 2.3 Unblocker: 语义向量搜索

```
核心思路:
  把商品和查询都转化为高维向量 (embedding)
  → 语义相近的内容在向量空间中距离更近
  → "comfortable heels for standing" 与 "block heels" 向量距离近
  → "floral dress for winter wedding" 与 "long-sleeve velvet dress" 向量距离近

技术实现:
  Embedding Model  → 将文本/图片转为向量
  Vector Database  → 高效近邻搜索 (ANN)
  Hybrid Retrieval → 语义 + 关键词 + 过滤 + 业务规则
```

---

# 3. 技术选型

> 演讲第 3 环节: "Technical Deep-Dive"，约 12 分钟。

## 3.1 Google Cloud 组件清单

| 功能 | 组件 | 说明 |
|---|---|---|
| 文本 Embedding | **gemini-embedding-001** | 最新旗舰，3072 维，2048 tokens，多语言 |
| 图片 Embedding | **multimodalembedding@001** | 1408 维，图文同空间，图片搜索专用 |
| 向量搜索 | **Vertex AI Vector Search** | ScaNN 算法，支持 dense+sparse 混合，原生过滤 |
| Re-ranking | **Vertex AI Ranking API** | 预训练语言模型精排，提升排序质量 |
| 结构化存储 | **AlloyDB** 或 **Cloud SQL** | 商品属性、价格、库存等结构化数据 |
| 缓存 | **Memorystore (Redis)** | 热门查询缓存，降低延迟 |
| 应用层 | **Cloud Run** | 无服务器，自动扩缩容 |
| 数据管道 | **Dataflow** | 商品入库时实时生成 embedding |
| 监控 | **Cloud Monitoring + BigQuery** | 搜索日志分析、质量监控 |

### 关键模型参数 (官方文档验证)

**gemini-embedding-001:**
- 输出维度: 最高 3072，可通过 `output_dimensionality` 降至 256/512/768
- 序列长度: 2048 tokens
- Task types: `RETRIEVAL_DOCUMENT` (商品端), `RETRIEVAL_QUERY` (查询端)
- 距离: cosine / dot product / Euclidean

**multimodalembedding@001:**
- 输出维度: 1408 (可选 128/256/512)
- 文本限制: **32 tokens** (仅短文本)
- 图片: 最大 20MB，内部缩放到 512x512
- 图文在同一向量空间

**Vertex AI Vector Search:**
- 算法: TreeAH (ScaNN)，`approximateNeighborsCount=150`
- 距离: 推荐 DOT_PRODUCT_DISTANCE + UNIT_L2_NORM (官方建议优于 COSINE)
- 过滤: Token restricts (分类 AND/OR) + Numeric restricts (范围比较)
- 更新: 支持 Batch 和 Streaming 两种模式
- 扩缩容: 按 QPS 自动扩缩

---

# 4. 架构设计

## 4.1 整体架构

```
                        ┌──────────────────┐
                        │  Web / Mobile    │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Cloud CDN +     │
                        │  Load Balancer   │
                        └────────┬─────────┘
                                 │
               ┌─────────────────▼─────────────────┐
               │        API Layer (Cloud Run)       │
               │                                    │
               │  ┌────────────┐  ┌──────────────┐ │
               │  │ Query      │  │ Image Upload │ │
               │  │ Handler    │  │ Handler      │ │
               │  └─────┬──────┘  └──────┬───────┘ │
               └────────┼────────────────┼─────────┘
                        │                │
          ┌─────────────▼────────────────▼──────────────┐
          │          Retrieval Layer                      │
          │                                              │
          │  ┌──────────────────────────────────────┐   │
          │  │       Memorystore (Redis)             │   │
          │  │       Hot query cache                 │   │
          │  └──────────────┬───────────────────────┘   │
          │                 │ miss                       │
          │  ┌──────────────▼───────────────────────┐   │
          │  │    Vertex AI Vector Search            │   │
          │  │    ┌─────────────┐ ┌───────────────┐ │   │
          │  │    │ Text Index  │ │ Image Index   │ │   │
          │  │    │ (768-dim)   │ │ (1408-dim)    │ │   │
          │  │    │ dense+sparse│ │ multimodal    │ │   │
          │  │    └─────────────┘ └───────────────┘ │   │
          │  └──────────────────────────────────────┘   │
          │                                              │
          │  ┌──────────────────────────────────────┐   │
          │  │    Vertex AI Ranking API              │   │
          │  │    Cross-encoder re-ranking            │   │
          │  └──────────────────────────────────────┘   │
          │                                              │
          │  ┌──────────────────────────────────────┐   │
          │  │    Business Rules Engine              │   │
          │  │    Boost / Bury / Pin / Personalize   │   │
          │  └──────────────────────────────────────┘   │
          └──────────────────────────────────────────────┘

          ┌──────────────────────────────────────────────┐
          │        Data Layer                             │
          │  AlloyDB: 商品属性/价格/库存                   │
          │  Bigtable: 用户画像/行为历史                   │
          │  BigQuery: 搜索日志/离线评估                   │
          └──────────────────────────────────────────────┘

          ┌──────────────────────────────────────────────┐
          │        Ingestion Pipeline (Dataflow)          │
          │  新商品 → embedding 生成 → Vector Search 索引  │
          │         → 结构化数据写入 AlloyDB               │
          └──────────────────────────────────────────────┘
```

## 4.2 文本搜索管线 (Text Search Pipeline)

```
用户输入: "floral dress for winter wedding"
  │
  │ ① Query Understanding
  ▼
┌──────────────────────────────┐
│ 拼写纠正 → 查询扩展 → 意图识别 │
└──────────────┬───────────────┘
               │
  │ ② Embedding Generation
  ▼
┌──────────────────────────────┐
│ gemini-embedding-001          │
│ task_type = RETRIEVAL_QUERY   │
│ output_dimensionality = 768   │
│ → 768-dim query vector        │
└──────────────┬───────────────┘
               │
  │ ③ Parallel Retrieval (两路并行)
  ▼
┌──────────────────┐  ┌──────────────────┐
│ Dense Search     │  │ Sparse Search    │
│ (语义向量, ANN)  │  │ (关键词, BM25)   │
│ → top 100        │  │ → top 100        │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
  │ ④ Fusion
  ▼
┌──────────────────────────────┐
│ Reciprocal Rank Fusion (RRF) │
│ score = Σ 1/(k + rank)       │
│ → merged top 100              │
└──────────────┬───────────────┘
               │
  │ ⑤ Structured Filtering
  ▼
┌──────────────────────────────┐
│ Token restricts: category,    │
│   color, brand                │
│ Numeric restricts: price,     │
│   inventory > 0               │
└──────────────┬───────────────┘
               │
  │ ⑥ Re-ranking
  ▼
┌──────────────────────────────┐
│ Vertex AI Ranking API         │
│ → 精排 top 50                 │
└──────────────┬───────────────┘
               │
  │ ⑦ Business Rules + Personalization
  ▼
┌──────────────────────────────┐
│ 促销商品 boost +20%           │
│ 低库存 bury -30%              │
│ 用户偏好 re-weight            │
│ → final top 20                │
└──────────────┬───────────────┘
               │
               ▼
         返回搜索结果
```

### 为什么两路并行 + RRF？

| 查询类型 | Dense (语义) | Sparse (关键词) | 谁赢？ |
|---|---|---|---|
| "comfortable heels for standing" | 理解功能需求 | 无法匹配 | Dense |
| "Nike Air Max 90" | 相似运动鞋都返回 | 精确匹配型号 | Sparse |
| "blue dress under $50" | 理解蓝色裙子 | 匹配 "blue" + "dress" | 互补 |

**Vector Search 原生支持 dense + sparse 混合索引**，一次查询同时做两路检索，无需维护两套系统。

## 4.3 图片搜索管线 (Image Search Pipeline)

```
用户上传穿搭照片
  │
  │ ① Object Detection
  ▼
┌──────────────────────────────┐
│ Gemini Vision API             │
│ 识别图中各个单品:              │
│ → dress (bounding box 1)      │
│ → shoes (bounding box 2)      │
│ → bag   (bounding box 3)      │
└──────────────┬───────────────┘
               │
  │ ② Per-item Embedding (每个单品分别处理)
  ▼
┌──────────────────────────────┐
│ multimodalembedding@001       │
│ 每个 crop → 1408-dim vector   │
│ (图文共享向量空间)             │
└──────────────┬───────────────┘
               │
  │ ③ Vector Search (per item)
  ▼
┌──────────────────────────────┐
│ Image Index (1408-dim)        │
│ 每件单品 → top 20 相似商品    │
└──────────────┬───────────────┘
               │
  │ ④ Category Grouping
  ▼
┌──────────────────────────────┐
│ "Complete the Look" 分组展示  │
│ Similar dresses | shoes | bags│
└──────────────────────────────┘
```

### 文本 vs 图片: 两套索引 + 统一体验

| 维度 | Text Index | Image Index |
|---|---|---|
| 模型 | gemini-embedding-001 | multimodalembedding@001 |
| 维度 | 768 | 1408 |
| 商品入库 | title + description → vector | product image → vector |
| 查询入口 | 搜索框文本输入 | 图片上传 |
| 混合查询 | "like this but in black" → 图片 embedding + 文本 embedding 加权 |

**为什么分两套索引**:
- `multimodalembedding@001` 文本仅支持 32 tokens，不足以编码完整商品描述
- `gemini-embedding-001` 支持 2048 tokens 但没有图片输入能力
- 两套索引通过 **应用层融合** (RRF / weighted ensemble) 实现统一搜索体验

---

# 5. Black Friday 扩缩容策略

> 演讲必答问题: "How do you handle the 10x traffic spike?"

## 5.1 流量对比

| 指标 | 日常 | Black Friday | 倍数 |
|---|---|---|---|
| QPS (平均) | 12 | 350 | 29x |
| QPS (峰值) | 50 | 1,000 | 20x |
| 持续时间 | 全天 | 8 小时集中 | - |

## 5.2 四层防御策略

```
Layer 1: CDN + Edge (Cloud CDN)
  │ 缓存 autocomplete 建议、类目页
  │ → 削峰 ~20%
  │
Layer 2: Application Cache (Redis)
  │ 热门查询结果缓存 (TTL 5 min)
  │ 提前 2 天预热 top 10,000 查询
  │ 目标: 85% cache hit → 实际到 Vector Search 仅 150 QPS
  │
Layer 3: Auto-scaling (Vector Search + Cloud Run)
  │ Vector Search: min 5 → max 25 nodes
  │ Cloud Run: min 2 → max 50 instances
  │ 提前 1 小时 warm-up 到 15 nodes
  │
Layer 4: Graceful Degradation (熔断)
  │ 如果延迟超标: fallback 到纯 keyword + cache
  │ Circuit breaker: 10 秒内 >5% 错误 → 自动切换
  │ 用户感知: 质量略降，但服务不中断
```

## 5.3 延迟预算 (P99 < 200ms)

```
200ms 总预算分配:
├── 网络 (CDN → LB → Cloud Run)     20ms
├── 缓存查找 (Redis)                  5ms  ← cache hit 直接返回
├── Embedding 生成                    30ms
├── Vector Search (ANN)              40ms
├── Ranking API (re-rank)            30ms
├── 结构化过滤                        20ms  ← 与 Vector Search 并行
├── 业务规则 + 个性化                 15ms
├── 序列化 + 响应                     10ms
└── 缓冲                             30ms
```

---

# 6. 搜索质量评估框架

> 演讲必答问题: "How do you measure search quality?"

## 6.1 离线评估 (上线前)

| 指标 | 含义 | 目标 |
|---|---|---|
| **NDCG@10** | 前 10 结果排序质量 | > 0.65 |
| **MRR** | 首个相关结果排名倒数 | > 0.45 |
| **Recall@100** | top-100 候选包含相关商品比例 | > 0.90 |
| **Zero-result Rate** | 无结果查询占比 | < 3% |

**评估数据**: 从搜索日志采样 2000 查询，点击且购买=高相关，点击=中相关，展现未点击=不相关。

## 6.2 在线评估 (上线后)

| 指标 | 当前 | 目标 |
|---|---|---|
| 搜索点击率 CTR | 12% | **25%** |
| 零结果率 | 15% | **< 3%** |
| P99 延迟 | 2.5s | **< 200ms** |
| 搜索转化率 | baseline | **+30%** |
| Revenue per Search | baseline | **+20%** |

## 6.3 A/B 测试策略

- 灰度放量: 5% → 20% → 50% → 100%
- 最小检测效应: CTR 提升 2pp，置信度 95%
- **Black Friday 冻结实验** — 全量使用已验证的最优配置

---

# 7. 补充设计要点

## 7.1 冷启动 — 新品无行为数据怎么办

```
新品上架:
  ├── 商品描述 + 图片 → 生成 embedding → 立即可被语义搜索发现
  ├── 属性元数据 (color, material) → 结构化过滤可用
  ├── 同品牌/品类热销品 → 作为 proxy signal
  └── Exploration boost → 新品 7 天内小幅曝光加权

核心优势: 向量搜索基于内容 (content-based)，不依赖行为数据
         → 这是相对于协同过滤的天然优势
```

## 7.2 个性化 — 怎样做而不形成 "filter bubble"

```
final_score = 0.7 × relevance + 0.2 × personalization + 0.1 × business_rules

个性化信号:
  - 购买历史 embedding (历史购买商品向量的加权平均)
  - 浏览偏好 (近 30 天)
  - 价格偏好 (平均客单价)
  - 尺码/品牌偏好

防 filter bubble:
  - 个性化仅影响排序，不影响召回
  - 每页保留 "exploration slots" (第 8-10 位放非个性化推荐)
  - 提供 "See all results" 开关
```

## 7.3 商品描述质量参差 — Enrichment Pipeline

```
低质量描述 (< 20 words):
  商品标题 + 图片 + 属性 → Gemini → 生成丰富描述 (100-200 words)

Embedding 输入 = title + enriched_description + attributes
定期评估 enriched vs original 搜索质量差异
```

---

# 8. 业务价值 (ROI)

> 演讲第 5 环节: "Business & Strategic Value"，约 3 分钟。

## 8.1 营收影响测算

```
年线上营收:                    $2,000,000,000
搜索驱动营收 (预估 60%):        $1,200,000,000
搜索相关购物车放弃:             35%

改善预期:
  CTR:         12% → 22% (+83%)
  零结果率:     15% → 3%
  延迟跳出率:   40% → 10%

保守测算:
  搜索转化提升:    +15~20%
  年营收增量:      $180M ~ $240M
  首年投入:        ~$2~3M (基础设施 + 实施)
  ROI:            60~80x
```

## 8.2 月度成本预估

| 组件 | 月费用 |
|---|---|
| Vector Search (5 nodes 常态) | ~$1,800 |
| Embeddings API (1M queries/day) | ~$750 |
| AlloyDB / Cloud SQL | ~$1,500 |
| Memorystore Redis | ~$500 |
| Cloud Run | ~$800 |
| 其他 (监控/日志/存储) | ~$650 |
| **月度总计** | **~$6,000** |

**对比**: $6K/月 vs $15M+/月营收增量 = **2500x 投入产出比**

---

# 9. 实施路线图

> 演讲第 6 环节: "Implementation & Next Steps"，约 3 分钟。

```
Phase 1: Text Semantic Search MVP (第 1-4 周)
  ├── embedding 管线搭建 (gemini-embedding-001)
  ├── Vector Search 索引 (200 万 SKU)
  ├── RRF 混合检索 (dense + sparse)
  ├── 基础结构化过滤
  └── 交付物: 离线评估 NDCG > 0.55，Demo 可演示

Phase 2: Image Search (第 5-8 周)
  ├── multimodalembedding@001 集成
  ├── Gemini Vision object detection
  ├── "Shop the Look" 端到端流程
  └── 交付物: 图片搜索可用，recall > 0.70

Phase 3: Personalization + Business Rules (第 9-12 周)
  ├── 用户画像构建 (Bigtable)
  ├── Ranking API re-ranking
  ├── Business rules engine (boost/bury/pin)
  └── 交付物: A/B 测试框架上线，个性化可用

Phase 4: Production Hardening (第 13-16 周)
  ├── Black Friday 负载测试 (1000 QPS)
  ├── 缓存预热策略
  ├── 自动扩缩容验证
  ├── Graceful degradation + circuit breaker
  └── 交付物: 生产就绪，Runbook 完成
```

---

# 10. 方案选型对比 — Trade-offs

> 演讲第 4 环节: "Architectural Decisions & Trade-offs"，约 5 分钟。

## 10.1 全托管 vs 自建

| 维度 | 方案 A: Vertex AI Search for Commerce | 方案 B: 自建 (Vector Search + Embeddings) |
|---|---|---|
| 上线时间 | 2-3 个月 | 4-6 个月 |
| 自定义程度 | 中 (通过 serving controls) | 高 (完全可控) |
| 图片搜索 | 不原生支持 | 自由实现 |
| 维护成本 | 低 | 高 |
| 个性化 | 内置 (基于行为) | 自建 (更灵活) |
| 成本 | 按查询付费 | 按节点付费 |
| 面试技术深度 | 较浅 | **深 (加分)** |

**演讲策略: 推荐分阶段实施**
- Phase 1: 用自建方案做 MVP (展示技术深度) + Demo
- Phase 2-3: 评估是否迁移到 Commerce Search (由数据和效果决定)
- 原因: "Let the data decide" — 与 Google 官方迁移方法论一致

## 10.2 向量空间: 统一 vs 分离

| 维度 | 统一 (一套索引) | 分离 (两套索引，推荐) |
|---|---|---|
| 架构简单性 | 简单 | 稍复杂 |
| 文本搜索质量 | 受限 (multimodal 文本仅 32 tokens) | **高** (gemini-embedding 2048 tokens) |
| 图片搜索质量 | 高 | 高 |
| 混合查询 | 天然支持 | 需应用层融合 |
| **推荐理由** | - | **文本搜索是 80% 的场景，不能妥协质量** |

## 10.3 Pre-filter vs Post-filter

| 策略 | Pre-filter (推荐) | Post-filter |
|---|---|---|
| 流程 | 先按结构化条件缩小范围，再向量搜索 | 先向量搜索全量，再过滤 |
| 延迟 | 更低 (搜索空间小) | 更高 |
| 准确性 | 可能漏掉一些语义相关结果 | 更准确但更慢 |
| 适用场景 | 大目录 + 强过滤条件 | 小目录 + 弱过滤 |
| **推荐理由** | **200 万 SKU，"size M + under $50" 可缩小到 10 万条** |

---

# 11. Q&A 准备

## Q1: 为什么不继续用 Elasticsearch + kNN?

> ES kNN 在百万级向量上 P99 > 500ms；扩缩容需手动管理 shard；没有原生多模态支持。
> Vector Search 基于 ScaNN 算法，百万级 P99 < 50ms，自动扩缩容，原生 dense+sparse 混合搜索。
> 备选方案: AlloyDB AI (ScaNN 索引 + SQL 接口) 适合低迁移风险场景。

## Q2: Embedding 是否需要 fine-tune?

> 三步走: ① gemini-embedding-001 直接用 → ② 用搜索日志评估 → ③ 不够再 fine-tune。
> 时尚领域 "cottagecore" "coastal grandmother" 等术语可能需要 fine-tune。
> Vertex AI 提供 embedding tuning 功能。

## Q3: 个性化搜索会不会造成 filter bubble?

> 个性化只影响 re-ranking (排序)，不影响 retrieval (召回)。
> 权重: 70% 相关性 + 20% 个性化 + 10% 业务规则。
> 每页保留 exploration slots，提供关闭开关。

## Q4: 成本?

> 月度约 $6K (详见第 8 节)。对比搜索改善带来的 $15M+/月增量，投入产出比 2500x。

## Q5: 数据隐私?

> 所有数据在客户 GCP project 内处理。
> Vertex AI Embeddings API 不用客户数据训练模型。
> 搜索日志存 BigQuery，受 IAM + VPC Service Controls 保护。
> 个性化数据有 TTL，30 天无活动自动清除。

## Q6: 如果 Embedding 模型效果不够好?

> 三条路: ① 换更大维度 (768 → 1536 → 3072) ② Fine-tune ③ Two-tower 模型自训练。
> Google 官方提供 "Two-Tower Retrieval" 参考架构，用用户行为数据训练专用检索模型。

---

# 12. 演讲节奏 (30 分钟)

| 环节 | 时长 | 内容 | 对应本文章节 |
|---|---|---|---|
| Title & Context | 2 min | 自我介绍 + 场景设定 | §1 |
| Specialized Challenge | 5 min | ES 为什么不够 + Unblocker | §2 |
| **Technical Deep-Dive + Demo** | **12 min** | 架构 + 管线 + Live Demo | §3, §4 |
| Architectural Decisions | 5 min | 选型对比 + Trade-offs | §10 |
| Business Value | 3 min | ROI 测算 | §8 |
| Implementation | 3 min | 4 阶段路线图 | §9 |

### Demo 设计 (5-7 分钟，包含在 Technical Deep-Dive 内)

```
准备: Colab / Jupyter Notebook，预先跑通

流程:
  1. 展示 100 件时尚商品数据 (JSON)
  2. 调用 gemini-embedding-001 生成 embedding
  3. 创建 Vector Search index
  4. 关键词 vs 语义搜索并排对比
     → "floral dress for winter wedding" 的两种结果
  5. 图片上传搜索 demo
  6. 混合搜索: "summer dress" + filter "under $50"

Backup: 录屏备份 (防 API 超时)
```

---

## References

### 官方文档
- [Vector Search Overview](https://cloud.google.com/vertex-ai/docs/vector-search/overview)
- [Vector Search Index Config](https://cloud.google.com/vertex-ai/docs/vector-search/configuring-indexes)
- [Text Embeddings (gemini-embedding-001)](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)
- [Multimodal Embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings)
- [Vertex AI Search for Commerce](https://cloud.google.com/retail/docs/overview)
- [AlloyDB AI Embeddings](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings)

### 官方 Notebooks
- Vector Search Quickstart
- Hybrid Search Tutorial (Semantic + Keyword)
- Two-Tower Retrieval for Large-Scale Candidate Generation
- Multimodal Search with Rank-Biased Reciprocal Rank Ensemble

### 客户案例 (官方引用)
- eBay: Vector Search 推荐系统
- Mercado Libre: 市场平台搜索
- Bloomreach: 电商搜索性能优化
