# Airwallex 沙箱验证 Spike 报告

> 验证日期：2026-10-20  
> 目标：确认沙箱能否支撑 Agentic Payment Demo 三条轨道  
> 方法：直接调 REST API，凭证来自 AWS Secrets Manager

---

## 0. 环境与余额

| 字段 | 值 |
|---|---|
| Base URL | `https://api.sandbox.airwallex.com` |
| 鉴权端点 | `POST /api/v1/authentication/login` |
| Token 有效期 | 30 分钟 |
| SGD available | ~10,000,000 |
| USD available | ~10,000,000 |
| EUR available | ~10,000,000 |
| VND available | ~256,000（少，但换汇即补） |
| HKD available | 0（靠 USD 自动换汇支付） |

余额充足，无需模拟入账。所有 API 调用均带 `request_id`（UUID）作幂等键。

---

## 1. 转账状态模拟（Transfer Simulation）

### 1.1 现有转账状态

两笔已创建的 SG FAST 转账（SGD 15 / SGD 8）均处于 `PROCESSING`，无法进一步推进。

| 转账 ID | 状态 | 原因 |
|---|---|---|
| `a1d7943c-08a9-...` | `PROCESSING` | 等待模拟推进 |
| `8bcc69d0-d5bb-...` | `PROCESSING` | 等待模拟推进 |

### 1.2 正确路径（已确认）

```
POST /api/v1/simulations/transfers/{id}/succeed
POST /api/v1/simulations/transfers/{id}/fail
```

路径本身正确（返回 401，而非 404），但需要 **Admin 级别 API 密钥**。

### 1.3 根本原因：权限不足

```json
HTTP 401 {"code":"unauthorized","message":"Admin permissions required"}
```

标准 API Key（`client_id`/`api_key`）不具备 admin 权限。以下端点均受此限制：
- `POST /api/v1/simulations/transfers/{id}/succeed`
- `POST /api/v1/simulations/transfers/{id}/fail`
- `GET  /api/v1/beneficiaries/bank_details/form_schema`
- `GET  /api/v1/issuing/programs`
- `GET  /api/v1/issuing/simulations/transactions`
- `GET  /api/v1/fx/quotes` 和 `/fx/rates`

### 1.4 用户控制台操作（必须）

> **在沙箱演示前，操作员须在 Airwallex 沙箱控制台完成以下步骤：**
> 
> 1. 登录 sandbox console → Account → Developer → API Keys
> 2. 点击「Create API Key」，Role 选择 **Admin**
> 3. 将新的 `client_id` 和 `api_key` 更新到 AWS Secrets Manager `apdemo/airwallex-sandbox`
> 4. 重新运行 `01_auth_and_balances.py` 刷新 token
> 5. 重新运行转账创建脚本，再用 `/simulations/transfers/{id}/succeed` 推进

### ⚠️ 结论：路径已知、流程已验证；仅缺 Admin Key，需控制台操作。

---

## 2. 本地实时付款

### 2.1 新加坡 FAST（SGD）

✅ **已完全验证（上轮完成）**

- 收款方 ID: `9591eb5d-7424-4290-8869-a03a17b42666`
- 转账端点: `POST /api/v1/transfers/create`
- 字段: `transfer_method: "LOCAL"`, `local_clearing_system: "FAST"`, `source_currency: "SGD"`
- 转账状态: `PROCESSING`（需 Admin Key 推进到 SETTLED）

### 2.2 越南 NAPAS（VND）

✅ **本轮已修复并验证**

**失败根因分析：**
1. 之前用 `payment_methods` — 字段名应为 `transfer_methods`（顶层字段）
2. 直接指定 `local_clearing_system: "NAPAS"` 报 `beneficiary_type_unsupported`（此沙箱账号未开通 NAPAS 直连）
3. VN 本地付款必须通过 `swift_code` 标识收款行，系统内部路由到 NAPAS

**正确创建请求：**

```json
POST /api/v1/beneficiaries/create
{
  "request_id": "<uuid>",
  "transfer_methods": ["LOCAL"],
  "payer_entity_type": "COMPANY",
  "beneficiary": {
    "entity_type": "PERSONAL",
    "type": "BANK_ACCOUNT",
    "first_name": "Van An",
    "last_name": "Nguyen",
    "address": {
      "country_code": "VN",
      "city": "Ho Chi Minh City",
      "street_address": "123 Le Loi Street",
      "postcode": "700000",
      "state": "SG"
    },
    "bank_details": {
      "bank_country_code": "VN",
      "account_currency": "VND",
      "account_name": "Nguyen Van An",
      "account_number": "1234567890123",
      "swift_code": "BFTVVNVX"
    }
  }
}
```

响应关键字段：`id: "459bd865-3b05-4f34-838e-cb74ca2de473"`, `bank_name: "JOINT STOCK COMMERCIAL BANK FOR FOREIGN TRADE OF VIETNAM (VIETCOM BANK)"`

**转账（USD→VND 7,000,000）：**

```json
POST /api/v1/transfers/create
{
  "beneficiary_id": "459bd865-3b05-4f34-838e-cb74ca2de473",
  "transfer_currency": "VND",
  "transfer_amount": "7000000",
  "source_currency": "USD",
  "transfer_method": "LOCAL",
  "reason": "goods_purchased"
}
```

响应：`id: "a3160863-923a-4814-8880-7042481c2fe2"` status=`SCHEDULED` → `PROCESSING`  
`amount_payer_pays: 273.31 USD`, `amount_beneficiary_receives: 7000000 VND`

**坑：**
- `reason` 字段对 VN 有枚举约束，不能用 `paying_suppliers`，必须用 `goods_purchased` / `business_expenses` 等
- VN 地址必须带 `postcode` 和 `state`
- VND 账号无需 `account_routing_type1`，但必须有 `swift_code`（与 SG/HK 不同）

### 2.3 香港 FPS（HKD）

✅ **本轮已修复并验证**

**失败根因：** 缺少 `account_routing_type1` 和 `account_routing_value1`（FPS 专有字段）

**正确 bank_details：**

```json
{
  "bank_country_code": "HK",
  "account_currency": "HKD",
  "account_name": "Chan Tai Man",
  "account_number": "004123456789",
  "account_routing_type1": "bank_code",
  "account_routing_value1": "004",
  "local_clearing_system": "FPS"
}
```

routing_type1 可选值：`bank_code` | `phone_number` | `email_address` | `fps_identifier` | `personal_id_number`

收款方 ID: `7f54195d-c524-40ba-bcfd-27714d6c6115`  
转账 ID: `9dc20b7d-e5dc-4d17-b27b-83c20221111d` status=`PROCESSING`  
`amount_payer_pays: 127.94 USD` → `1000 HKD`（自动换汇）

---

## 3. 换汇（Transactional FX）

✅ **已完全验证**

**关键发现：** 需要 `x-api-version: 2023-12-01` 请求头，否则返回 `incorrect_version` 错误。更新的版本（如 2024-06-30）反而不支持 Conversions 端点。

| 换汇对 | 卖出 | 买入 | client_rate | conversion_id | status |
|---|---|---|---|---|---|
| SGD→USD | 100 SGD | 77.88 USD | 1.2840 | `7313e7f1-...` | SETTLED |
| USD→VND | 300 USD | 7,683,448 VND | 25,611 | `256f0cc4-...` | SETTLED |
| SGD→EUR | 1000 SGD | 694.07 EUR | 1.4408 | `e5234b83-...` | SETTLED |
| SGD→HKD | 1635.85 SGD | 10,000 HKD | 6.113 | `e29be719-...` | SETTLED |

**沙箱行为：** 状态直接跳转 `SETTLED`（无需等待或模拟）。

**正确请求：**

```json
POST /api/v1/conversions/create
Headers: x-api-version: 2023-12-01
{
  "sell_currency": "USD",
  "buy_currency": "VND",
  "sell_amount": 300,
  "buy_or_sell": "sell",
  "reason": "paying_suppliers",
  "term_agreement": true,
  "request_id": "<uuid>"
}
```

**坑：**
- `POST /api/v1/fx/quotes` 和 `GET /api/v1/fx/rates` 均返回 401 Admin Required（不能直接用于报价展示）
- 替代方案：`conversion` 响应本身包含 `client_rate`、`mid_rate`、`rate_details`，可直接展示汇率

---

## 4. 发卡（Issuing / Virtual Card）

### 4.1 Issuing 配置

✅ Issuing 模块已激活，配置如下：
- `primary_currency: SGD`
- `enable_auto_conversion: true`
- 单笔限额 (USD): default 10,000, max 50,000

### 4.2 DELEGATE 持卡人

✅ **已成功创建**

```json
POST /api/v1/issuing/cardholders/create
{
  "request_id": "<uuid>",
  "type": "DELEGATE",
  "first_name": "Demo",
  "last_name": "User",
  "email": "demo@example.com",
  "mobile": "+6591234567",
  "billing_address": {
    "street_address": "1 Marina Blvd",
    "city": "Singapore",
    "postcode": "018989",
    "country_code": "SG"
  }
}
```

响应：`cardholder_id: "956a9fd9-3df6-4bd1-8db6-9dd08d633a06"` status=`READY`

### 4.3 INDIVIDUAL 持卡人

⚠️ **部分验证 — 已定位 schema 问题**

- `type: "INDIVIDUAL"` 时，API 要求 `individual.name` 字段
- 提供 `individual: {"name": "Demo Individual"}` 报 `BAD_REQUEST: Invalid request content`
- 提供 `individual: {"first_name": "...", "last_name": "..."}` 报 `individual.name is mandatory`
- 疑似 API 文档中的 `individual.name` 为一个特殊对象结构（`{first: ..., last: ...}` 或 `{display: ...}`），文档未公开展示

**推荐做法：** Demo 使用 `DELEGATE` 类型，等待官方文档或沙箱技术支持确认 `individual.name` 的精确格式。

### 4.4 虚拟卡创建

❌ **被阻：card program 未知**

- `POST /api/v1/issuing/cards/create` 报 `program is mandatory`
- `GET /api/v1/issuing/programs` 报 `401 Admin permissions required`

**用户控制台操作（必须）：**

> 1. 登录 sandbox console → Issuing → Programs（或 Settings）
> 2. 找到或创建一个 card program，复制其 program ID（UUID 格式）
> 3. 在脚本中传入：`"program": "<program_id>"`
> 4. 若 programs 下为空，联系 Airwallex 沙箱技术支持开通 issuing program

一旦获得 program ID，完整卡创建请求：

```json
POST /api/v1/issuing/cards/create
{
  "request_id": "<uuid>",
  "cardholder_id": "956a9fd9-3df6-4bd1-8db6-9dd08d633a06",
  "program": "<program_id>",
  "currency": "USD",
  "nick_name": "Demo Agent Card",
  "form_factor": "VIRTUAL",
  "spending_controls": {
    "spending_limits": [{"amount": 500, "interval": "PER_AUTHORIZATION"}],
    "allowed_merchant_category_codes": ["5411", "5812", "4111"],
    "transaction_count_limit": {"count": 10, "interval": "MONTHLY"}
  },
  "valid_to": "2026-12-31"
}
```

### 4.5 卡模拟（授权 / 超额被拒）

⚠️ **路径已知，待 Admin Key**

```
POST /api/v1/issuing/simulations/transactions        → 模拟授权（成功）
POST /api/v1/issuing/simulations/transactions/{id}/capture → 模拟扣款
```

通过 `spending_controls` 设置的限额，沙箱中会自动触发拒绝逻辑（无需模拟）。

---

## 5. 余额与模拟入账

无需模拟入账。沙箱初始余额：
- SGD 9,999,962（够用）
- USD 9,999,982（够用）
- EUR 9,999,990（够用）
- VND 256,110（少，但 USD→VND 换汇即补）

---

## 6. Developer MCP

### 6.1 端点

| 服务 | URL | 认证 | 用途 |
|---|---|---|---|
| Developer MCP | `https://mcp.sandbox.airwallex.com/developer` | OAuth（Airwallex 账号） | 沙箱工具 + 文档 |
| Docs MCP | `https://mcp.sandbox.airwallex.com/docs` | 无 | 只读文档 |

### 6.2 Claude Code 接入

```shell
# Developer MCP（沙箱操作）
claude mcp add-json airwallex-dev '{ "type": "http", "url": "https://mcp.sandbox.airwallex.com/developer" }'

# Docs MCP（文档查询，无需 OAuth）
claude mcp add-json airwallex-docs '{ "type": "http", "url": "https://mcp.sandbox.airwallex.com/docs" }'
```

### 6.3 能否接入 AgentCore Gateway

✅ **可以**。Developer MCP 是 HTTP 远程 MCP（StreamableHTTP），使用 OAuth 认证（标准 OIDC/OAuth2 流），符合 AgentCore Gateway 对远程 MCP 的要求。接入流程：

1. 在 AgentCore Gateway 中配置 target 指向 `https://mcp.sandbox.airwallex.com/developer`
2. 配置 OAuth credential（需要 Airwallex 沙箱账号的 OAuth client_id/client_secret）

**已提供工具：** 文档搜索、沙箱资源创建、模拟事件触发、sandbox 端点读取

**限制：** 仅限沙箱；生产账号操作使用 Airwallex AgentOS MCP（独立产品）

---

## 7. Webhooks

✅ **沙箱支持 Webhooks**

```
GET /api/v1/webhooks → 200 {"items": []}
```

- 在 sandbox console Account → Developer → Webhooks 配置 URL
- 支持查看和重触发历史 webhook 事件
- 建议将沙箱 IP 加白名单

---

## 8. 坑汇总

| 坑 | 说明 | 修复方式 |
|---|---|---|
| `payment_methods` vs `transfer_methods` | 创建收款方时字段名不同，导致 VN/HK 一直返回 null | 改用 `transfer_methods` |
| VN 不能用 `local_clearing_system: "NAPAS"` | 沙箱账号未开通 NAPAS 直连，报 `beneficiary_type_unsupported` | 改用 `swift_code` 指定银行，系统内部路由 |
| HK FPS 需要 `account_routing_type1/value1` | 文档里有但 API 错误信息没明确提 | 加 `account_routing_type1: "bank_code"`, `value1: "004"` |
| Conversion 需要 `x-api-version: 2023-12-01` | 最新版本不支持，要用旧版本 | 固定加请求头 |
| VN transfer 的 reason 枚举 | "paying_suppliers" 对 VN 不合法 | 改用 "goods_purchased" |
| 模拟端点需要 Admin Key | `/simulations/transfers/{id}/succeed` 等全部 401 | 控制台创建 Admin API Key |
| 卡创建需要 `program` | `/issuing/programs` 也需要 admin，无法 API 获取 | 控制台获取 program ID |
| INDIVIDUAL 持卡人 `individual.name` 格式不明 | 错误信息矛盾 | 使用 DELEGATE 类型规避 |
| Sandbox 换汇直接 SETTLED | 无法演示「锁汇中」状态 | Demo 展示 rate 信息即可 |

---

## 9. 对 Demo 设计的影响

### 需要会前预置（不能现场实时跑）

| 项目 | 原因 | 预估耗时 |
|---|---|---|
| 创建 Admin API Key | 控制台操作，需人工 | 5 分钟 |
| 获取 Issuing Program ID | 控制台操作 | 5 分钟 |
| 创建 SG/VN/HK 收款方 | 已完成，ID 已保存 | 0（已就绪） |
| 创建虚拟卡（有 program ID 后） | 约 5 秒 API 调用 | 可现场 |
| 预热 token | 调 auth 接口 | 2 秒 |
| VN 转账（有 Admin Key 后验证仿真） | 推进状态需模拟 | 现场可演 |

### 可以现场实时跑（建议 live demo）

| 动作 | 端点 | 预估响应时间 |
|---|---|---|
| 鉴权 | POST /authentication/login | ~0.5s |
| 查余额 | GET /balances/current | ~0.3s |
| 创建收款方 | POST /beneficiaries/create | ~1s |
| 创建转账（SG FAST） | POST /transfers/create | ~1s，状态 SCHEDULED→PROCESSING |
| 推进转账到 SETTLED | POST /simulations/transfers/{id}/succeed | ~1s（有 Admin Key） |
| FX 报价（通过 conversion 获取） | POST /conversions/create | ~0.5s，直接 SETTLED |
| 创建持卡人（DELEGATE） | POST /issuing/cardholders/create | ~1s |
| 创建虚拟卡 | POST /issuing/cards/create | ~1s（需 program ID） |
| 模拟卡授权 | POST /issuing/simulations/transactions | ~1s（需 Admin Key） |

### 各轨道评估

| 轨道 | 可演状态 | 限制 |
|---|---|---|
| 本地实时付款（SG FAST / VN NAPAS / HK FPS） | ✅ 端到端可跑 | 需 Admin Key 演示 SETTLED 状态 |
| 换汇（FX） | ✅ 完整可跑 | 状态直接 SETTLED，无「锁汇中」 |
| Agent 虚拟卡 | ⚠️ 部分可跑 | 需 program ID（控制台获取）和 Admin Key（模拟授权） |

---

## 10. 已创建资源汇总

| 类型 | ID | 描述 |
|---|---|---|
| 收款方（SG FAST） | `9591eb5d-7424-4290-8869-a03a17b42666` | DBS SGD LOCAL |
| 收款方（VN NAPAS） | `459bd865-3b05-4f34-838e-cb74ca2de473` | Vietcombank VND |
| 收款方（HK FPS） | `7f54195d-c524-40ba-bcfd-27714d6c6115` | HSBC HKD FPS |
| 转账（SG FAST 成功测试） | `a1d7943c-08a9-4549-891d-1a0d37acc0b2` | 15 SGD，PROCESSING |
| 转账（SG FAST 失败测试） | `8bcc69d0-d5bb-4e6a-af85-9d1f813e5d8c` | 8 SGD，PROCESSING |
| 转账（HK FPS） | `9dc20b7d-e5dc-4d17-b27b-83c20221111d` | 1000 HKD，PROCESSING |
| 转账（VN NAPAS） | `a3160863-923a-4814-8880-7042481c2fe2` | 7M VND，PROCESSING |
| 换汇（SGD→USD） | `7313e7f1-ed15-4b33-9e8f-d5fc3df6ca44` | SETTLED |
| 换汇（USD→VND） | `256f0cc4-439e-4258-b1a5-7012f567071e` | SETTLED |
| 换汇（SGD→EUR） | `e5234b83-f25a-4fa4-84dc-88589513b785` | SETTLED |
| 换汇（SGD→HKD） | `e29be719-b9b8-4583-93d4-982a42cf7072` | SETTLED |
| 持卡人（DELEGATE） | `956a9fd9-3df6-4bd1-8db6-9dd08d633a06` | status=READY |

---

## 补核（主线程，2026-10-20）：不需要 Admin Key，三条轨道全部跑通 ✅

背景：Airwallex 正在下线 Admin API Key（2026-08-24 之后注册的新账号不再发放）。本节确认**用现有的普通 API key 即可完成全部演示动作**，之前报告中"需 Admin Key"的两项结论作废。

| 动作 | 接口 | 结果 |
|---|---|---|
| 推进转账状态（成功路径） | `POST /api/v1/simulation/transfers/{id}/transition`，body `{"next_status":"SENT"}` → `{"next_status":"PAID"}` | ✅ SG 转账 `a1d7…0b2` PROCESSING → SENT → **PAID** |
| 越南本地付款到账 | 同上 | ✅ VND 7,000,000（LOCAL）`a316…fe2` PROCESSING → **PAID** |
| 香港 FPS 到账 | 同上 | ✅ HKD 1,000（LOCAL）`9dc2…11d` PROCESSING → **PAID** |
| 失败路径 | 同上，`SENT` → `FAILED` | ✅ 返回 200，最终状态显示为 **CANCELLED**（直接 PROCESSING → FAILED 返回 500，需先到 SENT） |
| 发虚拟卡 | `POST /api/v1/issuing/cards/create`，`program: {"purpose":"COMMERCIAL","type":"DEBIT"}`（**不需要 program ID**），`form_factor: VIRTUAL`，`authorization_controls.transaction_limits` = USD 50 / PER_TRANSACTION，持卡人 DELEGATE | ✅ 卡 `2f27…a8c`，**VISA**，状态 **ACTIVE**，卡号尾号 5787（脱敏） |
| 模拟授权：额度内 | `POST /api/v1/simulation/issuing/create`，`transaction_amount: 49`，USD，MCC 5734（软件） | ✅ `process_result: APPROVED`，交易列表中出现 -49 USD，商户类别 Software |
| 模拟授权：超额 | 同上，`transaction_amount: 80` | ✅ `process_result: DECLINED` |

说明与坑：
- `GET /api/v1/issuing/programs`、`/card_programs` 与 `/api/v1/simulations/transfers/{id}/succeed` 仍返回 "Admin permissions required"，但**演示不需要它们**：状态推进改用 `/simulation/transfers/{id}/transition`，发卡用 `program.purpose/type` 而非 program ID。
- `GET /api/v1/issuing/config` 可读到发卡侧单笔限额默认值与上限（USD 默认 10,000、上限 50,000），可在控制台展示"发卡侧硬限额"。
- 结论：**Airwallex 场景的本地实时付款（VN / HK / SG）、换汇、agent 虚拟卡（含授权通过 / 超额拒绝）全部可在现场实时演示，每步约 1 秒**；客户新账号同样可复现（不依赖 Admin Key）。
