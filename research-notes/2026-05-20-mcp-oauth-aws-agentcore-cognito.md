# MCP 授权落地实战：从 OAuth 2.1 RFC 规范到 AWS AgentCore + Cognito 的最小可行适配器

- **Date:** 2026-05-20
- **Tags:** MCP, OAuth 2.1, AgentCore, Cognito, API Gateway, Lambda, RFC 9728, RFC 8414, RFC 7591, PKCE, AWS

## Context

如果你尝试过把一个 MCP Server 托管到 AWS，让 Claude Code、Cursor 这类客户端通过 `claude mcp add <https-url>` 一键接入，大概率会撞到一堵墙：**Cognito 作为 OIDC 身份提供商，本身没有为 MCP 授权规范设计。**

这堵墙到底是什么？为什么撞？怎么绕？Kane Zhu 在他的 MCP 系列里用六篇连贯的文章把这件事讲透了。本文把其中最关键的几篇串起来读：

- **规范层**（2025-11）[Technical Deconstruction of MCP Authorization](https://kane.mx/posts/2025/mcp-authorization-oauth-rfc-deep-dive/) — 从 IETF RFC 栈出发，论证 MCP 是 OAuth 2.1 的 domain-specific profile，给出 IdaaS 兼容性矩阵
- **AS 实现层 A**（2025-11）[Implementing MCP OAuth 2.1 with Keycloak on AWS](https://kane.mx/posts/2025/deploy-keycloak-aws-mcp-oauth/) — Keycloak + ECS Fargate 完整方案，audience mapper 变通解决 RFC 8707 历史包袱
- **AS 实现层 B**（2026-05）[MCP OAuth on AgentCore Gateway + Cognito via APIGW Façade](https://kane.mx/posts/2026/agentcore-gateway-cognito-mcp-oauth/) — 用 APIGW + Lambda 适配 Cognito 缺失的四个 RFC 表面
- **客户端层**（2025-09）[Leveraging MCP Client's OAuthClientProvider for Seamless AWS AgentCore Authentication](https://kane.mx/posts/2025/use-mcp-client-oauthclientprovider-invoke-mcp-hosted-on-aws-agentcore/) — MCP SDK 自带 OAuth provider 在 AgentCore 上的几个 workaround
- **协议演进**（2025-12）[MCP OAuth Evolution: SEP-991 Simplifies Client Registration](https://kane.mx/posts/2025/mcp-oauth-sep-991-simplified-registration/) — Client ID Metadata Documents 如何颠覆 DCR 的角色

这篇博客从「为什么」走到「怎么做」再到「未来怎么变」，重点是把**规范、AS 侧、客户端侧、协议演进**这四个视角拼成一张完整的图。

## Main Content

### 一、MCP 授权不是 OAuth 2.0，是 OAuth 2.1

很多人第一次读 MCP 授权规范会觉得「不就是 OAuth 吗，加个 Bearer token 不就行了」。这种直觉是错的。

MCP 的授权模型刻意拒绝了 2012 年原版 OAuth 2.0（RFC 6749）的灵活性，转向**正在成为标准的 OAuth 2.1**。三大支柱：

| 支柱 | 涉及 RFC | 解决什么 |
|---|---|---|
| 强制安全性 | RFC 9700（安全 BCP）+ RFC 7636（PKCE）| 杜绝授权码拦截、CSRF |
| 令牌特异性 | RFC 8707（资源指示符）+ RFC 7519（JWT）| 杜绝 Confused Deputy 攻击、支持无状态验证 |
| 动态联邦 | RFC 7591（DCR）+ RFC 8414（AS 元数据）+ RFC 9728（受保护资源元数据）| 客户端零配置自动发现 |

OAuth 2.1 相对于 2.0 的关键变化：
- **删除** Implicit Flow 和 ROPC Flow
- **强制** PKCE（包括机密客户端）
- **强制** `redirect_uri` 精确字符串匹配
- **禁止** 在 URL query 中传 bearer token

MCP 在此基础上还加了一道：**所有 MCP 客户端必须用 RFC 9728 做 AS 发现**，即从 `/.well-known/oauth-protected-resource` 起步。

### 二、那个完整的"动态联邦"流程

理解 MCP 授权最关键的一张图，是新客户端第一次连一个完全陌生的 MCP server 时发生了什么——**它不需要预先知道任何端点**：

```
1.  Claude Code 拿到一个 MCP URL: https://model.example.com/mcp
2.  GET /.well-known/oauth-protected-resource         ← RFC 9728
    返回: { issuer, scopes_supported, resource }
3.  GET <issuer>/.well-known/oauth-authorization-server ← RFC 8414
    返回: { authorization_endpoint, token_endpoint,
           registration_endpoint, code_challenge_methods_supported }
4.  POST <registration_endpoint>                       ← RFC 7591
    返回: { client_id, client_secret }
5.  浏览器跳转 authorization_endpoint
    带: code_challenge (S256), resource, scope, state ← RFC 7636 + 8707
6.  用户在 IdP 登录 → 回调到 loopback redirect_uri
7.  POST token_endpoint                                ← RFC 6749 §4.1.3
    带: code, code_verifier
    返回: JWT (aud=resource, iss=as)
8.  POST /mcp  Authorization: Bearer <jwt>
    Resource Server 本地验证签名、aud、exp、iss → 200
```

注意第 7 步那个 `aud` 声明——这是 RFC 8707 的灵魂。它把令牌**绑定到一个特定资源**，使得为低权限 API 颁发的令牌**无法**在高权限 API 上重放。这就是 Confused Deputy 攻击的解药。

### 三、残酷的 IdaaS 兼容性现实

读完规范你会以为大厂 IdaaS 早就支持了。Kane 的兼容性矩阵给所有人浇了一盆冷水：

| 提供商 | OAuth 2.1 + PKCE | RFC 8707 资源指示符 | RFC 7591 DCR | 评价 |
|---|---|---|---|---|
| Auth0 | ✅ | ❌ 用专有 audience 参数 | ✅ | 部分兼容 |
| Okta | ✅ | ❌ 文档明确不支持 | ✅ | 部分兼容 |
| **Amazon Cognito** | ✅ | ✅ | ⚠️ 只有 admin API | 几乎，但缺 DCR |
| Microsoft Entra ID | ✅ | ❌ 用 `scope={resource}/.default` | ❌ | 不兼容 |
| Google Cloud Identity | ⚠️ PKCE 仍要 secret | ❌ | ❌ | 不合规 |
| Keycloak (OSS) | ✅ | ⚠️ 用 Audience Mapper 变通 | ✅ | 可绕 |
| **Ping Identity** | ✅ | ✅ | ✅ | **唯一完全合规** |
| Zitadel (OSS) | ✅ | ❌ | ❌ | 不兼容 |

**只有 Ping Identity 一家完全合规。** 其他人或多或少都缺一块。这意味着：**今天你想生产部署 MCP，要么换 IdP，要么写适配器。**

### 四、AWS 这条路上的具体痛点

如果你已经在 AWS，最自然的栈是 **Bedrock AgentCore Gateway + Cognito**：

- AgentCore Gateway：托管的多目标 MCP runtime，前端一个 endpoint 后端可挂 OpenAPI、Lambda、Smithy 模型，自带工具级拦截器和会话隔离
- Cognito：托管 OIDC 提供商，企业 IdP 联邦（Feishu、Google、SAML）现成可用

但当 Claude Code 试图发起 per-user OAuth 流时，**正好踩中 Cognito 缺失的四个 RFC 表面**：

| RFC | MCP 客户端要做什么 | Cognito 给不了什么 |
|---|---|---|
| RFC 9728 | 访问 `/.well-known/oauth-protected-resource` | 完全没有这个端点 |
| RFC 8414 | 访问 `/.well-known/oauth-authorization-server` | 只有 OIDC discovery，路径是 `cognito-idp.<region>.amazonaws.com/<pool-id>/.well-known/openid-configuration`，路径不对 |
| RFC 7591 | POST 到 `registration_endpoint` 注册 | 只有管理员的 `CreateUserPoolClient` API，没有公开 DCR |
| RFC 6749 §3.1.2.3 | 在**随机临时 loopback 端口**起回调监听器 | Hosted UI 只接受预注册的 callback URL，精确匹配 |

四个差距，每一个都是死结。要么你给每个用户预注册 callback（不可能），要么换 IdP（重）。

### 五、Façade 模式：500 行代码的最小可行适配器

Kane 的方案是写一个 **API Gateway + Lambda 门面**，挂在 Cognito 和 AgentCore Gateway 之前，做四件事：

```
Claude Code ─┐
Cursor ──────┼──→ Façade (APIGW + Lambda)
MCP Inspector┘        │
                      ├──→ Cognito Hosted UI（用户登录）
                      └──→ AgentCore Gateway（MCP 流量）
                              └──→ OpenAPI / Lambda / Smithy targets
```

**1. 伪造 metadata（RFC 9728 + 8414）**

把发现端点全部指向 Façade 自身，但 `issuer` 必须保留 Cognito 真实值——因为 JWT 里的 `iss` claim 不能改：

```typescript
// /.well-known/oauth-authorization-server
{
  issuer: COGNITO_ISSUER,                          // ← 不能改
  authorization_endpoint: `${base}/oauth/authorize`, // ← 重写到 Façade
  token_endpoint: `${base}/oauth/token`,
  registration_endpoint: `${base}/register`,
  code_challenge_methods_supported: ["S256"],       // ← 拒绝 plain
  ...
}
```

**关键洞察：Façade 重写的是路径，不是声明。** JWT 验证时仍然信任 Cognito 的真 issuer。

**2. 伪造 DCR（RFC 7591）**

所有客户端注册请求都返回**同一个**预配置的 Cognito app client：

```typescript
POST /register → 201 {
  client_id: USER_CLIENT_ID,        // 全局共享
  client_secret: USER_CLIENT_SECRET,
  redirect_uris: req.redirect_uris,  // 镜像回客户端要求的
  ...
}
```

这是**有意识的 trade-off**：所有 MCP 客户端共享一个 Cognito client_id，多租户隔离改用 Cognito group/scope 实现，而不是 client_id 维度。如果你的合规要求 client 级别审计隔离，这个模式不适合你。

**3. redirect_uri 代理重写（解决 loopback 难题）**

这是整个方案最巧妙的一块。Cognito 只能注册一个固定的 callback URL，而 MCP 客户端起的 loopback 端口是随机的。Façade 的做法是把客户端的 redirect_uri 用 HMAC 签名后塞进 `state`，过 Cognito 一圈再解出来：

```
1. Claude Code GET /oauth/authorize?redirect_uri=http://localhost:54321/cb&state=xyz
2. Façade:
   - 校验 redirect_uri 是 loopback（防开放重定向）
   - 校验 code_challenge 存在（强制 PKCE）
   - 生成 facadeState = HMAC-SHA256({cs: "xyz", r: "http://localhost:54321/cb"})
   - 302 to Cognito，redirect_uri 改成 ${facade}/oauth/callback，state 改成 facadeState
3. 用户 Cognito 登录 → Cognito 302 to ${facade}/oauth/callback?code=...&state=facadeState
4. Façade:
   - 验证 HMAC、loopback 二次校验
   - 302 to http://localhost:54321/cb?code=...&state=xyz
5. Claude Code POST /oauth/token
   - Façade 把 redirect_uri 换回 ${facade}/oauth/callback 满足 Cognito replay check
   - 转发给 Cognito，原样返回 token
```

**安全分析**：
- HMAC-SHA-256 + 10 分钟 TTL + 60 秒未来时钟容忍
- Loopback 双重校验（签 state 前 + 验 HMAC 后）
- 即便 HMAC key 泄露，攻击者只能跳到 localhost；PKCE 让没有 verifier 的 code 无法兑换
- state.ts 仅 50 行，只依赖 `node:crypto`，无 DynamoDB 无状态

**4. MCP 流量直通**

`/mcp` 路径带 Bearer JWT 时直接代理到 AgentCore Gateway，AgentCore 自己用 `CUSTOM_JWT` authorizer 验证：

```typescript
const gateway = new aws.bedrock.AgentcoreGateway("McpGateway", {
  protocolType: "MCP",
  authorizerType: "CUSTOM_JWT",
  authorizerConfiguration: {
    customJwtAuthorizer: {
      discoveryUrl: `${cognitoIssuer}/.well-known/openid-configuration`,
      allowedClients: [userAppClient.id, backendM2mClient.id],
    },
  },
});
```

### 六、那些不读完踩不到的坑

文档里找不到的工程细节，我列出最值得记的四个：

1. **`WWW-Authenticate` 头必须在 `/mcp` 代理路径上重写**。AgentCore Gateway 401 时返回的 `WWW-Authenticate` 指向自己的 RFC 9728 元数据；如果不改写，客户端 discovery 流就会绕过 Façade，整个适配器失效。

2. **`client_secret_basic` 必须原样转发**。Claude Code 用 HTTP Basic auth 发 token 请求，Lambda 代理时若把 `Authorization` 头吞掉，Cognito 返回 `invalid_client` (HTTP 400)，错误信息毫无指向性，调试一晚上都找不到。

3. **CORS 必须包含 `MCP-Protocol-Version`**。浏览器版 MCP Inspector 的预检会因这个 header 失败。

4. **HMAC key 应当按 stage 持久化，不要 CI 自动轮换**。轮换会让进行中的 OAuth 流全部失效，恼人但不致命；可控变更窗口里轮换更合适。

### 七、对照实现：Keycloak on AWS 是怎么解决同一道题的

把 Façade 方案放在更宽的视角看，它其实只是 MCP-on-AWS 的两条路之一。另一条是**自托管完整 IdP**——用 Keycloak。Kane 在 2025-11 的[那篇 Keycloak 实战](https://kane.mx/posts/2025/deploy-keycloak-aws-mcp-oauth/)里给出了完整 IaC，特别有意思的是：**Keycloak 同样不原生支持 RFC 8707**，它有自己的历史包袱要绕。

#### 7.1 Keycloak 的部署形态

```
ALB (HTTPS, ACM)
  └─ ECS Fargate × 2 AZ  (Keycloak 26.4.4 自构镜像，JDBC_PING 集群)
        └─ Aurora PostgreSQL Serverless v2  (PG 16.8, 0.5–2 ACU)
```

几个值得记的工程点：
- **JDBC_PING 替代 UDP multicast**：AWS VPC 不支持多播，且 Fargate 任务无静态 IP；Keycloak 通过在 PG 的 `JGROUPSPING` 表里登记成员发现集群，端口 7800 通信
- **Aurora Serverless v2 0.5 ACU 闲置态**约 $0.12/小时，比 RDS 起步便宜
- **健康检查 grace period 600 秒**——Keycloak 启动 60-120 秒，比一般 web 服务慢，初次部署时这个值给小了会被 ECS 不停重启杀掉

#### 7.2 RFC 8707 audience binding 的 Keycloak 变通

Keycloak 的 audience 实现**比 RFC 8707 还早**（2018 vs 2020），用的是专有 `audience` 参数而非标准 `resource`。MCP 客户端发送 `resource` 时它**直接忽略**，access token 里要么没有 `aud`，要么是错的。

变通方案三件套：

```hcl
# 1. 硬编码声明 mapper：把 aud 注入到 access token
resource "keycloak_openid_hardcoded_claim_protocol_mapper" "mcp_run_audience_mapper" {
  client_scope_id     = keycloak_openid_client_scope.mcp_run.id
  claim_name          = "aud"
  claim_value         = var.resource_server_uri  # MCP Server URL
  add_to_access_token = true   # 仅 access token，不进 ID token
  add_to_id_token     = false
  add_to_userinfo     = false
}

# 2. 把 mcp:run 设为 realm 默认 scope，所有 DCR 注册的客户端自动继承
default_scopes = ["profile", "email", "mcp:run", "roles", "web-origins", "acr", "basic"]

# 3. 通过 Admin REST API 把 mcp:run 加入 DCR 允许 scope 列表
#    (Terraform Provider 不支持，必须用 bash + curl)
```

**核心设计**：`aud` 仅出现在 access token（供资源服务器校验），不出现在 ID token（避免给前端误用）。Realm 默认 scope + DCR 允许列表是关键——它让所有动态注册的客户端**零配置**自动获得正确的 audience 映射。

#### 7.3 Keycloak vs Cognito 的 RFC 合规对比

| RFC | Keycloak | Cognito |
|---|---|---|
| RFC 7591 DCR | ✅ 原生（启用匿名 DCR）| ❌ 仅 admin API |
| RFC 7636 PKCE S256 | ✅ 强制 | ✅ |
| RFC 8414 AS metadata | ✅ OIDC discovery | ⚠️ 只有 OIDC，路径不对 |
| RFC 8707 资源指示符 | ⚠️ 用 audience mapper 变通 | ✅ 原生 |
| RFC 9728 受保护资源 metadata | ⚠️ 由 MCP Server 端实现 | ❌ 完全没有 |

Keycloak 缺一项，Cognito 缺三项——**这就是为什么 Keycloak 不需要 Façade，Cognito 必须套 Façade**。

#### 7.4 Keycloak 路线的"两阶段部署"陷阱

Keycloak 的 Terraform Provider 不能管 Client Registration Policies，所以必须分两步：

- **Phase 1（Terraform）**：Realm、安全策略、Client Scope、Audience Mapper、默认 scope
- **Phase 2（Bash + Admin REST API）**：
  - `fix-allowed-scopes.sh` —— 把 `mcp:run` 加进 DCR 允许列表
  - `disable-trusted-hosts.sh` —— 删 Trusted Hosts 策略，否则 `cursor://`、`vscode://`、`claude://` 这些自定义 scheme 会被拒
  - `enable-dcr.sh` —— 验证 DCR 真的能用

这是 IaC 落地的真实痛点——**没法把整个 IdP 配置塞进一份 Terraform**，CI/CD 必须双轨。

### 八、方案对比：Façade vs Keycloak

| 维度 | Keycloak（自管完整 IdP）| Cognito + AgentCore + Façade |
|---|---|---|
| RFC 9728/8414 metadata | 原生 | Façade 提供 |
| RFC 7591 DCR | 原生 | Façade 伪造（共享 client）|
| RFC 8707 audience 绑定 | 通过 mapper 变通 | 继承 Cognito 的 client_id-as-audience |
| PKCE S256 | 原生 | Façade 强制 |
| loopback redirect | 原生 | Façade HMAC state 重写 |
| 企业 IdP 联邦 | 原生丰富 | Cognito 原生（Feishu/SAML/OIDC）|
| 运维足迹 | ECS Fargate + Aurora Serverless | 两个 Lambda + HTTP API |
| 成本 | 容器按秒计算 | Lambda 按调用 |
| 多租户隔离 | client 级 | scope/group 级 |

**选型建议**：
- 已经在 AWS、Cognito 是既有资产、不需要 client 级合规隔离 → **Façade**
- 需要完整 IdaaS 能力（细粒度策略、丰富 social login、SCIM）→ **Keycloak**
- 不在 AWS 或愿意换栈 → **Ping Identity** 是唯一开箱合规的

### 九、客户端侧：MCP SDK 的 OAuthClientProvider 怎么用

讲完 AS 侧（Façade / Keycloak），还有半个故事在客户端侧。Kane 在 2025-09 那篇[OAuthClientProvider 实战](https://kane.mx/posts/2025/use-mcp-client-oauthclientprovider-invoke-mcp-hosted-on-aws-agentcore/)里揭示了一个反直觉的事：**MCP SDK 自带的 OAuthClientProvider 已经做了大部分活，但在 AgentCore 上你仍然要打三个补丁。**

#### 9.1 SDK 自带能力

MCP SDK 的 `OAuthClientProvider` 是面向客户端开发者的封装：
- **自动模式检测**：根据是否提供 `client_secret` 自动切换 M2M 与 interactive
- **令牌存储与刷新**：内置 `token_storage` 抽象，自动用 `refresh_token` 续期；M2M 模式自动重跑 `client_credentials`
- **Discovery 联动**：`authorization_endpoint` / `token_endpoint` 留空，运行时从 OIDC discovery 拉

```python
is_m2m_mode = bool(config.get("client_secret"))
# True → grant_type=client_credentials，无浏览器
# False → authorization_code + PKCE，浏览器跳转
```

**Scope 选择策略（针对 Cognito）**：
- 交互模式：`openid email aws.cognito.signin.user.admin`
- M2M 模式：自定义资源 server scope，如 `mcp-server/read mcp-server/write`

#### 9.2 AgentCore 上必须打的三个补丁

**Patch 1：401 → 403 的兼容**

AgentCore 在未授权时返回 **403 而不是标准 401**，SDK 默认只对 401 触发 OAuth 流。必须改：

```python
if response.status_code in (401, 403):
    # trigger OAuth flow
```

这个细节 spec 里没有，标准 MCP Server 会按 RFC 6750 返 401，**仅限 AgentCore 这一种 hosting**。

**Patch 2：跨域 metadata 手动注入**

MCP server 在 AgentCore 域，AS 在 Cognito 域。SDK 期待从 MCP server 的 RFC 9728 metadata 里发现 AS，但跨域元数据匹配很容易出错，所以作者直接手动注入：

```python
oauth_auth.context.protected_resource_metadata = protected_metadata
oauth_auth.context.auth_server_url = oauth_server_url
```

**这正是 Façade 方案要解决的根因**——Façade 把 RFC 9728/8414 端点放在同一域，客户端就不用打这种补丁。

**Patch 3：跳过 DCR**

Cognito 没有 DCR，所以**不能让 SDK 走默认的 `/register` 调用**。预先构造 `OAuthClientInformationFull` 并 `token_storage.set_client_info(client_info)` 注入：

```python
client_info = OAuthClientInformationFull(
    client_id=COGNITO_CLIENT_ID,
    client_secret=COGNITO_CLIENT_SECRET,  # 仅 M2M 用
    redirect_uris=["http://localhost:8080/callback"],
    grant_types=["authorization_code", "refresh_token"],
    ...
)
token_storage.set_client_info(client_info)
```

注意这是**客户端侧**绕过 DCR，对比 Façade 在 AS 侧伪造 `/register`——同一个问题的两层不同答案。

#### 9.3 配置坑：M2M 模式必须先开 client_credentials

Cognito 用户池客户端默认不允许 client_credentials grant，要 CLI 开：

```bash
aws cognito-idp update-user-pool-client \
  --user-pool-id <pool-id> \
  --client-id <client-id> \
  --allowed-o-auth-flows "client_credentials" \
  --generate-secret
```

少了这一步会得到一个语义不明的 `invalid_grant`，调试很痛。

### 十、协议演进：SEP-991 / Client ID Metadata Documents 颠覆 DCR

最后看最前沿的一块：**MCP 规范 2025-11-25 版正式纳入了 SEP-991**，这可能让 Façade 现在最复杂的"伪造 DCR"那一段彻底过时。Kane 在 2025-12 那篇[SEP-991 解读](https://kane.mx/posts/2025/mcp-oauth-sep-991-simplified-registration/)里说得很直白：**范式从「server registers client」转为「client proves identity」。**

#### 10.1 核心机制

`client_id` **不再是 AS 生成的字符串**，而是**客户端自托管的 HTTPS URL**：

```
client_id = https://my-mcp-client.com/.well-known/oauth-client.json
```

URL 指向客户端自己托管的元数据 JSON：

```json
{
  "client_id": "https://my-mcp-client.com/.well-known/oauth-client.json",
  "client_name": "My MCP Client",
  "redirect_uris": ["http://localhost:8080/cb"],
  "token_endpoint_auth_method": "none"
}
```

AS 在收到 `/authorize` 请求时**按需 GET 这个 URL**，校验：
1. 文档里的 `client_id` 与 URL 完全一致
2. `redirect_uri` 在 `redirect_uris` 列表里
3. 文档由可信 HTTPS 域名签发（基于域名所有权的隐式信任）

#### 10.2 流程对比：DCR vs CIMD

```
DCR (legacy):
  1. POST /register {redirect_uris, ...}
  2. 收到 server-generated client_id
  3. GET /authorize?client_id=<id>&...

CIMD (SEP-991):
  无注册步骤
  1. GET /authorize?client_id=https://client.example/.well-known/oauth-client.json&...
  2. AS 拉取该 URL 校验
  3. 完成授权
```

身份验证从「**注册时**」变成「**fetch-time**」，依据是 HTTPS 域名所有权——和 Web 上一切其他东西的信任根一样。

#### 10.3 优先级矩阵（来自 MCP 规范）

```
1. Pre-registered credentials       (已知客户端-服务器关系)
2. Client ID Metadata Documents     (AS 声明 client_id_metadata_document_supported=true)
3. Dynamic Client Registration      (RFC 7591 作为 fallback)
4. 用户手工录入                       (最终兜底)
```

AS 必须在 RFC 8414 metadata 里显式声明：

```json
{ "client_id_metadata_document_supported": true }
```

#### 10.4 SDK 现状：分裂的世界

| SDK | 语言 | CIMD 支持 |
|---|---|---|
| typescript-sdk | TypeScript | ✅ Full + capability detection |
| python-sdk | Python | ✅ Full + graceful fallback to DCR |
| rust-sdk | Rust | ❌ 仅 OAuth 2.1 + DCR |
| go-sdk | Go | ❌ 仅 RFC 8414 metadata |
| kotlin-sdk | Kotlin | ❓ 文档缺失 |
| csharp-sdk | C# | ❌ 无 OAuth |

#### 10.5 对 Façade 方案的影响

回到主线：**SEP-991 落地后，Façade 还需要伪造 DCR 吗？**

短期答案：**仍然需要**，因为：
1. AS 侧 Cognito 不支持 CIMD（不会主动 fetch 客户端 metadata URL）
2. 大部分 MCP 客户端 SDK（Rust/Go/Kotlin）还没实现 CIMD
3. RFC 9728 / 8414 / loopback 这三块差距 SEP-991 不解决

中期答案：**Façade 可以"瘦身"**——只保留 metadata 提供 + redirect 重写，DCR 那一段可以删掉，让 AS 接受 CIMD 形式的 `client_id`。

长期答案：**如果 Cognito 原生加上 CIMD + RFC 9728，Façade 就完全可以下线了。**

### 十一、再退一步看：MCP 授权规范的现实张力

把六篇文章合起来读，揭示出的其实是一个更宏观的三层张力：

**第一层：规范层 vs IdaaS 现实**

MCP 授权规范用 RFC 栈定义了一个理想中的「动态联邦」世界——客户端零配置接入任意服务。但现实是绝大多数 IdaaS 厂商的 OIDC 实现都没为这个世界设计过。Cognito 缺 RFC 9728/8414/7591；Auth0 / Okta / Entra 用各自的专有 audience 机制；Google 至今 PKCE 还要 client_secret。这些不是 bug，是十年来累积的不同优先级。

**第二层：AS 侧 vs 客户端侧的 workaround 对称性**

很有意思的一点是，**同一个差距可以从两端来打补丁**：
- DCR 缺失 → AS 侧用 Façade 伪造 `/register`，或客户端侧用 `set_client_info` 跳过
- 跨域 metadata → AS 侧用 Façade 同域托管，或客户端侧手动注入 context
- 401 vs 403 → 暂时只有客户端侧能补

**这是非常有用的工程直觉**：当一边补不动时，去看另一边。

**第三层：协议演进会让一部分 workaround 自然过期**

SEP-991 / CIMD 把 DCR 这块从「server registers client」翻转为「client proves identity」，让 Façade 现在最丑陋的"假动态注册"那一段有了下线路径。但它解不了 RFC 9728/8414 metadata 缺失，也解不了 loopback redirect 的精确匹配问题——这些仍然要 Façade 补，只是体量会更小。

> 一句话总结：**MCP 把 OAuth 2.1 的发现机制做到了规范级别强制，而 IdaaS 厂商的目录还没追上。Façade 是这道时间差里的最小可行解；它的形态会随 SEP-991 等协议演进而瘦身，但短期内不会消失。**

## Open Questions

- **SEP-991 普及节奏**：Cognito / Auth0 / Okta 何时实现 `client_id_metadata_document_supported`？目前只有 MCP 自己的 TS / Python SDK 实现了客户端侧；AS 侧主流 IdaaS 都没动。如果 AS 不动，CIMD 模式只能在 Keycloak 这种自托管栈上工作
- **CIMD 的信任模型**：让任意 HTTPS 域名都能"自证身份"会不会被滥用？AS 应该按白名单还是开放？企业部署里这个策略怎么定？
- **Cognito 的 RFC 9728 路线图**：AWS 正在 AgentCore Gateway 这一层补齐 metadata，但 Cognito 自身是否会原生支持 OAuth 2.1 / RFC 9728？目前没有公开信号
- **多租户隔离需要 client_id 级别**的场景下，Façade 的「共享 client」假设可否通过 per-tenant Cognito user pool 解决？N 个 pool 的运维负担如何评估？
- **Sender-constrained tokens（mTLS / DPoP）** 何时进 MCP？OAuth 2.1 推荐了，但客户端生态没动作；Bearer token 仍是默认
- **AgentCore Gateway 的 `allowedClients` 上限**多少？M2M 租户增长后是否会撞限制？这块文档里没说
- **客户端侧 401 vs 403 的兼容**何时能进 MCP SDK 主线？目前 AgentCore 的非标 403 行为只能各客户端自己打补丁，理想是 SDK 默认就识别两者

## References

### 主线文章（Kane Zhu MCP 系列）

- [Technical Deconstruction of MCP Authorization: A Deep Dive into OAuth 2.1 and IETF RFC Specifications](https://kane.mx/posts/2025/mcp-authorization-oauth-rfc-deep-dive/) — 2025-11-12，规范层
- [Implementing MCP OAuth 2.1 with Keycloak on AWS](https://kane.mx/posts/2025/deploy-keycloak-aws-mcp-oauth/) — 2025-11-21，AS 侧 Keycloak
- [MCP OAuth on AgentCore Gateway + Cognito via APIGW Façade](https://kane.mx/posts/2026/agentcore-gateway-cognito-mcp-oauth/) — 2026-05-19，AS 侧 Cognito + Façade
- [Leveraging MCP Client's OAuthClientProvider for Seamless AWS AgentCore Authentication](https://kane.mx/posts/2025/use-mcp-client-oauthclientprovider-invoke-mcp-hosted-on-aws-agentcore/) — 2025-09-04，客户端侧
- [MCP OAuth Evolution: SEP-991 Simplifies Client Registration](https://kane.mx/posts/2025/mcp-oauth-sep-991-simplified-registration/) — 2025-12-02，协议演进
- [How invoking remote MCP servers hosted on AWS AgentCore](https://kane.mx/posts/2025/invoke-mcp-hosted-on-aws-agentcore/) — 2025-08-22，背景资料

### 标准与规范

- [MCP Authorization Specification (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
- [SEP-991 GitHub Discussion](https://github.com/modelcontextprotocol/specification/discussions/991)
- RFC 9728 — OAuth 2.0 Protected Resource Metadata
- RFC 8414 — OAuth 2.0 Authorization Server Metadata
- RFC 7591 — OAuth 2.0 Dynamic Client Registration
- RFC 7636 — Proof Key for Code Exchange (PKCE)
- RFC 8707 — Resource Indicators for OAuth 2.0
- RFC 9700 — OAuth 2.0 Security Best Current Practice
- RFC 8252 — OAuth 2.0 for Native Apps（loopback redirect URI 规则）
- [draft-ietf-oauth-v2-1](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1-14)

### AWS 相关

- [Amazon Bedrock AgentCore Gateway](https://docs.aws.amazon.com/bedrock-agentcore/)
- [Amazon Cognito User Pools](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-identity-pools.html)
- [terraform-keycloak-aws (作者开源 IaC)](https://github.com/zxkane/terraform-keycloak-aws)
