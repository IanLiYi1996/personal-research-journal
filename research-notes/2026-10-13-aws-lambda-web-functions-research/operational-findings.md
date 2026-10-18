# AWS Lambda Web Functions：限制、价格、鉴权与运维核实

核实日期：2026-10-13（Asia/Shanghai）。范围仅为 operational 领域，不重写发布文章摘要、总体架构或传统 Lambda 对比。所有结论来自本次实际取得的 AWS Developer Guide、AWS API Reference、CloudFormation Template Reference、官方价格页及 What's New；通过公开 HTTPS 读取，不部署、不安装、不访问 AWS 账户。

标记约定：**事实**为原页明确说明；**推断**为基于所引事实的分析；**未知**表示本次所查公开材料不足以确认，不能写成支持、不支持、免费或 SLA。Developer Guide 的 `.md` 是原 HTML 页面提供的官方 Markdown 版本；下文引用对应可浏览的 `.html`。

## 1. 可直接用于主报告的结论

- **事实：发布状态与区域。** 2026-10-12 的 What's New 写的是 “available today”，列出五区域：`us-east-1`、`us-east-2`、`us-west-2`、`ap-northeast-1`、`eu-west-1`。该公告没有明确使用 preview 或 generally available/GA 标签；本报告保留公告原有的可用状态表述。[What's New](https://aws.amazon.com/about-aws/whats-new/2026/10/aws-lambda-web-functions/)
- **事实：当前只列 Node.js 24（`nodejs24.x`），仅 Arm64，固定每环境 2 vCPU、2 GB 内存；默认 V8 heap 为 1,024 MB。** 启动监听地址读取 `AWS_LAMBDA_HTTP_ENDPOINT` 的 `host:port`，不是假设服务提供 `PORT`。[运行时](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html)；[指令集](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-configuration-architecture.html)
- **事实：标准 HTTPS/HTTP 1.1，支持流式请求与响应、SSE，不支持 WebSocket upgrade；请求路径为半双工。** 单请求默认 30 秒，可设 3–900 秒；超时覆盖接收 body 和发送响应的整个流。[网络与协议](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html)；[超时](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions-timeout.html)
- **事实：VPC 是出站 connector，不是私网入口；只允许 HomeRegion endpoint 使用。** MultiRegion/PerRegion 不支持引用 connector 的 revision；原生 endpoint 均公网可达，无 PrivateLink 私有 endpoint、原生 custom domain 或直接 WAF association。[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html)
- **事实：鉴权为 `IamAuth` 或 `ApplicationManaged`。** 后者不由 Lambda 鉴权，invoke 时也不评估资源策略。前者使用 `lambda:InvokeWebFunctionEndpoint`，区域 endpoint 用 SigV4，MultiRegion 用 SigV4a。需要 WAF/custom domain 时可使用 CloudFront；仅加 CloudFront 而不限制源站不能防绕过。[访问控制](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-access-control.html)；[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html)
- **事实：基础单价不是未知。** 官方价格页的 N. Virginia/Arm 示例明确为 `$0.30/百万请求`、`$0.0000276944/vCPU-second`、`$0.0000036667/GB-second`，compute 粒度 1 ms；内存按 consumed memory 和环境存活时间计费，包括请求间空闲时间，不是把固定 2 GB 容量全部当作账单用量。示例排除 free tier 和 data transfer，且假设整月仅一个环境常驻，不能作为实际报价。[价格](https://aws.amazon.com/lambda/pricing/)

## 2. Service Quotas 与资源限制

下表全部为**事实**，来自官方专门的 **Lambda Web Functions** 配额部分，不能拿同页 Event Functions 的普通配额表替换。[配额原页](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html#web-functions-quotas)

| 作用域 | 已核实的限制 | 是否可申请增加 |
| --- | --- | --- |
| 每 Web Function endpoints | 10 | 否 |
| 每 Web Function revisions | 50 是自动清理目标，不是永远不能超过的硬上限；清理异步，可能暂时超过 | 否 |
| 每 revision 部署包 | 解压后 512 MB | 否 |
| 每 revision 环境变量 | 合计 32 KB | 否 |
| 每请求 headers | 128 KiB | 否 |
| 每环境 `/tmp` | 512 MB | 否 |
| 每环境 CPU / 内存 | 2 Arm64 vCPU / 2 GB | 否 |
| 每环境请求并发上界 | 默认 64，配置范围 1–128；实际动态并发可以低于配置值 | 范围固定 |
| 每请求 timeout | 默认 30 秒，范围 3–900 秒 | 范围固定 |
| 初始化监听期限 | 10 秒 | 否 |
| 每环境 file descriptors | 1,024 | 否 |
| 每环境 processes/threads | 1,024 | 否 |
| 每 endpoint `maxEnvironments` | 可不设；配置范围 2–10,000；每区域独立执行且受账户 vCPU 配额约束 | 否 |
| 每 endpoint RPS | 未设 `rateLimit` 时仍受 10,000 RPS 上限限制；设定值为 0，或 100–1,000 步长 100，或 2,000–10,000 步长 1,000 | 否 |
| 每 endpoint 分流 revisions | 最多 2；单个 weight 范围 1–100 | 否 |
| 每账户每区域 RPS | 默认 100,000 | 是 |
| 每账户每区域 Arm64 vCPU | 默认 2,000，每环境占 2 vCPU | 是 |

- **事实：新 AWS 账户可能有更低的 Web Functions vCPU/RPS 配额，随使用自动提高。** 文档中的默认值不等于任意新账户已经获批的值；`GetWebAccountSettings` 返回当前区域的配额与 usage。本研究未调用账户 API。[配额](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)；[GetWebAccountSettings API](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_GetWebAccountSettings.html)
- **推断：默认 2,000 vCPU 对应账户每区域约 1,000 个环境的容量预算，跨 endpoints 共享。** 这是 2,000 ÷ 2 的配额换算，不是单 endpoint 承诺，也不是 128,000 并发请求的保证。[配额](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html#web-functions-quotas)；[动态扩缩容](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-scaling.html)
- **事实：没有 minimum 或 reserved-environment setting，始终可缩至零。** 不应写 `minEnvironments`、预留常驻环境或完全消除冷启动；触顶且全部环境忙时通常返回 429，正在初始化时请求可等待；`rateLimit=0` 返回 503/`Platform.EndpointDisabled`。[扩缩容](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-scaling.html)
- **事实：`maxEnvironments` 在 revision weight 变化、内部 fleet cycling 等情况下可能短暂超调；RPS 短暂突发也可能被接受。** 因而限制不是绝对逐瞬间硬封顶。[扩缩容](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-scaling.html)
- **事实：revision 自动清理会删除最老且任何 endpoint 都未使用的 revisions，不删除仍被引用的 revisions 或刚激活的新 revision。** 回滚目标解除全部分流引用后可能被清理；源 S3 对象不随 revision 删除而删除。Web Functions 没有单独的 code storage quota，源包在自有 S3。[revisions](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions.html#web-functions-revisions-limits)
- **未知：账户 Web Functions 总数量上限、专用控制面 TPS、请求/响应 body 数值上限、专用流式吞吐限额。** 本次读取的专门 Web Functions 配额表和 HTTP 页没有给出这些数字；API 错误说明提到函数数目可能触发 quota，但未给数值。不能把普通 Lambda 的 6 MB/200 MB、2 MB/s 或普通控制面 15 TPS 填入 Web Functions。[配额](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html#web-functions-quotas)；[HTTP](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-http.html)；[CreateWebFunction API](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_CreateWebFunction.html)

## 3. Runtime、container 与 durable 支持边界

- **事实：runtime 当前只列 `nodejs24.x`，托管 Node.js runtime 自动应用安全补丁和修复，不支持 runtime management controls。** 无法固定 patch version 或关闭自动更新；不可把 immutable revision 解读成整个底层 runtime 永远不变。[运行时](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html)
- **事实：当前文档化部署为 ZIP，存入启用 versioning 的 S3 bucket，并把每个 revision 固定到特定 object version。** `CodeConfig` 的必需字段是 `s3Object`；当前创建 API 的 code/runtime 模型没有 container image URI、自定义 runtime 或 layers 配置字段。[部署包](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-configuration-deployment-package.html)；[CodeConfig API](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_CodeConfig.html)；[CreateWebFunction API](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_CreateWebFunction.html)
- **推断：不应宣称 Web Functions 当前支持 ECR container image、Python、自定义 runtime 或 Lambda layers。** 已核实的正式路径是 Node.js 24 + ZIP/S3；“在容器中编译依赖”不等于“按 container image 部署 Web Function”。其中非 Node.js runtime 的支持范围已有明确限制，container/layers 则以当前公开配置模型为依据，不假装有单独的禁止声明。[运行时](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html)；[部署包](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-configuration-deployment-package.html)；[指令集](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-configuration-architecture.html)
- **事实：原生二进制依赖必须为 Linux arm64/aarch64 构建；不能选择 x86。** 每个环境中的多个请求共享进程/内存，request-specific 凭证和状态不能放入全局可变变量。[指令集](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-configuration-architecture.html)；[运行时](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html)
- **未知：原生 durable execution 集成。** 本次 Web Functions 专用 runtime、revision、ServiceConfig API 未说明 durable checkpoint/replay 配置。durable guide 讲的是对 event handler 包装并传入 `DurableContext` 的模型，不能因二者都支持 Node.js 就断言 Web Functions 原生支持 durable，也不能把 HTTP 超时扩展为 durable 的一年。外部调用其他服务/函数与本产品原生 durable 是两件事。[ServiceConfig API](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_ServiceConfig.html)；[revisions](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions.html)；[durable guide](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html)

## 4. HTTP、SSE 与时限

- **事实：对外只有 HTTPS 443，Lambda 终止 TLS，内部 HTTP server 按 `AWS_LAMBDA_HTTP_ENDPOINT` 提供的地址监听；无需应用管理 endpoint 证书。** 托管入口/转发协议为 HTTP/1.1；HTTP/1.0 返回 505。支持 `GET/HEAD/POST/PUT/PATCH/OPTIONS/DELETE`，不支持的方法在到应用之前返回 400。[运行时](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html)；[HTTP](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-http.html)
- **事实：128 KiB 是 request line、headers 与终止空行的合计，不仅 header values。** 超出返回 431；`Host` 必须是 endpoint 域名，前置 CloudFront 不能直接转发 viewer Host，否则可能 404。[HTTP](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-http.html)；[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html)
- **事实：流式收发但半双工。** Lambda 完整转发 request body 后才交付应用响应；应用应先读完 body 再写 response。不支持 WebSocket upgrades，可在仅需 server-to-client 单向流时使用 SSE。[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html#web-functions-networking-protocols)
- **事实：非 streaming 超时返回 504/`Runtime.RequestTimeout`；已开始 streaming 的响应超时被截断，保持已经发出的 status（常为 200），且无 `lambda-web-error-type`。** 所以单靠 HTTP 5xx alarm 无法识别所有 SSE/LLM 流失败；需要应用完整性/完成标记与客户端重连逻辑。这后一条是**推断**。[超时](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions-timeout.html)；[metrics](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring-metrics.html)
- **事实：CORS 不由 Lambda 自动添加，应用或 front door 负责 OPTIONS/preflight 与实际响应；CORS 不是鉴权。** `X-Forwarded-For` 前序值与客户端 `X-Forwarded-Host` 未经验证，不能无条件信任为鉴权或 URL 生成依据。[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html)；[HTTP](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-http.html)

## 5. 网络、IAM、CloudFront/WAF

- **事实：三个 endpoint 类型为 `HomeRegion`、`MultiRegion`、`PerRegion`，不能写成 Regional/Global。** 网络 connector 不改变入口公网性；只有 HomeRegion 可接出站 VPC connector。每 revision 引用零或一个 connector，要求同账户、同区域、version-qualified ARN，compute type 必须是 `WebFunction`。[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html)；[ServiceConfig API](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_ServiceConfig.html)
- **事实：无 connector 时直接访问公网 IPv4，没有稳定源 IP、没有私网 VPC 资源路由。** connector 替换默认出站路径，但本身不提供互联网访问；公网 IPv4 要 NAT 路由，IPv6 要 DualStack、IPv6 子网/路由/规则，公网 IPv6 要 egress-only internet gateway。至少两个不同 AZ 的子网；connector operator role 管 ENI，与应用 execution role 分离。[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html)
- **事实：`IamAuth` 区域 endpoint 使用 SigV4；MultiRegion 使用 SigV4a、Region set `*`，签名 service name 为 `lambda`。** 每个签名请求必须有 `X-Amz-Content-SHA256`，无 body 也要有；不支持 chunked body signing。服务不校验 body 与所传 hash 是否匹配，需要 TLS 之外 body integrity 时应用自行实现。[访问控制](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-access-control.html#web-functions-endpoint-access-control-signing)
- **事实：同账户 invoke 可由 identity policy 或 resource policy 授权；跨账户必须二者都允许。** Web Function 上附着资源策略，invoke statement 的 Resource 可以限定 endpoint ARN。使用 `PutResourcePolicy/GetResourcePolicy/DeleteResourcePolicy`，不是普通 Lambda 的 `AddPermission/RemovePermission/GetPolicy`；公开资源策略不允许，未受限 wildcard Principal 会被拒绝。[权限](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-permissions.html)
- **事实：`ApplicationManaged` 对 invoke 不做 IAM 鉴权，也不评估 resource policy；不能用资源策略把这种 public endpoint 变为私有或限制调用者。** 配置治理可通过 SCP 配合 `lambda:Request/WebFunctionAuthType` 阻止创建/更新为 ApplicationManaged；文档的 CLI deploy 默认会建立 ApplicationManaged endpoint。[访问控制](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-access-control.html)
- **事实：custom domain、直接 WAF association 不受原生 endpoint 支持；CloudFront front door 提供 custom domain、缓存、viewer HTTP/2/3、压缩和 WAF。** 这不是 Web Function origin 原生支持 HTTP/2/3。[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html#web-functions-networking-front-door)
- **事实：CloudFront OAC 可接 `IamAuth`，origin type `lambda`，signing behavior 必须为 `always-amz-auth`，策略限定 `cloudfront.amazonaws.com` 和 distribution 的 `aws:SourceArn`。** CloudFront 可透传用户 JWT/cookie 认证，应用 Authorization 应配置转发；公网源域名仍可达，其他获 IAM 授权的 principal 仍可能直连。ApplicationManaged + CloudFront 不自动防止绕过 WAF，需应用检验 origin secret 等证明，或改为受限 IamAuth/OAC。[访问控制](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-access-control.html#web-functions-endpoint-access-control-cloudfront)；[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html)
- **事实：跨区域资源策略复制最终一致，新 grant 或 revoke 不是所有区域瞬间生效，也无客户可见复制完成信号。** 需逐区域测试；急停可停止流量路由或把 endpoint rateLimit 设为 0，而非只依赖策略 revoke。[权限](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-permissions.html#web-functions-permissions-resource-policy-replication)

## 6. 已发布价格、计量与成本边界

### 6.1 已核实单价与计量

下表数值来自本次实际取得的官方价格页 **Lambda Web Functions — Pricing Examples**，明确是 **US East (N. Virginia)、Arm**。不是套用其他 Lambda 产品价格。[价格原页](https://aws.amazon.com/lambda/pricing/)

| 维度 | 示例明确单价 | 计量范围 |
| --- | --- | --- |
| Requests | $0.30 / 1 million | 请求数量 |
| Active vCPU | $0.0000276944 / vCPU-second | 代码活跃消耗 CPU 的时间 |
| Memory | $0.0000036667 / GB-second | 每环境 consumed memory × 环境存活时间 |
| Compute 粒度 | 1 ms | 官方明确写 compute billed at 1 ms granularity |
| Data transfer | 标准 AWS Data Transfer 费率；还包含 Web Function 与 VPC 间传输 | 具体路径/区域相关，不能省略为免费 |

- **事实：内存实际消耗与容量不同。** 同一价格页示例使用 500 MB≈0.49 GB、300 MB≈0.29 GB、145 MB≈0.14 GB、512 MB=0.50 GB，而 runtime 容量固定 2 GB。内存计费覆盖环境存活期，包括请求间隔；多个并发请求共享同一环境的 CPU/内存使用，不按每请求重复计一整个环境。[价格](https://aws.amazon.com/lambda/pricing/)；[运行时](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html)
- **事实：官方四个示例假设 traffic 均匀分布于 30 天/2,592,000 秒，且恰好一个环境全程运行，排除 free tier 和 data transfer。** 文中的 $25.78/$19.61 等示例合计不是任意实际应用的总报价，也未包含其他 AWS 服务。[价格](https://aws.amazon.com/lambda/pricing/)
- **推断：预算模型应为 requests + 活跃 vCPU 秒 + 全部环境的内存时间用量 + 网络 + 依赖服务/运维收费。** 不能用单请求端到端耗时直接当 active CPU 秒，也不能只计算有请求时内存。[价格](https://aws.amazon.com/lambda/pricing/)
- **未知：内存 consumption 的精确账单采样/聚合算法、最小计费内存、初始化与 shutdown 的逐项 billable 边界、被拒绝/限流/鉴权失败请求是否计请求费。** 官方 metrics 提供 `EnvironmentCPUTime`（包括 initialization）、`EnvironmentActiveTime`（包括 initialization）、`EnvironmentMaxMemoryUsage`（环境寿命截至当前的峰值），但这些监控量不等于公开了精确账单算法；不要把 peak memory metric 直接宣称为 billing memory 定义。[metrics](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring-metrics.html)；[价格](https://aws.amazon.com/lambda/pricing/)

### 6.2 价格是否“齐全”

- **事实：价格页已有 Web Functions 专用维度、区域选择表与清晰的基础价格示例，不能写“价格未发布”或“三项单价未知”。** 本次非浏览器 HTML 中区域价格表出现 `{priceOf!...}` 动态占位，但示例单价是实际正文、可直接读取。[价格](https://aws.amazon.com/lambda/pricing/)
- **未知：本次未逐区域解析动态价格表，因此不能保证另外四区域与 N. Virginia 同价。** 数据传输是链接到标准 AWS 费率的路径相关费用，不存在本次已核实的一项统一 all-in transfer 单价。[价格](https://aws.amazon.com/lambda/pricing/)
- **未知：Web Functions 专属免费层额度与共享规则、Compute Savings Plans/其他折扣资格。** Web Functions 示例确实写“exclude the free tier”，但这一句没有给出额度或说明适用范围；页面其他产品段落的 100 万 requests/400,000 GB-seconds 不能自动继承为 Web Functions 的免费层。已取得材料不足以计算 free tier 抵扣，也不足以宣布没有任何免费层。[价格](https://aws.amazon.com/lambda/pricing/)
- **未知：Web Functions 是否存在独立 streaming processed-byte surcharge。** 当前 Web Functions 段落列 request/CPU/memory/标准 data transfer，没有单独列此费项；不能搬用页面其他产品 streaming 的首 6 MB 或 $0.008/GB 规则。[价格](https://aws.amazon.com/lambda/pricing/)

### 6.3 必须另算的费用/预算风险

- **事实：CloudWatch 服务发布的 `AWS/Lambda/Web` 和 `AWS/Lambda/Web/Usage` 指标免费；Logs 不是全部免费。** 各区域 `.local` log group 的写入和存储收费，home Region 集中副本存储也收费；Lambda 首次集中复制 log event 免费。EMF 自定义指标产生 Logs ingestion 和 CloudWatch custom metric charges。[monitoring/telemetry billing](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring.html#web-functions-monitoring-billing)；[CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/)
- **事实：CloudTrail invocation data events 额外收费，默认不记录。** 多区域调用事件写在实际请求执行区域，不自动集中到 home Region，需配置 multi-Region trail/store。[CloudTrail 文档](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-logging-cloudtrail.html)；[CloudTrail pricing](https://aws.amazon.com/cloudtrail/pricing/)
- **事实：customer managed KMS key 额外产生 KMS 费用，多区域 endpoint 要每目标区域均有 multi-Region key replica 且单独配置 key policy。** 默认为 AWS owned key；静态加密始终开启。撤销 KMS 权限也不能立即停止已缓存解密材料的现有环境。[encryption](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-encryption.html)；[KMS pricing](https://aws.amazon.com/kms/pricing/)
- **推断：使用自有 S3 版本化代码存储、NAT、CloudFront、WAF、数据库或模型服务时，应把各自账单加入总成本；“serverless”不代表这些依赖免费。** 其使用路径已由部署和网络文档明确；本研究不估算未给 workload 的总额。[部署包](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-configuration-deployment-package.html)；[网络](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html)；[S3 pricing](https://aws.amazon.com/s3/pricing/)；[VPC pricing](https://aws.amazon.com/vpc/pricing/)；[CloudFront pricing](https://aws.amazon.com/cloudfront/pricing/)；[WAF pricing](https://aws.amazon.com/waf/pricing/)
- **推断：`maxEnvironments`/`rateLimit` 是容量与下游保护，不是硬美元预算。** 每区域独立生效、临时环境超调、内存空闲期收费、日志/传输/下游费用都使其不能替代成本预算告警。[扩缩容](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-scaling.html)；[价格](https://aws.amazon.com/lambda/pricing/)；[monitoring](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring.html)

## 7. 运维工具、IaC、遥测与保障边界

- **事实：What's New 明确宣布 API、Console、CLI、CloudFormation、SAM、CDK、SDKs 与 Agent Toolkit for AWS 支持创建/配置。** 这是一手发布声明；本研究没有安装或验证任何具体 SDK/CDK/SAM 版本。[What's New](https://aws.amazon.com/about-aws/whats-new/2026/10/aws-lambda-web-functions/)
- **事实：CloudFormation 已有 `AWS::Lambda::WebFunction`、`AWS::Lambda::WebFunctionRevision`、`AWS::Lambda::WebFunctionEndpoint`，以及 `AWS::Lambda::NetworkConnector`、`AWS::Lambda::ResourcePolicy`。** WebFunction 资源只创建顶层 function，本身无 revision/endpoint、不服务流量；FunctionName 修改需 replacement，endpoint 类型/区域修改也需 replacement。[CloudFormation Lambda resources](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/AWS_Lambda.html)；[WebFunction](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-lambda-webfunction.html)；[WebFunctionEndpoint](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-lambda-webfunctionendpoint.html)
- **事实：AWS API 文档位于 `lambda/latest/lambda-web/`，不是普通 `lambda/latest/api/`。** 已核实 `CreateWebFunction` 和 `GetWebAccountSettings` 的独立正式 API 页，不能因尝试错误 URL 失败就判定 API 文档缺失。[CreateWebFunction](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_CreateWebFunction.html)；[GetWebAccountSettings](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_GetWebAccountSettings.html)
- **事实：请求/环境指标与 logs 集中到 home Region，quota usage/throttle 指标留在各执行区域。** 集中处理异步，通常数分钟，服务中断可以更晚；首次 metrics centralization 启用前产生的 replica 指标不会事后补入 home Region。服务可能丢失/重复 telemetry，不能视为审计账本。[monitoring](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring.html)
- **事实：指标无 revision dimension。** Canary 判断需应用日志/custom metrics 带 build/revision ID；4xx/5xx 指标不区分应用与平台来源，stream 已发 200 后截断仍按 200 计，5xx/ExecutedRequests 比率在平台提前拒绝时可能高估错误率。[metrics](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring-metrics.html)
- **事实：无原生 Lambda X-Ray service segments，不自动生成缺失的 `X-Amzn-Trace-Id`。** 有此 header 时原样透传，需视为不可信客户端输入；应用可用 ADOT 等自行发 trace 并授权 X-Ray 写入。[monitoring/X-Ray](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring.html#web-functions-monitoring-xray)
- **事实：CloudTrail management events 默认可在 Event history 看 90 天；invoke data events 默认关闭、另收费。** data event resource type `AWS::Lambda::WebFunction` 是 CloudTrail selector 类型，不能单靠这个字符串证明 CloudFormation 支持；本研究已另查 CFN 原页。缺失/错误/过期签名等未通过 authentication 的请求，以及 rateLimit=0 等提前拒绝，不记录 invocation data event。[CloudTrail](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-logging-cloudtrail.html)
- **事实：代码/config 多区域部署可能部分成功，失败区域继续上次成功 revision，不自动全局 rollback；需检查每区域 `regionalEndpoints.updateStatus`。** MultiRegion 路由按 latency，不按区域 health 自动 failover。监控的 home Region 故障也可能使集中遥测不可用，应保留各区域 `.local` 与 usage 观测。[复制部署](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-replication.html)；[monitoring](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring.html)
- **事实：启动需 10 秒内监听，SIGTERM 后最多 1 秒清理，in-flight drain best effort；环境寿命不保证，process 和 `/tmp` 状态会丢失。** 不应把长期 SSE、关机回调或后台任务完成当作保证。[生命周期](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-env-lifecycle.html)
- **未知：本次限定来源没有取得专门 Web Functions SLA、延迟/冷启动 percentile 保证、RTO/RPO 或补偿条款。** 发布宣传的“降低 cold starts”不是契约；“available today”也不是 SLA 定义。不能替产品写零冷启动、最邻近区域必达、自动健康故障转移或无损 drain。[What's New](https://aws.amazon.com/about-aws/whats-new/2026/10/aws-lambda-web-functions/)；[复制部署](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-replication.html)；[生命周期](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-env-lifecycle.html)

## 8. 来源审计、表述差异与待核实项

- **表述精度差异：CloudFormation Endpoint 的 `AuthType` 说明泛写 SigV4，access-control guide 明确 MultiRegion 用 SigV4a。** 多区域签名应采用更具体的鉴权 guide，不能由 CFN 简写推成全部 endpoints 均只需 SigV4。[CFN endpoint](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-lambda-webfunctionendpoint.html)；[鉴权](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-access-control.html)
- **表述精度差异：CFN 描述 MultiRegion “nearest”，专门复制/路由 guide 明确 latency-based best effort，非 health failover、非每请求最低延迟保证。** 不把前者简写当成更强承诺。[CFN endpoint](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-lambda-webfunctionendpoint.html)；[路由](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-replication.html)
- **注意：一般 API schema maximum length 可能描述可接受 name/ARN 容器，不能代替名称规则。** CreateWebFunction/CFN 原页文字均把 function name 限为 64 characters，但 schema 字符串长度标注 256；报告使用实际名称文字规则，不据 256 宣称支持 256 字符名称。[Create API](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_CreateWebFunction.html)；[CFN function](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-lambda-webfunction.html)
- **待核实，不阻碍本报告交付：** Web Functions 免费层/折扣详细规则；各区域完整动态 price table；memory billing 精确采样与 floor；平台拒绝请求收费规则；HTTP body/stream bandwidth 与控制面 TPS；原生 durable/layers/container 进一步正式声明；具体 SDK/CDK/SAM 最低版本；专门 SLA 与合同适用性。上述“未核实”均不等于不支持或不收费。[价格](https://aws.amazon.com/lambda/pricing/)；[专用配额](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html#web-functions-quotas)；[API](https://docs.aws.amazon.com/lambda/latest/lambda-web/API_CreateWebFunction.html)；[What's New](https://aws.amazon.com/about-aws/whats-new/2026/10/aws-lambda-web-functions/)

## 9. 主代理浏览与引用用主要 URL

1. 发布状态/五区域/工具支持：https://aws.amazon.com/about-aws/whats-new/2026/10/aws-lambda-web-functions/
2. Web Functions 专门配额：https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html#web-functions-quotas
3. Runtime/Node 24/patching/监听变量/2 GB：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html
4. Arm64/2 vCPU：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-configuration-architecture.html
5. ZIP/S3 部署包：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-configuration-deployment-package.html
6. 扩缩容/无预留环境/超调：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-scaling.html
7. HTTP 契约/headers：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-http.html
8. HTTP/SSE timeout：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions-timeout.html
9. VPC/PrivateLink/WAF/WebSocket/SSE/CORS：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html
10. IAM/SigV4a/OAC/应用鉴权：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-access-control.html
11. 资源策略/复制/执行角色：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-permissions.html
12. 单价/计量/数据传输/示例：https://aws.amazon.com/lambda/pricing/
13. Telemetry 支持与收费：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring.html
14. Metrics/计量定义/无 revision dimension：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-monitoring-metrics.html
15. CloudTrail/收费/数据事件边界：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-logging-cloudtrail.html
16. Revision 清理/回滚/S3 存储：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions.html
17. KMS/静态加密：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-encryption.html
18. Create API：https://docs.aws.amazon.com/lambda/latest/lambda-web/API_CreateWebFunction.html
19. CodeConfig API：https://docs.aws.amazon.com/lambda/latest/lambda-web/API_CodeConfig.html
20. ServiceConfig API：https://docs.aws.amazon.com/lambda/latest/lambda-web/API_ServiceConfig.html
21. GetWebAccountSettings API：https://docs.aws.amazon.com/lambda/latest/lambda-web/API_GetWebAccountSettings.html
22. CFN 资源列表：https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/AWS_Lambda.html
23. CFN WebFunction：https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-lambda-webfunction.html
24. CFN Endpoint：https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-lambda-webfunctionendpoint.html
25. Durable guide（仅作范围核对）：https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html
26. 区域复制/路由运维边界：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-replication.html
27. 生命周期：https://docs.aws.amazon.com/lambda/latest/dg/web-functions-env-lifecycle.html
