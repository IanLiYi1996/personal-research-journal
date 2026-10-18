# Lambda Web Adapter 与 Lambda Web Functions 对比

核查日期：2026-10-13，Asia/Shanghai。仅一手资料；未部署、未访问 AWS 账户、未进行性能实测。工作区未找到可确认用户此前方案的 Adapter 配置，因此主要基线为 **AWS Lambda Web Adapter（LWA）＋传统 Lambda on-demand**，不推定用户所用版本、语言或入口。当前官方 Adapter 仓库已从 `awslabs` 重定向到 `aws`。

## 1. 结论

**判断：LWA 解决“让标准 HTTP 应用适配既有 Lambda 调用接口”；Web Functions 改变了 HTTP 工作负载的托管入口、执行并发和计费模型。** 对已经使用 LWA 的应用，“业务代码不用改成 handler”不是新卖点。真正新增的选型价值在于多并发复用、活跃 CPU 计费和托管多区域发布；代价是当前语言、打包和资源范围更窄。[1]、[2]

旧方案的应用也监听本地端口，但外部请求先形成 Lambda event，Adapter 再转换为 HTTP，并把应用响应转换回 Lambda response。Web Functions 则由托管端点直接向应用 HTTP server 转发，不要求 LWA。[1]、[2]

## 2. 对照表

| 维度 | LWA＋传统 on-demand Lambda | Web Functions |
| --- | --- | --- |
| HTTP 应用接口 | 已可使用普通 Web 框架，无须改为业务 handler | 同样可用普通 Web 框架，但当前仅 Node.js |
| 请求路径 | Function URL / API Gateway / ALB → Lambda event → LWA → HTTP server | 托管 Web Function endpoint → HTTP server |
| 单环境请求并发 | 标准 on-demand 同时一个调用；框架内部 async 不会改变平台单并发 | 多请求并发，动态限制；上限默认 64，可配置 1–128 |
| 计算计费 | 配置内存 GB × 调用执行时间，包含 I/O 等待 | 活跃 CPU 时间＋消耗的内存 × 环境存续时间 |
| 语言、CPU 架构 | 可用 Python、Java、Node.js 等；Adapter 有 x86_64/Arm64 二进制 | 当前 Node.js 24、Arm64 |
| 包格式 | 支持 OCI 容器镜像和 ZIP＋Layer | 当前正式路径为 ZIP，不能直接搬已有 OCI image |
| 资源选择 | 标准 Lambda 内存可配 128–10,240 MB，CPU 随内存变化 | 每环境 2 vCPU、2 GB 内存 |
| 流式响应 | 已支持，需结合入口和 Adapter streaming 配置 | 原生响应流，无需 Adapter streaming 配置 |
| 冷启动容量控制 | 可结合 Provisioned Concurrency 预初始化 | 没有最低或预留环境设置，空闲后缩到零 |
| 入口与事件集成 | 既有 API Gateway、ALB、Function URL；支持非 HTTP event pass-through | HTTP 专用 endpoint；事件工作负载仍应使用 Event Functions |
| 多区域 | 需组织跨区域部署和入口选路；不是 Adapter 提供的功能 | 托管代码/config 复制，MultiRegion 延迟选路；不等于自动健康 failover |
| VPC | 使用传统 Lambda 的 VPC 出站访问能力 | connector 仅支持 HomeRegion，不能与 MultiRegion/PerRegion 直接组合 |

依据：[1]、[2]、[3]、[4]、[5]、[6]、[7]、[8]、[9]、[10]、[14]。本表比较的是平台和组合架构，不能把所有区别归因于 Adapter 代码。

## 3. 性能与成本为什么可能变好

以下是概念例子，不是实测或报价：某 HTTP 请求持续 3 秒，应用累计只消耗约 100 ms CPU，其余主要等待模型或数据库。

- **旧 on-demand：** 一个调用持续约 3 秒，消耗一个环境的调用并发，并按配置内存与执行时间计计算费用。[3]、[4]
- **Web Functions：** CPU 项按实际活跃 CPU 用量计费，等待期间仍有内存费用；环境可同时服务其他请求，具体复用率由动态并发和 CPU/内存决定。[2]、[5]

**判断：** 节省潜力主要来自 I/O 等待时不按完整墙钟时间计 CPU，以及多个请求共享环境；不能宣称只是去掉 Adapter 就必然快很多，也不能把 `3 s / 0.1 s = 30` 当成 30 倍省钱。两种模式的价格单位、内存计量、入口、免费额度和环境寿命不同，需要实测账单对照。

CPU 密集请求不会因为“HTTP 原生”而减少必需计算量；Node.js 同步 CPU 工作还会阻塞同一环境中的其他请求。Web Functions 下，全局可变的用户凭证、请求对象、临时文件名和数据库连接使用都需要做并发审计。[6]

## 4. Managed Instances 是重要的第三种情况

**当前 LWA 已支持 Lambda Managed Instances 的多并发调用，同时支持 buffered 和 streaming 模式。** 所以“Adapter 永远每环境只能处理一个请求”是错误的；单并发是上述传统 on-demand 基线的限制。[11]

LWA＋Managed Instances 依然通过 event-to-HTTP 转换，但可以复用实例内并发；其收费为请求费、EC2 实例费和管理费，而不是 Web Functions 的活跃 CPU/消耗内存模型。因此如果此前方案采用 Managed Instances，就应做三方比较，不能套用上表的 on-demand 并发及计费结论。[11]、[12]

## 5. 迁移判断

以下是建议，不是已执行的迁移：

- **已有 Node.js HTTP/SSR/模型 SSE 服务：** 值得做 Web Functions POC，主要看并发复用、p95/p99、冷启动和总账单。
- **已有 Python/FastAPI、Java、依赖自定义容器或超过 2 GB 的环境：** 当前不能按原样迁入 Web Functions，保留 LWA 更直接。
- **已有 API Gateway authorizer、复杂路径映射、event pass-through 或预初始化容量：** 先列出依赖并审计；Web Functions 不是移除 Adapter Layer 就完成的配置升级。
- **已有多区域＋私有 VPC 数据库访问需求：** 特别检查 Web Functions 的 HomeRegion-only connector 限制，不能只按公告的多区域能力选型。

监听地址需要从原来的 `AWS_LWA_PORT` / `PORT` 等配置迁至 `AWS_LAMBDA_HTTP_ENDPOINT` 的 `host:port`；Native 默认入口为 `index.js`，可通过 `AWS_LAMBDA_NODEJS_ENTRYPOINT` 指定其他入口。迁移需要新的 Web Function、Revision、Endpoint 及部署/权限配置，不是沿用原 Event Function 的 URL 和资源。[6]、[13]

更细的请求语义和迁移陷阱见 [兼容性核查](/research-notes/2026-10-13-aws-lambda-web-functions-research/lwa-migration-findings.md)；Web Functions 完整边界见 [此前调研](/research-notes/2026-10-13-aws-lambda-web-functions-research.md)。

其中值得优先检查的三项：

1. **就绪条件变了。** LWA 可以等自定义 HTTP readiness path/status；Web Functions 检测到指定地址开始监听就可以接入请求，必须 10 秒内监听。延迟初始化不能只保留原 health route，需要让相关 handler 等依赖真正就绪。[15]、[16]
2. **中间件语义可能变了。** LWA 的 base-path 裁剪、`x-amzn-*-context`、Authorization 恢复配置不再由 Adapter 提供；原有重定向、用户身份、客户端 IP 和路径前缀需做回归。[17]、[18]
3. **流式默认时限要重新确认。** Native 默认请求超时只有 30 秒，最长 900 秒；已发送 `200` 的流超时会截断正文，应用要检测完整性及取消上游操作，不能只看 5xx。[19]

## 来源

[1]: https://aws.github.io/aws-lambda-web-adapter/reference/architecture.html
[2]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions.html
[3]: https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.html
[4]: https://aws.amazon.com/lambda/pricing/
[5]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-scaling.html
[6]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html
[7]: https://aws.github.io/aws-lambda-web-adapter/
[8]: https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.html
[9]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-configuration-architecture.html
[10]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html
[11]: https://aws.github.io/aws-lambda-web-adapter/features/managed-instances.html
[12]: https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html
[13]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-how-it-works.html
[14]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-replication.html
[15]: https://aws.github.io/aws-lambda-web-adapter/configuration/readiness-check.html
[16]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-env-lifecycle.html
[17]: https://github.com/aws/aws-lambda-web-adapter/blob/2171d5a68bb5de323d5c7271676caf9674049cb2/src/lib.rs
[18]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-http.html
[19]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions-timeout.html
