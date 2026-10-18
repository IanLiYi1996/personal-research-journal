# AWS Lambda Web Functions 调研

调研日期：2026-10-13，Asia/Shanghai。仅依据公开 AWS 一手资料；未部署应用、访问 AWS 账户或进行性能实测。文中的“判断”和“建议”不代表 AWS 承诺。

## 1. 结论

**判断：这是 Lambda 面向标准 HTTP 服务的一种新执行形态，而不是给传统 Lambda 多加一个 URL。** 它把常驻 HTTP 进程、单环境请求并发、活跃 CPU 计费、端点路由和多区域代码部署整合起来。它适合评估为 Web 后端、SSR 和流式 HTTP API 的运行平台，但不能据此认定它解决了冷启动、数据跨区域复制或区域级容灾。[1]、[2]、[3]、[4]

原公告发布于 2026-10-12，列出初始五个区域：美国东部北弗吉尼亚和俄亥俄、美国西部俄勒冈、欧洲爱尔兰、亚太东京。中文博客地址不等于中国区域可用。[1]

## 2. 真正变化了什么

| 维度 | 传统 Lambda on-demand + HTTP 入口 | Lambda Web Functions |
| --- | --- | --- |
| 应用接口 | handler 接收调用事件；Function URL 把 HTTP 转为事件 | 应用启动自己的 HTTP server，通过 `AWS_LAMBDA_HTTP_ENDPOINT` 指定的地址接收请求 |
| 单环境并发 | 标准 on-demand 环境通常一次处理一个请求 | 同一运行环境处理多个 HTTP 请求 |
| 环境复用 | 环境可跨调用复用，但标准 on-demand 调用模型每次处理一个事件 | 同一个 HTTP server 跨请求复用，并在环境内并发处理请求 |
| 缩容 | 按 Lambda 调用及并发扩展 | 根据 CPU、并发请求和配置管理环境数，可以缩到零 |
| 主要计费 | 请求次数、按配置内存及执行时长计费 | 请求次数、实际使用的 vCPU 时间、消耗的内存时间 |
| 多区域 | 通常需自行部署、配置路由 | MultiRegion 端点复制代码和配置，并按网络延迟路由 |

依据：[2]、[3]、[4]、[5]、[6]、[7]。这里特意限定“传统 on-demand”：不能把表格泛化到 Lambda Managed Instances 等其他执行模式。

### 标准 HTTP 应用接口

Web Functions 不要求把 Express、Hono、Fastify、Next.js 等 Node.js 应用改写成 Lambda handler；应用需要启动 HTTP server，监听 `AWS_LAMBDA_HTTP_ENDPOINT` 指定的 `host:port`。默认入口是部署包根目录的 `index.js`，可以通过 `AWS_LAMBDA_NODEJS_ENTRYPOINT` 指定其他入口文件。**当前只支持 Node.js 24、Arm64 和 ZIP 部署；不能把“标准 HTTP server”理解为任意语言或容器镜像可部署。** 文档并没有要求 `/health`，就绪条件是应用开始监听指定地址。[1]、[2]、[8]、[9]

**判断：** 迁移主要从“适配调用事件”转为“满足启动、监听地址和无状态运行要求”。Node.js HTTP 应用代码较易复用，不代表部署配置、IAM、端点和监控也具有跨云可移植性；Python/FastAPI 应用目前不能直接迁入。

### 资源和发布模型

资源分为 Web Function、Revision 和 Endpoint。Revision 是代码和配置的不可变快照；Endpoint 是 HTTPS 入口，可以把流量路由到一个或两个 Revision，实现加权灰度与回滚。修改并发、超时和环境变量等 Revision 配置，也需要创建新 Revision。[3]

**重要运维细节：** Revision/Endpoint 显示 `Active` 并不证明应用能成功启动；创建端点时不会验证服务器监听。发布后仍须进行真实请求探测和日志检查。MultiRegion 的复制是最终一致的，更新失败的区域可能继续运行旧 Revision，部署流程需要逐区域检查状态。[3]、[10]

### 为什么延迟可能更稳定

Lambda 根据资源使用情况自动调整每环境的动态并发限制；`maxConcurrencyPerEnvironment` 设置其上限，默认 64、范围 1–128。已有环境接近动态限制时会扩容并保留一定余量；新环境启动期间，请求先进入有容量的环境，没有容量时可能等待启动完成。[3]、[4]

**没有最低或预留环境数设置。** 无流量时环境缩到零，下次请求重新冷启动，不能用不存在的 `minEnvironments` 解决首请求延迟。[4]

**判断：** 并发复用和提前扩容可以减少部分启动等待，但缩到零后的首批请求、突发流量、CPU 饱和和下游数据库延迟仍然需要实测。官方公告的定性性能描述不能视为应用 p95/p99、跨洲总延迟或冷启动 SLA。[1]、[4]、[9]

## 3. “全球部署”的准确含义

三种端点必须区分：[3]、[10]

- **HomeRegion：** 仅在创建 Web Function 的主区域运行，提供区域域名。
- **MultiRegion：** 把代码和配置复制到所选计算区域，通过全球入口按网络延迟选择区域。
- **PerRegion：** 复制到多个区域，每个区域有独立区域域名，由客户管理流量路由。

最重要的限制是：**MultiRegion 的选路基于延迟，不基于区域健康。** PerRegion 允许客户通过自己的 Route 53 策略或 CloudFront 管理路由。不能把 MultiRegion 当成自动区域容灾机制。[10]

端点类型以及 MultiRegion 区域列表创建后不能直接更改。需要改变拓扑时，应规划新端点及流量迁移。[10]

**判断：** MultiRegion 是托管的计算部署与延迟路由，不是完整的多区域业务系统。数据库、会话、对象存储、秘密、IAM 权限、写入冲突和数据驻留仍需要独立设计。

例如，把计算复制到东京但数据库仍在北弗吉尼亚，东京请求仍可能产生跨区域数据库往返。因此需要比较“客户端到计算”的收益与“计算到数据”的额外延迟，而不能只测静态页面或空接口。[3]、[10]

## 4. 执行环境与生命周期

扩缩容文档规定每个执行环境消耗 2 vCPU；运行时文档规定环境内存 2 GB、默认 Node.js V8 heap 1,024 MB，采用 Arm64。应用必须在 10 秒内开始监听指定地址。关闭时先尽力排空请求，再发送 `SIGTERM`，应用最多约 1 秒完成清理；不保证所有进行中的请求或清理回调完成。[1]、[4]、[8]、[9]

缩到零会回收全部环境；环境也可能因部署、维护和错误被替换。因此进程内存和临时磁盘不是持久状态。服务需要把业务状态、任务和文件放到外部持久化系统。[4]、[9]

应用和运行时活跃消耗的 CPU 会被计费；响应结束不等于环境立即回收或内存计费停止。**判断：** 不应把运行中环境视为免费后台 worker，更不应依赖它完成必须可靠交付的长任务。[2]、[9]

具体 HTTP、网络、鉴权和配额限制见本次配套的一手文档核查：
[运维核实笔记](/research-notes/2026-10-13-aws-lambda-web-functions-research/operational-findings.md)。这些限制也是上线前检查项，不能只依靠公告支持的框架名单判断兼容性。

### 协议、安全与网络的硬边界

- 支持流式请求、流式响应和 SSE，**不支持 WebSocket upgrade**。请求路径是半双工：完整请求 body 转发完后，才向客户端交付响应；不能当任意全双工长连接宿主。[13]
- 单请求默认 30 秒，可以设置为 3–900 秒，超时覆盖整个响应流。流式响应已发出 `200` 后超时会截断正文，仍可能保留 `200`，因此只监控 5xx 不足以判断流完整性。[14]
- **VPC connector 仅用于出站，且当前只支持 HomeRegion。** MultiRegion/PerRegion 不能路由到引用 connector 的 Revision。因此“多区域复制”和“直接连私有 VPC 数据库”不能当成当前可组合能力；公网服务路径或额外数据架构需要另行评估。[13]
- 入口域名始终公网可达；IAM 鉴权不使它成为网络层私有 endpoint。当前没有 Web Function 域名的 PrivateLink 接口端点。[13]
- 可选 `IamAuth` 或 `ApplicationManaged`。前者需 IAM 签名，区域端点用 SigV4、MultiRegion 用 SigV4a；后者把认证授权责任交给应用，调用时不评估资源策略。[15]
- 自定义域名、缓存和 WAF 要加 CloudFront 等前置服务；仅把 CloudFront 放在 `ApplicationManaged` 端点前面不能阻止客户端绕过 CDN/WAF 直连。生产应限制源站入口，例如采用受限 `IamAuth` 与 CloudFront OAC。[13]、[15]

这些边界使它更像受托管的 HTTP/SSE 服务平台，而不是通用容器、私网服务或任意实时通信平台。此句为选型判断。

## 5. 成本：省等待 CPU，不等于免费等待

官方计费维度为：[2]、[11]

1. **请求数量。**
2. **Active CPU：** 实际消耗的 vCPU 时间；HTTP 请求等待 I/O 时没有消耗的 CPU 时间不计入这一项。
3. **Memory：** 环境消耗的内存随时间计费，包含活跃环境无请求期间的内存；不能只按 handler 或请求持续时间算。

因此可以使用概念公式：

> 总成本 ≈ 请求数 × 请求单价 + 累计 vCPU 秒 × CPU 单价 + 累计消耗 GB 秒 × 内存单价 + 数据传输及下游服务成本。

以北弗吉尼亚、Arm 的官方定价示例为准，当前单价是每百万请求 **0.30 美元**、每 vCPU 秒 **0.0000276944 美元**、每 GB 秒内存 **0.0000036667 美元**；计算按 1 ms 粒度计量。这里不是东京、欧洲或所有区域统一报价。[11]

需注意，**执行环境的 2 GB 内存容量不等于定价例子中统一按 2 GB 全额计费**；价格示例使用具体消耗的内存值计算。[8]、[11]

官方 I/O 型 API 示例每月 1,000 万请求、端到端 200 ms、活跃 CPU 50 ms、内存消耗 0.29 GB，合计 19.61 美元。它假设 30 天请求均匀分布，始终恰好一个环境运行，并排除免费额度和数据传输。因此只是计费方法示例，不能直接用作生产总成本或低流量保底费用；数据库、日志、CDN 和模型调用等费用也需要另行纳入预算。[11]

环境尚未回收时，空闲等待仍有内存成本；若应用或运行时消耗 CPU，则另计活跃 CPU 成本。没有最低或预留环境设置，不能依赖可配置的常驻实例数。缩容到零降低无流量时的资源成本，但也重新暴露首请求启动开销。[2]、[4]、[9]、[11]

**判断：** I/O 密集、可复用连接、单环境能处理较多并发请求的服务值得优先测算；CPU 密集或内存占用较大的服务未必比传统 Lambda 更便宜。低流量而又要求多区域始终常驻的需求，也不能由当前最低容量设置直接满足。实际比较必须匹配区域、Arm 架构、总端到端时长、环境寿命、并发复用率和免费额度。

## 6. 候选场景与边界

以下是基于机制的选型判断，不是性能实测：

| 场景 | 初步判断 | 需要验证 |
| --- | --- | --- |
| Node.js SSR / Next.js / 标准 Web 后端 | 值得试点 | 冷启动、首字节、构建产物、静态资源与缓存策略 |
| 等待数据库、第三方 API 的 HTTP 服务 | 值得试点 | 实际 CPU/内存计量、连接池并发、下游容量 |
| 调用外部模型的 Node.js 流式 HTTP 服务 | 可评估 | SSE 首字节、15 分钟总时限、取消传播、截断检测和内存占用 |
| WebSocket 实时双向服务 | 当前不支持 | 明确不支持 upgrade，不能用 SSE 代替真正双向通信 |
| 多区域计算直接连接私有 VPC 数据库 | 不能按公告默认可组合 | VPC connector 当前仅 HomeRegion 支持 |
| 重 CPU 服务 | 谨慎 | 固定环境规格、Node.js 事件循环阻塞、排队和价格 |
| 长期后台任务、持久进程状态 | 不应直接依赖 | 任务持久化、关停宽限、环境替换与失败恢复 |
| 强容灾、强数据驻留系统 | 需要额外架构 | 区域故障切换、数据复制、权限和网络路径 |

事实依据：[2]、[3]、[4]、[8]、[9]、[10]、[13]、[14]；更多协议边界见配套运维核查。

## 7. 建议如何做 POC

以下是建议，不是已执行的测试：

1. **先单区域。** 选一个现有真实 HTTP 服务，保留相同下游数据库和上游 API，与当前部署做对照。
2. **测四种状态。** 缩到零后首次访问；持续有流量的稳定访问；阶跃式突发流量；持续高 CPU 或大量 I/O 等待。当前没有可设置的最低常驻环境数。
3. **记录用户指标。** p50/p95/p99、TTFB、5xx、超时、流式中断、取消传播，而非只看平均函数执行时间。
4. **记录资源指标。** 环境数、启动耗时、活跃 CPU、消耗内存时间、并发请求、下游连接数和费用。
5. **然后验证多区域。** 分别测“计算和数据同区”“计算跨区、数据单区”“数据也有区域化方案”。单独设计区域故障切换，不把延迟路由当容灾。
6. **检查关停和发布。** 在部署、缩容和实例替换时验证进行中的请求、数据库写入和后台操作，不依赖 1 秒清理完成不可恢复工作。

建议依据：[4]、[9]、[10]、[12]。

## 8. 证据状态

- 公告和 developer guide 是本调研的主要依据；价格和限制按调研日页面核查。
- 未取得独立性能基准，也未执行 AWS 账户内测试，因此不能给出具体延迟改善百分比或节省比例。
- 没有将中文站点路径、支持框架、全球入口或多区域代码复制推导为中国区域可用、所有协议可用或自动容灾。
- 配套运维笔记记录价格、协议、安全与支持范围的详细证据；最终选型应复核该笔记与当前服务文档。
- Web Functions 免费层详细额度、精确内存计量算法、被拒绝请求收费、完整区域价格表、专用 body/吞吐限制和原生 durable 集成在本次核查中仍未确认；“未确认”不代表免费、无限或不支持。

## 参考资料

[1]: https://aws.amazon.com/cn/blogs/aws/announcing-aws-lambda-web-functions-for-fast-consistent-web-app-performance-anywhere/
[2]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions.html
[3]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-how-it-works.html
[4]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-scaling.html
[5]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-create.html
[6]: https://docs.aws.amazon.com/lambda/latest/dg/urls-invocation.html
[7]: https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.html
[8]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html
[9]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-env-lifecycle.html
[10]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-replication.html
[11]: https://aws.amazon.com/lambda/pricing/
[12]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-testing.html
[13]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html
[14]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions-timeout.html
[15]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-endpoint-access-control.html
