# LWA → Lambda Web Functions：实际兼容性陷阱

核实日期：2026-10-13（Asia/Shanghai）。仅公开一手源码、README、AWS Developer Guide；未部署、安装、访问账户或实测。本笔记补充请求语义和迁移兼容性，其他选型维度见同日主研究。
LWA 官方 `awslabs` 仓库地址现重定向至 `aws`；本次固定核查提交 `2171d5a68bb5de323d5c7271676caf9674049cb2`（README 标示 1.1.0），不是假定客户已经运行该版本。旧 LWA 版本需另对照其固定 tag。
标记：**事实**＝来源明确说明或源码直接实现；**推断/建议**＝迁移分析；**未知**＝公开材料未确证。

## 1. 监听、入口与健康检查

- **事实：** LWA 默认连接 `127.0.0.1:8080`，`AWS_LWA_PORT` 可回退到 `PORT`；ZIP 启动约定为 `AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap`、handler 指向 `run.sh` 等脚本。[LWA README][L1]
- **事实：** Web Functions 要读取并精确绑定 `AWS_LAMBDA_HTTP_ENDPOINT` 的 host:port；入口默认为包根 `index.js`，可用 `AWS_LAMBDA_NODEJS_ENTRYPOINT` 改名；不读取 `package.json` 的 entry point。Next.js standalone 需入口先从该变量设置 `HOSTNAME`、`PORT` 再启动。[运行时][W1]；[环境变量][W2]
- **事实：** LWA 默认 HTTP GET `/`，可改端口/路径或使用 TCP；HTTP healthy 默认 **100–499**，不是“只有 2xx”。`AWS_LWA_ASYNC_INIT` 可在约 9.8 秒后结束等待，再于首请求等待就绪。[README][L1]；[readiness 源码][L3]；[初始化源码][L2]
- **事实：** Web Functions 必须 10 秒内开始监听；服务检测监听即认为 ready，立即可路由请求。官方没有文档化等价的自定义 HTTP readiness path/status gate；推迟初始化时，依赖它的 handler 应等待共享 promise。[生命周期][W3]
- **推断/建议：** 不能只搬 `PORT=8080`、`run.sh` 或 `AWS_LWA_READINESS_CHECK_*`。移除对 wrapper/handler 的启动依赖；将“依赖未就绪返回 503，由 LWA 阻挡”的逻辑改为应用自身请求门控，避免先 listen 后误接真实流量。[L1]、[W1]、[W2]、[W3]

## 2. headers、path、base64 与 event 来源

- **事实：** LWA 给请求注入 JSON `x-amzn-request-context` 和 `x-amzn-lambda-context`；`AWS_LWA_AUTHORIZATION_SOURCE` 会移走指定原始头并覆盖 `Authorization`。[fetch_response 源码][L2]
- **事实：** Web Functions 明确增加的是 `lambda-web-request-id`、`lambda-web-endpoint-name`；直连 IAM 签名请求验证后移除签名用的 `Authorization`。CloudFront OAC 的专门集成路径可以保留 viewer 的业务 `Authorization`，不能把前述规则泛化成所有业务令牌都会消失。清除 `X-Real-IP`、三种方法覆盖头、连接相关头、客户端 `lambda-web-*`；响应的连接级头和 HTTP trailers 也会移除。[HTTP 合同][W4]；[网络和 CloudFront 集成][W5]
- **推断/建议：** context 中的 authorizer claims、调用 deadline/request ID 不再有 LWA 自动构造路径；改为应用认证上下文和 `lambda-web-request-id`。不要把客户端自传 `x-amzn-*` 当可信平台身份。原来借 `AWS_LWA_AUTHORIZATION_SOURCE` 恢复用户令牌的方案需应用显式实现，IAM 签名头不等于业务 JWT。[L2]、[W4]
- **事实：** Web Functions 的 `Host` 必须为小写、无端口的 endpoint 域名，应用收到此域名；`X-Forwarded-For` 只追加连接者 IP，前序值未验证；`X-Forwarded-Host` 未验证。[HTTP 合同][W4]
- **推断/建议：** 旧的 `trust proxy`、公网 URL/redirect 生成、客户端 IP 获取和方法覆盖 middleware 必须复核；迁移后不能继续靠 `X-Real-IP` 或被删除的方法覆盖头。[L2]、[W4]
- **事实：** 本次 LWA 的 `AWS_LWA_REMOVE_BASE_PATH` 仅在段边界裁掉一次前缀：`/api/api/order`→`/api/order`，`/apiorder` 不裁；之后用 `Url::set_path` 与 URI query 构建转发地址。Web Functions 明确保留 request path、query、body，未文档化 LWA 的前缀裁剪。[路径源码][L2]；[HTTP 合同][W4]
- **推断/建议：** 原来依赖 stage/base-path 裁剪的路由需在应用或前置层恢复；用 `%2F`、重复 query、重复 cookie/header、非 UTF-8 二进制做回归，不把 event map 重建结果与直通 HTTP 假定为逐字节等价。[L2]、[W4]
- **事实：** LWA 声明自动编码二进制响应；其 `lambda_http` 转换按 event 的 `isBase64Encoded` 解码请求，再由 LWA 发送 `Body::Binary`。Web Functions 直接传 HTTP body，不要求 Lambda event/response JSON envelope。[README][L1]；[AWS Rust request 转换源码][L6]；[LWA body 转发][L2]；[HTTP 合同][W4]
- **推断/建议：** 框架本来收发字节的代码通常无需新增 base64；手写 `{statusCode, headers, body, isBase64Encoded}` HTTP 响应或再次解码 body 的兼容层应移除，避免返回 envelope JSON 或双重编码。[L1]、[L2]、[L6]、[W4]
- **事实：** LWA 可把 SQS/SNS/S3 等非 HTTP event JSON 转成 POST，默认 `/events`，`AWS_LWA_PASS_THROUGH_PATH` 可修改；Web Functions 的运行时合同是原生 HTTP server，不提供这一 LWA event→POST 功能。[非 HTTP 指南][L5]；[PassThrough 源码][L2]；[运行时][W1]
- **推断/建议：** 保留 `/events` 路由不等于原来的 event source mapping、批失败返回值和重试契约已迁移；保留独立 Event Function/LWA 消费者，或另设计明确的 HTTP 桥接。此结论不声称所有外部事件服务都无法调用 HTTP。[L2]、[L5]、[W1]
- **事实／安全提醒：** 该 LWA 版本会阻挡外部请求访问已配置 SnapStart hook 路径；README 明确警告去掉 adapter 后这些普通应用路由可能无认证暴露。迁往直连 HTTP 时应删除或自行保护，不能只保留 hook env 配置。[README 警告][L1]；[路径保护源码][L2]

## 3. streaming 配置、超时与停止传播

- **事实：** LWA 默认 buffered；流式需 `AWS_LWA_INVOKE_MODE=response_stream` 与 Function URL invoke mode 匹配。LWA 的 gzip/br 压缩在 streaming 模式自动禁用。[流式指南][L4]；[README][L1]
- **事实：** Web Functions 原生流式 request/response 和 SSE，无须 event→HTTP adapter；路径为**半双工**，完整 request body 转发完才交付 response，官方要求读完 body 再写响应；无 WebSocket upgrade。[运行时][W1]；[网络 streaming][W5]
- **推断/建议：** 旧 invoke-mode/压缩 env 不再替应用或前置层配置流式；复测 SSE flush 与压缩缓冲。边上传边期望收到 early error、ACK 或聊天 token 的双向交互不能由“支持 streaming”推导为可用；HTTP trailers 也不能充当完成标志。[L4]、[W4]、[W5]
- **事实：** `timeoutSeconds` 默认 30、范围 3–900 秒；覆盖收 body 与发 response 全流。非 streaming 超时为 504/`Runtime.RequestTimeout`；已开始流式则截断 body，保留已发 status（常为 200），无错误类型头。[timeout][W6]；[网络][W5]
- **推断/建议：** 显式重新设置请求超时；长 SSE 需要完成标记、断流/重连检测，不能只监控 5xx，也不能把心跳当延长总时限。[W5]、[W6]
- **事实：** 官方要求客户端断开时停止 upstream work；环境关停先停止路由新请求、尽力排空，再发 `SIGTERM`，仅最多 1 秒清理，不保证进行中请求完成。[网络][W5]；[生命周期][W3]
- **未知：** 所查 devguide 未承诺断开/平台超时何时转换成 Node.js 的哪种 socket/stream 事件，也未承诺自动取消下游 fetch、模型调用或数据库操作；不能将“截断客户端流”写成“上游任务已终止”。[W3]、[W5]、[W6]
- **推断/建议：** 请求级 AbortController 应连接到真实 response/socket close、应用 deadline 与 SIGTERM，并传到支持取消的下游；别把正常 request-body 读完的事件误判为客户端取消。检查断线后生成/写入是否停止及幂等性；长 drain 不能依赖超过 1 秒的 shutdown 宽限。以上是待验证方案，非本次实测结果。[W3]、[W5]、[W6]

## 一手来源（逐项编号；源码固定提交）

[L1]: https://github.com/aws/aws-lambda-web-adapter/blob/2171d5a68bb5de323d5c7271676caf9674049cb2/README.md
[L2]: https://github.com/aws/aws-lambda-web-adapter/blob/2171d5a68bb5de323d5c7271676caf9674049cb2/src/lib.rs
[L3]: https://github.com/aws/aws-lambda-web-adapter/blob/2171d5a68bb5de323d5c7271676caf9674049cb2/src/readiness.rs
[L4]: https://github.com/aws/aws-lambda-web-adapter/blob/2171d5a68bb5de323d5c7271676caf9674049cb2/docs/guide/src/configuration/response-streaming.md
[L5]: https://github.com/aws/aws-lambda-web-adapter/blob/2171d5a68bb5de323d5c7271676caf9674049cb2/docs/guide/src/features/non-http-events.md
[L6]: https://github.com/awslabs/aws-lambda-rust-runtime/blob/5670a2ed2efae728bf01b4de78297b670696d225/lambda-http/src/request.rs
[W1]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-runtimes.html
[W2]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions-env-vars.html
[W3]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-env-lifecycle.html
[W4]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-http.html
[W5]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-networking.html#web-functions-networking-protocols
[W6]: https://docs.aws.amazon.com/lambda/latest/dg/web-functions-revisions-timeout.html
