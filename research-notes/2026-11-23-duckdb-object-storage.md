# DuckDB Object Storage：定位、基准口径与采用边界

核查日期：2026-11-23（Asia/Shanghai）。文章：[A Real DuckDB Database in S3, With Nothing Else to Run][article]，页面署期 2026-10-09；本轮读取现存页面，不把它视作发布当天的归档。
源码基线：`ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd`（2026-11-09）。文章、历史基准与当前代码分开核对，不将旧基准当作当前版本性能。
证据分级：作者自报＝文章或基准表；静态确认＝固定 SHA 的文档、配置与脚本文本；本轮推算＝对已公布数字做算术。未运行第三方程序、DuckDB/扩展、SQL、安装器或云资源，未读凭据、下载数据集；不声称复现或安全认证。

## 1. 文章定位与机制：保存数据库，不是换一种 Parquet 查询

`duckdb_object_storage` 是 SlateDB 支撑的 DuckDB 虚拟文件系统，逻辑数据库经 `duckdb_objfs://` 附加，后端可选内存、本地目录或 S3 兼容存储。[README L3–5][readme]
文章的机制概述是：保持 DuckDB 存储引擎与数据库逻辑格式，由文件系统适配把数据库文件的字节映射至 SlateDB；SlateDB/OpenDAL 是进程内依赖，不需另起独立存储守护进程。这是作者的架构描述，不等于全功能或故障语义已实测。[文章：机制与布局][article]
“原生数据库格式”不等于“桶里有一份原样 `.duckdb`”：URI 是逻辑文件名，桶、前缀和物理布局另由设置与 SlateDB 决定。当前读测也分别生成原生单文件和 ObjFS 多对象；stock DuckDB 不能把这些对象当普通数据库文件直接打开。[README L231–262][placement]、[基准 L177–179][remote-method]、[文章：格式边界][article]
因此既不是 `read_parquet('s3://…')`，也不是每次本地改完数据库后整文件上传。前者读取数据文件，后者是原生基准的交付路径；本扩展把可读写的逻辑数据库放在另一套物理存储布局上。[官方 HTTPFS][httpfs]、[基准 L260–265][write-method]
“没有额外服务”只指独立服务数量；并非没有元数据、后台工作或本地磁盘需求。README 把持久数据库与 WAL 路由到对象存储，spill、扩展文件、日志等仍按本地/各自路径处理；文章说明 compaction/GC 在 writer 内运行。[README L244–262][placement]、[文章：内嵌后台工作][article]

### 1.1 源码补充：文件、持久化与一致性

调用路径是 `DuckDB FileSystem → C++/C FFI → Rust → SlateDB → OpenDAL → 对象存储`。当前配置依赖 SlateDB 0.18，并启用 OpenDAL 的 fs、memory、S3 后端；不能因为 OpenDAL 库支持更多服务，就写成本扩展已交付 GCS/Azure/HDFS。[依赖与后端配置][cargo]

默认 chunk 为 256 KiB，起始偏移为 12 KiB，配合 DuckDB 默认块布局减少跨 chunk 的读改写。**一块对应一个 SlateDB KV 键，不等于一块对应一个 S3 对象或一次 PUT**；WAL、SST、批量与 compaction 决定物理请求。[布局常量][geometry]、[整块写入][chunk-write]

FileSync 沿 FFI 调到 sync，把该 handle 的脏 chunk、删除及文件 metadata 放进同一个 WriteBatch，然后 write、flush，成功后才清空脏状态。重命名则用 SerializableSnapshot 事务更新路径映射，而不是复制整个逻辑文件。这是静态实现核对，不是所有故障场景下的数据库恢复认证。[FileSync 桥接][file-sync]、[sync 实现][sync]、[重命名][rename]

**单写者是使用约束，不是“第二个写者立即被拒绝”。**当前没有扩展层文件锁；第二个 writer 可以取得新 epoch，使旧 writer 在后续 flush/checkpoint 才失败。fencing 是兜底，不是多写者事务协调；应在部署或外层调度保证一个有效 writer。[单写者合同][single-writer]

**新鲜度与一致性分开。**跨进程 read-only reader 按文档约每 10 秒追踪 manifest，DETACH/ATTACH 仍复用缓存 reader；同进程若已有 writer，读取可复用 writer。源码采用 ManagedCheckpoint 并沿用 DbReaderOptions 默认值，没有把“开始 SQL 事务时刷新到最新”接成通用强一致开关；10 秒也不能当作网络故障下的硬 SLA。[reader 合同][read-consistency]、[reader 缓存][reader-cache]、[reader 构造][reader-build]

AWS S3 对对象 PUT/DELETE 提供强读后写一致性；扩展的可见性延迟不应归咎于“S3 本身最终一致”。对象级一致性也不自动提供上层跨文件/数据库事务的保证。[AWS S3 一致性说明][s3-consistency]

**共享存储不等于分布式查询统一快照。**截至核查时另一个项目 Duckherder 的固定 README（`3598d5422575eb05e901b852c52d8be30d028622`）把全 worker 快照选择、传播与协调标为目标架构/WIP；已有分区聚合展示不能证明这一完整合同已经完成。本文没有把存储扩展的约束、索引支持外推为 Duckherder 所有分布式 SQL 都正确。[Duckherder 固定文档 L124–133][distributed-snapshot]

源码仍直接拒绝 Glob、ListFiles/DirectoryExists；默认扩展数据/metadata 缓存约 512/128 MiB，独立于 DuckDB buffer-manager memory limit，不能仅用 DuckDB 的限制代表整个进程 RSS。当前 CHANGELOG 已转向 DuckDB v2.0.0，而基准使用 v1.5.5，应重新测量再下性能结论。[目录操作][directory-ops]、[缓存合同][cache-budget]、[版本变化][changelog]

## 2. 读基准：旧版本、有限样本与明确分母

作者报告 DuckDB `v1.5.5`、Apple M4、10 核、16 GB RAM、默认 10 线程。SF10 本地读结果来自 `21bae82c606ecf9d06232709cad9ea60e18cea57`，SF1 远端 ObjFS 来自 `72a21ed3da7db95dd4cfd2fa1c840400fe31532d`，都不是本轮 HEAD；短 SHA 已通过 GitHub commit API 展开。[基准 L10–18][read-method]、[21bae82][old-local]、[72a21ed][old-remote]
以下为 Q1–Q22 的作者表格；比值＝ObjFS 查询延迟中位数 / Native 中位数，越小越好。“±”是样本标准差，不是置信区间；脚本也使用 `statistics.median` 与 `statistics.stdev`。[基准 L16–18][read-method]、[统计 L483–487][read-stat]

| 场景 | 规模与重复 | 已公布逐查询比值 | 按表中中位数计数 |
|---|---|---|---|
| 内存 | SF10；每查询 1 次预热＋5 次测量 | 0.69–1.14× | ObjFS 20/22 更快；不是 S3 性能。[表 L22–45][memory-table] |
| 本地文件系统 | SF10；每查询 1 次预热＋5 次测量 | 0.93–1.40× | 7/22 更快，15/22 更慢；Q5 为 1.40×。[表 L49–77][local-table] |
| S3 | SF1；每查询 3 个新进程，无预热 | 0.92–1.58× | 仅 Q2 更快，其余 21/22 更慢。[表 L79–113][s3-table] |

文章概括为 0.92×、1.08×、1.20×，但未写总体聚合公式。按公布中位数的 22 个比值计算等权几何均值约为 0.919、1.080、1.203，与概括相符；这是本轮推算，不是取得原始样本后的作者统计。不能把它写成所有查询总耗时比或每条查询的保证。[文章：性能自报][article]、[三组表][memory-table] [本地][local-table] [S3][s3-table]
SF1 的脚本预期 `orders` 1,500,000 行、`lineitem` 6,001,215 行；SF10 分别为 15,000,000、59,986,052 行。SF 是数据生成规模参数，不是数据库实际占用 GB；SF10 与 SF1 不能直接横比。[脚本 L1063–1074][cardinality]、[基准 L113][s3-table]
S3 配置为台北客户端、同一 `ap-east-2` 桶；ObjFS 数据缓存 512 MiB、元数据缓存 128 MiB、持久缓存关闭，Native HTTPFS 外部文件缓存开启且新进程初始为空。“冷进程”不是全程禁用缓存，ATTACH 发生在查询计时之前。[基准 L182–189][s3-settings]、[脚本 L297–334][remote-query]
只取 DuckDB profiler 的查询 `latency`，不包含进程启动、认证、ATTACH、数据生成/上传。两种后端顺序执行而非随机交错；不能由 3 个样本证明统计显著性，亦不能外推同区客户端、长期热缓存、TB 级或 p95/p99。[脚本 L73–81][latency]、[基准 L81–84][s3-table]、[脚本 L297–334][remote-query]
读脚本将查询输出丢弃，基数校验只核对 `orders` 与 `lineitem` 行数；脚本本身未显式做逐查询结果对比。缓存/I/O 统计包含 ATTACH 与查询，而延迟只含查询；OpenDAL payload 字节也不等于 S3 线速流量或完整账单。[脚本 L125–129、L190–197](https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_read_benchmarks.py#L125-L197)、[统计说明 L751–765][io-stat]

## 3. 写基准：交付吞吐，不是随机写或事务提交延迟

| 作者自报场景 | Native 中位数 ± SD（行/秒） | ObjFS 中位数 ± SD（行/秒） | ObjFS/Native |
|---|---:|---:|---:|
| 本地 SF10 `lineitem`，每后端 3 个新库 | 8,014,063 ± 747,845 | 1,551,799 ± 813,842 | 0.19×。[基准 L119–130][write-table] |
| S3 交付 SF1 `lineitem`，每后端 3 个新库 | 188,642 ± 629（本地写＋上传） | 180,076 ± 8,813（直接写 S3） | 0.95×。[基准 L134–146][write-table] |

本地分母是 `CREATE TABLE AS SELECT`＋`CHECKPOINT` 的 profiler 时间；S3 Native 再加 AWS CLI 上传墙钟时间，ObjFS 不加独立上传步骤。吞吐＝行数 / 该路径耗时，越大越好；故 0.95× 约为吞吐低 4.5%，不应按读延迟方向解释。[脚本 L103–170][write-script]
脚本为各后端分别生成同 SF 的内存源，随后写入新库，Native 总是先执行；不是两后端共享一个内存表对象。源生成、数据库附加、启动及复核不计时；重开后仅比 `count(*)`、`sum(l_orderkey)`、`sum(l_quantity)`，不是逐行校验。[脚本 L46–56](https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_write_benchmarks.py#L46-L56)、[L96–170][write-script]、[L434–439](https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_write_benchmarks.py#L434-L439)
0.19× 与 0.95× 只能说明这些批量落库/交付路径。它们没有测单行 UPDATE、逐事务 fsync 延迟、并发客户端或持续写入时的长期 compaction 成本；本地 ObjFS 的 SD 813,842 也提示波动较大。[基准写表][write-table]、[计时实现][write-script]
文档没有为写表单独标注精确提交 SHA；不能顺带把它归到上述两个读测提交或当前 HEAD。当前读脚本默认仅 3 次测量，不能以默认参数重现表中的本地 5 次；原始 `.cache` 与 HTML 报告被 Git 忽略，本轮没有取得绑定结果的完整样本。[写表与方法][write-table] [方法][write-method]、[读脚本默认值 L1063][cardinality]、[.gitignore L1–3](https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/.gitignore#L1-L3)

## 4. 测试与 skip-list：有清单，不等于“所有 DuckDB 功能通过”

- 作者说运行数千份 SQLLogicTest，by-design 约 45、unsupported 10、extension bugs 无条目；这是作者自报，不是本轮执行所得。[文章：测试自报][article]
- HEAD 静态逐路径计数为 by-design **49**、unsupported **10**；前者含 catalog 名称/额外库、存储版本、checkpoint 指向、低内存假设及扩展自配置等多种原因，不能全压成“都只是 memory 名称不同”。数量是文件路径条目，不是 SQL 断言数。[by-design L3–82][by-design]、[unsupported L3–18][unsupported]
- Unsupported 的 10 项针对未由 `duckdb_objfs://` 写入的现有数据库 fixtures；不是任意现成 `.duckdb` 可自动导入。bugs JSON 只有说明、没有 skip 条目；空表不能证明未知缺陷为零。[unsupported L5–16][unsupported]、[bugs L1–3][bugs]
- 默认/磁盘配置均 `skip_compiled: true`；内存配置额外 `on_load: "skip"`。这些共享 skip-list 的 SQLLogicTest 适配配置不能代表 compiled tests、reload、真实 AWS 或全部扩展均获同等覆盖。[磁盘配置 L2–14][disk-config]、[内存配置 L2–15][memory-config]
- 有针对附加 catalog 的补测，涉及 VARIANT/latest storage、generated columns、sequence、macro、ART 数据与重开；CSV/Parquet 用例通过本地文件导入，Parquet 用例还依赖 `require parquet`。只能静态确认用例存在，不能代替通过日志。[兼容性用例 L10–205][compat]、[CSV L13–24](https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/sql/duckdb_object_storage_data_files_csv.test#L13-L24)、[Parquet L5–27][parquet-test]
- MinIO E2E 配方只写两行、checkpoint、重开只读并验和；不是 AWS 性能或系统性故障验证。`glob('duckdb_objfs://…')` 的用例明确预期“不支持”错误。[S3 配方 L21–30][s3-test]、[glob 用例 L7–11][glob-test]
- Makefile 有 skipped/survey/lint 入口，但都引用 `scripts/duckdb_test_skips.py`，该文件不在 HEAD 的受版本控制 scripts 清单中；本轮因此不能确认文章所称“跳过项一旦通过即自动报错”的入口在此 checkout 可用，也没有完整通过率分母。[Makefile L80–95][make-skips]、[固定 SHA scripts 目录](https://github.com/dentiny/duckdb-object-storage/tree/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts)

## 5. 原生 HTTPFS、本扩展与 lakehouse：比较层次不同

HTTPFS 是远程文件访问层：普通 HTTP(S) 只读；S3 API 可读、写和 glob 数据文件。原生 DuckDB 本来就可经 HTTP/S3 只读 ATTACH 单个远程数据库。因此不能说“DuckDB 原本只能查 Parquet”或“HTTPFS 完全不能写 S3”；缺口是这里的可变数据库存储路径。[官方 HTTPFS][httpfs]、[官方 ATTACH 的远端数据库语法][attach]
本扩展是数据库文件系统层，目标是保存 DuckDB 数据库对象与持久状态，不提供开放表格式互操作。Iceberg 是表格式/元数据提交层，以不可变数据文件、manifest 和快照组织表；DuckLake 则用 SQL catalog 记录元数据，数据文件采用 Parquet。这不是三个可以仅凭同一“查 S3”标签判胜负的替代品。[README][readme]、[文章：格式边界][article]、[Iceberg Spec][iceberg]、[DuckLake catalog][ducklake]、[DuckLake 数据文件规格][ducklake-data]
“Iceberg/DuckLake 必须另跑 catalog 服务”不准确：PyIceberg SQL catalog 官方支持本地 SQLite，但明确限探索/开发，不能据此承诺生产并发；DuckLake 官方支持本地 DuckDB catalog 的单客户端模式，也列出 SQLite/ PostgreSQL 选择。有 catalog 元数据不等于必须有独立网络服务，实际部署与并发合同要按实现判断。[PyIceberg SQL catalog][pyiceberg]、[DuckLake catalog 选型][ducklake]
采用判断：已有只读远端数据库或数据文件，先区分 HTTPFS 能否满足；需要 DuckDB 原生数据库语义再评估本扩展；需要开放表格式共享与其快照/提交模型，则按具体引擎、catalog 和格式版本评估 lakehouse。社区扩展分发与当前源码 SHA 也不能混同：官方说明二进制绑定 DuckDB 版本及平台，签名检查完整性，不是本轮安全认证。[官方 HTTPFS][httpfs]、[官方扩展分发][distribution]、[Iceberg Spec][iceberg]

## 6. 待实测清单与已读一手来源

- 固定扩展、DuckDB 子模块、HTTPFS/TPCH 的 SHA/版本及平台；索取旧读测和写测原始 JSON、完整命令、网络条件，记录真实存储字节、请求及源数据校验。HEAD 子模块配置指向 `v2.0-cyanoptera` 分支，而基准脚本硬校验 `v1.5.5`，需另行对齐，不能只抄 build 的提示文本。[.gitmodules L1–4](https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/.gitmodules#L1-L4)、[脚本 L984–1007](https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_read_benchmarks.py#L984-L1007)
- 同 SF 做随机交错、多重复的冷/热读测，把查询、ATTACH、启动与完整交付分别计时；报告逐查询结果校验、分位数及缓存条件，不用 SF1 冷读与 SF10 热读直接推导规模效应。[现有口径][s3-table]、[现有查询循环][remote-query]
- 分离批量 CTAS、单行写、事务提交和长期写入，测 compaction 后的成本；以完整数据与重开验证约束/索引及业务关键扩展，补充真实 S3 与故障场景，先解决测试入口缺失并公布全部 skip/失败分母。[写入范围][write-script]、[测试入口][make-skips]、[兼容性用例][compat]

采用判断：值得作为“保留 DuckDB 数据库语义、减少独立服务”的单写多读方案做小规模研究；不直接视为多写者 OLTP、开放湖表格式或成熟分布式数据库。若需求只是读远端静态数据库/Parquet，先检查原生 HTTPFS 是否已足够。这是基于上述边界的建议，不是生产可用性认证。

另需验收第二个 writer 接管、writer 崩溃恢复、跨进程读后写、长查询中版本变化、完整备份还原与凭据轮换；如使用 Duckherder，还需验证 all-worker 同快照及分区结果合并。以上均未执行，不能用单机读表代替。本轮只保存研究笔记，未部署、未改 Wiki、未提交或推送。

下列为本轮实际读取的一手资料索引；GitHub 行号是原文件行号，不是抓取器的页面编号。比较文章版本时使用 DuckDB 1.5 文档，当前官方文档另已转向 2.0；文章、DuckLake、Iceberg/PyIceberg 在线页面仍可能更新。

[article]: https://dentiny.github.io/posts/duckdb-on-object-storage/
[readme]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/README.md#L3-L5
[placement]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/README.md#L231-L262
[read-method]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/benchmark.md#L10-L18
[memory-table]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/benchmark.md#L22-L45
[local-table]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/benchmark.md#L49-L77
[s3-table]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/benchmark.md#L79-L113
[remote-method]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/benchmark.md#L177-L179
[s3-settings]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/benchmark.md#L182-L189
[write-table]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/benchmark.md#L119-L146
[write-method]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/benchmark.md#L230-L265
[old-local]: https://github.com/dentiny/duckdb-object-storage/commit/21bae82c606ecf9d06232709cad9ea60e18cea57
[old-remote]: https://github.com/dentiny/duckdb-object-storage/commit/72a21ed3da7db95dd4cfd2fa1c840400fe31532d
[latency]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_read_benchmarks.py#L73-L81
[remote-query]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_read_benchmarks.py#L297-L334
[read-stat]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_read_benchmarks.py#L483-L487
[cardinality]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_read_benchmarks.py#L1063-L1074
[io-stat]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_read_benchmarks.py#L751-L765
[write-script]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/scripts/run_write_benchmarks.py#L96-L170
[by-design]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/configs/skip/by_design.json#L3-L82
[unsupported]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/configs/skip/unsupported_features.json#L3-L18
[bugs]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/configs/skip/extension_bugs.json#L1-L3
[disk-config]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/configs/duckdb_slatedb_disk.json#L2-L14
[memory-config]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/configs/duckdb_slatedb_memory.json#L2-L15
[compat]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/sql/duckdb_object_storage_duckdb_compatibility.test#L10-L205
[parquet-test]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/sql/duckdb_object_storage_data_files_parquet.test#L5-L27
[s3-test]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/s3/duckdb_objfs_e2e.sql.in#L21-L30
[glob-test]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/test/sql/duckdb_object_storage.test#L7-L11
[make-skips]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/Makefile#L80-L95
[httpfs]: https://duckdb.org/docs/1.5/core_extensions/httpfs/overview.html
[attach]: https://duckdb.org/docs/1.5/sql/statements/attach.html#attach-syntax
[distribution]: https://duckdb.org/docs/1.5/extensions/extension_distribution.html
[ducklake]: https://ducklake.select/docs/stable/duckdb/usage/choosing_a_catalog_database
[ducklake-data]: https://ducklake.select/docs/stable/specification/tables/ducklake_data_file
[iceberg]: https://iceberg.apache.org/spec/
[pyiceberg]: https://py.iceberg.apache.org/configuration/#sql-catalog
[cargo]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/rust/Cargo.toml#L18-L31
[geometry]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/rust/src/file_metadata.rs#L8-L23
[chunk-write]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/rust/src/chunk_manager.rs#L105-L130
[file-sync]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/src/slatedb_file_system.cpp#L268-L271
[sync]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/rust/src/file_handle.rs#L253-L272
[rename]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/rust/src/database_metadata.rs#L71-L111
[single-writer]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/README.md#L285-L298
[read-consistency]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/README.md#L272-L283
[reader-cache]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/rust/src/fs.rs#L328-L347
[reader-build]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/rust/src/fs.rs#L414-L432
[s3-consistency]: https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel
[distributed-snapshot]: https://github.com/dentiny/duckdb-distributed-execution/blob/3598d5422575eb05e901b852c52d8be30d028622/README.md#L124-L133
[directory-ops]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/src/slatedb_file_system.cpp#L306-L323
[cache-budget]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/README.md#L100-L115
[changelog]: https://github.com/dentiny/duckdb-object-storage/blob/ebbdcd33d2bf4dc317e4b2f2aaa9f077f0a4d3cd/CHANGELOG.md#L1-L17
