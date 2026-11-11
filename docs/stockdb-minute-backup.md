# 本地 StockDB 分钟补缺备源

StockDB 是本地历史分钟主源。盘后已收盘股票日默认按 StockDB -> Promax -> TDX 路由：先由本地 StockDB 读取并校验，只把缺失或校验失败的股票日交给在线备源。
实时行情、盘中分钟、ETF、指数与日K主源不变。

## 配置

```dotenv
STOCKDB_ENABLED=1
STOCKDB_BASE_URL=http://stockdb-api:7899
STOCKDB_MINUTE_FALLBACK_ENABLED=1
STOCKDB_AFTER_CLOSE_PRIMARY_ENABLED=1
STOCKDB_TIMEOUT_SECONDS=3
STOCKDB_MAX_STOCK_DAYS=50
STOCKDB_ONLINE_FALLBACK_SYMBOLS=5
STOCKDB_READINESS_SAMPLE_SIZE=10
STOCKDB_READINESS_MIN_COVERAGE=0.60
STOCKDB_PUBLICATION_RETRY_SECONDS=600
STOCKDB_AUTO_UPDATE_ENABLED=0
STOCKDB_UPDATE_TRIGGER=/work/update.trigger
```

必须先完成本地数据同步及样本核验，才开启两个开关。未配置时默认关闭，旧版 Promax/TDX 路由保持原样。
地址仅允许 loopback、明确的内网 IP、host.docker.internal 和 stockdb-api；不允许公网体验地址、URL 内密钥或重定向。
盘后优先只读取本地已有镜像，逐股票日通过校验后才交给落盘流程。目标交易日会先按本批股票抽样判断发布覆盖率；覆盖率不足且返回为空/不完整时标记为 `stockdb_snapshot_stale`，但不会阻塞已校验通过的本地股票日，也不会禁止缺失项进入有上限的 Promax、TDX 在线补缺。首次确认可用的观测时间保存在 `user_data/stockdb_freshness.json`。为避免少量异常拖慢全市场首轮，每批最多同步回退指定数量的股票到 Promax、TDX，其余失败项保留在队列并轮转重试；如果本地服务故障或数据格式错误，则直接使用在线链路。不要把盘中分钟主源改成 StockDB，它不代表实时或两年全历史权限。

## 契约

- 只通过 HTTP `cmd=get` 读取原始 `分钟k` 表，按单股/日期查询，追加 `json=1`，不用原生 SDK 默认的前复权处理。
- 入参/记录/key 三方核对六位代码、交易所、日期和小时，不将股票接口冒充 ETF 或指数接口。
- `date` 使用北京时间 YYYYMMDDhhmmss；分钟标准时间列为北京 naive datetime。
- 原始 `volume` 为股，显式除以 100 转为现有验证分钟库的手；`amount` 保持元，OHLC 保持不复权元价。
- 仅忽略零量零额的 13:00 午休占位；非零午休、非法时段、重复时间、缺失分钟、非法 OHLC 均拒收。
- 完整时段、逐分钟量额单位检查后，再走现有 `checked_additions` 的日K价格/量额与旧值冲突校验。
- 单个源请求失败立即终止本地批次并记录退避。每批最多尝试配置数量的股票日；未请求项明确记录，不伪装成空行情。
- 日常补缺不覆盖旧值；明确授权替换时仍走原有备份发布流程。源切换日志写入 `import_reports/minute_failover`，源冷却独立记录。

## 隔离试运行

采用作者发布的 Linux x64 包，先对照随发行提供的 SHA-256。二进制目录只读挂载，数据目录单独挂载，不提供生产目录、`.env`、Docker socket、用户主目录或其他密钥。
`docker-compose.stockdb.yml` 是可选 sidecar 配置，不改变默认启动服务。提供 `STOCKDB_DIST_HOST` 与 `STOCKDB_DATA_HOST`，只启动 `stockdb-api`。
端口只绑定本机 127.0.0.1:7899。容器内配置监听 0.0.0.0，但不得把宿主机端口开放到公网；建议配置数据库只读查询。

首次镜像下载不等于分钟数据已经可用。必须等该镜像文件全部同步成功，停止更新器后再启动读取服务；不要查询半套 LevelDB 快照。
历史镜像和当前镜像可独立试验，不要求为近期缺口先下载全部历史。镜像可能包含原始日K/复权等数据，并不保证本镜像内就有分钟表。

生产方式使用 A/B 双快照。读取服务始终访问当前快照，更新器只修改另一份快照；清单和服务启动校验通过后，计划交易日只要有至少一只有效交易股票达到240根常规分钟K，就先切换并发布 `serving_partial`，不再因为少数股票迟到而丢弃整份可用更新。缺失股票日保留在应用的逐日补差队列，管理器按原计划键和退避时间重试；全部符合预期后才标记 `data_ready=true` 和 `serving`。92xxxx、日K成交量与成交额均为0的停牌/无成交标的不进入 StockDB 完整性分母。更新器单轮超过 `STOCKDB_UPDATER_TIMEOUT_SECONDS` 会停止并进入退避，不会长期占住服务。默认关闭自动快照更新，避免为近期分钟缺口反复维护整套历史库。需要明确升级或修复完整快照时，在宿主机的 `STOCKDB_MANAGED_DATA_HOST` 目录创建 `update.trigger` 文件，管理器会消费一次并执行完整更新。设置 `STOCKDB_AUTO_UPDATE_ENABLED=1` 才恢复按 `STOCKDB_UPDATE_TIMES` 自动维护完整快照。`manager-status.json` 中 `data_ready`、`partial_update` 和 `data_readiness` 分别记录完整/部分发布状态与缺口；后续应用仍会逐股票日做日K价格、成交量和成交额校验，而不是只看更新器退出码为零。若日K清单本身不可读或完全没有有效交易标的，更新仍不会被伪判为成功。

Compose 默认启动 `stockdb-api` 管理服务，不再依赖可选 profile。设置 `STOCKDB_DIST_HOST`、`STOCKDB_MANAGED_DATA_HOST`、`STOCKDB_UPDATE_TIMES`，数据目录需预置至少一个完整的 `slots/a` 或 `slots/b`。管理状态写入 `manager-status.json`，更新日志写入 `updater-a.log` 或 `updater-b.log`。

## 只读核验

在隔离进程中运行 `scripts/probe_stockdb_minute.py`，提供 `--base-url --symbols --start --end --data-dir --output`。
`--data-dir` 为只读的本地日K参考，`--output` 必须在生产目录之外。该脚本只暂存校验合格的样本并保存诊断，绝不改写生产记录。
启动成功、HTTP 200、空数据库健康返回都不算通过；需实际返回分钟数据且对照日K通过。

## 回退

将 `STOCKDB_MINUTE_FALLBACK_ENABLED=0` 后重建/重启主应用，即回到原来的 Promax/TDX 链路。本地隔离数据可保留，不影响生产 Parquet、预测快照和既有历史成绩。
