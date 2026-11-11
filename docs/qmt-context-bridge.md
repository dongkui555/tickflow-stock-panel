# 国信 iQuant 实时数据桥

系统使用 iQuant 官方策略运行时注入的 `ContextInfo` 获取盘中数据。外部
`xtquant.xtdata` 未通过多标的路由校验时不会参与行情回退，避免把缓存值误当
成实时行情。

## 组件

- `tools/qmt_bridge/qmt_http_api.py`：本机 HTTP 服务和 SQLite 推送缓存。
- `tools/qmt_bridge/iquant_context_collector.py`：iQuant 内运行的只读策略。
- `backend/app/plugins/qmt/`：TickFlow 数据源适配器。

默认缓存文件为 `tools/qmt_bridge/data/qmt_push.sqlite3`。其中保存最新快照、
当日及近期历史分钟、当日日线和 09:15-09:25 集合竞价快照。

历史分钟下载由同一策略中的定时任务执行，不放在 `handlebar` 的实时回调
里。任务按沪深北 A 股股票池分批调用 `ContextInfo.get_market_data_ex`，默认
获取最近 5 个自然日、每批 100 只股票，每 60 秒最多处理 2 批。桥接端只接收
连续交易时段内且 OHLC 关系正确的记录，并以 `symbol + bar_time` 去重更新。
工作日从 15:05 后运行，周六和周日全天运行。
同一策略还会独立读取 `/api/index/universe` 返回的本地指数维表，按每批
50 只、最近 30 个自然日同步指数日 K 到桥接缓存；代码、日期、OHLCV 校验
通过后才写入，取不到的指数继续由 Tushare/Promax 处理。

## 启动

1. 执行 `tools/qmt_bridge/start_qmt_bridge.ps1`。
2. 在 iQuant 的“策略开发”中新建 Python 策略，使用
   `D:\国信iQuant策略交易平台\python\TickFlow_QMT_Data_Bridge.py`，周期设为
   1 分钟。
3. 将策略加入“策略交易”，选择模拟模式并启动。该策略不调用交易或账户 API。
4. 访问 `http://127.0.0.1:8188/api/health`，确认 `push_active=true`；盘中还应看到
   `realtime=true`，有五档字段时 `depth5=true`，形成分钟后 `minute=true`。
5. 更新策略代码后，必须在 QMT 中停止旧策略、重新加载/保存，再重新加入策略交易；
   已运行的旧策略不会自动读取新文件。历史分钟任务的批次进度会写入策略日志，
   桥接端可通过 `capabilities` 查看 `push_minute_latest`、`push_minute_rows` 和
   `push_minute_sync`。

## 路由

- 实时和五档：当前主源 -> QMT 本地缓存 -> 公开 L1 备源。
- 盘中分钟：QMT 可作为当前交易日增量源。
- 近期历史分钟：QMT 认证策略按批次补齐；更长历史仍为 StockDB -> Promax -> TDX，
  不使用当日 QMT 缓存冒充多年历史全量。
- 日K、财务、资金流、历史竞价：Tushare/Promax 为主。
- 当日竞价：Promax 空缺时使用 QMT 在 09:15-09:25 捕获的本地快照。
- 核心指数：主源失败后，仅在 QMT 日线能力通过校验时回退。

最新行情超过 15 分钟自动失效。快照推送不等于逐笔成交；逐笔委托、撤单和
成交必须在确认沪深 L2 权限后使用独立 L2 通道，系统不会虚报该能力。
