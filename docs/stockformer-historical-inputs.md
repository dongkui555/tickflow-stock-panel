# StockFormer历史引用研究输入

这是独立研究输入版本，当前生产模型及95a568/717b4d3d冻结账户仍使用其各自捕获源码。原冻结模型不能直接接受修正后的特征，也不能直接改预期源码指纹。

## 输入

`stockformer_pit_inputs.prepare_panel(panel, raw_prices, names, industries, limits)` 返回独立面板及质量报告。

- `panel`：原调整价技术面板，证券/日期唯一；保留完整价格时序。
- `raw_prices`：按标准仓库读取的原始OHLC及成交量，证券/日期唯一。
- `names`：曾用名的公告日、生效起止日；或显式可用日期、来源、有效期的标准引用。历史快照仅在其确切交易日使用。
- `industries`：行业、有效期、可用日期及来源。最新快照只有实际捕获日之后的有效性，不回填过去。不同分类体系须另立研究数据版本。
- `limits`：证券、交易日、真实当日涨跌停价；缺失、0占位和不合法范围保留未知，不猜10%或把缺失当无涨跌幅限制。

涨跌停/炸板基于原始价与当日限价，连板/连跌重新计算；缺失引用或交易日缺口中断计数。未知证券状态不按普通股处理。`build_features` 在完整性不足时拒绝生成可训练特征，避免零填充和缺历史导致滚动特征或行业分组失真。

`daily_reference_history` 支持供应商按交易日返回的历史名称/行业；这是有效日快照，仍须披露事后查询不能独立证明原始档案未被重述。当前捕获的备用列表行业标记 `tushare_bak_basic_daily_v1`，不与同花顺或申万分类混用。

## 研究调用

先准备新目录，不覆盖任何运行账户：

```bash
PYTHONPATH=backend python scripts/prepare_stockformer_pit_inputs.py \
  --panel panel.parquet --raw-prices raw-prices.parquet \
  --names names.parquet --industries industries.parquet \
  --limits limits.parquet --output research-new-version
```

CLI拒绝已有输出目录。报告 `training_ready=false` 时只能检查缺口，不能直接进入训练。通过后重新生成目标标签，不能混用旧行业标签缓存。

信号研究显式调用 `_build_gated_signal_frame(..., historical_identity=True)`，输入必须包含有效的历史名称和质量标记；默认路径保持旧行为。回放显式调用 `replay_paper(..., limit_references=limits)`，创建账户前检查全部需用限价，复用原撮合器，限价适配在退出或异常后恢复。该适配只在隔离研究worker使用，不从主应用请求调用。

## 历史来源完整性

`stockformer_pit_sources.fetch_reference_day` 接收现有provider的API客户端和当日实际股票池。分页请求每页最多1000行，拒绝重复页、错误日期、冲突身份、分页中换源或股票池覆盖不足。429的冷却从异常链中的响应读取，调用方保存断点并停止，不能把短页当完整数据。

显式研究选项 `native_first=True` 先尝试不带分页参数的完整日查询；只有实际股票池全部覆盖才返回成功。若供应商截断响应，按实际返回行数继续分页，重复页和缺口仍被拒绝。默认分页路径不变。`ReferenceSourceError` 只暴露安全错误类别、HTTP状态及可重试属性，避免将带有凭据的供应商异常写入报告。

2026-10-22修复中发现此前研究抓取的4秒超时短于实测完整日响应，45秒超时及固定主源读取已取得完整日数据；该设置仅用于独立采集进程。采集任务逐条校验证券池，原子保存源记录和断点，遇限流退出当前批次并保存冷却，后台有总时限和重试上限。部分成功不会标为全历史可训练。

六个历史日期的5,743条日K因带非空 `quote_ts` 被正式收盘读取器排除。它们实际均为收盘后快照；经独立历史日线核对，价格全部一致。3,529条量额严格匹配，其余2,214条另经扶摇历史日线确认。仅在独立研究价格视图中追加经过核验的正式日线，共649,692行；未清空生产原始行的时间戳，也未把时间戳本身当作收盘完整性的证明。源捕获、对照及剩余历史引用情况见 `output/historical-reference-repair-20261022/README.md`。

首次修复证据见 `output/point-in-time-repair-20261021/README.md`。当前研究代码及缺价修复验收通过，历史名称、行业、限价仍在补齐；全历史数据验收尚未通过，不得激活生产。历史快照的原始档案版本仍未被独立证明，不能把这批研究数据宣称为严格PIT盈利证据。

## 已公告退市整理期与标签重建

研究版本 `known_name_industry_authoritative_limits_lifecycle_v2` 增加可选 `lifecycle_events=`，要求状态、公告可用日、有效区间与来源；不根据简称或当前D状态推断历史。`attach_lifecycle_exclusions` 保留真实行情行，以显式标记排除整理期普通股研究决策及行业上下文；这些行的未知行业/限价仍为空，其余普通股上下文继续严格验收。

`announced_liquidation_events` 仅接受曾用名来源明确注明“退市整理期”且有公告日、生效日的记录；缺公告或“其他”原因不会自动转成排除。供应商档案版本限制仍需披露。公司公告优先保留，所有应用区间均以当日已知信息为准。

`build_labels(panel, raw_prices, execution=..., horizons=(5,), price_panel=...)` 复用真实模拟费用目标函数：收盘信号的下一实际交易日开盘入场，五交易日后开盘退出；数量使用信号日原始收盘价，未来开盘只用于标签。行业基准在分钟/决策筛选前形成，之后执行原240分钟和决策股票池规则。`price_panel` 可提供更长的价格/交易日序列以形成未来标签，不扩充历史特征上下文。

完整前缀的诊断标签和全范围标签分别报告。整手预算不足、未来价格缺失或未成熟均保留空值；条件标签不能替代涨跌停、订单竞争和现金路径下的组合撮合。全量引用未齐时仍不标记全量训练可用。证据与有限后台采集见 `output/lifecycle-reference-repair-20261022/README.md`。

## 2026-10-22 历史成交时序修复补充

独立研究输入修复见`output/traded-price-repair-20261022/README.md`。对四只证券的完整历史经两源核验后补入1,017个真实成交股票日，保留原源文件；复权使用现有pipeline函数并逐行对照，换手率使用同日两源一致值。分钟可用性沿用240根/覆盖1，缺失保持缺失。

研究materialize脚本现在接受显式`--output-root`、`--reference-root`、`--panel`、`--raw`、`--price-report`、`--supporting-history`、`--events`和`--execution-state-evidence`。这些是离线证据流水线参数，未增加生产训练入口。默认仍处理原捕获输入；修正版保存在单独作用域，并用input-config.json固定价格、面板和引用文件指纹。成熟成交价缺口在全范围就绪前须有真实交易状态归因，未归因时不放行训练。

首个修正前缀为132交易日，有效成本后行业超额标签94,013条。它仍是诊断数据；全674日引用采集及供应商历史原始档案版本证明尚未完成，未训练或激活新模型。
