# Canonical Security Identity Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将系统内股票、指数、ETF及其补差任务统一为带市场后缀的完整证券身份，安全迁移已有裸代码数据，并让后续同步、校验、查询、前端展示都不再依赖六位代码猜测市场。

**Architecture:** 以 asset_type + exchange + code + date 为唯一身份边界。symbol 统一保存为 代码.市场，供应商请求参数只在 provider adapter 边界按供应商要求转换。所有写入先经过身份解析与校验；无法由维表、历史代码映射、源返回市场和请求身份共同确认的记录进入隔离区，不伪造市场。数据迁移采用只读预检、备份、分批临时写入、原子替换、迁移日志和可回滚版本闸门。增量队列、授权审计和前端 API 同步使用同一身份模型。

**Tech Stack:** Python 3.11+, FastAPI, SQLite, Polars, DuckDB, Parquet/PyArrow, React/TypeScript, Docker Compose, uv, pnpm, pytest。业务数据处理使用 Polars，不新增依赖。

**Spec:** docs/superpowers/specs/2026-09-27-canonical-security-identity-design.md

## Global Constraints

- 不直接覆盖用户已有数据；迁移前必须创建可验证的备份，并在每个数据集记录文件清单、行数、哈希和迁移版本。
- 禁止仅凭六位代码前缀推断市场作为运行时最终结论；只有显式后缀、证券维表、日期有效的历史代码映射、源返回市场彼此一致时才可入库。
- 000001.SH、000001.SZ、000001.BJ 必须始终是不同身份；股票、指数、ETF必须分开解析和存储。
- symbol 是内部关联字段，必须是规范格式；供应商所需的裸代码只能在 provider adapter 出口生成，不能回写内部数据。
- 新增、重复、校验通过的 239/240 根分钟数据自动处理；只有新出现且未授权的冲突替换需要确认。身份未知不得进入最终确认或高置信度训练。
- 已发布预测快照不可被之后的修复数据改写；预测与验证必须保存当时使用的数据版本和身份键。
- 保留当前分钟缺口补差和历史审计任务的已落盘结果；本迁移只修复身份键与关联关系，不以全量重算代替身份修复。
- 迁移过程支持 --dry-run，默认只读；正式迁移必须显式传入 --apply，失败时保留临时目录与日志，禁止半成品替换正式目录。
- 本计划不修改 OHLC、量价单位、复权、交易策略或历史预测成绩。239/240 根的合格判定仍须满足现有时间轴和量价规则，不是按数量无条件通过。
- 活动系统不保留裸代码查询兜底、不双写两种身份格式。旧格式仅由迁移器、显式用户搜索解析器和不可变产物旁路映射读取。
- AAPL 等海外 ticker 不适用 A 股规则。仅迁移新闻的可确认证券关联，不丢弃正文或强加 SH/SZ/BJ。
- 不清空队列、不扩大替换授权、不提交或推送代码；保留工作区已有改动。

## Review Focus

- 是否仍有任何生产写入路径把 symbol 变成裸代码。
- 是否仍有任何查询、缓存、队列或权限判断只按六位代码关联。
- 代码变更、换市场、停牌、退市、未上市日期是否能按有效日期解析。
- 冲突数据是否被误当成可自动覆盖；重复数据是否只在完整身份相同且字段一致时折叠。
- 迁移后模型、回测、推荐、资金、竞价、财务、新闻关联是否仍能按 asset_type|exchange|code|date 取数。

---

## Shared Interfaces And Test Cycle

以下是计划新增/修改的契约，不表示现在已经存在。每项先新增失败测试并运行，随后实现，再运行相同测试及列出的现有回归；每项完成后更新清单。

- 在 security_identity.py 复用 SecurityIdentity，新增只读 property `security_key: str`；`identity_key(symbol: object, when: date | datetime | str, *, asset_type: str) -> str` 只接受严格身份，日期必须有效。
- 新增 `IdentityIndex`，按资产、市场、代码及有效区间索引证据；`build_identity_index(frames: Iterable[pl.DataFrame]) -> IdentityIndex` 不再使用 code-only 字典。
- 新增 `resolve_identity(value: object, *, asset_type: str, index: IdentityIndex, when: date | None, exchange: str | None = None, legacy: bool = False) -> SecurityIdentity`。legacy 仅允许专用迁移/搜索入口使用；匹配不唯一抛 IdentityError，不做前缀猜测。历史供应商别名按来源和有效期核验。
- 新增 `IdentityBatch(accepted: pl.DataFrame, quarantined: pl.DataFrame)` 与 `canonicalize_identity_frame(frame: pl.DataFrame, *, asset_type: str, index: IdentityIndex, time_column: str | None, expected: tuple[SecurityIdentity, ...] = (), legacy: bool = False) -> IdentityBatch`。按唯一身份组合批量解析再 join，禁止逐分钟行做 Python 查表；隔离行携带原行号、原值和原因。
- 修改 validate_frame_identity 为活动写入门禁：任何身份错误抛 IdentityBatchError；成功返回规范帧，不再返回混合格式原帧。需要部分接受的导入调用 IdentityBatch 接口，先保存隔离报告再发布 accepted。
- 新增 `backend/app/services/identity_migration.py`：`inventory(root: Path) -> MigrationManifest`、`plan_migration(root: Path, *, datasets: tuple[str, ...]) -> MigrationPlan`、`apply_migration(root: Path, plan: MigrationPlan, *, batch_days: int) -> MigrationResult`、`resume_migration(root: Path, migration_id: str) -> MigrationResult`、`rollback_migration(root: Path, migration_id: str) -> MigrationResult`。这些结构使用 dataclass，manifest 固定包含 schema_version、输入哈希、备份哈希、行数、非身份字段哈希、隔离/去重计数、文件阶段和磁盘预算。
- 新增 `backend/app/services/identity_schema.py`：`require_identity_version(root: Path) -> None` 与 `migration_lock(root: Path)` 上下文管理器。缺少版本标记为 legacy；新代码在生产目录版本不匹配时拒绝启动写入者，空目录可初始化新版本。锁文件存在且校验未完成时，任何 worker 不得恢复写入。
- 新增测试模块的命令统一从 backend 执行：`uv run --frozen --extra dev pytest tests/<test_file>.py -q`；首次应失败于目标断言，不能把导入环境错误当作有效失败。实现后全部断言通过才勾选。

## Task 1: Harden the canonical identity contract

- [ ] 完善 backend/app/services/security_identity.py：定义不可变的 SecurityIdentity、security_key、stock_day_key、严格解析和日期有效解析接口。
- [ ] 将“前缀推断”降级为迁移诊断信息，不能在 require_explicit=True 的运行时路径返回身份；显式后缀和维表/历史映射冲突时返回结构化拒绝原因。
- [ ] 增加 canonicalize_identity_frame()：统一生成 asset_type、exchange、code、symbol 和可选 security_key，并返回 accepted、quarantined、reasons 统计。
- [ ] 明确指数、ETF、股票的交易所和代码规则，支持 SH/SZ/BJ，不让指数或ETF被股票前缀规则吞掉。
- [ ] 增加单元测试：同代码不同市场、92xxxx、历史换码、未知市场、资产类型冲突、日期前后身份变化、显式后缀与源返回冲突。
- [ ] 先写并运行 test_no_market_guess（000001 裸码活动请求被拒绝）、test_asset_market_collision（stock|SZ|000001 != index|SH|000001）、test_unknown_asset_rejected、test_bj_explicit_market、test_source_target_conflict、test_lifecycle_alias_effective_date；实现 Shared Interfaces 后再次运行。test_overseas_not_coerced 断言 AAPL 不被转换为 A股。

Files: backend/app/services/security_identity.py, backend/tests/test_security_identity.py.

## Task 2: Make dimensions and provider boundaries canonical

- [ ] 修改 backend/app/services/instrument_sync.py，写入维表前强制规范化 symbol，并把无法确认身份的记录写入隔离报告而不是默认为某市场。
- [ ] 检查 backend/app/services/security_lifecycle.py 的日期有效映射：优先使用历史映射和维表状态，处理改名、换码、停牌、退市、未上市；只在存在唯一历史证据时解析旧代码。
- [ ] 修改日K、分钟K、资金流、集合竞价、财务和新闻关联的 provider adapter：请求参数可转换为裸代码，但返回帧入库前必须通过统一身份门；源返回的代码和市场必须与请求身份一致。
- [ ] 保留宏观新闻的独立新闻身份；对关联股票的新闻只保存规范化关联表，不能把逗号分隔的裸代码直接作为证券主键。
- [ ] 增加 provider contract 测试，覆盖 Promax/Tushare、QMT、StockDB、TDX、雪球和离线文件的请求/返回身份一致性，以及源不可用、空响应、部分字段缺失的拒绝原因。

Files: 修改 backend/app/services/instrument_sync.py、security_lifecycle.py、kline_sync.py、tushare_ext_sync.py、financial_sync.py、news_sync.py；修改 backend/app/data_providers/schemas.py、normalizer.py、tickflow_provider.py、custom/mapper.py，以及 backend/app/plugins/{tushare,qmt,stockdb,tdx,xueqiu,fuyao,stocksdk,aiagents}/provider.py。新增 backend/tests/test_provider_canonical_identity.py。

- [ ] test_provider_roundtrip_identity 参数化各源：供应商格式仅在请求边界出现，落盘 symbol 必为完整身份；test_mismatched_return_rejected 拒绝源返回其他市场；test_macro_news_preserved 和 test_financial_versions_not_merged 保留新闻正文及财报公告/修订版本。
- [ ] 运行新测试，以及 test_security_lifecycle.py、test_minute_routing.py、test_promax_minute_contract.py、test_minute_timezone_contract.py、test_financial_incremental.py，保留既有时区和量价规则。

## Task 3: Canonicalize repositories, joins, caches and model inputs

- [ ] 修改 backend/app/tickflow/repository.py 及日K、分钟K、资金、竞价、财务、新闻数据访问层：内部过滤、去重和连接全部使用完整身份键；禁止裸代码作为跨数据集 join key。
- [ ] 活动仓库发现裸 symbol 拒绝读写并输出身份版本错误；历史裸码只进入 Task 5 迁移器，不在在线查询中扫描维表或剥后缀兜底。
- [ ] 更新指标、因子、预测、回测和模型特征构造使用 asset_type|exchange|code|date，避免 000001.SH 与 000001.SZ 串联。
- [ ] 更新缓存键、API 响应和导出字段，保证用户看到名称和 代码.市场，同时保留 provider_code 仅用于诊断。
- [ ] 增加跨数据集 join 测试和回归测试，验证股票、指数、ETF相同数字代码不串数据。

Files: 修改 backend/app/tickflow/repository.py、backend/app/services/minute_repair.py、minute_reference.py、minute_cross_source.py、minute_features.py、stockformer.py、stockformer_training.py、short_term_ai.py、fund_monitor.py、watchlist.py、quote_service.py；同步 backend/app/indicators/pipeline.py 和 backend/app/backtest/engine.py 的身份关联。新增 backend/tests/test_canonical_identity_consumers.py，扩展 test_minute_repository_reads.py、test_repository_index.py、test_watchlist_enriched_join.py、test_stockformer_training.py。

- [ ] test_same_code_never_joins_assets 验证上述消费链不串市场；test_cache_contains_asset_symbol_version 验证缓存包含资产类型、规范 symbol、schema version、时间/周期；test_legacy_model_requires_mapping 验证旧模型无可确认旁路映射时阻止新预测，不以中性值伪造特征；test_published_snapshot_bytes_unchanged 验证旧预测及报告字节不变。
- [ ] 修改后运行新增消费者测试和列出的四个现有模块回归；移除身份用途的 split/六位提取前逐个分类为业务关联或合法供应商参数。

## Task 4: Migrate queue, authorization and audit state

- [ ] 将 minute_gap_queue.sqlite3 升级为 v2 schema：SQLite 复合主键为 (asset_type, exchange, code, trade_date)，stock_day_key 由统一函数生成；保留规范 symbol、来源、状态、重试次数、终态原因、最后错误、审计时间和原始值。
- [ ] minute_repair_authorization.py 当前已要求带后缀 symbol，但仍按前缀校验；接入统一身份契约并保留现有授权 id、task/source/operation/scope 及审计。为旧符号生成的 scope/hash 存显式旧新映射，不扩大批准范围。
- [ ] 为旧队列执行一次只读映射预检：可唯一映射的自动迁移，无法映射的进入 identity_quarantine；不得把未知项归入 SZ/BJ/SH。
- [ ] 队列执行器按日期闸门、完整身份和既有状态逻辑工作；身份未知进入待核验，不等于缺口完成。保留 attempts、next_retry_at、task id 和历史事件；迁移不能重置重试或改完成状态。
- [ ] 增加队列 schema 迁移、授权复用、冲突确认、未知隔离和重试状态测试。

Files: 修改 backend/app/services/minute_gap_queue.py、minute_repair_authorization.py、daily_maintenance.py、maintenance_queue.py；扩展 backend/tests/test_minute_gap_queue.py、test_minute_gap_queue_storage.py、test_minute_repair_authorization.py、test_minute_queue_approved_repair.py。队列存储在 minute_gap_queue.py，不新建平行存储实现。

- [ ] test_queue_identity_collision 断言同代码不同市场/资产保持独立；test_queue_migration_preserves_progress 断言 attempts、重试时间和状态不变；test_authorization_scope_preserved 断言同源同操作授权沿用、新源被拦截；test_conflicting_queue_states_quarantined 禁止直接选 complete；test_unknown_identity_not_approval 验证未知身份不弹覆盖确认。
- [ ] 运行上述四个模块测试与 test_maintenance_queue.py。

## Task 5: Build the safe historical migration tool

- [ ] 新增 scripts/migrate_security_identity.py，支持：--dry-run、--apply、--dataset、--batch-days、--resume、--rollback migration_id、--report-dir。
- [ ] 预检扫描维表、日K、分钟K、资金、竞价、财务、新闻关联和 SQLite 队列，输出按数据集的裸代码数量、可唯一映射、冲突、重复、未知、样本前后哈希和预计磁盘空间。
- [ ] 正式迁移顺序固定为：维表/历史映射 -> 日K和复权 -> 分钟K -> 资金/竞价/财务/新闻关联 -> 派生引用/用户状态 -> 队列/授权。全部阶段保持维护状态，每批临时写入验收后原子替换，不能提前恢复读写。
- [ ] 备份目录使用 data/migration_backups/<migration_id>/，每个文件记录 sha256、行数、分区、schema；保存 migration_manifest.json、migration_journal.jsonl 和隔离数据。
- [ ] 同一规范身份和时间的完全重复只保留一条；量价或来源不一致的重复不覆盖，进入冲突报告，并带上旧值、新值、来源和证据。
- [ ] 迁移完成后自动运行完整性检查：身份覆盖、重复、日K/分钟汇总、市场错配、队列可解析性、模型输入可用性；未通过则不发布迁移版本。
- [ ] 将 adj_factor、enriched、扩展资金表、JSON 自选/模拟仓/监控配置纳入 manifest；原始载荷、备份、已发布预测、旧研究产物和无证券字段的表列入免改清单。逐文件证明非身份字段不变，行数满足 原记录 = 活动记录 + 有凭据去重 + 隔离。
- [ ] 使用 SQLite backup API 和事务复制授权/队列数据库，不能只复制带有未 checkpoint WAL 的主文件。预检空间包含所有备份、最大临时分区、SQLite journal 及20%余量；不足或校验和变化时停止，不删除备份。
- [ ] 先写 test_dry_run_no_mutation、test_resume_idempotent、test_source_changed_stops_batch、test_sqlite_wal_backup、test_backup_failure_keeps_original、test_insufficient_disk_keeps_original、test_lock_blocks_writer、test_rollback_restores_hashes、test_non_identity_values_unchanged。再实现 Shared Interfaces 中迁移器与版本锁，全部测试通过后才在生产数据运行预检。
- [ ] 接入 backend/app/main.py、tickflow/scheduler.py、services/minute_refresh.py 和仓库写入点的版本/维护锁检查；正式切换前盘点独立 Docker/主机写入者并停止，不能仅关闭 HTTP。保持旧镜像与生产版本对应；回滚时新增数据单独保留，不静默删除。

Files: 新增 scripts/migrate_security_identity.py、backend/app/services/identity_migration.py、backend/app/services/identity_schema.py、backend/tests/test_identity_migration.py、backend/tests/test_identity_schema.py、docs/identity-migration.md；修改上述现有启动、调度和写入点。脚本在 backend uv 环境运行，显式接收数据根目录，不在实现中硬编码本机盘符。

## Task 6: Update API and frontend symbol handling

- [ ] 修改 frontend/src/lib/symbol.ts：不再对裸代码默认推断市场；展示和请求优先使用后端返回的规范 symbol，无法确认时显示“身份待核验”，不生成伪造的 .SZ/.SH/.BJ。
- [ ] 更新股票列表、图表、短线AI、StockFormer、资金监控、自选、模拟仓、回测和数据页的 query key 与请求参数，统一传递完整 symbol 和 asset type。
- [ ] 活动 API 要求规范 symbol 和确定的资产类型；只有证券搜索/解析入口接受裸代码，多匹配时返回 409/结构化歧义结果供选择，不在行情或交易请求中选第一个。
- [ ] 页面表格统一显示名称 + 代码.市场，在诊断场景显示来源代码、身份状态和隔离原因。
- [ ] 增加前端测试和 Playwright 冒烟：相同六位代码的股票/指数/ETF不串图、不串资金、不串推荐。

Files: 修改 frontend/src/lib/symbol.ts、api.ts、queryKeys.ts、useSharedQueries.ts；修改 backend/app/api/kline.py、watchlist.py、stockformer.py、short_term_ai.py、financials.py 及其真实调用方；新增 backend/tests/test_api_canonical_identity.py、tools/check-canonical-identity-ui.mjs（沿用 tools/check-maintenance-ui.mjs 的浏览器运行方式，不新增测试框架）。

- [ ] test_api_rejects_bare_symbol、test_search_ambiguous_identity、test_asset_type_is_required、test_identical_digits_separate_payloads 先失败再通过。前端用 pnpm build 验证类型，并在浏览器桌面/窄屏验证名称+规范代码、自选/模拟仓、股票/指数/ETF切换无缓存串用，未知身份不可提交。

## Task 7: Verification, deployment and rollback rehearsal

- [ ] 在隔离测试环境运行身份、队列、迁移、provider、repository、训练、回测、通知的受影响回归和前端构建；不得对生产数据跑破坏性测试。
- [ ] 使用 --dry-run 生成当前数据迁移报告，确认磁盘空间足够、未知和冲突清单可解释；若空间不足，停止正式迁移并给出可删除/迁移路径。
- [ ] 备份后先迁移一小批日期和三个市场各一组样本，运行读写回归、图表、推荐、资金、训练输入检查。
- [ ] 通过样本校验后执行分批正式迁移；迁移期间暂停相关写入任务，保留日常增量同步的状态，不触发全历史市场环境重算。
- [ ] 重建 Docker 镜像并重启 3018；检查 /api/health、任务状态、数据页、StockFormer 和短线AI页面。
- [ ] 进行一次 rollback rehearsal：在临时副本回滚一个数据集，确认清单、日志和队列状态可恢复；确认后才报告迁移完成。

Verification commands:

~~~powershell
# workdir: backend
uv run --frozen --extra dev pytest tests/test_security_identity.py tests/test_security_lifecycle.py tests/test_identity_migration.py tests/test_identity_schema.py tests/test_provider_canonical_identity.py tests/test_canonical_identity_consumers.py tests/test_api_canonical_identity.py tests/test_minute_gap_queue.py tests/test_minute_gap_queue_storage.py tests/test_minute_repair_authorization.py tests/test_minute_queue_approved_repair.py tests/test_maintenance_queue.py tests/test_minute_repository_reads.py tests/test_repository_index.py tests/test_watchlist_enriched_join.py tests/test_stockformer_training.py -q
uv run --frozen --extra dev python ../scripts/migrate_security_identity.py --data-dir ../data --dry-run --report-dir ../data/import_reports/identity_migration
# workdir: frontend
pnpm build
# workdir: repository root, only after migration acceptance
docker compose config --quiet
docker compose up -d --no-deps app
Invoke-RestMethod http://localhost:3018/api/health
~~~

镜像预先在停写前构建并保留旧镜像引用；切换时不重建其他容器。所有测试应输出 PASS/0失败，前端构建退出0；健康检查不能替代任务读写、训练输入及身份覆盖审计。UI 脚本使用当前已认证会话或临时测试账号，不输出密钥。补充运行 test_minute_quality.py、test_minute_repair.py、test_minute_timezone_contract.py、test_financial_incremental.py 证明既有量价/稀疏日判定未变化。

## Plan Self-Review

- 已与设计核对：移除了永久裸码兼容层，迁移顺序与设计一致，新增海外新闻、复权/enriched/用户配置、不可变产物旁路映射、版本锁和 SQLite 一致性备份。
- 已按实际文件更正 repository 和队列存储路径，授权模块当前已要求后缀，不能把它误写成裸代码主键。
- 五项 Review Focus 分别由 provider_roundtrip、same_code_never_joins_assets、lifecycle_alias_effective_date、conflicting_queue_states_quarantined、消费者/API 回归覆盖。
- 计划中新增函数和测试是实施目标，不是已存在或已通过。当前仅完成方案与计划文档，尚未迁移生产数据或部署。

## Completion Criteria

- [ ] 生产数据中新写入的 symbol 不再出现裸代码。
- [ ] 维表、日K、分钟K、资金、竞价、财务、新闻关联和补差队列均能用完整身份键稳定关联。
- [ ] 未知身份、代码冲突、市场冲突、重复量价冲突均有隔离记录和可解释原因，不被中性值伪造为有效数据。
- [ ] 迁移备份、manifest、journal、quarantine 和 rollback 均可验证。
- [ ] 模型、回测和推荐在同代码不同市场场景下不会串数据，历史预测快照保持不可变。
- [ ] Docker 服务健康，增量任务可继续运行，且没有因身份迁移自动触发全历史重算。
