# 分钟数据第三方核验流 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将分钟导入失败后的第三方日K核验、裁决、退避和队列状态接入现有盘后维护链路。

**Architecture:** 保留 `minute_repair` 作为唯一分钟写入入口，新增无副作用的 `minute_cross_source` 服务负责第三方日K标准化和裁决。`daily_maintenance` 在分钟导入后调用该服务，并把结果写入 `minute_gap_queue`；只有明确通过既有量价合同的数据才进入标准分钟表。

**Tech Stack:** Python 3.11、FastAPI service layer、Polars、SQLite queue、Parquet、pytest、Docker。

**Spec:** `docs/superpowers/specs/2026-09-22-minute-third-party-verification-design.md`

## Global Constraints

- 不凭日K合成分钟K。
- 使用完整身份键 `资产类型+交易所+证券代码+时间`。
- 网络错误不能标记为源无记录。
- 日K修正和冲突替换必须先备份并原子回读验证。
- 只处理失败股票日队列，不扩大为全历史扫描。
- 未核验成交口径不得进入资金特征或高置信度训练。

## Review Focus

- 仅有单一外部日K源可用时不能自动改本地日K：由 Task 2 测试单源候选状态。
- Eastmoney/Promax连接断开不能变成空记录：由 Task 1 测试错误分类和退避。
- `92xxxx.BJ` 不能按六位代码误合并：由 Task 1 测试身份键。
- QMT 239根合法稀疏数据不能被强制要求240根：由 Task 2 测试既有QMT合同兼容。
- 已有残缺记录不能被新数据静默覆盖：由 Task 3 测试备份替换闸门。

### Task 1: 第三方日K标准化与裁决服务

**Files:**
- Create: `backend/app/services/minute_cross_source.py`
- Test: `backend/tests/test_minute_cross_source.py`

**Interfaces:**
- Consumes: `minute_repair` 的分钟汇总、`daily_reference`、provider返回的标准或供应商日K。
- Produces: `verify_stock_day(...) -> dict`，返回 `status`、`sources`、`evidence`、`retryable` 和 `retry_after_seconds`。

- [x] **Step 1: Write failing tests**

覆盖以下输入：两源与分钟一致返回 `minute_import_candidate`；本地日K不同但两源与分钟一致返回 `daily_baseline_repair_candidate`；本地日K与外部一致而分钟不同返回 `qmt_turnover_conflict`；网络异常返回 `source_unavailable`；同码不同市场返回身份隔离。

- [x] **Step 2: Run the focused tests and confirm failure**

Run: `cd backend; uv run pytest tests/test_minute_cross_source.py -q`

Expected: FAIL because `minute_cross_source` and `verify_stock_day` do not exist.

- [x] **Step 3: Implement the minimal service**

实现明确的 `DailySnapshot` 标准字段、成交量单位转换、价格/成交量/成交额容差、源错误分类和有限指数退避。服务只返回裁决，不写日K或分钟Parquet。

- [x] **Step 4: Run focused tests; Ruff unavailable in the local runtime**

Run: `cd backend; uv run pytest tests/test_minute_cross_source.py -q; uv run ruff check app/services/minute_cross_source.py tests/test_minute_cross_source.py`

Expected: all focused tests PASS and no Ruff errors.

### Task 2: 接入失败队列和盘后维护

**Files:**
- Modify: `backend/app/services/minute_gap_queue.py`
- Modify: `backend/app/services/daily_maintenance.py`
- Modify: `backend/app/services/minute_failover.py`
- Test: `backend/tests/test_minute_cross_source.py`
- Test: `backend/tests/test_minute_gap_queue.py`

**Interfaces:**
- Consumes: Task 1 `verify_stock_day` 返回的状态和重试信息。
- Produces: 队列状态 `cross_source_verify`、`daily_baseline_repair_candidate`、`source_unavailable`、`quarantine`，以及按股票日去重的下一次执行时间。

- [x] **Step 1: Add queue tests**

断言同一 `symbol+date` 重复写入不会产生重复任务；`source_unavailable` 进入 retry；`source_no_record` 进入终态；`cross_source_verify` 不被普通分钟抓取重复领取。

- [x] **Step 2: Run queue tests and confirm failure**

Run: `cd backend; uv run pytest tests/test_minute_gap_queue.py tests/test_minute_cross_source.py -q`

Expected: new queue state assertions FAIL before implementation.

- [x] **Step 3: Wire the maintenance stage**

在 `daily_maintenance.execute(kind="minute")` 的现有 validation failure 分支调用第三方核验；只有 `minute_import_candidate` 才允许进入既有 `repair_minute_gaps` 重试成功路径，其余状态只记录队列和证据。不得在持锁期间执行网络请求。

- [x] **Step 4: Preserve source error categories and bounded retry in the verification queue**

把源错误转换为现有 `minute_failover` 的稳定错误类别，复用已有 cooldown 和 retry state；限制每个股票日每轮最多一次每个第三方源请求。

- [ ] **Step 5: Run the affected backend tests**

Run: `cd backend; uv run pytest tests/test_minute_gap_queue.py tests/test_minute_failover.py tests/test_minute_source_recovery.py tests/test_minute_cross_source.py -q`

Expected: all tests PASS.

### Task 3: 证据报告、备份闸门和回归验证

**Files:**
- Modify: `backend/app/services/minute_cross_source.py`
- Modify: `backend/app/services/minute_gap_queue.py`
- Create: `scripts/run_cross_source_minute_verify.py`
- Test: `backend/tests/test_minute_cross_source.py`

**Interfaces:**
- Consumes: 失败股票日队列、源裁决结果、既有报告目录。
- Produces: `data/import_reports/cross_source_verify/<run_id>/report.json`、`details.csv`、`retry_queue.jsonl`，以及可供现有导入器使用的候选清单。

- [x] **Step 1: Test report persistence**

验证报告包含每个股票日的源状态、单位、误差、裁决和重试时间；异常时也能保存部分结果；报告中不出现Token、URL查询参数或原始异常文本。

- [x] **Step 2: Add the bounded CLI**

`run_cross_source_minute_verify.py` 只接受 `--data-dir`、`--queue`、`--output`、`--max-items`，拒绝生产目录之外的输出，默认只处理队列中的失败项。

- [ ] **Step 3: Test backup and replacement gating**

只在调用方显式传入授权替换标记且裁决为 `daily_baseline_repair_candidate` 或 `existing_values_conflict` 时生成备份；单源一致、源不可用和三方冲突均不得覆盖。

- [ ] **Step 4: Run the complete focused verification**

Run: `cd backend; uv run pytest tests/test_minute_cross_source.py tests/test_minute_gap_queue.py tests/test_minute_failover.py tests/test_minute_source_recovery.py -q; uv run ruff check app/services/minute_cross_source.py app/services/minute_gap_queue.py app/services/daily_maintenance.py app/services/minute_failover.py tests/test_minute_cross_source.py`

Expected: all tests PASS and no lint errors.

### Task 4: Docker smoke test and maintenance dry run

**Files:**
- Modify: `docs/minute-failover-diagnostics.md`
- Modify: `docs/full-market-data-maintenance.md`

- [ ] **Step 1: Build and start the app image**

Run: `docker-compose build app; docker-compose up -d app`

- [ ] **Step 2: Run the read-only queue verification**

Run: `docker run --rm --user 0:0 -v "${PWD}/data:/app/data" -v "${PWD}/scripts:/app/scripts:ro" tick-stock-panel-v022-integrated-app /app/.venv/bin/python /app/scripts/run_cross_source_minute_verify.py --data-dir /app/data --queue /app/data/import_reports/audit_bj_final_20260922/missing_or_partial.csv --output /app/data/import_reports/cross_source_verify_smoke --max-items 5`

Expected: no production Parquet changes; report contains bounded source statuses.

- [ ] **Step 3: Check service health and diff**

Run: `Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3018/health; git diff --check`

Expected: health returns `status=ok`; diff check is clean.
