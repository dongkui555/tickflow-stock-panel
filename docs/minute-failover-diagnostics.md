# Minute failover diagnostics

Promax remains the primary source and TDX the verified backup. A valid HTTP
response is not proof that minute bars are suitable for persistence.

## Validation And Partial Progress

- Closed stock days still require 240 regular minutes, valid OHLC, and agreement
  with the raw daily price, volume and amount reference. No tolerance was relaxed.
- A closed-day conflict with existing bars is not an outage. Replacement requires
  approval, a backup and the existing whole-day repair workflow.
- During trading, a contiguous prefix may be saved after every included bar
  passes the existing price and turnover checks. The prefix stops BEFORE the
  first missing minute. It cannot replace conflicting old bars.
- Saving a prefix does not mean completion. `source_incomplete` and missing times
  remain in the journal; maintenance verifies the requested end time and retains
  incomplete stocks for incremental retry.
- Validation failures no longer abort unrelated symbol batches or start-date
  groups. An actual shared upstream outage still stops requests and backs off.
- Chart reads continue to use this same validation path. No unchecked third
  source, timestamp interpolation or synthetic minute bar was added.

## Journal

`data/import_reports/minute_failover/YYYY-MM-DD/*.json` contains independent,
uniquely named fetch reports. Source issues include:

- `reason`, `primary_reason`, `symbol`, `date`, `source`
- `source_rows`: the number returned, before validation
- `rows`: accepted rows, including an incomplete but verified intraday prefix
- `missing_minutes`, `missing_times` (first eight examples)
- `first_time`, `last_time`, source and reference volume/amount totals

The error message distinguishes `minute_validation_failed` from
`upstream_unavailable`; it includes a bounded sample of symbols and reasons.

## Daily Batches And Source Recovery

- Routine `sync_and_persist_minute` no longer invokes the legacy whole-table
  null-datetime audit or symbol-partition migration. Those helpers are offline
  maintenance only, not automatic startup or daily work. Recent partition and
  per-stock completeness checks remain. No historical partitions are deleted.
- Preparation reports source resolution, the actual partition date range, local
  gap checks, and the start of source requests separately.
- Promax stops after three consecutive transient failures, including after a
  partially successful response. Unattempted stocks retain `not_requested` and
  the causal error; they are not classified as an empty server response.
- HTTP status and `Retry-After` survive provider, failover, batch and queue
  boundaries. A local coverage failure keeps `data_reason` alongside its
  `source_errors`. Successfully stored stocks are still checkpointed.
- `user_data/minute_source_state.json` persists bounded exponential source
  cooldowns (60 seconds up to 32 minutes, or longer when the server requires it;
  permission/endpoint errors wait at least an hour). Network calls run outside
  the state-file lock. Valid primary bars clear its failure state; valid TDX
  minute probes clear the backup state. Other datasets are unaffected.
- When both sources are cooling down, the same unfinished increment waits until
  the first route becomes eligible. It is not rotated into thousands of new
  gateway preparation requests. Actual per-stock data conflicts still rotate
  through the queue and are never overwritten without approval.
- TDX fallback probes actual minute bars for two liquid reference stocks,
  checking the requested session, freshness, OHLC and volume/amount consistency.
  Bridge/TCP health alone does not make the minute capability usable. Each
  downloaded target stock still passes the existing daily-reference validator.
- The local bridge tests a small minute sample before selecting a node for
  minute requests. Connected-but-empty nodes are skipped and cooled down;
  node selection has a four-second budget. Quote/day routes are unchanged.
- Optional diagnostic callbacks and additional queue fields preserve existing
  callers and UI contracts. The queue displays the cause, cooldown and retry
  time using its existing message/error fields. No market-data schema changes.

## Third-Party Cross-Source Verification

After a closed-day minute validation failure, the maintenance worker runs a
bounded stock-day verifier. It requests the same identity-qualified daily bar
from the configured Promax/Tushare gateway and the Eastmoney historical daily
endpoint, then compares both with the local daily reference and the stored
minute aggregate. It does not synthesize minute bars or overwrite either
partition.

The result is persisted under
`data/import_reports/cross_source_verify/<run_id>/`:

- `report.json`: complete per-stock-day decision and sanitized source errors;
- `details.csv`: tabular evidence for review;
- `retry_queue.jsonl`: only retryable source failures.

The queue distinguishes `daily_baseline_repair_candidate`,
`qmt_turnover_conflict`, `source_no_record` and `quarantine`. A transport,
timeout, rate-limit or gateway-preparation failure is `source_unavailable` and
gets delayed retry; it is never treated as a source with no record. QMT
turnover conflicts reopen the minute source chain, while daily-baseline
candidates and quarantined rows stay out of normal minute retries.

For a reviewed CSV manifest or the persistent SQLite queue, the same verifier
can be run manually without changing market Parquet:

```powershell
python scripts/run_cross_source_minute_verify.py `
  --data-dir data `
  --queue data/user_data/minute_gap_queue.sqlite `
  --output data/import_reports/cross_source_verify/manual-20260922 `
  --max-items 50
```

The output directory must be new and inside `data`; the command only updates
the minute-gap queue and writes evidence, so any daily/minute replacement still
requires the existing explicit backup-and-readback repair command.

Regression tests: `test_minute_source_recovery.py`, `test_local_tdx_market.py`,
plus existing minute routing, repair, normalization and maintenance tests.

## Read-Only Probe

From the application environment with `PYTHONPATH` pointing to `backend` locally
or `/app` in Docker:

```powershell
python tools/probe_minute_failover_readonly.py --symbols 603215,002136,000001 --start 2026-09-08 --end 2026-09-09T11:30:00
```

This tool queries at most ten stocks over five days. It compares both sources
against local minutes and daily references without changing market data. An
optional `--output` JSON path creates a new diagnostic report and refuses to
overwrite an existing file. Do not import `app.main` or initialize a second job
store in standalone production-data probes.

## Observed On 2026-09-09

For 2026-09-08, Promax returned 241 bars for 603215 and 002136, but turnover did not
match the daily reference. TDX returned 240 bars and passed daily validation.
Existing local fragments (10 and 14 rows) conflicted, including separately stored
09:30 opening volume versus the TDX 09:31 combined bar. They were not overwritten.

The initial morning probe of 2026-09-09 returned no Promax rows for the three
sampled stocks. TDX returned 119 regular bars through 11:29, plus a raw 13:00
bar outside the canonical session. The initial implementation left it pending.

The follow-up full-session audit identified TDX's lunch-close label convention:
119 morning end labels from 09:31 to 11:29, a 13:00 label, then 13:01 to 15:00.
The adapter now maps that 13:00 record to 11:30 before range filtering, ONLY for
minute1 batches containing all 119 preceding morning labels and neither 09:30
nor 11:30. Partial ambiguous responses and already canonical batches are not
changed. Raw prices, volume and amount are preserved. All 110 fetched stock days
(55 stocks, September 8 and 9) passed the unchanged full-day validators after
this normalization. This is source-specific timestamp normalization, not filling
an absent minute with a synthetic bar.

## Frozen Midday Tasks And Queue Recovery

- A request ending at 11:30 is not judged as a 240-minute request merely because
  it is retried after close. Range validation honors its requested cutoff.
- The minute maintenance worker explicitly promotes an old midday task to a
  full session once that date is closed (15:10 or later). Full-session data is
  validated against the daily reference before incremental persistence.
- Completion checks inspect every expected minute, not just the final timestamp.
  Per-symbol/date `validation_failures` include missing-time and turnover evidence
  in the maintenance queue API. Zero-trade daily records are not 240-minute gaps.
- Failed batches rotate once through the pending universe, then back off even if
  more than 50 stocks remain. Genuine completion progress resets minute retry
  backoff; previously completed items stay checkpointed.
- No conflict replacement is automatic. The read-only audit showed 53 of 55
  existing September 8 stock days were already valid despite source differences.
  Only two incomplete September 8 days and fourteen incomplete September 9 days
  required explicit conflict-replacement approval at the time of the audit.

Read-only whole-task comparison (writes a new JSON report, not market data):

```powershell
python scripts/audit_tdx_minute_queue.py --data-dir data --base-url http://localhost:8080 --task-id be41ce688d
```

## Repair Result

After explicit approval for those sixteen stock days, the application was stopped
between active jobs and `scripts/repair_minute_queue_20260909.py` performed the
bounded offline repair. It verifies the application's data mount, requires the
reviewed audit and stopped container, and keeps a fixed approval whitelist.
Full partitions are backed up with matching SHA-256 checksums before atomic
writes; merged data is validated first and read back afterward.

Run `minute_queue_repair_20260909T160123` retained 53 valid stock days, added 5,581
missing rows across 41 stock days, and replaced 1,494 incomplete/conflicting old
rows with 3,840 verified rows across the sixteen approved days. Post-write checks
passed all 110 stock days and checkpointed task `be41ce688d` to 55/55. This result
is scoped to that queue task, not a claim that all market history is complete.

The report is in `data/import_reports/minute_queue_repair_20260909T160123.json`;
the two backups are under `data/import_backups/minute_queue_repair_20260909T160123/`.
The targeted regression suite passed 177 tests, including offline backup and
unapproved-conflict preservation, source normalization, queue progress/backoff,
pre-listing dates, timezone conversion, and persistence validation.

Regression coverage includes tail/interior gaps, conflicting prefixes, distinct
start-date groups, streamed batches, actual network outages, and the original
whole-day repair/backup checks. The broader pre-existing minute-refresh tests
include two unrelated assertions expecting a six-second default while current
preferences return sixty seconds; those expectations were not changed here.
