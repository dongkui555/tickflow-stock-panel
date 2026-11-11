# Index Daily Incremental Sync

Daily scheduled sync, historical initialization and `POST /api/index/sync_daily`
use the same closed-day index gap planner. ETF and minute routing are unchanged.

- The planner checks every index and confirmed local market trading date in the
  requested window. A complete latest partition does not hide another index's gaps.
- Scheduled sync checks at least the recent 14 calendar days, but requests only
  missing index dates. Historical initialization checks the requested range.
- Valid existing index days are not downloaded or overwritten. Invalid OHLC or
  duplicate days are not counted complete. The current unfinished trading day is excluded.
- HTTP 429 retries honor numeric and HTTP-date `Retry-After`. Without a valid
  header, waits are 2, 4 and 8 seconds. At most three retries occur per request.
  Waits above 30 seconds and exhausted retries are handed to the persistent queue;
  the queue never retries before the server's specified deadline.
- A foreground batch processes at most 40 missing indices. Queued batches process
  at most 20. Every accepted index is persisted and checkpointed before continuing.
  Gateway-wide denial stops the batch; pending work survives a restart.
- The manual API queues work without doing network synchronization in the HTTP
  request. It retains `index_count` and `rows_written` and adds `index_sync` details.
  `status=queued` means pending work, not completed synchronization.
- `index_sync` distinguishes complete index count, pending index count, pending
  index-days, rows written this run, request errors and empty source responses.
  A partial result renders as `部分完成`, not a green complete state.

The Data page's maintenance queues show `指数日K增量补缺`, current progress and retry
time. `rows_written` is not the total local row count. An empty response is not
proof that an index is discontinued or unsupported, and does not close its gap.

The existing maintenance worker runs historical jobs outside trading hours
(before 07:00 or from 16:00, Beijing time). Once a gap is filled it is skipped on
subsequent attempts, including when old queue entries are resumed.
