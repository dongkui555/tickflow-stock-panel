# Promax Historical Minute Boundaries

This adapter applies only to the Promax `stk_mins` one-minute route. It does not
change the official Tushare client, another proxy, TDX, five-minute bars or the
live `rt_min_daily` session.

## Observed Contract

For 920000.BJ on 2025-07-03, requesting from 09:30 omitted the opening record.
Requesting from 09:25 returned the actual 09:30 record: 8,300 shares and 181,687
yuan. The 15:00 record had zero turnover; a 15:01 record contained 51,861 shares,
1,148,202.50 yuan and the closing price 22.14. The ordinary minute result ended
at 22.12 instead. Including both boundary records gives 1,757,565 shares, exactly
matching the independently queried daily endpoint. The amount differs by 0.53
yuan due to rounding. This is an observed gateway behavior, not a claim that all
Tushare minute sources use these timestamps.

## Acceptance Rules

- Fetch complete historical sessions using padded 09:25 to 15:10 bounds. Return
  only the caller's requested interval after validation.
- Preserve actual 09:30 records; never fabricate an opening auction amount.
- Map an actual positive-turnover 15:01 bucket to 15:00 only when that day's
  15:00 bucket is an unambiguous zero-turnover placeholder and there is no other
  post-close turnover. Do not shift the entire minute series.
- Require 240 distinct regular minutes, optional actual 09:30, valid OHLC and
  turnover, and agreement with a fresh same-stock/same-date daily response.
- Shared repair validation computes daily OHLC from bars with positive turnover
  (volume or amount):
  a zero-volume/zero-amount placeholder can carry a previous quote and is not
  the day's first transaction. Keep source timestamps and prices unchanged.
  A positive amount still participates even if source volume is zero; total
  volume, amount, units and price consistency retain the existing checks.
- Minute source volume is shares; persisted volume is lots. Daily `vol` is
  lots and daily `amount` is thousands of yuan. No magnitude-based unit guessing.
- Keep existing price tolerances (0.011 yuan or one ppm) and volume/amount
  tolerances (0.2% or one unit). Daily volume must match the known lots contract.
- Reject missing/ambiguous daily references, conflicting duplicates, incomplete
  sessions and ambiguous closing buckets. Do not fall back to unverified bars.
- Journal raw minute records, raw daily records and validation outcome under
  `data/import_reports/gateway_minute_contract/DATE/CODE/*.json.gz`. No API keys
  or request headers are recorded. Journals are content-addressed and atomic.
- Disk repair additionally checks the local daily reference. Replacement needs
  explicit opt-in, partition backup, readback verification and preservation of
  unrelated symbols. A local daily row aggregated from minutes is not a second
  independent source; the fresh gateway daily check remains required.
- Validate the final merged day as well as the incoming day. Small differences
  in retained old bars must not accumulate into a daily turnover mismatch.

After 15:10 Beijing time, Promax's completed current-day minute requests use
this verified history route instead of overwriting it with `rt_min_daily`.
Missing or delayed source publication remains a retryable gap, not success.

## Unconfirmed Alternate Labels

The bounded 2026-09-30 probe for 920701.BJ returned 240 unique labels covering
09:30–11:29 and 13:00–14:59 from both `stk_mins` and `a_share_mins`. Daily totals
agree, but this differs from the verified end-labeled contract above. A one-minute
shift was evaluated only as a diagnostic: 80 of 138 corresponding QMT bars differ
in at least one OHLC/turnover field, and 30 differ in volume. Of the 102 extra
Promax records, 100 carry zero turnover. These facts do not establish the source
bucket boundaries or justify manufacturing a complete execution axis.

Until the source timestamp contract is confirmed, these responses remain
`source_incomplete`. Do not shift the whole series, pad QMT, or splice providers
to satisfy strict minute execution. Keep the original raw journal and source
error classifications, including unavailable/pending/denied responses.

## Bounded Repair

`scripts/repair_gateway_minute_gaps.py` accepts the exact CSV manifest and an
optional previous repair report to avoid re-requesting already attempted pairs
in the same maintenance pass. It never turns that manifest into the Cartesian
product of all stocks and dates. It uses bounded workers and request counts,
and stops if the first 40 requests or two subsequent waves yield no verifiable
additions. Deferred and rejected pairs stay in the report, not marked complete.

Do not run against dates that another process is writing. The maintenance CLI
checks active pipeline stages; an explicitly bounded old-history run may coexist
only with a non-overlapping scheduled recent-minute stage. Restart the app after
maintenance to invalidate cached historical data. Scheduled full-market sync
remains incremental and its configured 15:30 schedule must be preserved.
# Asynchronous Historical Requests

For `stk_mins` and `rt_min_daily`, an HTTP 202 response is not an empty dataset.
The adapter reports `source_pending`, `retryable=true`, `http_status=202` and
`retry_after_seconds`, without exposing the upstream error body or credentials.
It honors `Retry-After` (seconds or HTTP date); invalid or absent values use a
30-second wait. The exact query is held in cooldown within the client instance;
other symbols or date ranges are not blocked. There is no unbounded polling.
Once the response is ready, the usual session and daily-reference validation
still applies. A pending response must not certify a date as having no data.
