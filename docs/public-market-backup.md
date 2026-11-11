# Public L1 Backup Routes

## Routing

- Realtime: configured primary -> Tencent -> Sina -> Eastmoney -> Xueqiu.
- Five-level depth: TickFlow (when entitled) -> Tencent -> Sina -> Eastmoney.
- Historical minutes: retain the existing Promax/Tushare -> TDX route and its
  session, daily OHLC, volume and amount validation. Public snapshots are NOT
  used to fabricate missing historical minutes.
- Historical fund flow: retain the AI Agents Tushare -> Eastmoney -> AkShare
  fallback. Public five-level data is NOT L2; net-flow snapshot fields are NOT
  gross buy/sell volumes and do not overwrite the `moneyflow` table.
- No public substitutes for historical auction, financial statements or news
  are introduced by this change. Their existing routes remain unchanged.

## Configuration

```dotenv
PUBLIC_MARKET_FALLBACK_ENABLED=true
PUBLIC_MARKET_FALLBACK_MAX_SYMBOLS=100
PUBLIC_MARKET_FALLBACK_MAX_AGE_S=180
```

The application default is disabled for backwards compatibility. Enable it in
the deployment `.env`, then recreate the app container. Selected primary
provider preferences and existing historical data are never changed.

## Failure and Recovery

Realtime primary calls use one in-flight worker with an 8-second foreground wait; slow
SDK retries cannot spawn an unbounded number of workers. Public requests have
3-second connect and 5-second read timeouts, no hidden retry loops, a per-pass
20-second soft budget (the final in-flight request may exceed the budget), and
exponential failure cooldown from 30 seconds up to 15 minutes. Primary recovery
is retried after cooldown; switching is per request, not a preference rewrite.

Only absent, stale or invalid priority symbols are sent to the next public
source. Tencent/Sina requests batch up to 50 symbols; Eastmoney is capped at
3 individual requests per pass. The shared 15-second cache avoids duplicate
quote/depth requests. The priority pool uses existing watchlist, positions,
explicit monitor symbols and core indices. A public fallback is NOT a full
market feed; missing counts stay visible. A sparse result is memory-only and
cannot truncate a full-market historical partition.

A few invalid rows in an otherwise healthy full-market batch are excluded,
not treated as a complete outage. Valid rows are merged by the local instrument
identity, so six-digit stored symbols are not duplicated with exchange suffixes.
Missing/unsupported symbols have per-symbol/per-capability negative caching;
they do not trip the whole source circuit breaker or block its valid depth data.

Provider timestamps must be today in Beijing. Intraday age defaults to three
minutes; lunch and after-close use the latest session endpoint. Yesterday's
close is never re-stamped as today. OHLC ordering, amount/volume consistency,
depth ordering, exchange-qualified symbol and finite numbers are validated.
Index price snapshots are supported, but unverified index volume units and
empty index order books are not used as equity volume/depth.

Xueqiu is the last realtime snapshot fallback. Its public snapshot endpoint
was verified for A-share stocks (including `92xxxx` identities), but it does
not provide a five-level book in this route. Historical daily K-lines are a
separate optional provider path and require `XUEQIU_TOKEN` (`xq_a_token`) or
`XUEQIU_COOKIE`; without credentials the daily route is skipped rather than
retrying a known 400 response. Xueqiu is not used for minutes, money flow,
auction, news or L2.

## Operational Checks

Settings -> Data Sources -> Automatic Backup Routes shows selected primary,
actual sources, degraded priority coverage, last check and cooldown. The same
read-only data is available from `GET /api/settings/backup-routes`. The quote
status adds `backup_route`; these endpoints never make market data requests.

An enabled backup denotes an implemented, best-effort path, not a paid SLA or
a recent successful health check. All-source failure preserves the historical
store and publishes no replacement quote. No provider rotations, TLS bypass
or authentication workarounds are used.

## Validation

`backend/tests/test_public_market_backup.py` covers variable field counts,
exact amount/shares conversion, depth direction, stale/invalid/empty results,
partial fallback, cooldown/cache, primary recovery and missing entitlements.
