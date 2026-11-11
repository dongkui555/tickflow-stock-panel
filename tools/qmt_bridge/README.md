# QMT local bridge

The bridge must run on the Windows host while QMT and its research Jupyter
service are running. It exposes only read-only market-data endpoints to the
Docker application.

```powershell
.\tools\qmt_bridge\start_qmt_bridge.ps1
```

For exact historical 1-minute volume, provide the verified local DAT directory:

```powershell
.\tools\qmt_bridge\start_qmt_bridge.ps1 -QmtDataRoot F:\QMT\datadir
```

This option additionally requires NumPy in the Windows bridge runtime. It reads
the shared verified DAT layout and restores fractional lots only when identity,
timestamps, prices, amount and the original whole-lot volume match. It adds no
minute records and rejects mismatches or unavailable DAT records. The directory
is read-only; no QMT file or panel partition is changed. Existing bridge processes
must be restarted to load new code/options, preserving their push database.

The bridge reloads the Jupyter token before kernel access because restarting
the QMT client rotates that token.

The default push database is `data/user_data/qmt_push.sqlite` relative to the
checkout containing `qmt_http_api.py`. Starting the bridge from another checkout
therefore changes the default cache. Before an upgrade, identify the currently
active cache and preserve it together with any divergent older cache. Set
`QMT_PUSH_DB` to the explicitly verified cache path before using the launcher,
or pass `--push-db` when starting the Python entry point directly. Stop external
collector/push writers as well as the bridge before making a filesystem snapshot.
Do not start two bridge writers against the same database or replace a newer
active cache with an older panel-directory copy.

Endpoints:

- The default listen address is `http://127.0.0.1:8188` (also reachable from Docker through `host.docker.internal`).
- `GET /api/health`
- `GET /api/capabilities` (cached state; `?refresh=1` requests a bounded kernel probe)
- `GET /api/quote?symbols=000001.SZ,600000.SH`
- `GET /api/kline?code=000001.SZ&freq=1m&count=240`
- `POST /api/push/minutes` (QMT strategy batch history-minute upload)
- `GET /api/tick?code=000001.SZ&count=5000`
- `GET /api/l2/order|transaction|quote` (only after QMT reports L2 data)

QMT post-close heartbeat ticks are removed. Minute bars are restricted to the
A-share continuous trading sessions. QMT volume is returned and stored in lots,
matching this repository's existing minute-data convention.

The collector now schedules a bounded recent-history job after the close on
weekdays and throughout Saturday and Sunday. It discovers the
Shanghai/Shenzhen/Beijing A-share universe, requests the latest five calendar
days in batches of 100 symbols, retries an empty or failed batch up to three
times, and pushes validated rows to
`/api/push/minutes`. The job keeps its cursor for the current run and never
blocks the live quote callback. A new QMT strategy run starts a new recent
window; older history remains the responsibility of StockDB, offline files,
Promax, or TDX.

The same strategy runs a separate index-daily job. It reads the exact local
index universe from `http://127.0.0.1:3018/api/index/universe`, requests 30
calendar days in batches of 50 symbols, validates OHLCV, and pushes accepted
rows to `/api/push/daily`. A failed index batch is retried and remains
available to the original Tushare/Promax route; it never marks the whole 612
symbol universe complete.

An explicit capability refresh probes `000001.SZ` and `600000.SH` independently.
Missing quotes or identical fingerprints disable kernel-backed realtime,
five-level depth and ticks. A recent validated ContextInfo push may separately
establish live routing. This prevents an unauthenticated formula session from
copying one stock's quote to other symbols. A health request alone does not
refresh the kernel or certify a stock/date for import.

Historical minute capability is checked independently of live routing, using
`000001.SZ` over the latest 14 calendar days with bounded timestamps and
`count=-1`. Only an actual positive-price trading-session bar establishes this
capability. `count=1` can return a post-close heartbeat and must not be used to
infer that historical minutes are missing. A validated push cache can also
establish minute availability; exact DAT verification still applies to bounded
records returned for import.

If `/api/capabilities?refresh=1` reports `minute=true` and
`symbol_routing=false`, historical minutes remain usable while live kernel
quotes are unavailable. The panel exposes only the confirmed datasets; it must
not advertise realtime or depth from minute availability. For live access, ask
the broker to enable MiniQMT/XtQuant API access and run the authenticated
collector. L2 order, transaction and quote endpoints additionally require the
corresponding Shanghai and Shenzhen Level-2 entitlement. Each capability is
accepted independently.
