"""Run a bounded, exact-manifest repair; API secrets stay in application settings."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from urllib.request import urlopen

import polars as pl

from app.services.kline_sync import _resolve_minute_provider
from app.services.minute_repair import repair_minute_gaps
from app.services.tushare_ext_sync import _temporary_promax_timeout
from app.plugins.tushare.provider import TushareProvider, build_datahub_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=160)
    # Historical gateway requests can legitimately take longer than the
    # normal live-query timeout.  Keep the bounded repair conservative about
    # concurrency, but avoid classifying a slow valid response as a source
    # failure by default.
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--single-day", action="store_true", help="Use one date per request when a range response omits dates")
    parser.add_argument("--start-date", type=date.fromisoformat, help="Restrict this run to one bounded date range")
    parser.add_argument("--date-gated", action="store_true",
                        help="Process and flush one trading date before moving to the next")
    parser.add_argument("--repair-conflicts", action="store_true", help="Replace verified conflicting stock-days only with explicit authorization")
    parser.add_argument("--source", choices=("auto", "promax", "datahub"), default="auto",
                        help="Minute gateway to use; datahub bypasses an empty Promax answer for this repair batch")
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--skip-report", type=Path, action="append", default=[])
    parser.add_argument("--api-url", default="http://127.0.0.1:3018")
    parser.add_argument("--stopped-container", help="Offline maintenance: verify this application container is stopped instead of querying its API")
    parser.add_argument("--assume-stopped", action="store_true",
                        help="Offline maintenance: caller has already stopped the application container")
    args = parser.parse_args()
    manifest = pl.read_csv(args.manifest)
    if args.start_date:
        manifest = manifest.filter(pl.col("date") >= str(args.start_date))
    if args.end_date:
        manifest = manifest.filter(pl.col("date") <= str(args.end_date))
    if "symbol" in manifest.columns:
        stock_days = [(row["symbol"], date.fromisoformat(str(row["date"]))) for row in manifest.iter_rows(named=True)]
    elif {"market", "code"}.issubset(set(manifest.columns)):
        stock_days = [
            (f"{str(row['code']).zfill(6)}.{str(row['market']).upper()}", date.fromisoformat(str(row["date"])))
            for row in manifest.iter_rows(named=True)
        ]
    else:
        raise ValueError("manifest must contain symbol or market+code columns")
    attempted = set()
    for path in args.skip_report:
        previous = json.loads(path.read_text(encoding="utf-8"))
        attempted.update((r["symbol"].split(".")[0], date.fromisoformat(r["date"])) for r in previous.get("details", []))
    stock_days = [(s, d) for s, d in stock_days if (s.split(".")[0], d) not in attempted]
    if not stock_days:
        print(json.dumps({"stock_days": 0}))
        return
    if args.assume_stopped:
        jobs = {"jobs": []}
    elif args.stopped_container:
        from import_staged_minute_gaps import require_stopped
        require_stopped(args.stopped_container)
        jobs = {"jobs": []}
    else:
        with urlopen(args.api_url + "/api/pipeline/jobs?limit=1", timeout=10) as response:
            jobs = json.load(response)
    for job in jobs.get("jobs", []):
        if job.get("status") != "running":
            continue
        # An explicitly bounded old-history run can coexist with the scheduled
        # recent-minute stage, but never with its current-date partitions.
        from app.market_time import cn_today
        from app.services import preferences
        disjoint = (args.end_date is not None and job.get("stage") == "sync_minute"
                    and args.end_date < cn_today() - timedelta(days=preferences.get_minute_sync_days() + 1))
        if not disjoint:
            raise RuntimeError("Pipeline active; specify an older disjoint end-date or wait for completion")
    if args.source == "datahub":
        provider = TushareProvider()
        # The repair path still uses TushareProvider's strict historical
        # normalization and local daily reference checks, but directs this
        # bounded batch to the new gateway instead of waiting on Promax.
        provider._api = build_datahub_client()
        fallback = False
    else:
        provider, fallback, _error = _resolve_minute_provider("tushare")
    if provider is None or fallback:
        raise RuntimeError("Configured Tushare source unavailable")
    began = time.monotonic()
    def progress(done, total, message):
        if done % 10 == 0 or done == total:
            print(json.dumps({"requests": done, "total": total, "elapsed_s": round(time.monotonic() - began), "message": message}), flush=True)
    # This maintenance process never opens or edits the live DuckDB database.
    # Existing partition views see atomic replacements; caches reload on restart.
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=args.data_dir), _write_lock=RLock(),
                           rebuild_views=lambda: None, clear_cache=lambda: None)
    with _temporary_promax_timeout(provider._ensure_api(), args.timeout):
        if not args.date_gated:
            result = repair_minute_gaps(
                repo, start_date=min(d for _, d in stock_days),
                end_date=max(d for _, d in stock_days), provider_name="tushare",
                workers=args.workers, progress=progress, stock_days=stock_days,
                repair_conflicts=args.repair_conflicts, max_requests=args.max_requests,
                single_day_requests=args.single_day,
                source_provider=provider,
                # An explicitly selected gateway must be used as-is.  The
                # normal failover chain is only for `auto`; otherwise a
                # historical BJ repair can be silently routed through QMT/TDX
                # and reported as source_empty even when Promax has verified
                # rows for the exact exchange-qualified symbol.
                bypass_failover=args.source in {"promax", "datahub"},
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return

        # Historical repair is date-gated: a finished date is flushed and
        # reported before any request for the next date is issued.
        results = []
        for index, day in enumerate(sorted({day for _, day in stock_days}), start=1):
            day_pairs = [(symbol, pair_day) for symbol, pair_day in stock_days if pair_day == day]
            def day_progress(done, total, message, *, _day=day, _index=index):
                progress(done, total, f"date {_day} ({_index}/{len(set(d for _, d in stock_days))}) · {message}")
            result = repair_minute_gaps(
                repo, start_date=day, end_date=day, provider_name="tushare",
                workers=args.workers, progress=day_progress, stock_days=day_pairs,
                repair_conflicts=args.repair_conflicts, max_requests=args.max_requests,
                single_day_requests=True,
                source_provider=provider,
                bypass_failover=args.source in {"promax", "datahub"},
            )
            results.append(result)
            print(json.dumps({
                "date": str(day), "completed": True,
                "requests_checked": result.get("requests_checked", 0),
                "added_rows": result.get("added_rows", 0),
                "rejected_rows": result.get("rejected_rows", 0),
                "reasons": result.get("reasons", {}),
                "report_path": result.get("report_path"),
            }, ensure_ascii=False, default=str), flush=True)
        totals = Counter()
        reports = []
        for result in results:
            for key in ("requests_checked", "added_rows", "replaced_rows", "removed_rows",
                        "rejected_rows", "duplicate_rows", "repaired_stock_days"):
                totals[key] += int(result.get(key, 0) or 0)
            totals.update(result.get("reasons", {}))
            reports.append(result.get("report_path"))
        print(json.dumps({
            "date_gated": True, "dates_completed": len(results),
            "totals": dict(totals), "reports": reports,
        }, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
