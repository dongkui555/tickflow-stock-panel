"""Backfill an exact minute-gap manifest from the local StockDB service."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import date
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import polars as pl

from app.config import settings
from app.plugins.stockdb.provider import StockDBProvider
from app.services.minute_repair import repair_minute_gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument(
        "--repair-conflicts",
        action="store_true",
        help="Replace existing rows only after the incoming full day passes independent validation.",
    )
    args = parser.parse_args()

    manifest = pl.read_csv(args.manifest, try_parse_dates=True)
    manifest = manifest.filter(pl.col("status").is_in(["missing_0", "invalid_partial"]))
    if args.start_date:
        manifest = manifest.filter(pl.col("date") >= args.start_date)
    if args.end_date:
        manifest = manifest.filter(pl.col("date") <= args.end_date)
    stock_days = [
        (f"{str(row['code']).zfill(6)}.{str(row['market']).upper()}", row["date"])
        for row in manifest.select("market", "code", "date").unique(["market", "code", "date"]).iter_rows(named=True)
    ]
    print(json.dumps({"selected_stock_days": len(stock_days)}, ensure_ascii=False), flush=True)
    if not stock_days:
        return

    settings.stockdb_base_url = args.base_url
    settings.stockdb_enabled = True
    provider = StockDBProvider()
    repo = SimpleNamespace(
        store=SimpleNamespace(data_dir=args.data_dir),
        _write_lock=RLock(),
        rebuild_views=lambda: None,
        clear_cache=lambda: None,
    )
    began = time.monotonic()

    def progress(done: int, total: int, message: str) -> None:
        if done % 10 == 0 or done == total:
            print(json.dumps({
                "requests": done,
                "total": total,
                "elapsed_s": round(time.monotonic() - began),
                "message": message,
            }, ensure_ascii=False), flush=True)

    try:
        results = []
        for index, day in enumerate(sorted({day for _, day in stock_days}), start=1):
            day_pairs = [(symbol, pair_day) for symbol, pair_day in stock_days if pair_day == day]
            result = repair_minute_gaps(
                repo,
                start_date=day,
                end_date=day,
                provider_name="stockdb",
                source_provider=provider,
                stock_days=day_pairs,
                workers=args.workers,
                single_day_requests=True,
                repair_conflicts=args.repair_conflicts,
                progress=lambda done, total, message, _day=day, _index=index: progress(
                    done, total, f"date {_day} ({_index}/{len(set(d for _, d in stock_days))}) · {message}"
                ),
                bypass_failover=True,
            )
            results.append(result)
            print(json.dumps({
                "date": str(day),
                "completed": True,
                "requests_checked": result.get("requests_checked", 0),
                "added_rows": result.get("added_rows", 0),
                "rejected_rows": result.get("rejected_rows", 0),
                "reasons": result.get("reasons", {}),
                "report_path": result.get("report_path"),
            }, ensure_ascii=False, default=str), flush=True)
        totals = Counter()
        reports = []
        for result in results:
            for key in (
                "requests_checked", "added_rows", "replaced_rows", "removed_rows",
                "rejected_rows", "duplicate_rows", "repaired_stock_days",
            ):
                totals[key] += int(result.get(key, 0) or 0)
            totals.update(result.get("reasons", {}))
            reports.append(result.get("report_path"))
        print(json.dumps({
            "date_gated": True,
            "dates_completed": len(results),
            "totals": dict(totals),
            "reports": reports,
        }, ensure_ascii=False, default=str), flush=True)
    finally:
        provider.close()


if __name__ == "__main__":
    main()
