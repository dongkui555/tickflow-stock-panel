"""Stage a bounded local StockDB sample; production partitions are read-only."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from app.config import settings
from app.plugins.stockdb.provider import StockDBProvider
from app.services.minute_repair import _daily_aggregate_exprs, checked_additions, daily_reference, keyed, regular


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.data_dir.resolve()):
        raise ValueError("Stage output must be outside production data")
    settings.stockdb_base_url = args.base_url
    client = StockDBProvider()
    try:
        frame, issues = client.get_minute_diagnostics(
            args.symbols.split(","), datetime.combine(args.start, time(9, 30)), datetime.combine(args.end, time(15)))
    finally:
        client.close()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    if not frame.is_empty():
        for day in sorted(frame["datetime"].dt.date().unique()):
            path = args.data_dir / f"kline_daily/date={day}/part.parquet"
            refs = daily_reference(pl.read_parquet(path), data_dir=args.data_dir) if path.exists() else {}
            sample = keyed(frame.filter(pl.col("datetime").dt.date() == day))
            old_path = args.data_dir / f"kline_minute/date={day}/part.parquet"
            old = keyed(pl.read_parquet(old_path)) if old_path.exists() else sample.clear()
            for code in sorted(sample["_code"].unique()):
                block = sample.filter(pl.col("_code") == code).drop("_code").sort("datetime")
                existing = old.filter(pl.col("_code") == code).drop("_code")
                accepted, reason = checked_additions(block, block.clear(), refs.get(code), day)
                delta, merge_reason = checked_additions(block, existing, refs.get(code), day)
                observed = block.select(_daily_aggregate_exprs()).row(0, named=True)
                observed["volume"] *= 100
                result = {"symbol": code, "date": str(day), "source_rows": block.height,
                          "validation": reason, "staged_rows": accepted.height,
                          "existing_regular_minutes": regular(existing, day).height,
                          "merge_validation": merge_reason, "merge_additions": delta.height,
                          "source_aggregate_shares_yuan": observed, "daily_reference": refs.get(code)}
                if reason == "accepted":
                    accepted.write_parquet(args.output / f"{code}-{day}.parquet")
                results.append(result)
    report = {"source": "local_stockdb", "production_written": False, "requests": issues, "validation": results}
    (args.output / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
