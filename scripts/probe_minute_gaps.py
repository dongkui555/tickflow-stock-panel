"""Read-only representative probes before requesting large historical backfills."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from app.services.kline_sync import _resolve_minute_provider
from app.services.minute_repair import checked_additions, daily_reference
from app.services.tushare_ext_sync import _temporary_promax_timeout


def main(root: Path, report: Path):
    gaps = pl.read_csv(report / "minute_required_stock_days.csv")
    # Sample both early and recent dates and both mainland/BJ markets.
    samples = gaps.with_columns(
        pl.col("date").str.slice(0, 7).alias("month"),
        pl.col("symbol").str.ends_with(".BJ").alias("beijing"),
    ).sort("date", "symbol").group_by("month", "beijing", maintain_order=True).first()
    indices = sorted(set(round(i * (samples.height - 1) / 9) for i in range(10))) if samples.height else []
    provider, fallback, error = _resolve_minute_provider("tushare")
    if not provider or fallback:
        raise RuntimeError(error or "Tushare unavailable")
    results = []
    with _temporary_promax_timeout(provider._ensure_api(), 8):
        for index in indices:
            row = samples.row(index, named=True)
            day = date.fromisoformat(row["date"])
            code = row["symbol"].split(".")[0]
            item = {"date": str(day), "symbol": row["symbol"]}
            try:
                source = provider.get_minute([code], datetime.combine(day, time(9, 30)),
                                             datetime.combine(day, time(15)), freq="1m")
                refs = daily_reference(pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet"), data_dir=root)
                if source.is_empty():
                    item.update(rows=0, result="source_empty")
                else:
                    verified, reason = checked_additions(source, source.clear(), refs.get(code), day)
                    item.update(rows=source.height, result=reason, verified_rows=verified.height)
            except Exception as exc:
                item.update(result=type(exc).__name__)
            results.append(item)
            (report / "minute_online_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(item), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.report_dir)
