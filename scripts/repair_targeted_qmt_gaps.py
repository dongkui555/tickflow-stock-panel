"""Repair the explicitly requested Shanghai/Beijing minute windows from local QMT."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from app.plugins.qmt.provider import QmtProvider
from app.services.minute_repair import repair_minute_gaps
from app.tickflow.repository import DataStore, KlineRepository


TARGETS = (
    ("BJ", date(2024, 1, 1), date(2024, 1, 31)),
    ("SH", date(2024, 1, 2), date(2024, 1, 12)),
    ("BJ", date(2024, 2, 22), date(2024, 2, 27)),
    ("BJ", date(2024, 4, 11), date(2024, 4, 16)),
    ("BJ", date(2024, 5, 31), date(2024, 6, 5)),
    ("BJ", date(2024, 7, 1), date(2024, 8, 23)),
    ("SH", date(2024, 7, 1), date(2024, 8, 23)),
    ("BJ", date(2025, 7, 3), date(2025, 7, 8)),
    ("BJ", date(2025, 8, 19), date(2025, 8, 22)),
    ("BJ", date(2025, 10, 13), date(2025, 10, 14)),
    ("BJ", date(2026, 3, 4), date(2026, 3, 4)),
    ("BJ", date(2026, 9, 7), date(2026, 9, 15)),
    ("SH", date(2026, 9, 7), date(2026, 9, 15)),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    issues = pl.read_csv(args.audit, try_parse_dates=True).with_columns(
        pl.col("code").cast(pl.String).str.pad_start(6, "0")
    )
    filters = []
    for market, start, end in TARGETS:
        filters.append(
            (pl.col("market") == market)
            & pl.col("date").is_between(start, end)
            & pl.col("status").is_in(["missing_0", "invalid_partial"])
        )
    selected = issues.filter(pl.any_horizontal(filters)).unique(["code", "date"])
    stock_days = sorted(selected.select("code", "date").iter_rows(), key=lambda row: (row[0], row[1]))
    print(json.dumps({
        "selected_stock_days": len(stock_days),
        "by_market": selected.group_by("market").len().sort("market").to_dicts(),
        "by_status": selected.group_by("status").len().sort("status").to_dicts(),
    }, ensure_ascii=False), flush=True)
    if not stock_days:
        return

    def progress(done: int, total: int, message: str) -> None:
        if done == total or done % 20 == 0:
            print(json.dumps({"done": done, "total": total, "message": message}, ensure_ascii=False), flush=True)

    repo = KlineRepository(DataStore())
    result = repair_minute_gaps(
        repo,
        start_date=min(day for _, day in stock_days),
        end_date=max(day for _, day in stock_days),
        provider_name="qmt",
        source_provider=QmtProvider(),
        stock_days=stock_days,
        workers=args.workers,
        repair_conflicts=True,
        single_day_requests=False,
        progress=progress,
    )
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
