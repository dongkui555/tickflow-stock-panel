"""Persist authoritative market calendar and historical suspension evidence.

This command is deliberately separate from minute repair.  A failed status
query never replaces an existing snapshot, and an empty response is reported
without being treated as proof that no historical suspensions exist.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from app.plugins.tushare.provider import TushareProvider


def _chunks(start: str, end: str, days: int = 180):
    current = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    finish = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
    while current <= finish:
        chunk_end = min(finish, current + timedelta(days=days - 1))
        yield current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        current = chunk_end + timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = str(args.start).replace("-", "")[:8]
    end = str(args.end).replace("-", "")[:8]
    if len(start) != 8 or len(end) != 8:
        raise ValueError("start/end 必须是 YYYY-MM-DD 或 YYYYMMDD")

    provider = TushareProvider()
    calendar = []
    suspensions = []
    errors = []
    try:
        for chunk_start, chunk_end in _chunks(start, end):
            try:
                calendar.extend(provider.get_trade_calendar(chunk_start, chunk_end))
            except Exception as exc:  # noqa: BLE001
                errors.append({"kind": "trade_calendar", "start": chunk_start,
                               "end": chunk_end, "error": type(exc).__name__})
            try:
                suspensions.extend(provider.get_historical_suspensions(chunk_start, chunk_end))
            except Exception as exc:  # noqa: BLE001
                errors.append({"kind": "suspension", "start": chunk_start,
                               "end": chunk_end, "error": type(exc).__name__})
    finally:
        provider.close()

    root = args.data_dir
    calendar_path = root / "market_calendar" / "trade_calendar.parquet"
    suspension_path = root / "instruments_history" / "suspensions.parquet"
    if not args.dry_run:
        if calendar:
            calendar_path.parent.mkdir(parents=True, exist_ok=True)
            frame = pl.DataFrame(calendar)
            if calendar_path.exists():
                frame = pl.concat([pl.read_parquet(calendar_path), frame], how="diagonal_relaxed")
            frame.unique(subset=["trade_date"], keep="last").sort("trade_date").write_parquet(
                calendar_path.with_suffix(".tmp.parquet")
            )
            calendar_path.with_suffix(".tmp.parquet").replace(calendar_path)
        if suspensions:
            suspension_path.parent.mkdir(parents=True, exist_ok=True)
            frame = pl.DataFrame(suspensions)
            if suspension_path.exists():
                frame = pl.concat([pl.read_parquet(suspension_path), frame], how="diagonal_relaxed")
            frame.unique(
                subset=["symbol", "suspend_date", "resume_date"], keep="last"
            ).sort(["symbol", "suspend_date"]).write_parquet(
                suspension_path.with_suffix(".tmp.parquet")
            )
            suspension_path.with_suffix(".tmp.parquet").replace(suspension_path)

    result = {
        "start": start,
        "end": end,
        "calendar_rows": len(calendar),
        "suspension_rows": len(suspensions),
        "errors": errors,
        "written": not args.dry_run,
        "calendar_path": str(calendar_path),
        "suspension_path": str(suspension_path),
        "warning": (
            "空响应未覆盖旧快照; 请检查权限或源状态, 不代表历史无停牌记录。"
            if not calendar or not suspensions else None
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
