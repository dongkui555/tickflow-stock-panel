from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _date_from_partition(path: Path) -> date | None:
    if not path.name.startswith("date="):
        return None
    try:
        return date.fromisoformat(path.name.removeprefix("date="))
    except ValueError:
        return None


def _partition_map(root: Path) -> dict[date, Path]:
    if not root.exists():
        return {}
    out: dict[date, Path] = {}
    for entry in root.glob("date=*"):
        day = _date_from_partition(entry)
        part = entry / "part.parquet"
        if day is not None and part.exists():
            out[day] = part
    return out


def _valid_parquet(path: Path) -> bool:
    try:
        if path.stat().st_size < 12:
            return False
        with path.open("rb") as fh:
            if fh.read(4) != b"PAR1":
                return False
            fh.seek(-4, 2)
            return fh.read(4) == b"PAR1"
    except OSError:
        return False


def _row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _low_day_detail(path: Path) -> dict[str, Any]:
    try:
        df = (
            pl.scan_parquet(str(path))
            .select(
                [
                    pl.len().alias("rows"),
                    pl.col("symbol").n_unique().alias("symbols"),
                    pl.col("datetime").min().alias("min_dt"),
                    pl.col("datetime").max().alias("max_dt"),
                ]
            )
            .collect()
        )
        return {
            "rows": int(df["rows"][0]),
            "symbols": int(df["symbols"][0]),
            "min_dt": df["min_dt"][0],
            "max_dt": df["max_dt"][0],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _group_ranges(days: list[date]) -> list[dict[str, Any]]:
    if not days:
        return []
    days = sorted(days)
    ranges: list[dict[str, Any]] = []
    start = prev = days[0]
    count = 1
    for day in days[1:]:
        if (day - prev).days <= 7:
            prev = day
            count += 1
            continue
        ranges.append({"start": start, "end": prev, "trading_days": count})
        start = prev = day
        count = 1
    ranges.append({"start": start, "end": prev, "trading_days": count})
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TickFlow kline_minute coverage against kline_daily trading days.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--min-full-rows", type=int, default=1_000_000)
    parser.add_argument("--latest-days", type=int, default=30)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    daily = _partition_map(data_dir / "kline_daily")
    minute = _partition_map(data_dir / "kline_minute")
    expected = sorted(day for day in daily if start <= day <= end)

    missing = [day for day in expected if day not in minute]
    invalid: list[date] = []
    low: list[dict[str, Any]] = []
    full_like = 0
    row_stats: dict[date, int] = {}
    for day in expected:
        path = minute.get(day)
        if path is None:
            continue
        if not _valid_parquet(path):
            invalid.append(day)
            continue
        rows = _row_count(path)
        row_stats[day] = rows
        if rows >= args.min_full_rows:
            full_like += 1
        else:
            detail = _low_day_detail(path)
            low.append({"date": day, **detail})

    latest_expected = expected[-args.latest_days :] if args.latest_days > 0 else []
    latest: list[dict[str, Any]] = []
    for day in latest_expected:
        path = minute.get(day)
        if path is None:
            latest.append({"date": day, "status": "missing"})
        elif day in invalid:
            latest.append({"date": day, "status": "invalid"})
        else:
            rows = row_stats.get(day, _row_count(path))
            item: dict[str, Any] = {"date": day, "status": "full_like" if rows >= args.min_full_rows else "low", "rows": rows}
            if rows < args.min_full_rows:
                item.update(_low_day_detail(path))
            latest.append(item)

    result = {
        "window": {"start": start, "end": end},
        "expected_trading_days": len(expected),
        "minute_partition_days": sum(1 for day in expected if day in minute),
        "full_like_days": full_like,
        "missing_full_day_count": len(missing),
        "missing_full_day_ranges": _group_ranges(missing),
        "invalid_parquet_days": [day.isoformat() for day in invalid],
        "low_coverage_count": len(low),
        "low_coverage_ranges": _group_ranges([item["date"] for item in low]),
        "low_coverage_days": low,
        "latest_days": latest,
    }
    print(json.dumps(result, default=_json_default, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
