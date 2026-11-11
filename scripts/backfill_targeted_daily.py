"""Backfill selected stock daily partitions without rewriting valid local rows.

Run inside the application container so it reuses the configured Tushare/Promax
provider and the production repository schema. Existing partitions are backed up
before missing symbol/date rows are appended.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pandas as pd

from app.plugins.tushare.provider import TushareProvider, _compact_symbol, _normalize_dates
from app.tickflow.repository import DataStore, KlineRepository


DAILY_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--markets",
        default="BJ,SH",
        help="Comma-separated markets: BJ,SH,SZ",
    )
    parser.add_argument(
        "--only-missing-dates",
        action="store_true",
        help="Skip dates that already have a local daily partition.",
    )
    return parser.parse_args()


def market_expr(markets: set[str]) -> pl.Expr:
    code = pl.col("symbol").cast(pl.Utf8).str.extract(r"(\d{6})", 1)
    tests: list[pl.Expr] = []
    if "SH" in markets:
        tests.append(code.str.contains(r"^6"))
    if "SZ" in markets:
        tests.append(code.str.contains(r"^(0|3)"))
    if "BJ" in markets:
        tests.append(code.str.contains(r"^(4|8|92)"))
    if not tests:
        raise ValueError("markets must contain at least one of BJ, SH, SZ")
    result = tests[0]
    for test in tests[1:]:
        result = result | test
    return result


def validate_daily(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    if frame.is_empty():
        return frame, {"received": 0, "valid": 0, "rejected": 0}
    missing = [column for column in DAILY_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"daily response missing columns: {missing}")
    frame = frame.select(DAILY_COLUMNS).with_columns(
        pl.col("symbol").cast(pl.Utf8).str.extract(r"(\d{6})", 1),
        pl.col("date").cast(pl.Date, strict=False),
        *[
            pl.col(column).cast(pl.Float64, strict=False)
            for column in ("open", "high", "low", "close", "volume", "amount")
        ],
    )
    received = frame.height
    frame = frame.filter(
        pl.col("symbol").is_not_null()
        & pl.col("date").is_not_null()
        & (pl.col("open") > 0)
        & (pl.col("high") >= pl.max_horizontal("open", "close"))
        & (pl.col("low") <= pl.min_horizontal("open", "close"))
        & (pl.col("low") > 0)
        & (pl.col("volume") >= 0)
        & (pl.col("amount") >= 0)
    )
    frame = frame.unique(subset=["symbol", "date"], keep="last").sort(["date", "symbol"])
    return frame, {
        "received": received,
        "valid": frame.height,
        "rejected": received - frame.height,
    }


def fetch_paged_daily(provider: TushareProvider, trade_days: list[date]) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Promax caps daily responses at 5,000 rows, so a full market needs paging."""
    api = provider._ensure_api()
    frames: list[pl.DataFrame] = []
    pages: list[dict[str, object]] = []
    for index, day in enumerate(trade_days, 1):
        for attempt in range(3):
            try:
                part = provider._promax_market_day(api.daily, day.strftime("%Y%m%d"), page_size=5000)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt * 3)
        if part.empty:
            raise RuntimeError(f"daily source returned no rows for trading day {day}")
        total = len(part)
        part = _normalize_dates(part, "trade_date")
        part["symbol"] = part["ts_code"].astype(str).map(_compact_symbol)
        part["date"] = part["trade_date"]
        part["volume"] = pd.to_numeric(part["vol"], errors="coerce") * 100.0
        part["amount"] = pd.to_numeric(part["amount"], errors="coerce") * 1000.0
        frames.append(pl.from_pandas(part[DAILY_COLUMNS]))
        pages.append({"date": str(day), "source_rows": total, "pages": (total - 1) // 5000 + 1})
        print(f"daily fetch {index}/{len(trade_days)}: {total} rows", flush=True)
    return pl.concat(frames, how="diagonal_relaxed"), pages


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start > end:
        raise ValueError("start must not be after end")
    markets = {item.strip().upper() for item in args.markets.split(",") if item.strip()}

    store = DataStore()
    repo = KlineRepository(store)
    provider = TushareProvider()
    start_key = start.strftime("%Y%m%d")
    end_key = end.strftime("%Y%m%d")
    trade_keys = provider._trade_dates(start_key, end_key)
    trade_days = [datetime.strptime(value, "%Y%m%d").date() for value in trade_keys]
    if args.only_missing_dates:
        trade_days = [
            day
            for day in trade_days
            if not (store.data_dir / "kline_daily" / f"date={day}" / "part.parquet").exists()
        ]
    if not trade_days:
        print(json.dumps({"status": "nothing_to_do", "start": str(start), "end": str(end)}, ensure_ascii=False))
        return

    print(f"fetching {len(trade_days)} trade dates from {trade_days[0]} to {trade_days[-1]}", flush=True)
    report_id = "targeted_daily_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = store.data_dir / "import_reports" / report_id
    backup_dir = store.data_dir / "import_backups" / report_id
    report_dir.mkdir(parents=True, exist_ok=False)

    source_pages: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    quality = {"received": 0, "valid": 0, "rejected": 0}
    market_rows = 0
    added_rows = 0
    per_date: list[dict[str, object]] = []
    for day in trade_days:
        try:
            fetched, pages = fetch_paged_daily(provider, [day])
            day_frame, day_quality = validate_daily(fetched)
            day_frame = day_frame.filter(market_expr(markets))
            if day_frame.is_empty():
                raise RuntimeError("provider returned no valid rows for requested markets")
            source_pages.extend(pages)
            for key in quality:
                quality[key] += day_quality[key]
            market_rows += day_frame.height
        except Exception as exc:
            failures.append({"date": str(day), "error": f"{type(exc).__name__}: {exc}"})
            print(f"daily fetch failed {day}: {type(exc).__name__}", flush=True)
            continue
        path = store.data_dir / "kline_daily" / f"date={day}" / "part.parquet"
        existing = pl.DataFrame()
        if path.exists():
            existing = pl.read_parquet(path)
            existing_codes = existing.select(
                pl.col("symbol").cast(pl.Utf8).str.extract(r"(\d{6})", 1).alias("symbol")
            ).unique()
            missing = day_frame.join(existing_codes, on="symbol", how="anti")
        else:
            missing = day_frame
        if not missing.is_empty():
            if path.exists():
                destination = backup_dir / f"date={day}" / "part.parquet"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            repo.append_daily(missing)
            added_rows += missing.height
        per_date.append(
            {
                "date": str(day),
                "provider_rows": day_frame.height,
                "existing_rows": existing.height,
                "added_rows": missing.height,
            }
        )
        stored = pl.read_parquet(path)
        if stored.height - stored.unique(subset=["symbol", "date"]).height:
            raise RuntimeError(f"duplicate daily keys after append: {day}")

    verification: list[dict[str, object]] = []
    for item in per_date:
        day = item["date"]
        path = store.data_dir / "kline_daily" / f"date={day}" / "part.parquet"
        stored = pl.read_parquet(path) if path.exists() else pl.DataFrame()
        duplicate_count = 0
        if not stored.is_empty():
            duplicate_count = stored.height - stored.unique(subset=["symbol", "date"]).height
        verification.append(
            {
                **item,
                "stored_rows": stored.height,
                "duplicate_keys": duplicate_count,
                "verified": path.exists() and duplicate_count == 0,
            }
        )

    pl.DataFrame(verification).write_csv(report_dir / "by_date.csv")
    pl.DataFrame(source_pages).write_csv(report_dir / "source_pages.csv")
    result = {
        "status": "completed",
        "start": str(start),
        "end": str(end),
        "markets": sorted(markets),
        "trade_dates": len(trade_days),
        "provider_quality": quality,
        "market_rows": market_rows,
        "added_rows": added_rows,
        "failed_dates": failures,
        "verified_dates": sum(bool(item["verified"]) for item in verification),
        "report_dir": str(report_dir),
        "backup_dir": str(backup_dir) if backup_dir.exists() else None,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
