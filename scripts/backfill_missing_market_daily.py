"""Append daily bars for market/date gaps without overwriting existing keys.

The minute audit can report a missing market on a date even when the daily
partition itself exists for other markets. This command consumes that exact
gap list, fetches one complete market page per date from the configured
Promax/Tushare provider, and appends only absent symbol/date keys.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import date
from pathlib import Path

import polars as pl

from app.plugins.tushare.provider import TushareProvider, _compact_symbol
from app.tickflow.repository import DataStore, KlineRepository


DAILY_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
MARKET_PATTERNS = {
    "BJ": r"^(43|83|87|88|92)",
    "SH": r"^(60|68)",
    "SZ": r"^(00|30)",
}


def _normalize_daily(frame, day: date, market: str) -> pl.DataFrame:
    """Convert one Promax market response to the local daily contract."""
    empty_schema = {
        "symbol": pl.String,
        "date": pl.Date,
        **{column: pl.Float64 for column in DAILY_COLUMNS[2:]},
    }
    if frame is None or frame.empty:
        return pl.DataFrame(schema=empty_schema)
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"daily response missing columns: {sorted(missing)}")
    expected = day.strftime("%Y%m%d")
    raw_dates = frame["trade_date"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    if not bool((raw_dates == expected).all()):
        raise RuntimeError("daily response contains another trade date")
    suffixes = frame["ts_code"].astype(str).str.upper().str.rsplit(".").str[-1]
    frame = frame.loc[suffixes == market].copy()
    frame["symbol"] = frame["ts_code"].astype(str).map(_compact_symbol)
    frame["date"] = day
    frame["volume"] = frame["vol"] * 100.0
    frame["amount"] = frame["amount"] * 1000.0
    out = pl.from_pandas(frame[DAILY_COLUMNS]).with_columns(
        pl.col("symbol").cast(pl.String).str.zfill(6),
        pl.col("date").cast(pl.Date),
        *[pl.col(column).cast(pl.Float64, strict=False) for column in DAILY_COLUMNS[2:]],
    )
    out = out.filter(pl.col("symbol").str.contains(MARKET_PATTERNS[market]))
    if out.height != out.unique(subset=["symbol", "date"]).height:
        raise RuntimeError("daily response contains duplicate symbol/date keys")
    valid = (
        (pl.col("open") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close"))
        & (pl.col("low") <= pl.min_horizontal("open", "close")) & (pl.col("low") > 0)
        & (pl.col("volume") >= 0) & (pl.col("amount") >= 0)
    )
    if not out.select(valid.all()).item():
        raise RuntimeError("daily response contains invalid OHLC or turnover")
    return out.sort("symbol")


def _existing_codes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(
        pl.read_parquet(path, columns=["symbol"])
        .get_column("symbol")
        .cast(pl.String)
        .str.extract(r"(\d{6})", 1)
        .drop_nulls()
        .to_list()
    )


def _official_daily(api, trade_date: str):
    """Fetch one complete official-Tushare daily snapshot for fallback only."""
    frame = api.daily(trade_date=trade_date, limit=6000, offset=0)
    if frame is None or frame.empty:
        return frame
    if len(frame) >= 6000:
        raise RuntimeError("official Tushare daily response reached the page limit")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--official-fallback",
        action="store_true",
        help="Use the configured official Tushare token only for dates Promax cannot fetch.",
    )
    args = parser.parse_args()
    root = args.data_dir.resolve()
    gap_path = args.gap_report.resolve()
    report_dir = args.report_dir.resolve()
    if not gap_path.is_relative_to(root / "import_reports"):
        raise ValueError("gap report must be inside data/import_reports")
    if not report_dir.is_relative_to(root / "import_reports") or report_dir == root / "import_reports":
        raise ValueError("report directory must be a new directory inside data/import_reports")
    if args.retries < 1:
        raise ValueError("retries must be positive")
    report_dir.mkdir(parents=True, exist_ok=False)

    gaps = pl.read_csv(gap_path).select(["date", "market"]).unique().sort(["date", "market"])
    targets = [(date.fromisoformat(row["date"]), row["market"]) for row in gaps.to_dicts()]
    if not targets:
        raise RuntimeError("gap report has no market/date targets")
    provider = TushareProvider()
    if provider._connection_mode() != "promax":
        raise RuntimeError("this targeted backfill requires TUSHARE_MODE=promax")
    official_api = None
    if args.official_fallback:
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if token:
            import tushare as ts

            official_api = ts.pro_api(token)
    store = DataStore(root)
    repo = KlineRepository(store)
    report_id = report_dir.name
    backup_root = root / "import_backups" / report_id
    rows: list[dict] = []
    total_added = 0
    try:
        api = provider._ensure_api()
        for index, (day, market) in enumerate(targets, 1):
            detail = {"date": str(day), "market": market, "attempts": 0}
            try:
                fetched = None
                error = None
                source = "promax"
                for attempt in range(1, args.retries + 1):
                    detail["attempts"] = attempt
                    try:
                        fetched = provider._promax_market_day(
                            api.daily, day.strftime("%Y%m%d"), page_size=5000
                        )
                        error = None
                        break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        if attempt < args.retries:
                            time.sleep(2 ** (attempt - 1) * 2)
                if (error is not None or fetched is None) and official_api is not None:
                    try:
                        fetched = _official_daily(official_api, day.strftime("%Y%m%d"))
                        error = None
                        source = "official_tushare_fallback"
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                if error is not None or fetched is None:
                    raise RuntimeError(error or "empty response")
                market_frame = _normalize_daily(fetched, day, market)
                path = root / "kline_daily" / f"date={day}" / "part.parquet"
                existing = _existing_codes(path)
                additions = market_frame.filter(~pl.col("symbol").is_in(sorted(existing)))
                detail.update(
                    source_rows=market_frame.height,
                    existing_market_rows=len(existing.intersection(set(market_frame["symbol"].to_list()))),
                    added_rows=additions.height,
                    status="no_change" if additions.is_empty() else "appended",
                )
                detail["source"] = source
                if not additions.is_empty():
                    if path.exists():
                        backup = backup_root / f"date={day}" / "part.parquet"
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, backup)
                    repo.append_daily(additions)
                    total_added += additions.height
                    stored = pl.read_parquet(path)
                    duplicate_keys = stored.height - stored.unique(subset=["symbol", "date"]).height
                    if duplicate_keys:
                        raise RuntimeError(f"duplicate daily keys after append: {duplicate_keys}")
                rows.append(detail)
                print(json.dumps({"done": index, "total": len(targets), **detail}, ensure_ascii=False), flush=True)
            except Exception as exc:
                detail.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                rows.append(detail)
                print(json.dumps({"done": index, "total": len(targets), **detail}, ensure_ascii=False), flush=True)
    finally:
        provider.close()
    result = {
        "status": "completed" if all(row["status"] != "failed" for row in rows) else "partial",
        "targets": len(targets),
        "added_rows": total_added,
        "by_status": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
        "report_dir": str(report_dir),
        "backup_dir": str(backup_root) if backup_root.exists() else None,
        "source": "Tushare Promax daily paginated by trade date with optional official fallback",
    }
    (report_dir / "by_date.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
