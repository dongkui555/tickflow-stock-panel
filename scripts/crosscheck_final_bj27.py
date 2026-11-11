"""Read-only cross-check for the remaining BSE 2025 minute/K-line conflicts.

This script deliberately performs no writes. It compares the persisted QMT
minutes with StockDB and the Tushare-compatible DataHub, with special
attention to the BSE 15:01-15:30 fixed-price session.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from app.plugins.stockdb.provider import StockDBProvider
from app.plugins.tushare.provider import TushareProvider, build_datahub_client
from app.services.minute_repair import (
    COLS,
    PRICE,
    _daily_aggregate_exprs,
    checked_additions,
    daily_reference,
    regular,
)


def code_of(symbol: str) -> str:
    return symbol.split(".", 1)[0].zfill(6)


def load_target(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_daily(root: Path, day: date) -> dict:
    frame = pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet")
    return daily_reference(frame, data_dir=root)


def load_local_minutes(root: Path, day: date, code: str) -> pl.DataFrame:
    path = root / "kline_minute" / f"date={day}" / "part.parquet"
    if not path.exists():
        return pl.DataFrame(schema={column: pl.Float64 for column in COLS})
    symbol_code = pl.col("symbol").cast(pl.String).str.extract(r"(\d{6})", 1)
    return (
        pl.scan_parquet(path)
        .select(COLS)
        .filter(symbol_code == code)
        .collect()
        .sort("datetime")
    )


def aggregate(frame: pl.DataFrame, day: date) -> tuple[dict | None, int, int]:
    if frame.is_empty():
        return None, 0, 0
    full = regular(frame.select(COLS), day, include_opening=True, include_bj_after_hours=True).sort("datetime")
    session = regular(full, day)
    if full.is_empty():
        return None, 0, 0
    row = full.select(_daily_aggregate_exprs()).row(0, named=True)
    after_hours = full.filter(
        (pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute()).is_between(901, 930)
    ).height
    return row, session.height, after_hours


def classify(frame: pl.DataFrame, day: date, ref: dict | None, source: str) -> tuple[str, dict]:
    if frame.is_empty():
        return "source_empty", {"rows": 0}
    agg, session_rows, after_hours = aggregate(frame, day)
    if agg is None:
        return "source_invalid", {"rows": frame.height}
    _, reason = checked_additions(frame, frame.clear(), ref, day, source=source)
    return reason, {
        "rows": frame.height,
        "session_rows": session_rows,
        "after_hours_rows": after_hours,
        "aggregate": agg,
    }


def mismatch(left: dict | None, right: dict | None) -> dict | None:
    if not left or not right:
        return None
    result = {}
    for column in (*PRICE, "volume", "amount"):
        try:
            result[column] = float(left[column]) - float(right[column])
        except (KeyError, TypeError, ValueError):
            result[column] = None
    return result


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("/app/data")
    target = root / "import_reports" / "audit_2025_bj_after_qmt_20260922" / "missing_or_partial.csv"
    output = root / "import_reports" / "audit_2025_bj_after_qmt_20260922" / "crosscheck_sources.json"
    targets = load_target(target)
    stockdb, datahub = StockDBProvider(), TushareProvider()
    # Force the independent DataHub path; do not silently fall back to Promax.
    datahub._api = build_datahub_client()
    results = []
    try:
        for index, item in enumerate(targets, 1):
            day = date.fromisoformat(item["date"])
            symbol = item.get("selected_symbol") or f"{item['code']}.BJ"
            code = code_of(symbol)
            ref = load_daily(root, day).get(code)
            local = load_local_minutes(root, day, code)
            start = datetime.combine(day, time(9, 25))
            end = datetime.combine(day, time(15, 35))
            stock_frame, stock_issues = stockdb.get_minute_diagnostics([symbol], start, end)
            hub_frame, hub_issues = datahub.get_minute_diagnostics([symbol], start, end)
            local_status, local_info = classify(local, day, ref, "qmt")
            stock_status, stock_info = classify(stock_frame, day, ref, "stockdb")
            hub_status, hub_info = classify(hub_frame, day, ref, "datahub")
            row = {
                "date": str(day),
                "symbol": symbol,
                "code": code,
                "daily": {column: ref.get(column) for column in (*PRICE, "volume", "amount")} if ref else None,
                "qmt": {"status": local_status, **local_info},
                "stockdb": {"status": stock_status, **stock_info, "issues": stock_issues},
                "datahub": {"status": hub_status, **hub_info, "issues": hub_issues},
                "qmt_vs_stockdb": mismatch(local_info.get("aggregate"), stock_info.get("aggregate")),
                "qmt_vs_datahub": mismatch(local_info.get("aggregate"), hub_info.get("aggregate")),
            }
            results.append(row)
            print(json.dumps({
                "checked": index,
                "total": len(targets),
                "date": str(day),
                "symbol": symbol,
                "stockdb": stock_status,
                "datahub": hub_status,
                "stockdb_after_hours": stock_info.get("after_hours_rows", 0),
                "datahub_after_hours": hub_info.get("after_hours_rows", 0),
            }, ensure_ascii=False), flush=True)
    finally:
        stockdb.close()
        datahub.close()
    summary = {
        "target_count": len(results),
        "stockdb_status": dict(Counter(row["stockdb"]["status"] for row in results)),
        "datahub_status": dict(Counter(row["datahub"]["status"] for row in results)),
        "stockdb_after_hours_rows": sum(row["stockdb"].get("after_hours_rows", 0) for row in results),
        "datahub_after_hours_rows": sum(row["datahub"].get("after_hours_rows", 0) for row in results),
    }
    report = {"summary": summary, "results": results}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
