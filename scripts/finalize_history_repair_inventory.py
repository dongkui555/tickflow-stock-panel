"""Publish a resumable inventory; do not promote incomplete staged factors."""
import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from app.services.minute_repair import daily_reference, keyed, regular


def factor_coverage(frame, daily, day):
    if set(frame["trade_date"].to_list()) != {day}:
        raise ValueError("Staged factor date mismatch")
    if frame["symbol"].n_unique() != frame.height:
        raise ValueError("Staged factor symbol duplicates")
    if frame.filter(pl.col("symbol").is_null() | pl.col("ex_factor").is_null()
                    | ~pl.col("ex_factor").is_finite() | (pl.col("ex_factor") <= 0)).height:
        raise ValueError("Invalid staged factors")
    codes = set(daily.filter((pl.col("volume") > 0) & (pl.col("close") > 0))
                ["symbol"].cast(pl.String).str.extract(r"^(\d{1,6})(?:\.(?:SH|SZ|BJ))?$", 1)
                .str.pad_start(6, "0").drop_nulls().to_list())
    codes = {s for s in codes if s.startswith(("00", "30", "60", "68", "4", "8", "92"))}
    return sorted(codes - set(frame["symbol"].to_list()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.data_dir, args.output
    factors = pl.read_parquet(root / "adj_factor/all.parquet")
    first = factors["trade_date"].min()
    requested = sorted(date.fromisoformat(p.parent.name[5:]) for p in
                       (root / "kline_daily").glob("date=*/part.parquet"))
    requested = [d for d in requested if date(2024, 1, 1) <= d < first]
    staged, staged_rows, partial = set(), 0, []
    for path in (output / "factors").glob("*.parquet"):
        frame = pl.read_parquet(path)
        expected = date.fromisoformat(path.stem)
        daily = pl.read_parquet(root / "kline_daily" / f"date={expected}" / "part.parquet")
        absent = factor_coverage(frame, daily, expected)
        if absent:
            partial.append({"date": str(expected), "missing_symbols": absent})
        else:
            staged.add(expected)
        staged_rows += frame.height
    remaining = [str(d) for d in requested if d not in staged]
    (output / "remaining-factor-dates.json").write_text(json.dumps(remaining, indent=2), encoding="utf-8")
    missing, coverage = [], []
    for text in ("2026-09-07", "2026-09-08", "2026-09-09"):
        day = date.fromisoformat(text)
        references = daily_reference(pl.read_parquet(root / "kline_daily" / f"date={text}" / "part.parquet"), data_dir=root)
        minute = pl.read_parquet(root / "kline_minute" / f"date={text}" / "part.parquet")
        counts = dict(regular(keyed(minute), day).group_by("_code").agg(pl.col("datetime").n_unique()).iter_rows())
        full = sum(counts.get(code, 0) == 240 for code in references)
        for code in sorted(references):
            count = counts.get(code, 0)
            if count < 240:
                missing.append({"date": text, "symbol": code, "minutes_present": count, "missing_minutes": 240 - count})
        coverage.append({"date": text, "active": len(references), "complete": full,
                         "coverage_pct": round(full / len(references) * 100, 4) if references else None})
    pl.DataFrame(missing).write_parquet(output / "remaining-recent-minute-stock-days.parquet")
    summary = {"factor_history_requested_dates": len(requested), "factor_staged_dates": len(staged),
               "factor_partial_dates": partial,
               "factor_staged_rows": staged_rows, "factor_remaining_dates": len(remaining),
               "factor_production_start": str(first), "factor_production_end": str(factors["trade_date"].max()),
               "factor_promoted": False, "recent_minute_coverage": coverage,
               "recent_missing_stock_days": len(missing),
               "recent_missing_minutes": sum(row["missing_minutes"] for row in missing),
               "note": "Incomplete factors remain staged. Published forecasts and raw daily prices are unchanged."}
    (output / "final-inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
