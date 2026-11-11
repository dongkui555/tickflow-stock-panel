"""Diagnose BSE minute/daily conflicts without modifying market data."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, time
from pathlib import Path

import polars as pl

from app.services.minute_repair import daily_reference, regular
from repair_targeted_qmt_dat import QmtDatProvider


def _aggregate(frame: pl.DataFrame) -> dict:
    if frame.is_empty():
        return {}
    return frame.sort("datetime").select(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("amount").sum().alias("amount"),
        pl.col("datetime").n_unique().alias("rows"),
    ).row(0, named=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qmt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = (
        pl.read_csv(args.manifest, try_parse_dates=True)
        .filter((pl.col("market") == "BJ") & pl.col("status").is_in(["missing_0", "invalid_partial"]))
        .with_columns(pl.col("code").cast(pl.String).str.zfill(6))
        .unique(["date", "code"])
        .sort(["date", "code"])
    )
    provider = QmtDatProvider(args.qmt_root)
    refs_by_day: dict = {}
    rows: list[dict] = []

    for item in targets.iter_rows(named=True):
        day = item["date"]
        code = item["code"]
        if day not in refs_by_day:
            daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
            refs_by_day[day] = (
                daily_reference(pl.read_parquet(daily_path), data_dir=root)
                if daily_path.exists()
                else {}
            )
        reference = refs_by_day[day].get(code)
        source = provider.get_minute(
            [f"{code}.BJ"],
            datetime.combine(day, time(9, 25)),
            datetime.combine(day, time(15, 30)),
        )
        regular_frame = regular(source, day)
        allowed_frame = regular(source, day, include_opening=True, include_bj_after_hours=True)
        aggregate = _aggregate(allowed_frame)
        regular_aggregate = _aggregate(regular_frame)
        clocks = (
            allowed_frame.select(
                (pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute()).alias("clock")
            )["clock"].to_list()
            if not allowed_frame.is_empty()
            else []
        )
        opening_rows = sum(1 for value in clocks if value == 570)
        after_hours_rows = sum(1 for value in clocks if 901 <= value <= 930)
        result = {
            "date": str(day),
            "symbol": f"{code}.BJ",
            "local_status": item["status"],
            "qmt_rows": int(aggregate.get("rows", 0)),
            "qmt_regular_rows": int(regular_aggregate.get("rows", 0)),
            "qmt_opening_rows": opening_rows,
            "qmt_after_hours_rows": after_hours_rows,
        }
        if reference is None:
            result["classification"] = "daily_reference_missing"
            rows.append(result)
            continue
        result.update({
            "daily_open": float(reference["open"]),
            "daily_high": float(reference["high"]),
            "daily_low": float(reference["low"]),
            "daily_close": float(reference["close"]),
            "daily_volume": float(reference["volume"]),
            "daily_amount": float(reference["amount"]),
        })
        if not aggregate:
            result["classification"] = "qmt_empty"
            rows.append(result)
            continue
        result.update({
            "qmt_open": float(aggregate["open"]),
            "qmt_high": float(aggregate["high"]),
            "qmt_low": float(aggregate["low"]),
            "qmt_close": float(aggregate["close"]),
            "qmt_volume": float(aggregate["volume"]),
            "qmt_amount": float(aggregate["amount"]),
        })
        tolerance = max(0.011, abs(float(reference["close"])) * 1e-6)
        price_ok = (
            abs(float(aggregate["close"]) - float(reference["close"])) <= tolerance
            and float(aggregate["high"]) <= float(reference["high"]) + tolerance
            and float(aggregate["low"]) >= float(reference["low"]) - tolerance
            and float(reference["low"]) - tolerance
            <= float(aggregate["open"])
            <= float(reference["high"]) + tolerance
        )
        amount_gap = float(reference["amount"]) - float(aggregate["amount"])
        amount_gap_pct = amount_gap / float(reference["amount"]) if reference["amount"] else 0.0
        best_volume_ratio = min(
            (abs(float(aggregate["volume"]) * scale - float(reference["volume"])), scale)
            for scale in (0.01, 1.0, 100.0)
        )[1]
        normalized_volume = float(aggregate["volume"]) * best_volume_ratio
        volume_gap = float(reference["volume"]) - normalized_volume
        volume_gap_pct = volume_gap / float(reference["volume"]) if reference["volume"] else 0.0
        result.update({
            "price_path_matches_daily": price_ok,
            "best_volume_scale": best_volume_ratio,
            "normalized_qmt_volume": normalized_volume,
            "volume_gap": volume_gap,
            "volume_gap_pct": volume_gap_pct,
            "amount_gap": amount_gap,
            "amount_gap_pct": amount_gap_pct,
        })
        if price_ok and amount_gap > 0 and opening_rows == 0 and after_hours_rows == 0:
            result["classification"] = "non_regular_turnover_missing"
        elif not price_ok:
            result["classification"] = "price_path_conflict"
        else:
            result["classification"] = "turnover_unit_or_reference_conflict"
        rows.append(result)

    details = pl.DataFrame(rows)
    details.write_csv(output / "details.csv")
    summary = {
        "stock_days": len(rows),
        "classifications": dict(Counter(row["classification"] for row in rows)),
        "no_opening_snapshot": sum(row.get("qmt_opening_rows", 0) == 0 for row in rows),
        "no_after_hours_rows": sum(row.get("qmt_after_hours_rows", 0) == 0 for row in rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
