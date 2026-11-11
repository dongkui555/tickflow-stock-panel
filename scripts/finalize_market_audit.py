"""Produce actionable gap lists and verify additive daily repairs against backups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from import_offline_minute_zip_batch import _symbol_expr


def finalize(root: Path, report: Path, verify: bool):
    minute = report / "minute_after_daily_repair"
    audit = json.loads((minute / "summary.json").read_text(encoding="utf-8"))
    issues = pl.read_csv(minute / "stock_day_issues.csv", schema_overrides={"symbol": pl.String})
    missing = issues.filter(pl.col("issue") == "missing")
    partial = issues.filter(pl.col("issue") == "partial_session")
    active_partial = []
    for group in partial.partition_by("date"):
        day = group["date"][0]
        daily = pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet")
        active = daily.filter(pl.all_horizontal([
            pl.col(c).is_finite() & (pl.col(c) > 0)
            for c in ("open", "high", "low", "close", "volume", "amount")
        ])).with_columns(_symbol_expr("symbol")).select("symbol").unique()
        active_partial.append(group.with_columns(_symbol_expr("symbol")).join(active, on="symbol", how="semi"))
    gaps = pl.concat([missing, *active_partial]).unique(subset=["date", "symbol"]).sort("date", "symbol")
    gaps = gaps.with_columns((240 - pl.col("session_minutes").fill_null(0)).alias("missing_minutes"))
    gaps.write_csv(report / "minute_required_stock_days.csv", include_bom=True)
    monthly = gaps.with_columns(pl.col("date").str.slice(0, 7).alias("month"))
    monthly = monthly.group_by("month").agg(
        pl.len().alias("stock_days"), pl.col("symbol").n_unique().alias("stocks"),
        pl.col("date").n_unique().alias("dates"), pl.col("missing_minutes").sum(),
    ).sort("month")
    monthly.write_csv(report / "minute_missing_by_month.csv", include_bom=True)
    conflicts = issues.filter(
        pl.col("issue").is_in(["price_mismatch", "turnover_mismatch"])
        & (pl.col("session_minutes") == 240)
    ).group_by("date", "symbol").agg(pl.col("issue").str.join(";")).sort("date", "symbol")
    conflicts.write_csv(report / "cross_source_conflicts.csv", include_bom=True)
    runs, failures = [], []
    for directory in sorted(root.glob("import_reports/market_repair_*")):
        path = directory / "summary.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("start") != audit["start"] or run.get("end") != audit["end"]:
            continue
        item = {k: run.get(k) for k in (
            "run_id", "provider", "requests", "api_requests", "rows_added",
            "partitions_added", "partitions_repaired", "complete",
        )}
        item["backups_checked"] = 0
        if verify:
            for write in run.get("writes", []):
                if not write.get("backup"):
                    continue
                old = pl.read_parquet(root / write["backup"])
                target = root / write["path"]
                new = pl.read_parquet(target)
                # Repairs only append absent codes; every old column must be unchanged.
                old = old.sort("symbol")
                retained = new.join(old.select("symbol").unique(), on="symbol", how="semi")
                retained = retained.select(old.columns).sort("symbol")
                if not old.equals(retained):
                    failures.append({"run_id": run["run_id"], "path": write["path"]})
                item["backups_checked"] += 1
        runs.append(item)
    result = {
        "start": audit["start"], "end": audit["end"],
        "minute_rows_checked": audit["total_rows"],
        "minute_stock_days_missing_or_partial": gaps.height,
        "missing_minute_rows": int(gaps["missing_minutes"].sum()),
        "full_day_cross_source_conflicts": conflicts.height,
        "conflict_note": "Not proof of which source is wrong; no forced replacement",
        "daily_reference_note": "Includes explicitly marked local minute aggregates; these are not independent verification",
        "minute_totals": audit["totals_completed_days"],
        "repairs": runs,
        "backup_verification_failures": failures,
    }
    market_path = report / "market_after_repair" / "summary.json"
    if market_path.exists():
        market = json.loads(market_path.read_text(encoding="utf-8"))
        remaining = [{"dataset": item["dataset"], "date": day,
                      "reason": "No records for a locally and remotely matched trading date"}
                     for item in market["datasets"] for day in item["missing_dates"]]
        pl.DataFrame(remaining, schema={"dataset": pl.String, "date": pl.String, "reason": pl.String}).write_csv(
            report / "missing_dataset_dates.csv", include_bom=True,
        )
        result["market_file_errors"] = sum(item["file_errors"] for item in market["datasets"])
        result["market_invalid_ohlc"] = sum(item["invalid_ohlc"] for item in market["datasets"])
        result["datasets_audited"] = len(market["datasets"])
    (report / "repair_acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit("Backup verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--verify-backups", action="store_true")
    args = parser.parse_args()
    finalize(args.data_dir, args.report_dir, args.verify_backups)
