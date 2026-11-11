"""Summarize an exact repair manifest and its immutable gateway evidence."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from datetime import date
from pathlib import Path

import polars as pl

from app.services.minute_repair import checked_additions, daily_reference, keyed, regular


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pl.read_csv(args.manifest)
    repairs = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
    attempted = {(r["symbol"].split(".")[0], r["date"]) for run in repairs for r in run["details"]}
    sync_results = {(r["symbol"].split(".")[0], r["date"]): r["reason"] for run in repairs for r in run["details"]}
    checked, remaining = [], []
    for group in manifest.partition_by("date"):
        day = date.fromisoformat(group["date"][0])
        path = args.data_dir / "kline_minute" / f"date={day}" / "part.parquet"
        daily_path = args.data_dir / "kline_daily" / f"date={day}" / "part.parquet"
        refs = daily_reference(pl.read_parquet(daily_path), data_dir=args.data_dir)
        frame = keyed(pl.read_parquet(path))
        codes = [str(s).split(".")[0] for s in group["symbol"]]
        blocks = {b["_code"][0]: b.drop("_code") for b in frame.filter(pl.col("_code").is_in(codes)).partition_by("_code")}
        for row in group.iter_rows(named=True):
            code = row["symbol"].split(".")[0]
            block = blocks.get(code, frame.clear().drop("_code"))
            count = regular(block, day)["datetime"].n_unique()
            _, validation = checked_additions(block, block.clear(), refs.get(code), day)
            item = {"date": str(day), "symbol": row["symbol"], "previous_missing_minutes": row["missing_minutes"],
                    "missing_minutes": max(0, 240 - count), "regular_minutes": count,
                    "validation": validation, "attempted": (code, str(day)) in attempted,
                    "sync_result": sync_results.get((code, str(day)), "not_requested")}
            checked.append(item)
            if count < 240:
                remaining.append(item)
    pl.DataFrame(checked).sort("date", "symbol").write_csv(args.output / "manifest_verification.csv", include_bom=True)
    pl.DataFrame(remaining, schema=pl.DataFrame(checked).schema).sort("date", "symbol").write_csv(
        args.output / "remaining_minute_stock_days.csv", include_bom=True)
    diagnostic = []
    for path in (args.data_dir / "import_reports" / "gateway_minute_contract").glob("*/*/*.json.gz"):
        event = json.loads(gzip.decompress(path.read_bytes()))
        raw, daily = event["raw_minute"], event["raw_daily"]
        raw = sorted(raw, key=lambda row: str(row.get("trade_time", "")))
        # A multi-day failure journal may contain several daily references.
        # Compare only this stock-day, never the first row from another date.
        daily = [row for row in daily
                 if str(row.get("ts_code", "")).split(".")[0] == event["symbol"]
                 and str(row.get("trade_date", "")).replace("-", "")[:8] == event["date"].replace("-", "")]
        record = {"date": event["date"], "symbol": event["symbol"], "result": event["result"],
                  "closing_mapped": event["closing_1501_mapped_to_1500"], "evidence": str(path.relative_to(args.data_dir))}
        if raw and daily:
            try:
                ref = daily[0]
                record.update(
                    minute_high=max(float(r["high"]) for r in raw), daily_high=float(ref["high"]),
                    minute_low=min(float(r["low"]) for r in raw), daily_low=float(ref["low"]),
                    minute_close=float(raw[-1]["close"]), daily_close=float(ref["close"]),
                    volume_delta_lots=sum(float(r["vol"]) for r in raw) / 100.0 - float(ref["vol"]),
                    amount_delta_yuan=sum(float(r["amount"]) for r in raw) - float(ref["amount"]) * 1000.0,
                )
            except (ValueError, TypeError, KeyError):
                record["comparison_error"] = True
        diagnostic.append(record)
    if diagnostic:
        pl.DataFrame(diagnostic, infer_schema_length=None).sort("date", "symbol").write_csv(
            args.output / "gateway_daily_comparisons.csv", include_bom=True)
    result = {
        "basis": "Re-read every stock-day from the original missing manifest; not a new audit of all historical stock-days",
        "original_stock_days": manifest.height, "original_missing_minutes": int(manifest["missing_minutes"].sum()),
        "attempted_stock_days": len(attempted), "requests_checked": sum(r["requests_checked"] for r in repairs),
        "unattempted_stock_days": sum((r["symbol"].split(".")[0], r["date"]) not in attempted for r in manifest.iter_rows(named=True)),
        "remaining_stock_days": len(remaining), "remaining_missing_minutes": sum(r["missing_minutes"] for r in remaining),
        "fully_daily_verified_in_manifest": sum(r["validation"] == "accepted" for r in checked),
        "full_but_conflicting_in_manifest": sum(r["regular_minutes"] >= 240 and r["validation"] != "accepted" for r in checked),
        "added_rows": sum(r["added_rows"] for r in repairs), "replaced_rows": sum(r["replaced_rows"] for r in repairs),
        "removed_rows": sum(r["removed_rows"] for r in repairs),
        "repair_reasons": dict(sum((Counter({k: v for k, v in r["reasons"].items() if not k.startswith("deferred")}) for r in repairs), Counter())),
        "gateway_evidence_results": dict(Counter(r["result"] for r in diagnostic)),
        "repair_reports": [str(p) for p in args.report],
    }
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
