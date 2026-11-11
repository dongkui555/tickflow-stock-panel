"""Revalidate saved raw responses after a contract fix; no network or production writes."""
import argparse
import gzip
import json
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from app.plugins.tushare.minute_contract import normalize_history
from app.services.minute_repair import checked_additions, daily_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dates", default="2026-09-07,2026-09-08,2026-09-09")
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output.resolve()
    if root == output or output.is_relative_to(root):
        raise ValueError("Output must be outside production data")
    (output / "minutes").mkdir(parents=True, exist_ok=True)
    dates = {date.fromisoformat(s) for s in args.dates.split(",")}
    date_labels = {str(day) for day in dates}
    refs = {day: daily_reference(pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet"), data_dir=root)
            for day in dates}
    results, accepted_keys = [], set()
    for folder in args.evidence:
        for path in sorted(folder.rglob("*.json.gz")):
            if not date_labels.intersection(path.parts):
                continue
            payload = json.loads(gzip.decompress(path.read_bytes()))
            if not {"date", "symbol", "raw_minute", "raw_daily"} <= set(payload):
                continue
            day, symbol = date.fromisoformat(payload["date"]), payload["symbol"]
            if day not in dates or len(symbol) != 6 or not symbol.isdigit() or (day, symbol) in accepted_keys:
                continue
            result = {"date": str(day), "symbol": symbol, "old_reason": payload.get("result"), "source_file": str(path)}
            outcomes = []
            try:
                frame = normalize_history(pd.DataFrame(payload["raw_minute"]), pd.DataFrame(payload["raw_daily"]),
                                          symbol, output / "evidence-v2", outcomes=outcomes)
                if not frame.is_empty():
                    accepted, reason = checked_additions(frame, frame.clear(), refs[day].get(symbol), day)
                    result["new_reason"] = reason
                    if reason == "accepted":
                        accepted.write_parquet(output / "minutes" / f"{symbol}_{day}_revalidated.parquet")
                        result["rows"] = accepted.height
                        accepted_keys.add((day, symbol))
                else:
                    result["new_reason"] = outcomes[-1]["reason"] if outcomes else "empty_source"
            except (ValueError, KeyError, TypeError) as exc:
                result["new_reason"] = type(exc).__name__
            results.append(result)
            if len(results) % 100 == 0:
                print(json.dumps({"checked": len(results), "validated_stock_days": len(accepted_keys)}), flush=True)
    summary = {"checked": len(results), "validated_stock_days": len(accepted_keys),
               "rows_staged": sum(r.get("rows", 0) for r in results), "production_writes": 0, "results": results}
    (output / "cached-minute-revalidation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
