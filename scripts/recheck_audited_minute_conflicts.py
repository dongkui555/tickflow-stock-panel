"""Reclassify audit flags using the production trade-aware session contract."""
import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from app.services.minute_repair import conflicting_complete_codes, daily_reference, keyed, regular


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit-issues", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = pl.read_csv(args.audit_issues).filter(pl.col("issue").is_in(["price_mismatch", "turnover_mismatch"]))
    days = sorted(original["date"].unique().to_list())
    results, issues = [], []
    for index, text in enumerate(days, 1):
        day = date.fromisoformat(text)
        minute = args.data_dir / "kline_minute" / f"date={day}" / "part.parquet"
        daily = args.data_dir / "kline_daily" / f"date={day}" / "part.parquet"
        if not minute.exists() or not daily.exists():
            continue
        reference = daily_reference(pl.read_parquet(daily), data_dir=args.data_dir)
        frame = pl.read_parquet(minute)
        conflicts = conflicting_complete_codes(frame, reference, day)
        flagged = set(original.filter(pl.col("date") == text)["symbol"].str.extract(r"(\d{6})", 1).to_list())
        full = regular(keyed(frame), day).group_by("_code").agg(pl.col("datetime").n_unique().alias("n"))
        full_codes = set(full.filter(pl.col("n") == 240)["_code"].to_list())
        eligible = flagged & set(reference) & full_codes
        verified_flags = eligible & conflicts
        row = {"date": text, "original_flagged": len(flagged), "recheckable": len(eligible),
               "confirmed_contract_conflicts": len(verified_flags),
               "cleared_after_correct_contract": len(eligible - conflicts),
               "unresolved_reference_or_completeness": len(flagged - eligible),
               "all_complete_session_conflicts": len(conflicts)}
        results.append(row)
        issues.extend({"date": text, "symbol": code, "issue": "daily_price_or_turnover_conflict"}
                      for code in sorted(conflicts))
        if index % 20 == 0 or index == len(days):
            print(json.dumps({"progress": index, "total": len(days), **row}), flush=True)
            (args.output / "minute-conflict-recheck.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pl.DataFrame(issues).write_parquet(args.output / "confirmed-minute-conflicts.parquet")
    summary = {key: sum(row[key] for row in results) for key in (
        "original_flagged", "recheckable", "confirmed_contract_conflicts", "cleared_after_correct_contract",
        "unresolved_reference_or_completeness", "all_complete_session_conflicts")}
    summary.update(days=len(results), production_writes=0)
    (args.output / "minute-recheck-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
