"""Drill down flagged dates; leave all production inputs unchanged."""
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import polars as pl

from audit_history_semantics import audit_symbol_expr
from audit_minute_integrity import summarize_minutes, compare_daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    root, reports = args.data_dir.resolve(), args.report_dir.resolve()
    if reports == root or root in reports.parents:
        raise ValueError("Output must be outside source data")
    minute = json.loads((reports / "minute/summary.json").read_text())
    legacy = [r["date"] for r in minute["days"] if r["status"] == "readable" and r["date"] < "2024-08-01"]
    pl.DataFrame({"date": legacy}).write_csv(reports / "legacy-dates.csv")
    subprocess.run([sys.executable, str(Path(__file__).with_name("audit_minute_integrity.py")),
                    "--data-dir", str(root), "--output-dir", str(reports / "minute-legacy-review"),
                    "--start-date", "1990-01-01", "--end-date", "2026-09-09", "--dates-file", str(reports / "legacy-dates.csv")], check=True)
    corrected = json.loads((reports / "minute-legacy-review/summary.json").read_text())
    revised = {r["date"]: r for r in corrected["days"]}
    days = [revised.get(r["date"], r) for r in minute["days"]]
    totals = {key: sum(r.get(key, 0) for r in days) for key in minute["totals_completed_days"]}
    flagged = sorted(days, key=lambda r: r.get("full_session_price_mismatches", 0), reverse=True)[:4]
    flagged += [r for r in days if r["date"] >= "2026-09-01" and r.get("full_session_price_mismatches", 0)]
    detail, day_summaries, cases = [], [], []
    for record in flagged:
        day = record["date"]
        frame = pl.read_parquet(root / "kline_minute" / ("date=" + day) / "part.parquet")
        daily = pl.read_parquet(root / "kline_daily" / ("date=" + day) / "part.parquet")
        compared, _ = compare_daily(summarize_minutes(frame, date.fromisoformat(day)), daily, data_dir=root)
        compared = compared.with_columns(*[
            ((pl.col(c) - pl.col(c + "_minute")).abs() > 0.011).alias(c + "_diff") for c in ["open", "high", "low", "close"]
        ])
        bad = compared.filter(~pl.col("incomplete") & (pl.col("price_mismatch") | pl.col("turnover_mismatch")))
        day_summaries.append({"date": day, "flags": bad.height, **{c: bad[c + "_diff"].sum() for c in ["open", "high", "low", "close"]}})
        rows = bad.to_dicts()
        detail.extend({"date": day, **r} for r in rows)
        chosen = sorted(rows, key=lambda r: abs((r.get("close_minute") or 0) - r["close"]), reverse=True)
        for row in chosen[:2]:
            cases.append({"symbol": row["symbol"], "date": day, "reason": "full_minute_vs_daily_discrepancy"})
        print(json.dumps(day_summaries[-1]), flush=True)
    semantics = json.loads((reports / "semantics-v2/summary.json").read_text())
    groups = defaultdict(list)
    for row in semantics["issues"]:
        if row["issue"] == "duplicate_key":
            groups[(row["date"], row["symbol"])].append(row)
    duplicates = []
    for (day, symbol), rows in groups.items():
        same_price = all(max(r[c] for r in rows) - min(r[c] for r in rows) <= 0.011 for c in ["open", "high", "low", "close"])
        duplicates.append({"date": day, "symbol": symbol, "rows": len(rows), "prices_equal": same_price})
    changes = json.loads((reports / "run.json").read_text())["source_changes"]
    rechecked = []
    for item in changes:
        path = root / item["file"]
        try:
            before = path.stat()
            df = pl.read_parquet(path)
            after = path.stat()
            rechecked.append({"file": item["file"], "rows": df.height,
                              "stable_read": (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)})
        except Exception as exc:
            rechecked.append({"file": item["file"], "error": type(exc).__name__})
    cal = next((r.get("rows", []) for r in json.loads((reports / "online/evidence.json").read_text()) if r["source"] == "promax_trade_cal"), [])
    calendar = {str(r["cal_date"])[:4] + "-" + str(r["cal_date"])[4:6] + "-" + str(r["cal_date"])[6:8] for r in cal if str(r.get("is_open")) == "1"}
    coverage = {}
    for dataset in ["kline_daily", "kline_minute", "kline_index_daily", "kline_etf_daily", "moneyflow", "auction_open"]:
        present = {p.name[5:] for p in (root / dataset).glob("date=*") if (p / "part.parquet").exists()}
        coverage[dataset] = {"gateway_calendar_days": len(calendar), "missing_partitions": sorted(calendar - present)}
    result = {"corrected_minute_totals": totals, "corrected_days": days, "drilldown": day_summaries,
              "duplicate_groups": len(duplicates), "duplicate_price_conflicts": sum(not r["prices_equal"] for r in duplicates),
              "duplicate_dates": dict(Counter(r["date"] for r in duplicates)), "changed_files_rechecked": rechecked,
              "external_calendar_coverage": coverage,
              "notes": "Legacy review corrects audit join keys only, not stored records. Missing partitions are not stock-level coverage proof."}
    (reports / "review.json").write_text(json.dumps(result, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "minute-comparison-evidence.json").write_text(json.dumps(detail, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "online-minute-cases.json").write_text(json.dumps(cases, indent=2))
    pl.DataFrame(duplicates).write_csv(reports / "duplicate-stock-days.csv")


if __name__ == "__main__":
    main()
