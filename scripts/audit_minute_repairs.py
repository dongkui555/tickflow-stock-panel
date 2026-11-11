"""Read-only verification of repair journals, current partitions and backups."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import polars as pl

from app.services.minute_repair import COLS, checked_additions, daily_reference, keyed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorized-start", type=date.fromisoformat, default=date(2026, 8, 24))
    parser.add_argument("--authorized-end", type=date.fromisoformat, default=date(2026, 9, 3))
    args = parser.parse_args()
    root = args.data_dir
    reports = []
    modified = defaultdict(set)
    replaced = defaultdict(set)
    for name in args.report:
        report = json.loads((root / "import_reports" / name / "report.json").read_text(encoding="utf-8"))
        # Reuse the source-specific unit contract that was used during import.
        # QMT minute volume is allowed to use its lots/share scale and its
        # auction rounding; treating it as an anonymous remote source creates
        # a false ``minute_turnover_unit_conflict`` during read-only audit.
        reports.append((name, report, str(report.get("provider") or "").strip() or None))
        for row in report["details"]:
            if row["reason"] == "repaired_existing_conflict":
                replaced[row["date"]].add(row["symbol"])
            if row["reason"] in {"accepted", "repaired_existing_conflict"}:
                modified[row["date"]].add(row["symbol"])

    validation = []
    preservation = []
    failures = []
    for name, report, source in reports:
        repair_dates = {r["date"] for r in report["details"] if r["reason"] == "repaired_existing_conflict"}
        for day in repair_dates:
            if not (root / "import_backups" / name / f"date={day}" / "part.parquet").is_file():
                failures.append({"run": name, "date": day, "error": "required_backup_missing"})
    checked_days = 0
    for day, codes in sorted(modified.items()):
        current = pl.read_parquet(root / "kline_minute" / f"date={day}" / "part.parquet")
        refs = daily_reference(pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet"), data_dir=root)
        selected = keyed(current).filter(pl.col("_code").is_in(sorted(codes)))
        blocks = {block["_code"][0]: block.drop("_code") for block in selected.partition_by("_code")}
        for code in sorted(codes):
            frame = blocks.get(code, current.clear()).select(COLS)
            _, reason = checked_additions(
                frame,
                frame.clear(),
                refs.get(code),
                date.fromisoformat(day),
                source=source,
            )
            validation.append({"date": day, "symbol": code, "rows": frame.height, "validation": reason})
            if reason != "accepted":
                failures.append({"date": day, "symbol": code, "error": reason})
        for name, _, _ in reports:
            path = root / "import_backups" / name / f"date={day}" / "part.parquet"
            if not path.exists():
                continue
            backup = pl.read_parquet(path)
            untouched = keyed(backup).filter(~pl.col("_code").is_in(sorted(replaced[day]))).drop("_code")
            actual = current.join(untouched.select("symbol", "datetime"), on=["symbol", "datetime"], how="semi")
            unchanged = actual.sort("symbol", "datetime").equals(untouched.sort("symbol", "datetime"))
            preservation.append({"run": name, "date": day, "backup_rows": backup.height,
                                 "unchanged_rows_checked": untouched.height, "unchanged": unchanged})
            if not unchanged:
                failures.append({"date": day, "run": name, "error": "unrelated_rows_changed"})
        checked_days += 1
        print(json.dumps({"date": day, "checked_days": checked_days, "stock_days": len(codes), "failures": len(failures)}), flush=True)
    unauthorized = [day for day, codes in replaced.items() if codes and not args.authorized_start <= date.fromisoformat(day) <= args.authorized_end]
    if unauthorized:
        failures.append({"error": "replacements_outside_authorized_window", "dates": unauthorized})
    result = {
        "basis": "Current modified stock-days validated against local daily data; exact backup comparison excludes only explicitly replaced stocks.",
        "authorized_replacement_window": [str(args.authorized_start), str(args.authorized_end)],
        "run_names": args.report,
        "runs": [{"run": name, **{k: v for k, v in report.items() if k != "details"}} for name, report, _ in reports],
        "added_rows": sum(r["added_rows"] for _, r, _ in reports),
        "replaced_rows": sum(r["replaced_rows"] for _, r, _ in reports),
        "removed_rows": sum(r["removed_rows"] for _, r, _ in reports),
        "repaired_stock_days": sum(r["repaired_stock_days"] for _, r, _ in reports),
        "unique_modified_stock_days": sum(len(codes) for codes in modified.values()),
        "unique_replaced_stock_days": sum(len(codes) for codes in replaced.values()),
        "backup_checks": len(preservation),
        "all_passed": not failures,
        "failures": failures,
        "validation": validation,
        "preservation": preservation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"runs", "validation", "preservation"}}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
