"""Replace only the final audited BSE conflicts after a strict DataHub check."""
from __future__ import annotations

import csv
import json
import shutil
import time
import uuid
from datetime import date, datetime, time as dtime
from pathlib import Path

import polars as pl

from app import config
from app.plugins.tushare.provider import TushareProvider, build_datahub_client
from app.services import kline_sync
from app.services.minute_repair import COLS, checked_additions, daily_reference, keyed


def main() -> None:
    root = Path("/app/data")
    audit = root / "import_reports" / "audit_2025_bj_after_datahub_20260922"
    source_csv = audit / "missing_or_partial.csv"
    report_dir = root / "import_reports" / f"datahub_final_bj11_{uuid.uuid4().hex[:12]}"
    backup_root = root / "import_backups" / report_dir.name
    report_dir.mkdir(parents=True, exist_ok=False)
    with source_csv.open(encoding="utf-8-sig", newline="") as fh:
        targets = list(csv.DictReader(fh))
    provider = TushareProvider()
    provider._api = build_datahub_client()
    results = []
    backups: set[date] = set()
    try:
        for index, item in enumerate(targets, 1):
            day = date.fromisoformat(item["date"])
            code = str(item["code"]).zfill(6)
            symbol = item.get("selected_symbol") or f"{code}.BJ"
            daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
            minute_path = root / "kline_minute" / f"date={day}" / "part.parquet"
            refs = daily_reference(pl.read_parquet(daily_path), data_dir=root)
            ref = refs.get(code)
            start = datetime.combine(day, dtime(9, 25))
            end = datetime.combine(day, dtime(15, 35))
            frame, issues = provider.get_minute_diagnostics([symbol], start, end, freq="1m")
            verified, validation = checked_additions(frame, frame.clear(), ref, day, source="tushare")
            result = {
                "date": str(day),
                "symbol": symbol,
                "rows_from_source": frame.height,
                "validated_rows": verified.height,
                "validation": validation,
                "issues": issues,
                "replaced": False,
            }
            if validation == "accepted" and verified.height > 0 and minute_path.exists():
                old = pl.read_parquet(minute_path, columns=COLS)
                old_by = keyed(old)
                current = old_by.filter(pl.col("_code") == code).drop("_code")
                if current.is_empty() or current["symbol"].n_unique() != 1:
                    result["validation"] = "existing_identity_conflict"
                else:
                    verified = verified.with_columns(pl.lit(current["symbol"][0]).alias("symbol"))
                    untouched = old_by.filter(pl.col("_code") != code).drop("_code")
                    expected = pl.concat([untouched, verified], how="diagonal_relaxed").sort(["symbol", "datetime"])
                    backup = backup_root / f"date={day}" / "part.parquet"
                    if day not in backups:
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(minute_path, backup)
                        backups.add(day)
                    kline_sync._atomic_write_parquet(expected, minute_path)
                    after = pl.read_parquet(minute_path, columns=COLS).sort(["symbol", "datetime"])
                    if not after.equals(expected):
                        kline_sync._atomic_write_parquet(old, minute_path)
                        result["validation"] = "readback_failed_restored"
                    else:
                        result["replaced"] = True
                        result["old_rows"] = current.height
                        result["new_rows"] = verified.height
            results.append(result)
            print(json.dumps({"checked": index, "total": len(targets), **result}, ensure_ascii=False, default=str), flush=True)
    finally:
        provider.close()
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_count": len(targets),
        "replaced_count": sum(bool(row["replaced"]) for row in results),
        "backup": str(backup_root),
        "results": results,
    }
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
