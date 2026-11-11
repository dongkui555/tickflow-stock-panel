"""Safely import a StockDB SDK staging file through the app's validators.

The staging file is created by ``stage_stockdb_sdk_minutes.py``. This runner
uses the production minute repair transaction, so existing rows are checked,
partitions are backed up before a write, and only missing timestamps are added.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path
from uuid import uuid4

import polars as pl

from app.services import kline_sync
from app.services.minute_repair import COLS, checked_additions, daily_reference, keyed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--staged-file", type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    day = date.fromisoformat(args.date)
    staged = args.staged_file.resolve()
    if not staged.exists() or staged.suffix.lower() != ".parquet":
        raise FileNotFoundError(staged)
    root = args.data_dir.resolve()
    source = pl.read_parquet(staged).select(COLS).sort(["symbol", "datetime"])
    daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
    minute_path = root / "kline_minute" / f"date={day}" / "part.parquet"
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)
    daily_refs = daily_reference(pl.read_parquet(daily_path), data_dir=root)
    old = pl.read_parquet(minute_path).select(COLS) if minute_path.exists() else pl.DataFrame(schema={
        "symbol": pl.String, "datetime": pl.Datetime("us"), "open": pl.Float64,
        "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
        "volume": pl.Float64, "amount": pl.Float64,
    })
    source = keyed(source)
    old = keyed(old)
    codes = sorted(set(source["_code"].to_list()))
    accepted: list[pl.DataFrame] = []
    reasons: Counter[str] = Counter()
    details: list[dict] = []
    for index, code in enumerate(codes, 1):
        incoming = source.filter(pl.col("_code") == code).drop("_code")
        current = old.filter(pl.col("_code") == code).drop("_code")
        verified, reason = checked_additions(
            incoming,
            incoming.clear(),
            daily_refs.get(code),
            day,
            source="stockdb",
        )
        if reason == "accepted":
            additions, reason = checked_additions(
                verified,
                current,
                daily_refs.get(code),
                day,
                source="stockdb",
            )
        else:
            additions = verified
        reasons[reason] += 1
        details.append({"date": str(day), "code": code, "reason": reason,
                        "source_rows": incoming.height, "existing_rows": current.height,
                        "addable_rows": additions.height})
        if not additions.is_empty():
            accepted.append(additions)
        if index % 250 == 0 or index == len(codes):
            print(json.dumps({"checked": index, "total": len(codes), "accepted_codes": sum(1 for x in details if x["reason"] == "accepted"),
                              "addable_rows": sum(x["addable_rows"] for x in details)}, ensure_ascii=False), flush=True)

    backup_dir = None
    added_rows = sum(frame.height for frame in accepted)
    if accepted:
        delta = pl.concat(accepted, how="vertical_relaxed").select(COLS)
        expected = pl.concat([old.drop("_code"), delta], how="vertical_relaxed").unique(
            subset=["symbol", "datetime"], keep="last", maintain_order=True
        ).sort(["symbol", "datetime"])
        if minute_path.exists():
            backup_dir = root / "import_backups" / f"stockdb_sdk_{day:%Y%m%d}_{uuid4().hex[:12]}"
            backup_target = backup_dir / f"date={day}" / "part.parquet"
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(minute_path, backup_target)
        minute_path.parent.mkdir(parents=True, exist_ok=True)
        kline_sync._atomic_write_parquet(expected, minute_path)
        readback = pl.read_parquet(minute_path).select(COLS).sort(["symbol", "datetime"])
        if not readback.equals(expected):
            if backup_dir is not None:
                shutil.copy2(backup_dir / f"date={day}" / "part.parquet", minute_path)
            raise RuntimeError("minute partition readback verification failed; original restored")
    report = {"source": "stockdb_sdk", "date": str(day), "source_rows": source.height,
              "source_codes": len(codes), "added_rows": added_rows, "existing_rows": old.height,
              "partition_rows_after": old.height + added_rows, "backup_dir": str(backup_dir) if backup_dir else None,
              "reasons": dict(reasons), "details": details}
    report_dir = root / "import_reports" / f"stockdb_sdk_{day:%Y%m%d}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
