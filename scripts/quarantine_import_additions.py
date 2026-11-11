"""Withdraw only one import's new rows, preserving all pre-existing data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import polars as pl


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["replaced_rows"] == report["removed_rows"] == 0
    details = [row for row in report["details"] if row["reason"] == "accepted"]
    plans = []
    for day in sorted({row["date"] for row in details}):
        path = args.data_dir / "kline_minute" / f"date={day}" / "part.parquet"
        before_path = args.before / f"date={day}" / "part.parquet"
        before = pl.read_parquet(before_path)
        original_hash = sha256(path)
        current = pl.read_parquet(path)
        day_rows = [row for row in details if row["date"] == day]
        symbols = [f'{row["symbol"]}.SZ' for row in day_rows]
        # This withdrawal is deliberately limited to the explicit SZ import.
        assert all(row["symbol"].startswith(("00", "30")) for row in day_rows)
        delta = current.filter(pl.col("symbol").is_in(symbols))
        assert delta.height == sum(row["added_rows"] for row in day_rows)
        assert delta.join(before, on=["symbol", "datetime"], how="semi").is_empty()
        remaining = current.join(delta.select("symbol", "datetime"), on=["symbol", "datetime"], how="anti")
        # Reject unexpected changes instead of restoring a stale whole partition.
        assert current.height - remaining.height == delta.height
        plans.append((day, path, original_hash, delta, remaining))
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"source_report": str(args.report), "applied": args.apply, "days": []}
    for day, path, original_hash, delta, remaining in plans:
        folder = args.output / f"date={day}"
        folder.mkdir()
        delta.write_parquet(folder / "quarantined.parquet")
        entry = {"date": day, "rows": delta.height, "before_sha256": original_hash,
                 "reason": "qmt_raw_intrabar_ohlc_conflict", "withdrawn": False}
        if args.apply:
            shutil.copy2(path, folder / "before_withdrawal.parquet")
            temp = path.with_name("part.quarantine-tmp.parquet")
            remaining.write_parquet(temp)
            assert pl.read_parquet(temp).equals(remaining)
            if sha256(path) != original_hash:
                raise RuntimeError(f"Concurrent partition change: {day}; no replacement performed")
            os.replace(temp, path)
            assert pl.read_parquet(path).equals(remaining)
            entry["withdrawn"] = True
            entry["after_sha256"] = sha256(path)
        result["days"].append(entry)
        (args.output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
