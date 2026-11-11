"""Canonicalize explicitly verified minute-symbol aliases for one stock-day.

This is intentionally narrow: it only repairs the two stock-days supplied on
the command line, only merges byte/value-equivalent duplicate timestamps, and
always writes a content-addressed backup before changing a parquet partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from datetime import date
from pathlib import Path

import polars as pl

from app.parquet import atomic_write_parquet
from app.services.security_identity import validate_frame_identity


COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
TARGETS = {
    "600935": "600935.SH",
    "601033": "601033.SH",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonicalize_partition(root: Path, day: date) -> dict:
    path = root / "kline_minute" / f"date={day}" / "part.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    before_hash = sha256(path)
    old = pl.read_parquet(path)
    required = set(COLS)
    if not required.issubset(old.columns):
        raise ValueError(f"minute partition missing columns: {sorted(required - set(old.columns))}")

    code = old.get_column("symbol").cast(pl.String).str.slice(0, 6)
    affected = old.filter(code.is_in(list(TARGETS)))
    if affected.is_empty():
        return {
            "status": "not_found",
            "date": str(day),
            "file": str(path.relative_to(root)),
            "rows_before": old.height,
            "rows_after": old.height,
            "duplicates_removed": 0,
        }

    normalized = old.with_columns(
        pl.when(code == "600935")
        .then(pl.lit(TARGETS["600935"]))
        .when(code == "601033")
        .then(pl.lit(TARGETS["601033"]))
        .otherwise(pl.col("symbol"))
        .alias("symbol")
    )
    changed = normalized.filter(
        normalized.get_column("symbol").is_in(list(TARGETS.values()))
    )

    # A duplicate timestamp may be removed only if every OHLCV field agrees.
    duplicate_groups = changed.group_by(["symbol", "datetime"]).agg(
        pl.len().alias("rows"),
        *[pl.col(column).n_unique().alias(f"{column}_unique") for column in COLS[2:]],
    ).filter(
        (pl.col("rows") > 1)
        | pl.any_horizontal(
            [pl.col(f"{column}_unique") > 1 for column in COLS[2:]]
        )
    )
    conflicting = duplicate_groups.filter(
        pl.any_horizontal(
            [pl.col(f"{column}_unique") > 1 for column in COLS[2:]]
        )
    )
    if not conflicting.is_empty():
        raise ValueError(
            "alias timestamps contain different OHLCV values; refusing to merge: "
            + conflicting.head(10).write_json()
        )

    deduped = changed.unique(
        subset=["symbol", "datetime"], keep="first", maintain_order=True
    )
    untouched = normalized.filter(
        ~normalized.get_column("symbol").is_in(list(TARGETS.values()))
    )
    new = pl.concat([untouched, deduped], how="vertical_relaxed").sort(
        ["symbol", "datetime"]
    )

    validate_frame_identity(
        deduped,
        asset_type="stock",
        expected_symbols=list(TARGETS.values()),
    )
    duplicate_keys = deduped.group_by(["symbol", "datetime"]).len().filter(pl.col("len") > 1)
    if not duplicate_keys.is_empty():
        raise ValueError("canonicalized target still contains duplicate timestamps")
    if not deduped.select(
        pl.all_horizontal(
            [pl.col(column).is_finite() for column in COLS[2:]]
        ).all()
    ).item():
        raise ValueError("canonicalized target contains non-finite values")

    removed = old.height - new.height
    run_id = "minute_alias_repair_" + uuid.uuid4().hex[:12]
    backup = root / "import_backups" / run_id / "kline_minute" / f"date={day}" / "part.parquet"
    backup.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(path, backup)
    if sha256(backup) != before_hash or sha256(path) != before_hash:
        raise RuntimeError("source changed or backup hash mismatch; no replacement")

    atomic_write_parquet(new, path)
    readback = pl.read_parquet(path)
    if not readback.sort(["symbol", "datetime"]).equals(new):
        atomic_write_parquet(old, path)
        raise RuntimeError("readback verification failed; original partition restored")

    report_dir = root / "import_reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "repaired",
        "date": str(day),
        "file": str(path.relative_to(root)),
        "targets": TARGETS,
        "rows_before": old.height,
        "rows_after": new.height,
        "duplicates_removed": removed,
        "backup": str(backup.relative_to(root)),
        "sha256_before": before_hash,
        "target_rows": {
            symbol: int(deduped.filter(pl.col("symbol") == symbol).height)
            for symbol in TARGETS.values()
        },
    }
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, default=date(2026, 9, 9))
    args = parser.parse_args()
    print(json.dumps(canonicalize_partition(args.data_dir.resolve(), args.date), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
