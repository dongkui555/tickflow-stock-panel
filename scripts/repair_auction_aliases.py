"""Normalize auction symbols and remove verified alias duplicates safely."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import polars as pl


CORE_COLUMNS = ("open", "high", "low", "close", "volume", "amount", "vwap")


def compact_symbol(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    for prefix in ("SH", "SZ", "BJ"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else text


def repair_partition(path: Path, backup_root: Path) -> dict[str, object]:
    original = pl.read_parquet(path)
    if original.is_empty() or "symbol" not in original.columns or "date" not in original.columns:
        return {"file": str(path), "changed": False, "reason": "missing symbol/date or empty"}

    normalized = original.with_columns(
        pl.col("symbol").map_elements(compact_symbol, return_dtype=pl.String).alias("symbol")
    ).filter(pl.col("symbol") != "")
    if normalized.is_empty():
        raise ValueError(f"all symbols became empty: {path}")

    duplicate_groups = (
        normalized.group_by(["symbol", "date"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    available_core = [column for column in CORE_COLUMNS if column in normalized.columns]
    conflicts = []
    if available_core and not duplicate_groups.is_empty():
        grouped = normalized.group_by(["symbol", "date"]).agg(
            pl.len().alias("n"),
            *[pl.col(column).n_unique().alias(column) for column in available_core],
        )
        conflicts = grouped.filter(
            pl.any_horizontal([pl.col(column) > 1 for column in available_core])
        ).to_dicts()
    if conflicts:
        raise ValueError(f"conflicting auction observations in {path}: {conflicts[:3]}")

    # Prefer the row carrying the most provider metadata, then keep one row per
    # canonical security/date. All duplicate core values were verified above.
    metadata_score = pl.sum_horizontal(
        [pl.col(column).is_not_null().cast(pl.UInt8) for column in normalized.columns]
    ).alias("_metadata_score")
    repaired = (
        normalized.with_columns(metadata_score)
        .sort(["_metadata_score", "symbol", "date"], descending=[True, False, False])
        .unique(subset=["symbol", "date"], keep="first", maintain_order=True)
        .drop("_metadata_score")
        .sort(["symbol", "date"])
    )
    changed = not original.equals(repaired)
    result: dict[str, object] = {
        "file": str(path),
        "changed": changed,
        "rows_before": original.height,
        "rows_after": repaired.height,
        "duplicate_groups": duplicate_groups.height,
        "removed_rows": original.height - repaired.height,
    }
    if not changed:
        return result

    relative = path.relative_to(path.parents[2])
    backup = backup_root / relative
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        repaired.write_parquet(tmp)
        os.replace(tmp, path)
        readback = pl.read_parquet(path)
        if not readback.equals(repaired):
            raise RuntimeError("parquet readback mismatch")
        result["backup"] = str(backup)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        if backup.exists():
            shutil.copy2(backup, path)
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--date", help="Only repair date=YYYY-MM-DD; defaults to all partitions")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    auction_dir = data_dir / "auction_open"
    if args.date:
        paths = [auction_dir / f"date={args.date}" / "part.parquet"]
    else:
        paths = sorted(auction_dir.glob("date=*/part.parquet"))
    paths = [path for path in paths if path.exists()]
    run_id = "auction_alias_normalize_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = data_dir / "import_backups" / run_id
    report_dir = data_dir / "import_reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    results = [repair_partition(path, backup_root) for path in paths]
    summary = {
        "run_id": run_id,
        "files_scanned": len(results),
        "files_changed": sum(bool(item.get("changed")) for item in results),
        "rows_removed": sum(int(item.get("removed_rows", 0)) for item in results),
        "results": results,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("run_id", "files_scanned", "files_changed", "rows_removed")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
