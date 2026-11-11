"""Stage and import one daily partition aggregated from QMT local DAT files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
from qmt_dat import QMT_DTYPE, qmt_minute_volume_lots

FIELDS = ("open", "high", "low", "close", "volume", "amount")


def _hash(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _market(symbol: str) -> str | None:
    text = str(symbol).upper()
    if ".BJ" in text:
        return "BJ"
    if ".SH" in text:
        return "SH"
    if ".SZ" in text:
        return "SZ"
    code = "".join(ch for ch in text if ch.isdigit())[-6:].zfill(6)
    if code.startswith("92"):
        return "BJ"
    if code.startswith(("6", "68")):
        return "SH"
    if code.startswith(("0", "3")):
        return "SZ"
    return None


def _code(symbol: str) -> str:
    return "".join(ch for ch in str(symbol) if ch.isdigit())[-6:].zfill(6)


def _read_qmt(path: Path, day: date) -> dict | None:
    size = path.stat().st_size
    if size < 72 or (size - 8) % QMT_DTYPE.itemsize:
        return None
    count = min((size - 8) // QMT_DTYPE.itemsize, 1000)
    with path.open("rb") as handle:
        handle.seek(size - count * QMT_DTYPE.itemsize)
        rows = np.frombuffer(handle.read(count * QMT_DTYPE.itemsize), dtype=QMT_DTYPE)
    tz = timezone(timedelta(hours=8))
    start = int(datetime.combine(day, datetime.min.time(), tzinfo=tz).timestamp())
    end = int(datetime.combine(day, datetime.max.time(), tzinfo=tz).timestamp())
    rows = rows[(rows["timestamp"] >= start) & (rows["timestamp"] <= end)]
    if rows.size == 0:
        return None
    rows = rows[np.argsort(rows["timestamp"])]
    return {
        "rows": int(rows.size),
        "open": float(rows["open"][0]) / 1000.0,
        "high": float(rows["high"].max()) / 1000.0,
        "low": float(rows["low"].min()) / 1000.0,
        "close": float(rows["close"][-1]) / 1000.0,
        # Aggregate complete minute quantities; the daily contract uses shares.
        "volume": float(qmt_minute_volume_lots(rows).sum()) * 100.0,
        "amount": float(rows["amount"].sum()),
        "first_timestamp": int(rows["timestamp"][0]),
        "last_timestamp": int(rows["timestamp"][-1]),
    }


def _price_match(qmt: dict, local: dict) -> bool:
    tolerance = max(0.011, abs(float(qmt["close"])) * 1e-6)
    return all(abs(float(qmt[field]) - float(local[field])) <= tolerance for field in ("open", "high", "low", "close"))


def _valid(qmt: dict) -> bool:
    return (
        qmt["rows"] == 240
        and qmt["open"] > 0
        and qmt["low"] > 0
        and qmt["high"] >= max(qmt["open"], qmt["close"])
        and qmt["low"] <= min(qmt["open"], qmt["close"])
        and qmt["volume"] >= 0
        and qmt["amount"] >= 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--qmt-root", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--confirm-replace", action="store_true")
    args = parser.parse_args()
    root = args.data_dir.resolve()
    daily_path = root / "kline_daily" / f"date={args.date}" / "part.parquet"
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)
    local = pl.read_parquet(daily_path)
    if "symbol" not in local.columns or "date" not in local.columns:
        raise RuntimeError("daily partition is missing symbol/date")
    if local.height != local.unique(["symbol", "date"]).height:
        raise RuntimeError("daily partition has duplicate symbol/date keys")

    report = root / "import_reports" / f"qmt_daily_{args.date:%Y%m%d}_{datetime.now():%H%M%S}"
    staging = root / "import_staging" / f"qmt_daily_{args.date:%Y%m%d}"
    report.mkdir(parents=True, exist_ok=False)
    # A dry-run creates the date staging directory before confirmation. Reuse
    # that directory on the confirmation run, while rejecting incomplete or
    # unrelated contents instead of silently mixing artifacts.
    if staging.exists():
        expected = (staging / "accepted.parquet", staging / "details.json")
        if not all(path.is_file() for path in expected):
            raise RuntimeError(f"staging directory is incomplete: {staging}")
    else:
        staging.mkdir(parents=True, exist_ok=False)
    local_rows = {str(row["symbol"]): row for row in local.to_dicts()}
    details: list[dict] = []
    accepted: list[dict] = []
    rejected: list[dict] = []
    for symbol, row in local_rows.items():
        code = _code(symbol)
        market = _market(symbol)
        item = {"symbol": symbol, "code": code, "market": market, "date": args.date.isoformat()}
        if not market:
            item.update(status="quarantine", reason="unknown_exchange")
            rejected.append(item)
            details.append(item)
            continue
        qmt_path = args.qmt_root / market / "60" / f"{code}.DAT"
        qmt = _read_qmt(qmt_path, args.date) if qmt_path.exists() else None
        item["qmt_file"] = str(qmt_path)
        if not qmt:
            item.update(status="source_no_record", reason="missing_file_or_date")
            rejected.append(item)
            details.append(item)
            continue
        item.update(qmt=qmt, local={field: row[field] for field in FIELDS})
        if not _valid(qmt):
            item.update(status="quarantine", reason="qmt_integrity_failed")
        elif not _price_match(qmt, row):
            item.update(status="quarantine", reason="price_conflict")
        else:
            item.update(status="candidate", reason="qmt_ohlc_consistent")
            accepted.append({"symbol": symbol, "date": args.date, **{field: qmt[field] for field in FIELDS}})
        if item["status"] != "candidate":
            rejected.append(item)
        details.append(item)

    # The staging artifact contains only validated replacement rows.
    accepted_frame = pl.DataFrame(accepted, schema={
        "symbol": pl.String, "date": pl.Date, "open": pl.Float64, "high": pl.Float64,
        "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64, "amount": pl.Float64,
    })
    accepted_frame.write_parquet(staging / "accepted.parquet", compression="zstd")
    (staging / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary = {
        "status": "staged",
        "date": args.date.isoformat(),
        "local_rows": local.height,
        "qmt_candidates": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(pl.DataFrame(rejected).group_by("reason").len().iter_rows()) if rejected else {},
        "report_dir": str(report),
        "staging_dir": str(staging),
    }
    (report / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.confirm_replace:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # Preserve the local symbol spelling, replace only validated rows, and verify atomically.
    updates = {row["symbol"]: row for row in accepted_frame.to_dicts()}
    output_rows = []
    for row in local.to_dicts():
        output_rows.append({**row, **updates.get(str(row["symbol"]), {})})
    # Keep nullable fields such as quote_ts on the source schema. Inference
    # from a list of dicts can see only nulls in the first rows and then fail
    # when a later row contains an integer timestamp.
    updated = pl.DataFrame(output_rows, schema=local.schema).select(local.columns).sort(["symbol", "date"])
    backup = root / "import_backups" / report.name / "kline_daily" / f"date={args.date}" / "part.parquet"
    backup.parent.mkdir(parents=True, exist_ok=True)
    before = _hash(daily_path)
    shutil.copy2(daily_path, backup)
    if _hash(backup) != before:
        raise RuntimeError("daily backup hash mismatch")
    tmp = daily_path.with_name(daily_path.name + ".qmt-tmp")
    updated.write_parquet(tmp, compression="zstd")
    readback = pl.read_parquet(tmp)
    if readback.height != local.height or readback.height != readback.unique(["symbol", "date"]).height:
        raise RuntimeError("daily replacement verification failed")
    tmp.replace(daily_path)

    enriched_path = root / "kline_daily_enriched" / f"date={args.date}" / "part.parquet"
    enriched_changed = False
    if enriched_path.exists():
        enriched = pl.read_parquet(enriched_path)
        enriched_updates = {row["symbol"]: row for row in accepted_frame.to_dicts()}
        enriched_rows = []
        for row in enriched.to_dicts():
            update = enriched_updates.get(str(row["symbol"]))
            enriched_rows.append({**row, **({field: update[field] for field in FIELDS} if update else {})})
        enriched = pl.DataFrame(enriched_rows, schema=enriched.schema).select(enriched.columns)
        enriched_backup = root / "import_backups" / report.name / "kline_daily_enriched" / f"date={args.date}" / "part.parquet"
        enriched_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(enriched_path, enriched_backup)
        enriched_tmp = enriched_path.with_name(enriched_path.name + ".qmt-tmp")
        enriched.write_parquet(enriched_tmp, compression="zstd")
        enriched_tmp.replace(enriched_path)
        enriched_changed = True
    summary.update({
        "status": "completed",
        "replaced_rows": len(accepted),
        "backup": str(backup),
        "before_sha256": before,
        "after_sha256": _hash(daily_path),
        "enriched_updated": enriched_changed,
    })
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
