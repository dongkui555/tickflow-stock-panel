"""Classify safe Beijing regular-session minute records without fabricating bars.

The BSE daily total can include the 15:00-15:30 fixed-price session while a
minute source contains only the 240 continuous-trading bars. This command
records that narrower quality state and keeps the original daily total intact.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


REASON = "北交所连续竞价240根量价通过，日K成交额包含未覆盖的盘后固定价格交易"


def _bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def classify_row(row: dict) -> str:
    """Return a conservative status for one audit row."""
    try:
        bars = int(row.get("bars") or 0)
        after_hours = int(row.get("after_hours_rows") or 0)
        daily_amount = float(row.get("amount"))
        minute_amount = float(row.get("minute_amount"))
    except (TypeError, ValueError):
        return "complete_conflict"
    if (
        str(row.get("market") or "").upper() == "BJ"
        and bars >= 240
        and after_hours == 0
        and _bool(row.get("price_match"))
        and _bool(row.get("volume_match"))
        and not _bool(row.get("amount_match"))
        and daily_amount > minute_amount
    ):
        return "regular_session"
    return "complete_conflict"


def _backup(path: Path, backup_dir: Path) -> str | None:
    if not path.exists():
        return None
    target = backup_dir / path.name
    shutil.copy2(path, target)
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_dir.resolve()
    audit = args.audit.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root / "import_reports") or output == root / "import_reports":
        raise ValueError("output must be a new directory inside data/import_reports")
    output.mkdir(parents=True, exist_ok=False)

    with audit.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row for row in rows
        if row.get("status") == "complete_conflict" and classify_row(row) == "regular_session"
    ]

    records = []
    for row in candidates:
        records.append({
            "symbol": f"{str(row['code']).zfill(6)}.BJ",
            "date": row["date"],
            "status": "regular_session",
            "expected_minutes": 240,
            "actual_minutes": int(row["bars"]),
            "missing_minutes": 0,
            "reason": REASON,
            "source": "local",
            "daily_turnover_complete": False,
            "minute_scope": "regular_session",
        })

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_dir = root / "import_backups" / f"bj_regular_session_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups = []
    for path in (
        root / "user_data" / "minute_coverage_status.json",
        root / "user_data" / "minute_gap_queue.sqlite",
    ):
        saved = _backup(path, backup_dir)
        if saved:
            backups.append(saved)

    from app.services import minute_status
    from app.services.minute_gap_queue import _connect, _ensure_schema, _upsert

    minute_status.upsert(root, records)
    now = datetime.now().isoformat(timespec="seconds")
    queue_updated = 0
    with _connect(root) as conn:
        _ensure_schema(conn)
        for record in records:
            old = conn.execute(
                "SELECT attempts, created_at FROM minute_gap_queue WHERE symbol=? AND trade_date=?",
                (record["symbol"].split(".")[0], record["date"]),
            ).fetchone()
            _upsert(conn, {
                **record,
                "trade_date": record["date"],
                "source_chain": "qmt,promax,tdx,stockdb,offline",
                "attempts": int(old["attempts"] if old else 0),
                "created_at": old["created_at"] if old else now,
                "next_retry_at": None,
                "last_error": None,
            }, now)
            queue_updated += 1

    report = {
        "status": "applied",
        "audit": str(audit),
        "candidate_stock_days": len(candidates),
        "coverage_status_updated": len(records),
        "queue_updated": queue_updated,
        "status_value": "regular_session",
        "reason": REASON,
        "daily_total_preserved": True,
        "minute_data_written": False,
        "backups": backups,
        "remaining_conflicts": len(rows) - len(candidates),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
