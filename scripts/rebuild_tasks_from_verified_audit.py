"""Rebuild minute repair tasks from a completed local coverage audit.

This command is deliberately conservative: only ``missing_0`` and
``invalid_partial`` rows are ordinary repair candidates.  Complete rows with
量价 conflicts stay out of the retry queue and remain available for the
cross-source verification workflow.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import polars as pl

from app.services import maintenance_queue
from reset_and_rescan_tasks import _archive_and_clear


def _symbol(row: dict) -> str:
    selected = row.get("selected_symbol")
    if selected:
        return str(selected).upper()
    market = str(row["market"]).upper()
    return f"{int(row['code']):06d}.{market}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/app/data")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    report_dir = Path(args.report_dir)
    source = report_dir / "missing_or_partial.csv"
    frame = pl.read_csv(source)
    frame = frame.with_columns(pl.col("date").str.slice(0, 10))
    frame = frame.filter(
        (pl.col("date") >= start.isoformat())
        & (pl.col("date") <= end.isoformat())
    )

    counts = {
        str(row["status"]): int(row["len"])
        for row in frame.group_by("status").len().iter_rows(named=True)
    }
    candidates = frame.filter(pl.col("status").is_in(["missing_0", "invalid_partial"]))
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in candidates.iter_rows(named=True):
        grouped[str(row["date"])].append(_symbol(row))

    archive_dir = _archive_and_clear(data_dir)
    task_ids: list[str] = []
    for day in sorted(grouped):
        items = sorted(set(grouped[day]))
        task_id = maintenance_queue.enqueue(
            data_dir,
            "minute",
            items,
            as_of=f"{day}T15:00",
            days=0,
            target_date=day,
            scope=f"verified-audit:history-gap:{day}",
            mode="history_gap_rescan",
            rescan=True,
            audit_report=str(report_dir),
        )
        if task_id:
            task_ids.append(task_id)

    result = {
        "archive_dir": archive_dir,
        "audit_report": str(report_dir),
        "range": [start.isoformat(), end.isoformat()],
        "audit_status_counts": counts,
        "repair_statuses": ["missing_0", "invalid_partial"],
        "repair_dates": len(grouped),
        "repair_stock_days": sum(len(set(items)) for items in grouped.values()),
        "tasks_rebuilt": len(task_ids),
        "task_ids": task_ids,
        "excluded_from_retry": {
            "complete_conflict": counts.get("complete_conflict", 0),
            "reason": "分钟已完整，量价冲突需第三方核验，不自动覆盖",
        },
    }
    output = report_dir / "maintenance_queue_rebuild.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
