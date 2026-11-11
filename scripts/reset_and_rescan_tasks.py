"""Archive maintenance tasks and rebuild only real minute gaps.

This is intentionally a data-preserving maintenance command.  It never deletes
minute data, audits, backups, authorization records, or the durable gap queue.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import polars as pl

from app.services import kline_sync, maintenance_queue, minute_gap_queue
from app.tickflow.repository import DataStore, KlineRepository


def _dates(data_dir: Path, start: date, end: date) -> list[str]:
    root = data_dir / "kline_daily"
    values = []
    for path in root.glob("date=*"):
        if not path.is_dir():
            continue
        value = path.name.removeprefix("date=")[:10]
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            continue
        if start <= parsed <= end:
            values.append(value)
    return sorted(set(values))


def _local_universe(data_dir: Path, repo: KlineRepository) -> list[str]:
    """Resolve the persisted stock universe without making a network call."""
    path = data_dir / "instruments" / "instruments.parquet"
    if not path.exists():
        return []
    frame = pl.read_parquet(path, columns=["symbol"])
    symbols = sorted({str(value).upper() for value in frame["symbol"].drop_nulls().to_list()})
    try:
        index_symbols = repo.get_index_symbol_set()
    except Exception:
        index_symbols = set()
    return [symbol for symbol in symbols if symbol not in index_symbols]


def _archive_and_clear(data_dir: Path) -> str:
    source = data_dir / "user_data" / "maintenance_queue.json"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive_dir = data_dir / "import_reports" / "maintenance_queue_reset" / stamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, archive_dir / "maintenance_queue-before-reset.json")
    (archive_dir / "reset_manifest.json").write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "action": "clear-worker-tasks-and-rescan-minute-gaps",
            "data_preserved": True,
            "gap_queue_preserved": True,
            "authorization_audit_preserved": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    temp = source.with_suffix(".reset.tmp")
    temp.write_text("[]", encoding="utf-8")
    temp.replace(source)
    return str(archive_dir)


def _pending_by_date(data_dir: Path, start: date, end: date) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with minute_gap_queue._connect(data_dir) as conn:  # noqa: SLF001
        rows = conn.execute(
            "SELECT trade_date, symbol FROM minute_gap_queue "
            "WHERE status IN ('pending','retry','qmt_turnover_conflict') "
            "ORDER BY trade_date, symbol"
        ).fetchall()
    for row in rows:
        day = str(row["trade_date"])
        try:
            parsed = date.fromisoformat(day[:10])
        except ValueError:
            continue
        if start <= parsed <= end:
            grouped[day].append(str(row["symbol"]))
    return dict(grouped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/app/data")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    parser.add_argument("--queue-only", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    archive_dir = _archive_and_clear(data_dir)
    store = DataStore()
    repo = KlineRepository(store)
    symbols = _local_universe(data_dir, repo)
    scope_label = "本地 instruments.parquet 股票池"
    dates = _dates(data_dir, start, end)
    scanned = []
    if not args.queue_only:
        for index, day in enumerate(dates, start=1):
            scanned.append(minute_gap_queue.ensure_date(data_dir, symbols, day))
            if index == 1 or index % 10 == 0 or index == len(dates):
                print(f"scanned {index}/{len(dates)} dates; latest={day}", flush=True)

    grouped = _pending_by_date(data_dir, start, end)
    task_ids = []
    for day, items in grouped.items():
        task_id = maintenance_queue.enqueue(
            data_dir, "minute", items,
            as_of=f"{day}T15:00", days=0, target_date=day,
            scope=f"all:history-gap-rescan:{day}",
            mode="history_gap_rescan", rescan=True,
        )
        if task_id:
            task_ids.append(task_id)

    result = {
        "archive_dir": archive_dir,
        "scope": scope_label,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": len(symbols),
        "dates_scanned": len(dates),
        "pending_dates": len(grouped),
        "pending_stock_days": sum(len(items) for items in grouped.values()),
        "tasks_rebuilt": len(task_ids),
        "task_ids": task_ids,
        "sample_scan": scanned[-3:],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
