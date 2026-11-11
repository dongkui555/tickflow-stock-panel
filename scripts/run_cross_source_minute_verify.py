"""Run the same bounded third-party minute-gap verifier used by the app.

This command reads a CSV manifest or the persistent minute-gap SQLite queue,
fetches only the selected stock-days, writes an audit report, and updates queue
state. It never writes daily or minute Parquet partitions.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys
from types import SimpleNamespace

import polars as pl

# Keep the operator-facing command runnable from the repository root as well
# as from a backend shell. The production package lives under ``backend``.
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

def _symbol_from_row(row: dict) -> str:
    raw = str(row.get("symbol") or "").strip().upper()
    if "." in raw:
        return raw
    code = str(row.get("code") or raw).strip().zfill(6)
    market = str(row.get("market") or row.get("exchange") or "").strip().upper()
    return f"{code}.{market}" if market else code


def _read_targets(queue_path: Path) -> list[dict]:
    if queue_path.suffix.lower() in {".sqlite", ".db"}:
        with sqlite3.connect(queue_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT symbol, trade_date AS date, status, reason, actual_minutes "
                "FROM minute_gap_queue WHERE status IN "
                "('pending','retry','qmt_turnover_conflict','existing_values_conflict',"
                "'quarantine','source_no_record') ORDER BY trade_date, symbol"
            ).fetchall()
        return [dict(row) for row in rows]

    frame = pl.read_csv(queue_path, try_parse_dates=True)
    required = {"date"}
    if not required.issubset(frame.columns) or not ({"symbol", "code"} & set(frame.columns)):
        raise ValueError("缺口清单必须包含 date，以及 symbol 或 code 列")
    rows = []
    for row in frame.to_dicts():
        status = str(row.get("status") or "pending")
        if status in {"complete", "no_trade", "pre_listing", "delisted", "code_changed"}:
            continue
        rows.append({
            "symbol": _symbol_from_row(row),
            "date": str(row.get("date"))[:10],
            "status": status,
            "reason": str(row.get("reason") or "source_incomplete"),
            "actual_minutes": int(row.get("actual_minutes") or 0),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True,
                        help="minute_gap_queue.sqlite 或缺口 CSV")
    parser.add_argument("--output", type=Path, required=True,
                        help="报告目录，必须位于 data-dir 内且不存在")
    parser.add_argument("--max-items", type=int, default=50)
    args = parser.parse_args()

    root = args.data_dir.resolve()
    queue_path = args.queue.resolve()
    output = args.output.resolve()
    if not queue_path.exists():
        raise FileNotFoundError(queue_path)
    if output == root or not output.is_relative_to(root):
        raise ValueError("报告输出目录必须位于 data-dir 内")
    targets = _read_targets(queue_path)
    deduped = {(str(row["symbol"]).upper(), str(row["date"])[:10]): row for row in targets}
    targets = list(deduped.values())[:max(1, args.max_items)]
    if not targets:
        print('{"checked": 0, "message": "没有可核验的股票日"}')
        return 0

    # Import the application service only after CLI validation so ``--help``
    # remains usable in a lightweight operator environment.
    from app.services.daily_maintenance import _cross_source_verify_failures
    from app.services.minute_cross_source import persist_verification_report
    from app.services.minute_gap_queue import record_cross_source_results

    repo = SimpleNamespace(store=SimpleNamespace(data_dir=root))
    results = _cross_source_verify_failures(repo, targets, max_items=len(targets))
    report_dir = persist_verification_report(
        root, results, task_kind="manual_minute", output_dir=output,
    )
    queue_result = record_cross_source_results(root, results) if results else {}
    print({
        "checked": len(results),
        "requested": len(targets),
        "report_dir": report_dir,
        "queue": queue_result,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
