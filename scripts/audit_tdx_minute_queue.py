"""Read-only source/old-data comparison for one existing minute maintenance task."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from app.plugins.tdx.provider import TdxProvider
from app.services.minute_repair import checked_additions, daily_reference, keyed, regular


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--limit", type=int, default=55)
    args = parser.parse_args()
    root = args.data_dir
    tasks = json.loads((root / "user_data/maintenance_queue.json").read_text(encoding="utf-8"))
    task = next(t for t in tasks if t["id"] == args.task_id)
    end = datetime.fromisoformat(task["options"]["as_of"])
    begin = (end - timedelta(days=task["options"]["days"])).replace(hour=0, minute=0)
    end = end.replace(hour=15, minute=0)
    symbols = task.get("known_items", task["items"])[:max(1, min(100, args.limit))]
    refs, old = {}, {}
    for folder in sorted((root / "kline_daily").glob("date=*")):
        day = date.fromisoformat(folder.name[5:])
        if not begin.date() <= day <= end.date():
            continue
        refs[day] = daily_reference(pl.read_parquet(folder / "part.parquet"), data_dir=root)
        path = root / "kline_minute" / f"date={day}" / "part.parquet"
        frame = keyed(pl.read_parquet(path)).filter(pl.col("_code").is_in(symbols)) if path.exists() else pl.DataFrame()
        old[day] = frame.partition_by("_code", as_dict=True) if not frame.is_empty() else {}
    provider = TdxProvider()
    provider.base_url = args.base_url.rstrip("/")
    provider.timeout = 15
    rows = []
    try:
        for index, code in enumerate(symbols, 1):
            source = provider.get_minute([code], begin, end)
            for day in refs:
                part = regular(source, day, include_opening=True) if not source.is_empty() else source
                previous = old[day].get((code,), part.clear())
                if "_code" in previous.columns:
                    previous = previous.drop("_code")
                _, validation = checked_additions(part, part.clear(), refs[day].get(code), day)
                _, old_validation = checked_additions(previous, previous.clear(), refs[day].get(code), day)
                delta, reason = checked_additions(part, previous, refs[day].get(code), day)
                rows.append(dict(symbol=code, date=str(day), source_rows=part.height,
                                 source_validation=validation, old_validation=old_validation,
                                 old_rows=previous.height, merge_reason=reason, additions=delta.height,
                                 has_lunch=not part.is_empty() and datetime.combine(day, time(11, 30)) in part["datetime"]))
            print(json.dumps({"progress": f"{index}/{len(symbols)}", "symbol": code}), flush=True)
    finally:
        provider.close()
    report = {"task_id": args.task_id, "mode": "read_only", "rows": rows,
              "source_counts": dict(Counter(r["source_validation"] for r in rows)),
              "old_counts": dict(Counter(r["old_validation"] for r in rows)),
              "merge_counts": dict(Counter(r["merge_reason"] for r in rows))}
    path = root / "import_reports" / f"tdx_queue_audit_{datetime.now():%Y%m%dT%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(path), **{k: report[k] for k in ("source_counts", "merge_counts")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
