"""Run with the app stopped; verify a queue-only audit did not touch market/forecast data."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from app.services import maintenance_queue


def protected_state(root):
    content = [root / "regime_history/part.parquet", root / "regime_history/sources.json",
               root / "user_data/sector_forecasts_v2.sqlite3"]
    content += list((root / "user_data/prediction_workflow/snapshots").glob("*.json"))
    hashes = {}
    for path in content:
        if path.exists():
            with path.open("rb") as handle:
                hashes[str(path.relative_to(root))] = hashlib.file_digest(handle, "sha256").hexdigest()
    tables = {}
    for table in ("kline_daily", "kline_daily_enriched", "kline_index_daily", "kline_minute", "adj_factor"):
        entries = []
        for path in sorted((root / table).rglob("*.parquet")):
            stat = path.stat()
            entries.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
        tables[table] = {"files": len(entries), "metadata_sha256": hashlib.sha256(json.dumps(entries).encode()).hexdigest()}
    return {"content_sha256": hashes, "table_metadata": tables}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        parser.error("Stop the application, then explicitly pass --apply")
    rows = maintenance_queue._read(args.data_dir)
    if any(r["status"] == "running" for r in rows):
        raise RuntimeError("Running queue entry present; do not audit concurrent writers")
    before = protected_state(args.data_dir)
    report = maintenance_queue.audit_regime(args.data_dir)
    after = protected_state(args.data_dir)
    if before != after:
        raise RuntimeError("Protected data changed; investigate before starting the application")
    tasks = maintenance_queue.status(args.data_dir)
    paths = {t["audit_report"] for t in tasks if t.get("audit_report")}
    verification = {"protected_unchanged": True, **after}
    for path in paths:
        target = (args.data_dir / path).with_name("protected-data-verification.json")
        if not target.exists():
            target.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    summary = [{"task_id": t["task_id"], "counts": t["counts"],
                "reasons": dict(Counter(d["reason"] for d in t["dates"]))} for t in report["tasks"]]
    print(json.dumps({"audit": summary, "protected_unchanged": True, "reports": sorted(paths)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
