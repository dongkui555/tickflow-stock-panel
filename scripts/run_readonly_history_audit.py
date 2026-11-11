"""Run existing audits against a read-only mount; keep evidence outside DATA_DIR."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def inventory(root: Path) -> dict:
    files = {}
    bases = [p for p in root.iterdir() if p.is_dir() and (
        p.name.startswith(("kline_", "instruments", "adj_factor")) or p.name in {
            "moneyflow", "auction_open", "financials", "ext_data", "news", "dragon_tiger", "depth5",
        }
    )]
    for path in sorted(path for base in bases for path in base.rglob("*.parquet")):
        try:
            stat = path.stat()
            files[path.relative_to(root).as_posix()] = {
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            }
        except FileNotFoundError:
            continue
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Audit output must be outside the source data directory")
    output.mkdir(parents=True, exist_ok=True)
    before = inventory(root)
    (output / "inventory-before.json").write_text(json.dumps(before, indent=2))
    scripts = Path(__file__).parent
    results = {"started_at": datetime.now(timezone.utc).isoformat(), "steps": []}
    print(json.dumps({"phase": "inventory", "files": len(before),
                      "bytes": sum(v["size"] for v in before.values())}), flush=True)
    try:
        commands = [
            ["audit_market_data.py", "--data-dir", str(root), "--output-dir", str(output / "market"),
             "--start", "1990-01-01", "--end", args.end],
            ["audit_minute_integrity.py", "--data-dir", str(root), "--output-dir", str(output / "minute"),
             "--start-date", "1990-01-01", "--end-date", args.end],
        ]
        for command in commands:
            code = subprocess.call([sys.executable, str(scripts / command[0]), *command[1:]])
            results["steps"].append({"script": command[0], "exit_code": code})
    finally:
        after = inventory(root)
        (output / "inventory-after.json").write_text(json.dumps(after, indent=2))
        results["source_changes"] = [
            {"file": key, "before": before.get(key), "after": after.get(key)}
            for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)
        ]
        results["completed_at"] = datetime.now(timezone.utc).isoformat()
        results["note"] = "Input mounted read-only. Background writers may change source files; affected observations require recheck."
        (output / "run.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"phase": "completed", "changed_source_files": len(results["source_changes"])}), flush=True)
    return int(any(step["exit_code"] for step in results["steps"]))


if __name__ == "__main__":
    raise SystemExit(main())
