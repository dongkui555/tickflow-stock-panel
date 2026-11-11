"""Offline maintenance import using the same add-only transaction as online repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import date
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import polars as pl

from app.services.minute_repair import COLS, repair_minute_gaps


class StagedMinuteProvider:
    def __init__(self, folders: list[Path]):
        self.files = {}
        self.inventory = []
        for folder in folders:
            folder = folder.resolve()
            rows = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            for row in rows:
                if row.get("reason") != "accepted" or not row.get("file"):
                    continue
                symbol = row["symbol"]
                day = date.fromisoformat(row["date"])
                if not re.fullmatch(r"\d{6}\.(SZ|SH|BJ)", symbol):
                    raise ValueError("Invalid staged symbol")
                path = (folder / row["file"]).resolve()
                if not path.is_relative_to(folder) or path.suffix != ".parquet":
                    raise ValueError("Staged file escapes its report directory")
                key = (symbol[:6], day)
                if key in self.files:
                    raise ValueError("Duplicate staged stock-day")
                self.files[key] = path
                self.inventory.append({"symbol": symbol, "date": str(day), "file": str(path),
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def get_minute(self, symbols, start, end, **kwargs):
        frames = []
        codes = {symbol.split(".")[0] for symbol in symbols}
        for (code, day), path in self.files.items():
            if code not in codes or not start.date() <= day <= end.date():
                continue
            frame = pl.read_parquet(path).select(COLS)
            if not frame.select((pl.col("datetime").dt.date() == day).fill_null(False).all()).item():
                raise ValueError("Staged file contains another date")
            frames.append(frame)
        return pl.concat(frames) if frames else pl.DataFrame()


def require_stopped(container: str) -> None:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State}}", container],
        capture_output=True, text=True, timeout=15, check=True,
    )
    state = json.loads(result.stdout)
    if state.get("Running") is not False or state.get("Restarting") is not False:
        raise RuntimeError("Stop the application container before offline maintenance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--staged-dir", type=Path, action="append", required=True)
    parser.add_argument("--container", default="TickFlow_Stock_Panel")
    parser.add_argument("--assume-stopped", action="store_true",
                        help="Use from a one-shot container when the host has already stopped the app")
    args = parser.parse_args()
    root = args.data_dir.resolve()
    if any(not folder.resolve().is_relative_to(root / "import_reports") for folder in args.staged_dir):
        raise ValueError("Only staging folders inside this data directory are accepted")
    if not args.assume_stopped:
        require_stopped(args.container)
    source = StagedMinuteProvider(args.staged_dir)
    if not source.files:
        print(json.dumps({"added_rows": 0, "reason": "no_accepted_staged_files"}))
        return
    # The application is stopped. Its caches and views will reload on startup;
    # this maintenance process does not open the live DuckDB database.
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=root), _write_lock=RLock(),
                           rebuild_views=lambda: None, clear_cache=lambda: None)
    pairs = list(source.files)
    result = repair_minute_gaps(
        repo, start_date=min(day for _, day in pairs), end_date=max(day for _, day in pairs),
        provider_name="staged_minute", source_provider=source, stock_days=pairs,
        repair_conflicts=False, single_day_requests=True, workers=1,
        progress=lambda done, total, message: print(json.dumps({"done": done, "total": total, "message": message}), flush=True),
    )
    inventory_path = Path(result["report_path"]).with_name("staged_sources.json")
    inventory_path.write_text(json.dumps(source.inventory, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
