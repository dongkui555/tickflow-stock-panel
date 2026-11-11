"""Back up and replace the final 27 BSE 2025 conflicts with validated DataHub minutes."""
from __future__ import annotations

import csv
import json
import time
from datetime import date
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import polars as pl

from app.plugins.tushare.provider import TushareProvider, build_datahub_client
from app.services.minute_repair import repair_minute_gaps


def main() -> None:
    root = Path("/app/data")
    csv_path = root / "import_reports" / "audit_2025_bj_after_qmt_20260922" / "missing_or_partial.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    stock_days = [(row.get("selected_symbol") or f"{row['code']}.BJ", date.fromisoformat(row["date"])) for row in rows]
    provider = TushareProvider()
    provider._api = build_datahub_client()
    repo = SimpleNamespace(
        store=SimpleNamespace(data_dir=root),
        _write_lock=RLock(),
        rebuild_views=lambda: None,
        clear_cache=lambda: None,
    )
    began = time.monotonic()

    def progress(done: int, total: int, message: str) -> None:
        print(json.dumps({
            "done": done,
            "total": total,
            "elapsed_s": round(time.monotonic() - began),
            "message": message,
        }, ensure_ascii=False), flush=True)

    try:
        result = repair_minute_gaps(
            repo,
            start_date=min(day for _, day in stock_days),
            end_date=max(day for _, day in stock_days),
            provider_name="tushare",
            source_provider=provider,
            stock_days=stock_days,
            workers=1,
            repair_conflicts=True,
            bypass_failover=True,
            progress=progress,
        )
    finally:
        provider.close()
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
