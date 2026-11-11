"""Remove byte-value identical duplicates only from keyed daily aggregates."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import polars as pl

from app.parquet import atomic_write_parquet


ALLOWED = {
    "auction_open", "moneyflow", "kline_daily",
    "ext_data/ext_tushare_sw_daily/timeseries",
    "ext_data/ext_tushare_ths_daily/timeseries",
    "ext_data/ext_tushare_dc_daily/timeseries",
    "ext_data/ext_tushare_stk_limit/timeseries",
    "ext_data/ext_tushare_top_list/timeseries",
    "ext_data/ext_tushare_top_inst/timeseries",
    "ext_data/ext_tushare_hsgt_top10/timeseries",
    "ext_data/ext_tushare_ggt_top10/timeseries",
}


def repair(root: Path, audit_path: Path):
    root = root.resolve()
    run_id = "market_exact_duplicates_" + uuid.uuid4().hex[:12]
    report = root / "import_reports" / run_id
    report.mkdir(parents=True)
    result = {"run_id": run_id, "removed_exact_duplicates": 0, "writes": []}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for item in audit["issues"]:
        if item["dataset"] not in ALLOWED or not item.get("duplicates"):
            continue
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("path outside data directory")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        old = pl.read_parquet(path)
        new = old.unique(maintain_order=True)
        if old.height == new.height:
            continue
        backup = root / "import_backups" / run_id / item["file"]
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise RuntimeError("file changed concurrently; no replacement")
        if hashlib.sha256(backup.read_bytes()).hexdigest() != before:
            raise RuntimeError("backup hash mismatch; no replacement")
        atomic_write_parquet(new, path)
        if not pl.read_parquet(path).equals(new):
            atomic_write_parquet(old, path)
            raise RuntimeError("readback mismatch; restored original rows")
        removed = old.height - new.height
        result["removed_exact_duplicates"] += removed
        result["writes"].append({"file": item["file"], "removed": removed,
                                 "backup": str(backup.relative_to(root)), "sha256_before": before})
        (report / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report), "files": len(result["writes"]),
                      "removed": result["removed_exact_duplicates"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    repair(args.data_dir, args.audit)
