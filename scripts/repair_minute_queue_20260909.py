"""One-time approved, offline repair of the September 9 minute queue.

Run only with the application container stopped and the TDX bridge running.
Already valid stock days are retained even when another source differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from app.plugins.tdx.provider import TdxProvider
from app.services.kline_sync import _atomic_write_parquet
from app.services.minute_repair import checked_additions, daily_reference, keyed, regular

APPROVED = {
    "2026-09-08": {"002136", "603215"},
    "2026-09-09": {"000531", "002643", "003816", "301197", "301616", "600405", "600505",
                   "600579", "600642", "600955", "603169", "603212", "603612", "603616"},
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--apply-approved-replacements", action="store_true", required=True)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    container = json.loads(subprocess.check_output(
        ["docker", "inspect", "TickFlow_Stock_Panel"], text=True))[0]
    if container["State"]["Running"]:
        raise RuntimeError("Stop the application container before offline repair")
    mount = next(m for m in container["Mounts"] if m["Destination"] == "/app/data")
    if Path(mount["Source"]).resolve() != root:
        raise RuntimeError("Data directory must match the stopped application's bind mount")
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if audit["task_id"] != "be41ce688d" or len(audit["rows"]) != 110:
        raise RuntimeError("Expected the reviewed 55-stock, two-day audit")
    approved = {(d, s) for d, symbols in APPROVED.items() for s in symbols}
    reviewed = {(r["date"], r["symbol"]) for r in audit["rows"]
                if r["source_validation"] == "accepted" and r["old_validation"] != "accepted"
                and r["merge_reason"] in {"existing_values_conflict", "existing_symbol_alias_conflict"}}
    if reviewed != approved:
        raise RuntimeError("Audit conflicts differ from the 16 explicitly approved stock days")
    run = f"minute_queue_repair_{datetime.now():%Y%m%dT%H%M%S}"
    report_path = root / "import_reports" / f"{run}.json"
    report = {"task_id": audit["task_id"], "approved_stock_days": sorted(approved),
              "source": "tdx", "backups": [], "rows": [], "status": "running"}
    provider = TdxProvider()
    provider.base_url, provider.timeout = args.base_url.rstrip("/"), 15
    try:
        for day_text in sorted(APPROVED):
            day = date.fromisoformat(day_text)
            if datetime.now() < datetime.combine(day, time(15, 10)):
                raise RuntimeError("Target session is not closed")
            path = root / "kline_minute" / f"date={day}" / "part.parquet"
            daily = pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet")
            refs = daily_reference(daily, data_dir=root)
            old = pl.read_parquet(path)
            old_by = keyed(old)
            groups = old_by.partition_by("_code", as_dict=True)
            changes, replace_codes, touched = [], [], []
            for row in (r for r in audit["rows"] if r["date"] == day_text):
                code = row["symbol"]
                current = groups.get((code,), old_by.clear()).drop("_code")
                _, old_reason = checked_additions(current, current.clear(), refs.get(code), day)
                detail = {"date": day_text, "symbol": code, "old_rows": current.height}
                if old_reason == "accepted":
                    report["rows"].append({**detail, "result": "retained_valid"})
                    continue
                source = provider.get_minute([code], datetime.combine(day, time.min),
                                             datetime.combine(day, time(15)))
                source = regular(source, day, include_opening=True) if not source.is_empty() else source
                delta, reason = checked_additions(source, current, refs.get(code), day)
                if reason in {"existing_values_conflict", "existing_symbol_alias_conflict"} and (day_text, code) in approved:
                    delta, reason = checked_additions(source, source.clear(), refs.get(code), day)
                    if reason == "accepted":
                        delta = delta.with_columns(pl.lit(refs[code]["symbol"]).alias("symbol"))
                        replace_codes.append(code)
                        reason = "approved_replacement"
                report["rows"].append({**detail, "result": reason, "written_rows": delta.height})
                if not delta.is_empty():
                    changes.append(delta)
                    touched.append(code)
                print(json.dumps({"date": day_text, "symbol": code, "result": reason}), flush=True)
            if not changes:
                continue
            untouched = old_by.filter(~pl.col("_code").is_in(replace_codes)).drop("_code")
            expected = pl.concat([untouched, *changes], how="diagonal_relaxed").sort(["symbol", "datetime"])
            expected_groups = keyed(expected).partition_by("_code", as_dict=True)
            for code in touched:
                part = expected_groups[(code,)].drop("_code")
                _, reason = checked_additions(part, part.clear(), refs.get(code), day)
                if reason != "accepted":
                    raise RuntimeError(f"Merged day failed validation: {day} {code} {reason}")
            backup = root / "import_backups" / run / f"date={day}" / "part.parquet"
            backup.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(path, backup)
            sha = digest(path)
            if digest(backup) != sha:
                raise RuntimeError("Backup checksum mismatch; original partition untouched")
            report["backups"].append({"path": str(backup), "sha256": sha})
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            _atomic_write_parquet(expected, path)
            if not pl.read_parquet(path).sort(["symbol", "datetime"]).equals(expected):
                _atomic_write_parquet(old, path)
                raise RuntimeError("Readback failed; original partition restored")
        from app.services.daily_maintenance import minute_validation_failures
        symbols = sorted({r["symbol"] for r in audit["rows"]})
        failures = minute_validation_failures(SimpleNamespace(store=SimpleNamespace(data_dir=root)),
            symbols, datetime(2026, 9, 8), datetime(2026, 9, 9, 15))
        report["validation_failures"] = failures
        report["completed_symbols"] = sorted(set(symbols) - {r["symbol"] for r in failures})
        if (root / "user_data/maintenance_queue.json").exists():
            from app.services.maintenance_queue import checkpoint
            checkpoint(root, audit["task_id"], report["completed_symbols"])
        report["status"] = "complete" if not failures else "partial"
    finally:
        provider.close()
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "status": report["status"]}))


if __name__ == "__main__":
    main()
