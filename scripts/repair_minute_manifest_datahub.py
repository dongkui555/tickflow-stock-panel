"""Repair audited minute stock-days from the configured Tushare DataHub.

The manifest must contain ``date`` and either ``selected_symbol`` or
``code``/``market``. Writes remain guarded by the shared identity, session,
daily OHLCV, backup, and read-back checks in ``repair_minute_gaps``.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import uuid
from datetime import date, datetime, time as wall_time
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _stock_days(path: Path) -> list[tuple[str, date]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: list[tuple[str, date]] = []
    for row in rows:
        day = date.fromisoformat(str(row.get("date") or "")[:10])
        symbol = str(row.get("selected_symbol") or row.get("symbol") or "").strip().upper()
        if not symbol:
            code = str(row.get("code") or "").strip().zfill(6)
            market = str(row.get("market") or row.get("exchange") or "").strip().upper()
            symbol = f"{code}.{market}"
        result.append((symbol, day))
    return sorted(set(result), key=lambda item: (item[1], item[0]))


def _direct_repair(root: Path, stock_days: list[tuple[str, date]], progress_every: int) -> dict:
    """Fetch and write on the main thread for clients that are not thread-safe."""
    from app.plugins.tushare.provider import TushareProvider, build_datahub_client
    from app.services import kline_sync
    from app.services.minute_repair import COLS, checked_additions, daily_reference, keyed
    from app.services.security_lifecycle import validate_supplement_identity

    provider = TushareProvider()
    provider._api = build_datahub_client()  # noqa: SLF001 - operator-selected gateway
    run_id = f"datahub_direct_{uuid.uuid4().hex[:12]}"
    report_dir = root / "import_reports" / run_id
    backup_root = root / "import_backups" / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict] = []
    backed_up: set[date] = set()
    started = time.monotonic()
    try:
        for index, (symbol, day) in enumerate(stock_days, 1):
            code = symbol.split(".")[0]
            daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
            minute_path = root / "kline_minute" / f"date={day}" / "part.parquet"
            result = {"date": str(day), "symbol": symbol, "status": "pending"}
            if not daily_path.exists():
                result["status"] = "daily_partition_missing"
                results.append(result)
                continue
            reference = daily_reference(pl.read_parquet(daily_path), data_dir=root).get(code)
            if reference is None:
                result["status"] = "daily_reference_missing"
                results.append(result)
                continue
            old = pl.read_parquet(minute_path, columns=COLS) if minute_path.exists() else pl.DataFrame(schema={
                "symbol": pl.String, "datetime": pl.Datetime,
                "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
                "close": pl.Float64, "volume": pl.Float64, "amount": pl.Float64,
            })
            old_by = keyed(old)
            current = old_by.filter(pl.col("_code") == code).drop("_code")
            if not current.is_empty():
                _, current_reason = checked_additions(
                    current, current.clear(), reference, day, source="tushare",
                )
                if current_reason == "accepted":
                    result["status"] = "already_complete"
                    result["old_rows"] = current.height
                    results.append(result)
                    continue

            frame, issues = provider.get_minute_diagnostics(
                [symbol],
                datetime.combine(day, wall_time(9, 25)),
                datetime.combine(day, wall_time(15, 35)),
                freq="1m",
            )
            verified, validation = checked_additions(
                frame, frame.clear(), reference, day, source="tushare",
            )
            result.update({
                "source_rows": frame.height,
                "validation": validation,
                "issues": issues,
            })
            if validation != "accepted" or verified.is_empty():
                result["status"] = "rejected"
                results.append(result)
                continue

            source_symbol = str(verified["symbol"][0])
            identity = validate_supplement_identity(symbol, source_symbol, day, root)
            if not identity.get("ok") and identity.get("reason") != "identity_history_unknown":
                result["status"] = identity.get("reason", "market_identity_conflict")
                result["identity"] = identity
                results.append(result)
                continue

            canonical = str(reference.get("symbol") or symbol).upper()
            verified = verified.with_columns(pl.lit(canonical).alias("symbol"))
            untouched = old_by.filter(pl.col("_code") != code).drop("_code")
            expected = pl.concat([untouched, verified], how="diagonal_relaxed").sort(["symbol", "datetime"])
            if minute_path.exists() and day not in backed_up:
                backup = backup_root / f"date={day}" / "part.parquet"
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(minute_path, backup)
                backed_up.add(day)
            kline_sync._atomic_write_parquet(expected, minute_path)
            after = pl.read_parquet(minute_path, columns=COLS).sort(["symbol", "datetime"])
            if not after.equals(expected):
                kline_sync._atomic_write_parquet(old, minute_path)
                raise RuntimeError(f"readback verification failed; restored {day} {symbol}")
            result.update({
                "status": "replaced" if not current.is_empty() else "inserted",
                "old_rows": current.height,
                "new_rows": verified.height,
            })
            results.append(result)
            if index == len(stock_days) or index % max(1, progress_every) == 0:
                print(json.dumps({
                    "done": index,
                    "total": len(stock_days),
                    "elapsed_s": round(time.monotonic() - started),
                    "inserted": sum(row["status"] == "inserted" for row in results),
                    "replaced": sum(row["status"] == "replaced" for row in results),
                    "rejected": sum(row["status"] == "rejected" for row in results),
                }, ensure_ascii=False), flush=True)
    finally:
        provider.close()

    counts: dict[str, int] = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = {
        "run_id": run_id,
        "planned": len(stock_days),
        "counts": counts,
        "backup_root": str(backup_root),
        "elapsed_s": round(time.monotonic() - started),
        "results": results,
    }
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, default=str, indent=2), encoding="utf-8",
    )
    return {key: value for key, value in report.items() if key != "results"} | {
        "report_path": str(report_dir / "report.json")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--direct", action="store_true",
        help="fetch and write on the main thread instead of the shared repair thread pool",
    )
    args = parser.parse_args()

    root = args.data_dir.resolve()
    manifest = args.manifest.resolve()
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if args.workers < 1:
        parser.error("--workers must be positive")

    stock_days = _stock_days(manifest)
    if not stock_days:
        print(json.dumps({"planned": 0, "message": "manifest is empty"}))
        return 0

    if args.direct:
        print(json.dumps(
            _direct_repair(root, stock_days, args.progress_every),
            ensure_ascii=False, default=str,
        ), flush=True)
        return 0

    from app.plugins.tushare.provider import TushareProvider, build_datahub_client
    from app.services.minute_repair import repair_minute_gaps

    provider = TushareProvider()
    provider._api = build_datahub_client()  # noqa: SLF001 - operator-selected gateway
    repo = SimpleNamespace(
        store=SimpleNamespace(data_dir=root),
        _write_lock=RLock(),
        rebuild_views=lambda: None,
        clear_cache=lambda: None,
    )
    started = time.monotonic()

    def progress(done: int, total: int, message: str) -> None:
        if done == total or done % max(1, args.progress_every) == 0:
            print(json.dumps({
                "done": done,
                "total": total,
                "elapsed_s": round(time.monotonic() - started),
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
            workers=args.workers,
            repair_conflicts=True,
            bypass_failover=True,
            single_day_requests=True,
            progress=progress,
            identity_lookup={symbol.split(".")[0]: symbol for symbol, _ in stock_days},
        )
    finally:
        provider.close()

    summary = {key: result.get(key) for key in (
        "requests_planned", "requests_checked", "stock_days_planned",
        "written_rows", "replaced_rows", "removed_rows", "repaired_stock_days",
        "duplicate_rows", "rejected_rows", "reasons", "report_path", "complete",
    )}
    summary["elapsed_s"] = round(time.monotonic() - started)
    print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
