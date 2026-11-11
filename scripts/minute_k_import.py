"""Reusable QMT minute-K gap importer.

The command consumes an audit CSV, processes one trading date at a time, and
delegates identity, session, daily OHLCV and atomic-write checks to the same
``minute_repair`` service used by the application. It is intentionally
idempotent: rerunning the same audit only revisits unresolved stock-days.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

import polars as pl

from app.services.minute_repair import repair_minute_gaps
from app.tickflow.repository import DataStore, KlineRepository
from repair_targeted_qmt_dat import QmtDatProvider


DEFAULT_PENDING_STATUSES = ("missing_0", "invalid_partial")
CONFLICT_STATUSES = ("complete_conflict", "invalid_time_axis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import audited QMT minute-K gaps with date-gated validation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--qmt-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True,
                        help="missing_or_partial.csv from audit_market_minute_coverage.py")
    parser.add_argument("--markets", nargs="+", choices=("BJ", "SH", "SZ"))
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int,
                        help="Maximum stock-days to process, useful for a dry run.")
    parser.add_argument("--include-conflicts", action="store_true",
                        help="Include complete_conflict/invalid_time_axis rows.")
    parser.add_argument("--repair-conflicts", action="store_true",
                        help="Back up and replace only independently validated conflicts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the durable queue summary without writing minute data.")
    return parser.parse_args()


def load_queue(path: Path, args: argparse.Namespace) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"audit CSV not found: {path}")
    # A zero-byte missing_or_partial.csv is the valid audit representation for
    # an empty repair queue. Treat it as success instead of letting Polars
    # raise NoDataError and turning a completed audit into a failed import.
    if path.stat().st_size == 0:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "code": pl.String,
                "market": pl.String,
                "status": pl.String,
            }
        )
    # Audit rows can contain mostly-null optional text columns (for example
    # ``selected_symbol``). Letting Polars infer those columns from the first
    # rows can incorrectly choose an integer dtype and reject later symbols
    # such as ``920580.BJ``. Keep optional identity fields textual.
    with path.open("r", encoding="utf-8", newline="") as handle:
        columns = next(csv.reader(handle), [])
    frame = pl.read_csv(
        path,
        try_parse_dates=True,
        schema_overrides={
            column: pl.String
            for column in ("code", "market", "status", "selected_symbol", "symbol")
            if column in columns
        },
    )
    required = {"date", "code", "market", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"audit CSV is missing columns: {sorted(missing)}")

    statuses = list(DEFAULT_PENDING_STATUSES)
    if args.include_conflicts:
        statuses.extend(CONFLICT_STATUSES)
    queue = (
        frame
        .with_columns(
            pl.col("code").cast(pl.String).str.extract(r"(\d{6})", 1).alias("code"),
            pl.col("market").cast(pl.String).str.to_uppercase(),
            pl.col("date").cast(pl.Date),
        )
        .filter(pl.col("status").is_in(statuses))
    )
    if args.markets:
        queue = queue.filter(pl.col("market").is_in(args.markets))
    if args.start:
        queue = queue.filter(pl.col("date") >= args.start)
    if args.end:
        queue = queue.filter(pl.col("date") <= args.end)
    queue = queue.filter(
        pl.col("code").is_not_null()
        & pl.col("market").is_in(("BJ", "SH", "SZ"))
    ).unique(["code", "market", "date"], keep="last").sort(["date", "market", "code"])
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        queue = queue.head(args.limit)
    return queue


def queue_summary(queue: pl.DataFrame) -> dict:
    if queue.is_empty():
        return {"stock_days": 0, "dates": 0, "by_market": {}, "by_status": {}}
    return {
        "stock_days": queue.height,
        "dates": queue["date"].n_unique(),
        "by_market": {
            row["market"]: row["len"]
            for row in queue.group_by("market").len().sort("market").to_dicts()
        },
        "by_status": {
            row["status"]: row["len"]
            for row in queue.group_by("status").len().sort("status").to_dicts()
        },
        "first_date": str(queue["date"].min()),
        "last_date": str(queue["date"].max()),
    }


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    qmt_root = args.qmt_root.resolve()
    if not qmt_root.is_dir():
        raise NotADirectoryError(f"QMT root not found: {qmt_root}")
    queue = load_queue(args.audit.resolve(), args)
    summary = queue_summary(queue)
    print(json.dumps({"queue": summary, "dry_run": args.dry_run}, ensure_ascii=False), flush=True)
    if args.dry_run or queue.is_empty():
        return 0

    stock_days = [
        (f"{row['code']}.{row['market']}", row["date"])
        for row in queue.select("code", "market", "date").iter_rows(named=True)
    ]
    # The audit is an explicit market-qualified identity source. Pass only
    # unambiguous code -> symbol mappings into lifecycle validation so a
    # live-only instrument snapshot cannot reject valid historical rows.
    identity_candidates: dict[str, set[str]] = {}
    for symbol, _day in stock_days:
        code = symbol.split(".", 1)[0]
        identity_candidates.setdefault(code, set()).add(symbol)
    identity_lookup = {
        code: next(iter(symbols))
        for code, symbols in identity_candidates.items()
        if len(symbols) == 1
    }
    provider = QmtDatProvider(qmt_root)
    repo = KlineRepository(DataStore(data_dir=data_dir))
    results: list[dict] = []
    dates = sorted({day for _, day in stock_days})
    pairs_by_date = {
        day: [(symbol, pair_day) for symbol, pair_day in stock_days if pair_day == day]
        for day in dates
    }

    for index, day in enumerate(dates, 1):
        day_pairs = pairs_by_date[day]
        print(json.dumps({
            "date": str(day),
            "date_index": index,
            "date_total": len(dates),
            "stock_days": len(day_pairs),
            "message": "processing one trading date",
        }, ensure_ascii=False), flush=True)

        def progress(done: int, total: int, message: str) -> None:
            print(json.dumps({
                "date": str(day), "done": done, "total": total, "message": message,
            }, ensure_ascii=False), flush=True)

        result = repair_minute_gaps(
            repo,
            start_date=day,
            end_date=day,
            provider_name="qmt",
            source_provider=provider,
            stock_days=day_pairs,
            workers=args.workers,
            repair_conflicts=args.repair_conflicts,
            single_day_requests=True,
            bypass_failover=True,
            identity_lookup=identity_lookup,
            progress=progress,
        )
        results.append(result)
        print(json.dumps({
            "date": str(day),
            "completed": True,
            "added_rows": result.get("added_rows", 0),
            "replaced_rows": result.get("replaced_rows", 0),
            "rejected_rows": result.get("rejected_rows", 0),
            "reasons": result.get("reasons", {}),
            "report_path": result.get("report_path"),
        }, ensure_ascii=False, default=str), flush=True)

    totals = Counter()
    reports = []
    for result in results:
        for key in ("added_rows", "replaced_rows", "removed_rows", "rejected_rows",
                    "duplicate_rows", "repaired_stock_days", "requests_checked"):
            totals[key] += int(result.get(key, 0) or 0)
        totals.update(result.get("reasons", {}))
        reports.append(result.get("report_path"))
    final = {
        "queue": summary,
        "dates_completed": len(results),
        "totals": dict(totals),
        "reports": reports,
        "idempotent": True,
        "note": "Unresolved source-empty or validation rows remain in the next audit queue; no synthetic bars were written.",
    }
    print(json.dumps(final, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
