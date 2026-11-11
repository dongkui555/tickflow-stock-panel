"""Read-only, bounded cross-check of staged StockDB minutes against online data."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from app.plugins.tushare.provider import TushareProvider
from app.services.minute_repair import (
    COLS, PRICE, _daily_aggregate_exprs, checked_additions, daily_reference,
    keyed, regular,
)


def matches(left: dict, right: dict) -> bool:
    if not left or not right:
        return False
    prices = all(
        abs(float(left[c]) - float(right[c])) <= max(0.011, abs(float(right[c])) * 1e-6)
        for c in PRICE
    )
    volume = any(
        abs(float(left["volume"]) * scale - float(right["volume"]))
        <= max(1.0, abs(float(right["volume"])) * 0.002)
        for scale in (1, 100)
    )
    amount = abs(float(left["amount"]) - float(right["amount"])) <= max(
        1.0, abs(float(right["amount"])) * 0.002
    )
    return prices and volume and amount


def summarize(frame: pl.DataFrame, day: date) -> tuple[dict | None, str]:
    if frame.is_empty():
        return None, "source_empty"
    frame = regular(frame.select(COLS), day, include_opening=True).sort("datetime")
    if frame.is_empty():
        return None, "source_empty"
    aggregate = frame.select(_daily_aggregate_exprs()).row(0, named=True)
    if any(aggregate[c] is None for c in (*PRICE, "volume", "amount")):
        return None, "source_invalid"
    _, reason = checked_additions(frame, frame.clear(), aggregate, day)
    return aggregate, reason


def classify(local_daily: dict | None, online_daily: dict | None,
             stockdb: dict | None, stockdb_reason: str,
             online_minute: dict | None, online_minute_reason: str) -> str:
    if local_daily is None:
        return "local_daily_missing"
    if online_daily is None:
        return "online_daily_missing"
    if stockdb_reason != "accepted":
        return "stockdb_invalid"
    if online_minute_reason == "not_requested":
        return "online_minute_not_requested"
    if online_minute is None or online_minute_reason != "accepted":
        if online_minute_reason in {"source_pending", "rate_limited", "request_timeout", "transport_error", "upstream_unavailable"}:
            return "online_minute_retry"
        return "online_minute_unverified"
    online_local = matches(online_minute, local_daily)
    online_stockdb = matches(online_minute, stockdb)
    daily_confirmed = matches(online_daily, local_daily)
    if online_local and not online_stockdb and daily_confirmed:
        return "replacement_candidate"
    if online_stockdb and not online_local:
        return "minute_sources_agree_daily_conflict"
    if online_local and online_stockdb:
        return "all_sources_agree"
    return "three_way_conflict"


def _symbol(code: str, local: dict | None) -> str:
    if local and "." in str(local.get("symbol", "")):
        return str(local["symbol"])
    suffix = "BJ" if code.startswith(("4", "8", "92")) else "SH" if code.startswith(("5", "6")) else "SZ"
    return f"{code}.{suffix}"


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(root: Path, staged_file: Path, day: date, output: Path,
        limit: int, max_minute_requests: int, max_daily_fallbacks: int) -> dict:
    report_path = root / "import_reports" / f"stockdb_sdk_{day:%Y%m%d}" / "report.json"
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if source_report.get("date") != str(day):
        raise ValueError("StockDB report date does not match requested date")
    all_conflicts = [row for row in source_report["details"] if row["reason"] in {
        "daily_price_conflict", "daily_turnover_conflict", "existing_values_conflict",
    }]
    selected = all_conflicts[:limit] if limit else all_conflicts
    local_path = root / "kline_daily" / f"date={day}" / "part.parquet"
    refs = daily_reference(pl.read_parquet(local_path), data_dir=root)
    staged = keyed(pl.read_parquet(staged_file, columns=COLS))
    wanted = {row["code"] for row in selected}
    staged = staged.filter(pl.col("_code").is_in(wanted))
    staged_by_code = {
        key[0] if isinstance(key, tuple) else key: frame.drop("_code")
        for key, frame in staged.partition_by("_code", as_dict=True).items()
    }
    symbols = [_symbol(row["code"], refs.get(row["code"])) for row in selected]
    provider = TushareProvider()
    result = {
        "date": str(day), "source": "StockDB staged / local daily / online Tushare-Promax",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_conflicts": len(all_conflicts), "selected": len(selected),
        "minute_request_cap": max_minute_requests, "daily_fallback_cap": max_daily_fallbacks,
        "production_data_writes": 0,
        "online_daily_may_share_lineage_with_local": True,
        "results": [],
    }
    try:
        start, end = datetime.combine(day, datetime.min.time()), datetime.combine(day, datetime.max.time())
        try:
            online = provider.get_daily(symbols, start, end)
            online_map = {str(row["symbol"]): row for row in online.to_dicts()}
            result["online_daily_batch_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            online_map = {}
            result["online_daily_batch_status"] = type(exc).__name__
        missing_symbols = [symbol for symbol in symbols if symbol.split(".")[0] not in online_map]
        result["online_daily_batch_missing"] = len(missing_symbols)
        result["online_daily_fallback_attempted"] = 0
        for symbol in missing_symbols[:max_daily_fallbacks]:
            try:
                # Some gateways truncate the all-market daily response. A
                # single-symbol query distinguishes truncation from no coverage.
                frame = provider.get_daily([symbol], start, end)
                for record in frame.to_dicts():
                    online_map[str(record["symbol"])] = record
            except Exception:  # noqa: BLE001
                pass
            result["online_daily_fallback_attempted"] += 1
        result["online_daily_missing_after_fallback"] = sum(
            symbol.split(".")[0] not in online_map for symbol in symbols
        )
        for index, row in enumerate(selected):
            code = row["code"]
            local, remote = refs.get(code), online_map.get(code)
            source = staged_by_code.get(code, pl.DataFrame(schema=staged.drop("_code").schema))
            source_aggregate, source_reason = summarize(source, day)
            minute_aggregate, minute_reason, minute_rows, issues = None, "not_requested", 0, []
            if remote is not None and index < max_minute_requests:
                try:
                    frame, issues = provider.get_minute_diagnostics(
                        [_symbol(code, local)], start, end, freq="1m",
                    )
                    minute_rows = frame.height
                    minute_aggregate, minute_reason = summarize(frame, day)
                    if minute_reason == "source_empty" and issues:
                        minute_reason = str(issues[-1].get("reason", "source_empty"))
                except Exception as exc:  # noqa: BLE001
                    minute_reason = "request_exception"
                    issues = [{"reason": minute_reason, "error_type": type(exc).__name__}]
            elif remote is None:
                minute_reason = "online_daily_missing"
            status = classify(local, remote, source_aggregate, source_reason,
                              minute_aggregate, minute_reason)
            retry_seconds = next((item.get("retry_after_seconds") for item in issues
                                  if item.get("retry_after_seconds")), None)
            entry = {
                "code": code, "original_reason": row["reason"], "status": status,
                "local_daily": {c: local[c] for c in (*PRICE, "volume", "amount")} if local else None,
                "online_daily": {c: remote[c] for c in (*PRICE, "volume", "amount")} if remote else None,
                "online_daily_matches_local": matches(remote, local),
                "stockdb_minutes": {"rows": source.height, "validation": source_reason,
                                     "aggregate": source_aggregate},
                "online_minutes": {"rows": minute_rows,
                                    "validation": minute_reason, "aggregate": minute_aggregate,
                                    "issues": issues},
                "next_retry_at_utc": (
                    datetime.now(timezone.utc) + timedelta(seconds=float(retry_seconds))
                ).isoformat() if retry_seconds else None,
            }
            result["results"].append(entry)
            result["status_counts"] = dict(Counter(x["status"] for x in result["results"]))
            if (index + 1) % 25 == 0 or index + 1 == len(selected):
                _write_json(output, result)
                print(json.dumps({"checked": index + 1, "total": len(selected), "code": code,
                                  "status": status, "minute_reason": minute_reason}, ensure_ascii=False), flush=True)
    finally:
        provider.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--staged-file", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20, help="0 means all conflict rows")
    parser.add_argument("--max-minute-requests", type=int, default=5)
    parser.add_argument("--max-daily-fallbacks", type=int, default=20,
                        help="Single-symbol rechecks for a truncated batch daily response")
    args = parser.parse_args()
    if args.limit < 0 or args.max_minute_requests < 0 or args.max_daily_fallbacks < 0:
        parser.error("limits must be nonnegative")
    root, staged, output = args.data_dir.resolve(), args.staged_file.resolve(), args.output.resolve()
    if not staged.exists() or output.exists():
        parser.error("staged file must exist and output report must not exist")
    if output.is_relative_to(root) and not output.is_relative_to(root / "import_reports"):
        parser.error("output inside production data must be under import_reports")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(root, staged, args.date, output, args.limit,
                 args.max_minute_requests, args.max_daily_fallbacks)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
