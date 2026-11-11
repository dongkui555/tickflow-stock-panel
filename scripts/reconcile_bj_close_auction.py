"""Classify BSE minute rows whose only price mismatch is the closing auction.

This command never creates a synthetic 15:00 auction bar. It only records a
separate quality state when the 240 continuous-session bars have matching
open/high/low, matching volume, a non-decreasing daily turnover total, and a
single verified BSE minute identity.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import polars as pl


REASON = (
    "北交所240根连续竞价量价通过，日K收盘价与分钟末端存在收盘竞价口径差异；"
    "盘后成交额未纳入分钟"
)


def _bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def classify_row(row: dict, *, ohl_match: bool) -> str:
    """Return a conservative status for an audit row."""
    try:
        bars = int(row.get("bars") or 0)
        after_hours = int(row.get("after_hours_rows") or 0)
        alias_count = int(row.get("alias_count") or 0)
        daily_amount = float(row.get("amount"))
        minute_amount = float(row.get("minute_amount"))
    except (TypeError, ValueError):
        return "complete_conflict"
    if (
        str(row.get("market") or "").upper() == "BJ"
        and str(row.get("selected_symbol") or "").upper().endswith(".BJ")
        and bars == 240
        and after_hours == 0
        and alias_count == 1
        and not _bool(row.get("price_match"))
        and _bool(row.get("volume_match"))
        and ohl_match
        and daily_amount >= minute_amount
    ):
        return "regular_session_close_auction"
    return "complete_conflict"


def _price_tolerance(close: float) -> float:
    return max(0.011, abs(close) * 1e-6)


def _ohl_matches(data_dir: Path, rows: list[dict]) -> dict[tuple[str, str], bool]:
    """Compare all requested rows while reading each date partition once."""
    by_day: dict[str, list[dict]] = {}
    for row in rows:
        by_day.setdefault(str(row.get("date") or "")[:10], []).append(row)
    result: dict[tuple[str, str], bool] = {}
    for day, day_rows in by_day.items():
        minute_path = data_dir / "kline_minute" / f"date={day}" / "part.parquet"
        daily_path = data_dir / "kline_daily" / f"date={day}" / "part.parquet"
        symbols = [str(row.get("selected_symbol") or "").upper() for row in day_rows]
        codes = [str(row.get("code") or "").zfill(6) for row in day_rows]
        if not minute_path.exists() or not daily_path.exists():
            continue
        try:
            minute = pl.read_parquet(
                minute_path,
                columns=["symbol", "datetime", "open", "high", "low", "volume", "amount"],
            ).with_columns(
                pl.col("symbol").cast(pl.String).str.to_uppercase().alias("_symbol"),
            ).filter(pl.col("_symbol").is_in(symbols))
            daily = (
                pl.read_parquet(daily_path, columns=["symbol", "open", "high", "low", "close"])
                .with_columns(
                    pl.col("symbol").cast(pl.String).str.to_uppercase().alias("_symbol"),
                    pl.col("symbol").cast(pl.String).str.extract(r"(\d{6})", 1).alias("_code"),
                )
                .filter(pl.col("_code").is_in(codes))
            )
            if minute.is_empty() or daily.is_empty():
                continue
            clock = (
                pl.col("datetime").dt.hour().cast(pl.Int32) * 60
                + pl.col("datetime").dt.minute().cast(pl.Int32)
            )
            session = clock.is_between(571, 690) | clock.is_between(781, 900)
            traded = (pl.col("volume") > 0) | (pl.col("amount") > 0)
            minute_agg = minute.sort(["_symbol", "datetime"]).group_by("_symbol").agg(
                pl.col("datetime").filter(session).n_unique().alias("bars"),
                pl.col("open").filter(session & traded).first().alias("open"),
                pl.col("high").filter(session & traded).max().alias("high"),
                pl.col("low").filter(session & traded).min().alias("low"),
            )
            minute_map = {row["_symbol"]: row for row in minute_agg.to_dicts()}
            daily_by_symbol = {
                row["_symbol"]: row for row in daily.to_dicts()
            }
            daily_by_code: dict[str, list[dict]] = {}
            for row in daily.to_dicts():
                daily_by_code.setdefault(row["_code"], []).append(row)
            for row in day_rows:
                symbol = str(row.get("selected_symbol") or "").upper()
                code = str(row.get("code") or "").zfill(6)
                aggregate = minute_map.get(symbol)
                references = daily_by_symbol.get(symbol)
                if references is None and len(daily_by_code.get(code, [])) == 1:
                    references = daily_by_code[code][0]
                if aggregate is None or references is None:
                    continue
                if int(aggregate["bars"] or 0) != 240:
                    continue
                if any(
                    aggregate[field] is None or references[field] is None
                    for field in ("open", "high", "low")
                ):
                    continue
                tolerance = _price_tolerance(float(references["close"]))
                result[(day, code)] = all(
                    abs(float(aggregate[field]) - float(references[field])) <= tolerance
                    for field in ("open", "high", "low")
                )
        except (OSError, pl.exceptions.PolarsError, TypeError, ValueError):
            continue
    return result


def _backup(path: Path, backup_dir: Path) -> str | None:
    if not path.exists():
        return None
    target = backup_dir / path.name
    shutil.copy2(path, target)
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_dir.resolve()
    audit = args.audit.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root / "import_reports") or output == root / "import_reports":
        raise ValueError("output must be a new directory inside data/import_reports")
    output.mkdir(parents=True, exist_ok=False)

    with audit.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    conflict_rows = [row for row in rows if row.get("status") == "complete_conflict"]
    ohl_matches = _ohl_matches(root, conflict_rows)
    candidates = []
    skipped = Counter()
    for row in conflict_rows:
        key = (str(row.get("date") or "")[:10], str(row.get("code") or "").zfill(6))
        ohl = ohl_matches.get(key, False)
        status = classify_row(row, ohl_match=ohl)
        if status == "regular_session_close_auction":
            candidates.append(row)
        elif row.get("bars") == "240" and row.get("price_match") == "false":
            skipped["price_path_or_volume_conflict"] += 1
        elif row.get("bars") == "240":
            skipped["other_complete_conflict"] += 1

    records = [
        {
            "symbol": str(row["code"]).zfill(6) + ".BJ",
            "date": row["date"],
            "status": "regular_session_close_auction",
            "expected_minutes": 240,
            "actual_minutes": int(row["bars"]),
            "missing_minutes": 0,
            "reason": REASON,
            "source": "local",
            "daily_turnover_complete": False,
            "minute_scope": "regular_session_close_auction",
        }
        for row in candidates
    ]

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_dir = root / "import_backups" / f"bj_close_auction_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups = []
    for path in (
        root / "user_data" / "minute_coverage_status.json",
        root / "user_data" / "minute_gap_queue.sqlite",
    ):
        saved = _backup(path, backup_dir)
        if saved:
            backups.append(saved)

    from app.services import minute_status
    from app.services.minute_gap_queue import _connect, _ensure_schema, _upsert

    minute_status.upsert(root, records)
    now = datetime.now().isoformat(timespec="seconds")
    queue_updated = 0
    with _connect(root) as conn:
        _ensure_schema(conn)
        for record in records:
            old = conn.execute(
                "SELECT attempts, created_at FROM minute_gap_queue WHERE symbol=? AND trade_date=?",
                (record["symbol"].split(".")[0], record["date"]),
            ).fetchone()
            _upsert(conn, {
                **record,
                "trade_date": record["date"],
                "source_chain": "qmt,promax,tdx,stockdb,offline",
                "attempts": int(old["attempts"] if old else 0),
                "created_at": old["created_at"] if old else now,
                "next_retry_at": None,
                "last_error": None,
            }, now)
            queue_updated += 1

    report = {
        "status": "applied",
        "audit": str(audit),
        "candidate_stock_days": len(candidates),
        "coverage_status_updated": len(records),
        "queue_updated": queue_updated,
        "status_value": "regular_session_close_auction",
        "reason": REASON,
        "daily_total_preserved": True,
        "minute_data_written": False,
        "backups": backups,
        "remaining_conflicts": len(rows) - len(candidates),
        "skipped": dict(skipped),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
