"""Recover canonical identities from an immutable migration quarantine.

The default mode is read-only.  ``--apply`` publishes only rows that passed
date-aware identity validation and an independent daily OHLCV evidence gate.
Rows without such evidence remain in the original quarantine and are listed
as ``pending_cross_source``; they are never represented by zero or neutral
values.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from uuid import uuid4

import polars as pl
import duckdb

from app.services.identity_migration import _asset_type_for_dataset, _build_lookups
from app.services.identity_recovery import (
    IdentityRecoveryDecision,
    dedupe_key,
    is_valid_ohlcv_row,
    resolve_quarantined_identity,
    row_trade_date,
)
from app.services.security_lifecycle import load_lifecycle_index, parse_day
from app.services.identity_schema import migration_lock
from app.services.minute_cross_source import verify_stock_day


PRICE_DATASETS = frozenset({"kline_daily", "kline_daily_enriched", "kline_minute"})
def _partition_date(path: Path) -> date | None:
    match = re.search(r"date=(\d{4}-\d{2}-\d{2})", str(path))
    return parse_day(match.group(1)) if match else None


def _target_from_quarantine(root: Path, quarantine_root: Path, path: Path) -> Path:
    relative = path.relative_to(quarantine_root)
    name = relative.name
    if name.endswith(".parquet.parquet"):
        name = name[:-len(".parquet")]
    target_relative = relative.with_name(name)
    return root / target_relative


def _evidence_for(root: Path, symbol: str, day: date) -> tuple[dict | None, list[dict]]:
    path = root / "minute_reference_evidence" / f"date={day.isoformat()}" / f"{symbol.split('.')[0]}.json"
    if not path.exists():
        return None, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None, []
    if str(payload.get("symbol") or "").upper() != symbol or str(payload.get("date") or "")[:10] != day.isoformat():
        return None, []
    snapshot = payload.get("daily_snapshot")
    if not isinstance(snapshot, dict):
        return None, []
    base = {"symbol": symbol, "date": day.isoformat(), **snapshot, "volume_unit": payload.get("daily_volume_unit") or "shares"}
    sources = payload.get("sources") or []
    external = []
    seen_sources: set[str] = set()
    for item in sources:
        provider = str(item.get("provider") or "").strip()
        if provider and provider.lower() not in seen_sources:
            seen_sources.add(provider.lower())
            external.append({"source": provider, **base})
    # Two labels copied from one upstream response are not independent
    # evidence.  Require distinct providers before a price row can advance.
    return base, external if len(seen_sources) >= 2 else []


def _canonical_row(row: dict, decision: IdentityRecoveryDecision) -> dict:
    cleaned = {
        key: value
        for key, value in row.items()
        if key not in {"original_symbol", "identity_reason", "identity_row"}
    }
    cleaned.update({
        "symbol": decision.symbol,
        "asset_type": decision.asset_type,
        "exchange": decision.exchange,
        "code": decision.code,
        "security_key": f"{decision.asset_type}|{decision.exchange}|{decision.code}",
    })
    return cleaned


def _minute_aggregate(frame: pl.DataFrame, symbol: str, day: date) -> dict | None:
    if frame.is_empty():
        return None
    values = frame.sort("datetime")
    needed = {"datetime", "open", "high", "low", "close", "volume", "amount"}
    if not needed.issubset(values.columns):
        return None
    rows = values.to_dicts()
    if any(not is_valid_ohlcv_row(item) for item in rows):
        return None
    first, last = rows[0], rows[-1]
    return {
        "symbol": symbol,
        "date": day.isoformat(),
        "open": first["open"],
        "high": max(float(item["high"]) for item in rows),
        "low": min(float(item["low"]) for item in rows),
        "close": last["close"],
        "volume": sum(float(item["volume"]) for item in rows),
        "amount": sum(float(item["amount"]) for item in rows),
        "volume_unit": "lots",
        "rows": len(rows),
    }


def _price_status(dataset: str, root: Path, symbol: str, day: date, rows: list[dict]) -> tuple[str, str]:
    if dataset in {"kline_daily", "kline_daily_enriched"}:
        if not rows or not is_valid_ohlcv_row(rows[0]):
            return "quarantine", "ohlcv_invalid"
        candidate = {"symbol": symbol, "date": day.isoformat(), **rows[0], "volume_unit": "lots"}
        evidence, external = _evidence_for(root, symbol, day)
        if evidence is None or len(external) < 2:
            return "pending_cross_source", "cross_source_evidence_missing"
        result = verify_stock_day(
            symbol=symbol,
            trade_date=day,
            local_daily=candidate,
            minute_aggregate=candidate,
            external_snapshots=external,
        )
        return ("approved", "independent_daily_agreement") if result["write_allowed"] else (result["status"], result["reason"])
    return "identity_verified", "identity_and_lifecycle_valid"


def scan_quarantine(
    root: Path,
    migration_id: str,
    *,
    allow_structural_stock_market: bool,
    build_frames: bool = False,
) -> tuple[list[dict], dict[str, pl.DataFrame]]:
    quarantine_root = root / "migration_backups" / migration_id / "quarantine"
    if not quarantine_root.exists():
        raise FileNotFoundError(quarantine_root)
    lookups = _build_lookups(root)
    lifecycle = load_lifecycle_index(root)
    lifecycle_code_index: dict[tuple[str, str], list[tuple[tuple[str, str, str], list[dict]]]] = defaultdict(list)
    for key, values in lifecycle.items():
        lifecycle_code_index[(key[0], key[2])].append((key, values))
    result_index: dict[tuple, dict] = {}
    accepted_frames: dict[str, pl.DataFrame] = {}
    decision_cache: dict[tuple[str, str, str], IdentityRecoveryDecision] = {}

    def record(item: dict, rows: int = 1) -> None:
        key = tuple(item.get(name) for name in ("path", "dataset", "symbol", "date", "status", "reason"))
        current = result_index.get(key)
        if current is None:
            current = dict(item)
            current["rows"] = 0
            result_index[key] = current
        current["rows"] = int(current.get("rows") or 0) + rows

    if not build_frames:
        # Read each dataset once through DuckDB.  The report is intentionally
        # stock-day level; raw rows remain immutable in the migration backup.
        dataset_dirs = [item for item in quarantine_root.iterdir() if item.is_dir()]
        for dataset_dir in sorted(dataset_dirs):
            dataset = dataset_dir.name
            asset_type = _asset_type_for_dataset(dataset)
            if not asset_type:
                record({"path": str(dataset_dir.relative_to(root)), "dataset": dataset, "status": "quarantine", "reason": "asset_type_unknown"})
                continue
            pattern = str(dataset_dir / "**" / "*.parquet")
            price = dataset in PRICE_DATASETS
            try:
                schema_rows = duckdb.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)", [pattern]
                ).fetchall()
                columns = {str(row[0]) for row in schema_rows}
            except Exception as exc:  # noqa: BLE001
                record({"path": str(dataset_dir.relative_to(root)), "dataset": dataset, "status": "quarantine", "reason": "parquet_schema_error", "error": type(exc).__name__})
                continue

            def ident(name: str) -> str:
                return f'"{name}"'

            date_parts = [
                f"TRY_CAST(CAST({ident(name)} AS VARCHAR) AS DATE)"
                for name in ("date", "trade_date", "datetime", "timestamp", "ann_date", "end_date", "period_end", "announce_date")
                if name in columns
            ]
            date_expr = "COALESCE(" + ",".join(date_parts) + ")" if date_parts else "NULL::DATE"
            order_name = "datetime" if "datetime" in columns else next((name for name in ("date", "trade_date", "ann_date", "end_date", "period_end", "announce_date") if name in columns), None)
            order_expr = ident(order_name) if order_name else "1"
            value_expr = {
                name: (f"TRY_CAST({ident(name)} AS DOUBLE)" if name in columns else "NULL::DOUBLE")
                for name in ("open", "high", "low", "close", "volume", "amount")
            }
            unique_expr = f"COUNT(DISTINCT {order_expr})" if dataset == "kline_minute" else "COUNT(*)"
            valid_fields_present = all(name in columns for name in ("open", "high", "low", "close", "volume", "amount"))
            invalid_expr = (
                f"SUM(CASE WHEN {value_expr['open']} > 0 AND {value_expr['high']} > 0 AND {value_expr['low']} > 0 AND {value_expr['close']} > 0 "
                f"AND {value_expr['high']} >= GREATEST({value_expr['open']}, {value_expr['low']}, {value_expr['close']}) "
                f"AND {value_expr['low']} <= LEAST({value_expr['open']}, {value_expr['high']}, {value_expr['close']}) "
                f"AND {value_expr['volume']} >= 0 AND {value_expr['amount']} >= 0 THEN 0 ELSE 1 END)"
                if valid_fields_present else "0"
            )
            query = f"""
                SELECT
                    CAST(original_symbol AS VARCHAR) AS original_symbol,
                    {date_expr} AS trade_day,
                    COUNT(*) AS rows,
                    {unique_expr} AS unique_rows,
                    arg_min({value_expr['open']}, {order_expr}) AS first_open,
                    MAX({value_expr['high']}) AS high,
                    MIN({value_expr['low']}) AS low,
                    arg_max({value_expr['close']}, {order_expr}) AS last_close,
                    SUM({value_expr['volume']}) AS volume,
                    SUM({value_expr['amount']}) AS amount,
                    {invalid_expr} AS invalid_rows
                FROM read_parquet(?, union_by_name=true)
                WHERE original_symbol IS NOT NULL
                GROUP BY 1, 2
            """
            try:
                groups = duckdb.execute(query, [pattern]).fetchall()
            except Exception as exc:  # noqa: BLE001
                record({"path": str(dataset_dir.relative_to(root)), "dataset": dataset, "status": "quarantine", "reason": "parquet_grouping_error", "error": type(exc).__name__})
                continue
            for raw_symbol, raw_day, row_count, unique_rows, first_open, high, low, last_close, volume, amount, invalid_rows in groups:
                day = parse_day(raw_day)
                raw_text = str(raw_symbol or "")
                cache_key = (asset_type, raw_text, day.isoformat() if day else "")
                decision = decision_cache.get(cache_key)
                if decision is None:
                    decision = resolve_quarantined_identity(
                        raw_text,
                        asset_type=asset_type,
                        trade_date=day,
                        lookup=lookups[asset_type],
                        lifecycle_index=lifecycle,
                        lifecycle_code_index=lifecycle_code_index,
                        data_dir=root,
                        allow_structural_stock_market=allow_structural_stock_market,
                    )
                    decision_cache[cache_key] = decision
                base = {"path": str(dataset_dir.relative_to(root)), "dataset": dataset, "symbol": decision.symbol, "date": str(day) if day else None}
                count = int(row_count or 0)
                if not decision.ok or day is None:
                    record({**base, "status": "quarantine", "reason": decision.reason}, rows=count)
                    continue
                if not price:
                    record({**base, "status": "identity_verified", "reason": "identity_and_lifecycle_valid", "basis": decision.basis}, rows=count)
                    continue
                candidate = {
                    "symbol": decision.symbol,
                    "date": day.isoformat(),
                    "open": first_open,
                    "high": high,
                    "low": low,
                    "close": last_close,
                    "volume": volume,
                    "amount": amount,
                    "volume_unit": "lots",
                }
                if int(invalid_rows or 0) > 0 or (dataset == "kline_minute" and int(unique_rows or 0) != count):
                    record({**base, "status": "quarantine", "reason": "minute_ohlcv_invalid" if dataset == "kline_minute" else "ohlcv_invalid"}, rows=count)
                    continue
                evidence, external = _evidence_for(root, decision.symbol, day)
                if evidence is None or len(external) < 2:
                    record({**base, "status": "pending_cross_source", "reason": "cross_source_evidence_missing"}, rows=count)
                    continue
                if dataset == "kline_minute":
                    candidate["rows"] = count
                result = verify_stock_day(
                    symbol=decision.symbol,
                    trade_date=day,
                    local_daily=candidate,
                    minute_aggregate=candidate,
                    external_snapshots=external,
                )
                status = "approved" if result["write_allowed"] else result["status"]
                reason = "independent_daily_agreement" if status == "approved" else result["reason"]
                record({**base, "status": status, "reason": reason, "basis": decision.basis}, rows=count)
        return list(result_index.values()), accepted_frames

    # Minute quarantine is the largest dataset.  Aggregate it in DuckDB so
    # the identity pass operates at stock-day granularity instead of decoding
    # every bar into Python dictionaries.
    if not build_frames:
        minute_root = quarantine_root / "kline_minute"
        minute_files = str(minute_root / "**" / "*.parquet")
        if minute_root.exists():
            minute_groups = duckdb.execute(
                """
                SELECT
                    CAST(original_symbol AS VARCHAR) AS original_symbol,
                    CAST(datetime AS DATE) AS trade_day,
                    COUNT(*) AS rows,
                    COUNT(DISTINCT datetime) AS unique_rows,
                    arg_min(open, datetime) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    arg_max(close, datetime) AS close,
                    SUM(volume) AS volume,
                    SUM(amount) AS amount,
                    SUM(CASE WHEN open > 0 AND high > 0 AND low > 0 AND close > 0
                              AND high >= GREATEST(open, low, close)
                              AND low <= LEAST(open, high, close)
                              AND volume >= 0 AND amount >= 0
                             THEN 0 ELSE 1 END) AS invalid_rows
                FROM read_parquet(?, union_by_name=true)
                GROUP BY 1, 2
                """,
                [minute_files],
            ).fetchall()
            for raw_symbol, raw_day, row_count, unique_rows, open_value, high_value, low_value, close_value, volume, amount, invalid_rows in minute_groups:
                day = parse_day(raw_day)
                raw_text = str(raw_symbol or "")
                cache_key = ("stock", raw_text, day.isoformat() if day else "")
                decision = decision_cache.get(cache_key)
                if decision is None:
                    decision = resolve_quarantined_identity(
                        raw_text,
                        asset_type="stock",
                        trade_date=day,
                        lookup=lookups["stock"],
                        lifecycle_index=lifecycle,
                        lifecycle_code_index=lifecycle_code_index,
                        data_dir=root,
                        allow_structural_stock_market=allow_structural_stock_market,
                    )
                    decision_cache[cache_key] = decision
                base = {"path": str(minute_root.relative_to(root)), "dataset": "kline_minute", "symbol": decision.symbol, "date": str(day) if day else None}
                count = int(row_count or 0)
                if not decision.ok or day is None:
                    record({**base, "status": "quarantine", "reason": decision.reason}, rows=count)
                    continue
                if int(invalid_rows or 0) > 0 or int(unique_rows or 0) != count:
                    record({**base, "status": "quarantine", "reason": "minute_ohlcv_invalid"}, rows=count)
                    continue
                evidence, external = _evidence_for(root, decision.symbol, day)
                if evidence is None or len(external) < 2:
                    record({**base, "status": "pending_cross_source", "reason": "cross_source_evidence_missing"}, rows=count)
                    continue
                aggregate = {"symbol": decision.symbol, "date": day.isoformat(), "open": open_value, "high": high_value, "low": low_value, "close": close_value, "volume": volume, "amount": amount, "volume_unit": "lots", "rows": count}
                result = verify_stock_day(symbol=decision.symbol, trade_date=day, local_daily=evidence, minute_aggregate=aggregate, external_snapshots=external)
                status = "approved" if result["write_allowed"] else result["status"]
                reason = "independent_daily_agreement" if status == "approved" else result["reason"]
                record({**base, "status": status, "reason": reason}, rows=count)
    for path in sorted(quarantine_root.rglob("*.parquet")):
        try:
            frame = pl.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            record({"path": str(path.relative_to(root)), "status": "quarantine", "reason": "parquet_read_error", "error": type(exc).__name__})
            continue
        if "original_symbol" not in frame.columns:
            continue
        dataset = path.relative_to(quarantine_root).parts[0]
        if dataset == "kline_minute" and not build_frames:
            continue
        asset_type = _asset_type_for_dataset(dataset)
        if not asset_type:
            record({"path": str(path.relative_to(root)), "status": "quarantine", "reason": "asset_type_unknown"})
            continue
        if build_frames and dataset in PRICE_DATASETS:
            # The dry-run already classified every price row.  Applying this
            # command is intentionally identity-only; price datasets require
            # a later external-evidence run before they may be staged.
            continue
        partition_day = _partition_date(path)
        accepted_rows: list[dict] = []

        if build_frames and dataset not in PRICE_DATASETS:
            date_column = next((name for name in ("date", "trade_date", "datetime", "timestamp", "ann_date", "end_date", "period_end", "announce_date") if name in frame.columns), None)
            if date_column:
                if date_column in {"datetime", "timestamp"}:
                    date_expr = pl.col(date_column).cast(pl.Datetime, strict=False).dt.date()
                else:
                    date_expr = pl.col(date_column).cast(pl.String).str.slice(0, 10).str.strptime(pl.Date, strict=False)
            else:
                date_expr = pl.lit(partition_day, dtype=pl.Date)
            working = frame.with_columns(date_expr.alias("__recovery_date"))
            groups = working.group_by(["original_symbol", "__recovery_date"]).len(name="__rows")
            mapping_rows: list[dict] = []
            for group in groups.to_dicts():
                raw_symbol = str(group.get("original_symbol") or "")
                day = parse_day(group.get("__recovery_date")) or partition_day
                cache_key = (asset_type, raw_symbol, day.isoformat() if day else "")
                decision = decision_cache.get(cache_key)
                if decision is None:
                    decision = resolve_quarantined_identity(
                        raw_symbol,
                        asset_type=asset_type,
                        trade_date=day,
                        lookup=lookups[asset_type],
                        lifecycle_index=lifecycle,
                        lifecycle_code_index=lifecycle_code_index,
                        data_dir=root,
                        allow_structural_stock_market=allow_structural_stock_market,
                    )
                    decision_cache[cache_key] = decision
                count = int(group.get("__rows") or 0)
                base = {"path": str(path.relative_to(root)), "dataset": dataset, "symbol": decision.symbol, "date": str(day) if day else None}
                if not decision.ok or day is None:
                    record({**base, "status": "quarantine", "reason": decision.reason}, rows=count)
                    continue
                record({**base, "status": "identity_verified", "reason": "identity_and_lifecycle_valid", "basis": decision.basis}, rows=count)
                mapping_rows.append({
                    "original_symbol": raw_symbol,
                    "__recovery_date": day,
                    "__canonical_symbol": decision.symbol,
                    "__canonical_code": decision.code,
                    "__canonical_exchange": decision.exchange,
                    "__canonical_asset_type": decision.asset_type,
                    "__canonical_security_key": f"{decision.asset_type}|{decision.exchange}|{decision.code}",
                })
            if mapping_rows:
                mapping = pl.DataFrame(mapping_rows)
                base = working.drop([column for column in ("symbol", "asset_type", "exchange", "code", "security_key", "identity_reason", "identity_row") if column in working.columns])
                accepted = base.join(mapping, on=["original_symbol", "__recovery_date"], how="inner")
                accepted = accepted.drop(["original_symbol", "__recovery_date"]).rename({
                    "__canonical_symbol": "symbol",
                    "__canonical_code": "code",
                    "__canonical_exchange": "exchange",
                    "__canonical_asset_type": "asset_type",
                    "__canonical_security_key": "security_key",
                })
                accepted_frames[str(path)] = accepted
            continue

        if not build_frames:
            date_column = next((name for name in ("date", "trade_date", "datetime", "timestamp", "ann_date", "end_date") if name in frame.columns), None)
            if date_column:
                if date_column in {"datetime", "timestamp"}:
                    date_expr = pl.col(date_column).cast(pl.Datetime, strict=False).dt.date()
                else:
                    date_expr = pl.col(date_column).cast(pl.String).str.slice(0, 10).str.strptime(pl.Date, strict=False)
            else:
                date_expr = pl.lit(partition_day, dtype=pl.Date)
            working = frame.with_columns(date_expr.alias("__recovery_date"))
            unique_groups = (
                working.group_by(["original_symbol", "__recovery_date"])
                .len(name="__rows")
                .sort(["original_symbol", "__recovery_date"])
            )
            for group in unique_groups.to_dicts():
                raw_symbol = str(group.get("original_symbol") or "")
                day = parse_day(group.get("__recovery_date")) or partition_day
                count = int(group.get("__rows") or 0)
                cache_key = (asset_type, raw_symbol, day.isoformat() if day else "")
                decision = decision_cache.get(cache_key)
                if decision is None:
                    decision = resolve_quarantined_identity(
                        raw_symbol,
                        asset_type=asset_type,
                        trade_date=day,
                        lookup=lookups[asset_type],
                        lifecycle_index=lifecycle,
                        lifecycle_code_index=lifecycle_code_index,
                        data_dir=root,
                        allow_structural_stock_market=allow_structural_stock_market,
                    )
                    decision_cache[cache_key] = decision
                base = {"path": str(path.relative_to(root)), "dataset": dataset, "symbol": decision.symbol, "date": str(day) if day else None}
                if not decision.ok or day is None:
                    record({**base, "status": "quarantine", "reason": decision.reason}, rows=count)
                    continue
                if dataset == "kline_minute":
                    evidence, external = _evidence_for(root, decision.symbol, day)
                    if evidence is None or len(external) < 2:
                        record({**base, "status": "pending_cross_source", "reason": "cross_source_evidence_missing"}, rows=count)
                        continue
                    subset = working.filter(
                        (pl.col("original_symbol") == raw_symbol) & (pl.col("__recovery_date") == day)
                    ).drop("__recovery_date")
                    canonical_rows = [_canonical_row(row, decision) for row in subset.to_dicts()]
                    aggregate = _minute_aggregate(pl.DataFrame(canonical_rows), decision.symbol, day)
                    if aggregate is None:
                        record({**base, "status": "quarantine", "reason": "minute_ohlcv_invalid"}, rows=count)
                    else:
                        result = verify_stock_day(symbol=decision.symbol, trade_date=day, local_daily=evidence, minute_aggregate=aggregate, external_snapshots=external)
                        status = "approved" if result["write_allowed"] else result["status"]
                        reason = "independent_daily_agreement" if status == "approved" else result["reason"]
                        record({**base, "status": status, "reason": reason}, rows=count)
                    continue
                representative = working.filter(
                    (pl.col("original_symbol") == raw_symbol) & (pl.col("__recovery_date") == day)
                ).head(1).drop("__recovery_date").to_dicts()
                status, reason = _price_status(dataset, root, decision.symbol, day, representative)
                record({**base, "status": status, "reason": reason, "basis": decision.basis}, rows=count)
            continue

        identity_rows: list[tuple[dict, IdentityRecoveryDecision, date]] = []
        for row in frame.to_dicts():
            day = row_trade_date(row, partition_day)
            raw_symbol = str(row.get("original_symbol") or row.get("symbol") or "")
            cache_key = (asset_type, raw_symbol, day.isoformat() if day else "")
            decision = decision_cache.get(cache_key)
            if decision is None:
                decision = resolve_quarantined_identity(
                    raw_symbol,
                    asset_type=asset_type,
                    trade_date=day,
                    lookup=lookups[asset_type],
                    lifecycle_index=lifecycle,
                    lifecycle_code_index=lifecycle_code_index,
                    data_dir=root,
                    allow_structural_stock_market=allow_structural_stock_market,
                )
                decision_cache[cache_key] = decision
            if not decision.ok or day is None:
                record({"path": str(path.relative_to(root)), "dataset": dataset, "status": "quarantine", "reason": decision.reason, "symbol": decision.symbol, "date": str(day) if day else None})
                continue
            identity_rows.append((row, decision, day))

        if dataset == "kline_minute" and identity_rows:
            grouped: dict[tuple[str, date], list[tuple[dict, IdentityRecoveryDecision]]] = defaultdict(list)
            for row, decision, day in identity_rows:
                grouped[(decision.symbol, day)].append((row, decision))
            for (symbol, day), items in grouped.items():
                candidate_frame = pl.DataFrame([_canonical_row(row, decision) for row, decision in items])
                aggregate = _minute_aggregate(candidate_frame, symbol, day)
                evidence, external = _evidence_for(root, symbol, day)
                if aggregate is None:
                    status, reason = "quarantine", "minute_ohlcv_invalid"
                elif evidence is None or len(external) < 2:
                    status, reason = "pending_cross_source", "cross_source_evidence_missing"
                else:
                    result = verify_stock_day(
                        symbol=symbol, trade_date=day, local_daily=evidence,
                        minute_aggregate=aggregate, external_snapshots=external,
                    )
                    status = "approved" if result["write_allowed"] else result["status"]
                    reason = "independent_daily_agreement" if status == "approved" else result["reason"]
                record({"path": str(path.relative_to(root)), "dataset": dataset, "symbol": symbol, "date": day.isoformat(), "status": status, "reason": reason}, rows=len(items))
                if status == "approved":
                    accepted_rows.extend(_canonical_row(row, decision) for row, decision in items)
        else:
            for row, decision, day in identity_rows:
                status, reason = _price_status(dataset, root, decision.symbol, day, [row])
                record({"path": str(path.relative_to(root)), "dataset": dataset, "symbol": decision.symbol, "date": day.isoformat(), "status": status, "reason": reason, "basis": decision.basis})
                if status in {"approved", "identity_verified"}:
                    accepted_rows.append(_canonical_row(row, decision))
        if accepted_rows:
            accepted_frames[str(path)] = pl.DataFrame(accepted_rows)
    return list(result_index.values()), accepted_frames


def apply_accepted(
    root: Path,
    recovery_id: str,
    migration_id: str,
    accepted_frames: dict[str, pl.DataFrame],
) -> int:
    recovery_root = root / "identity_recovery" / recovery_id
    backup_root = recovery_root / "backups"
    applied = 0
    with migration_lock(root):
        for quarantine_path, incoming in accepted_frames.items():
            quarantine_root = root / "migration_backups" / migration_id / "quarantine"
            target = _target_from_quarantine(root, quarantine_root, Path(quarantine_path))
            if not target.parent.exists():
                continue
            backup = backup_root / target.relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.copy2(target, backup)
                existing = pl.read_parquet(target)
                merged = pl.concat([existing, incoming], how="diagonal_relaxed")
                keys = dedupe_key(target.relative_to(root).parts[0], merged)
                if keys:
                    merged = merged.unique(keys, keep="first", maintain_order=True)
            else:
                merged = incoming
            temporary = target.with_name(f".{target.name}.{recovery_id}.tmp")
            merged.write_parquet(temporary)
            temporary.replace(target)
            applied += incoming.height
    return applied


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover identities from canonical migration quarantine")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--migration-id", required=True)
    parser.add_argument("--allow-structural-stock-market", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.data_dir.resolve()
    recovery_id = f"recovery-{date.today().strftime('%Y%m%d')}-{uuid4().hex[:8]}"
    results, accepted = scan_quarantine(
        root,
        args.migration_id,
        allow_structural_stock_market=args.allow_structural_stock_market,
        build_frames=args.apply,
    )
    report_root = root / "identity_recovery" / recovery_id
    report_root.mkdir(parents=True, exist_ok=False)
    applied = apply_accepted(root, recovery_id, args.migration_id, accepted) if args.apply and accepted else 0
    counts = Counter(str(item.get("status") or "unknown") for item in results)
    reasons = Counter(str(item.get("reason") or "unknown") for item in results)
    row_counts = Counter()
    reason_row_counts = Counter()
    for item in results:
        rows = int(item.get("rows") or 0)
        row_counts[str(item.get("status") or "unknown")] += rows
        reason_row_counts[str(item.get("reason") or "unknown")] += rows
    payload = {
        "recovery_id": recovery_id,
        "migration_id": args.migration_id,
        "structural_stock_market_enabled": args.allow_structural_stock_market,
        "apply": bool(args.apply),
        "result_count": len(results),
        "accepted_rows_staged": sum(frame.height for frame in accepted.values()),
        "applied_rows": applied,
        "status_counts": dict(counts),
        "reason_counts": dict(reasons),
        "status_row_counts": dict(row_counts),
        "reason_row_counts": dict(reason_row_counts),
        "results": results,
    }
    (report_root / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_root / "summary.json").write_text(json.dumps({key: payload[key] for key in payload if key != "results"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "results"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
