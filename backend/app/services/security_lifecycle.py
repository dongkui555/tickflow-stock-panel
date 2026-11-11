"""Date-aware security lifecycle validation for market-data backfills.

Market-data gaps are not all missing-data problems.  A symbol can be before
its listing date, after delisting, temporarily suspended, or have changed
identity after a corporate action.  This module keeps those cases explicit so
minute/daily repair jobs do not retry impossible requests or merge a successor
security into its predecessor.

The validator is deliberately conservative:

* a missing lifecycle record is ``identity_history_unknown`` rather than an
  implicit SZ/BJ assignment;
* a current ``P`` status is metadata only unless an effective date is present;
* a name containing ``退`` or ``重组`` is never treated as proof of delisting
  or a code change;
* code aliases are accepted only when an explicit effective interval or
  ``successor_symbol`` record is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping

import polars as pl

from app.services.security_identity import (
    IdentityError,
    compact_code,
    exchange_for_code,
    explicit_exchange,
    normalize_exchange,
    parse_identity,
)


LIFECYCLE_TERMINAL_REASONS = frozenset({
    "not_listed_yet",
    "delisted_on_date",
    "identity_history_unknown",
    "market_identity_conflict",
    "lifecycle_date_missing",
})


def parse_day(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "nat"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        except ValueError:
            return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _first(row: dict, *names: str):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() not in {"", "None", "nan", "NaT"}:
            return value
    return None


def _status(value: object) -> str:
    text = str(value or "").strip().upper()
    return {
        "L": "listed",
        "D": "delisted",
        "P": "suspended",
        "LISTED": "listed",
        "DELISTED": "delisted",
        "SUSPENDED": "suspended",
        "PAUSED": "suspended",
    }.get(text, text.lower() if text else "unknown")


def _normalise_row(row: dict, source: str) -> dict | None:
    raw_symbol = _first(row, "symbol", "ts_code", "security", "code")
    code = compact_code(raw_symbol or _first(row, "code", "ts_code"))
    if not code:
        return None
    exchange = (
        normalize_exchange(_first(row, "exchange", "exchange_code"))
        or explicit_exchange(raw_symbol)
        or exchange_for_code(code, "stock")
    )
    if not exchange:
        return None
    asset_type = str(_first(row, "asset_type", "type") or "stock").strip().lower()
    if asset_type not in {"stock", "equity"}:
        asset_type = "stock"
    listed = parse_day(_first(row, "listing_date", "list_date", "effective_from", "start_date"))
    delisted = parse_day(_first(row, "delist_date", "delisting_date"))
    effective_from = parse_day(_first(row, "effective_from", "start_date", "listing_date", "list_date"))
    effective_to = parse_day(_first(row, "effective_to", "end_date", "delist_date", "delisting_date"))
    status = _status(_first(row, "list_status", "status", "lifecycle_status"))
    return {
        "symbol": f"{code}.{exchange}",
        "code": code,
        "exchange": exchange,
        "asset_type": asset_type,
        "name": str(_first(row, "name", "security_name") or ""),
        "status": status,
        "listing_date": listed,
        "delist_date": delisted,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "successor_symbol": str(_first(row, "successor_symbol", "superseded_by", "next_symbol") or ""),
        "alias_group": str(_first(row, "alias_group", "code_group") or ""),
        "event_type": str(_first(row, "event_type", "change_reason", "corporate_action") or ""),
        "source": source,
    }


def _read_rows(path: Path, source: str) -> list[dict]:
    if not path.exists():
        return []
    try:
        rows = []
        for raw in pl.read_parquet(path).to_dicts():
            row = _normalise_row(raw, source)
            if row:
                rows.append(row)
        return rows
    except Exception:
        return []


def refresh_observed_security_dimension(data_dir: str | Path) -> int:
    """Build a conservative identity lower-bound from canonical daily data.

    This is not an official listing/delisting table.  It only records that a
    fully canonical ``symbol`` was observed on or after its first local daily
    bar.  It is useful for historical B-share/BJ rows missing from provider
    lifecycle snapshots, while dates before the observation remain
    unresolved.  No delist date is inferred from the last local bar.
    """
    root = Path(data_dir)
    files = list((root / "kline_daily").glob("**/*.parquet"))
    if not files:
        return 0
    try:
        frame = (
            pl.scan_parquet([str(path) for path in files], extra_columns="ignore", missing_columns="insert")
            .select(["symbol", "date", "asset_type", "exchange", "code"])
            .filter(
                pl.col("asset_type").fill_null("stock").cast(pl.String).str.to_lowercase().is_in(["stock", "equity"])
                & pl.col("symbol").cast(pl.String).str.contains(r"^\d{6}\.(SH|SZ|BJ)$")
            )
            .with_columns([
                pl.col("symbol").cast(pl.String).str.to_uppercase(),
                pl.col("date").cast(pl.Date, strict=False),
                pl.col("exchange").cast(pl.String).str.to_uppercase(),
                pl.col("code").cast(pl.String).str.zfill(6),
            ])
            .drop_nulls(["symbol", "date"])
            .group_by(["symbol", "exchange", "code"])
            .agg(pl.col("date").min().alias("listing_date"))
            .collect()
        )
    except Exception:
        return 0
    if frame.is_empty():
        return 0
    frame = frame.with_columns([
        pl.lit("stock").alias("asset_type"),
        pl.lit("listed").alias("list_status"),
        pl.col("listing_date").alias("effective_from"),
        pl.lit("active_canonical_observation").alias("source"),
        pl.lit("observed_lower_bound").alias("event_type"),
        pl.lit(date.today()).alias("as_of"),
    ])
    out = root / "instruments_history" / "observed_security_dimension.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    frame.write_parquet(tmp)
    tmp.replace(out)
    return frame.height


def load_lifecycle_index(data_dir: str | Path) -> dict[tuple[str, str, str], list[dict]]:
    """Load current, historical and provider lifecycle dimensions.

    ``instruments_lifecycle`` is preferred for status/effective-date data but
    current and historical dimensions remain valid identity fallbacks.  Rows
    are retained rather than collapsed because one code may have multiple
    historical intervals.
    """
    root = Path(data_dir)
    paths = [
        (root / "instruments_history" / "code_changes.parquet", "code_change_dimension"),
        (root / "instruments_lifecycle" / "instruments_lifecycle.parquet", "provider_lifecycle"),
        (root / "instruments_history" / "instruments_history.parquet", "historical_dimension"),
        (root / "instruments_history" / "observed_security_dimension.parquet", "canonical_observation_dimension"),
        (root / "instruments" / "instruments.parquet", "current_dimension"),
    ]
    index: dict[tuple[str, str, str], list[dict]] = {}
    explicit_alias_keys: set[tuple[str, str, str]] = set()
    for path, source in paths:
        for row in _read_rows(path, source):
            key = (row["asset_type"], row["exchange"], row["code"])
            if source == "code_change_dimension" and row.get("successor_symbol"):
                explicit_alias_keys.add(key)
            if source != "code_change_dimension" and key in explicit_alias_keys:
                continue
            # Current dimensions are explicitly live even though older files
            # predate the list_status column.
            if source == "current_dimension" and row["status"] == "unknown":
                row["status"] = "listed"
            if row not in index.setdefault(key, []):
                index[key].append(row)
    return index


@dataclass(frozen=True)
class LifecycleDecision:
    ok: bool
    reason: str
    symbol: str
    code: str
    exchange: str
    asset_type: str
    status: str = "unknown"
    listing_date: date | None = None
    delist_date: date | None = None
    successor_symbol: str = ""
    event_type: str = ""
    source: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "symbol": self.symbol,
            "code": self.code,
            "exchange": self.exchange,
            "asset_type": self.asset_type,
            "status": self.status,
            "listing_date": self.listing_date.isoformat() if self.listing_date else None,
            "delist_date": self.delist_date.isoformat() if self.delist_date else None,
            "successor_symbol": self.successor_symbol,
            "event_type": self.event_type,
            "source": self.source,
        }


def validate_security_lifecycle(
    symbol: object,
    when: date | datetime | str,
    data_dir: str | Path,
    *,
    asset_type: str = "stock",
    expected_exchange: str | None = None,
    index: dict[tuple[str, str, str], list[dict]] | None = None,
    code_index: Mapping[tuple[str, str], list[tuple[tuple[str, str, str], list[dict]]]] | None = None,
    require_metadata: bool = True,
) -> LifecycleDecision:
    """Validate whether a security may have data on a specific trading date."""
    day = parse_day(when)
    if day is None:
        return LifecycleDecision(False, "invalid_trade_date", "", "", "", asset_type)
    try:
        identity = parse_identity(
            symbol,
            asset_type=asset_type,
            expected_exchange=expected_exchange,
            require_lookup=False,
        )
    except IdentityError as exc:
        return LifecycleDecision(False, "market_identity_conflict", str(symbol), "", "", asset_type)
    symbol_key = identity.symbol
    index = index if index is not None else load_lifecycle_index(data_dir)
    same_code = (
        list(code_index.get((asset_type, identity.code), []))
        if code_index is not None
        else [
            (key, values)
            for key, values in index.items()
            if key[0] == asset_type and key[2] == identity.code
        ]
    )
    if any(key[1] != identity.exchange for key, _ in same_code):
        return LifecycleDecision(False, "market_identity_conflict", symbol_key, identity.code,
                                 identity.exchange, asset_type)
    records = index.get((asset_type, identity.exchange, identity.code), [])
    if not records:
        reason = "identity_history_unknown" if require_metadata else "no_lifecycle_metadata"
        return LifecycleDecision(False, reason, symbol_key, identity.code, identity.exchange, asset_type)

    # A known listing date is a hard lower bound.  Never request a source for
    # a security before it existed under this identity.
    listing_dates = [r["listing_date"] for r in records if r.get("listing_date")]
    earliest_listing = min(listing_dates) if listing_dates else None
    if earliest_listing and day < earliest_listing:
        row = min(records, key=lambda r: r.get("listing_date") or date.max)
        return LifecycleDecision(False, "not_listed_yet", symbol_key, identity.code,
                                 identity.exchange, asset_type, row.get("status", "unknown"),
                                 earliest_listing, row.get("delist_date"),
                                 row.get("successor_symbol", ""), row.get("event_type", ""),
                                 row.get("source", ""))

    active: list[dict] = []
    ended: list[dict] = []
    for row in records:
        start = row.get("effective_from") or row.get("listing_date")
        end = row.get("effective_to") or row.get("delist_date")
        if start and day < start:
            continue
        if end and day > end:
            ended.append(row)
            continue
        if row.get("status") == "delisted" and end is None:
            # A D snapshot without its effective date cannot prove that this
            # historical date was already delisted.  Quarantine it instead
            # of deleting potentially valid history.
            continue
        active.append(row)
    if active:
        row = sorted(active, key=lambda r: (r.get("effective_from") or date.min), reverse=True)[0]
        return LifecycleDecision(True, "active", symbol_key, identity.code, identity.exchange,
                                 asset_type, row.get("status", "listed"), row.get("listing_date"),
                                 row.get("delist_date"), row.get("successor_symbol", ""),
                                 row.get("event_type", ""), row.get("source", ""))
    if ended:
        row = max(ended, key=lambda r: r.get("effective_to") or r.get("delist_date") or date.min)
        successor = row.get("successor_symbol", "")
        reason = "code_changed" if successor else "delisted_on_date"
        return LifecycleDecision(False, reason, symbol_key, identity.code, identity.exchange,
                                 asset_type, row.get("status", "delisted"), row.get("listing_date"),
                                 row.get("delist_date"), successor, row.get("event_type", ""),
                                 row.get("source", ""))
    return LifecycleDecision(False, "lifecycle_date_missing", symbol_key, identity.code,
                             identity.exchange, asset_type)


def validate_supplement_identity(
    target_symbol: object,
    source_symbol: object,
    when: date | datetime | str,
    data_dir: str | Path,
    *,
    asset_type: str = "stock",
    index: dict[tuple[str, str, str], list[dict]] | None = None,
) -> dict:
    """Cross-check requested, source-returned and stored identities."""
    try:
        target = parse_identity(target_symbol, asset_type=asset_type)
        source = parse_identity(source_symbol, asset_type=asset_type)
    except IdentityError:
        return {"ok": False, "reason": "market_identity_conflict",
                "target_symbol": str(target_symbol), "source_symbol": str(source_symbol)}
    if target != source:
        return {"ok": False, "reason": "market_identity_conflict",
                "target_symbol": target.symbol, "source_symbol": source.symbol}
    decision = validate_security_lifecycle(
        target.symbol, when, data_dir, asset_type=asset_type, index=index,
    )
    return {**decision.as_dict(), "target_symbol": target.symbol, "source_symbol": source.symbol}


def is_terminal_lifecycle_reason(reason: str | None) -> bool:
    return str(reason or "") in LIFECYCLE_TERMINAL_REASONS or str(reason or "") == "code_changed"
