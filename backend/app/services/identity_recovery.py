"""Conservative recovery of rows quarantined by canonical identity migration.

The initial canonical migration intentionally failed closed when a row did not
carry an exchange.  This module is the explicit second pass: it may use the
historical dimension and deterministic stock code namespaces, but only after
the identity is unique and valid on the row's date.  It does not write data;
the command-line recovery job owns reports, backups and atomic publication.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Mapping

import polars as pl

from app.services.security_identity import (
    IdentityError,
    compact_code,
    exchange_for_code,
    explicit_exchange,
    parse_identity,
)
from app.services.security_lifecycle import (
    LifecycleDecision,
    parse_day,
    validate_security_lifecycle,
)


@dataclass(frozen=True)
class IdentityRecoveryDecision:
    ok: bool
    reason: str
    symbol: str = ""
    code: str = ""
    exchange: str = ""
    asset_type: str = "stock"
    basis: str = ""
    lifecycle: LifecycleDecision | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "symbol": self.symbol,
            "code": self.code,
            "exchange": self.exchange,
            "asset_type": self.asset_type,
            "basis": self.basis,
            "lifecycle": self.lifecycle.as_dict() if self.lifecycle else None,
        }


def _dimension_exchanges(
    code: str,
    asset_type: str,
    lifecycle_index: Mapping[tuple[str, str, str], list[dict]],
    code_index: Mapping[tuple[str, str], list[tuple[tuple[str, str, str], list[dict]]]] | None = None,
) -> set[str]:
    if code_index is not None:
        return {key[1] for key, _ in code_index.get((asset_type, code), [])}
    return {key[1] for key in lifecycle_index if key[0] == asset_type and key[2] == code}


def resolve_quarantined_identity(
    raw_symbol: object,
    *,
    asset_type: str,
    trade_date: date | datetime | str | None,
    lookup: Mapping[str, str],
    lifecycle_index: Mapping[tuple[str, str, str], list[dict]],
    lifecycle_code_index: Mapping[tuple[str, str], list[tuple[tuple[str, str, str], list[dict]]]] | None = None,
    data_dir=None,
    allow_structural_stock_market: bool = False,
) -> IdentityRecoveryDecision:
    """Resolve one quarantined row without guessing a market.

    ``lookup`` is the preferred source.  Structural stock namespaces are an
    explicit fallback only, and still require a matching date-aware lifecycle
    record.  Index and ETF rows never use structural inference.
    """
    asset_type = "stock" if asset_type == "equity" else asset_type
    code = compact_code(raw_symbol)
    if not code:
        return IdentityRecoveryDecision(False, "code_unparseable", asset_type=asset_type)
    if parse_day(trade_date) is None:
        return IdentityRecoveryDecision(False, "identity_date_missing", code=code, asset_type=asset_type)

    mapped = str(lookup.get(code) or "").strip().upper()
    explicit = explicit_exchange(raw_symbol)
    if mapped:
        try:
            mapped_identity = parse_identity(mapped, asset_type=asset_type, require_explicit=True)
        except IdentityError:
            return IdentityRecoveryDecision(False, "dimension_identity_invalid", code=code, asset_type=asset_type)
        if explicit and explicit != mapped_identity.exchange:
            return IdentityRecoveryDecision(
                False, "dimension_exchange_conflict", code=code, exchange=mapped_identity.exchange,
                asset_type=asset_type,
            )
        exchange = mapped_identity.exchange
        basis = "dimension"
    else:
        if asset_type != "stock" or not allow_structural_stock_market:
            return IdentityRecoveryDecision(False, "exchange_unknown", code=code, asset_type=asset_type)
        exchange = exchange_for_code(code, asset_type)
        if not exchange:
            return IdentityRecoveryDecision(False, "exchange_unknown", code=code, asset_type=asset_type)
        if explicit and explicit != exchange:
            return IdentityRecoveryDecision(False, "code_exchange_conflict", code=code, exchange=exchange, asset_type=asset_type)
        basis = "code_rule"

    exchanges = _dimension_exchanges(code, asset_type, lifecycle_index, lifecycle_code_index)
    if len(exchanges) > 1:
        return IdentityRecoveryDecision(
            False, "market_identity_ambiguous", code=code, exchange=exchange,
            asset_type=asset_type, basis=basis,
        )
    if exchanges and exchange not in exchanges:
        return IdentityRecoveryDecision(
            False, "market_identity_conflict", code=code, exchange=exchange,
            asset_type=asset_type, basis=basis,
        )

    symbol = f"{code}.{exchange}"
    lifecycle = validate_security_lifecycle(
        symbol,
        trade_date,
        data_dir or ".",
        asset_type=asset_type,
        index=lifecycle_index,
        code_index=lifecycle_code_index,
        require_metadata=True,
    )
    if not lifecycle.ok:
        return IdentityRecoveryDecision(
            False, lifecycle.reason, symbol=symbol, code=code, exchange=exchange,
            asset_type=asset_type, basis=basis, lifecycle=lifecycle,
        )
    return IdentityRecoveryDecision(
        True, "identity_verified", symbol=symbol, code=code, exchange=exchange,
        asset_type=asset_type, basis=basis, lifecycle=lifecycle,
    )


def row_trade_date(row: Mapping[str, object], partition_date: date | None = None) -> date | None:
    """Extract a logical market date, falling back to a partition date."""
    for name in ("date", "trade_date", "datetime", "timestamp", "ann_date", "end_date", "period_end", "announce_date"):
        if name not in row or row.get(name) is None:
            continue
        value = row.get(name)
        parsed = parse_day(value)
        if parsed:
            return parsed
    return partition_date


def is_valid_ohlcv_row(row: Mapping[str, object]) -> bool:
    """Validate the raw bar without inferring units or repairing values."""
    values = []
    for field in ("open", "high", "low", "close"):
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            return False
        if not isfinite(value) or value <= 0:
            return False
        values.append(value)
    try:
        volume = float(row.get("volume"))
        amount = float(row.get("amount"))
    except (TypeError, ValueError):
        return False
    if not isfinite(volume) or not isfinite(amount) or volume < 0 or amount < 0:
        return False
    return values[1] >= max(values[0], values[3], values[2]) and values[2] <= min(values[0], values[3], values[1])


RECOVERY_KEY_COLUMNS = {
    "kline_daily": ["symbol", "date"],
    "kline_daily_enriched": ["symbol", "date"],
    "kline_minute": ["symbol", "datetime"],
    "moneyflow": ["symbol", "date"],
    "auction_open": ["symbol", "date"],
    "financials": ["symbol", "period_end", "announce_date"],
    "adj_factor": ["symbol", "trade_date"],
}


def dedupe_key(dataset: str, frame: pl.DataFrame) -> list[str]:
    """Return a lossless identity/time key for an accepted recovery frame."""
    candidates = RECOVERY_KEY_COLUMNS.get(dataset, ["symbol", "date"])
    keys = [column for column in candidates if column in frame.columns]
    # Never collapse a historical series to one row per security when a
    # provider used a different date-column name than the default contract.
    if dataset in {"financials", "adj_factor", "kline_daily", "kline_daily_enriched", "kline_minute"} and len(keys) < 2:
        return []
    return keys
