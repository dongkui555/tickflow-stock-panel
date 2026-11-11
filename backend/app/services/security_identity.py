"""Fail-closed security identity checks for imported market data.

The parquet layout is partitioned by trading date, but the logical key is
asset type + exchange + code + timestamp/date.  This module keeps the
identity contract in one place without forcing an immediate rewrite of old
bare-code partitions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import polars as pl


EXCHANGES = {"SH", "SZ", "BJ"}
STOCK_PREFIXES = {
    # 20xxxx/90xxxx are the Shenzhen/Shanghai B-share namespaces.
    "SH": ("60", "68", "90"),
    "SZ": ("00", "20", "30"),
    "BJ": ("43", "83", "87", "88", "92"),
}
EXCHANGE_ALIASES = {
    "SSE": "SH",
    "SZSE": "SZ",
    "BSE": "BJ",
    "SHANGHAI": "SH",
    "SHENZHEN": "SZ",
    "BEIJING": "BJ",
}
ASSET_TYPES = frozenset({"stock", "index", "etf"})


class IdentityError(ValueError):
    """A source row cannot be assigned to one unambiguous security identity."""


class IdentityBatchError(IdentityError):
    """A batch contains one or more rows with invalid identity metadata."""

    def __init__(self, issues: list[dict[str, str]]) -> None:
        self.issues = issues
        sample = "; ".join(
            f"{item.get('raw', '')}: {item.get('reason', 'identity_conflict')}"
            for item in issues[:5]
        )
        suffix = f" (还有 {len(issues) - 5} 条)" if len(issues) > 5 else ""
        super().__init__(f"证券身份校验失败: {sample}{suffix}")


@dataclass(frozen=True)
class SecurityIdentity:
    asset_type: str
    exchange: str
    code: str

    @property
    def symbol(self) -> str:
        return f"{self.code}.{self.exchange}"

    @property
    def security_key(self) -> str:
        return f"{self.asset_type}|{self.exchange}|{self.code}"

    def stock_day_key(self, when: date | datetime | str) -> str:
        return f"{self.security_key}|{_date_text(when)}"


@dataclass(frozen=True)
class IdentityBatch:
    accepted: pl.DataFrame
    quarantined: pl.DataFrame
    reasons: dict[str, int]


def normalize_exchange(value: object) -> str | None:
    text = str(value or "").strip().upper().lstrip(".")
    if not text:
        return None
    return EXCHANGE_ALIASES.get(text, text) if text in EXCHANGE_ALIASES else text if text in EXCHANGES else None


def compact_code(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return None
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    if match:
        return match.group(1)
    if text.isdigit() and len(text) < 6:
        return text.zfill(6)
    return None


def explicit_exchange(value: object) -> str | None:
    text = str(value or "").strip().upper()
    suffix = re.search(r"\.([A-Z]+)$", text)
    if suffix:
        return normalize_exchange(suffix.group(1))
    prefix = re.match(r"^(SH|SZ|BJ|SSE|SZSE|BSE)[._-]?\d{6}$", text)
    return normalize_exchange(prefix.group(1)) if prefix else None


def exchange_for_code(code: str, asset_type: str = "stock") -> str | None:
    """Infer only structurally unique stock markets; never infer index/ETF."""
    if asset_type != "stock":
        return None
    for exchange, prefixes in STOCK_PREFIXES.items():
        if code.startswith(prefixes):
            return exchange
    return None


def _date_text(when: date | datetime | str) -> str:
    if isinstance(when, datetime):
        return when.date().isoformat()
    if isinstance(when, date):
        return when.isoformat()
    text = str(when).strip()
    if not text:
        raise IdentityError("date_missing")
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise IdentityError("date_invalid") from exc


def _asset_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "equity":
        text = "stock"
    if text not in ASSET_TYPES:
        raise IdentityError("asset_type_unknown")
    return text


def build_instrument_lookup(frames: Iterable[pl.DataFrame]) -> dict[str, str]:
    """Build code -> canonical symbol, dropping ambiguous duplicate codes."""
    candidates: dict[str, set[str]] = {}
    for frame in frames:
        if frame is None or frame.is_empty() or "code" not in frame.columns:
            continue
        columns = [column for column in ("code", "symbol", "exchange") if column in frame.columns]
        for row in frame.select(columns).drop_nulls(subset=["code"]).to_dicts():
            code = compact_code(row.get("code") or row.get("symbol"))
            if not code:
                continue
            exchange = normalize_exchange(row.get("exchange")) or explicit_exchange(row.get("symbol"))
            if exchange:
                candidates.setdefault(code, set()).add(f"{code}.{exchange}")
    return {code: next(iter(values)) for code, values in candidates.items() if len(values) == 1}


def parse_identity(
    value: object,
    *,
    asset_type: str,
    lookup: dict[str, str] | None = None,
    expected_exchange: str | None = None,
    require_lookup: bool = False,
    require_explicit: bool = False,
) -> SecurityIdentity:
    asset_type = _asset_type(asset_type)
    code = compact_code(value)
    if not code:
        raise IdentityError("code_unparseable")
    explicit = explicit_exchange(value)
    expected = normalize_exchange(expected_exchange)
    mapped = lookup.get(code) if lookup else None
    mapped_exchange = explicit_exchange(mapped) if mapped else None
    if explicit and expected and explicit != expected:
        raise IdentityError("expected_exchange_conflict")
    if explicit and mapped_exchange and explicit != mapped_exchange:
        raise IdentityError("dimension_exchange_conflict")
    if expected and mapped_exchange and expected != mapped_exchange:
        raise IdentityError("expected_dimension_conflict")
    if require_explicit and not (explicit or mapped_exchange or expected):
        raise IdentityError("exchange_unknown")
    exchange = explicit or mapped_exchange or expected
    if not exchange:
        exchange = exchange_for_code(code, asset_type)
    if not exchange:
        raise IdentityError("exchange_unknown")
    structural = exchange_for_code(code, asset_type)
    if asset_type == "stock" and structural and structural != exchange:
        raise IdentityError("code_exchange_conflict")
    if require_lookup and not mapped:
        raise IdentityError("security_not_in_dimension")
    return SecurityIdentity(asset_type=asset_type, exchange=exchange, code=code)


def canonicalize_identity_frame(
    frame: pl.DataFrame,
    *,
    asset_type: str,
    lookup: dict[str, str] | None = None,
    expected_symbols: Iterable[str] | None = None,
    require_explicit: bool = True,
) -> IdentityBatch:
    """Resolve a frame into canonical identities and isolate rejected rows.

    The original row is retained in the quarantine output with an explicit
    ``original_symbol`` and ``identity_reason`` column. No row is inferred to
    a market when the source does not provide enough evidence.
    """
    if frame.is_empty():
        return IdentityBatch(frame, frame, {})
    if "symbol" not in frame.columns:
        quarantine = frame.with_columns(
            pl.lit("").alias("original_symbol"),
            pl.lit("symbol_column_missing").alias("identity_reason"),
        )
        return IdentityBatch(frame.clear(), quarantine, {"symbol_column_missing": frame.height})

    expected: dict[str, SecurityIdentity] = {}
    for symbol in expected_symbols or ():
        try:
            resolved = parse_identity(symbol, asset_type=asset_type, lookup=lookup, require_explicit=True)
            expected[resolved.code] = resolved
        except IdentityError:
            continue

    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    reasons: dict[str, int] = {}
    for row_number, row in enumerate(frame.to_dicts()):
        raw = row.get("symbol")
        try:
            target = expected.get(compact_code(raw) or "")
            resolved = parse_identity(
                raw,
                asset_type=asset_type,
                lookup=lookup,
                expected_exchange=(
                    normalize_exchange(row.get("exchange"))
                    or (target.exchange if target else None)
                ),
                require_explicit=require_explicit,
            )
            row.update({
                "asset_type": resolved.asset_type,
                "exchange": resolved.exchange,
                "code": resolved.code,
                "symbol": resolved.symbol,
                "security_key": resolved.security_key,
            })
            accepted_rows.append(row)
        except IdentityError as exc:
            reason = str(exc)
            reasons[reason] = reasons.get(reason, 0) + 1
            row["original_symbol"] = raw
            row["identity_reason"] = reason
            row["identity_row"] = row_number
            rejected_rows.append(row)

    accepted = pl.DataFrame(accepted_rows) if accepted_rows else frame.clear()
    quarantined = pl.DataFrame(rejected_rows) if rejected_rows else frame.clear().with_columns(
        pl.lit(None).cast(pl.String).alias("original_symbol"),
        pl.lit(None).cast(pl.String).alias("identity_reason"),
        pl.lit(None).cast(pl.Int64).alias("identity_row"),
    )
    return IdentityBatch(accepted, quarantined, reasons)


def validate_frame_identity(
    frame: pl.DataFrame,
    *,
    asset_type: str,
    lookup: dict[str, str] | None = None,
    expected_symbols: Iterable[str] | None = None,
    require_lookup: bool = False,
) -> pl.DataFrame:
    """Validate every row before an active write is allowed."""
    if frame.is_empty():
        return frame
    if "symbol" not in frame.columns:
        raise IdentityBatchError([{"raw": "", "reason": "symbol_column_missing"}])
    expected: dict[str, SecurityIdentity] = {}
    for symbol in expected_symbols or ():
        try:
            identity = parse_identity(symbol, asset_type=asset_type, lookup=lookup, require_lookup=False)
            expected[identity.code] = identity
        except IdentityError:
            continue
    issues: list[dict[str, str]] = []
    for raw in frame.get_column("symbol").cast(pl.String).to_list():
        code = compact_code(raw)
        target = expected.get(code or "")
        try:
            parse_identity(
                raw,
                asset_type=asset_type,
                lookup=lookup,
                expected_exchange=target.exchange if target else None,
                require_lookup=require_lookup and target is None,
                require_explicit=False,
            )
        except IdentityError as exc:
            issues.append({"raw": str(raw), "code": str(code or ""), "reason": str(exc)})
    if issues:
        raise IdentityBatchError(issues)
    return frame


def identity_key(
    symbol: object,
    when: date | datetime | str,
    *,
    asset_type: str,
    lookup: dict[str, str] | None = None,
) -> str:
    identity = parse_identity(symbol, asset_type=asset_type, lookup=lookup, require_lookup=False)
    stamp = when.isoformat() if isinstance(when, datetime) else _date_text(when)
    return f"{identity.asset_type}|{identity.exchange}|{identity.code}|{stamp}"
