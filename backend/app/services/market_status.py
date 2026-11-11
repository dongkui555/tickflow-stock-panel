"""Fail-closed market-session and historical suspension gates.

The minute repair queue must distinguish a closed market or a stock that was
explicitly suspended from a data-source failure.  This module only consumes
persisted evidence; an absent calendar or suspension file is ``unknown`` and
therefore does not suppress a repair request.

Supported persisted evidence:

* ``data/market_calendar/trade_calendar.parquet`` with ``trade_date`` and
  ``is_open`` columns;
* ``data/instruments_history/suspensions.parquet`` with an identity column
  (``symbol`` or ``code`` + ``exchange``), ``suspend_date`` and optional
  ``resume_date``.

``resume_date`` is exclusive, matching exchange suspension calendars.  A
missing or ambiguous identity never matches a stock-day.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl

from app.services.security_identity import IdentityError, normalize_exchange, parse_identity


_CALENDAR_PATHS = (
    Path("market_calendar/trade_calendar.parquet"),
    Path("market_calendar/market_calendar.parquet"),
    Path("trade_calendar/trade_calendar.parquet"),
)
_SUSPENSION_PATHS = (
    Path("instruments_history/suspensions.parquet"),
    Path("instruments_history/suspend_d.parquet"),
    Path("suspensions/suspensions.parquet"),
)


def _parse_day(value: object) -> date | None:
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
        if value is not None and str(value).strip().lower() not in {"", "none", "nan", "nat"}:
            return value
    return None


def _bool_open(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "open", "trade"}:
        return True
    if text in {"0", "0.0", "false", "no", "n", "closed", "holiday"}:
        return False
    return None


def _read_first_existing(data_dir: str | Path, paths: tuple[Path, ...]) -> pl.DataFrame | None:
    root = Path(data_dir)
    for relative in paths:
        path = root / relative
        if not path.exists():
            continue
        try:
            return pl.read_parquet(path)
        except Exception:
            # A corrupt optional status file is not proof of a closed market or
            # suspension.  Treat it as unavailable and keep the repair gate open.
            continue
    return None


def market_day_status(data_dir: str | Path, when: date | datetime | str) -> dict:
    """Return ``open``, ``market_closed`` or ``unknown`` for one date.

    Weekends are deterministic.  For weekdays, an explicitly persisted
    exchange calendar wins; a non-empty local daily partition is only a
    positive fallback.  Missing evidence never becomes ``market_closed``.
    """
    day = _parse_day(when)
    if day is None:
        return {"status": "unknown", "reason": "invalid_trade_date"}
    if day.weekday() >= 5:
        return {"status": "market_closed", "reason": "weekend", "date": day.isoformat()}

    frame = _read_first_existing(data_dir, _CALENDAR_PATHS)
    if frame is not None and not frame.is_empty():
        date_column = next((name for name in ("trade_date", "cal_date", "date") if name in frame.columns), None)
        open_column = next((name for name in ("is_open", "open", "status") if name in frame.columns), None)
        if date_column and open_column:
            matches = []
            for row in frame.to_dicts():
                if _parse_day(row.get(date_column)) == day:
                    matches.append(_bool_open(row.get(open_column)))
            known = [value for value in matches if value is not None]
            if known:
                return {
                    "status": "open" if known[-1] else "market_closed",
                    "reason": "trade_calendar",
                    "date": day.isoformat(),
                }

    daily_path = Path(data_dir) / "kline_daily" / f"date={day.isoformat()}" / "part.parquet"
    if daily_path.exists():
        try:
            if pl.read_parquet(daily_path).height > 0:
                return {"status": "open", "reason": "local_daily_partition", "date": day.isoformat()}
        except Exception:
            pass
    return {"status": "unknown", "reason": "market_calendar_missing", "date": day.isoformat()}


def _canonical_symbol(row: dict) -> str | None:
    raw = _first(row, "symbol", "ts_code", "security")
    exchange = normalize_exchange(_first(row, "exchange", "market", "exchange_code"))
    if raw is None and row.get("code") is not None:
        raw = row.get("code")
    try:
        return parse_identity(raw, asset_type="stock", expected_exchange=exchange).symbol
    except (IdentityError, TypeError, ValueError):
        return None


def historical_suspension_status(
    data_dir: str | Path,
    symbol: object,
    when: date | datetime | str,
) -> dict:
    """Check an explicit historical suspension interval for one stock-day.

    This intentionally does not infer historical suspension from a current
    ``P`` lifecycle snapshot.  A source must provide an effective interval.
    """
    day = _parse_day(when)
    if day is None:
        return {"status": "unknown", "reason": "invalid_trade_date"}
    try:
        target = parse_identity(symbol, asset_type="stock").symbol
    except (IdentityError, TypeError, ValueError):
        return {"status": "unknown", "reason": "identity_conflict"}

    frame = _read_first_existing(data_dir, _SUSPENSION_PATHS)
    if frame is None or frame.is_empty():
        return {"status": "unknown", "reason": "historical_suspension_data_missing"}
    start_names = ("suspend_date", "suspension_date", "effective_from", "start_date", "date", "trade_date")
    end_names = ("resume_date", "effective_to", "end_date")
    exact_identity_seen = False
    for row in frame.to_dicts():
        canonical = _canonical_symbol(row)
        if canonical != target:
            continue
        exact_identity_seen = True
        start = _parse_day(_first(row, *start_names))
        if start is None:
            continue
        resume = _parse_day(_first(row, *end_names))
        # resume_date is the first trading date on which the security may trade.
        if day < start or (resume is not None and day >= resume):
            continue
        status = str(_first(row, "status", "suspend_status") or "suspended").strip().lower()
        if status in {"resume", "resumed", "active", "listed", "0", "false"}:
            continue
        return {
            "status": "suspended",
            "reason": str(_first(row, "reason", "suspend_reason", "event_type") or "historical_suspension"),
            "symbol": target,
            "start_date": start.isoformat(),
            "resume_date": resume.isoformat() if resume else None,
        }
    if exact_identity_seen:
        return {"status": "not_suspended", "reason": "no_interval_match", "symbol": target}
    return {"status": "unknown", "reason": "historical_identity_not_found", "symbol": target}
