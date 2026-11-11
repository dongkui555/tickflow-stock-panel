"""Bounded calendar refresh through the configured provider; no weekday fill."""
from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from app.market_time import CN_TZ, cn_now
from app.services.fs_utils import atomic_write_parquet, atomic_write_text
from app.services.market_status import _bool_open, _parse_day

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
CALENDAR = Path('market_calendar/trade_calendar.parquet')
STATE = Path('user_data/market_calendar_sync.json')


def _days(start: date, end: date):
    return [start + timedelta(days=i) for i in range((end-start).days + 1)]


def _normalize(rows: list[dict]) -> dict[date, int]:
    values = {}
    for row in rows:
        day, opened = _parse_day(row.get('trade_date')), _bool_open(row.get('is_open'))
        if day is None or opened is None:
            raise ValueError('invalid_calendar_row')
        value = int(opened)
        if day in values and values[day] != value:
            raise ValueError('conflicting_calendar_day')
        values[day] = value
    return values


def refresh(data_dir: Path, *, now: datetime | None = None, provider=None) -> dict:
    """Called in the pipeline execution slot; failures retain the prior file.

    Calendar days include explicit closed sessions. A complete date interval
    must be provided by the source; missing rows never become weekday guesses.
    """
    if not _LOCK.acquire(blocking=False):
        return {'status': 'busy'}
    root = Path(data_dir)
    now = (now or cn_now()).astimezone(CN_TZ)
    state_path = root / STATE
    state = {}
    result = None
    try:
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
        path = root / CALENDAR
        old = _normalize(pl.read_parquet(path).to_dicts()) if path.exists() else {}
        missing = [day for day in _days(now.date()-timedelta(days=7), now.date()+timedelta(days=30))
                   if day not in old]
        if not missing:
            return {'status': 'current', 'through': str(max(old)), 'checked_at': now.isoformat()}
        retry = state.get('next_retry_at')
        if retry and datetime.fromisoformat(retry) > now:
            return {**state, 'status': 'backoff'}
        start = min(missing)
        start = min(start, max(old)+timedelta(days=1)) if old else now.date()-timedelta(days=45)
        end = now.date()+timedelta(days=45)
        if provider is None:
            from app.data_providers import custom
            from app.services import preferences

            source = preferences.get_daily_data_provider()
            provider = custom.get_provider(source)
        else:
            source = type(provider).__name__
        fetch = getattr(provider, 'get_trade_calendar', None)
        if not callable(fetch):
            raise NotImplementedError('configured_provider_has_no_calendar')
        incoming = _normalize(fetch(start.strftime('%Y%m%d'), end.strftime('%Y%m%d')) or [])
        expected = set(_days(start, end))
        if set(incoming) != expected:
            raise ValueError('calendar_source_range_incomplete_or_unexpected')
        merged = {**old, **incoming}
        frame = pl.DataFrame({'trade_date': [day.strftime('%Y%m%d') for day in sorted(merged)],
                              'is_open': [merged[day] for day in sorted(merged)]})
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(frame, path)
        result = {'status': 'succeeded', 'checked_at': now.isoformat(), 'source': source,
                  'start': str(start), 'through': str(end), 'source_rows': len(incoming),
                  'total_rows': len(merged), 'attempts': 0, 'next_retry_at': None}
    except Exception as exc:  # noqa: BLE001 — retain data, sanitize credentials, retry later.
        attempts = state.get('attempts', 0)+1 if state.get('checked_at', '')[:10] == str(now.date()) else 1
        result = {'status': 'retry', 'checked_at': now.isoformat(), 'attempts': attempts,
                  'error_type': type(exc).__name__,
                  'next_retry_at': (now+timedelta(minutes=min(120, 30*2**min(attempts-1, 2)))).isoformat()}
        logger.warning('Market calendar refresh deferred: %s', type(exc).__name__)
    finally:
        try:
            if result is not None:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(state_path, json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            _LOCK.release()
    return result
