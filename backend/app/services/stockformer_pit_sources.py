"""Coverage-checked research reference reads through an existing provider API."""
from __future__ import annotations

import time
from datetime import date
from email.utils import parsedate_to_datetime


class ReferenceRateLimitError(RuntimeError):
    def __init__(self, seconds):
        self.retry_after_seconds = max(1., float(seconds))
        super().__init__('Historical reference source is rate limited')


class ReferenceCoverageError(RuntimeError):
    def __init__(self, missing, *, records=None, verification=None):
        self.missing_symbols = sorted(missing)
        self.records = records or []
        self.verification = verification or {}
        super().__init__(f'Historical reference coverage incomplete: {len(missing)} symbols')


class ReferenceSourceError(RuntimeError):
    def __init__(self, error_type, status_code=None):
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = status_code in {408, 425, 500, 502, 503, 504} or (
            status_code is None and error_type in {'ReadTimeout', 'ConnectTimeout', 'ReadError',
                'ConnectError', 'RemoteProtocolError', 'RuntimeError'})
        super().__init__(f'Historical reference request failed ({error_type}, status={status_code})')


def should_fetch_reference(previous, plan_sha256, *, now, max_failures=3):
    """Stable coverage gaps need adjudication, while quota cooldown is resumable."""
    if not previous or previous.get('plan_sha256') != plan_sha256:
        return True
    if previous.get('status') in {'complete', 'coverage_gap', 'permission_denied', 'invalid_response'}:
        return False
    if previous.get('failures', 0) >= max_failures:
        return False
    return now >= previous.get('retry_at', 0)


def _rate_limit(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        response = getattr(exc, 'response', None)
        if response is not None and getattr(response, 'status_code', None) == 429:
            value = response.headers.get('Retry-After', '')
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                try:
                    seconds = parsedate_to_datetime(value).timestamp()-time.time()
                except (TypeError, ValueError, OverflowError):
                    seconds = 60.
            raise ReferenceRateLimitError(seconds) from None
        exc = exc.__cause__ or exc.__context__


def fetch_reference_day(api, endpoint, day: date, required_symbols, *, page_size=1000, native_first=False):
    """Require actual panel coverage, not an API success or short page alone.

    Gateways can cap below the documented upstream limit. Keep the requested
    page size at most 1000, check repeated pages/keys/date and fail closed when
    any required stock is absent. No short page is called a complete market.
    Native-first mode allows a pinned gateway to return its full day directly;
    an incomplete native response resumes at the actual returned offset.
    """
    if endpoint not in {'bak_basic', 'stk_limit'} or not 1 <= page_size <= 1000:
        raise ValueError('Unsupported reference endpoint or page size')
    required = set(required_symbols)
    if not required:
        raise ValueError('Empty expected reference universe')
    collected, fingerprints, sources, page_rows = {}, set(), set(), []

    def coverage_error(missing):
        return ReferenceCoverageError(missing, records=list(collected.values()), verification={
            'endpoint': endpoint, 'date': str(day), 'expected_symbols': len(required),
            'covered_symbols': len(collected), 'page_rows': list(page_rows),
            'sources': sorted(sources) or [type(api).__name__],
            'coverage_complete': False, 'native_first': native_first})

    offset = 0
    for page in range(8+int(native_first)):
        native = native_first and page == 0
        params = {'trade_date': day.strftime('%Y%m%d')}
        if not native:
            params.update(limit=page_size, offset=offset)
        if endpoint == 'bak_basic':
            params['fields'] = 'trade_date,ts_code,name,industry'
        try:
            records = getattr(api, endpoint)(**params).to_dict(orient='records')
        except Exception as exc:  # noqa: BLE001 -- Vendor exceptions can contain credentials.
            _rate_limit(exc)
            response = getattr(exc, 'response', None)
            raise ReferenceSourceError(type(exc).__name__, getattr(response, 'status_code', None)) from None
        source = getattr(api, 'last_source', None)
        if isinstance(source, str):
            sources.add(source)
        if len(sources) > 1:
            raise ValueError('Reference source changed during pagination')
        page_rows.append(len(records))
        identities = tuple(str(r.get('ts_code', '')) for r in records)
        if records and identities in fingerprints:
            raise ValueError('Repeated reference pagination page')
        fingerprints.add(identities)
        for row in records:
            if str(row.get('trade_date', '')).replace('-', '') != day.strftime('%Y%m%d'):
                raise ValueError('Unexpected historical reference date')
            symbol = str(row.get('ts_code', '')).split('.')[0]
            if symbol not in required:
                continue
            if symbol in collected and collected[symbol] != row:
                raise ValueError('Conflicting historical reference identity')
            collected[symbol] = row
        missing = required-set(collected)
        if not missing:
            return list(collected.values()), {'endpoint': endpoint, 'date': str(day),
                'expected_symbols': len(required), 'covered_symbols': len(collected),
                'page_rows': page_rows, 'sources': sorted(sources) or [type(api).__name__],
                'coverage_complete': True, 'native_first': native_first}
        if not records or (not native and len(records) < page_size):
            raise coverage_error(missing)
        offset += len(records)
    raise coverage_error(required-set(collected))
