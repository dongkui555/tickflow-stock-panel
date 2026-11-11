"""Closed-session quality gate for research, never for a live partial session."""
from __future__ import annotations

import polars as pl

from app.services.minute_repair import (
    COLS,
    PRICE,
    canonical_minute_aliases,
    conflicting_complete_codes,
    daily_reference,
    keyed,
    regular,
)


def filter_minute_sessions(frame: pl.DataFrame, daily: pl.DataFrame | None = None, *, data_dir=None):
    """Return complete stock-days and explicit exclusions; never fabricate bars.

    With raw daily references, also require same-day OHLC and turnover agreement.
    Without daily references this checks structure only (legacy repository stubs).
    """
    if frame.is_empty():
        return frame, []
    if not set(COLS).issubset(frame.columns):
        raise ValueError('Minute data lacks required OHLC/turnover fields')
    if frame.schema['datetime'].time_zone is not None:
        frame = frame.with_columns(pl.col('datetime').dt.convert_time_zone('Asia/Shanghai').dt.replace_time_zone(None))
    kept, issues = [], []
    for tagged in frame.sort('datetime').with_columns(pl.col('datetime').dt.date().alias('_day')).partition_by('_day'):
        part = tagged.drop('_day')
        day = part['datetime'][0].date()
        source = keyed(canonical_minute_aliases(part, day)).with_columns(
            pl.col('_code').fill_null(pl.col('symbol'))
        )
        counts = dict(regular(source, day).group_by('_code').agg(pl.col('datetime').n_unique()).iter_rows())
        clock_rows = dict(regular(source, day, include_opening=True).group_by('_code').len().iter_rows())
        good = pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in PRICE])
        good &= (pl.col('high') >= pl.max_horizontal(PRICE)) & (pl.col('low') <= pl.min_horizontal(PRICE))
        good &= pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) >= 0) for c in ('volume', 'amount')])
        summary = source.group_by('_code').agg(pl.col('symbol').first(), pl.len().alias('rows'),
                   pl.col('datetime').n_unique().alias('unique'), good.fill_null(False).all().alias('valid'))
        refs = daily_reference(daily.filter(pl.col('date') == day), data_dir=data_dir) if daily is not None and not daily.is_empty() else {}
        conflicts = conflicting_complete_codes(part, refs, day) if refs else set()
        allowed = []
        for row in summary.iter_rows(named=True):
            code = row['_code']
            reason = None
            if code is None:
                reason = 'invalid_symbol'
            elif counts.get(code, 0) != 240:
                reason = 'incomplete_session'
            elif row['rows'] != row['unique'] or clock_rows.get(code, 0) != row['rows']:
                reason = 'invalid_session_timestamps'
            elif not row['valid']:
                reason = 'invalid_minute_fields'
            elif daily is not None and code not in refs:
                reason = 'daily_reference_missing'
            elif code in conflicts:
                reason = 'daily_price_or_turnover_conflict'
            if reason:
                issues.append({'date': str(day), 'symbol': row['symbol'], 'reason': reason})
            else:
                allowed.append(code)
        kept.append(source.filter(pl.col('_code').is_in(allowed)).drop('_code'))
    return pl.concat(kept) if kept else frame.clear(), issues
