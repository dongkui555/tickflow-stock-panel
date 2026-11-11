"""Matched inputs for the 2af research pair, separate from frozen generators."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_decision_research as decision
from app.services import stockformer_pit_inputs as pit
from app.services.stockformer_research_price_basis import feature_price_view
from app.services.stockformer_research_turnover import (
    mask_turnover_context,
    turnover_quality_context,
)

KEYS = ['symbol', 'date']
VERSION = 'matched_repaired_inputs_daily_v1'


def matched_features(panel, raw, factors, disputed):
    """Exactly the frozen training producer, without future labels or imputation."""
    eligible_chronology=panel.filter(~pl.col('_pit_lifecycle_excluded').fill_null(False))
    return mask_turnover_context(pit.build_features(feature_price_view(eligible_chronology, raw, factors)),
                                 turnover_quality_context(panel, disputed.select(KEYS)))


def inference_pool(features, day, columns):
    today = features.filter(pl.col('date') == day)
    strict, minute = sf.prepare_training_features(today, 'minute_enhanced')
    _, _, complete = sf._prepare_xy(strict.with_columns(pl.lit(0.).alias('_predict_mask')),
                                    '_predict_mask', columns)
    selected, audit = decision.decision_pool(complete)
    return selected, {**minute, 'complete_rows': complete.height, 'pool_audit': audit,
                       'decision_pool_rows': selected.height,
                       'incomplete_symbols': sorted(set(today['symbol'])-set(complete['symbol'])),
                       'pool_excluded': sorted(set(complete['symbol'])-set(selected['symbol']))}


def dated_frame(item, endpoint, day, required):
    if item.get('date') != str(day) or item.get('endpoint') != endpoint:
        raise ValueError('Dated source capture does not match requested session')
    frame = pit._keys(pl.DataFrame(item['records']), endpoint)
    if frame.is_empty() or frame['date'].unique().to_list() != [day]:
        raise ValueError('Empty or stale dated source capture')
    if not set(required) <= set(frame.columns):
        raise ValueError('Missing required source fields: '+endpoint)
    return frame.select(*KEYS, *required)


def clean_panel(frame):
    # prepare_panel must reconstruct availability and limit streaks across the
    # seam; stale metadata must not survive a join under suffixed columns.
    return frame.drop([c for c in frame.columns if c.startswith('_pit_')
                       or (c.startswith('_raw_') and c != '_raw_auction_open_price')
                       or c == 'lifecycle'], strict=False)


def raw_session(inputs: Path, day: date, universe):
    from app.indicators.pipeline import _normalize_raw_volume_unit
    from app.services.prediction_workflow import _closed_prices, _frame

    raw = _normalize_raw_volume_unit(_closed_prices(_frame(inputs, 'kline_daily', day), day))
    # Existing daily partitions can contain compact and qualified copies.
    # Remove only identical closed observations; conflicting prices still fail.
    raw = sf._normalize_symbol_column(raw).filter(pl.col('symbol').is_in(universe)).select(
        *KEYS, 'open', 'high', 'low', 'close', 'volume', 'amount').unique()
    raw = pit._keys(raw, 'closed research prices')
    if raw.is_empty():
        raise ValueError('No closed raw prices in frozen universe')
    return raw.select(*KEYS, 'open', 'high', 'low', 'close', 'volume', 'amount')


def auxiliary_session(inputs: Path, raw):
    start, end = raw['date'].min(), raw['date'].max()
    frame = raw.with_columns(pl.col('close').alias('raw_close'),
                             pl.lit(None, dtype=pl.String).alias('industry'))
    frame = sf._attach_auxiliary_features(inputs, frame, start, end)
    # Here the auction is still native; training multiplied it once by the
    # absolute same-day factor, independently of OHLC's feature units.
    if '_auction_open_price' not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias('_auction_open_price'))
    return frame.with_columns(pl.col('_auction_open_price').alias('_raw_auction_open_price'))


def extend_panel(seed, raw_seed, factors_seed, names, industries, limits, events,
                 additions, raw_additions, dated_sources):
    """Rebuild references across the seam, keeping original dated observations."""
    if not additions:
        return seed, raw_seed, factors_seed, limits
    panel = pl.concat([clean_panel(seed), *[clean_panel(p) for p in additions]], how='diagonal_relaxed')
    raw = pl.concat([raw_seed.select(raw_additions[0].columns), *raw_additions], how='vertical_relaxed')
    factors = pit._keys(factors_seed, 'seed factors').select(*KEYS, 'adj_factor')
    turns, fresh_names, fresh_industries, fresh_limits, fresh_factors = [], [], [], [], []
    for day, captures in dated_sources:
        reference = dated_frame(captures['bak_basic'], 'bak_basic', day, ['name', 'industry'])
        fresh_names.append(pit.daily_reference_history(reference, 'name'))
        fresh_industries.append(pit.daily_reference_history(reference, 'industry'))
        fresh_limits.append(dated_frame(captures['stk_limit'], 'stk_limit', day, ['up_limit', 'down_limit']))
        fresh_factors.append(dated_frame(captures['adj_factor'], 'adj_factor', day, ['adj_factor']))
        turns.append(dated_frame(captures['daily_basic'], 'daily_basic', day, ['turnover_rate']))
    names = pl.concat([names, *fresh_names], how='vertical_relaxed')
    industries = pl.concat([industries, *fresh_industries], how='vertical_relaxed')
    limits = pl.concat([limits, *fresh_limits], how='vertical_relaxed')
    factors = pl.concat([factors, *fresh_factors], how='vertical_relaxed')
    panel, _ = pit.prepare_panel(panel, raw, names, industries, limits, lifecycle_events=events)
    turnover = pl.concat(turns, how='vertical_relaxed').rename({'turnover_rate': '_dated_turnover'})
    panel = panel.join(turnover, on=KEYS, how='left')
    fresh = pl.col('date') > seed['date'].max()
    invalid = fresh & ~pl.col('_pit_lifecycle_excluded') & ~(
        pl.col('_dated_turnover').is_finite() & (pl.col('_dated_turnover') >= 0)).fill_null(False)
    if panel.filter(invalid).height:
        raise ValueError('Missing/invalid dated turnover in appended chronology')
    panel = panel.with_columns(pl.when(fresh).then('_dated_turnover').otherwise('turnover_rate')
                               .alias('turnover_rate')).drop('_dated_turnover')
    return panel, raw, factors, limits
