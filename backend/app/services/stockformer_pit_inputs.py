"""Opt-in research inputs with explicit historical availability; no production I/O.

Unknown references remain null and cannot become training or trading rows.
Existing frozen model generators and their accounts deliberately remain intact.
"""
from __future__ import annotations

from datetime import date

import polars as pl

from app.services import stockformer as sf

VERSION = 'known_name_industry_authoritative_limits_lifecycle_v2'
SIGNALS = ('signal_limit_up', 'signal_limit_down', 'signal_broken_limit_up',
           'signal_limit_down_recovery')


def _date(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.String).str.strip_chars()
    return pl.coalesce(text.str.strptime(pl.Date, '%Y-%m-%d', strict=False),
                       text.str.strptime(pl.Date, '%Y%m%d', strict=False))


def _keys(frame: pl.DataFrame, label: str, *, dated: bool = True) -> pl.DataFrame:
    if 'symbol' not in frame.columns and 'ts_code' in frame.columns:
        frame = frame.rename({'ts_code': 'symbol'})
    if dated and 'date' not in frame.columns and 'trade_date' in frame.columns:
        frame = frame.rename({'trade_date': 'date'})
    required = ['symbol', 'date'] if dated else ['symbol']
    if any(c not in frame.columns for c in required):
        raise ValueError(f'Missing {label} identity/date keys')
    frame = sf._normalize_symbol_column(frame)
    if dated:
        frame = frame.with_columns(_date('date').alias('date'))
    if frame.select(pl.any_horizontal([pl.col(c).is_null() for c in required]).any()).item():
        raise ValueError(f'Null {label} identity/date keys')
    if frame.filter(pl.col('symbol') == '').height:
        raise ValueError(f'Empty {label} symbol')
    if dated and frame.select(pl.struct(required).is_duplicated().any()).item():
        raise ValueError(f'Duplicate {label} identity/date keys')
    return frame


def name_history(frame: pl.DataFrame) -> pl.DataFrame:
    """Tushare namechange: announcement and inclusive effective interval."""
    frame = _keys(frame, 'name history', dated=False)
    if {'valid_from', 'valid_to', 'known_on', 'source'}.issubset(frame.columns):
        return frame.select('symbol', pl.col('name').alias('value'),
                            'valid_from', 'valid_to', 'known_on', 'source')
    for c in ['name', 'start_date', 'ann_date']:
        if c not in frame.columns:
            raise ValueError(f'Missing name history {c}')
    if 'end_date' not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias('end_date'))
    return frame.select('symbol', pl.col('name').cast(pl.String).alias('value'),
        _date('start_date').alias('valid_from'), _date('end_date').alias('valid_to'),
        _date('ann_date').alias('known_on'), pl.lit('tushare.namechange').alias('source'))


def daily_reference_history(frame: pl.DataFrame, field: str) -> pl.DataFrame:
    """Provider historical daily snapshot, with its own industry taxonomy.

    A dated bak_basic row is used only on that trade date. Its provenance must
    disclose that retrieval now does not establish the original archive vintage.
    The source's daily industry classification is never mixed with THS or SW.
    """
    if field not in {'name', 'industry'}:
        raise ValueError('Unsupported daily reference field')
    frame = _keys(frame, 'daily reference')
    if field not in frame.columns:
        raise ValueError('Missing daily reference '+field)
    return frame.select('symbol', pl.col(field).cast(pl.String),
        pl.col('date').alias('valid_from'), pl.col('date').alias('valid_to'),
        pl.col('date').alias('known_on'),
        pl.lit('tushare.bak_basic:historical_daily_snapshot').alias('source'))


def industry_snapshot(frame: pl.DataFrame, captured_on: date, *, source: str) -> pl.DataFrame:
    """A current snapshot becomes evidence only from its actual capture date."""
    frame = _keys(frame, 'industry snapshot', dated=False)
    column = next((c for c in ['industry', '所属同花顺行业'] if c in frame.columns), None)
    if column is None or not source.strip():
        raise ValueError('Missing industry snapshot value/source')
    return frame.select('symbol', pl.col(column).cast(pl.String).alias('industry'),
        pl.lit(captured_on).alias('valid_from'), pl.lit(None, dtype=pl.Date).alias('valid_to'),
        pl.lit(captured_on).alias('known_on'), pl.lit(source).alias('source'))


def _attach_history(panel, history, field):
    history = _keys(history, field+' history', dated=False)
    required = ['value', 'valid_from', 'valid_to', 'known_on', 'source']
    if any(c not in history.columns for c in required):
        raise ValueError(f'Missing {field} availability metadata')
    history = history.select('symbol', *required).with_columns(
        *[_date(c).alias(c) for c in ['valid_from', 'valid_to', 'known_on']],
        pl.col('value').cast(pl.String).str.strip_chars(),
        pl.col('source').cast(pl.String).str.strip_chars())
    history = history.filter(pl.col('valid_from').is_not_null() & pl.col('known_on').is_not_null()
        & pl.col('value').is_not_null() & (pl.col('value') != '') & (pl.col('source') != '')
        & (pl.col('valid_to').is_null() | (pl.col('valid_to') >= pl.col('valid_from')))).unique()
    keys = panel.select('symbol', 'date')
    is_point = pl.col('valid_from') == pl.col('valid_to')
    points = history.filter(is_point).rename({'valid_from': 'date'}).drop('valid_to')
    points = keys.join(points, on=['symbol', 'date']).filter(pl.col('known_on') <= pl.col('date'))
    intervals = history.filter(~is_point.fill_null(False))
    matches = keys.join(intervals, on='symbol').filter(
        (pl.col('date') >= pl.col('valid_from')) & (pl.col('date') >= pl.col('known_on'))
        & (pl.col('valid_to').is_null() | (pl.col('date') <= pl.col('valid_to'))))
    matches = pl.concat([points.select('symbol', 'date', 'value', 'source'),
                         matches.select('symbol', 'date', 'value', 'source')])
    values = matches.group_by('symbol', 'date').agg(pl.col('value').n_unique().alias('_values'),
        pl.col('value').first().alias('_value'), pl.col('source').unique().sort().str.join('|').alias('_source'))
    values = values.with_columns(
        pl.when(pl.col('_values') == 1).then(pl.col('_value')).otherwise(None).alias(field),
        (pl.col('_values') > 1).alias(f'_pit_{field}_conflict'),
        pl.col('_source').alias(f'_pit_{field}_source')).drop('_values', '_value', '_source')
    panel = panel.drop(field, strict=False).join(values, on=['symbol', 'date'], how='left')
    return panel.with_columns(pl.col(f'_pit_{field}_conflict').fill_null(False),
                             pl.col(field).is_not_null().alias(f'_pit_{field}_known'))


def attach_lifecycle_exclusions(panel, events=None):
    """Keep real bars; exclude only dated, announced liquidation intervals.

    A name suffix or today's delisted status is not evidence for a historical
    exclusion. Missing industry/limits on an excluded liquidation row remain
    unknown and never become part of an ordinary-stock feature cross section.
    """
    panel = _keys(panel, 'lifecycle panel')
    if events is None or events.is_empty():
        return panel.with_columns(pl.lit(False).alias('_pit_lifecycle_excluded'))
    events = _keys(events, 'lifecycle events', dated=False)
    if 'state' not in events.columns or events.filter(~pl.col('state').is_in(['delisting_period'])).height:
        raise ValueError('Unsupported historical lifecycle state')
    events = events.rename({'state': 'value'})
    panel = _attach_history(panel, events, 'lifecycle')
    if panel.filter('_pit_lifecycle_conflict').height:
        raise ValueError('Conflicting historical lifecycle states')
    return panel.with_columns((pl.col('lifecycle') == 'delisting_period').fill_null(False)
        .alias('_pit_lifecycle_excluded'))


def announced_liquidation_events(name_changes):
    """Use an explicit change reason and announcement, never a name heuristic."""
    history = name_history(name_changes).rename({'value': 'name'})
    raw = _keys(name_changes, 'name change reasons', dated=False)
    if 'change_reason' not in raw.columns:
        raise ValueError('Missing historical lifecycle change reason')
    raw = raw.with_columns(_date('start_date').alias('valid_from')).select('symbol', 'valid_from', 'change_reason')
    result = history.join(raw, on=['symbol', 'valid_from']).filter(
        (pl.col('change_reason') == '退市整理期') & pl.col('known_on').is_not_null()
        & pl.col('valid_from').is_not_null())
    return result.select('symbol', 'name', 'valid_from', 'valid_to', 'known_on',
        pl.lit('tushare.namechange:announced_delisting_period').alias('source'),
        pl.lit('delisting_period').alias('state')).unique()


def prepare_panel(panel, raw_prices, names, industries, limits, *, lifecycle_events=None):
    """Join known references and compute flags from raw OHLC and same-day limits.

    Industry records require their original taxonomy, validity interval and an
    evidenced known_on. Effective dates alone are not publication evidence.
    No latest name/industry, calculated price or missing-limit fallback is used.
    """
    panel = attach_lifecycle_exclusions(panel, lifecycle_events)
    raw = _keys(raw_prices, 'raw prices')
    authority = _keys(limits, 'limits')
    panel = _attach_history(panel, name_history(names), 'name')
    panel = _attach_history(panel, industries.rename({'industry': 'value'}), 'industry')
    raw_columns = ['open', 'high', 'low', 'close', 'volume']
    if any(c not in raw.columns for c in raw_columns) or any(c not in authority.columns for c in ['up_limit', 'down_limit']):
        raise ValueError('Missing raw OHLC/volume or authoritative daily limits')
    panel = panel.drop([*SIGNALS, 'up_limit', 'down_limit'], strict=False)
    panel = panel.join(raw.select('symbol', 'date', *[pl.col(c).cast(pl.Float64).alias('_raw_'+c)
        for c in raw_columns]), on=['symbol', 'date'], how='left').join(
            authority.select('symbol', 'date', pl.col('up_limit', 'down_limit').cast(pl.Float64)),
            on=['symbol', 'date'], how='left')
    raw_valid = (pl.all_horizontal([pl.col('_raw_'+c).is_finite() & (pl.col('_raw_'+c) > 0)
                    for c in raw_columns])
        & (pl.col('_raw_high') >= pl.max_horizontal('_raw_open', '_raw_low', '_raw_close'))
        & (pl.col('_raw_low') <= pl.min_horizontal('_raw_open', '_raw_high', '_raw_close')))
    limits_valid = (pl.col('up_limit').is_finite() & pl.col('down_limit').is_finite()
        & (pl.col('down_limit') > 0) & (pl.col('up_limit') > pl.col('down_limit'))
        & (pl.col('_raw_low') >= pl.col('down_limit')-.005)
        & (pl.col('_raw_high') <= pl.col('up_limit')+.005))
    panel = panel.with_columns((raw_valid & limits_valid).fill_null(False).alias('_pit_limits_known'))
    up = pl.col('_raw_close') >= pl.col('up_limit')-.005
    down = pl.col('_raw_close') <= pl.col('down_limit')+.005
    for column, expression in zip(SIGNALS, [up, down,
        (pl.col('_raw_high') >= pl.col('up_limit')-.005) & ~up,
        (pl.col('_raw_low') <= pl.col('down_limit')+.005) & ~down], strict=True):
        panel = panel.with_columns(pl.when(pl.col('_pit_limits_known')).then(expression)
            .otherwise(None).cast(pl.Boolean).alias(column))
    sessions = panel.select('date').unique().sort('date').with_row_index('_pit_session')
    panel = panel.sort('symbol', 'date').join(sessions, on='date', how='left')
    gap = (pl.col('_pit_session')-pl.col('_pit_session').shift(1).over('symbol') != 1).fill_null(True)
    for signal, output in [('signal_limit_up', 'consecutive_limit_ups'),
                           ('signal_limit_down', 'consecutive_limit_downs')]:
        panel = panel.with_columns((~pl.col(signal).fill_null(False) | gap).cast(pl.UInt32)
            .cum_sum().over('symbol').alias('_pit_streak_group'))
        panel = panel.with_columns(pl.when(pl.col('_pit_limits_known')).then(
            pl.col(signal).fill_null(False).cast(pl.UInt32).cum_sum().over('symbol', '_pit_streak_group'))
            .otherwise(None).cast(pl.UInt32).alias(output))
    panel = panel.drop('_pit_session', '_pit_streak_group')
    panel = panel.with_columns(pl.when(pl.col('_pit_name_known')).then(sf._special_treatment_expr())
        .otherwise(None).cast(pl.Boolean).alias('_pit_is_st'))
    panel = panel.with_columns((pl.col('_pit_name_known') & pl.col('_pit_industry_known')
        & pl.col('_pit_limits_known') & ~pl.col('_pit_is_st')
        & ~pl.col('_pit_lifecycle_excluded')).alias('_pit_input_ready'))
    report = {'version': VERSION, 'input_rows': panel.height,
        'eligible_rows': panel.filter('_pit_input_ready').height,
        'st_excluded_rows': panel.filter('_pit_name_known', '_pit_is_st').height,
        'lifecycle_excluded_rows': panel['_pit_lifecycle_excluded'].sum()}
    for key in ['name', 'industry', 'limits']:
        report[key+'_unknown_rows'] = panel.filter(~pl.col('_pit_'+key+'_known')).height
        report[key+'_required_unknown_rows'] = panel.filter(~pl.col('_pit_lifecycle_excluded')
            & ~pl.col('_pit_'+key+'_known')).height
    for key in ['name', 'industry']:
        report[key+'_conflict_rows'] = panel.filter('_pit_'+key+'_conflict').height
    report['nonzero_flags'] = {c: panel[c].sum() or 0 for c in SIGNALS}
    return panel.sort('symbol', 'date'), report


def build_features(panel):
    """Keep full price chronology for rolling features; reject unknown outputs."""
    if '_pit_input_ready' not in panel.columns or not panel.filter('_pit_input_ready').height:
        raise ValueError('No eligible point-in-time research inputs')
    if '_pit_lifecycle_excluded' in panel.columns:
        panel = panel.filter(~pl.col('_pit_lifecycle_excluded').fill_null(False))
    known = pl.all_horizontal([pl.col('_pit_'+key+'_known') for key in ['name', 'industry', 'limits']])
    if panel.filter(~known.fill_null(False)).height:
        raise ValueError('Incomplete point-in-time reference coverage; rolling/industry context is unknown')
    frame = sf._feature_frame(panel, include_targets=False)
    metadata = panel.select('symbol', 'date', 'name', '_pit_name_known', '_pit_is_st', '_pit_input_ready')
    return frame.join(metadata, on=['symbol', 'date'], how='left').filter('_pit_input_ready')


def build_labels(panel, raw_prices, *, execution, horizons=(5,), price_panel=None):
    """Rebuild labels, then apply the unchanged minute and decision-pool rules.

    The industry benchmark is formed before minute/entry filtering. Exact
    ledger fees and next-session opens come from the existing target builder;
    these conditional labels do not assert that a portfolio could fill them.
    """
    from app.services import stockformer_training as training
    from app.services.stockformer_decision_research import decision_pool

    features = build_features(panel)
    labels = training.paper_execution_targets(features, panel if price_panel is None else price_panel, raw_prices,
        execution=execution, horizons=horizons)
    minute_ready, minute_report = sf.prepare_training_features(labels, 'minute_enhanced')
    selected, pool_report = decision_pool(minute_ready)
    report = {'version': VERSION, 'cost_version': training.PAPER_COST_TARGET_VERSION,
        'base_feature_rows': features.height, 'label_rows': labels.height,
        'minute_coverage': minute_report, 'decision_pool': pool_report,
        'label_benchmark': 'signal_date_industry_mean_before_minute_and_decision_pool_filters',
        'conditional_labels_are_not_portfolio_returns': True, 'horizons': {}}
    for horizon in horizons:
        target = f'target_{horizon}d_rel_net'
        report['horizons'][str(horizon)] = {'finite_net_industry_labels': selected[target].is_finite().sum(),
            'unmatured_rows': selected[f'label_end_{horizon}d'].is_null().sum(),
            'statuses': selected.group_by(f'label_paper_status_{horizon}d').len().to_dicts()}
    return labels, selected, report


def execution_limit_map(raw_prices, limits, calendar):
    """Validate all replay bars before account creation; never guess a limit."""
    raw = _keys(raw_prices, 'execution raw prices')
    limits = _keys(limits, 'execution limits')
    if any(c not in limits.columns for c in ['up_limit', 'down_limit']):
        raise ValueError('Missing authoritative execution limit columns')
    limits = limits.filter(pl.col('up_limit').is_finite() & pl.col('down_limit').is_finite()
        & (pl.col('down_limit') > 0) & (pl.col('up_limit') > pl.col('down_limit')))
    required = raw.filter(pl.col('date').is_in(calendar) & (pl.col('volume') > 0)).select('symbol', 'date')
    missing = required.join(limits.select('symbol', 'date'), on=['symbol', 'date'], how='anti')
    if missing.height:
        raise ValueError(f'Missing authoritative execution limits: {missing.height} bars')
    return {(r['symbol'], r['date']): (float(r['up_limit']), float(r['down_limit']))
        for r in limits.select('symbol', 'date', 'up_limit', 'down_limit').to_dicts()}
