"""Offline selection alignment diagnostics; never connected to live inference."""
from __future__ import annotations

import bisect

import polars as pl

ARMS = {
    'strict': (),
    'without_entry_score': ('entry_score_62',),
    'without_risk_score': ('risk_score_38',),
    'hard_risk_only': ('entry_score_62', 'risk_score_38', 'entry_rank_p35'),
}
LABELS = ('target_5d_rel_net', 'target_5d_net', 'label_cost_5d')


def research_split(calendar, start, stop, *, validation_sessions=63, calibration_sessions=20):
    """Choose boundaries by sessions only, leaving six settlement sessions."""
    dates = sorted(set(calendar))
    cutoff = bisect.bisect_left(dates, stop)
    last = cutoff-6
    first = last-validation_sessions
    cal_first = first-6-calibration_sessions
    if cal_first <= 0 or dates[cal_first] <= start or validation_sessions <= 0:
        raise ValueError('Insufficient research split chronology')
    return dict(train_start=start, calibration_start=dates[cal_first],
        validation_start=dates[first], validation_dates=dates[first:last],
        label_cut_exclusive=stop, settlement_dates=dates[last:cutoff],
        validation_sessions=validation_sessions, calibration_sessions=calibration_sessions,
        embargo_sessions=6)


def entry_masks(frame):
    required = ['symbol', 'date', 'entry_score', 'risk_score', 'amplitude',
        'limit_down_flag', 'broken_limit_up_flag', 'price_chase_risk', 'moneyflow_chase_risk']
    missing = set(required)-set(frame.columns)
    if missing:
        raise ValueError('Missing entry inputs: '+','.join(sorted(missing)))
    if frame.select(pl.struct('symbol','date').is_duplicated().any()).item():
        raise ValueError('Duplicate research symbol/date')
    rules = {
        'entry_score_62': pl.col('entry_score') >= 62,
        'not_limit_down': pl.col('limit_down_flag') < .5,
        'not_broken_limit': pl.col('broken_limit_up_flag') < .5,
        'price_chase': pl.col('price_chase_risk').fill_null(0.) <= .82,
        'moneyflow_chase': pl.col('moneyflow_chase_risk').fill_null(0.) <= .88,
        'risk_score_38': pl.col('risk_score').fill_null(50.) >= 38,
        'amplitude_p95': pl.col('amplitude') <= pl.col('amplitude').quantile(.95).over('date'),
        'entry_rank_p35': (pl.col('entry_score').count().over('date') < 8) | (
            pl.col('entry_score').rank(method='average').over('date') >=
            pl.col('entry_score').count().over('date')*.35),
    }
    masks = frame.select('symbol','date',*[expr.fill_null(False).alias(name) for name,expr in rules.items()])
    return masks, list(rules)


def arm_signal(frame, arm):
    if arm not in ARMS:
        raise ValueError('Unregistered research arm')
    masks, names = entry_masks(frame)
    accepted = masks.select(pl.all_horizontal([pl.col(k) for k in names if k not in ARMS[arm]]).alias('_research_entry'))
    return frame.with_columns(accepted['_research_entry'].alias('_entry_rules_pass'))


def _mean_daily(frame, column):
    finite = frame.filter(pl.col(column).is_finite())
    return finite.group_by('date').agg(pl.col(column).mean())[column].mean() if finite.height else None


def quality(frame):
    """Equal-date overlapping label diagnostics, explicitly not portfolio returns."""
    valid = frame.filter(pl.col('target_5d_rel_net').is_finite() & pl.col('pred_ret_raw_5d').is_finite())
    daily_ic = valid.group_by('date').agg(pl.corr('pred_ret_raw_5d','target_5d_rel_net',method='spearman').alias('ic'))
    bias = valid.with_columns((pl.col('pred_ret_5d')-pl.col('target_5d_rel_net')).alias('_bias'))
    return dict(rows=frame.height, labeled_rows=valid.height, dates=frame['date'].n_unique(),
        daily_rank_ic=daily_ic.filter(pl.col('ic').is_finite())['ic'].mean(),
        mean_net_industry_label=_mean_daily(frame,'target_5d_rel_net'),
        mean_net_absolute_label=_mean_daily(frame,'target_5d_net'),
        mean_estimated_cost=_mean_daily(frame,'label_cost_5d'),
        date_equal_calibration_bias=_mean_daily(bias,'_bias'),
        negative_absolute_despite_positive_excess_rows=frame.filter(
            (pl.col('target_5d_rel_net')>0)&(pl.col('target_5d_net')<0)).height,
        kind='overlapping_conditional_labels_not_account_returns')


def diagnose(frame):
    masks, names = entry_masks(frame)
    baseline = arm_signal(frame,'strict')['_entry_rules_pass']
    if not baseline.equals(frame['_entry_rules_pass'].fill_null(False)):
        raise ValueError('Research predicate differs from frozen entry result')
    n = pl.col('pred_ret_raw_5d').count().over('date')
    top = pl.col('pred_ret_raw_5d').rank(method='average',descending=True).over('date') <= pl.max_horizontal(n*.2,pl.lit(1.))
    groups = {
        'all': pl.lit(True), 'raw_top20': top,
        'forecast_net': pl.col('_forecast_pass'),
        'forecast_net_entry': pl.col('_forecast_pass')&pl.col('_entry_rules_pass'),
        'forecast_net_entry_rejected': pl.col('_forecast_pass')&~pl.col('_entry_rules_pass'),
        'all_gates': pl.col('_forecast_pass')&pl.col('_entry_rules_pass')&pl.col('_market_gate_pass'),
    }
    high = frame.select(pl.col('_forecast_pass').fill_null(False))['_forecast_pass']
    failure = {}
    for name in names:
        failed = ~masks[name]
        only = masks.select(pl.all_horizontal([pl.col(k) for k in names if k!=name])).to_series()
        failure[name] = dict(independent_high_failures=int((high&failed).sum()),
                            sole_high_blocker=int((high&failed&only).sum()))
    high_frame = frame.filter('_forecast_pass')
    correlations = high_frame.group_by('date').agg(*[
        pl.corr('pred_ret_raw_5d',c,method='spearman').alias(c)
        for c in ['entry_score','risk_score','technical_score','fund_score','sector_score']])
    return dict(rows=frame.height, dates=frame['date'].n_unique(),
        groups={name:quality(frame.filter(expr.fill_null(False))) for name,expr in groups.items()},
        failures=failure, high_score_component_rank_correlation={
            c:correlations.filter(pl.col(c).is_finite())[c].mean() for c in correlations.columns if c!='date'},
        market_states=frame.group_by('market_gate').agg(pl.len().alias('rows'),
            pl.col('_forecast_pass').sum().alias('forecast_net'),
            (pl.col('_forecast_pass')&pl.col('_entry_rules_pass')).sum().alias('with_entry'),
            pl.col('_market_gate_pass').sum().alias('market_pass')).sort('market_gate').to_dicts())


def assert_label_boundaries(train, calibration, validation, split):
    """Reject cross-boundary labels before fitting or reporting a validation."""
    for frame,boundary in [(train,split['calibration_start']),
                           (calibration,split['validation_start']),
                           (validation,split['label_cut_exclusive'])]:
        if frame.is_empty() or frame.filter(pl.col('label_end_5d').is_null() |
            (pl.col('label_end_5d')>=boundary)).height:
            raise ValueError('Empty or overlapping label boundary')
    if train['date'].n_unique()<100 or calibration['date'].n_unique()<20:
        raise ValueError('Insufficient isolated fitting history')
    if validation['date'].n_unique()!=split['validation_sessions']:
        raise ValueError('Incomplete validation sessions')
    if train['date'].max()>=calibration['date'].min() or calibration['date'].max()>=validation['date'].min():
        raise ValueError('Overlapping signal dates')
    return dict(train_dates=train['date'].n_unique(),train_rows=train.height,
        train_max_label_end=train['label_end_5d'].max(),calibration_dates=calibration['date'].n_unique(),
        calibration_rows=calibration.height,calibration_max_label_end=calibration['label_end_5d'].max(),
        validation_dates=validation['date'].n_unique(),validation_rows=validation.height,
        validation_max_label_end=validation['label_end_5d'].max())
