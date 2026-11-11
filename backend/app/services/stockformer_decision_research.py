"""Prediction-independent research pool shared by fitting and prediction."""
from __future__ import annotations

import polars as pl

POOL_POLICY = {
    'version': 'known_day_liquidity_chase_pool_v1',
    'amount_min_exclusive_yuan': 0.,
    'price_chase_risk_max': .82,
    'moneyflow_chase_risk_max': .88,
    'limit_down_flag_max_exclusive': .5,
    'broken_limit_up_flag_max_exclusive': .5,
    'missing_input': 'exclude',
    'prediction_or_future_label_in_membership': False,
}
REQUIRED = ('amount', 'close', 'price_chase_risk', 'moneyflow_chase_risk',
            'limit_down_flag', 'broken_limit_up_flag')


def decision_pool(frame):
    """Apply only known-day hard exclusions; final entry/market gates still apply.

    Positive known turnover is the liquidity prerequisite. No new fitted
    liquidity cutoff, model-dependent entry score, or future execution flag is
    used to select training rows. OHLC affordability uses raw prices later in
    the existing order/settlement engine, never adjusted close here.
    """
    missing = [column for column in ('symbol', 'date', *REQUIRED) if column not in frame.columns]
    if missing:
        raise ValueError('Missing decision pool inputs: ' + ', '.join(missing))
    if frame.select(pl.struct('symbol', 'date').is_duplicated().any()).item():
        raise ValueError('Duplicate decision pool symbol/date')
    rules = {
        'finite_known_day_inputs': pl.all_horizontal([pl.col(c).is_finite().fill_null(False) for c in REQUIRED]),
        'positive_amount': pl.col('amount') > POOL_POLICY['amount_min_exclusive_yuan'],
        'positive_close': pl.col('close') > 0.,
        'price_chase': pl.col('price_chase_risk') <= POOL_POLICY['price_chase_risk_max'],
        'moneyflow_chase': pl.col('moneyflow_chase_risk') <= POOL_POLICY['moneyflow_chase_risk_max'],
        'limit_down': pl.col('limit_down_flag') < POOL_POLICY['limit_down_flag_max_exclusive'],
        'broken_limit': pl.col('broken_limit_up_flag') < POOL_POLICY['broken_limit_up_flag_max_exclusive'],
    }
    accepted = pl.all_horizontal(list(rules.values())).fill_null(False)
    selected = frame.filter(accepted).sort('date', 'symbol')
    report = {'policy': dict(POOL_POLICY), 'input_rows': frame.height, 'accepted_rows': selected.height,
        'input_dates': frame['date'].n_unique(), 'accepted_dates': selected['date'].n_unique(),
        'excluded_rows': frame.height-selected.height,
        'independent_failure_counts': {name: int(frame.select((~expr.fill_null(False)).sum()).item())
                                       for name, expr in rules.items()}}
    return selected, report
