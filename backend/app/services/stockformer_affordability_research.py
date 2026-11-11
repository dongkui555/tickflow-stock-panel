"""Known-day raw-price affordability shared by isolated fit/calibration/predict."""
from __future__ import annotations

import math

import polars as pl

from app.services.stockformer_decision_research import decision_pool
from app.strategy import paper

POLICY = {
    'version': 'known_day_raw_close_fixed_budget_stock_pool_v1',
    'price': 'same_signal_session_canonical_raw_close',
    'budget': 'initial_cash_divided_by_max_positions',
    'quantity': 'paper_qty_from_amount_after_minimum_commission_and_slippage',
    'minimum_buy_cost': 'one_stock_lot_with_buy_fee',
    'missing_raw_price': 'exclude_without_imputation_or_asof_fill',
    'future_prices_labels_predictions_or_account_nav_in_membership': False,
    'next_open_cash_quantity_limits_rechecked_by_original_matcher': True,
}
KEYS = ['symbol', 'date']
OUTPUT = ['_afford_raw_close', '_afford_qty', '_afford_one_lot_cost']


def affordable_pool(features, raw_prices, execution):
    """Apply the existing decision pool then check fixed known-day buying power.

    The reference sizing expression is identical to training labels and daily
    shadow orders. This does not certify next-session execution or predict cash
    remaining after other fills. Adjusted feature prices never size shares.
    """
    required = ['initial_cash', 'max_positions', 'slippage_bps', 'commission_pct', 'minimum_commission']
    if any(k not in execution for k in required):
        raise ValueError('Missing affordability execution parameters')
    if any(isinstance(execution[k], bool) or not math.isfinite(float(execution[k])) for k in required):
        raise ValueError('Invalid affordability execution parameters')
    capital = float(execution['initial_cash'])
    positions = float(execution['max_positions'])
    slip = float(execution['slippage_bps'])
    commission = float(execution['commission_pct'])
    if capital <= 0 or positions < 1 or not positions.is_integer() or slip < 0 or commission < 0:
        raise ValueError('Invalid cash, positions or cost policy')
    if execution['minimum_commission'] != paper.MIN_COMMISSION or paper.LOT_SIZE != 100:
        raise ValueError('Affordability policy requires the original stock lot and minimum commission')
    if any(c not in raw_prices.columns for c in [*KEYS, 'close']):
        raise ValueError('Missing canonical raw price keys or close')
    if features.schema.get('date') != pl.Date or raw_prices.schema.get('date') != pl.Date:
        raise ValueError('Expected canonical Date keys')
    if set(OUTPUT) & set(features.columns):
        raise ValueError('Affordability annotations already present; use canonical inputs')
    if raw_prices.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise ValueError('Duplicate canonical raw symbol/date')
    if raw_prices.select(pl.any_horizontal([pl.col(k).is_null() for k in KEYS]).any()).item():
        raise ValueError('Null canonical raw identity/date')
    base, original = decision_pool(features)
    raw = raw_prices.select(*KEYS,pl.col('close').cast(pl.Float64,strict=False).alias('_afford_raw_close'))
    joined = base.join(raw,on=KEYS,how='left',validate='1:1')
    budget = capital/positions
    buffer = max(0., budget-float(execution['minimum_commission']))

    def quantity(price):
        if price is None or not math.isfinite(price) or price <= 0:
            return 0
        return paper.qty_from_amount(buffer,price*(1+slip/10000))

    def minimum_cost(price):
        if price is None or not math.isfinite(price) or price <= 0:
            return None
        reference = price*(1+slip/10000)
        return paper.LOT_SIZE*reference+paper.buy_fee(paper.LOT_SIZE,reference,commission)

    annotated = joined.with_columns(
        pl.col('_afford_raw_close').map_elements(quantity,return_dtype=pl.Int64,skip_nulls=False).alias('_afford_qty'),
        pl.col('_afford_raw_close').map_elements(minimum_cost,return_dtype=pl.Float64,skip_nulls=False).alias('_afford_one_lot_cost'))
    valid_raw = pl.col('_afford_raw_close').is_finite() & (pl.col('_afford_raw_close')>0)
    affordable = ((pl.col('_afford_qty')>=paper.LOT_SIZE)&
                  (pl.col('_afford_one_lot_cost')<=budget)).fill_null(False)
    selected = annotated.filter(valid_raw.fill_null(False)&affordable).sort('date','symbol')
    report = dict(policy=dict(POLICY),execution={k:execution[k] for k in required},
        fixed_position_budget_yuan=budget,input_rows=features.height,decision_rows=base.height,
        accepted_rows=selected.height,excluded_by_affordability=base.height-selected.height,
        invalid_or_missing_raw_rows=annotated.filter(~valid_raw.fill_null(False)).height,
        one_lot_budget_rows=annotated.filter(valid_raw.fill_null(False)&~affordable).height,
        input_dates=features['date'].n_unique(),accepted_dates=selected['date'].n_unique(),
        original_decision_pool=original)
    return selected,report
