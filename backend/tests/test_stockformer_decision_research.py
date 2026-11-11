from datetime import date, timedelta

import polars as pl
import pytest

from app.services.stockformer_decision_research import decision_pool


def inputs():
    return pl.DataFrame([{'symbol': str(i), 'date': date(2024, 1, 2), 'amount': 1e8,
        'close': 10., 'price_chase_risk': .82, 'moneyflow_chase_risk': .88,
        'limit_down_flag': 0., 'broken_limit_up_flag': 0., 'target_5d_rel_net': i/100.,
        'pred_ret_5d': i/1000., 'pred_ret_raw_5d': i/1000.} for i in range(8)])


def test_pool_uses_only_known_inputs_and_fixed_boundaries():
    frame = inputs().with_columns(
        pl.when(pl.col('symbol') == '1').then(.82001).otherwise(pl.col('price_chase_risk')).alias('price_chase_risk'),
        pl.when(pl.col('symbol') == '2').then(.88001).otherwise(pl.col('moneyflow_chase_risk')).alias('moneyflow_chase_risk'),
        pl.when(pl.col('symbol') == '3').then(0.).when(pl.col('symbol') == '6')
        .then(float('inf')).otherwise(pl.col('amount')).alias('amount'),
        pl.when(pl.col('symbol') == '4').then(.5).otherwise(pl.col('limit_down_flag')).alias('limit_down_flag'),
        pl.when(pl.col('symbol') == '5').then(.5).otherwise(pl.col('broken_limit_up_flag')).alias('broken_limit_up_flag'),
        pl.when(pl.col('symbol') == '7').then(None).otherwise(pl.col('close')).alias('close'))
    selected, report = decision_pool(frame)
    assert selected['symbol'].to_list() == ['0']
    assert report['excluded_rows'] == 7


def test_labels_predictions_order_and_later_sessions_cannot_change_pool():
    frame = inputs()
    keys = ['symbol', 'date']
    original, _ = decision_pool(frame)
    changed = frame.with_columns(pl.lit(-999.).alias('target_5d_rel_net'),
        pl.lit(None, dtype=pl.Float64).alias('pred_ret_5d'), pl.lit(999.).alias('pred_ret_raw_5d'))
    selected, _ = decision_pool(changed.reverse())
    assert original.select(keys).equals(selected.select(keys))
    label_free, _ = decision_pool(frame.drop('target_5d_rel_net', 'pred_ret_5d', 'pred_ret_raw_5d'))
    assert original.select(keys).equals(label_free.select(keys))
    later = frame.with_columns((pl.col('date')+timedelta(days=5)).alias('date'), pl.lit(1.).alias('amount'))
    combined, _ = decision_pool(pl.concat([frame, later]))
    assert combined.filter(pl.col('date') == date(2024, 1, 2)).select(keys).equals(original.select(keys))


def test_missing_and_duplicate_pool_inputs_fail_closed():
    with pytest.raises(ValueError, match='Missing'):
        decision_pool(inputs().drop('amount'))
    with pytest.raises(ValueError, match='Duplicate'):
        decision_pool(pl.concat([inputs(), inputs()]))
    empty, report = decision_pool(inputs().clear())
    assert empty.is_empty() and report['accepted_rows'] == 0
