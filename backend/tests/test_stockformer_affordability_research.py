from datetime import date, timedelta

import polars as pl
import pytest

from app.services.stockformer_affordability_research import affordable_pool
from app.strategy import paper

EXECUTION=dict(initial_cash=3015.,max_positions=3,slippage_bps=0.,commission_pct=.0003,minimum_commission=5.)


def inputs():
    frame=pl.DataFrame([dict(symbol=str(i).zfill(6),date=date(2026,1,5),amount=1e8,close=100.,
        price_chase_risk=0.,moneyflow_chase_risk=0.,limit_down_flag=0.,broken_limit_up_flag=0.,
        target_5d_rel_net=100.,pred_ret_5d=10.) for i in range(4)])
    raw=frame.select('symbol','date').with_columns(pl.Series('close',[10.,10.0001,20.,5.]))
    return frame,raw


def test_raw_close_budget_slippage_and_minimum_fee_boundary():
    frame,raw=inputs()
    selected,report=affordable_pool(frame,raw,EXECUTION)
    assert selected['symbol'].to_list()==['000000','000003']
    assert selected['_afford_qty'].to_list()==[100,200]
    assert report['fixed_position_budget_yuan']==1005
    assert report['one_lot_budget_rows']==2
    slipped,_=affordable_pool(frame,raw,{**EXECUTION,'slippage_bps':5.})
    assert slipped['symbol'].to_list()==['000003']


def test_reference_quantity_is_identical_to_original_sizer():
    frame,raw=inputs()
    selected,_=affordable_pool(frame,raw,EXECUTION)
    for row in selected.to_dicts():
        assert row['_afford_qty']==paper.qty_from_amount(1000,row['_afford_raw_close'])
        assert row['_afford_one_lot_cost']<=1005


def test_adjusted_close_predictions_labels_future_prices_and_input_order_do_not_change_pool():
    frame,raw=inputs()
    before,_=affordable_pool(frame,raw,EXECUTION)
    changed=frame.with_columns(pl.lit(1e6).alias('close'),pl.lit(-999.).alias('target_5d_rel_net'),
        pl.lit(None,dtype=pl.Float64).alias('pred_ret_5d'))
    future=raw.with_columns((pl.col('date')+timedelta(days=1)).alias('date'),pl.lit(.01).alias('close'))
    after,_=affordable_pool(changed.reverse(),pl.concat([future,raw.reverse()]),EXECUTION)
    assert before.select('symbol','date', '_afford_qty').equals(after.select('symbol','date','_afford_qty'))
    without,_=affordable_pool(frame.drop('target_5d_rel_net','pred_ret_5d'),raw,EXECUTION)
    assert before.select('symbol','date').equals(without.select('symbol','date'))


def test_missing_stale_nan_or_nonpositive_raw_prices_are_excluded_without_fill():
    frame,raw=inputs()
    raw=raw.with_columns(pl.Series('close',[None,float('nan'),0.,5.]))
    result,report=affordable_pool(frame,raw,EXECUTION)
    assert result['symbol'].to_list()==['000003'] and report['invalid_or_missing_raw_rows']==3
    stale=raw.with_columns((pl.col('date')-timedelta(days=1)).alias('date'))
    result,report=affordable_pool(frame,stale,EXECUTION)
    assert result.is_empty() and report['invalid_or_missing_raw_rows']==4


def test_existing_hard_exclusions_still_apply():
    frame,raw=inputs()
    frame=frame.with_columns(pl.when(pl.col('symbol')=='000003').then(1.).otherwise(0.).alias('limit_down_flag'))
    result,report=affordable_pool(frame,raw,EXECUTION)
    assert result['symbol'].to_list()==['000000']
    assert report['original_decision_pool']['excluded_rows']==1


@pytest.mark.parametrize('patch',[{'initial_cash':float('nan')},{'max_positions':0},{'max_positions':1.5},
    {'commission_pct':-.1},{'minimum_commission':0.},{'slippage_bps':-1.}])
def test_invalid_execution_contract_fails_closed(patch):
    frame,raw=inputs()
    with pytest.raises(ValueError):
        affordable_pool(frame,raw,{**EXECUTION,**patch})


def test_missing_keys_duplicate_raw_and_double_annotation_are_rejected():
    frame,raw=inputs()
    with pytest.raises(ValueError,match='Duplicate'):
        affordable_pool(frame,pl.concat([raw,raw]),EXECUTION)
    with pytest.raises(ValueError,match='Missing canonical'):
        affordable_pool(frame,raw.drop('close'),EXECUTION)
    selected,_=affordable_pool(frame,raw,EXECUTION)
    with pytest.raises(ValueError,match='already present'):
        affordable_pool(selected,raw,EXECUTION)
