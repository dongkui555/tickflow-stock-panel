from datetime import date, timedelta

import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_selection_research as research


def signals():
    frame=pl.DataFrame([dict(symbol=str(i),date=date(2026,1,5),entry_score=70.,risk_score=50.,
        amplitude=.02,limit_down_flag=0.,broken_limit_up_flag=0.,price_chase_risk=0.,
        moneyflow_chase_risk=0.,pred_ret_raw_5d=i/100.,pred_ret_5d=.01,
        _forecast_pass=True,_market_gate_pass=True,market_gate='open',technical_score=50.,fund_score=50.,sector_score=50.,
        target_5d_rel_net=.02,target_5d_net=-.01,label_cost_5d=.005) for i in range(10)])
    frame=frame.with_columns(pl.when(pl.col('symbol')=='0').then(61.).otherwise(pl.col('entry_score')).alias('entry_score'),
        pl.when(pl.col('symbol')=='1').then(37.).otherwise(pl.col('risk_score')).alias('risk_score'))
    rank=pl.col('entry_score').rank(method='average').over('date')
    return frame.with_columns((sf._entry_eligible_expr(frame)&(rank>=pl.len().over('date')*.35)).alias('_entry_rules_pass'))


def test_baseline_matches_actual_frozen_entry_and_ablations_remove_only_named_conditions():
    frame=signals()
    assert research.arm_signal(frame,'strict')['_entry_rules_pass'].equals(frame['_entry_rules_pass'])
    entry=research.arm_signal(frame,'without_entry_score')
    # Symbol 0 still fails the unchanged relative entry rank.
    assert not entry.filter(pl.col('symbol')=='0')['_entry_rules_pass'].item()
    risk=research.arm_signal(frame,'without_risk_score')
    assert risk.filter(pl.col('symbol')=='1')['_entry_rules_pass'].item()
    hard=research.arm_signal(frame,'hard_risk_only')
    assert hard['_entry_rules_pass'].all()
    assert hard['_market_gate_pass'].equals(frame['_market_gate_pass'])


def test_future_labels_and_future_sessions_cannot_change_arm_membership():
    original=signals()
    changed=original.with_columns(pl.lit(-999.).alias('target_5d_rel_net'))
    later=changed.with_columns((pl.col('date')+timedelta(days=1)).alias('date'),pl.lit(0.).alias('entry_score'))
    for arm in research.ARMS:
        expected=research.arm_signal(original,arm).select('symbol','date','_entry_rules_pass')
        assert expected.equals(research.arm_signal(changed,arm).select(expected.columns))
        joined=research.arm_signal(pl.concat([changed,later]),arm).filter(pl.col('date')==date(2026,1,5))
        assert expected.equals(joined.select(expected.columns))


def test_diagnostics_distinguish_joint_and_sole_blockers_and_reject_changed_original_flag():
    frame=signals()
    report=research.diagnose(frame)
    assert report['failures']['entry_score_62']==dict(independent_high_failures=1,sole_high_blocker=0)
    assert report['failures']['risk_score_38']==dict(independent_high_failures=1,sole_high_blocker=1)
    assert report['groups']['all']['negative_absolute_despite_positive_excess_rows']==10
    with pytest.raises(ValueError,match='differs'):
        research.diagnose(frame.with_columns((~pl.col('_entry_rules_pass')).alias('_entry_rules_pass')))


def test_equal_date_quality_is_not_row_weighted_and_empty_groups_are_valid():
    frame=signals().with_columns(pl.lit(.2).alias('target_5d_rel_net'))
    next_day=frame.head(1).with_columns((pl.col('date')+timedelta(days=1)).alias('date'),
                                      pl.lit(0.).alias('target_5d_rel_net'))
    assert research.quality(pl.concat([frame,next_day]))['mean_net_industry_label']==pytest.approx(.1)
    result=research.quality(frame.clear())
    assert result['rows']==0 and result['daily_rank_ic'] is None
    assert result['mean_net_industry_label'] is None


def test_research_boundaries_are_purged_and_training_validation_are_separate():
    calendar=[date(2025,1,1)+timedelta(days=i) for i in range(251)]
    split=research.research_split(calendar,calendar[0],calendar[250])
    assert len(split['validation_dates'])==63 and len(split['settlement_dates'])==6
    def sample(start,stop):
        return pl.DataFrame([dict(date=calendar[i],label_end_5d=calendar[i+6]) for i in range(start,stop)])
    cal_index=calendar.index(split['calibration_start'])
    val_index=calendar.index(split['validation_start'])
    train,cal,valid=sample(0,cal_index-6),sample(cal_index,val_index-6),sample(val_index,244)
    assert research.assert_label_boundaries(train,cal,valid,split)['calibration_dates']==20
    with pytest.raises(ValueError,match='boundary'):
        research.assert_label_boundaries(train,cal.with_columns(pl.lit(split['validation_start']).alias('label_end_5d')),valid,split)


def test_invalid_inputs_fail_closed():
    frame=signals()
    with pytest.raises(ValueError,match='Missing'):
        research.entry_masks(frame.drop('amplitude'))
    with pytest.raises(ValueError,match='Duplicate'):
        research.entry_masks(pl.concat([frame,frame]))
    with pytest.raises(ValueError,match='Unregistered'):
        research.arm_signal(frame,'pick_best_threshold')
