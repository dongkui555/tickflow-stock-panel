from datetime import date, timedelta

import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_ranking_research as research


def sample(days=1, stocks=20):
    return pl.DataFrame([dict(symbol=str(i),date=date(2026,1,5)+timedelta(days=d),
        score=float(i),target_5d_rel_net=i/100.,target_5d_net=i/100.,entry_score=70.,risk_score=50.,
        amplitude=.02,limit_down_flag=0.,broken_limit_up_flag=0.,price_chase_risk=0.,
        moneyflow_chase_risk=0.,_forecast_pass=True,_market_gate_pass=True)
        for d in range(days) for i in range(stocks)])


def with_entry(frame):
    rank=pl.col('entry_score').rank(method='average').over('date')
    return frame.with_columns((sf._entry_eligible_expr(frame)&(rank>=pl.len().over('date')*.35)).alias('_entry_rules_pass'))


def test_constant_scores_have_no_rank_groups_or_correlations():
    frame=sample().with_columns(pl.lit(.009163635198125162).alias('score'))
    daily=research.daily_quality(frame,'score')
    assert daily['rank_ic'].item() is None
    assert daily['top_rows'].item()==daily['bottom_rows'].item()==0
    assert daily['top_target'].item() is None
    result=research.quality_summary(daily,frame['date'].unique().to_list())
    assert result['rank_ic_mean'] is None and result['rank_ic_block_ci95'] is None


def test_labels_are_not_used_to_rank_and_missing_top_labels_do_not_promote_lower_scores():
    frame=sample().with_columns(pl.when(pl.col('score')==19).then(None).otherwise(pl.col('target_5d_rel_net')).alias('target_5d_rel_net'))
    daily=research.daily_quality(frame,'score')
    assert daily['top_rows'].item()==4 and daily['top_labeled_rows'].item()==3
    assert daily['top_target'].item()==pytest.approx(.17)
    altered=frame.with_columns(pl.lit(-999.).alias('target_5d_rel_net'))
    cols=['symbol','date','_rank_top','_rank_bottom']
    assert research.rank_groups(frame,'score').select(cols).equals(research.rank_groups(altered,'score').select(cols))
    later=frame.with_columns((pl.col('date')+timedelta(days=1)).alias('date'),pl.lit(-999.).alias('score'))
    joined=research.rank_groups(pl.concat([frame,later]),'score').filter(pl.col('date')==date(2026,1,5))
    assert joined.select(cols).equals(research.rank_groups(frame,'score').select(cols))


def test_equal_date_statistics_and_nonoverlap_offsets_use_actual_calendar_positions():
    one=sample()
    two=sample(stocks=200).with_columns(pl.lit(date(2026,1,7)).alias('date'),(-pl.col('score')).alias('score'))
    daily=research.daily_quality(pl.concat([one,two]),'score')
    calendar=[date(2026,1,5)+timedelta(days=i) for i in range(3)]
    result=research.quality_summary(daily,calendar)
    assert result['rank_ic_mean']==pytest.approx(0.)
    assert result['rank_ic_positive_fraction']==pytest.approx(.5)
    assert result['nonoverlapping_offsets'][0]['rank_ic_days']==1
    assert result['nonoverlapping_offsets'][1]['rank_ic_days']==0
    assert result['nonoverlapping_offsets'][2]['rank_ic_days']==1
    assert result['rank_ic_block_ci95'] is None


def test_block_intervals_are_reproducible_and_constant_targets_have_undefined_ic():
    frame=sample(days=21)
    daily=research.daily_quality(frame,'score')
    calendar=frame['date'].unique().sort().to_list()
    first=research.quality_summary(daily,calendar)
    second=research.quality_summary(daily,calendar)
    assert first==second and first['rank_ic_mean']==pytest.approx(1.)
    assert first['rank_ic_block_ci95']==pytest.approx([1.,1.])
    constant=research.daily_quality(frame.with_columns(pl.lit(.01).alias('target_5d_rel_net')),'score')
    assert constant['rank_ic'].null_count()==21


def test_refit_agreement_measures_same_cross_section_and_rejects_mismatched_identity():
    original=sample().select('symbol','date','score')
    changed=original.with_columns((-pl.col('score')).alias('score')).reverse()
    result=research.refit_agreement(original,changed,'score')
    assert result['daily'][0]['rank_correlation']==pytest.approx(-1.)
    assert result['daily'][0]['top_overlap_fraction']==0
    assert result['daily'][0]['top_jaccard']==0
    with pytest.raises(ValueError,match='universe'):
        research.refit_agreement(original,changed.head(19),'score')
    with pytest.raises(ValueError,match='Duplicate'):
        research.refit_agreement(original,pl.concat([changed,changed]),'score')


def test_entry_attribution_distinguishes_simultaneous_and_sole_blockers():
    frame=sample().with_columns(pl.when(pl.col('score')==19).then(61.).otherwise(pl.col('entry_score')).alias('entry_score'),
        pl.when(pl.col('score')==18).then(37.).otherwise(pl.col('risk_score')).alias('risk_score'))
    frame=with_entry(frame)
    result=research.entry_alignment(frame,'score')
    high=result['populations']['raw_top20']['rules']
    assert high['entry_score_62']['failed']['rows']==1
    assert high['entry_score_62']['sole_blocker_rows']==0
    assert high['risk_score_38']['failed']['rows']==1
    assert high['risk_score_38']['sole_blocker_rows']==1
    changed=frame.with_columns(pl.lit(-999.).alias('target_5d_rel_net'))
    other=research.entry_alignment(changed,'score')
    for rule in high:
        assert high[rule]['failed']['rows']==other['populations']['raw_top20']['rules'][rule]['failed']['rows']
    with pytest.raises(ValueError,match='predicate'):
        research.entry_alignment(frame.with_columns((~pl.col('_entry_rules_pass')).alias('_entry_rules_pass')),'score')


def test_duplicate_and_unknown_calendar_inputs_fail_closed():
    frame=sample()
    with pytest.raises(ValueError,match='Duplicate'):
        research.rank_groups(pl.concat([frame,frame]),'score')
    daily=research.daily_quality(frame,'score')
    with pytest.raises(ValueError,match='calendar'):
        research.quality_summary(daily,[date(2026,1,6)])


def test_empty_groups_keep_undefined_metrics_and_zero_counts():
    daily=research.daily_quality(sample().clear(),'score')
    result=research.quality_summary(daily,[])
    assert result['rows']==result['dates']==result['rank_ic_days']==0
    assert result['rank_ic_mean'] is None
    assert research.label_summary(sample().clear())['absolute_net_daily_mean'] is None
