from datetime import date, timedelta

import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_necessary_oof_research as research
from app.services import stockformer_portfolio as portfolio

EXECUTION=dict(initial_cash=10000.,max_positions=3,slippage_bps=5.,commission_pct=.0003,minimum_commission=5.)


def inputs(days=1,stocks=30):
    rows=[]
    for d in range(days):
        for i in range(stocks):
            row={c:float(i)/100 for c in sf.training_feature_columns('minute_enhanced')}
            row.update(symbol=f'{i:06d}',date=date(2024,1,1)+timedelta(days=d),name='普通股票',
                _pit_name_known=True,_pit_input_ready=True,industry='测试行业',amount=1e8*(i+1),close=10.,
                turnover_rate=5.,price_chase_risk=0.,moneyflow_chase_risk=0.,limit_down_flag=0.,broken_limit_up_flag=0.,
                minute_data_available=1.,minute_bar_count=240.,minute_session_coverage=1.,
                target_5d_net=.01,target_5d_rel_net=.005,label_end_5d=date(2024,1,1)+timedelta(days=d+6))
            rows.append(row)
    features=pl.DataFrame(rows)
    raw=features.select('symbol','date',pl.lit(10.).alias('close'))
    return features,raw


def gate(root,frame):
    return frame.with_columns(pl.lit('open').alias('market_gate'),pl.lit(0.).alias('market_buy_threshold_add'),
        pl.lit(True).alias('market_allow_new_position'),pl.lit(1.).alias('market_position_multiplier'),
        pl.lit(70.).alias('market_regime_score'))


def test_pool_and_reference_scores_ignore_predictions_and_future_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(sf,'_attach_market_gate_for_oos',gate)
    frame,raw=inputs()
    reference,report=research.reference_pool(tmp_path,frame,raw,EXECUTION)
    assert report['reference_rows']==30 and 0<report['necessary_rows']<30
    changed=frame.with_columns(pl.lit(-999.).alias('target_5d_net'),pl.lit(None).cast(pl.Float64).alias('target_5d_rel_net'),
        pl.lit(999.).alias('pred_ret_raw_5d'),pl.lit(999.).alias('stockformer_score'))
    other,_=research.reference_pool(tmp_path,changed,raw,EXECUTION)
    cols=['symbol','date','_necessary_entry_pool','_reference_risk_score','_reference_entry_without_model']
    assert reference.select(cols).equals(other.select(cols))
    later=frame.with_columns((pl.col('date')+timedelta(days=1)).alias('date'),pl.lit(0.).alias('ret_1d'))
    later_raw=raw.with_columns((pl.col('date')+timedelta(days=1)).alias('date'))
    extended,_=research.reference_pool(tmp_path,pl.concat([frame,later]),pl.concat([raw,later_raw]),EXECUTION)
    assert reference.select(cols).equals(extended.filter(pl.col('date')==date(2024,1,1)).select(cols))


def test_signal_scoring_keeps_full_reference_and_does_not_renormalize_after_filter(tmp_path,monkeypatch):
    monkeypatch.setattr(sf,'_attach_market_gate_for_oos',gate)
    frame,raw=inputs()
    reference,_=research.reference_pool(tmp_path,frame,raw,EXECUTION)
    predictions=reference.select('symbol','date').with_columns(pl.Series('pred_ret_raw_5d',[i/100 for i in range(30)]),
        pl.lit(.02).alias('pred_ret_5d'))
    baseline=portfolio._build_gated_signal_frame(tmp_path,predictions,reference,
        dict(target_mode='industry_relative_net',signal_weights={'5d':1.}),horizon=5,min_net_edge=.0005,
        score_source='raw',historical_identity=True)
    full,candidate=research.reference_signals(tmp_path,reference,predictions,min_net_edge=.0005,signal_policy=None)
    assert full.select('symbol','date').equals(baseline.select('symbol','date'))
    expected=baseline.join(reference.select('symbol','date','_necessary_entry_pool'),on=['symbol','date']).filter('_necessary_entry_pool')
    for c in ['entry_score','risk_score','stockformer_score','_entry_rank','net_margin_score','_entry_rules_pass']:
        assert candidate[c].equals(expected[c])
    assert candidate['_necessary_entry_pool'].all() and candidate.height<full.height
    with pytest.raises(ValueError,match='universe'):
        research.reference_signals(tmp_path,reference,predictions.head(20),min_net_edge=.0005,signal_policy=None)


def test_time_parts_use_only_matured_labels_and_never_training_predictions(tmp_path,monkeypatch):
    monkeypatch.setattr(sf,'_attach_market_gate_for_oos',gate)
    frame,raw=inputs(days=311,stocks=500)
    reference,_=research.reference_pool(tmp_path,frame,raw,EXECUTION)
    start=date(2024,1,1)
    window=dict(train_start=start,calibration_start=start+timedelta(days=258),
        validation_start=start+timedelta(days=284),validation_dates=[start+timedelta(days=i) for i in range(284,305)])
    fitting,coverage=research.audited_fitting_frame(reference,window)
    assert coverage['train_dates']==252 and coverage['calibration_dates']==20
    assert fitting['_necessary_entry_pool'].all()
    assert fitting['date'].max()<window['validation_start']
    assert coverage['tree_max_label_end']<window['calibration_start']
    assert coverage['calibration_max_label_end']<window['validation_start']
    changed=reference.with_columns(pl.lit(999.).alias('pred_ret_raw_5d'))
    other,_=research.audited_fitting_frame(changed,window)
    assert fitting.select('symbol','date').equals(other.select('symbol','date'))
    future=reference.with_columns(pl.when(pl.col('date')==start).then(start+timedelta(days=400))
        .otherwise(pl.col('label_end_5d')).alias('label_end_5d'))
    with pytest.raises(ValueError,match='252'):
        research.audited_fitting_frame(future,window)


def test_reference_rejects_schema_or_identity_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(sf,'_attach_market_gate_for_oos',gate)
    frame,raw=inputs()
    with pytest.raises(ValueError,match='Duplicate'):
        research.reference_pool(tmp_path,pl.concat([frame,frame]),raw,EXECUTION)
    reference,_=research.reference_pool(tmp_path,frame,raw,EXECUTION)
    predictions=reference.select('symbol','date').with_columns(pl.lit(.01).alias('pred_ret_raw_5d'),pl.lit(.01).alias('pred_ret_5d'))
    changed=reference.with_columns((pl.col('_reference_risk_score')+1).alias('_reference_risk_score'))
    with pytest.raises(ValueError,match='drift'):
        research.reference_signals(tmp_path,changed,predictions,min_net_edge=.0005,signal_policy=None)
