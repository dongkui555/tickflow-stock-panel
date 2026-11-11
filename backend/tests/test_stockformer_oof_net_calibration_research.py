from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_calibration_research as calibration
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_oof_net_calibration_research as research


def observations(days=84):
    rows=[]
    for d in range(days):
        for i in range(25):
            p=(i+.5)/25
            state='open' if d%2==0 else 'observe'
            rows.append(dict(symbol=str(i),date=date(2025,1,1)+timedelta(days=d),percentile=p,
                target=.02*p+(.01 if state=='open' else -.01),market_gate=state,
                label_end_5d=date(2025,1,1)+timedelta(days=d+6)))
    return pl.DataFrame(rows)


def test_maps_use_actual_return_units_and_state_effects_are_shrunk():
    frame=observations()
    before=date(2025,5,1)
    affine=research.fit_map(frame,'affine_percentile',before=before)
    state=research.fit_map(frame,'isotonic_state',before=before)
    assert affine['base']['slope']==pytest.approx(.02)
    assert affine['base']['intercept']==pytest.approx(0.)
    # Both states have the same percentile distribution. The existing 20-bin
    # isotonic interpolation can add a shared residual, but their difference
    # must retain the observed 2pp state gap with the fixed 42/(42+20) shrinkage.
    assert state['states']['open']['offset']-state['states']['observe']['offset']==pytest.approx(.02*42/(42+20))
    assert 0<state['states']['open']['offset']<.01
    assert -.01<state['states']['observe']['offset']<0
    pred=research.apply_map(frame,state)
    assert pred[24]>pred[0] and pred[0]>pred[25]
    shuffled=frame.with_columns(pl.lit(-999.).alias('target'),pl.lit(date(2030,1,1)).alias('label_end_5d'))
    np.testing.assert_array_equal(research.apply_map(shuffled,state),pred)


def test_unsupported_state_falls_back_and_negative_constant_is_not_forced_positive():
    frame=observations().with_columns(pl.lit(-.01).alias('target'))
    spec=research.fit_map(frame,'isotonic_state',before=date(2025,5,1))
    unknown=frame.with_columns(pl.lit('unknown').alias('market_gate'))
    assert np.all(research.apply_map(unknown,spec)<0)
    sparse=frame.with_columns(pl.when(pl.col('date')==date(2025,1,1)).then(pl.lit('strict'))
        .otherwise(pl.lit('open')).alias('market_gate'))
    spec=research.fit_map(sparse,'isotonic_state',before=date(2025,5,1))
    assert not spec['states']['strict']['supported'] and spec['states']['strict']['offset']==0.


def test_duplicate_unmatured_and_out_of_range_percentiles_fail_closed():
    frame=observations()
    with pytest.raises(ValueError,match='Duplicate'):
        research.fit_map(pl.concat([frame,frame]),'affine_percentile',before=date(2025,5,1))
    with pytest.raises(ValueError,match='mature'):
        research.fit_map(frame,'affine_percentile',before=date(2025,2,1))
    with pytest.raises(ValueError,match='percentile'):
        research.fit_map(frame.with_columns(pl.lit(1.1).alias('percentile')),'affine_percentile',before=date(2025,5,1))


def test_percentiles_use_full_reference_before_candidate_and_label_filter():
    frame=pl.DataFrame(dict(symbol=['a','b','c','d'],date=[date(2025,1,1)]*4,raw=[1.,2.,3.,4.]))
    ref=frame.select('symbol','date').with_columns(pl.Series('_necessary_entry_pool',[True,True,False,False]),
        pl.lit('open').alias('market_gate'))
    result=research.percentile_candidates(frame,ref)
    assert result['percentile'].to_list()==[.125,.375]
    changed=ref.with_columns(pl.lit(-999.).alias('target_5d_net'))
    assert result.select('symbol','date','percentile').equals(research.percentile_candidates(frame,changed).select('symbol','date','percentile'))
    with pytest.raises(ValueError,match='universe'):
        research.percentile_candidates(frame.head(3),ref)


def test_teacher_only_fits_mature_history_and_never_scores_its_training_dates(tmp_path,monkeypatch):
    rows=[]
    for d in range(280):
        for i in range(25):
            x=(i+d%17)/1000
            rows.append(dict(symbol=str(i),date=date(2024,1,1)+timedelta(days=d),ret_1d=x,
                target_5d_rel_net=.003+.02*x,target_5d_net=.01+x,label_end_5d=date(2024,1,1)+timedelta(days=d+6),
                minute_data_available=1.,minute_bar_count=240.,minute_session_coverage=1.))
    frame=pl.DataFrame(rows)
    seen=[]
    class Dummy:
        booster_=property(lambda self:self)
        def save_model(self,path):
            Path(path).write_text('fixed-teacher-tree')
    def fit(x,y,**kwargs):
        seen.append(y.copy())
        return Dummy()
    monkeypatch.setattr(sf,'_fit_xgboost',fit)
    monkeypatch.setattr(sf,'_fit_lightgbm',fit)
    before=date(2024,1,1)+timedelta(days=258)
    manifest=research.fit_teacher(tmp_path/'teacher',frame,['ret_1d'],start=date(2024,1,1),before=before,head='industry_rank')
    assert manifest['train_dates']==252 and manifest['max_label_end']<str(before)
    assert max(seen[0])<.5 and min(seen[0])>-.5
    assert not (tmp_path/'teacher/calibration.json').exists()
    with pytest.raises(ValueError,match='training dates'):
        research.predict_teacher(tmp_path/'teacher',frame.head(25))
    with pytest.raises(ValueError,match='empty'):
        research.fit_teacher(tmp_path/'teacher',frame,['ret_1d'],start=date(2024,1,1),before=before,head='industry_rank')


def test_frozen_raw_forecast_never_uses_old_calibration_parameters(monkeypatch):
    day=date(2025,1,2)
    frame=pl.DataFrame(dict(symbol=['a'],date=[day],ret_1d=[.01],minute_data_available=[1.],
        minute_bar_count=[240.],minute_session_coverage=[1.]))
    monkeypatch.setattr(pair,'verify_manifest',lambda root:dict(feature_columns=['ret_1d'],provenance=dict(tree_max_label_end='2025-01-01')))
    monkeypatch.setattr(pair,'_load_trees',lambda root:(None,None))
    monkeypatch.setattr(sf,'_predict_xgboost',lambda model,x:x[:,0])
    monkeypatch.setattr(sf,'_predict_lightgbm',lambda model,x:x[:,0])
    def forbidden(*args,**kwargs):
        raise AssertionError('Old calibration parameters must not influence this forecast')
    monkeypatch.setattr(calibration,'apply_method',forbidden)
    result=research.frozen_raw_prediction('unused',frame)
    assert result['raw'].item()==pytest.approx(.01)
