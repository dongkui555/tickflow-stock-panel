from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_rank_objective_research as research


def test_rank_target_is_bounded_equal_date_centered_and_not_a_return_alias():
    frame=pl.DataFrame(dict(symbol=['a','b','c','d','e'],date=[date(2025,1,1)]*3+[date(2025,1,2)]*2,
        target_5d_rel_net=[-.02,.01,100.,.03,.03]))
    result=research.training_rank_labels(frame)
    assert result['target_5d_rel_net'].equals(frame['target_5d_rel_net'])
    assert result[research.RANK_TARGET].to_list()==pytest.approx([-1/3,0.,1/3,0.,0.])
    changed=frame.with_columns(pl.col('target_5d_rel_net')*100)
    assert research.training_rank_labels(changed)[research.RANK_TARGET].equals(result[research.RANK_TARGET])
    assert research.training_rank_labels(frame.head(3))[research.RANK_TARGET].equals(result[research.RANK_TARGET].head(3))


def test_nonfinite_labels_are_not_ranked_and_duplicate_identity_is_rejected():
    frame=pl.DataFrame(dict(symbol=['a','b','c','d'],date=[date(2025,1,1)]*4,target_5d_rel_net=[1.,2.,None,float('nan')]))
    result=research.training_rank_labels(frame)
    assert result[research.RANK_TARGET].to_list()==[-.25,.25,None,None]
    with pytest.raises(ValueError,match='Duplicate'):
        research.training_rank_labels(pl.concat([frame,frame]))


def test_calibration_error_decomposition_is_equal_date_and_support_is_from_calibration_only():
    cal=pl.DataFrame(dict(symbol=['a','b','c','d'],date=[date(2025,1,1)]*4,
        raw=[0.,1.,2.,3.],target=[0.,1.,2.,3.],calibrated=[1.,2.,3.,4.]))
    result=research.calibration_diagnostic(cal,cal)
    assert result['check']['calibrated']['mse']==pytest.approx(1.)
    assert result['check']['calibrated']['daily_bias_mse']==pytest.approx(1.)
    assert result['check']['calibrated']['within_day_error_mse']==pytest.approx(0.)
    other=cal.head(1).with_columns(pl.lit(date(2025,1,2)).alias('date'),pl.lit(2.).alias('calibrated'))
    result=research.calibration_diagnostic(cal,pl.concat([cal,other]))
    assert result['check']['calibrated']['bias']==pytest.approx(1.5)
    altered=cal.with_columns(pl.lit(999.).alias('target'))
    assert research.calibration_diagnostic(altered,cal)['support']==research.calibration_diagnostic(cal,cal)['support']


def test_rank_fitting_uses_rank_y_but_calibrates_to_actual_cost_after_return(tmp_path,monkeypatch):
    rows=[]
    for d in range(175):
        for i in range(25):
            x=(i+d%17)/1000
            rows.append(dict(symbol=str(i),date=date(2024,1,1)+timedelta(days=d),ret_1d=x,
                target_5d_rel_net=.003+.02*x,label_end_5d=date(2024,1,1)+timedelta(days=d+6),
                minute_data_available=1.,minute_bar_count=240.,minute_session_coverage=1.))
    frame=pl.DataFrame(rows)
    seen=[]
    class Dummy:
        booster_=property(lambda self:self)
        def save_model(self,path):
            Path(path).write_text('fixed-rank-tree')
    def fit(x,y,**kwargs):
        seen.append(y.copy())
        return Dummy()
    monkeypatch.setattr(sf,'_fit_xgboost',fit)
    monkeypatch.setattr(sf,'_fit_lightgbm',fit)
    monkeypatch.setattr(sf,'_predict_xgboost',lambda model,x:x[:,0])
    monkeypatch.setattr(sf,'_predict_lightgbm',lambda model,x:x[:,0])
    manifest=research.fit_rank_pair(tmp_path/'rank',frame.reverse(),['ret_1d'],
        calibration_start=date(2024,5,1),label_cut=date(2024,6,25))
    assert max(seen[0])<.5 and min(seen[0])>-.5
    assert np.mean(seen[0])==pytest.approx(0.)
    assert manifest['provenance']['target_column']==research.RANK_TARGET
    assert manifest['provenance']['calibration_target']=='target_5d_rel_net'
    import json
    spec=json.loads((tmp_path/'rank/calibration.json').read_text())
    assert spec['slope']==pytest.approx(.02,rel=1e-5)
    assert spec['intercept']==pytest.approx(.003,rel=1e-5)
    with pytest.raises(ValueError,match='empty'):
        research.fit_rank_pair(tmp_path/'rank',frame,['ret_1d'],calibration_start=date(2024,5,1),label_cut=date(2024,6,25))
