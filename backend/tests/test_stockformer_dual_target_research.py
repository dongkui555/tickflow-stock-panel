from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_dual_target_research as dual
from app.services import stockformer_frozen_pair as pair


def signals():
    return pl.DataFrame([dict(symbol=str(i),date=date(2026,1,5),amount=1e9/(i+1),
        pred_ret_raw_5d=.01*i,pred_ret_5d=.02,net_margin_score=.019,
        _prediction_pass=True,_net_margin_pass=True,_forecast_pass=i!=3,
        _entry_rules_pass=i!=2,_market_gate_pass=i!=1) for i in range(4)])


def test_joint_gate_is_subset_preserves_risk_and_does_not_double_charge_cost():
    original=signals()
    absolute=original.select('symbol','date').with_columns(pl.lit(.02).alias('pred_ret_raw_5d'),
        pl.Series('pred_ret_5d',[.0006,-.01,.02,.02]))
    joint=dual.joint_signal(original,absolute,min_net_edge=.0005)
    assert joint['_forecast_pass'].to_list()==[True,False,True,False]
    assert joint['absolute_net_margin_score'][0]==pytest.approx(.0006)
    for c in ['_entry_rules_pass','_market_gate_pass','pred_ret_raw_5d','net_margin_score']:
        assert joint[c].equals(original[c])
    assert not joint.filter(pl.col('_forecast_pass')&~pl.col('_industry_forecast_pass')).height


def test_missing_nonfinite_absolute_predictions_cannot_pass_and_mismatched_rows_fail():
    original=signals()
    absolute=original.select('symbol','date').with_columns(pl.lit(.01).alias('pred_ret_raw_5d'),
        pl.Series('pred_ret_5d',[None,float('nan'),0.,-.01]))
    joint=dual.joint_signal(original,absolute,min_net_edge=.0005)
    assert not joint['_forecast_pass'].fill_null(False).any()
    assert not joint['_absolute_net_margin_pass'].fill_null(False).any()
    with pytest.raises(ValueError,match='universe'):
        dual.joint_signal(original,absolute.head(3),min_net_edge=.0005)
    with pytest.raises(ValueError,match='Duplicate'):
        dual.joint_signal(original,pl.concat([absolute,absolute]),min_net_edge=.0005)


def test_absolute_fit_uses_absolute_target_and_purges_calibration(tmp_path,monkeypatch):
    rows=[]
    for i in range(175):
        for s in range(25):
            day=date(2024,1,1)+timedelta(days=i)
            x=(i%17+s)/1000
            rows.append(dict(symbol=str(s),date=day,ret_1d=x,target_5d_net=.02+x*.1,
                target_5d_rel_net=-.01+x*.2,label_end_5d=day+timedelta(days=6),
                minute_data_available=1.,minute_bar_count=240.,minute_session_coverage=1.))
    frame=pl.DataFrame(rows)
    seen=[]
    class DummyTree:
        booster_=property(lambda self:self)
        def save_model(self,path):
            Path(path).write_text('fixed-absolute-tree')
    def fit(x,y,**kwargs):
        seen.append((x.copy(),y.copy()))
        return DummyTree()
    monkeypatch.setattr(sf,'_fit_xgboost',fit)
    monkeypatch.setattr(sf,'_fit_lightgbm',fit)
    monkeypatch.setattr(sf,'_predict_xgboost',lambda model,x:x[:,0])
    monkeypatch.setattr(sf,'_predict_lightgbm',lambda model,x:x[:,0])
    manifest=dual.fit_absolute_pair(tmp_path/'absolute',frame.reverse(),['ret_1d'],
        calibration_start=date(2024,5,1),label_cut=date(2024,6,25))
    np.testing.assert_allclose(seen[0][1],.02+seen[0][0][:,0]*.1)
    assert manifest['provenance']['target_column']=='target_5d_net'
    assert manifest['provenance']['tree_max_label_end']<'2024-05-01'
    assert manifest['provenance']['calibration_max_label_end']<'2024-06-25'
    with pytest.raises(ValueError,match='target mismatch'):
        dual.predict_head(tmp_path/'absolute',frame,dual.INDUSTRY_TARGET)
    assert pair.verify_manifest(tmp_path/'absolute')['activation_allowed'] is False
    with pytest.raises(ValueError,match='empty'):
        dual.fit_absolute_pair(tmp_path/'absolute',frame,['ret_1d'],calibration_start=date(2024,5,1),label_cut=date(2024,6,25))
