import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_gross_rank_research as gross


def test_ablation_keeps_full_feature_cohort_and_never_promotes_missing_minute_values():
    columns=sf.training_feature_columns('minute_enhanced')
    reduced=[c for c in columns if c not in sf.MINUTE_FEATURE_COLUMNS]
    assert len(columns)==80 and len(reduced)==68
    rows=[]
    for i in range(3):
        row={c:float(i+1) for c in columns}
        row.update(symbol=str(i),date=date(2025,1,1),target_5d_rel_net=.01*i,
            minute_data_available=1.,minute_bar_count=240. if i!=2 else 239.,minute_session_coverage=1.)
        if i==1:
            row['minute_intraday_return']=None
        rows.append(row)
    raw=pl.DataFrame(rows)
    complete,_=sf.prepare_training_features(raw,'minute_enhanced')
    _,_,reference=sf._prepare_xy(complete,'target_5d_rel_net',columns)
    assert reference['symbol'].to_list()==['0']
    _,_,unrestricted=sf._prepare_xy(complete,'target_5d_rel_net',reduced)
    assert unrestricted['symbol'].to_list()==['0','1']
    x,y,full=sf._prepare_xy(reference,'target_5d_rel_net',columns)
    sx,sy,lean=sf._prepare_xy(reference,'target_5d_rel_net',reduced)
    assert full.select('symbol','date').equals(lean.select('symbol','date'))
    np.testing.assert_array_equal(sx,x[:,[columns.index(c) for c in reduced]])
    np.testing.assert_array_equal(y,sy)


def test_existing_gross_fit_accepts_68_columns_with_same_samples_target_and_240_gate(tmp_path,monkeypatch):
    columns=sf.training_feature_columns('minute_enhanced')
    reduced=[c for c in columns if c not in sf.MINUTE_FEATURE_COLUMNS]
    start=date(2024,1,1)
    rows=[]
    for d in range(284):
        for i in range(56):
            x=i/1000
            row={c:x for c in columns}
            row.update(symbol=str(i),date=start+timedelta(days=d),ret_1d=x,
                target_5d_rel_net=.003+.02*x,label_cost_5d=.005,
                label_end_5d=start+timedelta(days=d+6),minute_data_available=1.,
                minute_bar_count=239. if i==55 else 240.,minute_session_coverage=1.)
            rows.append(row)
    data=pl.DataFrame(rows)
    seen=[]
    class Dummy:
        booster_=property(lambda self:self)
        def save_model(self,path):
            Path(path).write_text('ablation-test-tree')
    def fit(x,y,**kwargs):
        seen.append((x.copy(),y.copy()))
        return Dummy()
    monkeypatch.setattr(sf,'_fit_xgboost',fit)
    monkeypatch.setattr(sf,'_fit_lightgbm',fit)
    monkeypatch.setattr(sf,'_predict_xgboost',lambda model,x:x[:,0])
    monkeypatch.setattr(sf,'_predict_lightgbm',lambda model,x:x[:,0])
    manifests=[]
    for name,features in [('full',columns),('without',reduced)]:
        manifests.append(gross.fit_gross_pair(tmp_path/name,data,features,
            calibration_start=start+timedelta(days=258),label_cut=start+timedelta(days=284)))
    assert seen[0][0].shape==(252*55,80) and seen[2][0].shape==(252*55,68)
    np.testing.assert_array_equal(seen[0][1],seen[2][1])
    np.testing.assert_array_equal(seen[2][0],seen[0][0][:,[columns.index(c) for c in reduced]])
    assert all(m['minute_policy']==sf.MINUTE_AVAILABILITY_POLICY for m in manifests)
    assert all(m['provenance']['calibration_rows']==20*55 for m in manifests)
    assert all(m['provenance']['train_dates']==252 for m in manifests)
    assert not set(manifests[1]['feature_columns'])&set(sf.MINUTE_FEATURE_COLUMNS)
    for name in ['full','without']:
        mapping=json.loads((tmp_path/name/'calibration.json').read_text())
        assert mapping['slope']==pytest.approx(.02,rel=1e-5)
        assert mapping['intercept']==pytest.approx(.003,rel=1e-5)
