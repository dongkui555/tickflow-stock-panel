import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_gross_rank_research as research


def test_gross_rank_removes_cost_from_tree_target_but_preserves_actual_net_labels():
    frame=pl.DataFrame(dict(symbol=['a','b','c'],date=[date(2025,1,1)]*3,
        target_5d_rel_net=[.01,.012,.03],label_cost_5d=[.005,.001,.001]))
    ranked=research.training_gross_rank_labels(frame)
    assert ranked['target_5d_rel_net'].equals(frame['target_5d_rel_net'])
    assert ranked['label_cost_5d'].equals(frame['label_cost_5d'])
    assert ranked[research.RANK_TARGET].to_list()==pytest.approx([0.,-1/3,1/3])
    later=frame.with_columns(pl.lit(date(2025,1,2)).alias('date'),pl.lit(9.).alias('target_5d_rel_net'))
    assert research.training_gross_rank_labels(pl.concat([frame,later]))[research.RANK_TARGET].head(3).equals(ranked[research.RANK_TARGET])


def test_missing_cost_is_not_a_zero_cost_and_duplicates_or_negative_cost_fail():
    frame=pl.DataFrame(dict(symbol=['a','b','c'],date=[date(2025,1,1)]*3,
        target_5d_rel_net=[1.,2.,3.],label_cost_5d=[.01,None,float('nan')]))
    assert research.training_gross_rank_labels(frame)[research.RANK_TARGET].to_list()==[0.,None,None]
    with pytest.raises(ValueError,match='Duplicate'):
        research.training_gross_rank_labels(pl.concat([frame,frame]))
    with pytest.raises(ValueError,match='negative'):
        research.training_gross_rank_labels(frame.with_columns(pl.lit(-.1).alias('label_cost_5d')))


def test_gross_fit_is_purged_and_calibrates_to_actual_net_not_gross_or_rank(tmp_path,monkeypatch):
    rows=[]
    start=date(2024,1,1)
    for d in range(284):
        for i in range(50):
            x=i/1000
            rows.append(dict(symbol=str(i),date=start+timedelta(days=d),ret_1d=x,
                target_5d_rel_net=.003+.02*x,label_cost_5d=.05-.03*x,
                label_end_5d=start+timedelta(days=d+6),minute_data_available=1.,
                minute_bar_count=240.,minute_session_coverage=1.))
    frame=pl.DataFrame(rows)
    seen=[]
    class Dummy:
        booster_=property(lambda self:self)
        def save_model(self,path):
            Path(path).write_text('frozen-gross-test-tree')
    def fit(x,y,**kwargs):
        seen.append(y.copy())
        return Dummy()
    monkeypatch.setattr(sf,'_fit_xgboost',fit)
    monkeypatch.setattr(sf,'_fit_lightgbm',fit)
    monkeypatch.setattr(sf,'_predict_xgboost',lambda model,x:x[:,0])
    monkeypatch.setattr(sf,'_predict_lightgbm',lambda model,x:x[:,0])
    m=research.fit_gross_pair(tmp_path/'pair',frame,['ret_1d'],
        calibration_start=start+timedelta(days=258),label_cut=start+timedelta(days=284))
    assert seen[0][0]>seen[0][-1] and abs(np.mean(seen[0]))<1e-12
    assert m['provenance']['train_dates']==252 and m['provenance']['calibration_dates']==20
    spec=json.loads((tmp_path/'pair/calibration.json').read_text())
    assert spec['slope']==pytest.approx(.02,rel=1e-5) and spec['intercept']==pytest.approx(.003,rel=1e-5)
    with pytest.raises(ValueError,match='empty'):
        research.fit_gross_pair(tmp_path/'pair',frame,['ret_1d'],calibration_start=start+timedelta(days=258),label_cut=start+timedelta(days=284))


def test_diagnostic_uses_full_reference_groups_and_does_not_promote_missing_top():
    f=pl.DataFrame(dict(symbol=['a','b','c','d'],date=[date(2025,1,1)]*4,
        raw=[1.,2.,3.,4.],percentile=[.125,.375,.625,.875],_necessary_entry_pool=[True]*4,
        _entry_rules_pass=[True,True,True,False],_market_gate_pass=[True]*4,_shared_entry=[True,True,True,False],
        gross_industry=[.01,.02,.03,None],net_industry=[0.,.01,.02,None],net_absolute=[.01,.02,.03,None],cost=[.01]*4))
    result=research.daily_quality(f,[date(2025,1,1)],'complete_entry')
    assert result.filter(pl.col('group')=='top20')['rows'].item()==0
    result=research.daily_quality(f,[date(2025,1,1)],'necessary')
    top=result.filter(pl.col('group')=='top20')
    assert top['rows'].item()==1 and top['labeled_rows'].item()==0 and top['net_industry'].item() is None
