from datetime import date

import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_entry_feasibility_research as research


def signals():
    frame=pl.DataFrame([dict(symbol=str(i),date=date(2026,1,5),stockformer_score=float(i*5),
        amount=1e6*(i+1),ret_20d=i/100.,amplitude=i/100.,risk_20d=i/200.,
        market_buy_threshold_add=0.,target_5d_net=i/100.) for i in range(20)])
    frame=sf._add_multifactor_scores(frame)
    rank=pl.col('entry_score').rank(method='average').over('date')
    return frame.with_columns((sf._entry_eligible_expr(frame)&(rank>=pl.len().over('date')*.35)).alias('_entry_rules_pass'))


def test_attainability_bounds_actual_score_preserves_entry_and_ignores_future_labels():
    original=signals()
    result=research.feasibility(original)
    assert (result['_minimum_entry_score']<=result['entry_score']+1e-10).all()
    assert (result['_maximum_entry_score']>=result['entry_score']-1e-10).all()
    assert (result['_maximum_entry_score']-result['_minimum_entry_score']).max()==pytest.approx(10.4)
    assert not result.filter(pl.col('_entry_rules_pass')&~pl.col('_known_entry_necessary')).height
    other=research.feasibility(original.with_columns(pl.lit(-999.).alias('target_5d_net')))
    assert result['_known_entry_necessary'].equals(other['_known_entry_necessary'])


def test_reconstructed_entry_checks_market_penalty_and_rejects_changed_score():
    original=signals().with_columns(pl.lit(4.).alias('market_buy_threshold_add'),
        (pl.col('entry_score')-2.2).clip(0,100).alias('entry_score'),pl.lit(False).alias('_entry_rules_pass'))
    result=research.feasibility(original)
    assert (result['_maximum_entry_score']>=result['entry_score']-1e-10).all()
    with pytest.raises(ValueError,match='reconstruction'):
        research.feasibility(original.with_columns((pl.col('entry_score')+1).alias('entry_score')))
