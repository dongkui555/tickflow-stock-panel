"""Necessary entry feasibility using fixed known-day factors, without model labels."""
from __future__ import annotations

import polars as pl

from app.services import stockformer as sf
from app.services.stockformer_selection_research import entry_masks

POLICY = dict(version='fixed_entry_score_attainability_v1',existing_minimum_entry_score=62.,
    model_score_range=[0.,100.],model_entry_coefficient=.52*sf.MULTIFACTOR_WEIGHTS['stockformer_score'],
    market_penalty_coefficient=.55,relative_rank_feasibility='not_claimed',model_fits=0,
    use_for_live_or_training_pool=False)


def feasibility(frame):
    """Bound the same entry formula; this is not a replacement entry predicate."""
    masks,rules=entry_masks(frame)
    original=sf._add_multifactor_scores(frame)
    penalty=pl.col('market_buy_threshold_add').cast(pl.Float64,strict=False).fill_null(0.)*.55
    restored=original.with_columns((pl.col('entry_score')-penalty).clip(0,100).alias('_restored_entry'))
    difference=restored.select((pl.col('_restored_entry')-frame['entry_score']).abs().max()).item()
    if difference is not None and difference>1e-10:
        raise ValueError('Entry score reconstruction differs from frozen cross-section')
    scores={}
    for name,value in [('minimum',0.),('maximum',100.)]:
        bound=sf._add_multifactor_scores(frame.with_columns(pl.lit(value).alias('stockformer_score')))
        scores[name]=bound.select((pl.col('entry_score')-penalty).clip(0,100)).to_series()
    known=masks.select(pl.all_horizontal([pl.col(n) for n in rules if n not in {'entry_score_62','entry_rank_p35'}])).to_series()
    result=frame.with_columns(scores['minimum'].alias('_minimum_entry_score'),scores['maximum'].alias('_maximum_entry_score'),
        (scores['maximum']>=POLICY['existing_minimum_entry_score']).alias('_entry_score_attainable'),
        ((scores['maximum']>=POLICY['existing_minimum_entry_score'])&known).alias('_known_entry_necessary'))
    if result.filter(pl.col('_entry_rules_pass')&~pl.col('_known_entry_necessary')).height:
        raise ValueError('Necessary entry condition excluded an actual entry')
    return result
