"""Read-only ranking/entry diagnostics around existing frozen research pairs."""
from __future__ import annotations

import numpy as np
import polars as pl

from app.services.stockformer_selection_research import entry_masks

TARGET = 'target_5d_rel_net'
ABSOLUTE = 'target_5d_net'
POLICY = dict(version='frozen_ranking_entry_diagnostic_v1',min_cross_section=20,top_fraction=.2,
    block_sessions=6,bootstrap_samples=1000,seed=20261025,min_interval_days=20,
    constant_score_tolerance=1e-12,model_fits=0,threshold_search=False,activation_allowed=False)


def _keys(frame):
    if frame.select(pl.struct('symbol','date').is_duplicated().any()).item():
        raise ValueError('Duplicate diagnostic symbol/date')


def _varies(column):
    return (pl.col(column).n_unique()>1)&((pl.col(column).max()-pl.col(column).min())>
        POLICY['constant_score_tolerance']*(1+pl.col(column).abs().max()))


def rank_groups(frame, score):
    """Rank all available scores before looking at the availability of labels."""
    _keys(frame)
    value=pl.when(pl.col(score).is_finite()).then(pl.col(score)).otherwise(None)
    available=frame.with_columns(value.alias('_rank_score'))
    count=pl.col('_rank_score').count().over('date')
    varied=_varies('_rank_score').over('date')
    rankable=(count>=POLICY['min_cross_section'])&varied
    rank=pl.col('_rank_score').rank(method='average',descending=True).over('date')
    return available.with_columns(rankable.fill_null(False).alias('_rank_available'),
        (rankable&(rank<=count*POLICY['top_fraction'])).fill_null(False).alias('_rank_top'),
        (rankable&(rank>count*(1-POLICY['top_fraction']))).fill_null(False).alias('_rank_bottom'))


def daily_quality(frame, score, target=TARGET):
    ranked=rank_groups(frame,score)
    valid=ranked.filter(pl.col('_rank_score').is_finite()&pl.col(target).is_finite())
    corr=valid.group_by('date').agg(pl.len().alias('labeled_rows'),
        pl.when((pl.len()>=POLICY['min_cross_section'])&_varies('_rank_score')&_varies(target)).then(
            pl.corr('_rank_score',target,method='spearman')).otherwise(None).alias('rank_ic'))
    def mean(column, mask=None):
        if mask is None:
            mask=pl.lit(True)
        return pl.col(column).filter(mask&pl.col(column).is_finite()).mean()
    daily=ranked.group_by('date').agg(pl.len().alias('rows'),
        pl.col('_rank_score').count().alias('scored_rows'),pl.col('_rank_available').all().alias('rankable'),
        pl.col('_rank_top').sum().alias('top_rows'),pl.col('_rank_bottom').sum().alias('bottom_rows'),
        (pl.col('_rank_top')&pl.col(target).is_finite()).sum().alias('top_labeled_rows'),
        mean(target).alias('mean_target'),mean(ABSOLUTE).alias('mean_absolute'),
        mean(target,pl.col('_rank_top')).alias('top_target'),mean(target,pl.col('_rank_bottom')).alias('bottom_target'),
        mean(ABSOLUTE,pl.col('_rank_top')).alias('top_absolute'),mean(ABSOLUTE,pl.col('_rank_bottom')).alias('bottom_absolute'))
    return daily.join(corr,on='date',how='left',validate='1:1').with_columns(pl.col('labeled_rows').fill_null(0),
        (pl.col('top_target')-pl.col('bottom_target')).alias('top_bottom_spread')).sort('date')


def _mean(values):
    finite=np.asarray(values,dtype=float)
    finite=finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else None


def _interval(values):
    values=np.asarray(values,dtype=float)
    if np.isfinite(values).sum()<POLICY['min_interval_days']:
        return None
    n=len(values)
    rng=np.random.default_rng(POLICY['seed'])
    block=POLICY['block_sessions']
    start=rng.integers(0,n,size=(POLICY['bootstrap_samples'],(n+block-1)//block))
    indices=((start[:,:,None]+np.arange(block))%n).reshape(POLICY['bootstrap_samples'],-1)[:,:n]
    draws=values[indices]
    count=np.isfinite(draws).sum(axis=1)
    means=np.nansum(draws,axis=1)/np.maximum(count,1)
    return np.quantile(means[count>0],[.025,.975]).tolist()


def quality_summary(daily, calendar):
    dates=sorted(set(calendar))
    index={d:i for i,d in enumerate(dates)}
    if not set(daily['date'])<=set(dates):
        raise ValueError('Diagnostic date missing from calendar')
    fields=['rank_ic','top_target','top_absolute','top_bottom_spread','mean_target','mean_absolute']
    result=dict(rows=int(daily['rows'].sum()),dates=daily.height,labeled_rows=int(daily['labeled_rows'].sum()),
        rankable_dates=int(daily['rankable'].sum()),kind='overlapping_labels_not_account_returns')
    for field in fields:
        result[field+'_mean']=_mean(daily[field].to_numpy())
    valid=daily.filter(pl.col('rank_ic').is_finite())
    result.update(rank_ic_days=valid.height,rank_ic_positive_fraction=
        float((valid['rank_ic']>0).mean()) if valid.height else None)
    maps={field:dict(zip(daily['date'],daily[field],strict=True)) for field in ['rank_ic','top_bottom_spread']}
    grid=[d for d in dates if daily.height and daily['date'].min()<=d<=daily['date'].max()]
    for field,mapping in maps.items():
        result[field+'_block_ci95']=_interval([mapping.get(d) for d in grid])
    result['nonoverlapping_offsets']=[]
    for offset in range(POLICY['block_sessions']):
        part=daily.filter(pl.Series([index[d]%POLICY['block_sessions']==offset for d in daily['date']],dtype=pl.Boolean))
        result['nonoverlapping_offsets'].append(dict(offset=offset,
            rank_ic_days=part.filter(pl.col('rank_ic').is_finite()).height,
            rank_ic_mean=_mean(part['rank_ic'].to_numpy()),top_bottom_spread_mean=_mean(part['top_bottom_spread'].to_numpy())))
    return result


def label_summary(frame):
    valid=frame.filter(pl.col(TARGET).is_finite()&pl.col(ABSOLUTE).is_finite())
    means=valid.group_by('date').agg(pl.col(TARGET).mean(),pl.col(ABSOLUTE).mean())
    return dict(rows=frame.height,dates=frame['date'].n_unique(),labeled_rows=valid.height,
        labeled_dates=means.height,industry_net_daily_mean=_mean(means[TARGET].to_numpy()),
        absolute_net_daily_mean=_mean(means[ABSOLUTE].to_numpy()),not_account_returns=True)


def entry_alignment(frame, score):
    masks,names=entry_masks(frame)
    strict=masks.select(pl.all_horizontal([pl.col(n) for n in names])).to_series()
    if not strict.equals(frame['_entry_rules_pass'].fill_null(False)):
        raise ValueError('Frozen entry predicate differs from diagnostic predicate')
    ranked=rank_groups(frame,score)
    populations={}
    for name,selected in [('all',pl.Series([True]*frame.height)),('raw_top20',ranked['_rank_top']),
                          ('forecast_net',frame['_forecast_pass'].fill_null(False))]:
        rules={}
        for rule in names:
            others=masks.select(pl.all_horizontal([pl.col(n) for n in names if n!=rule])).to_series()
            passed=selected&masks[rule]
            failed=selected&~masks[rule]
            rules[rule]=dict(passed=label_summary(frame.filter(passed)),failed=label_summary(frame.filter(failed)),
                sole_blocker_rows=int((failed&others).sum()))
        populations[name]=dict(summary=label_summary(frame.filter(selected)),rules=rules)
    high=ranked['_rank_top']
    entry=frame['_entry_rules_pass'].fill_null(False)
    market=frame['_market_gate_pass'].fill_null(False)
    forecast=frame['_forecast_pass'].fill_null(False)
    groups={name:label_summary(frame.filter(mask)) for name,mask in [
        ('top20_entry_pass',high&entry),('top20_entry_fail',high&~entry),
        ('forecast_entry_pass',forecast&entry),('forecast_entry_fail',forecast&~entry),
        ('forecast_entry_market',forecast&entry&market)]}
    return dict(populations=populations,groups=groups,predicate_parity=True,thresholds_unchanged=True)


def component_correlations(frame, score, columns):
    valid=frame.filter(pl.col(score).is_finite())
    expr=[]
    for column in columns:
        pairwise=pl.col(column).is_finite()
        x,y=pl.col(score).filter(pairwise),pl.col(column).filter(pairwise)
        expr.append(pl.when((pairwise.sum()>=POLICY['min_cross_section'])&(x.n_unique()>1)&(y.n_unique()>1)).then(
            pl.corr(x,y,method='spearman')).otherwise(None).alias(column))
    daily=valid.group_by('date').agg(*expr)
    return {c:dict(mean_daily_spearman=_mean(daily[c].to_numpy()),
        valid_dates=daily.filter(pl.col(c).is_finite()).height) for c in columns}


def refit_agreement(old, new, score):
    for frame in [old,new]:
        _keys(frame)
    keys=['symbol','date']
    if not old.select(keys).sort(keys).equals(new.select(keys).sort(keys)):
        raise ValueError('Refit prediction universe mismatch')
    left=rank_groups(old,score).select(*keys,pl.col('_rank_score').alias('old_score'),pl.col('_rank_top').alias('old_top'))
    right=rank_groups(new,score).select(*keys,pl.col('_rank_score').alias('new_score'),pl.col('_rank_top').alias('new_top'))
    frame=left.join(right,on=keys,validate='1:1')
    finite=pl.col('old_score').is_finite()&pl.col('new_score').is_finite()
    correlations=frame.filter(finite).group_by('date').agg(
        pl.when((pl.len()>=POLICY['min_cross_section'])&_varies('old_score')&_varies('new_score')).then(
            pl.corr('old_score','new_score',method='spearman')).otherwise(None).alias('rank_correlation'))
    daily=frame.group_by('date').agg(pl.len().alias('rows'),pl.col('old_top').sum().alias('old_top'),
        pl.col('new_top').sum().alias('new_top'),(pl.col('old_top')&pl.col('new_top')).sum().alias('intersection'),
        (pl.col('old_top')|pl.col('new_top')).sum().alias('union')).join(correlations,on='date',how='left').with_columns(
            pl.when(pl.min_horizontal('old_top','new_top')>0).then(
                pl.col('intersection')/pl.min_horizontal('old_top','new_top')).otherwise(None).alias('top_overlap_fraction'),
            pl.when(pl.col('union')>0).then(pl.col('intersection')/pl.col('union')).otherwise(None).alias('top_jaccard')).sort('date')
    return dict(daily=daily.to_dicts(),rank_correlation_mean=_mean(daily['rank_correlation'].to_numpy()),
        top_overlap_fraction_mean=_mean(daily['top_overlap_fraction'].to_numpy()),
        top_jaccard_mean=_mean(daily['top_jaccard'].to_numpy()),same_prediction_keys=True)
