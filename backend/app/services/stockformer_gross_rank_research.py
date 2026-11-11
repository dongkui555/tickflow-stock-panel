"""One offline gross-industry rank intervention; net calibration stays unchanged."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_calibration_research as calibration
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_rank_objective_research as ordinal
from app.services import stockformer_ranking_research as ranking
from app.services import stockformer_training as training

RANK_TARGET='rank_5d_industry_gross'
POLICY=dict(version='gross_industry_rank_actual_net_calibration_v1',tree_target=RANK_TARGET,
    gross_label='target_5d_rel_net_plus_recorded_label_cost_5d',calibration_target=pair.TARGET,
    tree_training_sessions=252,calibration_sessions=20,tree_rounds=300,
    fitting_pool='prediction_independent_necessary_pool',score_reference='unchanged_full_reference',
    evaluation_scopes=['necessary','complete_entry','entry_market','shared_entry'],
    top_and_bottom_percentile=[.8,.2],parameter_search=False,activation_allowed=False)


def training_gross_rank_labels(frame):
    if frame.filter(pl.col('label_cost_5d').is_finite()&(pl.col('label_cost_5d')<0)).height:
        raise ValueError('Gross rank cannot use negative recorded costs')
    value=pl.when(pl.col(pair.TARGET).is_finite()&pl.col('label_cost_5d').is_finite()).then(
        pl.col(pair.TARGET)+pl.col('label_cost_5d')).otherwise(None)
    temporary=frame.select('symbol','date',value.alias(pair.TARGET))
    ranked=ordinal.training_rank_labels(temporary)
    return frame.with_columns(ranked[ordinal.RANK_TARGET].alias(RANK_TARGET))


def fit_gross_pair(root, features, columns, *, calibration_start, label_cut, context=None):
    root=Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Gross pair directory must be empty')
    root.mkdir(parents=True,exist_ok=True)
    ready,coverage=sf.prepare_training_features(features,'minute_enhanced')
    ready=ready.sort('date','symbol')
    train=training.purged_slice(ready,5,ready['date'].min(),calibration_start)
    x,_,used_train=sf._prepare_xy(train,pair.TARGET,columns)
    ranked=training_gross_rank_labels(used_train)
    y=ranked[RANK_TARGET].to_numpy()
    if used_train['date'].n_unique()!=252 or len(y)<1000 or not np.isfinite(y).all():
        raise ValueError('Gross tree requires 252 mature dates and finite gross ranks')
    xgb=sf._fit_xgboost(x,y,n_estimators=300)
    lgb=sf._fit_lightgbm(x,y,n_estimators=300)
    sample=ready.filter((pl.col('date')>=calibration_start)&(pl.col('date')<label_cut)).with_columns(
        pl.when(pl.col('label_end_5d')<label_cut).then(pl.col(pair.TARGET)).otherwise(None).alias(pair.TARGET))
    cx,_,used=sf._prepare_xy(sample.with_columns(pl.lit(0.).alias('_prediction_mask')),'_prediction_mask',columns)
    raw=.5*sf._predict_xgboost(xgb,cx)+.5*sf._predict_lightgbm(lgb,cx)
    cal=calibration.with_percentile(used.select('symbol','date',pl.col(pair.TARGET).alias('target')).with_columns(pl.Series('raw',raw)))
    mapping=calibration.fit_method(cal,'affine')
    if mapping['dates']!=20:
        raise ValueError('Gross pair requires 20 mature actual-net calibration dates')
    xgb.save_model(str(root/'xgboost.json'))
    lgb.booster_.save_model(str(root/'lightgbm.txt'))
    (root/'calibration.json').write_text(json.dumps(mapping,indent=2,allow_nan=False),encoding='utf-8')
    cal.write_parquet(root/'calibration-predictions.parquet')
    return pair.freeze_manifest(root,feature_columns=columns,provenance=dict(target_column=RANK_TARGET,
        calibration_target=pair.TARGET,gross_rank_source_sha256=pair.digest(__file__),gross_rank_policy=POLICY,
        tree_max_label_end=str(used_train['label_end_5d'].max()),calibration_start=str(calibration_start),
        calibration_max_label_end=str(used.filter(pl.col(pair.TARGET).is_finite())['label_end_5d'].max()),
        frozen_as_of=str(label_cut),train_rows=used_train.height,train_dates=252,
        calibration_rows=mapping['rows'],calibration_dates=20,minute_coverage=coverage,n_estimators=300,
        cost_target_version=training.PAPER_COST_TARGET_VERSION,refit_after_calibration=False,context=context or {}))


def predict_gross_pair(root, features):
    meta=pair.verify_manifest(root)
    p=meta['provenance']
    if (p.get('target_column')!=RANK_TARGET or p.get('calibration_target')!=pair.TARGET
            or p.get('gross_rank_source_sha256')!=pair.digest(__file__)):
        raise ValueError('Gross target or source mismatch')
    if features.height and features['date'].min()<date.fromisoformat(p['frozen_as_of']):
        raise ValueError('Gross predictions must begin after fitting and calibration')
    return pair.predict_pair(root,features)


def daily_quality(frame, calendar, scope):
    ranking._keys(frame)
    predicates={'necessary':pl.col('_necessary_entry_pool'),'complete_entry':pl.col('_entry_rules_pass'),
        'entry_market':pl.col('_entry_rules_pass')&pl.col('_market_gate_pass'),'shared_entry':pl.col('_shared_entry')}
    if scope not in predicates:
        raise ValueError('Unknown gross quality scope')
    dates=sorted(set(calendar))
    if not set(frame['date'])<=set(dates):
        raise ValueError('Date missing from quality calendar')
    grid=pl.DataFrame({'date':dates},schema={'date':pl.Date})
    selected=frame.filter(predicates[scope])
    result=[]
    for group,mask in [('all',pl.lit(True)),('top20',pl.col('percentile')>=.8),('bottom20',pl.col('percentile')<=.2)]:
        rows=selected.filter(mask)
        valid=pl.all_horizontal([pl.col(c).is_finite() for c in ['gross_industry','net_industry','net_absolute','cost']])
        aggregate=rows.group_by('date').agg(pl.len().alias('rows'),valid.sum().alias('labeled_rows'),
            *[pl.col(c).filter(valid).mean().alias(c) for c in ['gross_industry','net_industry','net_absolute','cost']])
        corr=rows.filter(valid).group_by('date').agg(pl.when((pl.len()>=ranking.POLICY['min_cross_section'])&
            ranking._varies('raw')&ranking._varies('gross_industry')).then(pl.corr('raw','gross_industry',method='spearman'))
                .otherwise(None).alias('gross_rank_ic'))
        result.append(grid.join(aggregate,on='date',how='left',validate='1:1').join(corr,on='date',how='left',validate='1:1')
            .with_columns(pl.col('rows','labeled_rows').fill_null(0),pl.lit(scope).alias('scope'),pl.lit(group).alias('group')))
    return pl.concat(result,how='vertical_relaxed').sort('group','date')


def summarize(daily):
    if daily['date'].n_unique()!=daily.height:
        raise ValueError('Gross summary requires one scope/group and unique dates')
    result=dict(rows=int(daily['rows'].sum()),labeled_rows=int(daily['labeled_rows'].sum()),
        labeled_days=daily.filter(pl.col('labeled_rows')>0).height,not_account_returns=True)
    for c in ['gross_industry','net_industry','net_absolute','cost','gross_rank_ic']:
        values=daily[c].to_numpy()
        result[c]=dict(mean=ranking._mean(values),ci95=ranking._interval(values),days=int(np.isfinite(values).sum()))
    return result
