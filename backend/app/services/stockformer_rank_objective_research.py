"""Bounded ordinal tree target with actual net-return calibration, offline only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_calibration_research as calibration
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_training as training

RANK_TARGET = 'rank_5d_industry_net'
POLICY = dict(version='industry_daily_net_rank_tree_actual_net_affine_v1',
    tree_target=RANK_TARGET,calibration_target=pair.TARGET,
    transform='finite_same_date_average_rank_minus_half_divided_by_count_minus_half',
    raw_score_unit='dimensionless_ordinal_score_not_return',
    actual_net_labels_unchanged=True,absolute_head_unchanged=True,threshold_search=False,
    activation_allowed=False)


def training_rank_labels(frame):
    if frame.select(pl.struct('symbol','date').is_duplicated().any()).item():
        raise ValueError('Duplicate rank-target identity')
    value=pl.when(pl.col(pair.TARGET).is_finite()).then(pl.col(pair.TARGET)).otherwise(None)
    ready=frame.with_columns(value.alias('_rank_training_value'))
    return ready.with_columns(((pl.col('_rank_training_value').rank(method='average').over('date')-.5)/
        pl.col('_rank_training_value').count().over('date')-.5).alias(RANK_TARGET)).drop('_rank_training_value')


def fit_rank_pair(root, features, columns, *, calibration_start, label_cut, context=None):
    root=Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Rank pair directory must be empty')
    root.mkdir(parents=True,exist_ok=True)
    ready,coverage=sf.prepare_training_features(features,'minute_enhanced')
    ready=ready.sort('date','symbol')
    train=training.purged_slice(ready,5,ready['date'].min(),calibration_start)
    x,_,used_train=sf._prepare_xy(train,pair.TARGET,columns)
    ranked=training_rank_labels(used_train)
    y=ranked[RANK_TARGET].to_numpy()
    if used_train['date'].n_unique()<100 or len(y)<1000:
        raise ValueError('Insufficient rank tree history')
    xgb=sf._fit_xgboost(x,y,n_estimators=300)
    lgb=sf._fit_lightgbm(x,y,n_estimators=300)
    sample=ready.filter((pl.col('date')>=calibration_start)&(pl.col('date')<label_cut)).with_columns(
        pl.when(pl.col('label_end_5d')<label_cut).then(pl.col(pair.TARGET)).otherwise(None).alias(pair.TARGET))
    cx,_,used=sf._prepare_xy(sample.with_columns(pl.lit(0.).alias('_prediction_mask')),'_prediction_mask',columns)
    raw=.5*sf._predict_xgboost(xgb,cx)+.5*sf._predict_lightgbm(lgb,cx)
    cal=calibration.with_percentile(used.select('symbol','date',pl.col(pair.TARGET).alias('target')).with_columns(pl.Series('raw',raw)))
    spec=calibration.fit_method(cal,'affine')
    xgb.save_model(str(root/'xgboost.json'))
    lgb.booster_.save_model(str(root/'lightgbm.txt'))
    (root/'calibration.json').write_text(json.dumps(spec,indent=2,allow_nan=False),encoding='utf-8')
    cal.write_parquet(root/'calibration-predictions.parquet')
    return pair.freeze_manifest(root,feature_columns=columns,provenance=dict(target_column=RANK_TARGET,
        calibration_target=pair.TARGET,rank_source_sha256=pair.digest(__file__),rank_policy=POLICY,
        tree_max_label_end=str(used_train['label_end_5d'].max()),calibration_start=str(calibration_start),
        calibration_max_label_end=str(used.filter(pl.col(pair.TARGET).is_finite())['label_end_5d'].max()),
        frozen_as_of=str(label_cut),train_rows=used_train.height,train_dates=used_train['date'].n_unique(),
        calibration_rows=spec['rows'],calibration_dates=spec['dates'],minute_coverage=coverage,n_estimators=300,
        cost_target_version=training.PAPER_COST_TARGET_VERSION,refit_after_calibration=False,context=context or {}))


def predict_rank_pair(root, features):
    manifest=pair.verify_manifest(root)
    provenance=manifest['provenance']
    if (provenance.get('target_column')!=RANK_TARGET or provenance.get('calibration_target')!=pair.TARGET
            or provenance.get('rank_source_sha256')!=pair.digest(__file__)):
        raise ValueError('Rank tree/calibration target or source mismatch')
    return pair.predict_pair(root,features)


def _errors(frame, column):
    valid=frame.filter(pl.col(column).is_finite()&pl.col('target').is_finite()).with_columns(
        (pl.col(column)-pl.col('target')).alias('_error'))
    if valid.is_empty():
        return dict(rows=0,dates=0,bias=None,mse=None,daily_bias_mse=None,within_day_error_mse=None,
            prediction_mean=None,target_mean=None)
    daily=valid.group_by('date').agg(pl.col('_error').mean().alias('bias'),
        pl.col('_error').pow(2).mean().alias('mse'),pl.col(column).mean().alias('prediction_mean'),
        pl.col('target').mean().alias('target_mean'))
    total=float(daily['mse'].mean())
    between=float(daily['bias'].pow(2).mean())
    return dict(rows=valid.height,dates=daily.height,bias=float(daily['bias'].mean()),mse=total,
        daily_bias_mse=between,within_day_error_mse=max(0.,total-between),
        prediction_mean=float(daily['prediction_mean'].mean()),target_mean=float(daily['target_mean'].mean()))


def calibration_diagnostic(calibration_frame, check_frame):
    values=calibration_frame.filter(pl.col('raw').is_finite())['raw'].to_numpy()
    if not len(values):
        raise ValueError('Missing finite calibration score support')
    low,high=np.quantile(values,[.01,.99]).tolist()
    finite=check_frame.filter(pl.col('raw').is_finite())
    return dict(calibration={c:_errors(calibration_frame,c) for c in ['raw','calibrated']},
        check={c:_errors(check_frame,c) for c in ['raw','calibrated']},
        support=dict(calibration_raw_q01=low,calibration_raw_q99=high,
            check_finite_rows=finite.height,below=int((finite['raw']<low).sum()),above=int((finite['raw']>high).sum())),
        support_bounds_are_diagnostics_not_trading_thresholds=True)
