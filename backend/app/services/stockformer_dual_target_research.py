"""Offline absolute-net head and fixed joint gates around existing frozen pairs."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_calibration_research as calibration
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_training as training

ABSOLUTE_TARGET = 'target_5d_net'
INDUSTRY_TARGET = pair.TARGET
POLICY = {
    'version': 'absolute_net_and_industry_net_fixed_conjunction_v1',
    'absolute_target': ABSOLUTE_TARGET,
    'industry_target': INDUSTRY_TARGET,
    'both_calibrated_positive_and_both_net_margin_required': True,
    'ordering': 'unchanged_raw_industry_prediction',
    'entry_and_market_rules': 'unchanged_industry_head_inputs_and_predicates',
    'absolute_cost_mode': 'net_no_second_fixed_cost',
    'threshold_selection': False,
}


def fit_absolute_pair(root, features, columns, *, calibration_start, label_cut, context=None):
    """Use the existing estimators/calibrator; keep the actual absolute target named."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Absolute pair directory must be empty')
    root.mkdir(parents=True, exist_ok=True)
    ready, coverage = sf.prepare_training_features(features, 'minute_enhanced')
    ready = ready.sort('date', 'symbol')
    train = training.purged_slice(ready, 5, ready['date'].min(), calibration_start)
    x, y, used_train = sf._prepare_xy(train, ABSOLUTE_TARGET, columns)
    if used_train['date'].n_unique() < 100 or len(y) < 1000:
        raise ValueError('Insufficient absolute tree training history')
    xgb = sf._fit_xgboost(x, y, n_estimators=300)
    lgb = sf._fit_lightgbm(x, y, n_estimators=300)
    sample = ready.filter((pl.col('date')>=calibration_start)&(pl.col('date')<label_cut)).with_columns(
        pl.when(pl.col('label_end_5d')<label_cut).then(pl.col(ABSOLUTE_TARGET)).otherwise(None).alias(ABSOLUTE_TARGET))
    cx, _, used = sf._prepare_xy(sample.with_columns(pl.lit(0.).alias('_mask')), '_mask', columns)
    if used.is_empty():
        raise ValueError('No absolute calibration observations')
    raw = .5*sf._predict_xgboost(xgb,cx)+.5*sf._predict_lightgbm(lgb,cx)
    cal = calibration.with_percentile(used.select('symbol','date',pl.col(ABSOLUTE_TARGET).alias('target')).with_columns(
        pl.Series('raw',raw)))
    mapping = calibration.fit_method(cal,'affine')
    xgb.save_model(str(root/'xgboost.json'))
    lgb.booster_.save_model(str(root/'lightgbm.txt'))
    (root/'calibration.json').write_text(json.dumps(mapping,allow_nan=False,indent=2),encoding='utf-8')
    cal.write_parquet(root/'calibration-predictions.parquet')
    return pair.freeze_manifest(root,feature_columns=columns,provenance=dict(
        target_column=ABSOLUTE_TARGET,target_mode='net',head_source_sha256=pair.digest(__file__),
        tree_max_label_end=str(used_train['label_end_5d'].max()),calibration_start=str(calibration_start),
        calibration_max_label_end=str(used.filter(pl.col(ABSOLUTE_TARGET).is_finite())['label_end_5d'].max()),
        frozen_as_of=str(label_cut),train_rows=used_train.height,train_dates=used_train['date'].n_unique(),
        calibration_rows=mapping['rows'],calibration_dates=mapping['dates'],minute_coverage=coverage,
        n_estimators=300,cost_target_version=training.PAPER_COST_TARGET_VERSION,
        refit_after_calibration=False,context=context or {}))


def predict_head(root, features, expected_target):
    manifest = pair.verify_manifest(root)
    actual = manifest['provenance'].get('target_column', INDUSTRY_TARGET)
    if expected_target not in {ABSOLUTE_TARGET,INDUSTRY_TARGET} or actual!=expected_target:
        raise ValueError('Research head target mismatch')
    if actual==ABSOLUTE_TARGET and manifest['provenance'].get('head_source_sha256')!=pair.digest(__file__):
        raise ValueError('Absolute head source changed')
    return pair.predict_pair(root,features)


def joint_signal(industry_signal, absolute_prediction, *, min_net_edge):
    """Keep the industry ranking/risk rules; require a separate absolute net edge."""
    keys = ['symbol','date']
    for frame in [industry_signal,absolute_prediction]:
        if frame.select(pl.struct(keys).is_duplicated().any()).item():
            raise ValueError('Duplicate dual head identity')
    if not industry_signal.select(keys).sort(keys).equals(absolute_prediction.select(keys).sort(keys)):
        raise ValueError('Dual head prediction universe mismatch')
    absolute = absolute_prediction.select(*keys,pl.col('pred_ret_raw_5d').alias('absolute_raw_5d'),
        pl.col('pred_ret_5d').alias('absolute_net_5d'))
    frame = industry_signal.join(absolute,on=keys,how='left',validate='1:1')
    # The absolute target already includes matcher fees/slippage. Reuse only
    # the existing liquidity surcharge and minimum edge, never charge it twice.
    adjusted = portfolio._attach_net_margin(frame,pred_col='absolute_net_5d',
        min_net_edge=min_net_edge,target_mode='net')
    frame = frame.with_columns(adjusted['net_margin_score'].alias('absolute_net_margin_score'),
        (adjusted['_net_margin_pass']&frame['absolute_net_5d'].is_finite().fill_null(False)).alias('_absolute_net_margin_pass'),
        (pl.col('absolute_net_5d').is_finite()&(pl.col('absolute_net_5d')>0)).fill_null(False).alias('_absolute_positive'))
    return frame.with_columns(
        pl.col('_forecast_pass').alias('_industry_forecast_pass'),
        (pl.col('_prediction_pass')&pl.col('_absolute_positive')).alias('_prediction_pass'),
        (pl.col('_net_margin_pass')&pl.col('_absolute_net_margin_pass')).alias('_net_margin_pass'),
        (pl.col('_forecast_pass')&pl.col('_absolute_positive')&pl.col('_absolute_net_margin_pass')).alias('_forecast_pass'))
