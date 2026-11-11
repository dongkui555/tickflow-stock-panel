"""Immutable research tree/calibrator pairs; no production registration."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path

import polars as pl

from app.market_time import cn_now
from app.services import minute_features
from app.services import stockformer as sf
from app.services import stockformer_calibration_research as calibration
from app.services import stockformer_training as training

FILES = ('xgboost.json', 'lightgbm.txt', 'calibration.json')
TARGET = 'target_5d_rel_net'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _source_hashes():
    return {module.__name__: digest(module.__file__) for module in (sf, calibration, training, minute_features)}


def _identity(manifest):
    return hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != 'pair_id'},
        sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def freeze_manifest(root, *, feature_columns, provenance):
    root = Path(root)
    path = root/'pair.json'
    if path.exists():
        raise ValueError('Frozen pair already exists')
    if not (date.fromisoformat(provenance['tree_max_label_end']) < date.fromisoformat(provenance['calibration_start'])
            and date.fromisoformat(provenance['calibration_max_label_end']) < date.fromisoformat(provenance['frozen_as_of'])):
        raise ValueError('Tree/calibration label overlap')
    mapping = json.loads((root/'calibration.json').read_text())
    if (mapping.get('method') != 'affine' or mapping.get('slope', -1) < 0
            or not all(math.isfinite(mapping.get(k, float('nan'))) for k in ('slope', 'intercept'))):
        raise ValueError('Frozen experiment requires nondecreasing affine calibration')
    manifest = {'version': 'minute_complete_frozen_tree_calibration_v2', 'frozen_at': cn_now().isoformat(),
        'feature_columns': list(feature_columns), 'feature_version': sf.FEATURE_VERSION,
        'row_order': ['date', 'symbol'],
        'minute_policy': dict(sf.MINUTE_AVAILABILITY_POLICY), 'calibration_method': 'affine',
        'weights': {'xgboost': .5, 'lightgbm': .5}, 'files': {name: digest(root/name) for name in FILES},
        'source_sha256': _source_hashes(), 'pair_source_sha256': digest(__file__),
        'provenance': provenance, 'activation_allowed': False}
    manifest = sf._json_safe(manifest)
    manifest['pair_id'] = _identity(manifest)
    path.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    return manifest


def verify_manifest(root):
    root = Path(root)
    manifest = json.loads((root/'pair.json').read_text())
    if manifest.get('pair_id') != _identity(manifest):
        raise ValueError('Frozen pair manifest fingerprint mismatch')
    if manifest.get('minute_policy') != sf.MINUTE_AVAILABILITY_POLICY:
        raise ValueError('Frozen minute policy mismatch')
    if (manifest['source_sha256'] != _source_hashes()
            or manifest['pair_source_sha256'] != digest(__file__)):
        raise ValueError('Frozen pair source fingerprint mismatch')
    if set(manifest['files']) != set(FILES) or any(digest(root/name) != manifest['files'][name] for name in FILES):
        raise ValueError('Frozen tree/calibration fingerprint mismatch')
    return manifest


def _load_trees(root):
    import lightgbm as lgb
    from xgboost import XGBRegressor

    xgb = XGBRegressor()
    xgb.load_model(str(Path(root)/'xgboost.json'))
    return xgb, lgb.Booster(model_file=str(Path(root)/'lightgbm.txt'))


def fit_pair(root, features, columns, *, calibration_start, label_cut, context=None):
    """Fit once before calibration, then freeze the exact calibrated trees."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Pair fit directory must be empty')
    root.mkdir(parents=True, exist_ok=True)
    ready, coverage = sf.prepare_training_features(features, 'minute_enhanced')
    # Seeded tree subsampling still depends on input order. Canonicalize before
    # fitting so repartitioning the same observations cannot change the model.
    ready = ready.sort('date', 'symbol')
    train = training.purged_slice(ready, 5, ready['date'].min(), calibration_start)
    x, y, used_train = sf._prepare_xy(train, TARGET, columns)
    if used_train['date'].n_unique() < 100 or len(y) < 1000:
        raise ValueError('Insufficient complete-minute tree training dates')
    xgb, lgb = sf._fit_xgboost(x, y, n_estimators=300), sf._fit_lightgbm(x, y, n_estimators=300)
    sample = ready.filter((pl.col('date') >= calibration_start) & (pl.col('date') < label_cut)).with_columns(
        pl.when(pl.col('label_end_5d') < label_cut).then(pl.col(TARGET)).otherwise(None).alias(TARGET))
    cx, _, used = sf._prepare_xy(sample.with_columns(pl.lit(0.).alias('_prediction_mask')), '_prediction_mask', columns)
    if used.is_empty():
        raise ValueError('No complete calibration observations')
    raw = .5*sf._predict_xgboost(xgb, cx)+.5*sf._predict_lightgbm(lgb, cx)
    calibration_frame = calibration.with_percentile(used.select('symbol', 'date', pl.col(TARGET).alias('target')).with_columns(
        pl.Series('raw', raw)))
    mapping = calibration.fit_method(calibration_frame, 'affine')
    xgb.save_model(str(root/'xgboost.json'))
    lgb.booster_.save_model(str(root/'lightgbm.txt'))
    (root/'calibration.json').write_text(json.dumps(mapping, allow_nan=False, indent=2), encoding='utf-8')
    calibration_frame.write_parquet(root/'calibration-predictions.parquet')
    return freeze_manifest(root, feature_columns=columns, provenance={
        'tree_max_label_end': str(used_train['label_end_5d'].max()),
        'calibration_start': str(calibration_start), 'calibration_max_label_end':
            str(used.filter(pl.col(TARGET).is_finite())['label_end_5d'].max()),
        'frozen_as_of': str(label_cut), 'train_rows': used_train.height,
        'train_dates': used_train['date'].n_unique(), 'calibration_rows': mapping['rows'],
        'calibration_dates': mapping['dates'], 'minute_coverage': coverage,
        'n_estimators': 300, 'cost_target_version': training.PAPER_COST_TARGET_VERSION,
        'refit_after_calibration': False, 'context': context or {}})


def predict_pair(root, features):
    """Verify artifacts before evaluating a label-free complete cross section."""
    manifest = verify_manifest(root)
    ready, coverage = sf.prepare_training_features(features, 'minute_enhanced')
    ready = ready.sort('date', 'symbol')
    x, _, used = sf._prepare_xy(ready.with_columns(pl.lit(0.).alias('_prediction_mask')),
                              '_prediction_mask', manifest['feature_columns'])
    if used.is_empty():
        return pl.DataFrame(schema={'symbol': pl.String, 'date': pl.Date, 'pred_ret_raw_5d': pl.Float64,
                                    'pred_ret_5d': pl.Float64}), coverage
    xgb, lgb = _load_trees(root)
    raw = .5*sf._predict_xgboost(xgb, x)+.5*sf._predict_lightgbm(lgb, x)
    frame = used.select('symbol', 'date').with_columns(pl.Series('raw', raw))
    mapping = json.loads((Path(root)/'calibration.json').read_text())
    corrected = calibration.apply_method(frame, mapping)
    return frame.rename({'raw': 'pred_ret_raw_5d'}).with_columns(pl.Series('pred_ret_5d', corrected)), coverage
