"""Time-excluded raw teachers and fixed net-return calibration, offline only."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from app.market_time import cn_now
from app.services import stockformer as sf
from app.services import stockformer_calibration_research as calibration
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_rank_objective_research as rank_objective
from app.services import stockformer_training as training

POLICY = dict(version='oof_percentile_net_isotonic_state_shrink_v1',oof_sessions=84,teacher_block_sessions=21,
    teacher_mature_training_sessions=252,settlement_embargo_sessions=6,
    methods=['affine_percentile','isotonic_state'],minimum_map_rows=1000,minimum_map_dates=20,
    state_minimum_rows=1000,state_minimum_dates=20,state_pseudo_dates=20,
    percentile_scope='full_fixed_scoring_reference_before_candidate_or_label_filter',
    state='existing_known_day_market_gate',unsupported_state='global_mapping_without_offset',
    actual_net_labels_and_trading_thresholds_unchanged=True,activation_allowed=False)


def _unique(frame):
    if frame.select(pl.struct('symbol','date').is_duplicated().any()).item():
        raise ValueError('Duplicate OOF observation identity')


def _identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _sources():
    return {m.__name__:pair.digest(m.__file__) for m in [sf,training,rank_objective]}


def fit_teacher(root, features, columns, *, start, before, head, context=None):
    if head not in {'industry_rank','absolute_net'}:
        raise ValueError('Unknown OOF teacher head')
    root=Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Teacher directory must be empty')
    root.mkdir(parents=True,exist_ok=True)
    ready,_=sf.prepare_training_features(features,'minute_enhanced')
    train=training.purged_slice(ready.sort('date','symbol'),5,start,before)
    target='target_5d_rel_net' if head=='industry_rank' else 'target_5d_net'
    x,y,used=sf._prepare_xy(train,target,columns)
    if used['date'].n_unique()!=252 or used.height<1000:
        raise ValueError('OOF teacher requires 252 mature training dates and 1000 rows')
    if head=='industry_rank':
        y=rank_objective.training_rank_labels(used)[rank_objective.RANK_TARGET].to_numpy()
    xgb=sf._fit_xgboost(x,y,n_estimators=300)
    lgb=sf._fit_lightgbm(x,y,n_estimators=300)
    xgb.save_model(str(root/'xgboost.json'))
    lgb.booster_.save_model(str(root/'lightgbm.txt'))
    meta=dict(version='raw_oof_teacher_v1',trained_at=cn_now().isoformat(),head=head,
        tree_target=rank_objective.RANK_TARGET if head=='industry_rank' else target,
        feature_columns=list(columns),minute_policy=sf.MINUTE_AVAILABILITY_POLICY,feature_version=sf.FEATURE_VERSION,
        train_dates=used['date'].n_unique(),train_rows=used.height,start=str(start),score_not_before=str(before),
        max_label_end=str(used['label_end_5d'].max()),sources=_sources(),teacher_source_sha256=pair.digest(__file__),
        files={f:pair.digest(root/f) for f in ['xgboost.json','lightgbm.txt']},context=context or {},
        calibrated=False,activation_allowed=False)
    meta=sf._json_safe(meta)
    meta['teacher_id']=_identity(meta)
    (root/'teacher.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
    return meta


def verify_teacher(root):
    root=Path(root)
    meta=json.loads((root/'teacher.json').read_text())
    if meta['teacher_id']!=_identity({k:v for k,v in meta.items() if k!='teacher_id'}):
        raise ValueError('OOF teacher identity mismatch')
    if meta['sources']!=_sources() or meta['teacher_source_sha256']!=pair.digest(__file__):
        raise ValueError('OOF teacher source mismatch')
    if any(pair.digest(root/f)!=sha for f,sha in meta['files'].items()):
        raise ValueError('OOF teacher artifact mismatch')
    if date.fromisoformat(meta['max_label_end'])>=date.fromisoformat(meta['score_not_before']):
        raise ValueError('OOF teacher label overlap')
    return meta


def predict_teacher(root, features):
    meta=verify_teacher(root)
    if features.height and features['date'].min()<date.fromisoformat(meta['score_not_before']):
        raise ValueError('OOF teacher cannot score its training dates')
    ready,_=sf.prepare_training_features(features,'minute_enhanced')
    x,_,used=sf._prepare_xy(ready.sort('date','symbol').with_columns(pl.lit(0.).alias('_mask')),'_mask',meta['feature_columns'])
    xgb,lgb=pair._load_trees(root)
    raw=.5*sf._predict_xgboost(xgb,x)+.5*sf._predict_lightgbm(lgb,x)
    return used.select('symbol','date').with_columns(pl.Series('raw',raw))


def frozen_raw_prediction(root, features):
    """Use existing trees without applying the parent's later-fitted calibration."""
    meta=pair.verify_manifest(root)
    if features.height and features['date'].min()<=date.fromisoformat(meta['provenance']['tree_max_label_end']):
        raise ValueError('Frozen tree labels overlap validation dates')
    ready,_=sf.prepare_training_features(features,'minute_enhanced')
    x,_,used=sf._prepare_xy(ready.sort('date','symbol').with_columns(pl.lit(0.).alias('_mask')),'_mask',meta['feature_columns'])
    xgb,lgb=pair._load_trees(root)
    raw=.5*sf._predict_xgboost(xgb,x)+.5*sf._predict_lightgbm(lgb,x)
    return used.select('symbol','date').with_columns(pl.Series('raw',raw))


def percentile_candidates(prediction, reference, *, candidate_only=True):
    _unique(prediction)
    _unique(reference)
    keys=['symbol','date']
    if not prediction.select(keys).sort(keys).equals(reference.select(keys).sort(keys)):
        raise ValueError('OOF scoring reference universe mismatch')
    ranked=calibration.with_percentile(prediction)
    frame=ranked.join(reference.select(*keys,'market_gate','_necessary_entry_pool'),on=keys,validate='1:1')
    return frame.filter('_necessary_entry_pool') if candidate_only else frame


def fit_map(frame, method, *, before):
    if method not in POLICY['methods']:
        raise ValueError('Unregistered OOF calibration method')
    _unique(frame)
    if frame.filter(~pl.col('percentile').is_finite()|(pl.col('percentile')<0)|(pl.col('percentile')>1)).height:
        raise ValueError('Invalid calibration percentile')
    valid=frame.filter(pl.col('target').is_finite())
    if valid.filter(pl.col('label_end_5d').is_null()|(pl.col('label_end_5d')>=before)|(pl.col('date')>=before)).height:
        raise ValueError('OOF calibration labels must be mature before validation')
    if valid.height<1000 or valid['date'].n_unique()<20:
        raise ValueError('Insufficient OOF calibration rows/dates')
    if valid.group_by('date').agg(pl.col('market_gate').n_unique().alias('states')).filter(pl.col('states')!=1).height:
        raise ValueError('Market state must be fixed for each date')
    data=valid.with_columns(pl.col('percentile').alias('raw'))
    base_method='affine' if method=='affine_percentile' or valid['percentile'].n_unique()<2 else 'isotonic_rank'
    base=calibration.fit_method(data,base_method)
    result=dict(version=POLICY['version'],method=method,policy=POLICY,base=base,states={},
        rows=valid.height,dates=valid['date'].n_unique(),first=str(valid['date'].min()),last=str(valid['date'].max()),
        max_label_end=str(valid['label_end_5d'].max()),frozen_before=str(before),source_sha256=pair.digest(__file__),
        targets_are_cost_after_returns=True,activation_allowed=False)
    if method=='isotonic_state':
        residual=data.with_columns(pl.Series('_residual',data['target'].to_numpy()-calibration.apply_method(data,base)))
        for state in sorted(residual['market_gate'].unique().to_list()):
            sample=residual.filter(pl.col('market_gate')==state)
            days=sample['date'].n_unique()
            supported=days>=20 and sample.height>=1000
            mean=sample.group_by('date').agg(pl.col('_residual').mean())['_residual'].mean()
            offset=float(mean*days/(days+POLICY['state_pseudo_dates'])) if supported else 0.
            result['states'][state]=dict(rows=sample.height,dates=days,supported=supported,offset=offset)
    return result


def apply_map(frame, spec):
    if spec.get('source_sha256')!=pair.digest(__file__) or spec.get('version')!=POLICY['version']:
        raise ValueError('OOF calibration source/version mismatch')
    data=frame.with_columns(pl.col('percentile').alias('raw'))
    values=calibration.apply_method(data,spec['base'])
    if spec['method']=='isotonic_state':
        values=values+np.asarray([spec['states'].get(s,{}).get('offset',0.) for s in frame['market_gate']])
    return values
