"""Prediction-independent candidates and a fixed scoring reference for time OOF."""
from __future__ import annotations

import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_affordability_research as affordability
from app.services import stockformer_entry_feasibility_research as feasibility
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_training as training

KEYS = ['symbol','date']
SCORES = ('technical_score','risk_score','fund_score','sector_score','sentiment_score','fundamental_score')
POLICY = dict(version='necessary_candidates_fixed_scoring_reference_time_oof_v1',
    reference='known_day_affordable_nonst_identity_ready_complete_80_feature_pool',
    necessary='existing_maximum_entry_score_62_and_known_risk_rules_without_prediction_or_relative_entry_rank',
    fit_calibration_and_candidate_eligibility='same_necessary_pool',
    background_predictions='reference_scoring_only_never_candidate_or_calibration_observations',
    all_score_percentiles_amplitude_rank_forecast_rank_and_liquidity_scope='full_fixed_reference',
    market_eligibility='unchanged_final_gate_not_training_date_filter',
    minimum_effective_tree_sessions=252,minimum_calibration_sessions=20,
    fitting_selection_by_training_predictions=False,activation_allowed=False)


def reference_pool(data_dir, features, raw_prices, execution):
    reference,pool_report=affordability.affordable_pool(features,raw_prices,execution)
    reference,_=sf.prepare_training_features(reference,'minute_enhanced')
    columns=sf.training_feature_columns('minute_enhanced')
    _,_,reference=sf._prepare_xy(reference.with_columns(pl.lit(0.).alias('_complete_mask')),'_complete_mask',columns)
    identity=['name','_pit_name_known','_pit_input_ready']
    if any(c not in reference.columns for c in identity):
        raise ValueError('Missing historical reference identity')
    reference=sf._filter_special_treatment(reference.filter(
        pl.col('_pit_name_known').fill_null(False)&pl.col('_pit_input_ready').fill_null(False)))
    if reference.is_empty():
        raise ValueError('Empty fixed scoring reference')
    scored=sf._add_multifactor_scores(reference.with_columns(pl.lit(0.).alias('stockformer_score')))
    scored=sf._attach_market_gate_for_oos(data_dir,scored)
    scored=scored.with_columns(*[pl.col(c).alias('_reference_'+c) for c in SCORES],
        pl.col('entry_score').alias('_reference_entry_without_model'),
        pl.len().over('date').alias('_reference_rows'))
    scored=scored.with_columns((pl.col('entry_score')-
        pl.col('market_buy_threshold_add').cast(pl.Float64,strict=False).fill_null(0.)*.55).clip(0,100).alias('entry_score'))
    rank=pl.col('entry_score').rank(method='average').over('date')
    count=pl.len().over('date')
    scored=scored.with_columns((sf._entry_eligible_expr(scored)&((count<8)|(rank>=count*.35))).alias('_entry_rules_pass'))
    bounded=feasibility.feasibility(scored).with_columns(pl.col('_known_entry_necessary').alias('_necessary_entry_pool'))
    report=dict(policy=POLICY,reference_rows=bounded.height,reference_dates=bounded['date'].n_unique(),
        necessary_rows=int(bounded['_necessary_entry_pool'].sum()),
        necessary_dates=bounded.filter('_necessary_entry_pool')['date'].n_unique(),affordability=pool_report,
        per_date=bounded.group_by('date').agg(pl.len().alias('reference_rows'),
            pl.col('_necessary_entry_pool').sum().alias('necessary_rows')).sort('date').to_dicts())
    return bounded.sort('date','symbol'),report


def audited_fitting_frame(reference, window):
    eligible=reference.filter(pl.col('_necessary_entry_pool')&(pl.col('date')>=window['train_start'])&
        (pl.col('date')<window['validation_start']))
    joint=pl.col('target_5d_rel_net').is_finite()&pl.col('target_5d_net').is_finite()
    eligible=eligible.with_columns([pl.when(joint).then(pl.col(c)).otherwise(None).alias(c)
        for c in ['target_5d_rel_net','target_5d_net']])
    train=training.purged_slice(eligible,5,window['train_start'],window['calibration_start'])
    cal=eligible.filter((pl.col('date')>=window['calibration_start'])&
        (pl.col('label_end_5d')<window['validation_start']))
    columns=sf.training_feature_columns('minute_enhanced')
    _,_,used_train=sf._prepare_xy(train,'target_5d_rel_net',columns)
    _,_,used_cal=sf._prepare_xy(cal,'target_5d_rel_net',columns)
    if used_train['date'].n_unique()!=252 or used_train.height<1000:
        raise ValueError('Fixed OOF requires 252 effective tree sessions and 1000 rows')
    if used_cal['date'].n_unique()!=20 or used_cal.height<1000:
        raise ValueError('Fixed OOF requires 20 mature calibration sessions and 1000 rows')
    if not (used_train['label_end_5d'].max()<window['calibration_start'] and
            used_cal['label_end_5d'].max()<window['validation_start']):
        raise ValueError('OOF label overlap')
    if set(used_train['date'])&set(window['validation_dates']) or set(used_cal['date'])&set(window['validation_dates']):
        raise ValueError('OOF scoring dates overlap fitting dates')
    return eligible,dict(train_rows=used_train.height,train_dates=used_train['date'].n_unique(),
        calibration_rows=used_cal.height,calibration_dates=used_cal['date'].n_unique(),
        tree_max_label_end=used_train['label_end_5d'].max(),calibration_max_label_end=used_cal['label_end_5d'].max(),
        no_in_sample_predictions_in_selection=True,scoring_dates_excluded_from_fitting=True)


def reference_signals(data_dir, reference, prediction, *, min_net_edge, signal_policy):
    if not reference.select(KEYS).sort(KEYS).equals(prediction.select(KEYS).sort(KEYS)):
        raise ValueError('Fixed reference prediction universe mismatch')
    full=portfolio._build_gated_signal_frame(data_dir,prediction,reference,
        dict(target_mode='industry_relative_net',signal_weights={'5d':1.}),horizon=5,
        signal_policy=signal_policy,min_net_edge=min_net_edge,score_source='raw',historical_identity=True)
    if not reference.select(KEYS).sort(KEYS).equals(full.select(KEYS).sort(KEYS)):
        raise ValueError('Fixed reference identity universe changed')
    metadata=[c for c in reference.columns if c.startswith('_reference_')]+['_necessary_entry_pool']
    full=full.join(reference.select(*KEYS,*metadata),on=KEYS,validate='1:1')
    for column in SCORES:
        bad=full.select((~pl.col(column).is_finite()|~pl.col('_reference_'+column).is_finite()|
            ((pl.col(column)-pl.col('_reference_'+column)).abs()>1e-10)).any()).item()
        if bad:
            raise ValueError('Fixed reference score drift: '+column)
    expected=((pl.col('_reference_entry_without_model')+
        pl.col('stockformer_score')*feasibility.POLICY['model_entry_coefficient']).clip(0,100)-
        pl.col('market_buy_threshold_add').cast(pl.Float64,strict=False).fill_null(0.)*.55).clip(0,100)
    if full.select(((pl.col('entry_score')-expected).abs()>1e-10).any()).item():
        raise ValueError('Fixed reference entry score drift')
    if full.filter(pl.col('_entry_rules_pass')&~pl.col('_necessary_entry_pool')).height:
        raise ValueError('Actual entry escaped the necessary pool')
    return full,full.filter('_necessary_entry_pool')
