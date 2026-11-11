"""Frozen five-day attribution plus an opt-in paper-cost research control.

No model registration, activation, original artifact writes or threshold search.
Old holdout analysis is descriptive; fixed ridge controls use pre-holdout only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import polars as pl
from app.market_time import cn_now
from app.services import stockformer as sf
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_training as training
from app.strategy import paper


def write(path, value):
    path.write_text(json.dumps(sf._json_safe(value), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(frame, prediction='pred_ret_5d', actual='target_5d'):
    selected = frame.filter(pl.col(prediction).is_finite() & pl.col(actual).is_finite())
    if selected.is_empty():
        return {'rows': 0}
    result = training.metrics(selected, selected[prediction].to_numpy(), selected[actual].to_numpy(),
                              target_mode='industry_relative_net')
    result.update(positive_predictions=selected.filter(pl.col(prediction) > 0).height,
                  prediction_min=selected[prediction].min(), prediction_max=selected[prediction].max())
    return result


def fixed_ridge_controls(legacy, aligned, meta, payload, out):
    """Reproduce selected ridge folds, then change costs without reselecting."""
    spec = meta['validation']['horizons']['5d']
    columns = payload['linear_baseline_models']['5']['feature_columns']
    old_target = 'target_5d_rel_net'
    selected = aligned.select('symbol', 'date', pl.col(old_target).alias('_paper_y'))
    joined = legacy.join(selected, on=['symbol', 'date'], how='left')
    results = []
    predictions = []
    for index, fold in enumerate(meta['validation']['folds']['5d']):
        start, end = date.fromisoformat(fold['start']), date.fromisoformat(fold['end_exclusive'])
        train = training.purged_slice(joined, 5, joined['date'].min(), start)
        test = training.purged_slice(joined, 5, start, end)
        assert train['label_end_5d'].max() < start
        x, y, used_train = sf._prepare_xy(train, old_target, columns)
        tx, ty, used = sf._prepare_xy(test, old_target, columns)
        weights, scaler = sf._fit_ridge(x, y, spec['alpha'], len(columns), clip_quantiles=(1, 99))
        pred = sf._predict(tx, weights, scaler)
        reproduced = training.metrics(used, pred, ty, target_mode='industry_relative_net')
        expected = meta['validation']['metrics']['5d']['folds'][index]
        assert abs(reproduced['top2_after_cost_proxy'] - expected['top2_after_cost_proxy']) < 1e-9
        assert abs(reproduced['rank_ic'] - expected['rank_ic']) < 1e-9
        frame = used.select('symbol', 'date', pl.col(old_target).alias('old_y'), pl.col('_paper_y').alias('paper_y')).with_columns(pl.Series('prediction', pred))
        same = frame.filter(pl.col('paper_y').is_finite())
        cost_only = {
            'legacy_labels_same_affordable_rows': summary(same, 'prediction', 'old_y'),
            'paper_labels_same_predictions_and_rows': summary(same, 'prediction', 'paper_y'),
        }
        nx, ny, ntrain = sf._prepare_xy(train, '_paper_y', columns)
        ntx, nty, nused = sf._prepare_xy(test, '_paper_y', columns)
        nw, ns = sf._fit_ridge(nx, ny, spec['alpha'], len(columns), clip_quantiles=(1, 99))
        npred = sf._predict(ntx, nw, ns)
        new_score = training.metrics(nused, npred, nty, target_mode='industry_relative_net')
        write(out/f'paper-cost-ridge-fold-{index+1}.json', {'kind': 'research_control_not_serving_model',
              'fold': index+1, 'trained_before': str(start), 'max_label_end': str(ntrain['label_end_5d'].max()),
              'feature_columns': columns, 'alpha': spec['alpha'], 'weights': nw.tolist(), 'scaler': ns,
              'label_cost_version': training.PAPER_COST_TARGET_VERSION})
        predictions.append(nused.select('symbol', 'date', pl.col('_paper_y').alias('target_5d')).with_columns(
            pl.Series('pred_ret_5d', npred), pl.lit(index+1).alias('fold')))
        # Audit the original tree-fitting inner split; this is structural
        # overlap evidence, not a causal estimate of its performance effect.
        dates = sorted(train['date'].unique().to_list())
        inner_cut = dates[max(1, int(len(dates)*.8))]
        overlap = train.filter((pl.col('date') < inner_cut) & (pl.col('label_end_5d') >= inner_cut))
        results.append({'fold': index+1, 'start': str(start), 'end_exclusive': str(end),
                        'original_reproduced': reproduced, 'cost_only_control': cost_only,
                        'paper_cost_fixed_ridge': new_score,
                        'train_rows': used_train.height, 'paper_train_rows': ntrain.height,
                        'validation_rows': used.height, 'paper_validation_rows': nused.height,
                        'original_inner_split_overlap_rows': overlap.height,
                        'original_inner_split_overlap_dates': overlap['date'].n_unique()})
        print(json.dumps({'stage': 'fixed_ridge_fold_complete', 'fold': index+1,
                          'old_top2': reproduced['top2_after_cost_proxy'],
                          'paper_top2': new_score['top2_after_cost_proxy']}), flush=True)
    pl.concat(predictions).write_parquet(out/'paper-cost-ridge-validation-predictions.parquet')
    return {'kind': 'pre_holdout_fixed_family_alpha_ridge_control_not_a_gbdt_winner',
            'family': spec['family'], 'alpha': spec['alpha'], 'folds': results,
            'thresholds_selected': False, 'feature_families_reselected': False,
            'holdout_used_for_training_or_selection': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Live model artifacts, mounted read-only')
    parser.add_argument('--reference-dir', type=Path, required=True, help='Pinned Oct16 research inputs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    args = parser.parse_args()
    root, reference, out = args.data_dir, args.reference_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise ValueError('Completed research result exists; preserve it')
    model = root/'stockformer/models'/args.model_id
    protected = {str(p): digest(p) for p in [*model.glob('*'), root/'stockformer/latest_model.txt',
                                            root/'stockformer/daily_shadow/config.json'] if p.is_file()}
    write(out/'protected-before.json', protected)
    meta = sf._model_meta(root, args.model_id)
    payload = sf._load_model_payload(sf._resolve_artifact_path(root, meta['model_file']))
    cfg = json.loads((root/'stockformer/daily_shadow/config.json').read_text())
    execution = cfg['protocol']['execution']
    write(out/'research-protocol.json', {'fixed_before_run': cn_now(), 'model_id': args.model_id,
          'execution': execution, 'horizons': [5], 'target_mode': 'industry_relative_net',
          'family': meta['validation']['horizons']['5d']['family'],
          'alpha': meta['validation']['horizons']['5d']['alpha'],
          'validation_end_exclusive': meta['oos_start'], 'status': 'isolated_research_only',
          'label_cost_version': training.PAPER_COST_TARGET_VERSION,
          'old_holdout_policy': 'diagnosis_only_no_threshold_or_model_selection'})
    print(json.dumps({'stage': 'read_pinned_inputs', 'started_at': str(cn_now())}), flush=True)
    panel = pl.read_parquet(reference/'training-panel.parquet')
    legacy = pl.read_parquet(reference/'training-features-labels-corrected.parquet')
    raw, conflicts = training.load_paper_raw_prices(reference/'data', panel['symbol'].unique().to_list())
    conflicts.write_parquet(out/'excluded-raw-identity-conflicts.parquet')
    all_features = sf._feature_frame(panel, include_targets=False)
    cost = training.paper_execution_targets(all_features, panel, raw, execution=execution, horizons=(5,))
    # Do not alter the original minute gate, imputations, benchmark membership,
    # feature values or symbol pool. Join only new label fields onto pinned X.
    target_columns = ['symbol', 'date', 'target_5d', 'target_5d_rel', 'target_5d_net', 'target_5d_rel_net',
                      'label_paper_qty', 'label_cost_5d', 'label_paper_status_5d', 'label_cost_version']
    cost = cost.select(target_columns)
    check = legacy.select('symbol', 'date', pl.col('target_5d_rel').alias('_legacy_rel')).join(
        cost.select('symbol', 'date', 'target_5d_rel'), on=['symbol', 'date'], how='left')
    assert check.filter((pl.col('_legacy_rel') - pl.col('target_5d_rel')).abs() > 1e-9).is_empty(), 'Industry benchmark changed'
    new = legacy.drop([c for c in target_columns if c not in ('symbol', 'date') and c in legacy.columns]).join(cost, on=['symbol', 'date'], how='left')
    assert new.height == legacy.height
    new.write_parquet(out/'research-features-paper-cost-v1.parquet')
    print(json.dumps({'stage': 'paper_cost_labels_ready', 'rows': new.height}), flush=True)
    pre_holdout = new.filter(pl.col('date') < date.fromisoformat(meta['oos_start']))
    cost_summary = pre_holdout.group_by('label_paper_status_5d').len().sort('len', descending=True).to_dicts()
    priced = pre_holdout.filter(pl.col('label_cost_5d').is_finite())
    controls = fixed_ridge_controls(legacy, new, meta, payload, out)
    write(out/'pre-holdout-cost-controls.json', controls)
    signals = pl.read_parquet(reference/f'fixed_signals_{args.model_id}.parquet')
    distributions = {'raw': summary(signals, 'pred_ret_raw_5d'), 'calibrated': summary(signals)}
    strata = []
    for key in ('market_gate',):
        for group in signals.partition_by(key, maintain_order=True):
            strata.append({'group': key, 'value': str(group[key][0]),
                           'raw': summary(group, 'pred_ret_raw_5d'), 'calibrated': summary(group)})
    calibration = payload['models']['5']['prediction_calibration']
    low, high = calibration['prediction_low_quantile'], calibration['prediction_high_quantile']
    tails = signals.with_columns(pl.when(pl.col('pred_ret_raw_5d') <= low).then(pl.lit('low_tail'))
             .when(pl.col('pred_ret_raw_5d') >= high).then(pl.lit('high_tail')).otherwise(pl.lit('middle')).alias('calibration_segment'))
    for group in tails.partition_by('calibration_segment', maintain_order=True):
        strata.append({'group': 'calibration_segment', 'value': group['calibration_segment'][0],
                       'raw': summary(group, 'pred_ret_raw_5d'), 'calibrated': summary(group)})
    ranking = signals.group_by('date').agg(pl.corr(pl.col('pred_ret_raw_5d').rank(), pl.col('pred_ret_5d').rank()).alias('raw_calibrated_rank_correlation'))
    # Today's available prior-close reconstruction is a diagnostic, not a
    # timestamped pre-entry observation. Keep it out of live shadow artifacts.
    latest_path = root.parent/'unavailable'
    latest = None
    read_only_check = reference.parent/'daily-shadow-20261020/readonly-check-signals.parquet'
    if read_only_check.exists():
        latest_path = read_only_check
        latest_frame = pl.read_parquet(latest_path)
        day = latest_frame['date'].max()
        daily_raw, _ = training.load_paper_raw_prices(root, latest_frame['symbol'].to_list(), start=day, end=day)
        daily_raw = daily_raw.filter(pl.col('date') == day)
        latest_frame = latest_frame.join(daily_raw.select('symbol', pl.col('close').alias('_raw_close')), on='symbol', how='left')
        def extra_cost(price):
            qty = paper.qty_from_amount(execution['initial_cash']/execution['max_positions'] - paper.MIN_COMMISSION,
                                       price*(1+execution['slippage_bps']/10000))
            if qty <= 0:
                return None
            buy, sell = paper.apply_slippage(price, 'buy', execution['slippage_bps']), paper.apply_slippage(price, 'sell', execution['slippage_bps'])
            paid = qty*buy + paper.buy_fee(qty, buy, execution['commission_pct'])
            received = qty*sell - paper.sell_fee(qty, sell, execution['commission_pct'], execution['stamp_tax_pct'])
            return -(received-paid)/paid - training.DEFAULT_COST_AFTER_RETURN
        latest_frame = latest_frame.with_columns(pl.col('_raw_close').map_elements(extra_cost, return_dtype=pl.Float64).alias('diagnostic_extra_cost'))
        latest_frame = latest_frame.with_columns((pl.col('net_margin_score') - pl.col('diagnostic_extra_cost')).alias('diagnostic_paper_margin'))
        latest_frame.write_parquet(out/'latest-reconstruction-cost-diagnostic.parquet')
        latest = {'as_of': str(day), 'kind': 'late_reconstruction_and_flat_price_fee_sensitivity_not_new_scores',
                  'rows': latest_frame.height, 'raw_positive': latest_frame.filter(pl.col('pred_ret_raw_5d') > 0).height,
                  'calibrated_positive': latest_frame.filter(pl.col('pred_ret_5d') > 0).height,
                  'affordable_rows': latest_frame.filter(pl.col('diagnostic_extra_cost').is_finite()).height,
                  'maximum_calibrated': latest_frame['pred_ret_5d'].max(),
                  'old_net_margin_pass': latest_frame.filter(pl.col('net_margin_score') >= .0005).height,
                  'paper_cost_diagnostic_margin_pass': latest_frame.filter(pl.col('diagnostic_paper_margin') >= .0005).height}
    write(out/'historical-strata-diagnosis.json', {'scope': 'old_holdout_diagnosis_only', 'metrics': distributions,
          'strata': strata, 'calibration': calibration, 'daily_mean_raw_calibrated_rank_correlation': ranking['raw_calibrated_rank_correlation'].mean(),
          'layer_stats': portfolio._gated_layer_stats(signals, 'pred_ret_5d')})
    protected_after = {p: digest(Path(p)) for p in protected}
    assert protected == protected_after, 'Protected model or shadow policy changed'
    report = {'completed_at': cn_now(), 'model_id': args.model_id, 'signal_weights': meta['signal_weights'],
              'official_block': 'frozen_horizon_weights_empty',
              'weight_selection_model': 'ridge', 'serving_model': payload['model_kind'],
              'original_five_day_validation': meta['validation']['metrics']['5d'],
              'calibration': calibration, 'old_holdout_diagnostic': distributions,
              'pre_holdout_cost_statuses': cost_summary, 'raw_conflict_keys_excluded': conflicts.height,
              'pre_holdout_cost_quantiles': {str(q): priced['label_cost_5d'].quantile(q) for q in (0., .1, .5, .9, 1.)},
              'fixed_ridge_controls': controls, 'latest_reconstruction_diagnostic': latest,
              'cost_model': execution, 'cost_target_version': training.PAPER_COST_TARGET_VERSION,
              'benchmark_unchanged': True, 'production_and_shadow_artifacts_unchanged': True,
              'activation_allowed': False, 'original_labels_overwritten': False,
              'limitations': ['Conditional fill labels do not replace T+1/limit/cash/portfolio matching.',
                             'Adjusted/raw return divergence above 1bp is excluded; no synthetic corporate-action cash flows.',
                             'Pre-holdout ridge controls are not executable GBDT profitability evidence.',
                             'Fixed prior holdout is descriptive only; no threshold tuning or model selection.']}
    write(out/'report.json', report)
    write(out/'protected-after.json', protected_after)
    print(json.dumps(sf._json_safe({k: report[k] for k in ('completed_at', 'pre_holdout_cost_quantiles', 'latest_reconstruction_diagnostic', 'production_and_shadow_artifacts_unchanged')}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
