"""Train-only calibration comparisons and refit distribution controls."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from app.market_time import cn_now
from app.services import stockformer as sf
from app.services import stockformer_calibration_research as research
from app.services import stockformer_training as training

TARGET = 'target_5d_rel_net'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(sf._json_safe(value), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def fit_pair(frame, columns, before, folder, name):
    assert frame['label_end_5d'].max() < before
    x, y, used = sf._prepare_xy(frame, TARGET, columns)
    if used['date'].n_unique() < 100 or len(y) < 1000:
        raise ValueError('Insufficient tree training history')
    xgb, lgb = sf._fit_xgboost(x, y, n_estimators=300), sf._fit_lightgbm(x, y, n_estimators=300)
    xgb.save_model(str(folder/f'{name}-xgb.json'))
    lgb.booster_.save_model(str(folder/f'{name}-lgb.txt'))
    return (xgb, lgb), {'rows': used.height, 'dates': used['date'].n_unique(),
                       'max_label_end': used['label_end_5d'].max(), 'fit_before': before}


def predict(pair, frame, columns):
    strict, _ = sf.prepare_training_features(frame, 'minute_enhanced', minimum_minute_bars=240,
                                              minimum_minute_coverage=1.)
    x, _, used = sf._prepare_xy(strict.with_columns(pl.lit(0.).alias('_predict_mask')), '_predict_mask', columns)
    if not used.height:
        raise ValueError('No complete minute predictions')
    raw = .5*sf._predict_xgboost(pair[0], x)+.5*sf._predict_lightgbm(pair[1], x)
    return research.with_percentile(used.select('symbol', 'date', pl.col(TARGET).alias('target')).with_columns(
        pl.Series('raw', raw))).sort('date', 'symbol')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise ValueError('Research protocol exists; preserve prior attempt')
    model_root = args.data_dir/'stockformer/models'/args.model_id
    protected = {str(p): digest(p) for p in [*model_root.glob('*'), args.data_dir/'stockformer/latest_model.txt',
                                           args.data_dir/'stockformer/daily_shadow/config.json'] if p.is_file()}
    write(out/'protected-before.json', protected)
    meta = sf._model_meta(args.data_dir, args.model_id)
    payload = sf._load_model_payload(sf._resolve_artifact_path(args.data_dir, meta['model_file']))
    columns = payload['linear_baseline_models']['5']['feature_columns']
    end = date.fromisoformat(meta['validation']['folds']['5d'][0]['start'])
    frame = pl.scan_parquet(args.features).filter(pl.col('date') < end).collect()
    calendar = frame['date'].unique().to_list()
    frame, minute_coverage = sf.prepare_training_features(frame, 'minute_enhanced')
    assert frame['label_cost_version'].unique().to_list() == [training.PAPER_COST_TARGET_VERSION]
    frame = frame.with_columns(pl.when(pl.col('label_end_5d') < end).then(pl.col(TARGET)).otherwise(None).alias(TARGET))
    windows, coverage_attempts = research.coverage_windows(frame, columns, end, calendar=calendar)
    protocol = {'version': 'train_only_refit_calibration_complete240_v4', 'frozen_at': cn_now(), 'model_reference': args.model_id,
                'minute_policy': sf.MINUTE_AVAILABILITY_POLICY, 'minute_coverage': minute_coverage,
                'development_end_exclusive': end, 'windows': windows, 'feature_columns': columns,
                'cost_label_version': training.PAPER_COST_TARGET_VERSION,
                'trees': {'n_estimators': 300, 'xgb_weight': .5, 'lgb_weight': .5, 'seed': 42,
                          'kind': 'fixed_capacity_no_hyperparameter_selection'},
                'methods': research.METHODS, 'isotonic_bins': 20, 'minimum_calibration_dates': 20,
                'calibration_sessions': 42, 'coverage_preflight': coverage_attempts,
                'geometry_selection': 'largest check length21..14 with100 training,20 calibration and10 check label dates; availability only',
                'weighting': 'equal_dates; legacy_tail_keeps_original_unweighted_median',
                'strategies': ['paired_frozen_tree', 'transfer_to_refitted_tree'],
                'selection_policy': 'compare_all_methods_no_automatic_winner_or_production_policy',
                'scope': 'training_internal_checks_only_no_original_validation_or_holdout_predictions',
                'input_sha256': digest(args.features), 'bounded_input': training._input_frame_fingerprint(frame),
                'source_sha256': {str(Path(m.__file__)): digest(Path(m.__file__)) for m in (sf, training, research)},
                'runner_sha256': digest(Path(__file__)),
                'activation_allowed': False}
    write(out/'protocol.json', protocol)
    print(json.dumps({'stage': 'protocol_frozen', 'end_exclusive': str(end), 'windows': sf._json_safe(windows)}), flush=True)
    reports = []
    for window in windows:
        folder = out/f"fold-{window['fold']}"
        folder.mkdir()
        cal_start, check_start = window['calibration_start'], window['check_start']
        base_train = training.purged_slice(frame, 5, frame['date'].min(), cal_start)
        refit_train = training.purged_slice(frame, 5, frame['date'].min(), check_start)
        assert base_train['label_end_5d'].max() < cal_start
        assert refit_train['label_end_5d'].max() < check_start
        print(json.dumps({'stage': 'base_fit', 'fold': window['fold']}), flush=True)
        base, base_meta = fit_pair(base_train, columns, cal_start, folder, 'base')
        calibration_input = frame.filter(pl.col('date').is_in(window['calibration_dates'])).with_columns(
            pl.when(pl.col('label_end_5d') < check_start).then(pl.col(TARGET)).otherwise(None).alias(TARGET))
        calibration = predict(base, calibration_input, columns)
        check = frame.filter(pl.col('date').is_in(window['check_dates']))
        check_base = predict(base, check, columns)
        mappings = {method: research.fit_method(calibration, method) for method in research.METHODS}
        write(folder/'calibrations.json', mappings)
        calibration.write_parquet(folder/'calibration-predictions.parquet')
        del base
        gc.collect()
        print(json.dumps({'stage': 'refit', 'fold': window['fold']}), flush=True)
        refit, refit_meta = fit_pair(refit_train, columns, check_start, folder, 'refit')
        check_refit = predict(refit, check, columns)
        comparison = research.compare_scores(check_base, check_refit)
        mapping_max_label = calibration_input.filter(pl.col(TARGET).is_finite())['label_end_5d'].max()
        assert mapping_max_label < check_start
        metrics = {}
        for name, sample in (('paired_frozen_tree', check_base), ('transfer_to_refitted_tree', check_refit)):
            metrics[name] = {}
            columns_out = []
            for method, mapping in mappings.items():
                calibrated = research.apply_method(sample, mapping)
                metrics[name][method] = research.evaluate(sample, calibrated)
                columns_out.append(pl.Series(f'cal_{method}', calibrated))
            sample.with_columns(columns_out).write_parquet(folder/f'check-{name}.parquet')
        item = {'window': window, 'base_training': base_meta, 'refit_training': refit_meta,
                'calibration_max_label_end': mapping_max_label, 'check_max_label_end':
                    check.filter(pl.col(TARGET).is_finite())['label_end_5d'].max(),
                'market_shift_fixed_base_ks': research.ks_distance(calibration['raw'].to_numpy(), check_base['raw'].to_numpy()),
                'same_check_refit_shift': comparison, 'metrics': metrics}
        write(folder/'summary.json', item)
        reports.append(item)
        print(json.dumps({'stage': 'fold_complete', 'fold': window['fold'], 'refit_shift': comparison}, default=str), flush=True)
        del refit, check_base, check_refit, calibration
        gc.collect()
    after = {p: digest(Path(p)) for p in protected}
    assert protected == after, 'Frozen live artifacts changed'
    write(out/'protected-after.json', after)
    aggregate = {}
    for strategy in protocol['strategies']:
        aggregate[strategy] = {}
        for method in research.METHODS:
            entries = [r['metrics'][strategy][method] for r in reports]
            baseline = [r['metrics'][strategy]['identity']['mse'] for r in reports]
            aggregate[strategy][method] = {'mean_mse': float(np.mean([e['mse'] for e in entries])),
                'mse_better_than_raw_folds': sum(e['mse'] < b for e, b in zip(entries, baseline, strict=True)),
                'positive_score_target_mean_by_fold': [e['positive']['mean_target'] for e in entries],
                'positive_score_rows_by_fold': [e['positive']['rows'] for e in entries],
                'mean_bias': float(np.mean([e['bias'] for e in entries]))}
    write(out/'report.json', {'completed_at': cn_now(), 'protocol': protocol, 'folds': reports, 'aggregate': aggregate,
                            'production_artifacts_unchanged': True, 'activation_allowed': False,
                            'automatic_selection': None, 'account_returns_computed': False})
    print(json.dumps({'stage': 'complete', 'aggregate': aggregate}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
