"""Unify minute readiness, freeze tree/calibrator pairs and replay privately."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from app.market_time import cn_now
from app.services import minute_features
from app.services import stockformer as sf
from app.services import stockformer_calibration_research as research
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_training as training
from app.services import stockformer_tree_validation as validation


def write(path, value):
    path.write_text(json.dumps(sf._json_safe(value), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def stage(name, **kwargs):
    print(json.dumps({'stage': name, 'at': str(cn_now()), **kwargs}, default=str), flush=True)


def minute_axis_audit(reference):
    observations = []
    for day in (date(2024, 7, 8), date(2024, 8, 1)):
        path = reference/'data/kline_minute'/f'date={day}'
        frame = pl.scan_parquet([str(p) for p in path.rglob('*.parquet')]).filter(
            pl.col('symbol').is_in(['002045', '002045.SZ', 'SZ002045', '002046', '002046.SZ', 'SZ002046'])).collect()
        frame = frame.with_columns(pl.col('symbol').map_elements(minute_features._compact_stock_symbol,
            return_dtype=pl.String).alias('code'))
        for symbol, group in frame.partition_by('code', as_dict=True).items():
            present = {stamp.hour*60+stamp.minute for stamp in group['datetime']}
            end_missing = sorted(set(minute_features._END_LABELLED_MINUTES)-present)
            start_missing = sorted(set(minute_features._START_LABELLED_MINUTES)-present)
            missing = end_missing if len(end_missing) <= len(start_missing) else start_missing
            observations.append({'date': day, 'symbol': symbol[0], 'unique_timestamps': group['datetime'].n_unique(),
                'best_axis': 'end' if len(end_missing) <= len(start_missing) else 'start',
                'missing_clock_minutes': [f'{m//60:02}:{m%60:02}' for m in missing]})
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--probe-date', type=date.fromisoformat, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise ValueError('Experiment exists; preserve prior outputs')
    model_root = args.data_dir/'stockformer/models'/args.model_id
    protected = {str(p): pair.digest(p) for p in [*model_root.glob('*'), args.data_dir/'stockformer/latest_model.txt',
        args.data_dir/'stockformer/daily_shadow/config.json'] if p.is_file()}
    write(out/'protected-before.json', protected)
    meta = sf._model_meta(args.data_dir, args.model_id)
    payload = sf._load_model_payload(sf._resolve_artifact_path(args.data_dir, meta['model_file']))
    columns = payload['linear_baseline_models']['5']['feature_columns']
    raw_features = pl.read_parquet(args.features)
    if raw_features['label_cost_version'].unique().to_list() != [training.PAPER_COST_TARGET_VERSION]:
        raise ValueError('Unexpected paper-cost label version')
    # Preserve actual global sessions, including dates with no eligible stocks.
    calendar = sorted(pl.scan_parquet(args.panel or args.reference_dir/'training-panel.parquet').select('date').unique().collect()['date'].to_list())
    features, coverage = sf.prepare_training_features(raw_features, 'minute_enhanced')
    symbols = sorted(raw_features['symbol'].unique().to_list())
    development_end = date.fromisoformat(meta['validation']['folds']['5d'][0]['start'])
    development = features.filter(pl.col('date') < development_end).with_columns(
        pl.when(pl.col('label_end_5d') < development_end).then(pl.col(pair.TARGET)).otherwise(None).alias(pair.TARGET))
    windows, attempts = research.coverage_windows(development, columns, development_end, calendar=calendar)
    latest_end = features['date'].max()
    history = [d for d in calendar if d <= latest_end]
    if len(history) < 252:
        raise ValueError('Missing rolling training history')
    latest_start, latest_calibration = history[-252], history[-42]
    policy = portfolio.load_frozen_signal_policy(args.data_dir, meta)
    execution = json.loads((args.data_dir/'stockformer/daily_shadow/config.json').read_text())['protocol']['execution']
    protocol = {'version': 'complete240_paired_affine_fixed_v2', 'frozen_at': cn_now(),
        'minute_policy': sf.MINUTE_AVAILABILITY_POLICY, 'coverage': coverage,
        'calibration': 'nondecreasing_affine_equal_date_weights_no_method_search',
        'tree_parameters': {'rounds': 300, 'xgboost_weight': .5, 'lightgbm_weight': .5, 'seed': 42},
        'row_order': ['date', 'symbol'],
        'post_calibration_refit': False, 'development_end_exclusive': development_end,
        'windows': windows, 'coverage_preflight': attempts, 'feature_columns': columns,
        'latest_pair': {'train_start': latest_start, 'calibration_start': latest_calibration,
                        'label_cut_exclusive': latest_end+timedelta(days=1), 'input_through': latest_end},
        'execution': execution, 'signal_policy': policy, 'research_horizon_weights': {'5d': 1.},
        'cost_target_version': training.PAPER_COST_TARGET_VERSION, 'input_sha256': pair.digest(args.features),
        'panel_sha256': pair.digest(args.panel or args.reference_dir/'training-panel.parquet'),
        'source_sha256': {m.__name__: pair.digest(m.__file__) for m in (sf, training, research, pair, validation, portfolio)},
        'runner_sha256': pair.digest(__file__), 'activation_allowed': False,
        'validation_kind': 'training_internal_development_diagnostic_and_future_frozen_candidate_not_new_forward'}
    write(out/'protocol.json', protocol)
    protocol_sha = pair.digest(out/'protocol.json')
    stage('protocol_frozen', windows=windows, latest_pair=protocol['latest_pair'])
    features.write_parquet(out/'features-complete240.parquet')
    write(out/'minute-axis-audit.json', minute_axis_audit(args.reference_dir))
    coverage_by_year = []
    for year in sorted(raw_features['date'].dt.year().unique().to_list()):
        before = raw_features.filter(pl.col('date').dt.year() == year)
        after = features.filter(pl.col('date').dt.year() == year)
        coverage_by_year.append({'year': year, 'before_rows': before.height, 'ready_rows': after.height,
            'before_dates': before['date'].n_unique(), 'ready_dates': after['date'].n_unique()})
    write(out/'input-manifest.json', {'minute_policy': sf.MINUTE_AVAILABILITY_POLICY,
        'coverage': coverage, 'coverage_by_year': coverage_by_year,
        'logical_input': training._input_frame_fingerprint(features),
        'daily_and_minute_sources_unchanged': True, 'targets_not_recomputed_or_rebased': True})
    raw_prices, conflicts = training.load_paper_raw_prices(args.reference_dir/'data', symbols)
    if conflicts.height:
        raise ValueError('Ambiguous raw execution prices')
    factors = sf._normalize_symbol_column(pl.read_parquet(args.reference_dir/'data/adj_factor/all.parquet')).select(
        'symbol', 'trade_date', 'ex_factor').drop_nulls()
    if factors.group_by('symbol', 'trade_date').agg(pl.col('ex_factor').n_unique().alias('n')).filter(pl.col('n') > 1).height:
        raise ValueError('Ambiguous corporate-action factors')
    factors = factors.unique(['symbol', 'trade_date'])
    research_meta = {**meta, 'target_mode': 'industry_relative_net', 'signal_weights': {'5d': 1.}}
    summaries, signals = [], []
    for window in windows:
        folder = out/f"window-{window['fold']}"
        folder.mkdir()
        stage('pair_fit', window=window['fold'])
        model = pair.fit_pair(folder/'pair', development, columns, calibration_start=window['calibration_start'],
            label_cut=window['check_start'], context={'input_sha256': protocol['input_sha256'], 'protocol_sha256': protocol_sha})
        future = development.filter(pl.col('date').is_in(window['check_dates']))
        prediction, predict_coverage = pair.predict_pair(folder/'pair', future)
        prediction.write_parquet(folder/'predictions.parquet')
        evaluated = prediction.join(future.select('symbol', 'date', pl.col(pair.TARGET).alias('target_5d')),
                                   on=['symbol', 'date']).filter(pl.col('target_5d').is_finite())
        metrics = training.metrics(evaluated, evaluated['pred_ret_5d'].to_numpy(), evaluated['target_5d'].to_numpy(),
                                   target_mode='industry_relative_net')
        signal = portfolio._build_gated_signal_frame(args.reference_dir/'data', prediction, features, research_meta,
            horizon=5, signal_policy=policy, min_net_edge=policy['prediction_rule']['min_net_edge'])
        signal.write_parquet(folder/'signals.parquet')
        signals.append(signal)
        last_index = calendar.index(window['check_dates'][-1])
        replay_dates = [d for d in calendar if window['check_start'] <= d <= calendar[last_index+6]]
        assert replay_dates[-1] < development_end
        result = validation.replay_paper(folder/'paper', raw_prices, signal, replay_dates, execution, factors=factors)
        summary = {'window': window, 'pair_id': model['pair_id'], 'provenance': model['provenance'],
            'prediction_coverage': predict_coverage, 'prediction_metrics': metrics,
            'layers': portfolio._gated_layer_stats(signal, 'pred_ret_5d'),
            'execution': {k: result[k] for k in ('candidate_count', 'completed_trades', 'total_return_pct',
                'max_drawdown_pct', 'settled', 'missing_position_marks', 'passed_execution_gate')}}
        write(folder/'summary.json', summary)
        summaries.append(summary)
        stage('window_complete', window=window['fold'], execution=summary['execution'])
        gc.collect()
    aggregate_signal = pl.concat(signals, how='diagonal_relaxed').sort('date', 'symbol')
    final_index = calendar.index(windows[-1]['check_dates'][-1])
    aggregate_dates = [d for d in calendar if windows[0]['check_start'] <= d <= calendar[final_index+6]]
    aggregate = validation.replay_paper(out/'aggregate-paper', raw_prices, aggregate_signal, aggregate_dates, execution, factors=factors)
    stage('latest_pair_fit', start=latest_start, calibration_start=latest_calibration)
    latest = pair.fit_pair(out/'latest-pair', features.filter(pl.col('date') >= latest_start), columns,
        calibration_start=latest_calibration, label_cut=latest_end+timedelta(days=1),
        context={'input_sha256': protocol['input_sha256'], 'protocol_sha256': protocol_sha,
                 'training_universe': symbols, 'execution': execution, 'signal_policy': policy})
    stage('read_only_probe', as_of=args.probe_date)
    live_features, actual_date, base_meta = sf._load_prediction_base(args.data_dir, args.probe_date,
        required_feature_columns=columns, strict_minute=True, symbols=symbols)
    if actual_date != args.probe_date:
        raise ValueError('Probe date mismatch')
    prediction, live_coverage = pair.predict_pair(out/'latest-pair', live_features)
    prediction.write_parquet(out/'latest-pair/retrospective-probe.parquet')
    signal = portfolio._build_gated_signal_frame(args.data_dir, prediction, live_features, research_meta,
        horizon=5, signal_policy=policy, min_net_edge=policy['prediction_rule']['min_net_edge'])
    signal.write_parquet(out/'latest-pair/retrospective-signals.parquet')
    probe = {'as_of': actual_date, 'generated_at': cn_now(), 'record_kind': 'retrospective_probe_not_forward',
        'eligible_for_forward': False, 'pair_id': latest['pair_id'], 'prediction_rows': prediction.height,
        'raw_positive_rows': prediction.filter(pl.col('pred_ret_raw_5d') > 0).height,
        'calibrated_positive_rows': prediction.filter(pl.col('pred_ret_5d') > 0).height,
        'layers': portfolio._gated_layer_stats(signal, 'pred_ret_5d'), 'minute_coverage': live_coverage,
        'base_metadata': base_meta, 'activation_allowed': False}
    write(out/'latest-pair/probe.json', probe)
    frozen_paths = [p for folder in [out/f"window-{w['fold']}/pair" for w in windows]+[out/'latest-pair']
                    for p in [folder/'pair.json', folder/'xgboost.json', folder/'lightgbm.txt', folder/'calibration.json']]
    for folder in [out/f"window-{w['fold']}/pair" for w in windows]+[out/'latest-pair']:
        pair.verify_manifest(folder)
    after = {p: pair.digest(p) for p in protected}
    assert after == protected, 'Frozen production artifacts changed'
    write(out/'protected-after.json', after)
    report = {'completed_at': cn_now(), 'protocol': protocol, 'windows': summaries,
        'aggregate': {k: aggregate[k] for k in ('candidate_count', 'completed_trades', 'total_return_pct', 'max_drawdown_pct',
            'missing_position_marks', 'settled', 'passed_execution_gate')}, 'latest_pair': latest, 'probe': probe,
        'frozen_pair_files': {str(p.relative_to(out)): pair.digest(p) for p in frozen_paths},
        'production_artifacts_unchanged': True, 'activation_allowed': False,
        'reason': 'internal_short_development_windows_not_three_new_complete63_session_forward_windows'}
    write(out/'report.json', report)
    stage('complete', aggregate=report['aggregate'], latest_pair_id=latest['pair_id'], probe_layers=probe['layers'])


if __name__ == '__main__':
    main()
