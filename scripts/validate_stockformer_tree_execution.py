"""Run the fixed retrospective tree/calibration/paper-ledger validation study."""
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
from app.services import stockformer_tree_validation as validation


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(sf._json_safe(value), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--cost-study-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    args = parser.parse_args()
    out, root, reference = args.output_dir, args.data_dir, args.reference_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise ValueError('Study protocol exists; preserve results and use a new repair run directory')
    model_root = root/'stockformer/models'/args.model_id
    protected = {str(p): digest(p) for p in [*model_root.glob('*'), root/'stockformer/latest_model.txt',
                                            root/'stockformer/daily_shadow/config.json'] if p.is_file()}
    write(out/'protected-before.json', protected)
    meta = sf._model_meta(root, args.model_id)
    payload = sf._load_model_payload(sf._resolve_artifact_path(root, meta['model_file']))
    columns = payload['linear_baseline_models']['5']['feature_columns']
    execution = json.loads((root/'stockformer/daily_shadow/config.json').read_text())['protocol']['execution']
    feature_path = args.cost_study_dir/'research-features-paper-cost-v1.parquet'
    features = pl.read_parquet(feature_path)
    assert features['label_cost_version'].drop_nulls().unique().to_list() == [training.PAPER_COST_TARGET_VERSION]
    calendar = sorted(features['date'].unique().to_list())
    windows = validation.make_windows(calendar, date.fromisoformat(meta['oos_start']))
    # A fixed single-horizon candidate is evaluated as a research hypothesis.
    # This does not set the old research model's frozen empty weights.
    research_meta = {**meta, 'signal_weights': {'5d': 1.}, 'target_mode': 'industry_relative_net'}
    policy = portfolio.load_frozen_signal_policy(root, meta)
    protocol = {'version': 'tree_calibration_paper_complete240_rolling12m_3x63_v2', 'frozen_at': cn_now(),
                'horizon': 5, 'reference_model': args.model_id, 'windows': windows,
                'train_sessions': 252, 'prediction_minutes': 240, 'prediction_minute_coverage': 1.,
                'training_minutes': 240, 'minute_availability_policy': sf.MINUTE_AVAILABILITY_POLICY,
                'feature_family': 'fixed_full_from_prior_model',
                'feature_columns': columns, 'label_cost_version': training.PAPER_COST_TARGET_VERSION,
                'execution': execution, 'signal_policy': policy, 'research_horizon_weight': {'5d': 1.},
                'research_weight_kind': 'single_horizon_hypothesis_not_production_activation',
                'tree_parameters': 'existing_xgboost_lightgbm_helpers_max600_rounds',
                'pre_window_choices': 'early_stop_and_blend_on_purged_tuning_then_separate_calibration',
                'primary_arm': 'calibrated', 'control_arm': 'raw_diagnostic_never_select_winner_from_results',
                'holdout_policy': 'no_old_holdout_signals_or_selection; all_three_windows_before_old_holdout',
                'evidence_kind': 'retrospective_development_validation_not_new_forward',
                'input_sha256': {str(feature_path): digest(feature_path),
                                 str(reference/'training-panel.parquet'): digest(reference/'training-panel.parquet')},
                'source_sha256': {str(Path(module.__file__)): digest(Path(module.__file__)) for module in (sf, training, validation, portfolio)},
                'activation_allowed': False}
    write(out/'protocol.json', protocol)
    print(json.dumps({'stage': 'protocol_frozen', 'windows': sf._json_safe(windows), 'at': str(cn_now())}), flush=True)
    raw, conflicts = training.load_paper_raw_prices(reference/'data', features['symbol'].unique().to_list())
    write(out/'raw-price-quarantine.json', conflicts.to_dicts())
    factors = None
    factor_path = reference/'data/adj_factor/all.parquet'
    if factor_path.exists():
        factors = sf._normalize_symbol_column(pl.read_parquet(factor_path)).select('symbol', 'trade_date', 'ex_factor').drop_nulls()
        factors = factors.filter(pl.col('ex_factor').is_finite() & (pl.col('ex_factor') > 0))
        if factors.group_by('symbol', 'trade_date').agg(pl.col('ex_factor').n_unique().alias('n')).filter(pl.col('n') > 1).height:
            raise ValueError('Conflicting corporate-action factors')
        factors = factors.unique(['symbol', 'trade_date'])
    reports, all_signals = [], {'calibrated': [], 'raw_control': []}
    for window in windows:
        folder = out/f"window-{window['window']}"
        def emit(stage, number):
            print(json.dumps({'stage': stage, 'window': number, 'at': str(cn_now())}), flush=True)
        prediction, specification = validation.fit_window(features, columns, window, folder, emit=emit)
        evaluated = prediction.filter(pl.col('target_5d').is_finite())
        prediction_metrics = {}
        arms = {}
        for name, column in (('calibrated', 'pred_ret_5d'), ('raw_control', 'pred_ret_raw_5d')):
            prediction_metrics[name] = training.metrics(evaluated, evaluated[column].to_numpy(), evaluated['target_5d'].to_numpy(),
                                                       target_mode='industry_relative_net')
            factor = prediction if name == 'calibrated' else prediction.with_columns(pl.col('pred_ret_raw_5d').alias('pred_ret_5d'))
            signal = portfolio._build_gated_signal_frame(reference/'data', factor, features, research_meta,
                                                        horizon=5, signal_policy=policy,
                                                        min_net_edge=policy['prediction_rule']['min_net_edge'])
            signal.write_parquet(folder/f'signals-{name}.parquet')
            all_signals[name].append(signal)
            result = validation.replay_paper(folder/f'paper-{name}', raw, signal, window['replay_dates'], execution, factors=factors)
            arms[name] = {k: result[k] for k in ('candidate_count', 'completed_trades', 'total_return_pct', 'max_drawdown_pct',
                                               'final_nav', 'settled', 'missing_position_marks', 'passed_execution_gate')}
            arms[name]['layers'] = portfolio._gated_layer_stats(signal, 'pred_ret_5d')
        item = {'window': window, 'model': specification, 'prediction_metrics': prediction_metrics, 'execution': arms}
        reports.append(item)
        write(folder/'summary.json', item)
        print(json.dumps({'stage': 'window_complete', 'window': window['window'], 'execution': arms}, default=str), flush=True)
    aggregate = {}
    replay_dates = [d for d in calendar if windows[0]['start'] <= d <= windows[-1]['replay_dates'][-1]]
    for name, frames in all_signals.items():
        signal = pl.concat(frames, how='diagonal_relaxed').sort('date', 'symbol')
        result = validation.replay_paper(out/f'aggregate-paper-{name}', raw, signal, replay_dates, execution, factors=factors)
        aggregate[name] = {k: result[k] for k in ('candidate_count', 'completed_trades', 'total_return_pct', 'max_drawdown_pct',
                                               'final_nav', 'settled', 'missing_position_marks', 'passed_execution_gate')}
    after = {p: digest(Path(p)) for p in protected}
    assert after == protected, 'Frozen production/research/shadow artifact changed'
    write(out/'protected-after.json', after)
    primary_pass = all(r['execution']['calibrated']['passed_execution_gate'] for r in reports) and aggregate['calibrated']['passed_execution_gate']
    result = {'completed_at': cn_now(), 'protocol': protocol, 'windows': reports, 'aggregate': aggregate,
              'primary_performance_gate_passed': primary_pass, 'activation_allowed': False,
              'reason': 'historical_development_periods_not_independent_new_forward',
              'production_artifacts_unchanged': True,
              'limitations': ['Daily-open paper matching, not intraday queue replication.',
                             'Current-vintage historical features/classifications are not full point-in-time archives.',
                             'Existing paper corporate-action behavior; no complete cash-dividend/transfer-fee accounting.',
                             'Window accounts reset; continuous aggregate is simulated separately, never compound overlapping window returns.']}
    write(out/'report.json', result)
    print(json.dumps({'stage': 'complete', 'aggregate': aggregate, 'primary_performance_gate_passed': primary_pass,
                      'activation_allowed': False}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
