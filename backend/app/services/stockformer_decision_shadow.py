"""Adapters for the fixed raw-ranking research pair; no fitting or registration."""
from __future__ import annotations

import importlib
from pathlib import Path

import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_decision_research as decision
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_portfolio as portfolio

RESEARCH_WARNINGS = [
    '历史输入缺少涨跌停/炸板布尔列，冻结特征按原生成器默认零；尚未修正或重训。',
    '旧历史回放使用后来的证券名称和行业映射，已确认存在历史ST漏过滤；不能当作严格时点盈利证据。',
]


def training_universe(frozen: Path) -> list[str]:
    manifest = pair.verify_manifest(frozen/'latest-pair')
    record = daily._read(frozen/'universe.json')
    symbols = record.get('symbols', [])
    if (record.get('input_sha256') != manifest['provenance']['context']['input_sha256']
            or not symbols or symbols != sorted(set(symbols))):
        raise ValueError('Frozen decision universe mismatch')
    return symbols


def verify_bundle(frozen: Path) -> dict:
    manifest = pair.verify_manifest(frozen/'latest-pair')
    context = manifest['provenance']['context']
    parent = daily._read(frozen/'protocol.json')
    if (context.get('fixed_arm') != 'decision_raw' or context.get('pool_policy') != decision.POOL_POLICY
            or pair.digest(frozen/'protocol.json') != context.get('research_protocol_sha256')):
        raise ValueError('Frozen decision research protocol mismatch')
    modules = ['app.services.stockformer', 'app.services.stockformer_calibration_research',
        'app.services.stockformer_training', 'app.services.stockformer_frozen_pair',
        'app.services.stockformer_decision_research', 'app.services.stockformer_portfolio',
        'app.services.stockformer_tree_validation', 'app.strategy.paper']
    actual = {name: pair.digest(importlib.import_module(name).__file__) for name in modules}
    if actual != context['research_source_sha256'] or actual != parent['source_sha256']:
        raise ValueError('Frozen decision inference/gate source mismatch')
    if context['signal_policy'] != parent['raw_policy'] or context['execution'] != parent['cost_scenarios']['base_5bp']:
        raise ValueError('Frozen decision rules differ from research protocol')
    training_universe(frozen)
    return manifest


def compute(inputs: Path, cfg: dict, day, manifest: dict):
    from app.services.prediction_workflow import _closed_prices, _frame, _symbols
    from app.services.stockformer_pair_shadow import WaitingDataError

    protocol = cfg['protocol']
    universe = protocol['training_universe']
    active = _symbols(_closed_prices(_frame(inputs, 'kline_daily', day), day)) & set(universe)
    features, actual, _ = sf._load_prediction_base(inputs, day,
        required_feature_columns=manifest['feature_columns'], strict_minute=True, symbols=universe)
    if actual != day or features.is_empty():
        raise WaitingDataError('当天完整分钟特征尚未就绪（240根、覆盖率100%）')
    strict, coverage = sf.prepare_training_features(features, 'minute_enhanced')
    _, _, complete = sf._prepare_xy(strict.with_columns(pl.lit(0.).alias('_predict_mask')),
                                    '_predict_mask', manifest['feature_columns'])
    complete_active = complete.filter(pl.col('symbol').is_in(sorted(active)))
    if not active or complete_active.height < len(active)*protocol['minimum_active_coverage']:
        raise WaitingDataError(f'完整分钟预测覆盖不足：{complete_active.height}/{len(active)}，等待补齐')
    selected_pool, audit = decision.decision_pool(complete_active)
    predictions, _ = pair.predict_pair(Path(protocol['frozen_root'])/'latest-pair', selected_pool)
    if not predictions.is_empty() and (predictions['date'].unique().to_list() != [day]
                or predictions['symbol'].n_unique() != predictions.height
                or not predictions.select(pl.all_horizontal(
                    pl.col('pred_ret_5d').is_finite(), pl.col('pred_ret_raw_5d').is_finite()).all()).item()):
        raise ValueError('Decision prediction date/identity/value mismatch')
    excluded = complete_active.join(selected_pool.select('symbol', 'date'), on=['symbol', 'date'], how='anti')
    details = excluded.select('symbol', *decision.REQUIRED).to_dicts()
    signal = portfolio._build_gated_signal_frame(inputs, predictions, selected_pool,
        {'target_mode': 'industry_relative_net', 'signal_weights': {'5d': 1.}}, horizon=5,
        signal_policy=protocol['signal_policy'],
        min_net_edge=protocol['signal_policy']['prediction_rule']['min_net_edge'], score_source='raw')
    coverage = {**coverage, 'active_universe': len(active), 'complete_active': complete_active.height,
        'missing_features': sorted(active-set(complete_active['symbol'])),
        'decision_pool_rows': selected_pool.height, 'pool_audit': audit,
        'pool_excluded': [{**row, 'reason': 'failed_frozen_known_day_pool'} for row in details]}
    return predictions, signal, features, coverage
