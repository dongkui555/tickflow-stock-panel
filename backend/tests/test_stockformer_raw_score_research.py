from datetime import date

import numpy as np
import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_tree_validation as validation


def gate_inputs(monkeypatch):
    monkeypatch.setattr(sf, '_attach_instrument_names', lambda _root, f: f)
    monkeypatch.setattr(sf, '_filter_special_treatment', lambda f: f)
    monkeypatch.setattr(sf, '_attach_market_gate_for_oos', lambda _root, f: f.with_columns(
        pl.lit('strict').alias('market_gate'), pl.lit(80.).alias('market_regime_score'),
        pl.lit(0.).alias('market_buy_threshold_add')))
    features = pl.DataFrame([{'symbol': symbol, 'date': date(2024, 1, 2), 'amount': 1e8,
        **{column: .5 for column in sf.FEATURE_COLUMNS}} for symbol in ('000001', '000002', '000003')])
    prediction = features.select('symbol', 'date').with_columns(
        pl.Series('pred_ret_raw_5d', [.03, .01, .02]), pl.lit(-.01).alias('pred_ret_5d'))
    return features, prediction


def build(tmp_path, features, prediction, **kwargs):
    return portfolio._build_gated_signal_frame(tmp_path, prediction, features,
        {'target_mode': 'industry_relative_net', 'signal_weights': {'5d': 1.}}, horizon=5, **kwargs)


def test_raw_score_preserves_ranking_when_calibration_has_zero_slope(tmp_path, monkeypatch):
    features, prediction = gate_inputs(monkeypatch)
    legacy = build(tmp_path, features, prediction)
    raw = build(tmp_path, features, prediction, score_source='raw')
    assert np.ptp(legacy['stockformer_score'].to_numpy()) == 0
    assert raw.sort('stockformer_score', descending=True)['symbol'].to_list() == ['000001', '000003', '000002']
    assert raw['pred_ret_5d'].equals(prediction['pred_ret_5d'])
    assert not raw['_forecast_pass'].any()
    assert not raw['_net_margin_pass'].any()


def test_raw_entry_and_strict_gate_do_not_change_with_calibration(tmp_path, monkeypatch):
    features, prediction = gate_inputs(monkeypatch)
    first = build(tmp_path, features, prediction, score_source='raw')
    changed = prediction.with_columns((pl.col('pred_ret_raw_5d')*.2-.001).alias('pred_ret_5d'))
    second = build(tmp_path, features, changed, score_source='raw')
    columns = ['symbol', 'date', 'stockformer_score', 'entry_score', '_entry_rules_pass', '_market_gate_pass']
    assert first.select(columns).equals(second.select(columns))
    assert not first['net_margin_score'].equals(second['net_margin_score'])


def test_raw_source_fails_closed_and_default_is_legacy(tmp_path, monkeypatch):
    features, prediction = gate_inputs(monkeypatch)
    assert build(tmp_path, features, prediction).equals(build(tmp_path, features, prediction, score_source='calibrated'))
    with pytest.raises(ValueError, match='raw'):
        build(tmp_path, features, prediction.drop('pred_ret_raw_5d'), score_source='raw')
    with pytest.raises(ValueError, match='score_source'):
        build(tmp_path, features, prediction, score_source='unknown')


def test_raw_ranking_controls_actual_order_priority_without_bypassing_cost_gate(tmp_path):
    from test_stockformer_tree_validation import DAYS, EXECUTION, raw_prices, signal

    raw = pl.concat([raw_prices().with_columns(pl.lit(symbol).alias('symbol')) for symbol in ('000001', '000002')])
    signals = pl.concat([signal().with_columns(pl.lit(symbol).alias('symbol'),
        pl.lit(raw_score).alias('pred_ret_raw_5d'), pl.lit(margin).alias('net_margin_score'))
        for symbol, raw_score, margin in [('000001', .01, .03), ('000002', .02, .01)]])
    execution = {**EXECUTION, 'max_positions': 1}
    result = validation.replay_paper(tmp_path/'raw', raw, signals, DAYS[:7], execution,
                                     ranking_column='pred_ret_raw_5d')
    assert result['fills'][0]['symbol'] == '000002'
    assert result['fills'][0]['date'] == str(DAYS[1])
    assert result['fills'][1]['date'] == str(DAYS[6])
    assert result['ranking_column'] == 'pred_ret_raw_5d'
    assert result['settled'] is True
    blocked = signals.with_columns(pl.lit(False).alias('_forecast_pass'), pl.lit(False).alias('_net_margin_pass'))
    none = validation.replay_paper(tmp_path/'blocked', raw, blocked, DAYS[:7], execution,
                                   ranking_column='pred_ret_raw_5d')
    assert none['fills'] == []


def test_missing_replay_ranking_column_is_rejected_before_any_orders(tmp_path):
    from test_stockformer_tree_validation import DAYS, EXECUTION, raw_prices, signal

    with pytest.raises(ValueError, match='ranking_column'):
        validation.replay_paper(tmp_path/'bad', raw_prices(), signal(), DAYS[:7], EXECUTION,
                                ranking_column='pred_ret_raw_5d')
    assert not (tmp_path/'bad').exists()
