from datetime import date, datetime, timedelta

import polars as pl
import pytest

from app.market_time import CN_TZ
from app.services import stockformer_tree_validation as validation
from app.strategy import paper

EXECUTION = {"initial_cash": 10000., "max_positions": 3, "holding_sessions": 5,
             "commission_pct": .0003, "stamp_tax_pct": .0005, "slippage_bps": 5.,
             "minimum_commission": 5., "queue_limit_orders": False}
DAYS = [date(2025, 1, n) for n in (2, 3, 6, 7, 8, 9, 10, 13, 14)]


def raw_prices():
    return pl.DataFrame([{"symbol": "000001", "date": day, "open": 10., "high": 10.,
                          "low": 10., "close": 10., "volume": 1000., "amount": 10000.}
                         for day in DAYS])


def signal():
    return pl.DataFrame({"symbol": ["000001"], "date": [DAYS[0]], "net_margin_score": [.02],
                         "_forecast_pass": [True], "_prediction_pass": [True], "_net_margin_pass": [True],
                         "_entry_rules_pass": [True], "_market_gate_pass": [True], "pred_ret_5d": [.02]})


def test_actual_paper_round_trip_next_open_five_sessions_and_minimum_fee(tmp_path):
    result = validation.replay_paper(tmp_path/'replay', raw_prices(), signal(), DAYS[:7], EXECUTION)
    assert result['completed_trades'] == 1
    assert result['fills'][0]['date'] == str(DAYS[1])
    assert result['fills'][1]['date'] == str(DAYS[6])
    assert result['fills'][0]['qty'] == 300
    assert result['fills'][0]['fee'] == 5.
    assert result['fills'][1]['fee'] == 6.5
    assert result['final_nav'] == pytest.approx(9985.5)
    assert result['total_return_pct'] == pytest.approx(-.145)
    assert result['settled'] is True
    assert result['replay_kind'] == 'retrospective_not_forward'


def test_constant_cumulative_factor_does_not_multiply_shares_each_session(tmp_path):
    factors = raw_prices().select('symbol', pl.col('date').alias('trade_date')).with_columns(
        pl.lit(7.25).alias('adj_factor'))
    result = validation.replay_paper(tmp_path/'cumulative', raw_prices(), signal(), DAYS[:7], EXECUTION,
                                    cumulative_factors=factors)
    assert result['completed_trades'] == 1 and result['settled']
    assert [fill['qty'] for fill in result['fills']] == [300, 300]
    assert all(fill['kind'] == 'fill' for fill in result['fills'])
    assert result['final_nav'] == pytest.approx(9985.5)


def test_cumulative_change_cannot_be_assumed_to_be_share_bonus(tmp_path):
    factors = raw_prices().select('symbol', pl.col('date').alias('trade_date')).with_columns(
        pl.when(pl.col('trade_date') >= DAYS[3]).then(7.5).otherwise(7.25).alias('adj_factor'))
    with pytest.raises(ValueError, match='verified share/cash accounting'):
        validation.replay_paper(tmp_path/'event', raw_prices(), signal(), DAYS[:7], EXECUTION,
                                cumulative_factors=factors)
    fills = paper.load_fills(tmp_path/'event', 'tree_replay')
    assert len(fills) == 1 and fills[0]['qty'] == 300 and fills[0]['kind'] == 'fill'


def test_missing_cumulative_factor_on_held_day_fails_closed(tmp_path):
    factors = raw_prices().select('symbol', pl.col('date').alias('trade_date')).with_columns(
        pl.lit(7.25).alias('adj_factor')).filter(pl.col('trade_date') != DAYS[3])
    with pytest.raises(ValueError, match='Missing cumulative factor history'):
        validation.replay_paper(tmp_path/'missing-factor', raw_prices(), signal(), DAYS[:7], EXECUTION,
                                cumulative_factors=factors)


def test_limit_up_rejection_is_preserved_not_fabricated_fill(tmp_path):
    raw = raw_prices().with_columns([pl.when(pl.col('date') == DAYS[1]).then(11.).otherwise(pl.col(c)).alias(c)
                                    for c in ('open', 'high', 'low', 'close')])
    result = validation.replay_paper(tmp_path/'replay', raw, signal(), DAYS[:7], EXECUTION)
    assert result['completed_trades'] == 0
    assert result['fills'] == []
    assert any('涨停' in str(order.get('reason')) for order in result['orders'])
    assert result['passed_execution_gate'] is False


def test_suspension_delays_entry_and_unsettled_position_is_not_force_closed(tmp_path):
    raw = raw_prices().with_columns(pl.when(pl.col('date') == DAYS[1]).then(0.).otherwise(pl.col('volume')).alias('volume'))
    result = validation.replay_paper(tmp_path/'replay', raw, signal(), DAYS[:7], EXECUTION)
    assert result['fills'][0]['date'] == str(DAYS[2])
    assert result['completed_trades'] == 0
    assert result['settled'] is False
    assert result['passed_execution_gate'] is False


def test_replay_clock_is_restored_on_exception(tmp_path, monkeypatch):
    original_now, original_today, original_read = paper.cn_now, paper.cn_today, paper.read_daily_bar
    monkeypatch.setattr(paper, 'settle_day', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('failure')))
    with pytest.raises(RuntimeError, match='failure'):
        validation.replay_paper(tmp_path/'replay', raw_prices(), signal(), DAYS[:7], EXECUTION)
    assert paper.cn_now is original_now
    assert paper.cn_today is original_today
    assert paper.read_daily_bar is original_read


def test_windows_leave_label_settlement_before_holdout_and_require_training_history():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(520)]
    windows = validation.make_windows(days, days[-1])
    assert len(windows) == 3
    for window in windows:
        assert len(window['signal_dates']) == 63
        assert len(window['replay_dates']) == 69
        assert window['replay_dates'][-1] < days[-1]
        assert window['train_start'] < window['start']
    assert windows[0]['signal_dates'][-1] < windows[1]['start']
    with pytest.raises(ValueError, match='history'):
        validation.make_windows(days[:300], days[299])


def test_training_tuning_calibration_and_test_boundaries_are_purged():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(270)]
    frame = pl.DataFrame({'symbol': ['000001'] * 270, 'date': days,
                          'label_end_5d': [d + timedelta(days=6) for d in days]})
    train, core, tune, calibration, tuning_cut, calibration_cut = validation.training_parts(frame, days[0], days[252])
    assert core['label_end_5d'].max() < tuning_cut
    assert tune['label_end_5d'].max() < calibration_cut
    assert calibration['label_end_5d'].max() < days[252]
    assert train['label_end_5d'].max() < days[252]
    assert tune['date'].min() == tuning_cut
    assert calibration['date'].min() == calibration_cut


def test_empty_candidates_still_record_every_session(tmp_path):
    result = validation.replay_paper(tmp_path/'replay', raw_prices(), signal().clear(), DAYS[:7], EXECUTION)
    assert len(result['daily']) == 7
    assert result['candidate_count'] == 0
    assert result['total_return_pct'] == 0
    assert result['passed_execution_gate'] is False
    assert paper.cn_now().date() == datetime.now(CN_TZ).date()


def test_profitable_settled_account_with_missing_holding_mark_cannot_pass(tmp_path):
    symbols = ['000001', '000002', '000003']
    raw = pl.concat([raw_prices().with_columns(pl.lit(symbol).alias('symbol')) for symbol in symbols])
    raw = raw.with_columns([
        pl.when(pl.col('date') == DAYS[6]).then(10.5).otherwise(pl.col(c)).alias(c)
        for c in ('open', 'high', 'low', 'close')
    ]).with_columns(
        pl.when((pl.col('symbol') == '000001') & (pl.col('date') == DAYS[3]))
        .then(0.).otherwise(pl.col('volume')).alias('volume')
    )
    signals = pl.concat([signal().with_columns(pl.lit(symbol).alias('symbol')) for symbol in symbols])
    result = validation.replay_paper(tmp_path/'replay', raw, signals, DAYS[:7], EXECUTION)
    assert result['completed_trades'] == 3
    assert result['total_return_pct'] > 0
    assert result['settled'] is True
    assert result['missing_position_marks'] == 1
    assert result['passed_execution_gate'] is False
