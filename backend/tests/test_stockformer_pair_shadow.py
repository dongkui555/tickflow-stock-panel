from datetime import datetime, timedelta

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.market_time import CN_TZ
from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_pair_shadow as shadow
from app.strategy import paper

NOW = datetime(2026, 10, 21, 19, 10, tzinfo=CN_TZ)
REAL_COMPUTE = shadow._compute


@pytest.fixture
def run(tmp_path, monkeypatch):
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    manifest = {'pair_id': 'frozen-pair', 'feature_columns': ['minute_available'],
                'minute_policy': {'version': 'complete_session_240_v1'},
                'provenance': {'frozen_as_of': '2026-10-20', 'context': {
                    'execution': daily._EXECUTION, 'training_universe': ['000001'],
                    'signal_policy': {'prediction_rule': {'min_net_edge': .0005}}}}}
    monkeypatch.setattr(shadow, '_verify_pair', lambda *a: manifest)
    monkeypatch.setattr(shadow, '_assets', lambda *a: {'frozen': 'unchanged'})
    monkeypatch.setattr(shadow, 'cn_now', lambda: NOW)
    monkeypatch.setattr(daily, 'cn_now', lambda: NOW)
    monkeypatch.setattr(paper, '_now_iso', lambda: NOW.isoformat())
    monkeypatch.setattr(shadow, 'readiness', lambda *a: {'ready': True, 'checks': {'日K': True}})
    monkeypatch.setattr(shadow, '_input_version', lambda *a: 'stable')
    monkeypatch.setattr(shadow, '_settle_sessions', lambda *a: {'status': 'settled'})
    monkeypatch.setattr(paper, 'read_daily_bar', lambda *a: {'open': 10., 'close': 10.})
    frame = pl.DataFrame({'symbol': ['000001'], 'date': [NOW.date()], 'pred_ret_raw_5d': [.02],
        'pred_ret_5d': [.01], 'net_margin_score': [.009], '_prediction_pass': [True],
        '_net_margin_pass': [True], '_forecast_pass': [True], '_entry_rules_pass': [True],
        '_market_gate_pass': [True]})
    monkeypatch.setattr(shadow, '_compute', lambda *a: (frame, frame, frame, {'eligible': 1}))
    old = tmp_path / 'stockformer/daily_shadow/config.json'
    old.parent.mkdir(parents=True)
    old.write_text('{"old":true}')
    cfg = shadow.configure(tmp_path, frozen, start_date=NOW.date())
    return tmp_path, cfg, frame


def test_separate_run_preserves_old_config_and_refuses_reconfiguration(run):
    root, cfg, _ = run
    assert (root / 'stockformer/daily_shadow/config.json').read_text() == '{"old":true}'
    assert cfg['protocol']['pair_id'] == 'frozen-pair'
    assert cfg['account_id'].startswith('sf_pair_')
    with pytest.raises(ValueError, match='already configured'):
        shadow.configure(root, root / 'frozen', start_date=NOW.date())
    for table in shadow.INPUT_TABLES:
        assert (shadow.view(root) / table).resolve() == (root / table).resolve()


def test_new_pair_records_and_orders_once_in_its_own_account(run, monkeypatch):
    root, cfg, _ = run
    first = shadow.tick(root, now=NOW)
    assert first['status'] == 'recorded'
    assert first['pair_id'] == 'frozen-pair'
    assert first['eligible_for_forward']
    assert first['observation']['selected_count'] == 1
    assert first['raw_positive_rows'] == first['calibrated_positive_rows'] == 1
    monkeypatch.setattr(shadow, '_compute', lambda *a: pytest.fail('must not recompute'))
    shadow.tick(root, now=NOW + timedelta(minutes=5))
    assert len(paper.load_orders(shadow.view(root), cfg['account_id'])) == 1
    assert not (root / 'paper/accounts' / cfg['account_id']).exists()
    result = shadow.status(root)
    assert result['recorded_days'] == 1 and result['completed_trades'] == 0
    assert result['cost_after_return_pct'] is None
    assert result['activation_allowed'] is False


def test_no_candidates_are_evidence_not_positive_returns(run, monkeypatch):
    root, _, frame = run
    signal = frame.with_columns(pl.lit(False).alias('_entry_rules_pass'))
    monkeypatch.setattr(shadow, '_compute', lambda *a: (frame, signal, frame, {}))
    record = shadow.tick(root, now=NOW)
    assert record['observation']['no_trade_reason'] == 'entry_rules_empty'
    assert record['observation']['selected_count'] == 0
    assert record['observation']['entry_rules_pass'] == 0
    assert shadow.status(root)['cost_after_return_pct'] is None


@pytest.mark.parametrize('hour', [9, 15, 18])
def test_never_predicts_before_evening(run, monkeypatch, hour):
    root, _, _ = run
    monkeypatch.setattr(shadow, '_compute', lambda *a: pytest.fail('too early'))
    assert shadow.tick(root, now=NOW.replace(hour=hour))['status'] == 'waiting_close'


def test_missing_minutes_remain_visible_and_retry(run, monkeypatch):
    root, _, _ = run
    monkeypatch.setattr(shadow, '_compute', lambda *a: (_ for _ in ()).throw(shadow.WaitingDataError('完整分钟不足')))
    result = shadow.tick(root, now=NOW)
    assert result['status'] == 'waiting_data' and '分钟' in result['reason']
    assert not daily.due(shadow.view(root), NOW + timedelta(minutes=1))
    assert daily.due(shadow.view(root), NOW + timedelta(minutes=16))
    assert shadow.status(root)['recorded_days'] == 0


def test_changed_pair_or_calibrator_blocks_before_orders(run, monkeypatch):
    root, cfg, _ = run
    monkeypatch.setattr(shadow, '_assets', lambda *a: {'frozen': 'modified'})
    monkeypatch.setattr(shadow, '_compute', lambda *a: pytest.fail('modified pair'))
    assert shadow.tick(root, now=NOW)['status'] == 'blocked_protocol_changed'
    assert not paper.load_orders(shadow.view(root), cfg['account_id'])
    assert paper.get_account(shadow.view(root), cfg['account_id'])['cash'] == 10000.


def test_input_drift_is_discarded(run, monkeypatch):
    root, _, _ = run
    versions = iter(['before', 'after'])
    monkeypatch.setattr(shadow, '_input_version', lambda *a: next(versions))
    assert shadow.tick(root, now=NOW)['status'] == 'inputs_changed'
    assert shadow.status(root)['recorded_days'] == 0


def test_midnight_completion_does_not_backdate_orders(run, monkeypatch):
    root, cfg, _ = run
    monkeypatch.setattr(shadow, 'cn_now', lambda: NOW + timedelta(hours=6))
    result = shadow.tick(root, now=NOW)
    assert result['status'] == 'late_record' and not result['eligible_for_forward']
    assert result['selected'] == []
    assert not paper.load_orders(shadow.view(root), cfg['account_id'])


def test_snapshot_publication_recovers_orders_without_duplicate_inference(run, monkeypatch):
    root, cfg, _ = run
    original = daily._orders
    monkeypatch.setattr(daily, '_orders', lambda *a: (_ for _ in ()).throw(OSError('disk unavailable')))
    assert shadow.tick(root, now=NOW)['status'] == 'error'
    monkeypatch.setattr(daily, '_orders', original)
    monkeypatch.setattr(shadow, '_compute', lambda *a: pytest.fail('snapshot is immutable'))
    assert shadow.tick(root, now=NOW)['status'] == 'recorded'
    assert len(paper.load_orders(shadow.view(root), cfg['account_id'])) == 1


def test_tampered_published_snapshot_cannot_recover_orders(run, monkeypatch):
    root, cfg, _ = run
    original = daily._orders
    monkeypatch.setattr(daily, '_orders', lambda *a: (_ for _ in ()).throw(OSError('interrupted')))
    assert shadow.tick(root, now=NOW)['status'] == 'error'
    monkeypatch.setattr(daily, '_orders', original)
    path = daily._root(shadow.view(root)) / f'days/{NOW.date()}/predictions.parquet'
    path.write_bytes(b'corrupt')
    assert shadow.tick(root, now=NOW)['status'] == 'blocked_protocol_changed'
    assert not paper.load_orders(shadow.view(root), cfg['account_id'])


def test_input_version_includes_industry_classification(run):
    root, _, _ = run
    before = shadow._input_version_real(shadow.view(root), NOW.date())
    table = root / 'ext_data/ext_hy_ths'
    table.mkdir(parents=True)
    pl.DataFrame({'symbol': ['000001'], 'industry': ['银行']}).write_parquet(table / 'all.parquet')
    assert before != shadow._input_version_real(shadow.view(root), NOW.date())


def test_missing_current_minutes_do_not_scan_historical_features(run, monkeypatch):
    root, cfg, _ = run
    monkeypatch.setattr(shadow.sf, '_load_prediction_base', lambda *a, **k: pytest.fail('unneeded history scan'))
    with pytest.raises(shadow.WaitingDataError, match='分钟K尚未导入'):
        REAL_COMPUTE(shadow.view(root), cfg, NOW.date())


def test_api_read_is_lightweight_and_distinct(run, monkeypatch):
    from types import SimpleNamespace

    from app.custom.frozen_pair_shadow import router

    root, _, _ = run
    monkeypatch.setattr(shadow, '_compute', lambda *a: pytest.fail('API must not compute'))
    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=root))
    app.include_router(router)
    result = TestClient(app).get('/api/stockformer/frozen-pair-shadow')
    assert result.status_code == 200
    assert result.json()['pair_id'] == 'frozen-pair'
    assert result.json()['recorded_days'] == 0


def test_completed_trade_uses_real_minimum_fees_slippage_and_t_plus_one(run):
    root, cfg, _ = run
    scoped = shadow.view(root)
    daily._account(scoped, cfg)
    assert paper.buy_fee(100, 10., .0003) == 5.
    assert paper.sell_fee(100, 10., .0003, .0005) == 5.5
    assert paper.apply_slippage(10., 'buy', 5.) == pytest.approx(10.005)
    record = shadow.tick(root, now=NOW)
    order = paper.load_orders(scoped, cfg['account_id'])[0]
    assert order['order_type'] == 'next_open'
    assert order['qty'] % 100 == 0
    assert paper._placed_after_price_time(order, str(NOW.date()))
    assert record['entry_rule'] == 'next_actual_session_open_after_generated_at'


def test_valuation_gap_blocks_settlement(run, monkeypatch):
    root, cfg, _ = run
    from app.services.stockformer_pair_shadow import _settle_sessions_real
    scoped = shadow.view(root)
    (root / f'kline_daily/date={NOW.date()}').mkdir(parents=True)
    pl.DataFrame({'symbol': ['000001'], 'date': [NOW.date()], 'open': [10.], 'close': [10.],
                  'high': [10.], 'low': [10.], 'volume': [100.], 'amount': [100000.]}).write_parquet(
                      root / f'kline_daily/date={NOW.date()}/part.parquet')
    monkeypatch.setattr(paper, 'load_positions', lambda *a: {'000001': {'qty': 100}})
    monkeypatch.setattr(paper, 'read_daily_bar', lambda *a: None)
    with pytest.raises(shadow.WaitingDataError, match='估值'):
        _settle_sessions_real(scoped, cfg, NOW.date())
    assert not (daily._root(scoped) / f'settlements/{NOW.date()}.json').exists()
