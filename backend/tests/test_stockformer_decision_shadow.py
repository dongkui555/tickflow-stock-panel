from datetime import date

import polars as pl
import pytest
import test_stockformer_pair_shadow as legacy
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_pair_shadow as shadow
from app.strategy import paper

NOW = legacy.NOW
run = legacy.run


def configure_decision(root, monkeypatch):
    policy = {'effective_from': '2026-10-22', 'prediction_rule': {'min_net_edge': .0005,
        'ranking_column': 'pred_ret_raw_5d'}}
    manifest = {'pair_id': '717-research', 'feature_columns': ['ret_1d'], 'minute_policy': {},
        'provenance': {'frozen_as_of': '2026-10-20', 'context': {'execution': daily._EXECUTION,
        'fixed_arm': 'decision_raw', 'signal_policy': policy}}}
    monkeypatch.setattr(shadow, '_verify_pair', lambda *a: manifest)
    from app.services import stockformer_decision_shadow as decision
    monkeypatch.setattr(decision, 'training_universe', lambda *a: ['000001', '000002', '000003', '000004'])
    return shadow.configure(root, root/'frozen', start_date=date(2026, 10, 22), variant='decision')


def test_decision_scope_is_distinct_and_unknown_scope_rejected(tmp_path):
    assert shadow.scope(tmp_path, variant='decision') != shadow.scope(tmp_path)
    with pytest.raises(ValueError, match='variant'):
        shadow.scope(tmp_path, variant='../anything')


def test_independent_account_and_no_backdated_start(run, monkeypatch):
    root, old, _ = run
    old_bytes = (daily._root(shadow.view(root))/'config.json').read_bytes()
    cfg = configure_decision(root, monkeypatch)
    assert cfg['account_id'].startswith('sf_decision_') and cfg['account_id'] != old['account_id']
    assert cfg['protocol']['ranking_column'] == 'pred_ret_raw_5d'
    assert shadow.tick(root, now=NOW, variant='decision')['status'] == 'before_start'
    assert shadow.status(root, variant='decision')['cost_after_return_pct'] is None
    assert (daily._root(shadow.view(root))/'config.json').read_bytes() == old_bytes
    with pytest.raises(ValueError, match='already configured'):
        shadow.configure(root, root/'frozen', start_date=date(2026, 10, 22), variant='decision')


def test_raw_priority_and_idempotent_orders_do_not_touch_other_account(run, monkeypatch):
    root, old, frame = run
    cfg = configure_decision(root, monkeypatch)
    now = NOW.replace(day=22)
    monkeypatch.setattr(shadow, 'cn_now', lambda: now)
    monkeypatch.setattr(daily, 'cn_now', lambda: now)
    monkeypatch.setattr(paper, '_now_iso', lambda: now.isoformat())
    signals = pl.concat([frame.with_columns(pl.lit(symbol).alias('symbol'), pl.lit(now.date()).alias('date'),
        pl.lit(raw).alias('pred_ret_raw_5d'), pl.lit(margin).alias('net_margin_score'))
        for symbol, raw, margin in [('000001', .1, .5), ('000002', .4, .1), ('000003', .3, .2), ('000004', .2, .3)]])
    monkeypatch.setattr(shadow, '_compute', lambda *a: (signals, signals, signals, {}))
    first = shadow.tick(root, now=now, variant='decision')
    assert first['status'] == 'recorded' and first['eligible_for_forward']
    assert [r['symbol'] for r in first['selected']] == ['000002', '000003', '000004']
    shadow.tick(root, now=now, variant='decision')
    assert len(paper.load_orders(shadow.view(root, variant='decision'), cfg['account_id'])) == 3
    assert paper.load_orders(shadow.view(root), old['account_id']) == []
    assert shadow.status(root)['recorded_days'] == 0


def test_new_status_api_reads_without_computing_and_preserves_old_route(run, monkeypatch):
    from types import SimpleNamespace

    from app.custom.frozen_pair_shadow import router
    root, _, _ = run
    cfg = configure_decision(root, monkeypatch)
    monkeypatch.setattr(shadow, '_compute', lambda *a: pytest.fail('API must not infer'))
    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=root))
    app.include_router(router)
    client = TestClient(app)
    new = client.get('/api/stockformer/decision-pair-shadow')
    assert new.status_code == 200 and new.json()['account_id'] == cfg['account_id']
    assert client.get('/api/stockformer/frozen-pair-shadow').json()['pair_id'] == 'frozen-pair'


def test_new_start_cannot_precede_frozen_signal_policy(run, monkeypatch):
    root, _, _ = run
    configure_decision(root, monkeypatch)
    cfg = daily._read(daily._root(shadow.view(root, variant='decision'))/'config.json')
    assert cfg['protocol']['start_date'] >= '2026-10-22'
    other = root/'other'
    with pytest.raises(ValueError, match='effective'):
        shadow.configure(other, root/'frozen', start_date=date(2026, 10, 21), variant='decision')


def test_pool_exclusion_is_separate_from_missing_minute_coverage(tmp_path, monkeypatch):
    from app.services import prediction_workflow as workflow
    from app.services import stockformer as sf
    from app.services import stockformer_decision_shadow as decision
    from app.services import stockformer_frozen_pair as pair
    day = date(2026, 10, 22)
    frame = pl.DataFrame([{'symbol': s, 'date': day, 'amount': 1e8, 'close': 10.,
        'price_chase_risk': chase, 'moneyflow_chase_risk': .1,
        'limit_down_flag': 0., 'broken_limit_up_flag': 0., 'ret_1d': .01,
        'minute_data_available': 1., 'minute_session_coverage': 1., 'minute_bar_count': 240.}
        for s, chase in [('000001', .1), ('000002', .9), ('000003', .1)]])
    cfg = {'protocol': {'frozen_root': '/frozen', 'training_universe': frame['symbol'].to_list(),
        'minimum_active_coverage': .98, 'signal_policy': {'prediction_rule': {'min_net_edge': .0005}}}}
    monkeypatch.setattr(workflow, '_frame', lambda *a: frame)
    monkeypatch.setattr(workflow, '_closed_prices', lambda f, *a: f)
    monkeypatch.setattr(workflow, '_symbols', lambda f: set(f['symbol']))
    monkeypatch.setattr(sf, '_load_prediction_base', lambda *a, **k: (frame, day, {}))
    seen = []
    def predict(_root, data):
        seen.append(data.height)
        return data.select('symbol', 'date').with_columns(pl.lit(.01).alias('pred_ret_raw_5d'),
            pl.lit(.001).alias('pred_ret_5d')), {}
    monkeypatch.setattr(pair, 'predict_pair', predict)
    monkeypatch.setattr(decision.portfolio, '_build_gated_signal_frame', lambda *a, **k: a[1])
    manifest = {'feature_columns': ['ret_1d']}
    pred, _, _, coverage = decision.compute(tmp_path, cfg, day, manifest)
    assert seen == [2] and pred.height == 2
    assert coverage['complete_active'] == 3 and coverage['active_universe'] == 3
    assert coverage['pool_excluded'][0]['symbol'] == '000002'
    sparse = frame.with_columns(pl.when(pl.col('symbol') == '000003').then(239.).otherwise(240.).alias('minute_bar_count'))
    monkeypatch.setattr(sf, '_load_prediction_base', lambda *a, **k: (sparse, day, {}))
    with pytest.raises(shadow.WaitingDataError, match='覆盖不足'):
        decision.compute(tmp_path, cfg, day, manifest)
