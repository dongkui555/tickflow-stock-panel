from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.custom.frozen_pair_shadow import router
from app.market_time import cn_now
from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_pair_shadow as observer
from app.strategy import paper


def client(root):
    app=FastAPI()
    app.state.repo=SimpleNamespace(store=SimpleNamespace(data_dir=root))
    app.include_router(router)
    return TestClient(app)


def test_all_status_routes_work_without_loading_research_inputs(tmp_path):
    api=client(tmp_path)
    for variant in ['frozen','decision','repaired']:
        response=api.get(f'/api/stockformer/{variant}-pair-shadow')
        assert response.status_code==200
        state=response.json()
        assert state['recorded_days']==state['fills']==state['completed_trades']==0
        assert state['cost_after_return_pct'] is None
        assert not state['activation_allowed']


def test_repaired_status_never_mixes_existing_accounts_or_calls_compute(tmp_path,monkeypatch):
    scope=observer.scope(tmp_path,variant='repaired')
    root=daily._root(observer.view(tmp_path,variant='repaired'))
    daily._write(root/'config.json',{'enabled':True,'account_id':'sf_repaired_test','protocol_id':'new',
        'protocol':{'start_date':'2026-10-26','pair_id':'2af','model_id':'frozen_pair:2af',
                    'execution':daily._EXECUTION}})
    daily._write(root/'state.json',{'status':'before_start','as_of':'2026-10-24'})
    daily._write(scope/'worker.json',{'checked_at':cn_now().isoformat()})
    monkeypatch.setattr(observer,'_compute',lambda *a:(_ for _ in ()).throw(AssertionError('status computed features')))
    api=client(tmp_path)
    state=api.get('/api/stockformer/repaired-pair-shadow').json()
    assert state['account_id']=='sf_repaired_test' and state['pair_id']=='2af'
    assert state['worker']['alive'] and state['state']['status']=='before_start'
    assert not api.get('/api/stockformer/frozen-pair-shadow').json()['enabled']
    assert not api.get('/api/stockformer/decision-pair-shadow').json()['enabled']


def test_corporate_status_uses_only_published_matching_ledger_head(tmp_path,monkeypatch):
    inputs=observer.view(tmp_path,variant='repaired')
    root=daily._root(inputs)
    account='sf_corporate'
    daily._write(root/'config.json',dict(enabled=True,account_id=account,protocol_id='corporate-v1',
        protocol=dict(start_date='2026-10-26',execution=daily._EXECUTION,
                      corporate_action_policy='verified_cash_transfer_ledger_v1')))
    base=paper.accounts_base(inputs)/account
    base.mkdir(parents=True)
    (base/'fills.jsonl').write_text('{"kind":"cash_payment_v1","date":"2026-10-27"}\n')
    (base/'nav').mkdir()
    (base/'nav/daily.jsonl').write_text('{"date":"2026-10-27","nav":10020}\n')
    daily._write(root/'execution_metrics.json',dict(account_id=account,protocol_id='corporate-v1',
        ledger_sha256=daily._digest(base/'fills.jsonl'),nav_sha256=daily._digest(base/'nav/daily.jsonl'),
        completed_trades=1,cost_after_return_pct=.2))
    monkeypatch.setattr(paper,'round_trips',lambda *a,**k:(_ for _ in ()).throw(AssertionError('legacy corporate P&L')))
    api=client(tmp_path)
    response=api.get('/api/stockformer/repaired-pair-shadow').json()
    assert response['completed_trades']==1 and response['cost_after_return_pct']==.2
    with (base/'fills.jsonl').open('a') as stream:
        stream.write('{"kind":"cash_payment_v1","date":"2026-10-28"}\n')
    response=api.get('/api/stockformer/repaired-pair-shadow').json()
    assert response['completed_trades']==0 and response['cost_after_return_pct'] is None
    assert any('账务摘要' in line for line in response['research_warnings'])
