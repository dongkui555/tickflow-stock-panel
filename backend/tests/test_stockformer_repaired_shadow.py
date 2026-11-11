from datetime import date, datetime

import polars as pl
import pytest

from app.market_time import CN_TZ
from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_pair_shadow as observer
from app.services import stockformer_repaired_inputs as inputs
from app.services import stockformer_repaired_shadow as repaired
from app.services.stockformer_research_price_basis import feature_price_view
from app.services.stockformer_research_turnover import turnover_quality_context


def test_dated_source_cannot_be_stale_or_duplicate():
    day = date(2026, 10, 23)
    item = {'date':str(day),'endpoint':'daily_basic','records':[
        {'ts_code':'000001.SZ','trade_date':'20261023','turnover_rate':2.5}]}
    assert inputs.dated_frame(item,'daily_basic',day,['turnover_rate'])['turnover_rate'].item() == 2.5
    with pytest.raises(ValueError,match='match requested'):
        inputs.dated_frame(item,'daily_basic',date(2026,10,22),['turnover_rate'])
    item['records'] *= 2
    with pytest.raises(ValueError,match='Duplicate'):
        inputs.dated_frame(item,'daily_basic',day,['turnover_rate'])


def test_exact_closed_alias_duplicates_merge_but_conflicting_bars_block(tmp_path,monkeypatch):
    from app.indicators import pipeline
    from app.services import prediction_workflow as workflow

    day=date(2026,10,23)
    frame=pl.DataFrame({'symbol':['000001','000001.SZ'],'date':[day,day],
        'open':[10.,10.],'high':[10.,10.],'low':[10.,10.],'close':[10.,10.],
        'volume':[1000.,1000.],'amount':[10000.,10000.]})
    monkeypatch.setattr(pipeline,'_normalize_raw_volume_unit',lambda f:f)
    monkeypatch.setattr(workflow,'_closed_prices',lambda f,d:f)
    monkeypatch.setattr(workflow,'_frame',lambda *a:frame)
    assert inputs.raw_session(tmp_path,day,['000001']).height == 1
    frame=frame.with_columns(pl.Series('close',[10.,10.1]))
    with pytest.raises(ValueError,match='Duplicate'):
        inputs.raw_session(tmp_path,day,['000001'])


def test_absolute_factor_units_and_auction_are_applied_once():
    panel = pl.DataFrame({'symbol':['000001'],'date':[date(2026,10,23)],
        'open':[9.],'high':[9.],'low':[9.],'close':[9.],
        '_auction_open_price':[9.], '_raw_auction_open_price':[10.]})
    raw = panel.select('symbol','date').with_columns(*[pl.lit(10.).alias(c) for c in ['open','high','low','close']])
    factors = panel.select('symbol','date').with_columns(pl.lit(7.25).alias('adj_factor'))
    actual = feature_price_view(panel,raw,factors)
    assert actual['close'].item() == actual['_auction_open_price'].item() == 72.5
    assert inputs.clean_panel(panel)['_raw_auction_open_price'].item() == 10.
    with pytest.raises(AssertionError):
        feature_price_view(panel,raw,factors.with_columns(pl.lit(None).alias('adj_factor')))


def test_turnover_dispute_cannot_cross_stock_or_backfill():
    panel = pl.DataFrame({'symbol':['000001']*7+['000002']*7,
        'date':[date(2026,10,n) for n in range(15,22)]*2})
    bad = panel.filter((pl.col('symbol')=='000001') & (pl.col('date')==date(2026,10,16)))
    actual = turnover_quality_context(panel,bad)
    assert actual.filter(pl.col('symbol')=='000001')['turnover_context_ready'].to_list() == [True,False,False,False,False,False,True]
    assert actual.filter(pl.col('symbol')=='000002')['turnover_context_ready'].all()


def test_calendar_skips_weekend_and_uses_provider_trade_date(tmp_path,monkeypatch):
    monkeypatch.setattr(repaired,'_request_calendar',lambda *a:{'records':[]})
    path = tmp_path/'research_inputs/captures/2026-10-23-calendar.json'
    daily._write(path,{'date':'2026-10-23','records':[
        {'trade_date':'20261023','is_open':1},{'trade_date':'20261024','is_open':0},
        {'trade_date':'20261025','is_open':0},{'trade_date':'20261026','is_open':1}]})
    assert repaired.next_open(tmp_path,date(2026,10,23)) == '2026-10-26T09:30:00+08:00'
    daily._write(path,{'date':'2026-10-23','records':[]})
    with pytest.raises(observer.WaitingDataError,match='下一实际交易'):
        repaired.next_open(tmp_path,date(2026,10,23))
    daily._write(path,{'records':[{'is_open':1,'trade_date':None}]})
    with pytest.raises(observer.WaitingDataError,match='日历'):
        repaired.next_open(tmp_path,date(2026,10,23))


def test_empty_calendar_is_retried_after_cooldown_and_original_response_survives(tmp_path,monkeypatch):
    now=datetime(2026,10,26,19,10,tzinfo=CN_TZ)
    monkeypatch.setattr(repaired,'cn_now',lambda:now)
    calls=[]
    def fetch(*a):
        calls.append(1)
        return {'records':[] if len(calls)==1 else [{'trade_date':'20261027','is_open':1}]}
    monkeypatch.setattr(repaired,'_request_calendar',fetch)
    for _ in range(2):
        with pytest.raises(observer.WaitingDataError):
            repaired.next_open(tmp_path,now.date())
    assert len(calls)==1
    assert not (tmp_path/'research_inputs/captures/2026-10-26-calendar.json').exists()
    assert list((tmp_path/'research_inputs/captures/rejected').glob('*.json'))
    monkeypatch.setattr(repaired,'cn_now',lambda:now.replace(minute=26))
    assert repaired.next_open(tmp_path,now.date())=='2026-10-27T09:30:00+08:00'
    assert len(calls)==2


def test_partial_reference_cannot_become_a_successful_immutable_cache():
    day=date(2026,10,23)
    item={'date':str(day),'endpoint':'daily_basic','records':[
        {'ts_code':'000001.SZ','trade_date':'20261023','turnover_rate':0.,'float_share':1.}]}
    repaired.validate_capture(item,'daily_basic',day,{'000001'})
    with pytest.raises(ValueError,match='Partial'):
        repaired.validate_capture(item,'daily_basic',day,{'000001','000002'})
    item['records'][0]['turnover_rate']=None
    with pytest.raises(ValueError,match='values'):
        repaired.validate_capture(item,'daily_basic',day,{'000001'})


def test_repaired_scope_never_aliases_old_accounts(tmp_path):
    scopes = {observer.scope(tmp_path,variant=v) for v in ['frozen','decision','repaired']}
    assert len(scopes) == 3
    assert observer.scope(tmp_path,variant='repaired').name == 'repaired_pair_shadow'


def test_new_status_endpoint_reads_only_its_own_account(tmp_path,monkeypatch):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.custom.frozen_pair_shadow import router

    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.include_router(router)
    result = TestClient(app).get('/api/stockformer/repaired-pair-shadow')
    assert result.status_code == 200
    assert result.json()['recorded_days'] == result.json()['completed_trades'] == 0
    assert result.json()['cost_after_return_pct'] is None
    assert not result.json()['activation_allowed']


@pytest.fixture
def configured(tmp_path,monkeypatch):
    now=datetime(2026,10,26,19,10,tzinfo=CN_TZ)
    frozen=tmp_path/'frozen'
    frozen.mkdir()
    daily._write(frozen/'universe.json',{'symbols':['000001']})
    manifest={'pair_id':'repaired-research','feature_columns':['minute_bar_count'],
        'minute_policy':{'version':'complete_session_240_v1'},
        'provenance':{'frozen_as_of':'2026-10-20','context':{
            'feature_input_contract':{'matching_live_input_builder_required':True},
            'effective_signal_start':'2026-10-26','execution':daily._EXECUTION,
            'signal_policy':{'prediction_rule':{'min_net_edge':.0005}}}}}
    monkeypatch.setattr(observer,'_verify_pair',lambda *a:manifest)
    monkeypatch.setattr(observer,'_assets',lambda *a:{'immutable':'same'})
    monkeypatch.setattr(observer,'cn_now',lambda:now)
    monkeypatch.setattr(daily,'cn_now',lambda:now)
    cfg=observer.configure(tmp_path,frozen,start_date=now.date(),variant='repaired')
    return tmp_path,cfg,now


def test_repaired_account_freeze_start_and_before_evening_never_records(configured):
    root,cfg,now=configured
    assert cfg['account_id'].startswith('sf_repaired_')
    assert cfg['protocol']['ranking_column']=='pred_ret_raw_5d'
    assert cfg['protocol']['order_quantity_policy']=='whole_position_odd_lot_exit_v1'
    assert observer.tick(root,now=now.replace(minute=0),variant='repaired')['status']=='waiting_close'
    assert not list((daily._root(observer.view(root,variant='repaired'))/'days').glob('*/snapshot.json'))


def test_missing_research_reference_is_pending_not_a_zero_return(configured,monkeypatch):
    root,_cfg,now=configured
    monkeypatch.setattr(observer,'readiness',lambda *a:{'ready':True,'checks':{'日K':True}})
    def unavailable(*a):
        raise observer.WaitingDataError('缺少当日行业或换手率')
    monkeypatch.setattr(repaired,'prepare',unavailable)
    result=observer.tick(root,now=now,variant='repaired')
    assert result['status']=='waiting_data'
    status=observer.status(root,variant='repaired')
    assert status['recorded_days']==0 and status['cost_after_return_pct'] is None


def test_late_minutes_do_not_block_existing_queued_order_settlement(configured,monkeypatch):
    root,_cfg,now=configured
    settled=[]
    monkeypatch.setattr(observer,'readiness',lambda *a:{'ready':True,'checks':{'日K':True}})
    def unavailable(*a):
        raise observer.WaitingDataError('当天分钟尚未齐')
    monkeypatch.setattr(repaired,'prepare',unavailable)
    monkeypatch.setattr(repaired,'execution_ready',lambda *a:True)
    monkeypatch.setattr(observer,'_settle_sessions',lambda *a:settled.append(a[-1]) or {'status':'settled'})
    monkeypatch.setattr(daily,'_queue_exits',lambda *a:[])
    result=observer.tick(root,now=now,variant='repaired')
    assert settled==[now.date()] and result['settlement']['status']=='settled'
    assert result['status']=='waiting_data'
    assert observer.status(root,variant='repaired')['recorded_days']==0


def test_formal_record_stores_both_scores_reasons_and_next_open_once(configured,monkeypatch):
    from app.strategy import paper

    root,cfg,now=configured
    frame=pl.DataFrame({'symbol':['000001'],'date':[now.date()],
        'pred_ret_raw_5d':[.02],'pred_ret_5d':[.01],'net_margin_score':[.009],
        '_prediction_pass':[True],'_net_margin_pass':[True],'_forecast_pass':[True],
        '_entry_rules_pass':[True],'_market_gate_pass':[True]})
    coverage={'decision_pool_rows':1,'matched_inputs':{'next_actual_session_open':'2026-10-27T09:30:00+08:00'}}
    monkeypatch.setattr(observer,'readiness',lambda *a:{'ready':True,'checks':{'日K':True}})
    monkeypatch.setattr(observer,'_input_version',lambda *a:'stable')
    monkeypatch.setattr(observer,'_settle_sessions',lambda *a:{'status':'settled'})
    monkeypatch.setattr(observer,'_compute',lambda *a:(frame,frame,frame,coverage))
    monkeypatch.setattr(repaired,'prepare',lambda *a:{})
    monkeypatch.setattr(paper,'read_daily_bar',lambda *a:{'open':10.,'close':10.})
    monkeypatch.setattr(paper,'_now_iso',lambda:now.isoformat())
    first=observer.tick(root,now=now,variant='repaired')
    assert first['status']=='recorded' and first['eligible_for_forward']
    assert first['next_actual_session_open']=='2026-10-27T09:30:00+08:00'
    assert first['raw_positive_rows']==first['calibrated_positive_rows']==1
    scoped=observer.view(root,variant='repaired')
    saved=pl.read_parquet(daily._root(scoped)/f'days/{now.date()}/signals.parquet')
    assert saved['filter_reasons'].to_list()==[[]]
    observer.tick(root,now=now,variant='repaired')
    assert len(paper.load_orders(scoped,cfg['account_id']))==1
    assert observer.status(root,variant='repaired')['cost_after_return_pct'] is None


def test_quantity_context_is_restored_after_pipeline_failure(configured,monkeypatch):
    from app.strategy import paper

    root,_cfg,now=configured
    original=paper.create_order
    def broken(*a):
        raise RuntimeError('artificial readiness failure')
    monkeypatch.setattr(observer,'readiness',broken)
    assert observer.tick(root,now=now,variant='repaired')['status']=='error'
    assert paper.create_order is original and paper.LOT_SIZE==100


def test_missing_quantity_protocol_is_blocked_before_any_readiness_or_orders(configured,monkeypatch):
    root,cfg,now=configured
    cfg['protocol'].pop('order_quantity_policy')
    cfg['protocol_id']=daily._hash(cfg['protocol'])
    daily._write(daily._root(observer.view(root,variant='repaired'))/'config.json',cfg)
    monkeypatch.setattr(observer,'readiness',lambda *a:pytest.fail('unfrozen protocol entered pipeline'))
    assert observer.tick(root,now=now,variant='repaired')['status']=='blocked_protocol_changed'


def test_corporate_action_is_checked_before_a_pending_sell_can_escape(tmp_path,monkeypatch):
    from app.strategy import paper

    folder=tmp_path/'materialized'
    folder.mkdir()
    day=date(2026,10,23)
    raw=pl.DataFrame({'symbol':['000001'],'date':[day],'open':[10.],
        'high':[10.],'low':[10.],'close':[10.],'volume':[1000.],'amount':[10000.]})
    raw.write_parquet(folder/'raw.parquet')
    pl.DataFrame({'symbol':['000001','000001'],'date':[date(2026,10,22),day],
        'adj_factor':[7.25,7.5]}).write_parquet(folder/'factors.parquet')
    raw.select('symbol','date').with_columns(pl.lit(11.).alias('up_limit'),
        pl.lit(9.).alias('down_limit')).write_parquet(folder/'limits.parquet')
    daily._write(tmp_path/'research_inputs/latest.json',{'path':str(folder)})
    monkeypatch.setattr(paper,'load_positions',lambda *a:{'000001':{'qty':100}})
    monkeypatch.setattr(daily,'_settle',lambda *a:pytest.fail('ledger touched before ex-date review'))
    with pytest.raises(ValueError,match='公司行动'):
        repaired.settle(tmp_path,{'account_id':'test'},day)
