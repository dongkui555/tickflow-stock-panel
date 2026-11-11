"""Synthetic sessions exercise the real matcher, postings and restart boundaries."""
from contextlib import contextmanager
from datetime import date, datetime

import polars as pl
import pytest

from app.market_time import CN_TZ
from app.services import stockformer_corporate_ledger as ledger
from app.services import stockformer_corporate_receipts as receipts
from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_pair_shadow as observer
from app.services import stockformer_repaired_shadow as repaired
from app.services.stockformer_execution_quantities import environment as quantities
from app.strategy import paper

DAYS=[date(2026,10,n) for n in [23,26,27,28,29,30]]


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root=observer.view(tmp_path,variant='repaired')
    root.mkdir(parents=True)
    clock={'day':DAYS[0]}
    def now():
        return datetime.combine(clock['day'],datetime.min.time(),tzinfo=CN_TZ).replace(hour=clock.get('hour',22))
    monkeypatch.setattr(paper,'cn_today',lambda:clock['day'])
    monkeypatch.setattr(paper,'cn_now',now)
    monkeypatch.setattr(daily,'cn_now',now)
    monkeypatch.setattr(receipts,'cn_now',now)
    cfg=dict(account_id='corp_test',protocol_id='protocol-test',enabled=True,
        protocol=dict(start_date=str(DAYS[1]),execution=daily._EXECUTION,
                      corporate_action_policy=ledger.POLICY))
    paper.create_account(root,10000,account_id=cfg['account_id'],commission_pct=.0003,stamp_tax_pct=.0005)
    daily._write(daily._root(root)/'config.json',cfg)
    raw=pl.DataFrame([dict(symbol=s,date=d,open=p,high=p,low=p,close=p,volume=1000.,amount=1000*p)
        for d in DAYS for s,p in [('002943',10. if d<=DAYS[1] else 9.),('000001',.5)]])
    factors=raw.select('symbol','date').with_columns(pl.when((pl.col('symbol')=='002943')&(pl.col('date')>=DAYS[2]))
        .then(10.1).otherwise(10.).alias('adj_factor'))
    limits=raw.select('symbol','date').with_columns((raw['close']*1.1).alias('up_limit'),(raw['close']*.9).alias('down_limit'))
    material=root/'material'
    material.mkdir()
    for n,f in [('raw',raw),('factors',factors),('limits',limits)]:
        f.write_parquet(material/f'{n}.parquet')
    def manifest():
        daily._write(root/'research_inputs/latest_execution.json',dict(date=str(DAYS[-1]),path=str(material),
            files={f'{n}.parquet':daily._digest(material/f'{n}.parquet') for n in ['raw','factors','limits']}))
    manifest()
    daily._write(root/'research_inputs/captures'/f'{DAYS[-1]}-calendar.json',
        {'records':[{'trade_date':str(d),'is_open':1} for d in DAYS]})
    event=dict(event_id='test',symbol='002943',exchange='SZ',known_on='2026-10-23',record_date=str(DAYS[1]),
        ex_date=str(DAYS[2]),cash_per_eligible_share='1',transfer_per_eligible_share='0',bonus_per_eligible_share='0',
        pay_date=str(DAYS[4]),share_credit_date=None,share_listing_date=None,share_tax=None,
        tax_policy='cn_individual_unrestricted_fifo',cash_rounding='exact_fen_only')
    native=dict(ts_code='002943.SZ',div_proc='实施',ex_date=str(DAYS[2]).replace('-',''),
        record_date=str(DAYS[1]).replace('-',''),pay_date=str(DAYS[4]).replace('-',''),cash_div_tax=1,
        stk_div=0,stk_co_rate=0,div_listdate=None)
    folder=root/'research_inputs/corporate_receipts'/f'002943-{DAYS[2]}'
    folder.mkdir(parents=True)
    def receipt():
        daily._write(folder/'native.json',{'records':[native]})
        (folder/'notice.txt').write_text('SYNTHETIC UNIT TEST NOTICE; NOT PRODUCTION EVIDENCE')
        daily._write(folder/'receipt.json',dict(schema_version=1,event=event,factor_before=10.,
            factor_after=pl.read_parquet(material/'factors.parquet').filter((pl.col('symbol')=='002943')&(pl.col('date')==DAYS[2]))['adj_factor'].item(),
            review={'issuer_explicit_no_cash':True,'issuer_premium_capital_tax_exempt':True},
            evidence=[dict(role=role,file=file,sha256=daily._digest(folder/file),
                url='https://static.cninfo.com.cn/unit-test') for role,file in [('native','native.json'),('issuer','notice.txt')]]))
    receipt()
    monkeypatch.setattr(receipts,'capture',lambda root,symbol,day:({'records':[native] if symbol=='002943' and day==DAYS[2] else []},folder/'native.json'))
    @contextmanager
    def active():
        with quantities(root,cfg['account_id'],{'002943','000001'}),ledger.environment(root,cfg) as ctx:
            yield ctx
    def order(side,symbol='002943',qty=100):
        value,error=paper.create_order(root,symbol,side,account_id=cfg['account_id'],qty=qty,order_type='next_open')
        assert value,error
        return value
    def settle(day):
        clock['day']=day
        return repaired.settle(root,cfg,day)
    return dict(root=root,cfg=cfg,clock=clock,material=material,manifest=manifest,event=event,native=native,
                folder=folder,receipt=receipt,active=active,order=order,settle=settle,data_root=tmp_path)


def enter_position(s):
    s['order']('buy')
    s['settle'](DAYS[1])


def queue_ex_morning_sell(s):
    s['clock'].update(day=DAYS[2],hour=8)
    s['order']('sell')
    s['clock']['hour']=22


def test_cash_sale_keeps_rights_until_payment_and_metrics_follow_same_ledger(setup):
    s=setup
    with s['active']():
        enter_position(s)
        queue_ex_morning_sell(s)
        s['settle'](DAYS[2])
        acc=paper.get_account(s['root'],s['cfg']['account_id'])
        assert acc['cash']==9868.6
        assert paper.load_nav(s['root'],s['cfg']['account_id'])[-1]['dividend_receivable']==100
        assert paper.round_trips(s['root'],s['cfg']['account_id'])==[]
        s['settle'](DAYS[3])
        s['settle'](DAYS[4])
        assert paper.get_account(s['root'],s['cfg']['account_id'])['cash']==9968.6
        rounds=paper.round_trips(s['root'],s['cfg']['account_id'])
        assert len(rounds)==1 and rounds[0]['pnl']==-31.4
    status=observer.status(s['data_root'],variant='repaired')
    assert status['completed_trades']==1 and status['cost_after_return_pct']==pytest.approx(-.314)
    with s['active']():
        s['settle'](DAYS[4])
        rows=paper.load_fills(s['root'],s['cfg']['account_id'])
        assert sum(r.get('kind')=='cash_payment_v1' for r in rows)==1


def test_cash_due_at_close_cannot_finance_that_mornings_buy(setup):
    s=setup
    acc=paper.get_account(s['root'],s['cfg']['account_id'])
    acc.update(initial_cash=1020,cash=1020)
    paper.save_account(s['root'],acc,s['cfg']['account_id'])
    with s['active']():
        enter_position(s)
        s['settle'](DAYS[2])
        s['settle'](DAYS[3])
        order=s['order']('buy','000001')
        s['settle'](DAYS[4])
        assert paper.get_order(s['root'],order['id'],s['cfg']['account_id'])['status']=='expired'
        assert paper.get_account(s['root'],s['cfg']['account_id'])['cash']==94.5


def test_transfer_waits_for_listing_and_130_exit_counts_one_trade(setup):
    s=setup
    s['event'].update(cash_per_eligible_share='0',transfer_per_eligible_share='.3',pay_date=None,
        share_credit_date=str(DAYS[3]),share_listing_date=str(DAYS[4]),share_tax='premium_capital_exempt')
    s['native'].update(cash_div_tax=None,stk_div=.3,stk_co_rate=.3,pay_date=None,div_listdate=str(DAYS[4]).replace('-',''))
    s['receipt']()
    with s['active']():
        enter_position(s)
        s['settle'](DAYS[2])
        pos=paper.load_positions(s['root'],s['cfg']['account_id'])['002943']
        assert pos['qty']==130 and pos['pending_share_qty']==30 and pos['available_qty']==100
        assert paper.create_order(s['root'],'002943','sell',qty=130,account_id=s['cfg']['account_id'])[0] is None
        s['settle'](DAYS[3])
        s['settle'](DAYS[4])
        s['order']('sell',qty=130)
        s['settle'](DAYS[5])
        assert paper.load_positions(s['root'],s['cfg']['account_id'])=={}
        rounds=paper.round_trips(s['root'],s['cfg']['account_id'])
        assert len(rounds)==1 and rounds[0]['qty']==130
        nav=paper.load_nav(s['root'],s['cfg']['account_id'])[-1]
        assert rounds[0]['pnl']==pytest.approx(nav['nav']-10000)


def test_odd_lot_fractional_cent_gross_cash_matches_paper_nav(setup):
    s=setup
    s['event'].update(cash_per_eligible_share='0',transfer_per_eligible_share='.3',pay_date=None,
        share_credit_date=str(DAYS[2]),share_listing_date=str(DAYS[2]),share_tax='premium_capital_exempt')
    s['native'].update(cash_div_tax=None,stk_div=.3,stk_co_rate=.3,pay_date=None,
                       div_listdate=str(DAYS[2]).replace('-',''))
    s['receipt']()
    path=s['material']/'raw.parquet'
    pl.read_parquet(path).with_columns([pl.when((pl.col('date')==DAYS[-1])&(pl.col('symbol')=='002943'))
        .then(9.07).otherwise(pl.col(c)).alias(c) for c in ['open','high','low','close']]).write_parquet(path)
    s['manifest']()
    with s['active']():
        enter_position(s)
        for day in DAYS[2:-1]:
            s['settle'](day)
        s['order']('sell',qty=130)
        s['settle'](DAYS[-1])
        nav=paper.load_nav(s['root'],s['cfg']['account_id'])[-1]
        assert paper.get_account(s['root'],s['cfg']['account_id'])['cash']==nav['cash_gross']==nav['nav']


def test_restart_after_fill_append_before_order_save_never_duplicates_fill(setup):
    s=setup
    with s['active']() as ctx:
        order=s['order']('buy')
        original=ctx.saved['save_order']
        def crash(root,value,aid):
            if value['status']=='filled':
                raise RuntimeError('crash after append')
            return original(root,value,aid)
        ctx.saved['save_order']=crash
        with pytest.raises(RuntimeError,match='crash'):
            s['settle'](DAYS[1])
    with s['active']():
        result=s['settle'](DAYS[1])
        fills=[r for r in paper.load_fills(s['root'],s['cfg']['account_id']) if r.get('kind','fill')=='fill']
        assert result['filled']==len(fills)==1
        assert paper.get_order(s['root'],order['id'],s['cfg']['account_id'])['status']=='filled'


def test_restart_after_postpone_does_not_double_increment(setup):
    s=setup
    path=s['material']/'raw.parquet'
    pl.read_parquet(path).filter(~((pl.col('date')==DAYS[1])&(pl.col('symbol')=='000001'))).write_parquet(path)
    s['manifest']()
    with s['active']() as ctx:
        order=s['order']('buy','000001')
        original=ctx.nav
        paper.daily_nav=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('crash after postpone'))
        with pytest.raises(RuntimeError,match='crash'):
            s['settle'](DAYS[1])
        paper.daily_nav=original
    with s['active']():
        s['settle'](DAYS[1])
        assert paper.get_order(s['root'],order['id'],s['cfg']['account_id'])['postponed']==1


def test_restart_after_cash_posting_never_credits_twice(setup):
    s=setup
    with s['active']() as ctx:
        enter_position(s)
        queue_ex_morning_sell(s)
        s['settle'](DAYS[2])
        s['settle'](DAYS[3])
        original=ctx.append
        def crash(root,row,aid):
            original(root,row,aid)
            if row.get('kind')=='cash_payment_v1':
                raise RuntimeError('crash after cash posting')
        ctx.append=crash
        with pytest.raises(RuntimeError,match='crash'):
            s['settle'](DAYS[4])
    with s['active']():
        s['settle'](DAYS[4])
        assert paper.get_account(s['root'],s['cfg']['account_id'])['cash']==9968.6
        assert sum(r.get('kind')=='cash_payment_v1' for r in paper.load_fills(s['root'],s['cfg']['account_id']))==1


def test_unreviewed_or_changed_evidence_blocks_before_selling(setup):
    s=setup
    with s['active']():
        enter_position(s)
        queue_ex_morning_sell(s)
        path=s['folder']/'notice.txt'
        path.write_text('CHANGED')
        before=(paper.accounts_base(s['root'])/s['cfg']['account_id']/'fills.jsonl').read_bytes()
        with pytest.raises(ValueError,match='evidence changed'):
            s['settle'](DAYS[2])
        assert (paper.accounts_base(s['root'])/s['cfg']['account_id']/'fills.jsonl').read_bytes()==before


def test_corrupt_tail_fails_closed_and_status_hides_stale_profit(setup):
    s=setup
    with s['active']():
        enter_position(s)
    path=paper.accounts_base(s['root'])/s['cfg']['account_id']/'fills.jsonl'
    with path.open('a') as stream:
        stream.write('{"incomplete":')
    with s['active'](),pytest.raises(ValueError,match='Incomplete ledger'):
        s['settle'](DAYS[2])
    result=observer.status(s['data_root'],variant='repaired')
    assert result['completed_trades']==0 and result['cost_after_return_pct'] is None


def test_source_gate_waits_until_published_update_window(setup,monkeypatch):
    s=setup
    monkeypatch.undo()
    monkeypatch.setattr(receipts,'cn_now',lambda:datetime(2026,10,27,20,59,tzinfo=CN_TZ))
    with pytest.raises(observer.WaitingDataError,match='21:05'):
        receipts.capture(s['root'],'002943',DAYS[2])


def test_source_cache_cannot_certify_an_early_empty_response(setup,monkeypatch):
    s=setup
    monkeypatch.undo()
    monkeypatch.setattr(receipts,'cn_now',lambda:datetime(2026,10,27,22,tzinfo=CN_TZ))
    path=s['root']/'research_inputs/corporate_sources/2026-10-27-002943.json'
    item=dict(schema_version=1,date='2026-10-27',symbol='002943',ts_code='002943.SZ',
              complete=True,captured_at='2026-10-27T20:00:00+08:00',records=[])
    daily._write(path,item)
    with pytest.raises(ValueError,match='Invalid corporate source cache'):
        receipts.capture(s['root'],'002943',DAYS[2])
    item['captured_at']='2026-10-27T21:10:00+08:00'
    daily._write(path,item)
    assert receipts.capture(s['root'],'002943',DAYS[2])[0]==item


@pytest.mark.parametrize('change', ['mixed_native','fractional_grant','factor_version'])
def test_unsupported_or_mismatched_terms_never_reach_matching(setup,change):
    s=setup
    if change=='mixed_native':
        s['native']['stk_div']=.3
    elif change=='fractional_grant':
        s['event'].update(cash_per_eligible_share='0',transfer_per_eligible_share='.003',pay_date=None,
            share_credit_date=str(DAYS[2]),share_listing_date=str(DAYS[2]),share_tax='premium_capital_exempt')
        s['native'].update(cash_div_tax=None,stk_div=.003,stk_co_rate=.003,pay_date=None,
                           div_listdate=str(DAYS[2]).replace('-',''))
    s['receipt']()
    if change=='factor_version':
        value=daily._read(s['folder']/'receipt.json')
        value['factor_after']=11
        daily._write(s['folder']/'receipt.json',value)
    with s['active']():
        enter_position(s)
        before=(paper.accounts_base(s['root'])/s['cfg']['account_id']/'fills.jsonl').read_bytes()
        with pytest.raises(ValueError):
            s['settle'](DAYS[2])
        assert (paper.accounts_base(s['root'])/s['cfg']['account_id']/'fills.jsonl').read_bytes()==before


def test_partial_settlement_refuses_changed_price_inputs(setup):
    s=setup
    with s['active']():
        s['order']('buy')
        paper.daily_nav=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('interrupted'))
        with pytest.raises(RuntimeError,match='interrupted'):
            s['settle'](DAYS[1])
    path=s['material']/'raw.parquet'
    pl.read_parquet(path).with_columns((pl.col('open')+.01).alias('open')).write_parquet(path)
    s['manifest']()
    with s['active'](),pytest.raises(ValueError,match='Settlement inputs/protocol changed'):
        s['settle'](DAYS[1])


def test_closed_prefix_and_filled_order_financial_fields_are_reconciled(setup):
    s=setup
    with s['active']():
        enter_position(s)
    order=paper.load_orders(s['root'],s['cfg']['account_id'])[0]
    order['fill_price']+=.01
    paper.save_order(s['root'],order,s['cfg']['account_id'])
    with s['active'](),pytest.raises(ValueError,match='financial fields'):
        s['settle'](DAYS[1])
    path=paper.accounts_base(s['root'])/s['cfg']['account_id']/'fills.jsonl'
    path.write_bytes(path.read_bytes().replace(b'"fee": 5.0',b'"fee": 6.0'))
    with pytest.raises(ValueError,match='Closed ledger prefix changed'):
        ledger.strict_rows(path)


def test_current_source_must_match_pinned_native_receipt(setup):
    s=setup
    with s['active']():
        enter_position(s)
        s['native']['cash_div_tax']=2
        with pytest.raises(ValueError,match='Current corporate source disagrees'):
            s['settle'](DAYS[2])
