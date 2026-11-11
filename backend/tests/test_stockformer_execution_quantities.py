"""Real order preflight and settlement, confined to ephemeral research accounts."""
from datetime import date, datetime

import pytest

from app.market_time import CN_TZ
from app.services.stockformer_execution_quantities import POLICY, environment
from app.strategy import paper


@pytest.fixture
def account(tmp_path,monkeypatch):
    aid='odd_exit_test'
    paper.create_account(tmp_path,10000,account_id=aid)
    pos={'002943':{'asset_type':'stock','qty':130,'avg_cost':8.,'available_qty':130,
        'lots':[{'date':'2026-10-21','qty':130}]}}
    monkeypatch.setattr(paper,'load_positions',lambda *a,**k:pos)
    monkeypatch.setattr(paper,'cn_today',lambda:date(2026,10,22))
    monkeypatch.setattr(paper,'cn_now',lambda:datetime(2026,10,22,19,10,tzinfo=CN_TZ))
    return tmp_path,aid,pos


def active(root,aid):
    return environment(root,aid,{'002943'})


def test_frozen_default_rejects_but_new_policy_can_queue_130_share_exit(account):
    root,aid,_pos=account
    assert paper.create_order(root,'002943','sell',account_id=aid,qty=130)[0] is None
    with active(root,aid):
        order,error=paper.create_order(root,'002943','sell',account_id=aid,qty=130,order_type='next_open')
        assert error is None and order['qty']==130
        assert paper.get_order(root,order['id'],aid)['qty']==130
    assert paper.LOT_SIZE==100 and POLICY=='whole_position_odd_lot_exit_v1'


@pytest.mark.parametrize('qty',[1,29,30,129,131,130.5,True])
def test_partial_odd_sells_or_invalid_quantities_are_not_permitted(account,qty):
    root,aid,_pos=account
    with active(root,aid):
        assert paper.create_order(root,'002943','sell',account_id=aid,qty=qty)[0] is None
    assert paper.load_orders(root,aid)==[]


def test_full_small_balance_can_exit_once(account):
    root,aid,pos=account
    pos['002943'].update(qty=30,lots=[{'date':'2026-10-21','qty':30}])
    with active(root,aid):
        order,error=paper.create_order(root,'002943','sell',account_id=aid,qty=30)
        assert order and error is None
        assert paper.create_order(root,'002943','sell',account_id=aid,qty=30)[0] is None


def test_buy_quantities_remain_whole_lots(account):
    root,aid,_pos=account
    with active(root,aid):
        assert paper.create_order(root,'002943','buy',account_id=aid,qty=130)[0] is None
        assert paper.create_order(root,'002943','buy',account_id=aid,qty=100,ref_price=8)[0]


def test_t1_and_earlier_sell_reservations_still_apply(account):
    root,aid,pos=account
    with active(root,aid):
        pos['002943']['lots']=[{'date':'2026-10-22','qty':130}]
        order,error=paper.create_order(root,'002943','sell',account_id=aid,qty=130)
        assert order is None and 'T+1' in error
        pos['002943']['lots']=[{'date':'2026-10-21','qty':130}]
        assert paper.create_order(root,'002943','sell',account_id=aid,qty=100)[0]
        order,error=paper.create_order(root,'002943','sell',account_id=aid,qty=130)
        assert order is None and '待成交卖出' in error


def test_other_accounts_paths_symbols_and_etfs_cannot_use_exemption(account,tmp_path):
    root,aid,pos=account
    paper.create_account(root,10000,account_id='other')
    paper.create_account(root/'other-path',10000,account_id=aid)
    with active(root,aid):
        assert paper.create_order(root,'002943','sell',account_id='other',qty=130)[0] is None
        assert paper.create_order(root/'other-path','002943','sell',account_id=aid,qty=130)[0] is None
        assert paper.create_order(root,'unknown','sell',account_id=aid,qty=130)[0] is None
        pos['002943']['asset_type']='etf'
        assert paper.create_order(root,'002943','sell',account_id=aid,qty=130)[0] is None


def test_exception_restores_original_engine_constants_and_handler(account,monkeypatch):
    root,aid,_pos=account
    original=paper.create_order
    def failure(*a,**kw):
        raise OSError('artificial order-store failure')
    monkeypatch.setattr(paper,'create_order',failure)
    with active(root,aid):
        with pytest.raises(OSError):
            paper.create_order(root,'002943','sell',account_id=aid,qty=130)
        assert paper.LOT_SIZE==100
    assert paper.create_order is failure
    monkeypatch.setattr(paper,'create_order',original)


def test_next_open_fill_uses_all_130_shares_and_keeps_fees_limits_t1(tmp_path,monkeypatch):
    aid='settle_odd_test'
    paper.create_account(tmp_path,10000,account_id=aid)
    # An artificial verified 100 -> 130 share fixture, not a price factor.
    paper._append_fill(tmp_path,dict(kind='fill',date='2026-10-20',symbol='002943',
        asset_type='stock',side='buy',qty=100,price=10.,fee=5.),aid)
    paper._append_fill(tmp_path,dict(kind='corp_action',date='2026-10-21',
        symbol='002943',factor=1.3),aid)
    monkeypatch.setattr(paper,'cn_today',lambda:date(2026,10,21))
    monkeypatch.setattr(paper,'cn_now',lambda:datetime(2026,10,21,19,10,tzinfo=CN_TZ))
    paper.rebuild_positions(tmp_path,aid)
    monkeypatch.setattr(paper,'read_daily_bar',lambda *a:{'open':8.,'close':8.})
    monkeypatch.setattr(paper,'_prev_close',lambda *a:8.)
    monkeypatch.setattr(paper,'_index_close',lambda *a:None)
    monkeypatch.setattr(paper,'_factor_on',lambda *a:None)
    with active(tmp_path,aid):
        order,error=paper.create_order(tmp_path,'002943','sell',account_id=aid,qty=130,order_type='next_open')
        assert order and error is None
        monkeypatch.setattr(paper,'cn_today',lambda:date(2026,10,22))
        result=paper.settle_day(tmp_path,'2026-10-22',aid)
    assert result['filled']==1
    fill=paper.load_fills(tmp_path,aid)[-1]
    acc=paper.get_account(tmp_path,aid)
    assert fill['qty']==130 and fill['price']==paper.apply_slippage(8.,'sell',acc['slippage_bps'])
    assert fill['fee']==paper.sell_fee(130,fill['price'],acc['commission_pct'],acc['stamp_tax_pct'])
    assert acc['cash']==round(8995+130*fill['price']-fill['fee'],2)
    assert paper.load_positions(tmp_path,aid)['002943']['qty']==0


def test_limit_down_still_blocks_a_full_odd_exit(account,monkeypatch):
    root,aid,_pos=account
    with active(root,aid):
        order,error=paper.create_order(root,'002943','sell',account_id=aid,qty=130)
        assert order and error is None
        monkeypatch.setattr(paper,'_prev_close',lambda *a:10.)
        assert paper._fill_order(root,order,8.,'2026-10-22',aid) is None
        assert order['status']=='expired'


def test_nested_context_is_rejected_and_outer_policy_restored(account):
    root,aid,_pos=account
    original=paper.create_order
    with active(root,aid):
        with pytest.raises(RuntimeError,match='already'), active(root,aid):
            pass
        assert paper.create_order(root,'002943','sell',account_id=aid,qty=130)[0]
    assert paper.create_order is original and paper.LOT_SIZE==100


def test_reentrant_callback_cannot_bypass_another_accounts_quantity_gate(account,monkeypatch):
    root,aid,_pos=account
    def recursive(*a,**k):
        return paper.create_order(root,'002943','buy',account_id='other',qty=130)
    monkeypatch.setattr(paper,'create_order',recursive)
    with active(root,aid):
        with pytest.raises(RuntimeError,match='Reentrant'):
            paper.create_order(root,'002943','sell',account_id=aid,qty=130)
        assert paper.LOT_SIZE==100
