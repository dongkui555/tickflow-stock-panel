"""Independent accounting oracle; synthetic orders never enter forward accounts."""
from copy import deepcopy

import pytest

from app.services.stockformer_corporate_research import audit_account, tax_rate


def fill(day, side, qty=100, price='10', fee='5', symbol='002943'):
    return dict(date=day, side=side, qty=qty, price=price, fee=fee,
                symbol=symbol, kind='fill', order_id=f'{day}-{side}')


def event(**changes):
    item = dict(event_id='issuer-2026-040', symbol='002943', exchange='SZ',
        known_on='2026-06-02', record_date='2026-06-05', ex_date='2026-06-08',
        cash_per_eligible_share='0', transfer_per_eligible_share='0.3',
        bonus_per_eligible_share='0', share_credit_date='2026-06-08',
        share_listing_date='2026-06-08', pay_date=None,
        share_tax='premium_capital_exempt', tax_policy='cn_individual_unrestricted_fifo',
        price_reference_transfer='0.2976647', cash_rounding='floor_fen',
        evidence_verified=True)
    item.update(changes)
    return item


def audit(fills, events, as_of='2026-06-09', close='8'):
    return audit_account(fills, events, initial_cash='10000', as_of=as_of,
        sessions=['2026-06-04','2026-06-05','2026-06-08','2026-06-09','2026-06-10'],
        closes={'002943': {'date':as_of, 'close':close}})


def cash_event(**changes):
    item = dict(transfer_per_eligible_share='0', price_reference_transfer='0',
        cash_per_eligible_share='1', pay_date='2026-06-10', share_credit_date=None,
        share_listing_date=None)
    item.update(changes)
    return event(**item)


def test_real_issuer_transfer_uses_point_three_not_price_reference():
    result = audit([fill('2026-06-05','buy')],[event()])
    assert result['positions']['002943']['qty'] == 130
    assert result['cash_gross'] == 8995
    assert result['positions']['002943']['cost'] == 1005
    assert result['nav_after_reserve'] == 10035


def test_ex_date_new_purchase_is_not_entitled():
    result = audit([fill('2026-06-08','buy')],[event()])
    assert result['positions']['002943']['qty'] == 100
    assert result['entitlements'] == []


def test_record_date_sale_is_not_entitled():
    result = audit([fill('2026-06-04','buy'),fill('2026-06-05','sell')],[event()])
    assert result['entitlements'] == []


def test_cash_rights_survive_ex_date_sale_but_payment_cannot_be_spent_early():
    rows = [fill('2026-06-05','buy'),fill('2026-06-08','sell',price='9')]
    before = audit(rows,[cash_event()])
    assert before['cash_gross'] == 9890
    assert before['dividend_receivable'] == 100
    assert before['dividend_tax_accrued'] == 20
    assert before['cash_available'] == 9870
    assert before['completed_trades'] == 0
    assert before['trades'][0]['net_pnl'] == -30
    after = audit(rows,[cash_event()],as_of='2026-06-10')
    assert after['cash_gross'] == 9990
    assert after['dividend_receivable'] == 0
    assert after['completed_trades'] == 1
    assert after['nav_after_reserve'] == 9970


def test_t_plus_one_and_bonus_listing_are_separate_constraints():
    with pytest.raises(ValueError, match='Same-session'):
        audit([fill('2026-06-05','buy'),fill('2026-06-05','sell')],[])
    delayed = event(share_credit_date='2026-06-09',share_listing_date='2026-06-10')
    result = audit([fill('2026-06-05','buy')],[delayed],as_of='2026-06-08')
    assert result['positions']['002943']['available_qty'] == 100
    assert result['positions']['002943']['pending_share_qty'] == 30
    with pytest.raises(ValueError,match='available'):
        audit([fill('2026-06-05','buy'),fill('2026-06-09','sell',qty=130)],[delayed])


def test_transfer_round_trip_preserves_total_cost_and_both_fees():
    result = audit([fill('2026-06-05','buy'),fill('2026-06-08','sell',qty=130,price='8')],[event()])
    assert result['nav_after_reserve'] == 10030
    assert sum(t['net_pnl'] for t in result['trades']) == 30
    assert result['positions'] == {}


@pytest.mark.parametrize(('buy','sell','expected'),[
    ('2026-01-31','2026-02-28','0.2'),('2026-01-31','2026-03-01','0.1'),
    ('2025-06-05','2026-06-05','0.1'),('2025-06-05','2026-06-06','0'),
    ('2024-02-29','2025-02-28','0.1'),('2024-02-29','2025-03-01','0')])
def test_tax_uses_natural_calendar_boundaries(buy,sell,expected):
    assert str(tax_rate(buy,sell)) == expected


@pytest.mark.parametrize('changes,reason',[
    ({'known_on':'2026-06-06'},'known'),
    ({'share_listing_date':None},'listing'),
    ({'bonus_per_eligible_share':'0.1'},'taxable bonus'),
    ({'share_tax':'unknown'},'tax treatment'),
    ({'evidence_verified':False},'evidence'),
    ({'transfer_per_eligible_share':'NaN'},'finite'),
    ({'cash_per_eligible_share':None},'missing'),
    ({'cash_per_eligible_share':'1','transfer_per_eligible_share':'0','pay_date':None},'payment'),
])
def test_incomplete_or_unsupported_events_fail_closed(changes,reason):
    with pytest.raises(ValueError,match=reason):
        audit([fill('2026-06-05','buy')],[event(**changes)])


def test_fractional_shares_are_not_guessed():
    with pytest.raises(ValueError,match='fractional'):
        audit([fill('2026-06-05','buy')],[event(transfer_per_eligible_share='0.2976647')])


def test_missing_record_session_cannot_be_treated_as_zero_entitlement():
    bad = event(record_date='2026-06-03')
    with pytest.raises(ValueError,match='record session'):
        audit([fill('2026-06-04','buy')],[bad])


def test_duplicate_events_and_inconsistent_marks_are_rejected():
    with pytest.raises(ValueError,match='Duplicate event'):
        audit([], [event(),event()])
    with pytest.raises(ValueError,match='mark'):
        audit_account([fill('2026-06-05','buy')],[],initial_cash=10000,as_of='2026-06-09',
            sessions=['2026-06-05','2026-06-09'],closes={'002943':{'date':'2026-06-08','close':8}})


def test_input_objects_are_not_mutated_and_no_event_matches_normal_cash():
    rows=[fill('2026-06-05','buy'),fill('2026-06-08','sell',price='11',fee='6')]
    saved=deepcopy(rows)
    result=audit(rows,[])
    assert result['cash_available'] == 10089
    assert sum(t['net_pnl'] for t in result['trades']) == 89
    assert rows == saved


def test_cash_rounding_happens_per_account_not_per_purchase_lot():
    rows=[fill('2026-06-04','buy',qty=100),fill('2026-06-05','buy',qty=100)]
    result=audit(rows,[cash_event(cash_per_eligible_share='0.00009')],as_of='2026-06-10')
    assert sum(r['gross'] for r in result['cashflows']) == .01


def test_combined_distribution_is_explicitly_unsupported():
    with pytest.raises(ValueError,match='Mixed cash/transfer'):
        audit([], [event(cash_per_eligible_share='0.4',pay_date='2026-06-08')])


def test_cash_on_previously_transferred_lot_needs_tax_origin_evidence():
    first=event(known_on='2026-06-01',record_date='2026-06-04',ex_date='2026-06-05',
        share_credit_date='2026-06-05',share_listing_date='2026-06-05')
    second=cash_event(event_id='second')
    with pytest.raises(ValueError,match='tax origin'):
        audit([fill('2026-06-04','buy')],[first,second])


def test_duplicate_raw_fill_dates_and_unknown_action_kinds_do_not_silently_pass():
    bad=fill('2026-06-05','buy')
    bad['kind']='corp_action'
    with pytest.raises(ValueError,match='price-factor'):
        audit([bad],[])
    with pytest.raises(ValueError,match='actual session'):
        audit([],[],as_of='2026-06-07')


def test_no_unstated_fractional_fen_rounding():
    with pytest.raises(ValueError,match='fractional-fen'):
        audit([fill('2026-06-05','buy')],
              [cash_event(cash_per_eligible_share='0.00009',cash_rounding='exact_fen_only')])
