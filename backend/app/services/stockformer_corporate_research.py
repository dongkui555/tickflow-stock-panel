"""Pure accounting reducer and read-only audit for adjudicated research events.

This is not a matching engine. The execution adapter supplies verified facts.
Orders, raw fills, commission and slippage must come from the existing paper
engine. Evidence review is an upstream requirement; an API row alone is not an
adjudicated event. Tax accrual/reserve below is an economic simulation, not a
claim about a broker's actual debit timestamp.
"""
from __future__ import annotations

from calendar import monthrange
from copy import deepcopy
from datetime import date, timedelta
from decimal import ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP, Decimal

ZERO = Decimal('0')
CENT = Decimal('.01')


def _number(value, label):
    if value is None or isinstance(value, bool):
        raise ValueError('missing numeric '+label)
    try:
        number = Decimal(str(value))
    except ArithmeticError:
        raise ValueError('invalid numeric '+label) from None
    if not number.is_finite() or number < 0:
        raise ValueError('non-finite or negative '+label)
    return number


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('Expected ISO date')
    return date.fromisoformat(value)


def _anniversary(day, *, months=0, years=0):
    number = day.year*12+day.month-1+months+years*12
    year, month = divmod(number, 12)
    return date(year, month+1, min(day.day, monthrange(year, month+1)[1]))


def tax_rate(buy_date, sell_date):
    """Unrestricted individual A-share cash dividends; calendar months/years."""
    buy, sell = _day(buy_date), _day(sell_date)
    if sell < buy:
        raise ValueError('Sale precedes purchase')
    if sell <= _anniversary(buy, months=1):
        return Decimal('0.2')
    return Decimal('0.1') if sell <= _anniversary(buy, years=1) else ZERO


def _event(item, sessions):
    value = deepcopy(item)
    if not value.get('evidence_verified'):
        raise ValueError('Unreviewed event evidence')
    if not value.get('event_id') or not value.get('symbol') or value.get('exchange') not in {'SH','SZ','BJ'}:
        raise ValueError('Missing event identity')
    known, record, ex = [_day(value[k]) for k in ('known_on','record_date','ex_date')]
    if known > record or record >= ex:
        raise ValueError('Event not known before record date')
    if record not in sessions or ex not in sessions:
        raise ValueError('Missing record session or ex session')
    if next((d for d in sessions if d > record), None) != ex:
        raise ValueError('Record and ex dates are not adjacent actual sessions')
    cash, transfer, bonus = [_number(value.get(k),k) for k in (
        'cash_per_eligible_share','transfer_per_eligible_share','bonus_per_eligible_share')]
    if bonus:
        raise ValueError('Unsupported taxable bonus: verify tax-lot allocation first')
    if value.get('tax_policy') != 'cn_individual_unrestricted_fifo':
        raise ValueError('Unsupported shareholder tax policy')
    if not cash and not transfer:
        raise ValueError('Empty distribution')
    if cash and transfer:
        raise ValueError('Mixed cash/transfer requires verified tax-lot allocation')
    if cash and (not value.get('pay_date') or _day(value['pay_date']) < ex):
        raise ValueError('Missing or invalid payment date')
    if value.get('cash_rounding') not in {'floor_fen','half_up_fen','exact_fen_only'}:
        raise ValueError('Missing cash rounding evidence')
    if transfer:
        if value.get('share_tax') != 'premium_capital_exempt':
            raise ValueError('Unverified transfer tax treatment')
        if not value.get('share_listing_date') or not value.get('share_credit_date'):
            raise ValueError('Missing credit/listing date')
        if not ex <= _day(value['share_credit_date']) <= _day(value['share_listing_date']):
            raise ValueError('Invalid credit/listing order')
    value.update(cash=cash, transfer=transfer, record=record, ex=ex)
    return value


def replay_account(fills, events, *, initial_cash, as_of, sessions, closes,
                   phase='close', payment_dates=None, include_lots=False, require_marks=True):
    """Replay economic rights without modifying a file, account or input object.

    Supported: cash-only distributions and tax-exempt premium-capital transfers
    with integral per-lot grants. Mixed distributions, taxable bonus shares and
    stochastic fractional allocation fail closed. Cash is credited at pay-date
    close, so this oracle never finances that day's opening orders with it.
    Open entitlements reserve a maximum 20% cash-dividend tax; sale fixes FIFO
    tax using the actual holding period. A round is complete only after its
    cash distribution has been paid. Marks must belong to as_of exactly.
    """
    through = _day(as_of)
    if phase not in {'open','close'}:
        raise ValueError('Unknown accounting phase')
    calendar = sorted(_day(d) for d in sessions)
    if len(calendar) != len(set(calendar)) or not calendar:
        raise ValueError('Duplicate or empty session calendar')
    if through not in calendar:
        raise ValueError('Valuation date is not an actual session')
    actions = [_event(e, calendar) for e in events if _day(e['known_on']) <= through]
    if len({e['event_id'] for e in actions}) != len(actions):
        raise ValueError('Duplicate event')
    if len({(e['symbol'],e['ex']) for e in actions}) != len(actions):
        raise ValueError('Conflicting same-day distribution')
    rows = []
    for index, source in enumerate(fills):
        f = deepcopy(source)
        day = _day(f['date'])
        if day > through:
            continue
        if day not in calendar or f.get('kind','fill') != 'fill' or f['side'] not in {'buy','sell'}:
            raise ValueError('Only actual-session raw fills accepted; no price-factor actions')
        qty = _number(f['qty'],'fill qty')
        price, fee = _number(f['price'],'fill price'), _number(f.get('fee',0),'fill fee')
        if not qty or qty != qty.to_integral_value() or not price:
            raise ValueError('Invalid integral fill quantity or raw price')
        f.update(qty=int(qty), price=price, fee=fee, index=index, day=day)
        rows.append(f)
    for f in rows:
        if any(r['day']==f['day'] and r['symbol']==f['symbol'] and r['side']!=f['side'] for r in rows):
            raise ValueError('Same-session buy/sell needs custodian net-share tax reconciliation')
    cash = _number(initial_cash,'initial_cash')
    lots, rights, trades, cashflows = [], [], [], []
    timeline = set(d for d in calendar if d <= through)
    timeline |= {e['record'] for e in actions if e['record'] <= through}
    timeline |= {e['ex'] for e in actions if e['ex'] <= through}
    timeline |= {_day(e['pay_date']) for e in actions if e.get('pay_date') and _day(e['pay_date']) <= through}
    if payment_dates is not None:
        for eid,posted in payment_dates.items():
            event=next((e for e in actions if e['event_id']==eid),None)
            if event is None or not event.get('pay_date') or _day(posted)<_day(event['pay_date']):
                raise ValueError('Invalid cash payment fact')
        timeline |= {_day(d) for d in payment_dates.values() if _day(d)<=through}
    for day in sorted(timeline):
        for e in actions:
            if e['ex'] != day:
                continue
            for right in [r for r in rights if r['event_id']==e['event_id']]:
                lot = next(lot for lot in lots if lot['id']==right['lot_id'])
                if lot['qty'] != right['qty']:
                    raise ValueError('Entitlement changed between record close and ex open')
                right['active'] = True
                if e['transfer']:
                    if any(r['active'] and r['cash'] and r['lot_id']==lot['id'] for r in rights):
                        raise ValueError('Transfer with outstanding cash tax needs allocation evidence')
                    granted = Decimal(lot['qty'])*e['transfer']
                    if granted != granted.to_integral_value():
                        raise ValueError('Unresolved fractional share allocation')
                    if not granted:
                        continue
                    cost = lot['cost']*granted/(Decimal(lot['qty'])+granted)
                    lot['cost'] -= cost
                    lots.append(dict(id=lot['id']+':'+e['event_id'], symbol=lot['symbol'],
                        date=lot['date'], qty=int(granted), cost=cost,
                        available_on=_day(e['share_listing_date']),
                        credit_on=_day(e['share_credit_date'])))
        for f in sorted((r for r in rows if r['day']==day),key=lambda r:r['index']):
            if f['side']=='buy':
                cash -= f['qty']*f['price']+f['fee']
                lots.append(dict(id=str(f['index']),symbol=f['symbol'],date=day,qty=f['qty'],
                    cost=f['qty']*f['price']+f['fee'],available_on=day+timedelta(days=1),credit_on=day))
                continue
            pool = [lot for lot in lots if lot['symbol']==f['symbol'] and lot['qty'] and lot['available_on']<=day]
            if sum(lot['qty'] for lot in pool) < f['qty']:
                raise ValueError('Insufficient available shares (T+1/listing)')
            cash += f['qty']*f['price']-f['fee']
            remaining = f['qty']
            for lot in sorted(pool,key=lambda lot:(lot['date'],lots.index(lot))):
                take = min(remaining,lot['qty'])
                cost = lot['cost']*Decimal(take)/lot['qty']
                linked = []
                for right in rights:
                    if right['lot_id'] != lot['id'] or not right['active'] or not right['cash']:
                        continue
                    gross = right['cash']*Decimal(take)/right['qty']
                    tax = gross*tax_rate(str(lot['date']),str(day))
                    right['sold_qty'] += take
                    right['tax'] += tax
                    linked.append(dict(event_id=right['event_id'],gross=gross,tax=tax))
                trades.append(dict(symbol=lot['symbol'],open_date=str(lot['date']),close_date=str(day),
                    origin_lot_id=lot['id'].split(':')[0],sell_fill_index=f['index'],cost_basis=cost,
                    qty=take,price_pnl=f['price']*take-f['fee']*Decimal(take)/f['qty']-cost,
                    distributions=linked))
                lot['qty'] -= take
                lot['cost'] -= cost
                remaining -= take
                if not remaining:
                    break
        # Entitlements belong to the record-date close, including buys on it.
        for e in actions:
            if e['record'] == day:
                eligible = [lot for lot in lots if lot['symbol']==e['symbol'] and lot['qty']]
                rounding = ROUND_DOWN if e['cash_rounding']=='floor_fen' else ROUND_HALF_UP
                exact = e['cash']*sum(lot['qty'] for lot in eligible)
                gross = exact.quantize(CENT,rounding=rounding)
                if e['cash_rounding']=='exact_fen_only' and gross != exact:
                    raise ValueError('Unverified fractional-fen cash rounding')
                distributed = ZERO
                for index, lot in enumerate(eligible):
                    if lot['credit_on'] > day:
                        raise ValueError('Uncredited shares need verified registration entitlement')
                    if e['cash'] and ':' in lot['id']:
                        raise ValueError('Transferred-share tax origin requires independent evidence')
                    # The custodian pays one account total. Lot allocations are
                    # analytic only; retain its exact cents, including residue.
                    amount = ((e['cash']*lot['qty']).quantize(CENT,rounding=ROUND_DOWN)
                              if index<len(eligible)-1 else gross-distributed)
                    distributed += amount
                    rights.append(dict(event_id=e['event_id'],lot_id=lot['id'],qty=lot['qty'],
                        cash=amount,active=False,
                        sold_qty=0,tax=ZERO,paid=False))
            paid_on = payment_dates.get(e['event_id']) if payment_dates is not None else e.get('pay_date')
            if paid_on == str(day) and (day < through or phase=='close'):
                for right in [r for r in rights if r['event_id']==e['event_id'] and r['active']]:
                    right['paid'] = True
                    cash += right['cash']
                    cashflows.append(dict(date=str(day),event_id=e['event_id'],gross=right['cash']))
    active = [r for r in rights if r['active']]
    receivable = sum((r['cash'] for r in active if not r['paid']),ZERO)
    tax = sum((r['tax'] for r in active),ZERO)
    reserve = sum((r['cash']*(r['qty']-r['sold_qty'])/r['qty']*Decimal('.2') for r in active),ZERO)
    positions, mv = {}, ZERO
    for lot in lots:
        if not lot['qty']:
            continue
        symbol = lot['symbol']
        mark = closes.get(symbol)
        valid_mark=mark and mark.get('date') == as_of and _number(mark.get('close'),'mark')
        if require_marks and not valid_mark:
            raise ValueError('Missing exact-date raw mark: '+symbol)
        qty = lot['qty']
        if valid_mark:
            mv += qty*_number(mark['close'],'mark')
        pos = positions.setdefault(symbol,dict(qty=0,pending_share_qty=0,available_qty=0,cost=ZERO))
        pos['cost'] += lot['cost']
        pos['qty' if lot['credit_on']<=through else 'pending_share_qty'] += qty
        pos['available_qty'] += qty if lot['available_on']<=through else 0
        if include_lots:
            pos.setdefault('lots',[]).append({k:v for k,v in lot.items() if k!='symbol'})
    for trade in trades:
        trade['net_pnl'] = trade['price_pnl']+sum((d['gross']-d['tax'] for d in trade['distributions']),ZERO)
        if include_lots:
            trade['net_pnl_exact']=str(trade['net_pnl'])
            trade['cost_basis_exact']=str(trade['cost_basis'])
        trade['cash_settled'] = all(next(r['paid'] for r in active if r['event_id']==d['event_id'])
                                    for d in trade['distributions'])
    def numeric(value):
        if isinstance(value, Decimal):
            return float(value.quantize(CENT,rounding=ROUND_HALF_UP))
        if isinstance(value, date):
            return str(value)
        if isinstance(value, list):
            return [numeric(v) for v in value]
        if isinstance(value, dict):
            return {k:numeric(v) for k,v in value.items()}
        return value
    return numeric(dict(as_of=as_of,
        # Match the existing paper account's cent-valued gross cash first;
        # floor only the remaining tax-reserved spendable balance. Odd-lot
        # fills at four-decimal simulated prices can contain fractional cents.
        cash_gross=cash,cash_available=(cash.quantize(CENT,rounding=ROUND_HALF_UP)-tax-reserve).quantize(CENT,rounding=ROUND_FLOOR),dividend_receivable=receivable,
        dividend_tax_accrued=tax,dividend_tax_reserve=reserve,positions=positions,market_value=mv,
        nav_after_reserve=cash+mv+receivable-tax-reserve,entitlements=rights,cashflows=cashflows,
        trades=trades,completed_trades=sum(t['cash_settled'] for t in trades),
        net_pnl_total=sum((t['net_pnl'] for t in trades),ZERO),
        tax_debit_timing='sale_accrual_not_verified_broker_debit'))


def audit_account(fills, events, **kwargs):
    """Independent research audit; it never publishes or installs a forward run."""
    return {**replay_account(fills,events,**kwargs),
            'record_kind':'retrospective_accounting_audit_not_forward','live_installation_allowed':False}
