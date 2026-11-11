"""Scoped event accounting around the unchanged paper matcher, with retry stamps."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path

import polars as pl

from app.services import stockformer_corporate_receipts as receipts
from app.services import stockformer_daily_shadow as daily
from app.services.stockformer_corporate_research import replay_account
from app.strategy import paper

POLICY='verified_cash_transfer_ledger_v1'
_ACTIVE=None
KINDS={'fill','distribution_terms_v1','cash_payment_v1','session_close_v1'}


def strict_rows(path):
    if not path.exists():
        return []
    text=path.read_bytes().decode('utf-8')
    if text and not text.endswith('\n'):
        raise ValueError('Incomplete ledger tail; manual reconciliation required')
    rows=[]
    prefix=b''
    for line in text.splitlines(keepends=True):
        row=json.loads(line)
        if row.get('kind')=='session_close_v1' and row.get('prefix_sha256')!=hashlib.sha256(prefix).hexdigest():
            raise ValueError('Closed ledger prefix changed')
        rows.append(row)
        prefix+=line.encode('utf-8')
    if any(r.get('kind','fill') not in KINDS for r in rows):
        raise ValueError('Unsupported legacy or unverified corporate ledger row')
    for kind,key in [('fill','order_id'),('distribution_terms_v1','symbol'),('cash_payment_v1','event_id'),('session_close_v1','date')]:
        selected=[(r.get(key),r['date']) if kind=='distribution_terms_v1' else r.get(key)
                  for r in rows if r.get('kind','fill')==kind]
        if None in selected or (kind=='distribution_terms_v1' and any(not all(k) for k in selected)) or len(selected)!=len(set(selected)):
            raise ValueError('Duplicate or missing ledger fact identity: '+kind)
    return rows


def _file_sha(path):
    return daily._digest(path) if path.exists() else daily._hash([])


class Ledger:
    def __init__(self,inputs,cfg):
        self.inputs=Path(inputs).resolve()
        self.cfg=cfg
        self.aid=cfg['account_id']
        self.path=paper.accounts_base(self.inputs)/self.aid/'fills.jsonl'
        self.day=None
        self.phase='close'
        self.matching=False
        self.filter_order_read=False
        self.saved={}
        self.busy=False

    def owns(self,root,aid):
        return Path(root).resolve()==self.inputs and aid==self.aid

    def metadata(self):
        report=daily._read(self.inputs/'research_inputs/latest_execution.json')
        folder=Path(report['path'])
        if set(report['files'])!={'raw.parquet','factors.parquet','limits.parquet'} or any(
            daily._digest(folder/n)!=sha for n,sha in report['files'].items()):
            raise ValueError('Execution materialization changed')
        self.raw=pl.read_parquet(folder/'raw.parquet')
        self.factors=pl.read_parquet(folder/'factors.parquet')
        self.limits=pl.read_parquet(folder/'limits.parquet')
        from app.services.stockformer_repaired_shadow import calendar_dates

        self.calendar=sorted(calendar_dates(daily._read(self.inputs/'research_inputs/captures'/f"{report['date']}-calendar.json")))
        self.day=date.fromisoformat(report['date'])

    def rows(self):
        rows=strict_rows(self.path)
        if not self.busy and self.path.exists():
            metrics=daily._read(daily._root(self.inputs)/'execution_metrics.json')
            if metrics.get('ledger_sha256')!=_file_sha(self.path):
                raise ValueError('Ledger head differs from last completed settlement')
        return rows

    def events(self,rows):
        result=[]
        for r in rows:
            if r.get('kind')!='distribution_terms_v1':
                continue
            checked=receipts.reviewed(self.inputs,r['symbol'],date.fromisoformat(r['date']),
                self.calendar,r['factor_versions'])
            if checked!=r:
                raise ValueError('Pinned corporate receipt no longer matches ledger')
            result.append(r['event'])
        return result

    def state(self,*,rows=None,day=None,phase=None,marks=False,planned_payments=False):
        data=self.rows() if rows is None else rows
        if self.day is None:
            self.metadata()
        day=day or self.day
        closes={r['symbol']:{'date':str(day),'close':r['close']} for r in self.raw.filter(pl.col('date')==day).to_dicts()}
        payments=None if planned_payments else {r['event_id']:r['date'] for r in data
            if r.get('kind')=='cash_payment_v1' and r['date']<=str(day)}
        state=replay_account([r for r in data if r.get('kind','fill')=='fill'],self.events(data),
            initial_cash=paper.get_account(self.inputs,self.aid)['initial_cash'],as_of=str(day),
            sessions=[str(d) for d in self.calendar],closes=closes,include_lots=True,
            phase=phase or self.phase,payment_dates=payments,require_marks=marks)
        if not planned_payments:
            expected={}
            for flow in state['cashflows']:
                expected[flow['event_id']]=round(expected.get(flow['event_id'],0)+flow['gross'],2)
            if any(r['gross']!=expected.get(r['event_id']) for r in data
                   if r.get('kind')=='cash_payment_v1' and r['date']<=str(day)
                   and (r['date']<str(day) or (phase or self.phase)=='close')):
                raise ValueError('Cash payment amount disagrees with entitlement')
        for pos in state['positions'].values():
            pos.update(asset_type='stock',qty=pos['qty']+pos['pending_share_qty'])
            pos['avg_cost']=round(pos['cost']/pos['qty'],6) if pos['qty'] else 0.
        return state

    def append(self,root,row,account_id=paper.DEFAULT_ACCOUNT_ID):
        if not self.owns(root,account_id):
            return self.saved['_append_fill'](root,row,account_id)
        with paper.PAPER_LOCK:
            self.rows()  # A damaged tail must never be hidden by a subsequent append.
            if row.get('kind')=='session_close_v1':
                row={**row,'prefix_sha256':hashlib.sha256(self.path.read_bytes() if self.path.exists() else b'').hexdigest()}
            payload=json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a',encoding='utf-8',newline='\n') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())

    def available(self,pos,today=None):
        if pos is None:
            return 0
        return int(sum(lot['qty'] for lot in pos['lots'] if today is None or (
            lot['date']<today and lot.get('available_on',today)<=today and lot.get('credit_on',today)<=today)))

    def positions(self,root,account_id=paper.DEFAULT_ACCOUNT_ID):
        if not self.owns(root,account_id):
            return self.saved['load_positions'](root,account_id)
        if not self.path.exists():
            return {}
        return self.state()['positions']

    def replay(self,root,fills=None,account_id=paper.DEFAULT_ACCOUNT_ID):
        if not self.owns(root,account_id):
            return self.saved['replay_positions'](root,fills,account_id)
        if not self.path.exists() and fills is None:
            return {},paper.get_account(root,account_id)['initial_cash']
        state=self.state(rows=fills)
        return state['positions'],state['cash_available']

    def orders(self,root,account_id=paper.DEFAULT_ACCOUNT_ID):
        values=self.saved['load_orders'](root,account_id)
        if self.owns(root,account_id) and self.filter_order_read:
            self.filter_order_read=False
            return [o for o in values if o.get('last_settlement_session')!=str(self.day)]
        return values

    def save_order(self,root,order,account_id=paper.DEFAULT_ACCOUNT_ID):
        if self.owns(root,account_id) and self.matching:
            order['last_settlement_session']=str(self.day)
        return self.saved['save_order'](root,order,account_id)

    def recover(self):
        rows=self.rows()
        fills={f['order_id']:f for f in rows if f.get('kind','fill')=='fill'}
        orders={o['id']:o for o in self.saved['load_orders'](self.inputs,self.aid)}
        for oid,fill in fills.items():
            order=orders.get(oid)
            if not order or any(order[k]!=fill[k] for k in ('symbol','qty','side')):
                raise ValueError('Order/ledger mismatch during recovery')
            if order['status']=='filled' and any(order[k]!=fill[v] for k,v in
                [('fill_price','price'),('fees','fee'),('filled_at','ts')]):
                raise ValueError('Filled order financial fields disagree with ledger')
            order.update(status='filled',filled_at=fill['ts'],fill_price=fill['price'],fees=fill['fee'],
                         last_settlement_session=fill['date'])
            self.saved['save_order'](self.inputs,order,self.aid)
        if any(o['status']=='filled' and o['id'] not in fills for o in orders.values()):
            raise ValueError('Filled order has no ledger fact')
        paper.rebuild_positions(self.inputs,self.aid)

    def post_cash(self):
        rows=self.rows()
        existing={r['event_id'] for r in rows if r.get('kind')=='cash_payment_v1'}
        planned=self.state(rows=rows,planned_payments=True)
        amounts={}
        for flow in planned['cashflows']:
            if flow['event_id'] not in existing:
                amounts[flow['event_id']]=round(amounts.get(flow['event_id'],0)+flow['gross'],2)
        for eid,gross in amounts.items():
            self.append(self.inputs,dict(kind='cash_payment_v1',date=str(self.day),symbol='',
                event_id=eid,gross=gross,posted_at=daily.cn_now().isoformat(),
                policy='simulated_close_on_or_after_announced_pay_date'),self.aid)

    def nav(self,root,day,price_map=None,account_id=paper.DEFAULT_ACCOUNT_ID):
        if not self.owns(root,account_id):
            return self.saved['daily_nav'](root,day,price_map,account_id)
        self.phase='close'
        self.post_cash()
        paper.rebuild_positions(self.inputs,self.aid)
        state=self.state(marks=True)
        return dict(date=day,cash=state['cash_available'],cash_gross=state['cash_gross'],mv=state['market_value'],
            dividend_receivable=state['dividend_receivable'],tax_accrued=state['dividend_tax_accrued'],
            tax_reserve=state['dividend_tax_reserve'],nav=state['nav_after_reserve'])

    def trips(self,root,account_id=paper.DEFAULT_ACCOUNT_ID):
        if not self.owns(root,account_id):
            return self.saved['round_trips'](root,account_id)
        groups={}
        for trade in self.state()['trades']:
            key=(trade['origin_lot_id'],trade['sell_fill_index'])
            row=groups.setdefault(key,dict(symbol=trade['symbol'],open_date=trade['open_date'],
                close_date=trade['close_date'],qty=0,pnl=Decimal(0),cost=Decimal(0),cash_settled=True))
            row['qty']+=trade['qty']
            row['pnl']+=Decimal(trade['net_pnl_exact'])
            row['cost']+=Decimal(trade['cost_basis_exact'])
            row['cash_settled'] &= trade['cash_settled']
        return [{**r,'cost':float(r['cost']),'pnl':round(float(r['pnl']),2),
                 'pnl_pct':round(float(r['pnl']/r['cost']*100),2) if r['cost'] else 0.,
                 'holding_days':paper._days_between(r['open_date'],r['close_date'])}
                for r in groups.values() if r['cash_settled']]

    def publish(self):
        rounds=self.trips(self.inputs,self.aid) if self.path.exists() else []
        nav=paper.load_nav(self.inputs,self.aid)
        navpath=paper.accounts_base(self.inputs)/self.aid/'nav/daily.jsonl'
        daily._write(daily._root(self.inputs)/'execution_metrics.json',dict(account_id=self.aid,
            protocol_id=self.cfg['protocol_id'],as_of=str(self.day),ledger_sha256=_file_sha(self.path),
            nav_sha256=_file_sha(navpath),completed_trades=len(rounds),
            cost_after_return_pct=(nav[-1]['nav']/self.cfg['protocol']['execution']['initial_cash']-1)*100 if rounds and nav else None,
            net_realized_pnl=round(sum(r['pnl'] for r in rounds),2),
            tax_policy='sale_accrual_and_max_20_percent_open_reserve_not_broker_debit'))

    def prepare_terms(self):
        previous=max(d for d in self.calendar if d<self.day)
        before=self.state(day=previous,phase='close')['positions'] if self.path.exists() else {}
        points=self.factors.sort('symbol','date').with_columns(
            pl.col('adj_factor').shift(1).over('symbol').alias('_previous'),
            pl.col('date').shift(1).over('symbol').alias('_previous_date'))
        points={r['symbol']:r for r in points.filter(pl.col('date')==self.day).to_dicts()}
        rows=self.rows()
        pending=[]
        for symbol,pos in before.items():
            if pos['qty']<=0:
                continue
            point=points.get(symbol)
            if point is None or point['_previous'] is None:
                raise ValueError('Held stock factor history missing: '+symbol)
            capture,_=receipts.capture(self.inputs,symbol,self.day)
            implemented=[r for r in capture['records'] if r.get('ex_date')==str(self.day).replace('-','')]
            if point['adj_factor']!=point['_previous'] or implemented:
                if point['_previous_date']!=previous or not any(r.get('kind')=='session_close_v1' and r['date']==str(previous) for r in rows):
                    raise ValueError('Corporate entitlement lacks prior closed-session proof')
                fact=receipts.reviewed(self.inputs,symbol,self.day,self.calendar,point,
                                       source_records=capture['records'])
                old=next((r for r in rows if r.get('kind')=='distribution_terms_v1' and r['symbol']==symbol and r['date']==str(self.day)),None)
                if old and old!=fact:
                    raise ValueError('Applied corporate fact changed')
                if not old:
                    pending.append(fact)
        # Validate the entire prospective account before appending any event.
        self.state(rows=rows+pending,marks=True,phase='open')
        return pending

    def settle(self,day):
        self.metadata()
        self.day=day
        self.phase='open'
        path=daily._root(self.inputs)/'settlements'/f'{day}.json'
        journal=daily._read(path)
        signature=daily._hash([f.filter(pl.col('date')==day).sort('symbol').to_dicts()
                              for f in [self.raw,self.factors,self.limits]])
        if journal and (journal.get('ledger_policy')!=POLICY or journal.get('input_sha256')!=signature):
            raise ValueError('Settlement inputs/protocol changed; reconcile existing fills')
        if journal.get('status')=='settled':
            if not any(r.get('kind')=='session_close_v1' and r['date']==str(day)
                       and r['input_sha256']==signature for r in self.rows()):
                raise ValueError('Settled journal lacks its closed ledger proof')
            self.phase='close'
            self.recover()
            self.publish()
            return journal
        pending=self.prepare_terms()
        daily._write(path,dict(status='settling',date=str(day),ledger_policy=POLICY,input_sha256=signature))
        for fact in pending:
            self.append(self.inputs,fact,self.aid)
        self.recover()
        self.matching=True
        self.filter_order_read=True
        try:
            result=paper.settle_day(self.inputs,str(day),self.aid)
        finally:
            self.matching=False
            self.filter_order_read=False
        result['filled']=sum(r.get('kind','fill')=='fill' and r['date']==str(day) for r in self.rows())
        result['expired']=sum(o['status']=='expired' and o.get('last_settlement_session')==str(day)
                              for o in self.saved['load_orders'](self.inputs,self.aid))
        result['corp_actions']=sum(r.get('kind')=='distribution_terms_v1' and r['date']==str(day) for r in self.rows())
        if not any(r.get('kind')=='session_close_v1' and r['date']==str(day) for r in self.rows()):
            self.append(self.inputs,dict(kind='session_close_v1',date=str(day),symbol='',
                input_sha256=signature,nav=result['nav']),self.aid)
        self.publish()
        result.update(status='settled',date=str(day),ledger_policy=POLICY,input_sha256=signature)
        daily._write(path,result)
        return result


@contextmanager
def environment(inputs,cfg):
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError('Corporate ledger already installed')
    ctx=Ledger(inputs,cfg)
    replacements={'_append_fill':ctx.append,'load_fills':lambda root,account_id=paper.DEFAULT_ACCOUNT_ID:
        ctx.rows() if ctx.owns(root,account_id) else ctx.saved['load_fills'](root,account_id),
        'replay_positions':ctx.replay,'load_positions':ctx.positions,'_available_of':ctx.available,
        'load_orders':ctx.orders,'save_order':ctx.save_order,'daily_nav':ctx.nav,'round_trips':ctx.trips}
    with paper.PAPER_LOCK:
        originals={name:getattr(paper,name) for name in replacements}
        ctx.saved=originals.copy()
        for name,value in replacements.items():
            setattr(paper,name,value)
        _ACTIVE=ctx
    try:
        yield ctx
    finally:
        with paper.PAPER_LOCK:
            for name,value in originals.items():
                setattr(paper,name,value)
            _ACTIVE=None


def settle(inputs,cfg,day):
    if _ACTIVE is None or not _ACTIVE.owns(inputs,cfg['account_id']):
        raise ValueError('Corporate accounting requires its scoped worker context')
    _ACTIVE.busy=True
    try:
        return _ACTIVE.settle(day)
    finally:
        _ACTIVE.busy=False
