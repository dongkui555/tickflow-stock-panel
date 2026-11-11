"""Explicit research-source captures and issuer-reviewed corporate receipts."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

from app.market_time import cn_now
from app.services import stockformer_daily_shadow as daily
from app.services.stockformer_corporate_research import _event

FIELDS='ts_code,div_proc,stk_div,stk_bo_rate,stk_co_rate,cash_div_tax,record_date,ex_date,pay_date,div_listdate,imp_ann_date'


def capture(inputs, symbol, day):
    """Tushare-specific research evidence, through the configured provider route."""
    from app.services.stockformer_pair_shadow import WaitingDataError

    path=Path(inputs)/'research_inputs/corporate_sources'/f'{day}-{symbol}.json'
    if path.exists():
        item=daily._read(path)
        stamp=datetime.fromisoformat(item['captured_at'])
        if (item.get('schema_version')==1 and item['date']==str(day) and item['symbol']==symbol
            and item.get('complete') and stamp.tzinfo is not None
            and stamp>=datetime.combine(day,datetime.min.time(),tzinfo=cn_now().tzinfo).replace(hour=21,minute=5)
            and stamp<=cn_now() and item.get('ts_code') in {symbol+'.SH',symbol+'.SZ',symbol+'.BJ'}
            and isinstance(item.get('records'),list) and len(item['records'])<2000
            and all(r.get('ts_code')==item['ts_code'] for r in item['records'])):
            return item,path
        raise ValueError('Invalid corporate source cache')
    failure=daily._read(path.with_suffix('.failure.json'))
    if failure.get('retry_after','')>cn_now().isoformat():
        raise WaitingDataError('公司行动引用冷却：'+symbol)
    # The upstream contract updates 20:00-21:00; an early empty response is
    # insufficient evidence of no action. No held position means no request.
    if cn_now().strftime('%H:%M')<'21:05' and cn_now().date()==day:
        raise WaitingDataError('持仓公司行动引用等待21:05后更新：'+symbol)
    names=daily._read(Path(inputs)/'research_inputs/captures'/f'{day}-bak_basic.json')
    codes={r['ts_code'] for r in names.get('records',[]) if str(r.get('ts_code','')).split('.')[0]==symbol}
    if len(codes)!=1:
        raise WaitingDataError('公司行动缺少明确市场身份：'+symbol)
    code=codes.pop()
    try:
        import httpx

        from app.plugins.tushare.provider import TushareProvider

        api=TushareProvider()._ensure_api()
        try:
            for client in [getattr(api,'primary',None),getattr(api,'backup',None)]:
                if client is not None and hasattr(client,'_client'):
                    client._client.timeout=httpx.Timeout(25)
            frame=api.dividend(ts_code=code,fields=FIELDS,limit=2000,offset=0)
            if frame is None or len(frame)>=2000:
                raise ValueError('Missing or truncated corporate history')
            rows=json.loads(frame.to_json(orient='records'))
            if any(r.get('ts_code')!=code for r in rows):
                raise ValueError('Corporate identity mismatch')
            item=dict(schema_version=1,symbol=symbol,ts_code=code,date=str(day),captured_at=cn_now().isoformat(),
                      source=getattr(api,'last_source',None),complete=True,records=rows)
            daily._write(path,item)
            return item,path
        finally:
            api.close()
    except Exception as exc:  # noqa: BLE001 -- never expose credential-bearing source errors.
        daily._write(path.with_suffix('.failure.json'),dict(error_type=type(exc).__name__,
            retry_after=(cn_now()+timedelta(minutes=15)).isoformat()))
        raise WaitingDataError('公司行动源未就绪：'+symbol+' '+type(exc).__name__) from None


def reviewed(inputs, symbol, day, sessions, point, *, source_records=None):
    """Validate the receipt and its pinned native/issuer files; never trust a boolean alone."""
    from app.services.stockformer_pair_shadow import WaitingDataError

    folder=Path(inputs)/'research_inputs/corporate_receipts'/f'{symbol}-{day}'
    path=folder/'receipt.json'
    if not path.exists():
        raise WaitingDataError('公司行动需公告及分配口径核验：'+symbol+' '+str(day))
    receipt=daily._read(path)
    if receipt.get('schema_version')!=1:
        raise ValueError('Unsupported corporate receipt')
    refs=receipt.get('evidence',[])
    if sorted(r['role'] for r in refs)!=['issuer','native']:
        raise ValueError('Receipt requires issuer and native evidence')
    data={}
    for ref in refs:
        target=(folder/ref['file']).resolve()
        if not target.is_relative_to(folder.resolve()) or daily._digest(target)!=ref['sha256']:
            raise ValueError('Corporate evidence changed or escaped scope')
        if ref['role']=='issuer':
            parsed=urlparse(ref['url'])
            hosts={'static.cninfo.com.cn','disc.static.szse.cn','www.sse.com.cn','static.sse.com.cn',
                   'www.bse.cn','paper.cnstock.com','epaper.stcn.com'}
            if parsed.scheme!='https' or parsed.hostname not in hosts:
                raise ValueError('Issuer notice must use a primary disclosure source')
        else:
            data=daily._read(target)
    event={**receipt['event'],'evidence_verified':True}
    if event.get('symbol')!=symbol or event.get('ex_date')!=str(day) or event.get('exchange') not in {'SH','SZ'}:
        raise ValueError('Unsupported receipt identity or tax jurisdiction')
    if event.get('cash_rounding')!='exact_fen_only':
        raise ValueError('Live policy requires exact-cent cash entitlement')
    _event(event,[datetime.fromisoformat(str(d)).date() for d in sessions])
    for field,current in [('factor_before',point['_previous']),('factor_after',point['adj_factor'])]:
        if Decimal(str(receipt[field]))!=Decimal(str(current)):
            raise ValueError('Corporate receipt factor version mismatch')
    rows=[r for r in data.get('records',[]) if r.get('ts_code')==symbol+'.'+event['exchange']
          and r.get('ex_date')==str(day).replace('-','') and r.get('div_proc')=='实施']
    if len(rows)!=1:
        raise ValueError('Corporate event must match one implemented native record')
    row=rows[0]
    if source_records is not None:
        current=[r for r in source_records if r.get('ts_code')==row['ts_code']
                 and r.get('ex_date')==row['ex_date'] and r.get('div_proc')=='实施']
        if len(current)!=1 or any(current[0].get(k)!=row.get(k) for k in FIELDS.split(',')):
            raise ValueError('Current corporate source disagrees with the reviewed receipt')
    for native,field in [('record_date','record_date'),('div_listdate','share_listing_date'),('pay_date','pay_date')]:
        expected=event.get(field)
        if expected and row.get(native)!=expected.replace('-',''):
            raise ValueError('Corporate event date disagrees with source')
    cash=Decimal(str(event['cash_per_eligible_share']))
    transfer=Decimal(str(event['transfer_per_eligible_share']))
    if cash and Decimal(str(row.get('cash_div_tax')))!=cash:
        raise ValueError('Corporate cash amount disagrees with native source')
    if Decimal(str(row.get('stk_div')))!=transfer or (
        transfer and Decimal(str(row.get('stk_co_rate')))!=transfer):
        raise ValueError('Corporate shares disagree with native source')
    if row.get('stk_bo_rate') is not None and Decimal(str(row['stk_bo_rate']))!=0:
        raise ValueError('Unsupported taxable bonus in native source')
    if not cash and row.get('cash_div_tax') is not None and Decimal(str(row['cash_div_tax']))!=0:
        raise ValueError('Native source contains an unaccounted cash distribution')
    if not cash and not receipt.get('review',{}).get('issuer_explicit_no_cash'):
        raise ValueError('Null native cash is not proof of zero cash')
    if transfer and not receipt.get('review',{}).get('issuer_premium_capital_tax_exempt'):
        raise ValueError('Transfer tax exemption was not verified')
    eid=daily._hash({k:v for k,v in event.items() if k not in {'evidence_verified','event_id'}})
    event['event_id']=eid
    return dict(kind='distribution_terms_v1',date=str(day),symbol=symbol,event=event,
        factor_versions={'_previous':point['_previous'],'adj_factor':point['adj_factor']},
        receipt=str(path.relative_to(inputs)),receipt_sha256=daily._digest(path),
        evidence_sha256={r['file']:r['sha256'] for r in refs})
