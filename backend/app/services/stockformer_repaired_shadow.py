"""Daily adapter for the repaired research pair; never fits or registers models."""
from __future__ import annotations

import importlib
import time
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import polars as pl

from app.market_time import cn_now
from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_pit_inputs as pit
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_repaired_inputs as matched

ENDPOINTS = {'bak_basic': ['name', 'industry'], 'daily_basic': ['turnover_rate', 'float_share'],
             'stk_limit': ['up_limit', 'down_limit'], 'adj_factor': ['adj_factor']}
WARNINGS = ['独立研究账户；旧窗口不作为新增前向证据；零成交不算盈利。',
            '历史源版本尚不能完全证明；争议换手率上下文继续隔离。',
            '已核实的纯现金与免税转增按独立账本核算；未核实、混合分派与零碎股继续阻断。']


def verify_bundle(frozen):
    manifest = pair.verify_manifest(frozen/'latest-pair')
    parent = daily._read(frozen/'protocol.json')
    context = manifest['provenance']['context']
    if (parent.get('version') != 'repaired_inputs_fixed_decision_research_v2'
            or pair.digest(frozen/'protocol.json') != context.get('research_protocol_sha256')
            or not context.get('feature_input_contract', {}).get('matching_live_input_builder_required')):
        raise ValueError('Repaired pair training protocol mismatch')
    for name in ['app.services.stockformer', 'app.services.stockformer_calibration_research',
                 'app.services.stockformer_training', 'app.services.stockformer_frozen_pair',
                 'app.services.stockformer_portfolio', 'app.services.stockformer_decision_research',
                 'app.strategy.paper']:
        if pair.digest(importlib.import_module(name).__file__) != parent['source_sha256'][name]:
            raise ValueError('Repaired pair inference/rule source changed: '+name)
    if context['signal_policy'] != parent['signal_policy'] or context['execution'] != daily._EXECUTION:
        raise ValueError('Repaired pair rules or costs changed')
    seed = daily._read(frozen/'seed.json')
    for item in seed['files'].values():
        if pair.digest(item['path']) != item['sha256']:
            raise ValueError('Validated research seed changed')
    universe = daily._read(frozen/'universe.json')['symbols']
    if universe != sorted(set(universe)) or not universe:
        raise ValueError('Invalid frozen research universe')
    return manifest


def assets(frozen):
    modules = [importlib.import_module(name) for name in [
        'app.services.stockformer_repaired_shadow', 'app.services.stockformer_repaired_inputs',
        'app.services.stockformer_research_price_basis', 'app.services.stockformer_research_turnover',
        'app.services.stockformer_execution_quantities',
        'app.services.stockformer_corporate_research','app.services.stockformer_corporate_receipts',
        'app.services.stockformer_corporate_ledger',
        'app.services.stockformer_pit_inputs', 'app.services.stockformer_tree_validation']]
    return {'files': {name: pair.digest(frozen/name) for name in ['seed.json', 'universe.json']},
            'code': {m.__name__: pair.digest(m.__file__) for m in modules}}


def _root(inputs):
    return inputs/'research_inputs'


def input_versions(inputs, day):
    return [(str(p.relative_to(_root(inputs))), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted((_root(inputs)/'captures').glob('*.json'))
            if p.name[:10] <= str(day)]


def validate_capture(item,endpoint,day,required_symbols):
    frame=matched.dated_frame(item,endpoint,day,ENDPOINTS[endpoint])
    if required_symbols is None:
        return
    selected=frame.filter(pl.col('symbol').is_in(sorted(required_symbols)))
    if set(selected['symbol']) != set(required_symbols):
        raise ValueError('Partial dated source coverage')
    numeric=[c for c in ENDPOINTS[endpoint] if c!='float_share']
    if endpoint=='bak_basic':
        valid=pl.all_horizontal([pl.col(c).cast(pl.String).is_not_null()
            & (pl.col(c).cast(pl.String).str.strip_chars()!='') for c in numeric])
    else:
        valid=pl.all_horizontal([pl.col(c).cast(pl.Float64,strict=False).is_finite()
            & (pl.col(c).cast(pl.Float64,strict=False) >= (0. if c=='turnover_rate' else 1e-12)) for c in numeric])
    if selected.filter(~valid.fill_null(False)).height:
        raise ValueError('Incomplete dated source values')


def _reject(path):
    archive=path.parent/'rejected'/(path.stem+'-'+pair.digest(path)[:16]+'.json')
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        path.rename(archive)
    else:
        # Keep both observations and never overwrite the archived response.
        path.rename(archive.with_name(archive.stem+'-'+uuid4().hex[:8]+'.json'))


def capture(inputs, endpoint, day, *, required_symbols=None):
    """One bounded request per missing endpoint and round; failures cool down."""
    from app.plugins.tushare.provider import TushareProvider
    from app.services.stockformer_pair_shadow import WaitingDataError

    path = _root(inputs)/'captures'/f'{day}-{endpoint}.json'
    if path.exists():
        item = daily._read(path)
        try:
            validate_capture(item,endpoint,day,required_symbols)
            return item
        except ValueError:
            _reject(path)
    failure = daily._read(path.with_suffix('.failure.json'))
    if failure.get('retry_after', '') > cn_now().isoformat():
        raise WaitingDataError('研究引用源冷却：'+str(day)+' '+endpoint)
    try:
        import httpx

        api = TushareProvider()._ensure_api()
        for client in [getattr(api, 'primary', None), getattr(api, 'backup', None)]:
            if client is not None and hasattr(client, '_client'):
                client._client.timeout = httpx.Timeout(25)
        frame = getattr(api, endpoint)(trade_date=str(day).replace('-', ''),
            fields=','.join(['ts_code', 'trade_date', *ENDPOINTS[endpoint]]), limit=6000, offset=0)
        if frame is None or not 0 < len(frame) < 6000:
            raise ValueError('Empty or possibly truncated dated response')
        item = {'date': str(day), 'endpoint': endpoint, 'captured_at': cn_now().isoformat(),
                'source': getattr(api, 'last_source', None) or 'configured_primary',
                'records': frame.to_dict(orient='records')}
        try:
            validate_capture(item,endpoint,day,required_symbols)
        except ValueError:
            daily._write(path,item)
            _reject(path)
            raise
        daily._write(path, item)
        time.sleep(3)
        return item
    except Exception as exc:  # noqa: BLE001 - sanitize provider failures without logging credentials.
        response = getattr(exc, 'response', None)
        daily._write(path.with_suffix('.failure.json'), {'error_type': type(exc).__name__,
            'http_status': getattr(response, 'status_code', None),
            'retry_after': (cn_now()+timedelta(minutes=15)).isoformat()})
        raise WaitingDataError('研究引用源未就绪：'+str(day)+' '+endpoint+' '+type(exc).__name__) from None


def _request_calendar(start,end):
    from app.plugins.tushare.provider import TushareProvider

    provider=TushareProvider()
    rows=provider.get_trade_calendar(str(start).replace('-',''),str(end).replace('-',''))
    return {'records':rows,'source':getattr(provider._api,'last_source',None) or 'configured_provider',
            'captured_at':cn_now().isoformat()}


def _validate_calendar(item,day,history_start):
    opens=calendar_dates(item)
    future=sorted(d for d in opens if d>day)
    if not future:
        raise ValueError('No next actual trading session')
    present={str(r.get('trade_date') or r.get('cal_date') or r.get('date')).replace('-','') for r in item['records']}
    if history_start is not None and any(str(history_start+timedelta(days=i)).replace('-','') not in present
        for i in range((day-history_start).days+1)):
        raise ValueError('Incomplete historical exchange calendar')
    return str(future[0])+'T09:30:00+08:00'


def next_open(inputs, day, *, history_start=None):
    from app.services.stockformer_pair_shadow import WaitingDataError

    path = _root(inputs)/'captures'/f'{day}-calendar.json'
    item = daily._read(path)
    if item:
        try:
            return _validate_calendar(item,day,history_start)
        except ValueError:
            _reject(path)
    failure=daily._read(path.with_suffix('.failure.json'))
    if failure.get('retry_after','')>cn_now().isoformat():
        raise WaitingDataError('交易日历未就绪，等待冷却重试')
    try:
        item={**_request_calendar(history_start or day,day+timedelta(days=32)),'date':str(day)}
        opening=_validate_calendar(item,day,history_start)
        daily._write(path,item)
        return opening
    except Exception as exc:  # noqa: BLE001 - preserve failure; never cache an empty calendar as success.
        if item:
            daily._write(path,item)
            _reject(path)
        daily._write(path.with_suffix('.failure.json'),{'error_type':type(exc).__name__,
            'retry_after':(cn_now()+timedelta(minutes=15)).isoformat()})
        raise WaitingDataError('交易日历无效或缺少下一实际交易日，等待重试') from None


def calendar_dates(item):
    result = []
    seen=set()
    for row in item['records']:
        if str(row.get('is_open')) not in {'0','1'}:
            raise ValueError('Invalid exchange calendar state')
        text = str(row.get('trade_date') or row.get('cal_date') or row.get('date')).replace('-','')
        if len(text) != 8 or not text.isdigit():
            raise ValueError('Invalid exchange calendar date')
        value = date.fromisoformat(f'{text[:4]}-{text[4:6]}-{text[6:]}')
        if value in seen:
            raise ValueError('Duplicate calendar date')
        seen.add(value)
        if str(row['is_open']) == '1':
            result.append(value)
    if len(result) != len(set(result)):
        raise ValueError('Duplicate open calendar date')
    return result


def prepare(inputs, cfg, day):
    """Immutable daily materializations; same loader is used for parity checks."""
    from app.services.stockformer_pair_shadow import WaitingDataError, _input_version

    frozen = Path(cfg['protocol']['frozen_root'])
    seed = daily._read(frozen/'seed.json')
    files = {k: Path(v['path']) for k,v in seed['files'].items()}
    universe = cfg['protocol']['training_universe']
    seed_day = date.fromisoformat(seed['through'])
    events = pl.DataFrame(daily._read(files['events']))
    if 'supplemental_events' in files:
        events = pl.concat([events,pl.DataFrame(daily._read(files['supplemental_events']))],how='diagonal_relaxed')
    opening = next_open(inputs,day,history_start=seed_day)
    calendar = daily._read(_root(inputs)/'captures'/f'{day}-calendar.json')
    dates = sorted(d for d in calendar_dates(calendar) if seed_day < d <= day)
    raw_additions = [matched.raw_session(inputs, d, universe) for d in dates]
    excluded=pit.attach_lifecycle_exclusions(pl.concat([r.select('symbol','date') for r in raw_additions]),events) if raw_additions else None
    sources=[]
    for d in dates:
        required=set(excluded.filter((pl.col('date')==d)&~pl.col('_pit_lifecycle_excluded'))['symbol'])
        sources.append((d,{e:capture(inputs,e,d,required_symbols=required) for e in ENDPOINTS}))
    version = _input_version(inputs, day)
    saved = daily._read(_root(inputs)/'latest.json')
    if saved.get('date') == str(day) and saved.get('input_version') == version:
        if all(pair.digest(Path(saved['path'])/name) == digest for name,digest in saved['files'].items()):
            return saved
        raise ValueError('Saved matched input materialization changed')
    panel = pl.read_parquet(files['panel']).filter(pl.col('date') <= day)
    raw = pl.read_parquet(files['raw']).filter(pl.col('date') <= day)
    factors = pit._keys(pl.read_parquet(files['factors']), 'seed factors').filter(pl.col('date') <= day)
    names, industries, limits = [pl.read_parquet(files[k]) for k in ['names','industries','limits']]
    auxiliary = [matched.auxiliary_session(inputs, r) for r in raw_additions]
    try:
        panel, raw, factors, limits = matched.extend_panel(panel, raw, factors, names, industries,
            limits, events, auxiliary, raw_additions, sources)
        known=pl.all_horizontal([pl.col('_pit_'+k+'_known') for k in ['name','industry','limits']])
        if panel.filter(~known.fill_null(False)&~pl.col('_pit_lifecycle_excluded')).height:
            raise ValueError('Incomplete execution reference context')
        execution_folder=_root(inputs)/'execution-materializations'/f'{day}-{uuid4().hex[:12]}'
        execution_folder.mkdir(parents=True,exist_ok=False)
        for name,frame in [('raw',raw),('factors',factors),('limits',limits)]:
            frame.write_parquet(execution_folder/(name+'.parquet'))
        if version != _input_version(inputs,day):
            raise WaitingDataError('成交输入计算期间更新，等待重试')
        daily._write(_root(inputs)/'latest_execution.json',{'date':str(day),'path':str(execution_folder),
            'input_version':version,'files':{p.name:pair.digest(p) for p in execution_folder.glob('*.parquet')}})
        features = matched.matched_features(panel, raw, factors, pl.read_parquet(files['disputed']))
    except (ValueError, AssertionError) as exc:
        raise WaitingDataError('研究输入不完整：'+str(exc)) from None
    today = features.filter(pl.col('date') == day)
    selected, coverage = matched.inference_pool(features, day, cfg['protocol']['feature_columns'])
    active = set(raw.filter(pl.col('date') == day)['symbol'])
    nonst = set(panel.filter((pl.col('date') == day) & ~pl.col('_pit_is_st')
                 & ~pl.col('_pit_lifecycle_excluded'))['symbol'])
    if not active or coverage['complete_rows'] < len(nonst)*cfg['protocol']['minimum_active_coverage']:
        raise WaitingDataError(f'完整研究特征覆盖不足：{coverage["complete_rows"]}/{len(nonst)}')
    folder = _root(inputs)/'materializations'/f'{day}-{uuid4().hex[:12]}'
    folder.mkdir(parents=True,exist_ok=False)
    for name, frame in [('features',today),('selected',selected),('raw',raw),('factors',factors),('limits',limits)]:
        frame.write_parquet(folder/(name+'.parquet'))
    report = {'date': str(day),'path':str(folder),'input_version': version,
        'files':{p.name:pair.digest(p) for p in folder.glob('*.parquet')},
        'coverage':{**coverage,'active_universe':len(active),'nonst_active':len(nonst),
                    'pit_excluded':sorted(active-nonst)},
        'auxiliary_coverage': today.select(pl.col('moneyflow_data_available').sum().alias('moneyflow'),
            pl.col('auction_data_available').sum().alias('auction'),
            pl.col('news_data_available').sum().alias('news')).to_dicts(),
        'version':matched.VERSION,'created_at':cn_now().isoformat()}
    report['next_actual_session_open'] = opening
    report['source_captures']={str(p.relative_to(_root(inputs))):pair.digest(p)
        for p in (_root(inputs)/'captures').glob('*.json') if not p.name.endswith('.failure.json') and p.name[:10]<=str(day)}
    if version != _input_version(inputs,day):
        raise WaitingDataError('研究输入在计算期间更新；丢弃此次结果并重试')
    daily._write(folder/'report.json',report)
    daily._write(_root(inputs)/'latest.json',report)
    return report


def compute(inputs, cfg, day, manifest):
    report = prepare(inputs,cfg,day)
    folder = Path(report['path'])
    features = pl.read_parquet(folder/'features.parquet')
    selected = pl.read_parquet(folder/'selected.parquet')
    prediction, _ = pair.predict_pair(Path(cfg['protocol']['frozen_root'])/'latest-pair',selected)
    policy = cfg['protocol']['signal_policy']
    signal = portfolio._build_gated_signal_frame(inputs,prediction,selected,
        {'target_mode':'industry_relative_net','signal_weights':{'5d':1.}},horizon=5,
        signal_policy=policy,min_net_edge=policy['prediction_rule']['min_net_edge'],
        score_source='raw',historical_identity=True)
    return prediction,signal,features,{**report['coverage'],'matched_inputs':report}


def settle(inputs,cfg,day):
    """Validate held exposure before any fills, including sells on an ex date."""
    from app.services import stockformer_tree_validation as validation
    from app.strategy import paper

    report = daily._read(_root(inputs)/'latest_execution.json') or daily._read(_root(inputs)/'latest.json')
    folder = Path(report['path'])
    raw = pl.read_parquet(folder/'raw.parquet')
    factors = pl.read_parquet(folder/'factors.parquet')
    limits = pl.read_parquet(folder/'limits.parquet')
    history = factors.sort('symbol','date').with_columns(
        pl.col('adj_factor').shift(1).over('symbol').alias('_previous'))
    current = {r['symbol']:r for r in history.filter(pl.col('date')==day).to_dicts()}
    new_ledger=cfg.get('protocol',{}).get('corporate_action_policy')=='verified_cash_transfer_ledger_v1'
    for symbol,pos in ([] if new_ledger else paper.load_positions(inputs,cfg['account_id']).items()):
        if pos['qty'] <= 0:
            continue
        point = current.get(symbol)
        if point is None or point['_previous'] is None or point['adj_factor'] != point['_previous']:
            raise ValueError('持仓公司行动或因子缺口需核验真实股份与现金：'+symbol+' '+str(day))
    refs = pit.execution_limit_map(raw,limits,[day])
    real_now, real_shadow_now = paper.cn_now,daily.cn_now
    with validation._paper_environment(raw,None,refs,factors.rename({'date':'trade_date'})) as (clock,_):
        clock['day'] = day
        paper.cn_now,daily.cn_now = real_now,real_shadow_now
        # All prior holdings were checked before matching. Shares bought on an
        # ex date are not entitled to the previous record-date distribution.
        paper._factor_on = lambda *a: None
        if new_ledger:
            from app.services.stockformer_corporate_ledger import settle as settle_ledger
            return settle_ledger(inputs,cfg,day)
        return daily._settle(inputs,cfg,day)


def execution_ready(inputs,day):
    from app.services.stockformer_pair_shadow import _input_version

    saved=daily._read(_root(inputs)/'latest_execution.json')
    return (saved.get('date')==str(day) and saved.get('input_version')==_input_version(inputs,day)
            and set(saved.get('files',{}))=={'raw.parquet','factors.parquet','limits.parquet'}
            and all(pair.digest(Path(saved['path'])/name)==digest for name,digest in saved['files'].items()))
