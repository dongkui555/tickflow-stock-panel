"""Daily observations of an immutable tree/calibration pair in an isolated worker.

The worker has a read-only market-data mount and one writable research scope.
Its view links those inputs to a separate StockFormer and paper namespace, so
the existing paper execution functions can be reused without changing the old
shadow service (whose source is itself frozen).
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from contextlib import ExitStack
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from app.market_time import CN_TZ, cn_now
from app.services import stockformer as sf
from app.services import stockformer_daily_shadow as daily
from app.services import stockformer_portfolio as portfolio
from app.services.fs_utils import atomic_write_parquet
from app.services.prediction_workflow import readiness
from app.services.stockformer_paper import _daily_observations
from app.strategy import paper

INPUT_TABLES = ('kline_daily', 'kline_daily_enriched', 'kline_minute', 'moneyflow',
                'auction_open', 'kline_index_daily', 'regime_history', 'instruments',
                'instruments_ext', 'news', 'adj_factor', 'ext_data', 'user_data')
_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


class WaitingDataError(ValueError):
    """Data are not ready; no successful zero-return record is published."""


def scope(data_dir: Path, *, variant: str = 'frozen') -> Path:
    if variant not in {'frozen', 'decision', 'repaired'}:
        raise ValueError('Unknown shadow variant')
    name = {'frozen': 'frozen_pair_shadow', 'decision': 'decision_pair_shadow',
            'repaired': 'repaired_pair_shadow'}[variant]
    return Path(data_dir) / 'stockformer' / name


def view(data_dir: Path, *, variant: str = 'frozen') -> Path:
    return scope(data_dir, variant=variant) / 'view'


def _verify_pair(frozen: Path) -> dict:
    if daily._read(frozen/'protocol.json').get('version') == 'repaired_inputs_fixed_decision_research_v2':
        from app.services.stockformer_repaired_shadow import verify_bundle
        return verify_bundle(frozen)
    if daily._read(frozen/'protocol.json').get('version') == 'raw_rank_matching_pool_fixed_v2':
        from app.services.stockformer_decision_shadow import verify_bundle
        return verify_bundle(frozen)
    from app.services import stockformer_frozen_pair as pair
    manifest = pair.verify_manifest(frozen / 'latest-pair')
    context = manifest['provenance']['context']
    if pair.digest(frozen / 'protocol.json') != context['protocol_sha256']:
        raise ValueError('Frozen parent protocol mismatch')
    from app.services import (
        stockformer_calibration_research,
        stockformer_training,
        stockformer_tree_validation,
    )
    modules = (sf, stockformer_training, stockformer_calibration_research, pair,
               stockformer_tree_validation, portfolio)
    parent = daily._read(frozen / 'protocol.json')
    if parent['source_sha256'] != {m.__name__: pair.digest(m.__file__) for m in modules}:
        raise ValueError('Frozen inference/gate source mismatch')
    return manifest


def _assets(frozen: Path) -> dict:
    from app.services import market_gate, minute_features, prediction_workflow, regime_builder
    from app.services import stockformer_frozen_pair as pair
    manifest = _verify_pair(frozen)
    modules = (daily, paper, portfolio, market_gate, regime_builder, minute_features, prediction_workflow)
    files = ['protocol.json', *['latest-pair/' + n for n in (*pair.FILES, 'pair.json')]]
    if manifest['provenance']['context'].get('fixed_arm') == 'decision_raw':
        from app.services import stockformer_decision_research, stockformer_decision_shadow
        modules += (stockformer_decision_research, stockformer_decision_shadow)
        files.append('universe.json')
    if manifest['provenance']['context'].get('feature_input_contract', {}).get('matching_live_input_builder_required'):
        from app.services.stockformer_repaired_shadow import assets
        extra = assets(frozen)
        files.extend(extra['files'])
    else:
        extra = {'code': {}}
    return {'files': {name: daily._digest(frozen / name) for name in files},
            'code': {**{m.__name__: daily._digest(Path(m.__file__)) for m in modules}, **extra['code']},
            'observer': daily._digest(Path(__file__))}


def configure(data_dir: Path, frozen: Path, *, start_date: date, variant: str = 'frozen') -> dict:
    """Install one new run; neither the old config nor a published day is migrated."""
    target = daily._root(view(data_dir, variant=variant)) / 'config.json'
    if target.exists():
        raise ValueError('frozen pair shadow already configured')
    manifest = _verify_pair(frozen)
    if start_date < date.fromisoformat(manifest['provenance']['frozen_as_of']):
        raise ValueError('Start precedes frozen pair')
    context = manifest['provenance']['context']
    is_repaired = context.get('feature_input_contract', {}).get('matching_live_input_builder_required', False)
    is_decision = context.get('fixed_arm') == 'decision_raw'
    if is_decision != (variant == 'decision') or is_repaired != (variant == 'repaired'):
        raise ValueError('Frozen pair does not match shadow variant')
    if is_decision and start_date < date.fromisoformat(context['signal_policy']['effective_from']):
        raise ValueError('Start precedes effective signal policy date')
    if context['execution'] != daily._EXECUTION:
        raise ValueError('Frozen costs/holding rules differ from paper execution')
    inputs = view(data_dir, variant=variant)
    inputs.mkdir(parents=True, exist_ok=True)
    for table in INPUT_TABLES:
        link = inputs / table
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != (Path(data_dir) / table).resolve():
                raise ValueError('Unexpected research input view: ' + table)
        else:
            link.symlink_to(Path('../../../') / table, target_is_directory=True)
    protocol = {'version': 'frozen_pair_daily_shadow_v1', 'pair_id': manifest['pair_id'],
                'model_id': 'frozen_pair:' + manifest['pair_id'], 'start_date': str(start_date),
                'frozen_root': str(frozen), 'assets': _assets(frozen),
                'execution': context['execution'], 'signal_policy': context['signal_policy'],
                'minute_policy': manifest['minute_policy'], 'minimum_active_coverage': .98,
                'signal_weights': {'5d': 1.}, 'activation_allowed': False,
                'cutoff': 'same_session_19:00_to_23:59_Asia/Shanghai'}
    if is_decision:
        from app.services.stockformer_decision_shadow import RESEARCH_WARNINGS, training_universe
        protocol.update(variant='decision', ranking_column='pred_ret_raw_5d',
                        training_universe=training_universe(frozen), research_warnings=RESEARCH_WARNINGS)
    if is_repaired:
        from app.services.stockformer_corporate_ledger import POLICY as LEDGER_POLICY
        from app.services.stockformer_execution_quantities import POLICY
        from app.services.stockformer_repaired_shadow import WARNINGS
        if start_date < date.fromisoformat(context['effective_signal_start']):
            raise ValueError('Start precedes repaired pair effective signal date')
        protocol.update(variant='repaired', ranking_column='pred_ret_raw_5d',
            feature_columns=manifest['feature_columns'],
            training_universe=daily._read(frozen/'universe.json')['symbols'],
            feature_input_contract=context['feature_input_contract'],
            research_warnings=WARNINGS, run_after='19:10',
            corporate_action_policy=LEDGER_POLICY,corporate_reference_after='21:05',
            corporate_cash_policy='simulated_close_on_or_after_announced_pay_date',
            corporate_tax_policy='sale_accrual_and_max_20_percent_open_reserve_not_broker_debit',
            order_quantity_policy=POLICY,
            price_limit_mode='authoritative_day')
    pid = daily._hash(protocol)
    cfg = {'enabled': True, 'protocol': protocol, 'protocol_id': pid,
           'account_id': ('sf_repaired_' if is_repaired else 'sf_decision_' if is_decision else 'sf_pair_') + pid[:16], 'created_at': cn_now().isoformat()}
    daily._write(target, cfg)
    daily._account(inputs, cfg)
    return cfg


def _input_version_real(inputs: Path, day: date) -> str:
    # The old shadow's bounded manifest predates industry-relative targets.
    # Include the industry mapping and other external feature inputs here.
    extra = [(str(p.relative_to(inputs)), p.stat().st_size, p.stat().st_mtime_ns)
             for p in sorted((inputs / 'ext_data').rglob('*.parquet'))]
    matched = []
    if (inputs/'research_inputs').exists():
        from app.services.stockformer_repaired_shadow import input_versions
        matched = input_versions(inputs, day)
    return daily._hash([daily._input_version(inputs, day), extra, matched] if matched
                       else [daily._input_version(inputs, day), extra])


_input_version = _input_version_real


def _compute(inputs: Path, cfg: dict, day: date):
    if not any((inputs / 'kline_minute' / f'date={day}').glob('*.parquet')):
        raise WaitingDataError('当天分钟K尚未导入；等待完整240根分钟数据')
    from app.services import stockformer_frozen_pair as pair
    frozen = Path(cfg['protocol']['frozen_root'])
    manifest = _verify_pair(frozen)
    if cfg['protocol'].get('variant') == 'repaired':
        from app.services.stockformer_repaired_shadow import compute
        return compute(inputs, cfg, day, manifest)
    if cfg['protocol'].get('variant') == 'decision':
        from app.services.stockformer_decision_shadow import compute
        return compute(inputs, cfg, day, manifest)
    universe = manifest['provenance']['context']['training_universe']
    from app.services.prediction_workflow import _closed_prices, _frame, _symbols
    active = _symbols(_closed_prices(_frame(inputs, 'kline_daily', day), day)) & set(universe)
    features, actual, _ = sf._load_prediction_base(inputs, day,
        required_feature_columns=manifest['feature_columns'], strict_minute=True, symbols=universe)
    if actual != day or features.is_empty():
        raise WaitingDataError('当天完整分钟特征尚未就绪（要求240根、覆盖率100%）')
    predictions, coverage = pair.predict_pair(frozen / 'latest-pair', features)
    if not active or predictions.height < len(active) * cfg['protocol']['minimum_active_coverage']:
        raise WaitingDataError(f'完整分钟预测覆盖不足：{predictions.height}/{len(active)}，等待补齐')
    if predictions['date'].unique().to_list() != [day] or predictions['symbol'].n_unique() != predictions.height:
        raise ValueError('Frozen prediction date/identity mismatch')
    if not predictions.select(pl.all_horizontal(pl.col('pred_ret_5d').is_finite(),
                                                pl.col('pred_ret_raw_5d').is_finite()).all()).item():
        raise ValueError('Nonfinite frozen predictions')
    policy = cfg['protocol']['signal_policy']
    signal = portfolio._build_gated_signal_frame(inputs, predictions, features,
        {'target_mode': 'industry_relative_net', 'signal_weights': {'5d': 1.}}, horizon=5,
        signal_policy=policy, min_net_edge=policy['prediction_rule']['min_net_edge'])
    coverage = {**coverage, 'active_universe': len(active),
                'not_predicted': sorted(active - set(predictions['symbol']))}
    return predictions, signal, features, coverage


def _settle_sessions_real(inputs: Path, cfg: dict, through: date) -> dict:
    result = {}
    for partition in sorted((inputs / 'kline_daily').glob('date=????-??-??')):
        day = partition.name[5:]
        if not cfg['protocol']['start_date'] <= day <= str(through):
            continue
        if (daily._root(inputs) / 'settlements' / f'{day}.json').exists():
            saved = daily._read(daily._root(inputs) / 'settlements' / f'{day}.json')
            if saved.get('status') != 'settled' and cfg['protocol'].get('corporate_action_policy')!='verified_cash_transfer_ledger_v1':
                raise ValueError('影子结算中断，需核对成交台账')
            if saved.get('status')=='settled' and not (day==str(through)
                and cfg['protocol'].get('corporate_action_policy')=='verified_cash_transfer_ledger_v1'):
                continue
        from app.services.prediction_workflow import _closed_prices, _frame
        if _closed_prices(_frame(inputs, 'kline_daily', date.fromisoformat(day)), date.fromisoformat(day)).is_empty():
            raise WaitingDataError('历史交易日收盘数据未齐：' + day)
        positions = ({} if cfg['protocol'].get('corporate_action_policy')=='verified_cash_transfer_ledger_v1'
                     else paper.load_positions(inputs, cfg['account_id']))
        missing = [s for s, p in positions.items() if p['qty'] > 0
                   and not paper.read_daily_bar(inputs, s, 'stock', day)]
        if missing:
            raise WaitingDataError('持仓估值缺口：' + day + ' ' + ','.join(missing))
        if cfg['protocol'].get('variant') == 'repaired':
            from app.services.stockformer_repaired_shadow import settle
            result = settle(inputs, cfg, date.fromisoformat(day))
        else:
            result = daily._settle(inputs, cfg, date.fromisoformat(day))
    return result


_settle_sessions = _settle_sessions_real


def tick(data_dir: Path, *, now: datetime, variant: str = 'frozen') -> dict:
    if not _LOCK.acquire(blocking=False):
        return {'status': 'busy'}
    now = now.astimezone(CN_TZ)
    inputs = view(data_dir, variant=variant)
    root = daily._root(inputs)
    cfg = daily._read(root / 'config.json')
    execution_context = ExitStack()

    def state(kind, **extra):
        row = {'status': kind, 'as_of': str(now.date()), 'checked_at': now.isoformat(),
               'next_check_at': (now + timedelta(minutes=15)).isoformat(), **extra}
        daily._write(root / 'state.json', row)
        daily._write(root / 'sessions' / f'{now.date()}.json', row)
        return row

    try:
        if not cfg.get('enabled'):
            return {'status': 'disabled'}
        if cfg['protocol'].get('variant', 'frozen') != variant:
            return state('blocked_protocol_changed', reason='独立账户作用域与协议不匹配')
        if str(now.date()) < cfg['protocol']['start_date']:
            return state('before_start')
        if now.weekday() >= 5:
            return state('non_trading_day')
        if now.hour < 19:
            return state('waiting_close', reason='每日19:00后检查完整收盘与分钟特征')
        if variant == 'repaired' and now.strftime('%H:%M') < cfg['protocol']['run_after']:
            return state('waiting_close', reason='新独立账户19:10后计算，错开原账户内存高峰')
        try:
            if daily._hash(cfg['protocol']) != cfg['protocol_id'] or _assets(Path(cfg['protocol']['frozen_root'])) != cfg['protocol']['assets']:
                raise ValueError('模型、校准、规则或执行版本与冻结指纹不同')
        except (ValueError, OSError) as exc:
            return state('blocked_protocol_changed', reason=str(exc))
        if variant == 'repaired':
            from app.services.stockformer_execution_quantities import POLICY, environment
            if cfg['protocol'].get('order_quantity_policy') != POLICY:
                return state('blocked_protocol_changed', reason='研究账户数量规则未冻结')
            execution_context.enter_context(environment(inputs,cfg['account_id'],cfg['protocol']['training_universe']))
            from app.services.stockformer_corporate_ledger import POLICY as LEDGER_POLICY
            from app.services.stockformer_corporate_ledger import environment as ledger_environment
            if cfg['protocol'].get('corporate_action_policy')!=LEDGER_POLICY:
                return state('blocked_protocol_changed',reason='公司行动账务规则未冻结')
            execution_context.enter_context(ledger_environment(inputs,cfg))
        path = root / 'days' / str(now.date()) / 'snapshot.json'
        saved = daily._read(path)
        if saved:
            summary = daily._read(path.parent / 'summary.json')
            if (saved.get('protocol_id') != cfg['protocol_id']
                    or summary.get('snapshot_sha256') != daily._digest(path)
                    or any(daily._digest(path.parent / f'{name}.parquet') != digest
                           for name, digest in saved.get('artifacts', {}).items())
                    or set(saved.get('artifacts', {})) != {'features', 'predictions', 'signals'}):
                return state('blocked_protocol_changed', reason='已发布快照或配套文件指纹改变，暂停恢复下单')
            daily._orders(inputs, cfg, saved, cn_now())
            state(saved['status'], observation=saved['observation'])
            return saved
        ready = readiness(inputs, now.date())
        settlement = None
        if variant == 'repaired' and ready.get('checks', {}).get('日K'):
            from app.services.stockformer_repaired_shadow import execution_ready, prepare
            try:
                prepare(inputs, cfg, now.date())
            except WaitingDataError as exc:
                if execution_ready(inputs,now.date()):
                    daily._account(inputs,cfg)
                    settlement=_settle_sessions(inputs,cfg,now.date())
                    daily._queue_exits(inputs,cfg,now.date())
                return state('waiting_data',reason=str(exc),settlement=settlement)
        if ready.get('ready') or ready.get('checks', {}).get('日K'):
            daily._account(inputs, cfg)
            settlement = _settle_sessions(inputs, cfg, now.date())
            daily._queue_exits(inputs, cfg, now.date())
        if not ready['ready']:
            return state('waiting_data', readiness=ready, reason=ready['reason'])
        before = _input_version(inputs, now.date())
        state('computing_features', reason='加载完整分钟特征、冻结树预测与校准')
        prediction, signal, features, coverage = _compute(inputs, cfg, now.date())
        if before != _input_version(inputs, now.date()):
            return state('inputs_changed', reason='计算期间数据版本改变，丢弃后重试')
        if _assets(Path(cfg['protocol']['frozen_root'])) != cfg['protocol']['assets']:
            return state('blocked_protocol_changed')
        observations, _ = _daily_observations(signal, available_dates=[now.date()],
            pred_col='pred_ret_5d', max_positions=cfg['protocol']['execution']['max_positions'], forecast_ceiling=None)
        obs = observations[0]
        if variant in {'decision', 'repaired'} and coverage.get('decision_pool_rows') == 0:
            obs['no_trade_reason'] = 'decision_pool_empty'
        obs['prediction_net_entry_pass'] = signal.filter(pl.col('_prediction_pass') &
            pl.col('_net_margin_pass') & pl.col('_entry_rules_pass')).height if not signal.is_empty() else 0
        selected = signal
        if not signal.is_empty():
            selected = signal.filter(pl.col('_forecast_pass') & pl.col('_entry_rules_pass') & pl.col('_market_gate_pass'))
            ranking = cfg['protocol'].get('ranking_column', 'net_margin_score')
            selected = selected.sort([ranking, 'symbol'], descending=[True, False]).head(cfg['protocol']['execution']['max_positions'])
        if not signal.is_empty():
            reasons = [[name for name, column in [('prediction_gate', '_prediction_pass'),
                ('net_margin', '_net_margin_pass'), ('entry_rules', '_entry_rules_pass'),
                ('market_gate', '_market_gate_pass')] if not row[column]] for row in signal.to_dicts()]
            signal = signal.with_columns(pl.Series('filter_reasons', reasons, dtype=pl.List(pl.String)))
        path.parent.mkdir(parents=True, exist_ok=True)
        for name, frame in [('features', features), ('predictions', prediction), ('signals', signal)]:
            atomic_write_parquet(frame, path.parent / f'{name}.parquet')
        if before != _input_version(inputs, now.date()):
            return state('inputs_changed', reason='保存特征期间数据版本改变，丢弃后重试')
        finished = cn_now()
        timely = finished.date() == now.date() and finished.hour >= 19
        if not timely:
            selected = selected.clear()
            obs.update(selected_count=0, no_trade_reason='late_snapshot')
        snapshot = {'status': 'recorded' if timely else 'late_record',
            'id': daily._hash([cfg['protocol_id'], str(now.date())]), 'protocol_id': cfg['protocol_id'],
            'pair_id': cfg['protocol']['pair_id'], 'model_id': cfg['protocol']['model_id'],
            'account_id': cfg['account_id'], 'as_of': str(now.date()), 'started_at': now.isoformat(),
            'generated_at': finished.isoformat(), 'eligible_for_forward': timely,
            'entry_rule': 'next_actual_session_open_after_generated_at', 'input_version': before,
            'minute_coverage': coverage, 'observation': obs, 'prediction_rows': prediction.height,
            'raw_positive_rows': prediction.filter(pl.col('pred_ret_raw_5d') > 0).height,
            'calibrated_positive_rows': prediction.filter(pl.col('pred_ret_5d') > 0).height,
            'excluded_before_gates': sorted(set(prediction['symbol']) - set(signal['symbol'])) if not signal.is_empty() else prediction['symbol'].to_list(),
            'selected': selected.to_dicts(), 'settlement': settlement,
            'research_warnings': cfg['protocol'].get('research_warnings', []),
            'artifacts': {name: daily._digest(path.parent / f'{name}.parquet') for name in ('features', 'predictions', 'signals')},
            'activation_allowed': False}
        if variant == 'repaired':
            snapshot['next_actual_session_open'] = coverage['matched_inputs']['next_actual_session_open']
        daily._write(path, snapshot)
        summary = {k: snapshot[k] for k in ('as_of', 'generated_at', 'status', 'eligible_for_forward',
                                          'observation', 'raw_positive_rows', 'calibrated_positive_rows')}
        summary['snapshot_sha256'] = daily._digest(path)
        if variant == 'repaired':
            summary['next_actual_session_open'] = snapshot['next_actual_session_open']
        daily._write(path.parent / 'summary.json', summary)
        daily._orders(inputs, cfg, snapshot, finished)
        state(snapshot['status'], observation=obs)
        return snapshot
    except WaitingDataError as exc:
        return state('waiting_data', reason=str(exc))
    except Exception as exc:
        logger.exception('Frozen pair daily shadow failed')
        return state('error', reason=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            execution_context.close()
        finally:
            _LOCK.release()


def status(data_dir: Path, *, variant: str = 'frozen') -> dict:
    inputs = view(data_dir, variant=variant)
    result = daily.status(inputs)
    cfg = daily._read(daily._root(inputs) / 'config.json')
    corporate=cfg.get('protocol',{}).get('corporate_action_policy')=='verified_cash_transfer_ledger_v1'
    rounds = paper.round_trips(inputs, cfg['account_id']) if cfg and not corporate else []
    nav = paper.load_nav(inputs, cfg['account_id']) if cfg else []
    heartbeat = daily._read(scope(data_dir, variant=variant) / 'worker.json')
    try:
        age = (cn_now() - datetime.fromisoformat(heartbeat['checked_at'])).total_seconds()
    except (KeyError, ValueError):
        age = float('inf')
    result.update(pair_id=cfg.get('protocol', {}).get('pair_id'), record_kind='frozen_tree_calibration',
        worker={**heartbeat, 'alive': 0 <= age < 180}, completed_trades=len(rounds),
        cost_after_return_pct=(nav[-1]['nav'] / cfg['protocol']['execution']['initial_cash'] - 1) * 100 if rounds and nav else None,
        research_warnings=cfg.get('protocol', {}).get('research_warnings', []), activation_allowed=False)
    if corporate:
        metrics=daily._read(daily._root(inputs)/'execution_metrics.json')
        base=paper.accounts_base(inputs)/cfg['account_id']
        def fingerprint(path):
            return daily._digest(path) if path.exists() else daily._hash([])
        valid=(metrics.get('protocol_id')==cfg['protocol_id'] and metrics.get('account_id')==cfg['account_id']
               and metrics.get('ledger_sha256')==fingerprint(base/'fills.jsonl')
               and metrics.get('nav_sha256')==fingerprint(base/'nav/daily.jsonl'))
        result.update(completed_trades=metrics['completed_trades'] if valid else 0,
                      cost_after_return_pct=metrics['cost_after_return_pct'] if valid else None)
        if not valid and (base/'fills.jsonl').exists():
            result['research_warnings']=[*result['research_warnings'],'账务摘要未完成或指纹变化，收益暂不展示。']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('/app/data'))
    parser.add_argument('--configure', type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--variant', choices=['frozen', 'decision', 'repaired'], default='frozen')
    parser.add_argument('--start-date', type=date.fromisoformat)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.configure:
        configure(args.data_dir, args.configure, start_date=args.start_date or cn_now().date(), variant=args.variant)
    while True:
        now = cn_now()
        state = daily._read(daily._root(view(args.data_dir, variant=args.variant)) / 'state.json')
        if state.get('as_of') != str(now.date()) or daily.due(view(args.data_dir, variant=args.variant), now):
            result = tick(args.data_dir, now=now, variant=args.variant)
            logger.info('pair shadow %s', result.get('status'))
        daily._write(scope(args.data_dir, variant=args.variant) / 'worker.json', {'checked_at': cn_now().isoformat(), 'schedule': '19:00后每15分钟检查'})
        if args.once:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
