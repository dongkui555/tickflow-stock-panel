"""Closed-day index gap filling with bounded retries and durable continuation."""

from __future__ import annotations

import math
import time
from collections import Counter
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime

import httpx
import polars as pl

from app.indicators.pipeline import compute_enriched
from app.market_time import cn_now
from app.services import index_sync, kline_sync, maintenance_queue
from app.tickflow.capabilities import Cap

COLS = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']


def _valid():
    return (
        pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in COLS[2:6]])
        & (pl.col('high') >= pl.max_horizontal(COLS[2:6]))
        & (pl.col('low') <= pl.min_horizontal(COLS[2:6]))
        & pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) >= 0) for c in COLS[6:]])
    ).fill_null(False)


def _plan(root, symbols, start, end):
    now = cn_now()
    closed = now.date() if now.hour >= 15 else now.date() - timedelta(days=1)
    end = min(end, closed)
    # The local market calendar is evidence of trading days, not a weekday guess.
    dates = [d for d in kline_sync._partition_dates(root / 'kline_daily') if start <= d <= end]
    if not dates:
        raise ValueError('区间内没有已确认的收盘交易日, 不能认定指数同步完成')
    missing = {symbol: set(dates) for symbol in symbols}
    for day in dates:
        path = root / 'kline_index_daily' / f'date={day}' / 'part.parquet'
        if not path.exists():
            continue
        frame = pl.read_parquet(path).filter(pl.col('symbol').is_in(symbols))
        if not set(COLS) <= set(frame.columns):
            continue
        frame = frame.filter((pl.col('date') == day) & _valid())
        for symbol in frame.group_by('symbol').len().filter(pl.col('len') == 1)['symbol']:
            missing[symbol].discard(day)
    return {s: ds for s, ds in missing.items() if ds}, dates[-1]


def _retry_after(exc, attempt):
    value = exc.response.headers.get('Retry-After', '')
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            delay = 2 ** (attempt + 1)
    return max(0.1, delay) if math.isfinite(delay) else 2 ** (attempt + 1)


def message(result):
    label = '完成' if result['complete'] else '部分完成'
    return (f"指数日K{label}: 已完成 {result['completed_count']}/{result['total_count']} 只, "
            f"未完成 {result['remaining_count']} 只, 本轮写入 {result['written_rows']} 行"
            + ('; 剩余缺口已排队续补' if result.get('queue_id') and not result['complete'] else ''))


def sync(repo, capset, *, symbols, start_date: date, end_date: date,
         on_progress=None, max_symbols=40, queue_pending=True):
    symbols = sorted(set(symbols))
    if not symbols or any('.' not in s for s in symbols):
        raise ValueError('指数同步需要带交易所后缀的非空标的列表')
    if start_date > end_date or max_symbols < 0:
        raise ValueError('Invalid index synchronization range or batch size')
    root = repo.store.data_dir
    pending, target = _plan(root, symbols, start_date, end_date)
    original_pending = list(pending)
    provider = index_sync.custom_asset_provider()
    qmt_provider = index_sync.qmt_index_provider()
    if provider is None and not capset.has(Cap.KLINE_DAILY_BATCH):
        raise ValueError('当前数据源不支持指数日K')
    task_id = maintenance_queue.enqueue(
        root, 'index_daily', original_pending,
        start_date=str(start_date), end_date=str(target),
    ) if queue_pending and pending else None
    written = retried = attempted = 0
    changed = False
    retry_after = 0.0
    if task_id:
        existing_task = next(t for t in maintenance_queue.status(root) if t['id'] == task_id)
        if existing_task.get('retry_at') and existing_task['status'] == 'retry':
            retry_after = max(0.0, (datetime.fromisoformat(existing_task['retry_at']) - cn_now()).total_seconds())
            if retry_after:
                max_symbols = 0
    reasons = Counter()
    details = []
    unavailable_streak = 0

    def progress(text):
        if on_progress:
            on_progress(len(symbols) - len(pending), len(symbols), text)

    def checkpoint():
        if task_id:
            maintenance_queue.checkpoint(
                root, task_id, [s for s in original_pending if s not in pending],
                retry_after=retry_after,
            )

    try:
        progress(f'本地已完整 {len(symbols) - len(pending)} 只, 待补 {len(pending)} 只')
        for symbol in original_pending[:max_symbols]:
            days = pending[symbol]
            start = datetime.combine(min(days), datetime.min.time())
            end = datetime.combine(max(days), datetime.min.time())
            attempted += 1
            raw = None
            reason = None
            for attempt in range(4):
                try:
                    raw = (provider.get_asset_daily([symbol], start, end, 'index') if provider is not None
                           else kline_sync.sync_daily_batch([symbol], start_time=start, end_time=end))
                    break
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    reason = f'http_{status}'
                    if status != 429:
                        break
                    delay = _retry_after(exc, attempt)
                    if attempt == 3 or delay > 30:
                        retry_after = max(60.0, delay)
                        break
                    retried += 1
                    progress(f'{symbol} 接口限流, 等待 {delay:g} 秒后重试 {attempt + 1}/3')
                    time.sleep(delay)
                except Exception as exc:
                    # Keep only safe categories; HTTP exception strings may expose credentials.
                    reason = 'timeout' if isinstance(exc, (httpx.TimeoutException, TimeoutError)) else 'request_failed'
                    break
            if raw is None or raw.is_empty():
                if qmt_provider is not None and qmt_provider is not provider:
                    try:
                        qmt_raw = qmt_provider.get_asset_daily([symbol], start, end, 'index')
                        if qmt_raw is not None and not qmt_raw.is_empty():
                            raw = qmt_raw
                            reason = None
                            details.append({'symbol': symbol, 'source': 'qmt', 'reason': 'accepted_fallback'})
                    except Exception:
                        details.append({'symbol': symbol, 'source': 'qmt', 'reason': 'request_failed'})
            if raw is not None:
                reason = None
                if raw.is_empty():
                    reason = 'source_empty'
                elif (not set(COLS) <= set(raw.columns)
                      or set(raw['symbol']) != {symbol}
                      or not set(raw['date']) <= set(d for d in kline_sync._partition_dates(root / 'kline_daily')
                                                    if start.date() <= d <= end.date())
                      or not raw.select(_valid().all()).item()
                      or raw.unique(['symbol', 'date']).height != raw.height):
                    reason = 'invalid_response'
                else:
                    # Accept only missing keys. No valid existing index day is overwritten.
                    delta = raw.filter(pl.col('date').is_in(sorted(days))).select(COLS)
                    if not delta.is_empty():
                        enriched = compute_enriched(delta, factors=None, instruments=None)
                        # Raw rows act as the resume checkpoint, so publish them last.
                        changed = True
                        repo.append_index_enriched(enriched)
                        repo.append_index_daily(delta)
                        written += delta.height
                        remaining, _ = _plan(root, [symbol], start_date, target)
                        if symbol in remaining:
                            pending[symbol] = remaining[symbol]
                        else:
                            del pending[symbol]
                        checkpoint()
                    if symbol in pending:
                        reason = 'source_incomplete'
            if reason:
                reasons[reason] += 1
                details.append({'symbol': symbol, 'reason': reason})
            unavailable_streak = unavailable_streak + 1 if reason in {'timeout', 'request_failed', 'http_502', 'http_503', 'http_504'} else 0
            progress(f'指数补缺已处理 {attempted} 只, 本轮写入 {written} 行; 未完成 {len(pending)} 只')
            if reason in {'http_429', 'http_401', 'http_403'} or unavailable_streak >= 3:
                # Gateway-wide refusal: persist the cursor instead of hammering other symbols.
                retry_after = max(retry_after, 300)
                break
    finally:
        checkpoint()
        if changed:
            repo.refresh_index_views()
            repo.clear_cache()
    result = {
        'complete': not pending, 'status': 'ok' if not pending else 'partial',
        'target_date': str(target), 'start_date': str(start_date),
        'total_count': len(symbols), 'completed_count': len(symbols) - len(pending),
        'remaining_count': len(pending), 'written_rows': written,
        'requested_count': attempted, 'already_complete_count': len(symbols) - len(original_pending),
        'completed_symbols': [s for s in symbols if s not in pending],
        'remaining_symbols': list(pending), 'remaining_stock_days': sum(map(len, pending.values())),
        'rate_limit_retries': retried, 'retry_after_seconds': retry_after,
        'reasons': dict(reasons), 'details': details, 'queue_id': task_id,
    }
    progress(message(result))
    return result
