from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import polars as pl
import pytest

from app.services import index_incremental as sync
from app.services import maintenance_queue as queue

DAYS = [date(2026, 9, 4), date(2026, 9, 7), date(2026, 9, 8)]
SYMBOLS = ['000001.SH', '399001.SZ', '399006.SZ']


def bars(symbol, days):
    return pl.DataFrame({
        'symbol': [symbol] * len(days), 'date': days,
        **{c: [10.0] * len(days) for c in ['open', 'high', 'low', 'close']},
        'volume': [100.0] * len(days), 'amount': [1000.0] * len(days),
    })


class Repo:
    def __init__(self, root):
        self.store = SimpleNamespace(data_dir=root)
        self.refreshes = 0
        for day in DAYS:
            (root / 'kline_daily' / f'date={day}').mkdir(parents=True)

    def append_index_daily(self, frame):
        for part in frame.partition_by('date'):
            path = self.store.data_dir / 'kline_index_daily' / f"date={part['date'][0]}" / 'part.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            old = pl.read_parquet(path) if path.exists() else part.clear()
            pl.concat([old, part]).unique(['symbol', 'date'], keep='last').write_parquet(path)

    def append_index_enriched(self, frame):
        pass

    def refresh_index_views(self):
        self.refreshes += 1

    def clear_cache(self):
        pass


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, 'cn_now', lambda: datetime(2026, 9, 9, 1, tzinfo=ZoneInfo('Asia/Shanghai')))
    monkeypatch.setattr(queue, 'cn_now', sync.cn_now)
    monkeypatch.setattr(sync, 'compute_enriched', lambda raw, **kw: raw)
    return Repo(tmp_path)


def provider(monkeypatch, fetch):
    monkeypatch.setattr(sync.index_sync, 'custom_asset_provider', lambda: SimpleNamespace(get_asset_daily=fetch))


def run(repo, **kwargs):
    return sync.sync(repo, SimpleNamespace(), symbols=SYMBOLS,
                     start_date=DAYS[0], end_date=date(2026, 9, 9), **kwargs)


def test_only_missing_index_dates_requested_and_restart_skips_written(repo, monkeypatch):
    repo.append_index_daily(bars(SYMBOLS[0], DAYS))
    repo.append_index_daily(bars(SYMBOLS[1], [DAYS[0], DAYS[-1]]))
    calls = []

    def fetch(symbols, start, end, kind, callback=None):
        calls.append((symbols[0], start.date(), end.date(), kind))
        return bars(symbols[0], [d for d in DAYS if start.date() <= d <= end.date()])

    provider(monkeypatch, fetch)
    result = run(repo)
    assert calls == [(SYMBOLS[1], DAYS[1], DAYS[1], 'index'), (SYMBOLS[2], DAYS[0], DAYS[-1], 'index')]
    assert result['complete'] and result['completed_count'] == 3
    assert result['written_rows'] == 4 and result['remaining_count'] == 0
    calls.clear()
    assert run(repo)['written_rows'] == 0 and not calls
    assert all(t['status'] == 'succeeded' for t in queue.status(repo.store.data_dir))


def test_429_retries_after_server_delay(repo, monkeypatch):
    calls, sleeps = [], []

    def fetch(symbols, start, end, kind, callback=None):
        calls.append(symbols[0])
        if len(calls) == 1:
            httpx.Response(429, headers={'Retry-After': '7'}, request=httpx.Request('GET', 'https://example.test')).raise_for_status()
        return bars(symbols[0], DAYS)

    monkeypatch.setattr(sync.time, 'sleep', sleeps.append)
    provider(monkeypatch, fetch)
    result = run(repo)
    assert result['complete'] and sleeps == [7.0]
    assert calls[:2] == [SYMBOLS[0], SYMBOLS[0]]
    assert result['rate_limit_retries'] == 1


def test_persistent_429_keeps_partial_data_and_only_queues_remainder(repo, monkeypatch):
    calls, sleeps = [], []

    def fetch(symbols, start, end, kind, callback=None):
        calls.append(symbols[0])
        if symbols[0] != SYMBOLS[0]:
            httpx.Response(429, request=httpx.Request('GET', 'https://example.test')).raise_for_status()
        return bars(symbols[0], DAYS)

    monkeypatch.setattr(sync.time, 'sleep', sleeps.append)
    provider(monkeypatch, fetch)
    result = run(repo)
    assert not result['complete'] and result['remaining_count'] == 2
    assert result['written_rows'] == 3 and result['completed_symbols'] == [SYMBOLS[0]]
    assert calls == [SYMBOLS[0]] + [SYMBOLS[1]] * 4
    assert sleeps == [2, 4, 8]
    task = queue._read(repo.store.data_dir)[0]
    assert task['items'] == SYMBOLS[1:] and task['status'] == 'retry'
    queue.recover(repo.store.data_dir)
    calls.clear()
    provider(monkeypatch, lambda symbols, *a: calls.append(symbols[0]) or bars(symbols[0], DAYS))
    assert not run(repo)['complete'] and not calls
    def later():
        return datetime(2026, 9, 9, 2, tzinfo=ZoneInfo('Asia/Shanghai'))
    monkeypatch.setattr(sync, 'cn_now', later)
    monkeypatch.setattr(queue, 'cn_now', later)
    assert run(repo)['complete'] and calls == SYMBOLS[1:]


def test_long_retry_after_does_not_block_worker_or_query_next_index(repo, monkeypatch):
    calls = []

    def fetch(symbols, *args):
        calls.append(symbols[0])
        httpx.Response(429, headers={'Retry-After': '3600'}, request=httpx.Request('GET', 'https://example.test')).raise_for_status()

    provider(monkeypatch, fetch)
    monkeypatch.setattr(sync.time, 'sleep', lambda _: pytest.fail('Long blocking sleep'))
    result = run(repo)
    assert len(calls) == 1 and result['retry_after_seconds'] >= 3600
    assert result['remaining_count'] == 3


def test_empty_and_invalid_sources_are_not_success(repo, monkeypatch):
    def fetch(symbols, *args):
        if symbols[0] == SYMBOLS[0]:
            return pl.DataFrame()
        if symbols[0] == SYMBOLS[1]:
            return bars('600000.SH', DAYS)
        return bars(symbols[0], DAYS).with_columns(pl.lit(float('inf')).alias('close'))

    provider(monkeypatch, fetch)
    result = run(repo)
    assert result['remaining_count'] == 3 and not result['complete']
    assert result['written_rows'] == 0
    assert result['reasons']['source_empty'] == 1
    assert result['reasons']['invalid_response'] == 2


def test_bounded_unit_leaves_durable_unrequested_indices(repo, monkeypatch):
    calls = []
    provider(monkeypatch, lambda symbols, *a: calls.append(symbols[0]) or bars(symbols[0], DAYS))
    result = run(repo, max_symbols=1)
    assert calls == SYMBOLS[:1] and result['remaining_count'] == 2
    task = queue._read(repo.store.data_dir)[0]
    assert task['completed'] == 1 and task['items'] == SYMBOLS[1:]


def test_cancel_keeps_checkpoint_and_already_persisted_index(repo, monkeypatch):
    def fetch(symbols, *args):
        if symbols[0] != SYMBOLS[0]:
            raise KeyboardInterrupt()
        return bars(symbols[0], DAYS)

    provider(monkeypatch, fetch)
    with pytest.raises(KeyboardInterrupt):
        run(repo)
    assert pl.read_parquet(repo.store.data_dir / 'kline_index_daily' / f'date={DAYS[-1]}' / 'part.parquet').height == 1
    assert queue._read(repo.store.data_dir)[0]['items'] == SYMBOLS[1:]


def test_no_confirmed_calendar_fails_closed(repo, monkeypatch):
    provider(monkeypatch, lambda *a: pytest.fail('No confirmed trading dates'))
    with pytest.raises(ValueError, match='交易日'):
        sync.sync(repo, SimpleNamespace(), symbols=SYMBOLS,
                  start_date=date(2026, 10, 1), end_date=date(2026, 10, 5))


def test_queue_only_api_does_not_fetch_or_clear_existing_cooldown(repo, monkeypatch):
    provider(monkeypatch, lambda *a: pytest.fail('API must not run network work'))
    result = run(repo, max_symbols=0)
    queue.checkpoint(repo.store.data_dir, result['queue_id'], [], retry_after=3600)
    old = queue.status(repo.store.data_dir)[0]['retry_at']
    assert run(repo, max_symbols=0)['remaining_count'] == 3
    task = queue.status(repo.store.data_dir)[0]
    assert task['status'] == 'retry' and task['retry_at'] == old


def test_partial_source_retries_only_unwritten_days(repo, monkeypatch):
    provider(monkeypatch, lambda symbols, *a: bars(symbols[0], DAYS[:1]))
    result = run(repo)
    assert not result['complete'] and result['written_rows'] == 3
    calls = []
    provider(monkeypatch, lambda symbols, start, *a: calls.append(start.date()) or bars(symbols[0], DAYS[1:]))
    assert run(repo)['complete'] and calls == [DAYS[1]] * 3


def test_http_date_retry_after_and_permission_failure(repo, monkeypatch):
    waits, calls = [], []
    monkeypatch.setattr(sync.time, 'time', lambda: 0)
    monkeypatch.setattr(sync.time, 'sleep', waits.append)

    def fetch(symbols, *args):
        calls.append(symbols[0])
        status = 429 if len(calls) == 1 else 403
        httpx.Response(status, headers={'Retry-After': 'Thu, 01 Jan 1970 00:00:09 GMT'},
                       request=httpx.Request('GET', 'https://example.test?secret=private')).raise_for_status()

    provider(monkeypatch, fetch)
    result = run(repo)
    assert waits == [9] and len(calls) == 2 and result['remaining_count'] == 3
    assert result['reasons'] == {'http_403': 1} and 'private' not in str(result)


def test_invalid_local_price_is_not_counted_complete(repo, monkeypatch):
    repo.append_index_daily(bars(SYMBOLS[0], DAYS).with_columns(pl.lit(0.0).alias('close')))
    provider(monkeypatch, lambda symbols, *a: bars(symbols[0], DAYS))
    result = run(repo)
    assert result['complete'] and result['already_complete_count'] == 0


def test_api_queues_without_network_and_reports_remaining(repo, monkeypatch):
    from app import market_time
    from app.api.indices import sync_index_daily
    monkeypatch.setattr(market_time, 'cn_now', sync.cn_now)
    repo.get_index_instruments = lambda: pl.DataFrame({'symbol': SYMBOLS, 'asset_type': ['index'] * 3})
    caps = SimpleNamespace(has=lambda cap: False)
    provider(monkeypatch, lambda *a: pytest.fail('API only queues work'))
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repo=repo, capabilities=caps)))
    result = sync_index_daily(req, days=7)
    assert result['status'] == 'queued' and result['rows_written'] == 0
    assert result['index_sync']['remaining_count'] == 3


def test_queued_worker_returns_only_completed_symbols_and_invalidates(repo, monkeypatch):
    from app.jobs import daily_pipeline
    from app.services import daily_maintenance
    invalidated = []
    monkeypatch.setattr(daily_pipeline, '_invalidate', invalidated.append)
    provider(monkeypatch, lambda symbols, *a: bars(symbols[0], DAYS) if symbols[0] == SYMBOLS[0] else pl.DataFrame())
    result = daily_maintenance.execute(repo, SimpleNamespace(), {
        'items': SYMBOLS, 'kind': 'index_daily', 'label': 'index test',
        'options': {'start_date': str(DAYS[0]), 'end_date': str(DAYS[-1])},
    }, lambda *a, **kw: None)
    assert result['completed_items'] == SYMBOLS[:1]
    assert '部分完成' in result['error']
    assert result['index_sync']['remaining_count'] == 2
    assert invalidated == ['index_daily', 'index_enriched']


def test_derived_write_failure_does_not_mark_raw_complete(repo, monkeypatch):
    provider(monkeypatch, lambda symbols, *a: bars(symbols[0], DAYS))
    def fail(frame):
        raise OSError('test write failure')
    repo.append_index_enriched = fail
    with pytest.raises(OSError):
        run(repo)
    assert not (repo.store.data_dir / 'kline_index_daily').exists()
    assert queue._read(repo.store.data_dir)[0]['items'] == SYMBOLS


def test_worker_honors_long_provider_retry_after(repo, monkeypatch):
    now = queue.cn_now()
    queue.enqueue(repo.store.data_dir, 'index_daily', SYMBOLS)
    def runner(fn, label, on_started):
        on_started()
        fn()
        return True
    assert not queue.tick(repo.store.data_dir, runner, lambda *a: {
        'completed_items': [], 'retry_after_seconds': 3600, 'error': 'rate_limited',
    }, now=now)
    retry = datetime.fromisoformat(queue.status(repo.store.data_dir)[0]['retry_at'])
    assert (retry - now).total_seconds() >= 3600


def test_real_tushare_adapter_429_and_units(repo, monkeypatch):
    import pandas as pd

    from app.plugins.tushare.provider import TushareProvider

    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            httpx.Response(429, request=httpx.Request('GET', 'https://example.test/index_daily')).raise_for_status()
        return pd.DataFrame([{'ts_code': kwargs['ts_code'], 'trade_date': d.strftime('%Y%m%d'),
                              'open': 10, 'high': 10, 'low': 10, 'close': 10,
                              'vol': 100, 'amount': 1000} for d in DAYS])
    adapter = TushareProvider()
    monkeypatch.setattr(adapter, '_ensure_api', lambda: SimpleNamespace(index_daily=fetch))
    monkeypatch.setattr(sync.index_sync, 'custom_asset_provider', lambda: adapter)
    monkeypatch.setattr(sync.time, 'sleep', lambda _: None)
    result = run(repo)
    assert result['complete'] and result['rate_limit_retries'] == 1
    saved = pl.read_parquet(repo.store.data_dir / 'kline_index_daily' / f'date={DAYS[-1]}' / 'part.parquet')
    assert saved['volume'].to_list() == [10000.0] * 3
    assert saved['amount'].to_list() == [1000000.0] * 3


def test_tracked_partial_index_retains_counts_in_job_result(tmp_path, monkeypatch):
    from app.jobs.daily_pipeline import _run_tracked
    from app.services import pipeline_jobs

    store = pipeline_jobs.JobStore(store_dir=tmp_path / 'jobs')
    monkeypatch.setattr(pipeline_jobs, 'job_store', store)
    result = {'completed_items': ['000001.SH'], 'error': 'partial',
              'index_sync': {'complete': False, 'remaining_count': 2}}
    assert _run_tracked(lambda **kw: result, 'index partial test')
    saved = store.list_recent()[0]
    assert saved['status'] == 'succeeded'
    assert saved['result']['index_sync']['remaining_count'] == 2


def test_unavailable_market_does_not_rotate_forever_every_minute(repo, monkeypatch):
    from datetime import timedelta
    now = queue.cn_now()
    codes = [f'{i:06}.SH' for i in range(59)]
    queue.enqueue(repo.store.data_dir, 'index_daily', codes)
    def runner(fn, label, on_started):
        on_started()
        fn()
        return True
    delays = []
    for _ in range(5):
        monkeypatch.setattr(queue, 'cn_now', lambda value=now: value)
        queue.tick(repo.store.data_dir, runner, lambda *a: {'completed_items': [], 'error': 'source_empty'}, now=now)
        retry = datetime.fromisoformat(queue.status(repo.store.data_dir)[0]['retry_at'])
        delays.append((retry - now).total_seconds())
        now = retry + timedelta(seconds=1)
    assert delays == [60, 60, 60, 300, 600]
