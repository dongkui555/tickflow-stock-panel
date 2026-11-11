from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import polars as pl
import pytest
from test_minute_failover import DAY, NOW, bars, fetch, source
from test_minute_failover import setup as _failover_setup

from app.plugins.tushare.provider import TushareMinutePendingError, TushareProvider
from app.services import minute_failover as mf

setup = _failover_setup


def test_unrequested_symbols_keep_pending_cause(monkeypatch):
    call = Mock(side_effect=TushareMinutePendingError(180))
    provider = TushareProvider()
    monkeypatch.setattr(provider, '_ensure_api', lambda: SimpleNamespace(stk_mins=call))
    _, issues = provider.get_minute_diagnostics(
        ['000701', '000757', '000001', '000002'],
        datetime(2025, 7, 3, 9, 30), datetime(2025, 7, 3, 15))
    assert call.call_count == 3
    skipped = next(i for i in issues if i['symbol'] == '000002')
    assert skipped['reason'] == 'not_requested'
    assert skipped['cause'] == 'source_pending'
    assert skipped['retry_after_seconds'] == 180


def test_retry_after_survives_http_error():
    from app.plugins.tushare.provider import minute_error_details
    response = httpx.Response(429, headers={'Retry-After': '600'},
                              request=httpx.Request('GET', 'https://example.test'))
    exc = httpx.HTTPStatusError('private response', request=response.request, response=response)
    assert minute_error_details(exc)['retry_after_seconds'] == 600


def test_source_cooldown_blocks_repeated_probes_and_recovers(setup, monkeypatch):
    root, backup = setup
    primary = source(issues=[{'reason': 'rate_limited', 'retryable': True,
                             'http_status': 429, 'retry_after_seconds': 180}])
    backup.probe_minute = Mock(return_value={'usable': False, 'reason': 'source_empty',
                                           'retryable': True, 'retry_after_seconds': 180})
    _, issues = fetch(primary)
    assert any(i.get('http_status') == 429 for i in issues)
    assert mf.retry_state(root)['retry_after_seconds'] >= 180
    assert mf._deferred(root, 'tdx')['cause'] == 'source_empty'
    fetch(primary)
    assert primary.get_minute_diagnostics.call_count == 1
    assert backup.probe_minute.call_count == 1
    backup.get_minute.assert_not_called()
    monkeypatch.setattr(mf, 'cn_now', lambda: NOW + timedelta(seconds=181))
    primary.get_minute_diagnostics.return_value = (bars(), [])
    result, _ = fetch(primary)
    assert result.equals(bars())
    assert mf.retry_state(root)['retry_after_seconds'] == 0


def test_single_stock_empty_does_not_disable_source(setup):
    _, backup = setup
    primary = source()
    fetch(primary)
    fetch(primary)
    assert primary.get_minute_diagnostics.call_count == 2
    assert backup.get_minute.call_count == 2


def test_pending_request_does_not_cool_down_other_symbols(setup):
    root, backup = setup
    primary = source(issues=[dict(reason='source_pending', http_status=202,
                                 retryable=True, retry_after_seconds=180)])
    fetch(primary)
    fetch(primary, ['000757'])
    assert primary.get_minute_diagnostics.call_count == 2
    assert backup.get_minute.call_count == 2
    assert mf._deferred(root, 'promax') is None


def test_mixed_accepted_and_empty_coverage_does_not_fail(setup):
    root, _ = setup
    mf._record_source_coverage(root, [
        dict(source='tdx', symbol='000701', date=str(DAY), reason='accepted'),
        dict(source='tdx', symbol='000701', date=str(DAY + timedelta(days=1)), reason='source_empty'),
    ])
    assert mf.source_coverage(root)['tdx|000701']['last_accepted_date'] == str(DAY)
    assert not mf.source_unsupported(root, 'tdx', '000701')


def test_two_consecutive_historical_empty_days_disable_source(setup):
    root, _ = setup
    mf._record_source_coverage(root, [
        {'source': 'promax', 'symbol': '000701', 'date': str(DAY), 'reason': 'source_empty'},
        {'source': 'promax', 'symbol': '000701', 'date': str(DAY + timedelta(days=1)),
         'reason': 'source_empty'},
    ])
    state = mf.source_coverage(root)['promax|000701']
    assert state['consecutive_empty'] == 2
    assert mf.source_unsupported(root, 'promax', '000701')


def test_partial_success_does_not_hide_source_abort(setup):
    root, backup = setup
    primary = source(bars(), [dict(symbol='000757', reason='rate_limited', retryable=True,
                                  retry_after_seconds=180, batch_aborted=True)])
    backup.probe_minute = Mock(return_value={'usable': False, 'reason': 'source_empty', 'retryable': True})
    result, _ = fetch(primary, ['000701', '000757'])
    assert result.equals(bars())
    assert mf.retry_state(root)['source_retry']
    assert mf._deferred(root, 'promax')['cause'] == 'rate_limited'


def test_daily_sync_never_calls_legacy_full_history_checks(setup, monkeypatch):
    from test_minute_failover import route
    root, _ = setup
    ks, _ = route(monkeypatch)
    scan = Mock(side_effect=AssertionError('full history scan'))
    monkeypatch.setattr(ks, '_cleanup_null_datetime_minute', scan)
    monkeypatch.setattr(ks, '_migrate_symbol_to_date_partition', scan)
    monkeypatch.setattr(ks, '_latest_minute_datetime_by_symbol', lambda *a, **kw: {})
    monkeypatch.setattr(ks, '_closed_minute_gap_starts', lambda *a, **kw: {})
    monkeypatch.setattr(ks, '_minute_increment_groups', lambda *a, **kw: [])
    monkeypatch.setattr(ks, 'resolve_limit', lambda *a, **kw: SimpleNamespace(batch=50, rpm=None))
    progress = Mock()
    assert ks.sync_and_persist_minute(['000701'], SimpleNamespace(store=SimpleNamespace(data_dir=root)),
                                     Mock(), days=1, as_of=datetime(2026, 9, 7, 15),
                                     on_chunk_done=progress) == 0
    scan.assert_not_called()
    assert progress.called


@pytest.mark.parametrize('hour,minute,expected', [(14, 59, False), (15, 0, False), (15, 9, False), (15, 10, True), (19, 0, True)])
def test_same_day_closed_sync_can_try_local_backup(setup, monkeypatch, hour, minute, expected):
    from test_minute_failover import route

    root, _ = setup
    ks, _ = route(monkeypatch)
    monkeypatch.setattr(ks, 'cn_now', lambda: NOW.replace(hour=hour, minute=minute))
    retry = Mock(return_value={'source_retry': False})
    monkeypatch.setattr(mf, 'retry_state', retry)
    monkeypatch.setattr(ks, '_latest_minute_datetime_by_symbol', lambda *a, **kw: {})
    monkeypatch.setattr(ks, '_closed_minute_gap_starts', lambda *a, **kw: {})
    monkeypatch.setattr(ks, '_minute_increment_groups', lambda *a, **kw: [])
    monkeypatch.setattr(ks, 'resolve_limit', lambda *a, **kw: SimpleNamespace(batch=50, rpm=None))
    ks.sync_and_persist_minute(['000701'], SimpleNamespace(store=SimpleNamespace(data_dir=root)), Mock(), days=1)
    retry.assert_called_once_with(root, include_local=expected)


def test_tdx_transport_error_is_not_empty(monkeypatch):
    import requests

    from app.plugins.tdx.provider import TdxProvider
    provider = TdxProvider()
    monkeypatch.setattr(provider, '_request', Mock(side_effect=requests.ReadTimeout('private URL')))
    frame, issues = provider.get_minute_diagnostics(
        ['000001'], datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15))
    assert frame.is_empty()
    assert issues[-1]['reason'] == 'request_timeout'
    assert 'private' not in str(issues)


@pytest.mark.parametrize('bad', ['empty', 'stale', 'price', 'amount'])
def test_tdx_probe_requires_fresh_valid_bars(monkeypatch, bad):
    from app.plugins.tdx.provider import TdxProvider
    provider = TdxProvider()
    frame = bars('000001')
    if bad == 'empty':
        frame = pl.DataFrame()
    elif bad == 'stale':
        frame = bars('000001', DAY - timedelta(days=1))
    elif bad == 'price':
        frame = frame.with_columns(pl.lit(0.).alias('close'))
    else:
        frame = frame.with_columns(pl.lit(1.).alias('amount'))
    monkeypatch.setattr(provider, 'get_minute_diagnostics', Mock(return_value=(frame, [])))
    assert not provider.probe_minute(datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15))['usable']


def test_tdx_probe_accepts_actual_valid_bars(monkeypatch):
    from app.plugins.tdx.provider import TdxProvider
    provider = TdxProvider()
    monkeypatch.setattr(provider, 'get_minute_diagnostics', Mock(return_value=(bars('000001'), [])))
    assert provider.probe_minute(datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15))['usable']


def test_local_gap_keeps_original_source_evidence():
    from app.services.daily_maintenance import annotate_minute_failures
    failures = [{'symbol': '000001', 'date': '2026-09-07', 'reason': 'source_empty'}]
    issues = [dict(symbol='000001', date='2026-09-07', source='promax', reason='not_requested',
                   cause='source_pending', http_status=202, retry_after_seconds=180),
              dict(symbol='000001', date='2026-09-07', source='tdx', reason='backup_unavailable', cause='source_empty')]
    annotate_minute_failures(failures, issues)
    assert failures[0]['reason'] == 'source_pending'
    assert failures[0]['data_reason'] == 'source_empty'
    assert failures[0]['source_errors'][0]['http_status'] == 202
    assert failures[0]['source_errors'][1]['cause'] == 'source_empty'


def test_pending_source_queue_retries_same_batch_after_cooldown(tmp_path, monkeypatch):
    from test_maintenance_queue import NOW as QUEUE_NOW
    from test_maintenance_queue import runner

    from app.services import maintenance_queue as queue
    monkeypatch.setattr(queue, 'cn_now', lambda: QUEUE_NOW)
    codes = [f'{i:06}' for i in range(100)]
    queue.enqueue(tmp_path, 'minute', codes)
    result = {'completed_items': [], 'error': 'source_pending', 'source_retry': True,
              'retry_after_seconds': 600, 'source_errors': [{'reason': 'source_pending', 'http_status': 202}]}
    queue.tick(tmp_path, runner, lambda *a: result, now=QUEUE_NOW)
    row = queue.status(tmp_path)[0]
    assert row['source_errors'][0]['http_status'] == 202
    queue.recover(tmp_path)
    assert not queue.tick(tmp_path, runner, lambda *a: pytest.fail('cooldown'), now=QUEUE_NOW + timedelta(seconds=599))
    called = []
    queue.tick(tmp_path, runner, lambda t, p: called.extend(t['items']), now=QUEUE_NOW + timedelta(seconds=601))
    assert called == codes[:50]


def test_partial_download_keeps_accepted_data_and_marks_rest_unrequested(monkeypatch):
    import pandas as pd
    calls = []
    def fetch(**kw):
        calls.append(kw['ts_code'])
        if len(calls) > 1:
            raise TushareMinutePendingError(180)
        return pd.DataFrame([{'trade_time': '2025-07-03 10:00:00', 'open': 10., 'high': 10.,
                              'low': 10., 'close': 10., 'vol': 100., 'amount': 1000.}])
    provider = TushareProvider()
    monkeypatch.setattr(provider, '_ensure_api', lambda: SimpleNamespace(stk_mins=fetch))
    frame, issues = provider.get_minute_diagnostics([f'{i:06}' for i in range(10)],
        datetime(2025, 7, 3, 9, 30), datetime(2025, 7, 3, 15))
    assert frame.height == 1 and len(calls) == 4
    assert len([i for i in issues if i['reason'] == 'not_requested']) == 6


def test_maintenance_keeps_valid_local_stock_when_source_fails(setup, monkeypatch):
    from app.jobs import daily_pipeline
    from app.services import daily_maintenance, minute_repair
    root, _ = setup
    path = root / f'kline_minute/date={DAY}/part.parquet'
    path.parent.mkdir(parents=True)
    bars().write_parquet(path)
    before = path.read_bytes()
    issues = [dict(source='promax', reason='source_pending', retryable=True, http_status=202, retry_after_seconds=180),
              dict(source='tdx', reason='backup_unavailable', retryable=True, retry_after_seconds=180)]
    for issue in issues:
        mf._save_source(root, issue['source'], issue)
    def fail(*args, **kwargs):
        exc = mf.MinuteSourcesUnavailableError(issues)
        exc.written_rows = 240
        raise exc
    monkeypatch.setattr(minute_repair, 'repair_minute_gaps', fail)
    monkeypatch.setattr(daily_pipeline, '_invalidate', lambda *a: None)
    result = daily_maintenance.execute(SimpleNamespace(store=SimpleNamespace(data_dir=root)), None,
        {'id': 'source-recovery-test', 'kind': 'minute', 'label': 'test', 'items': ['000701', '000757'],
         'options': {'as_of': '2026-09-07T15:00', 'days': 0}}, lambda *a, **kw: None)
    assert result['completed_items'] == ['000701']
    assert result['minute_rows'] == 240
    assert result['validation_failures'][0]['reason'] == 'source_pending'
    assert result['source_retry'] and result['retry_after_seconds'] >= 180
    assert path.read_bytes() == before
