"""A rejected SSE preflight must not strand the same request after recovery."""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from datetime import date
from types import SimpleNamespace

import pytest

from app.api import backtest as api
from app.config import settings
from app.tickflow.capabilities import Cap, CapabilitySet


def _request(capabilities, repo):
    async def connected():
        return False

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(capabilities=capabilities, repo=repo)),
        is_disconnected=connected,
    )


def _read(response):
    async def consume():
        return ''.join([chunk async for chunk in response.body_iterator])

    return asyncio.run(consume())


@pytest.mark.parametrize('rejection', ['capability', 'minute_range', 'server_range'])
def test_rejected_request_starts_once_after_preflight_recovers(monkeypatch, rejection):
    from app.backtest import worker
    from app.services.heavy_job_limiter import shared_heavy_job_limiter

    jobs = {}
    monkeypatch.setattr(api, '_running_jobs', jobs)
    monkeypatch.setattr(settings, 'backtest_range_guard', rejection == 'server_range')
    capset = CapabilitySet()
    if rejection != 'capability':
        capset.grant(Cap.KLINE_MINUTE_BATCH)
    earliest = [date(2026, 10, 1) if rejection == 'minute_range' else date(2024, 1, 1)]
    repo = SimpleNamespace(earliest_minute_date=lambda: earliest[0])
    request = _request(capset, repo)
    arguments = dict(request=request, strategy_id='retry_preflight_fixture',
                     start='2026-01-01' if rejection == 'server_range' else '2026-09-21',
                     end='2026-10-08', minute_fill=True, symbols='000001.SZ')
    response = asyncio.run(api.strategy_stream(**arguments))
    error = _read(response)
    assert 'event: error' in error
    assert not jobs, 'Rejected preflight cached an unstarted task; identical retry will hang'

    calls = []

    class ImmediateThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

    class NoopTimer:
        daemon = False

        def __init__(self, interval, callback):
            pass

        def start(self):
            pass

    def run(task, progress, cancel_event):
        calls.append(task)
        return {'trades': [], 'stats': {'total_trades': 0}}

    monkeypatch.setattr(api.threading, 'Thread', ImmediateThread)
    monkeypatch.setattr(api.threading, 'Timer', NoopTimer)
    monkeypatch.setattr(worker, 'make_worker_task', lambda kind, data_dir, cfg: cfg)
    monkeypatch.setattr(worker, 'run_worker_task', run)
    monkeypatch.setattr(shared_heavy_job_limiter, 'slot', lambda *args, **kwargs: nullcontext())
    capset.grant(Cap.KLINE_MINUTE_BATCH)
    earliest[0] = date(2024, 1, 1)
    monkeypatch.setattr(settings, 'backtest_range_guard', False)

    first = _read(asyncio.run(api.strategy_stream(**arguments)))
    reconnect = _read(asyncio.run(api.strategy_stream(**arguments)))
    assert 'event: done' in first
    assert 'event: done' in reconnect
    assert len(calls) == 1
    assert calls[0].minute_fill is True
    assert calls[0].symbols == ['000001.SZ']


def test_preflight_rejection_preserves_an_existing_running_job(monkeypatch):
    jobs = {}
    monkeypatch.setattr(api, '_running_jobs', jobs)
    monkeypatch.setattr(settings, 'backtest_range_guard', False)
    request = _request(CapabilitySet(), SimpleNamespace(
        earliest_minute_date=lambda: date(2024, 1, 1)))
    key = api._make_job_key(
        'existing_preflight_fixture', None, '2026-09-21', '2026-10-08',
        'open_t+1', None, None, 0.0002, 5.0, 10, 1.0, 1_000_000.0,
        'equal', None, None, minute_fill=True,
    )
    job = api._BacktestJob(key)
    jobs[key] = job
    response = asyncio.run(api.strategy_stream(
        request=request, strategy_id='existing_preflight_fixture',
        start='2026-09-21', end='2026-10-08', minute_fill=True))
    assert 'event: error' in _read(response)
    assert jobs == {key: job}
    assert not job.done
    assert not job.cancel_event.is_set()
