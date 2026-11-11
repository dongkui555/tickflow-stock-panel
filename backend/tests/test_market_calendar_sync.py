"""Calendar refresh must preserve history and reject invented/partial days."""
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from app.market_time import CN_TZ
from app.services import market_calendar_sync as sync
from app.services.market_status import market_day_status

NOW = datetime(2026, 10, 22, 17, 0, tzinfo=CN_TZ)


def seed(root):
    path = root / 'market_calendar/trade_calendar.parquet'
    path.parent.mkdir(parents=True)
    pl.DataFrame({'trade_date': ['20260921', '20260922'], 'is_open': [1, 1]}).write_parquet(path)
    return path


def rows(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return [{'trade_date': str(first + timedelta(days=i)),
             'is_open': int((first + timedelta(days=i)).weekday() < 5)}
            for i in range((last - first).days + 1)]


def test_refresh_fills_gap_preserves_history_and_avoids_repeat(tmp_path):
    path = seed(tmp_path)
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        result = rows(str(date(2026, 9, 23)), str(date(2026, 12, 6)))
        # A holiday on a weekday must remain closed, not become a weekday guess.
        return [{**r, 'is_open': 0} if r['trade_date'] == '2026-10-01' else r for r in result]

    provider = SimpleNamespace(get_trade_calendar=fetch)
    result = sync.refresh(tmp_path, now=NOW, provider=provider)
    assert result['status'] == 'succeeded'
    assert calls == [('20260923', '20261206')]
    assert pl.read_parquet(path).head(2).to_dicts() == [
        {'trade_date': '20260921', 'is_open': 1}, {'trade_date': '20260922', 'is_open': 1}]
    assert market_day_status(tmp_path, '20261001')['status'] == 'market_closed'
    assert market_day_status(tmp_path, '20261023')['status'] == 'open'
    assert sync.refresh(tmp_path, now=NOW, provider=provider)['status'] == 'current'
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['empty', 'missing_day', 'conflict', 'invalid_open', 'out_of_range', 'exception'])
def test_bad_source_never_overwrites_previous_and_obeys_backoff(tmp_path, kind):
    path = seed(tmp_path)
    before = path.read_bytes()
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        result = rows('2026-09-23', '2026-12-06')
        if kind == 'exception':
            raise TimeoutError('not suitable for printing: credentials')
        if kind == 'empty':
            return []
        if kind == 'missing_day':
            return result[1:]
        if kind == 'conflict':
            return [*result, {**result[0], 'is_open': 1-result[0]['is_open']}]
        if kind == 'invalid_open':
            result[0]['is_open'] = None
        if kind == 'out_of_range':
            result.append({'trade_date': '20270101', 'is_open': 1})
        return result

    provider = SimpleNamespace(get_trade_calendar=fetch)
    result = sync.refresh(tmp_path, now=NOW, provider=provider)
    assert result['status'] == 'retry'
    assert 'credentials' not in str(result)
    assert path.read_bytes() == before
    assert sync.refresh(tmp_path, now=NOW+timedelta(minutes=1), provider=provider)['status'] == 'backoff'
    assert len(calls) == 1
    assert sync.refresh(tmp_path, now=NOW+timedelta(minutes=31), provider=provider)['status'] == 'retry'
    assert len(calls) == 2


def test_unsupported_source_does_not_invent_calendar(tmp_path):
    path = seed(tmp_path)
    before = path.read_bytes()
    result = sync.refresh(tmp_path, now=NOW, provider=SimpleNamespace())
    assert result['status'] == 'retry'
    assert result['error_type'] == 'NotImplementedError'
    assert path.read_bytes() == before


def test_internal_gap_is_not_hidden_by_later_cached_dates(tmp_path):
    path = seed(tmp_path)
    future = pl.DataFrame({'trade_date': ['20261231'], 'is_open': [1]})
    pl.concat([pl.read_parquet(path), future]).write_parquet(path)
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return rows('2026-10-15', '2026-12-06')

    assert sync.refresh(tmp_path, now=NOW, provider=SimpleNamespace(get_trade_calendar=fetch))['status'] == 'succeeded'
    assert calls == [('20261015', '20261206')]
    assert pl.read_parquet(path)['trade_date'].max() == '20261231'


def test_calendar_scheduler_runs_through_shared_execution_slot(tmp_path, monkeypatch):
    from app.jobs import daily_pipeline
    from app.services import preferences

    class Scheduler:
        def __init__(self, **kwargs):
            self.jobs = {}

        def add_job(self, fn, **kwargs):
            self.jobs[kwargs['id']] = (fn, kwargs)

        def start(self):
            pass

    calls = []
    monkeypatch.setattr(daily_pipeline, 'AsyncIOScheduler', Scheduler)
    monkeypatch.setattr(preferences, 'load', lambda: {})
    monkeypatch.setattr(sync, 'refresh', lambda root: calls.append(root) or {'status': 'current'})
    monkeypatch.setattr(daily_pipeline, '_run_tracked', lambda task, name: calls.append(name) or task())
    scheduler = daily_pipeline.start_scheduler(SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)), None)
    fn, opts = scheduler.jobs['market_calendar_sync']
    assert opts['max_instances'] == 1 and opts['coalesce'] is True
    fn()
    assert calls == ['market_calendar_sync', tmp_path]
