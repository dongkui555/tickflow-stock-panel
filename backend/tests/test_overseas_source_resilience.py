import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from app.services import aiagents_sources as sources


def service(tmp_path):
    return sources.OverseasMarketData(str(tmp_path / "overseas.db"))


@pytest.fixture
def yahoo_session(monkeypatch):
    """Use a fake transport alongside the fake optional yfinance module."""
    class Session:
        def __init__(self, *, impersonate):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, **kwargs):
            return self.request("GET", url, **kwargs)

        def request(self, *args, **kwargs):
            pytest.fail("Unexpected HTTP request")

    monkeypatch.setitem(sys.modules, "curl_cffi.requests", SimpleNamespace(Session=Session))
    return Session


def test_failed_refresh_has_cooldown_even_when_forced(tmp_path, monkeypatch):
    svc = service(tmp_path)
    calls = []
    monkeypatch.setattr(sources.time, "monotonic", lambda: 100.)

    def fail(instruments):
        calls.append(instruments)
        raise TimeoutError("upstream unavailable")

    monkeypatch.setattr(svc, "_fetch_quotes", fail)
    assert svc.get_market_snapshot()["cache_fallback"]
    assert svc.get_market_snapshot(force_refresh=True)["cache_fallback"]
    assert len(calls) == 1
    assert svc.get_market_snapshot()["data_success"] is False


def test_failed_refresh_retries_after_cooldown_and_recovers(tmp_path, monkeypatch):
    svc = service(tmp_path)
    clock = [100.]
    monkeypatch.setattr(sources.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(svc, "_fetch_quotes", lambda _: [])
    assert svc.get_market_snapshot()["cache_fallback"]
    clock[0] += svc.QUOTE_TTL_MINUTES * 60 + 1
    monkeypatch.setattr(svc, "_fetch_quotes", lambda instruments: [{
        "symbol": item["symbol"], "price": 100., "previous_close": 99.,
        "quote_time": "2026-10-20 16:00:00"} for item in instruments])
    result = svc.get_market_snapshot()
    assert result["cache_fallback"] is False
    assert result["quotes"][0]["quote_time"] == "2026-10-20 16:00:00"


def test_parallel_refresh_uses_existing_cache_instead_of_duplicate_network(tmp_path, monkeypatch):
    svc = service(tmp_path)
    assert svc._quote_refresh_lock.acquire(blocking=False)
    monkeypatch.setattr(svc, "_fetch_quotes", lambda _: (_ for _ in ()).throw(AssertionError("duplicate fetch")))
    try:
        result = svc.get_market_snapshot(force_refresh=True)
        assert result["cache_fallback"] and not result["data_success"]
    finally:
        svc._quote_refresh_lock.release()


def test_yahoo_failure_stops_after_small_probe_batch(tmp_path, monkeypatch, yahoo_session):
    svc = service(tmp_path)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(sources, "_optional_import", lambda _: SimpleNamespace(download=download))
    assert svc._get_multi_ticker_history([str(i) for i in range(30)]).empty
    assert len(calls) == 1
    assert len(calls[0]["tickers"]) <= 2
    assert calls[0]["timeout"] <= 4
    assert calls[0]["threads"] is False


def test_yahoo_success_fetches_remaining_symbols_with_bounded_concurrency(tmp_path, monkeypatch, yahoo_session):
    svc = service(tmp_path)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return pd.concat({symbol: pd.DataFrame({"Close": [99., 100.]}) for symbol in kwargs["tickers"]}, axis=1)

    monkeypatch.setattr(sources, "_optional_import", lambda _: SimpleNamespace(download=download))
    frame = svc._get_multi_ticker_history([str(i) for i in range(30)])
    assert set(frame.columns.get_level_values(0)) == {str(i) for i in range(30)}
    assert len(calls) == 2
    assert calls[1]["threads"] == 4
    assert calls[1]["timeout"] <= 4


def test_quote_time_comes_from_last_valid_bar(tmp_path, monkeypatch):
    svc = service(tmp_path)
    data = pd.DataFrame({"Close": [99., 100., float("nan")]}, index=pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20"]))
    monkeypatch.setattr(svc, "_get_multi_ticker_history", lambda *a, **kw: data)
    quote = svc._fetch_quotes([svc.INSTRUMENTS[0]])[0]
    assert quote["quote_time"] == "2026-10-19T00:00:00"
    assert quote["change_pct"] == 1.0101


def test_yahoo_metadata_requests_obey_shared_deadline(tmp_path, monkeypatch, yahoo_session):
    svc = service(tmp_path)
    clock = [100.]
    monkeypatch.setattr(sources.time, "monotonic", lambda: clock[0])
    requests = []
    monkeypatch.setattr(yahoo_session, "request", lambda self, *a, **kw: requests.append(kw) or SimpleNamespace())

    def download(**kwargs):
        session = kwargs["session"]
        session.get("https://example.test/metadata", timeout=10)
        clock[0] += 29
        session.get("https://example.test/history", timeout=10)
        clock[0] += 2
        with pytest.raises(TimeoutError):
            session.get("https://example.test/late", timeout=10)
        return pd.DataFrame()

    monkeypatch.setattr(sources, "_optional_import", lambda _: SimpleNamespace(download=download))
    assert svc._get_multi_ticker_history(["SPY"]).empty
    assert [request["timeout"] for request in requests] == [4, 1]
