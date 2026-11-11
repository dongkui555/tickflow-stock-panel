from datetime import datetime

import httpx
import pytest

from app.plugins.tushare.provider import (
    TushareMinutePendingError,
    TushareMinuteSourceError,
    TushareProvider,
    _PromaxTushareAPI,
    minute_error_details,
)


def test_market_sync_stops_after_three_pending_symbols(monkeypatch):
    from types import SimpleNamespace

    calls = []
    def pending(**kwargs):
        calls.append(kwargs)
        raise TushareMinutePendingError(30)

    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(stk_mins=pending))
    with pytest.raises(TushareMinutePendingError):
        provider.get_minute(
            [f"600{i:03d}" for i in range(100)],
            datetime(2025, 7, 3, 9, 30), datetime(2025, 7, 3, 15),
        )
    assert len(calls) == 3


def test_pending_batch_is_not_retried_for_the_entire_market(monkeypatch):
    from types import SimpleNamespace

    from app.services import kline_sync

    calls = []
    def pending(symbols, **kwargs):
        calls.append(symbols)
        raise TushareMinutePendingError(30)

    monkeypatch.setattr(kline_sync, "_resolve_minute_provider",
                        lambda name: (SimpleNamespace(get_minute=pending), False, None))
    monkeypatch.setattr(kline_sync, "sleep_between_batches", lambda *args: None)
    with pytest.raises(TushareMinutePendingError):
        kline_sync._try_custom_minute_batched(
            [f"600{i:03d}" for i in range(100)],
            datetime(2025, 7, 3, 9, 30), datetime(2025, 7, 3, 15),
            "stock", batch_size=10, provider_name="tushare",
        )
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["timeout", "503"])
def test_repeated_transient_errors_stop_market_probe(monkeypatch, kind):
    from types import SimpleNamespace
    calls = []
    def unavailable(**kwargs):
        calls.append(kwargs)
        if kind == "timeout":
            raise httpx.ReadTimeout("temporary")
        response = httpx.Response(503, request=httpx.Request("GET", "https://example.test"))
        response.raise_for_status()
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(stk_mins=unavailable))
    with pytest.raises(TushareMinuteSourceError) as caught:
        provider.get_minute([f"600{i:03d}" for i in range(100)],
                            datetime(2025, 7, 3, 9, 30), datetime(2025, 7, 3, 15))
    assert caught.value.retryable is True
    assert len(calls) == 3


def test_successful_empty_response_breaks_transient_streak(monkeypatch):
    from types import SimpleNamespace

    import pandas as pd
    calls = []
    def sometimes_pending(**kwargs):
        calls.append(kwargs)
        if len(calls) in (1, 2, 4):
            raise TushareMinutePendingError(30)
        return pd.DataFrame()
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(stk_mins=sometimes_pending))
    with pytest.raises(TushareMinutePendingError):
        provider.get_minute([f"600{i:03d}" for i in range(5)],
                            datetime(2025, 7, 3, 9, 30), datetime(2025, 7, 3, 15))
    assert len(calls) == 5


def pending_api(monkeypatch, retry_after="30"):
    calls = []
    clock = [100.0]
    monkeypatch.setattr("app.plugins.tushare.provider.time.monotonic", lambda: clock[0])
    api = _PromaxTushareAPI("https://example.test", "secret-must-not-leak")
    api.close()

    def respond(request):
        calls.append(request)
        if len(calls) > 1 and clock[0] >= 130:
            return httpx.Response(200, json={"code": 0, "data": []})
        return httpx.Response(202, headers={"Retry-After": retry_after},
            json={"ok": False, "error": "secret-must-not-leak"})

    api._client = httpx.Client(transport=httpx.MockTransport(respond))
    return api, calls, clock


@pytest.mark.parametrize("endpoint", ["stk_mins", "rt_min_daily"])
def test_accepted_request_is_pending_not_empty(monkeypatch, endpoint):
    api, calls, _ = pending_api(monkeypatch)
    try:
        with pytest.raises(TushareMinuteSourceError) as caught:
            getattr(api, endpoint)(ts_code="920230.BJ")
        assert minute_error_details(caught.value) == {
            "reason": "source_pending", "retryable": True, "http_status": 202,
            "retry_after_seconds": 30,
        }
        assert len(calls) == 1
        assert "secret-must-not-leak" not in str(caught.value)
    finally:
        api.close()


def test_pending_cooldown_is_scoped_to_exact_query(monkeypatch):
    api, calls, clock = pending_api(monkeypatch)
    try:
        for _ in range(2):
            with pytest.raises(TushareMinuteSourceError):
                api.stk_mins(ts_code="920230.BJ", start_date="2025-07-03")
        assert len(calls) == 1
        with pytest.raises(TushareMinuteSourceError):
            api.stk_mins(ts_code="920230.BJ", start_date="2025-07-04")
        assert len(calls) == 2
        clock[0] = 131
        assert api.stk_mins(ts_code="920230.BJ", start_date="2025-07-03").empty
        assert len(calls) == 3
    finally:
        api.close()


@pytest.mark.parametrize("header,expected", [("", 30), ("bad", 30), ("nan", 30),
    ("inf", 30), ("-1", 30), ("0.5", 1), ("60", 60),
    ("Thu, 01 Jan 1970 00:01:00 GMT", 60)])
def test_pending_retry_after_is_finite_and_preserved(monkeypatch, header, expected):
    monkeypatch.setattr("app.plugins.tushare.provider.time.time", lambda: 0)
    api, _, _ = pending_api(monkeypatch, header)
    try:
        with pytest.raises(TushareMinuteSourceError) as caught:
            api.stk_mins(ts_code="920230.BJ")
        assert minute_error_details(caught.value)["retry_after_seconds"] == expected
    finally:
        api.close()


def test_pending_reason_survives_provider_diagnostics(monkeypatch):
    api, calls, _ = pending_api(monkeypatch)
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: api)
    try:
        frame, issues = provider.get_minute_diagnostics(
            ["920230.BJ"], datetime(2025, 7, 3, 9, 30), datetime(2025, 7, 3, 15)
        )
        assert frame.is_empty()
        assert len(calls) == 1
        assert issues[0]["reason"] == "source_pending"
        assert issues[0]["retry_after_seconds"] == 30
        assert issues[0]["http_status"] == 202
        assert issues[0]["retryable"] is True
        assert "secret-must-not-leak" not in str(issues)
    finally:
        api.close()
