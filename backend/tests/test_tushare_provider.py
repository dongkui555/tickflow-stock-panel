import sys
from datetime import datetime
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from app.plugins.tushare.provider import (
    TushareMinuteRateLimitError,
    TushareMinuteSourceError,
    TushareProvider,
    _normalize_minute_frame,
    _PromaxTushareAPI,
    _to_ts_code,
)


@pytest.mark.parametrize("symbol", ["920006", "920006.BJ", "430047", "830799.BJ"])
def test_beijing_symbols_keep_exchange(symbol):
    assert _to_ts_code(symbol) == symbol.split(".")[0] + ".BJ"


def test_calendar_filters_gateway_closed_days_and_out_of_range(monkeypatch):
    provider = TushareProvider()
    api = SimpleNamespace(trade_cal=lambda **kw: pd.DataFrame({
        "cal_date": ["20260903", "20260904", "20260905", "20260906", "20260907"],
        "is_open": [1, 1, 0, 0, 1],
    }))
    monkeypatch.setattr(provider, "_ensure_api", lambda: api)
    assert provider._trade_dates("20260904", "20260906") == ["20260904"]
    assert provider._trade_dates("20260905", "20260906") == []


def test_status_queries_expose_trade_calendar_and_historical_suspensions(monkeypatch):
    provider = TushareProvider()
    api = SimpleNamespace(
        trade_cal=lambda **kw: pd.DataFrame({
            "cal_date": ["20261001", "20261002"], "is_open": [0, 1],
        }),
        suspend_d=lambda **kw: pd.DataFrame({
            "ts_code": ["000001.SZ"], "suspend_date": ["20261002"],
            "resume_date": ["20261003"], "suspend_reason": ["重大事项"],
        }),
    )
    monkeypatch.setattr(provider, "_ensure_api", lambda: api)

    calendar = provider.get_trade_calendar("20261001", "20261002")
    suspensions = provider.get_historical_suspensions("20261001", "20261003")

    assert calendar == [
        {"trade_date": "20261001", "is_open": 0},
        {"trade_date": "20261002", "is_open": 1},
    ]
    assert suspensions[0]["symbol"] == "000001.SZ"
    assert suspensions[0]["suspend_date"] == "20261002"


def test_financial_announcement_gateway_cannot_ignore_date(monkeypatch):
    from datetime import date
    provider = TushareProvider()
    api = SimpleNamespace(income=lambda **kw: pd.DataFrame({
        "ts_code": ["000001.SZ"], "ann_date": ["20260902"], "end_date": ["20260630"],
    }))
    monkeypatch.setattr(provider, "_ensure_api", lambda: api)
    with pytest.raises(ValueError, match="announcement date"):
        provider.get_financial_updates("income", date(2026, 9, 3))


def test_financial_announcements_preserve_stock_identity(monkeypatch):
    from datetime import date
    provider = TushareProvider()
    api = SimpleNamespace(income=lambda **kw: pd.DataFrame({
        "ts_code": ["000001.SZ", "600000.SH"], "ann_date": ["20260903"] * 2,
        "end_date": ["20260630"] * 2, "revenue": [100.0, 200.0],
    }))
    monkeypatch.setattr(provider, "_ensure_api", lambda: api)
    df = provider.get_financial_updates("income", date(2026, 9, 3))
    assert df["symbol"].to_list() == ["000001", "600000"]
    assert df["announce_date"].to_list() == [date(2026, 9, 3)] * 2


@pytest.mark.parametrize("actual, expected", [
    ("20260806", "2026-08-06"),
    ("20260401", "2026-04-29"),
    (None, "2026-04-29"),
    ("", "2026-04-29"),
    ("invalid-date", None),
])
def test_financial_restatement_not_available_before_actual_announcement(actual, expected):
    source = pd.DataFrame({"ann_date": [20260429], "f_ann_date": [actual],
                           "end_date": [20251231], "n_cashflow_act": [100.0]})
    result = TushareProvider()._normalize_financial_frame(source, "cash_flow", "000001")
    value = result["announce_date"][0]
    assert (str(value) if value is not None else None) == expected
    assert str(result["period_end"][0]) == "2025-12-31"
    assert "announce_date" not in source.columns


def test_financial_restatement_keeps_different_announcement_versions():
    source = pd.DataFrame({"ann_date": ["20260429"] * 2,
                           "f_ann_date": ["20260429", "20260806"],
                           "end_date": ["20251231"] * 2, "n_cashflow_act": [100.0, 80.0]})
    result = TushareProvider()._normalize_financial_frame(source, "cash_flow", "000001")
    assert [str(d) for d in result["announce_date"]] == ["2026-04-29", "2026-08-06"]
    assert result["net_operating_cash_flow"].to_list() == [100.0, 80.0]


def test_official_client_can_read_http_url_from_settings(monkeypatch):
    from app.config import settings

    provider = TushareProvider()
    api = SimpleNamespace()
    monkeypatch.setattr(provider, "_connection_mode", lambda: "official")
    monkeypatch.setattr(provider, "_tushare_token", lambda: "test-only-token")
    monkeypatch.setenv("TUSHARE_HTTP_URL", "")
    monkeypatch.setattr(settings, "tushare_http_url", "https://example.test/tushare")
    monkeypatch.setitem(sys.modules, "tushare", SimpleNamespace(set_token=lambda token: None, pro_api=lambda: api))
    assert provider._ensure_api() is api
    assert api._DataApi__http_url == "https://example.test/tushare"


def test_minute_shares_convert_once_and_opening_record_is_preserved():
    source = pd.DataFrame({
        "ts_code": ["920006.BJ"] * 5,
        "trade_time": ["2026-09-03 " + t for t in ["09:25:00", "09:30:00", "09:31:00", "15:00:00", "15:01:00"]],
        "open": [10.] * 5, "high": [10.] * 5, "low": [10.] * 5, "close": [10.] * 5,
        "vol": [100.] * 5, "amount": [1000.] * 5,
    })
    result = _normalize_minute_frame(source)
    assert result["datetime"].dt.strftime("%H:%M").to_list() == ["09:30", "09:31", "15:00"]
    assert result["volume"].to_list() == [1., 1., 1.]
    assert result["amount"].to_list() == [1000.] * 3
    assert source["vol"].tolist() == [100.] * 5


def test_mixed_timestamp_precision_is_not_silently_dropped():
    source = pd.DataFrame({
        "ts_code": ["920006.BJ"] * 2,
        "trade_time": ["2026-09-03 09:30:00", "2026-09-03 09:31:00.000"],
        "open": [10.] * 2, "high": [10.] * 2, "low": [10.] * 2, "close": [10.] * 2,
        "vol": [100.] * 2, "amount": [1000.] * 2,
    })
    assert _normalize_minute_frame(source).height == 2


def test_rate_limit_is_not_returned_as_empty_data(monkeypatch):
    def fail(**kwargs):
        response = httpx.Response(429, request=httpx.Request("GET", "https://example.test/stk_mins"))
        response.raise_for_status()
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(stk_mins=fail))
    with pytest.raises(TushareMinuteRateLimitError, match="429"):
        provider.get_minute(["920006"], datetime(2026, 5, 8), datetime(2026, 5, 8, 15))


def test_network_error_and_genuine_empty_are_distinguished(monkeypatch):
    provider = TushareProvider()
    def fail(**kwargs):
        raise httpx.ConnectError("failed")
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(stk_mins=fail))
    with pytest.raises(TushareMinuteSourceError):
        provider.get_minute(["000001"], datetime(2026, 5, 8), datetime(2026, 5, 8, 15))
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(stk_mins=lambda **k: pd.DataFrame()))
    assert provider.get_minute(["000001"], datetime(2026, 5, 8), datetime(2026, 5, 8, 15)).is_empty()


def test_gateway_rate_limit_retries_are_bounded_and_respect_retry_after(monkeypatch):
    calls, waits = [], []
    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"code": 0, "data": [{"vol": 100}]})
    api = _PromaxTushareAPI("https://example.test", "test-key")
    api.close()
    api._client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("app.plugins.tushare.provider.time.sleep", waits.append)
    try:
        assert len(api.stk_mins(ts_code="920006.BJ")) == 1
        assert len(calls) == 2
        assert waits[0] > 0.9
        api._client.close()
        calls.clear()
        def limited(request):
            calls.append(request)
            return httpx.Response(429, headers={"Retry-After": "0"})
        api._client = httpx.Client(transport=httpx.MockTransport(limited))
        with pytest.raises(httpx.HTTPStatusError):
            api.stk_mins(ts_code="920006.BJ")
        assert len(calls) == 4
    finally:
        api.close()


def test_long_retry_after_blocks_further_requests_without_waiting_an_hour():
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "3600"})
    api = _PromaxTushareAPI("https://example.test", "test-key")
    api.close()
    api._client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            api.stk_mins(ts_code="920006.BJ")
        with pytest.raises(TushareMinuteRateLimitError, match="cooldown"):
            api.stk_mins(ts_code="920006.BJ")
        assert len(calls) == 1
    finally:
        api.close()


def test_shares_use_daily_basic_batch_snapshot(monkeypatch):
    class FakeApi:
        def __init__(self):
            self.daily_basic_calls = []

        def trade_cal(self, **kwargs):
            return pd.DataFrame({"cal_date": ["20260901"]})

        def daily_basic(self, **kwargs):
            self.daily_basic_calls.append(kwargs)
            return pd.DataFrame({
                "ts_code": ["000001.SZ", "300413.SZ", "600000.SH"],
                "trade_date": [kwargs["trade_date"]] * 3,
                "total_share": [1_940_591.8198, 187_072.0815, 293_520.80397],
                "float_share": [1_940_568.4991, 102_169.9958, 293_520.80397],
            })

    fake = FakeApi()
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_ensure_api", lambda: fake)

    df = provider.get_financials(
        "shares",
        ["000001", "300413"],
        latest_only=True,
        start_time=datetime(2026, 9, 1),
        end_time=datetime(2026, 9, 1),
    )

    assert len(fake.daily_basic_calls) == 1
    assert set(df["symbol"].to_list()) == {"000001", "300413"}
    by_symbol = {row["symbol"]: row for row in df.to_dicts()}
    assert by_symbol["000001"]["period_end"].isoformat() == "2026-09-01"
    assert by_symbol["000001"]["float_shares"] == pytest.approx(19_405_684_991.0)
    assert by_symbol["300413"]["total_shares"] == pytest.approx(1_870_720_815.0)
# Scheduled shares must never fall back to thousands of per-stock calls.
def test_incremental_shares_use_market_batch_only(monkeypatch):
    from app.plugins.tushare.provider import TushareProvider
    import polars as pl
    from datetime import date

    provider = TushareProvider()
    calls = []
    def batch(symbols, start, end, **kw):
        calls.append((symbols, start, end, kw))
        return pl.DataFrame()
    monkeypatch.setattr(provider, "_shares_from_daily_basic_by_trade_dates", batch)
    monkeypatch.setattr(provider, "_fetch_financial_frame", lambda *a, **kw: pytest.fail("per-stock fallback"))
    assert provider.get_financial_updates("shares", date(2026, 9, 3)).is_empty()
    assert len(calls) == 1
    assert calls[0][:3] == ([], "20260903", "20260903")
