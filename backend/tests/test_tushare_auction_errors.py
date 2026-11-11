from datetime import datetime
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from app.plugins.tushare.provider import TushareProvider

DAY = datetime(2026, 9, 8)


def provider_with(monkeypatch, method):
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_connection_mode", lambda: "promax")
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(stk_auction_o=method))
    return provider


def test_empty_success_is_not_silently_swallowed_in_strict_sync(monkeypatch):
    provider = provider_with(monkeypatch, lambda **kw: pd.DataFrame())
    assert provider.get_auction(["000001"], DAY, DAY, trade_dates=[DAY]).is_empty()
    with pytest.raises(RuntimeError) as exc:
        provider.get_auction(["000001"], DAY, DAY, trade_dates=[DAY], strict=True)
    assert exc.value.reason == "upstream_empty"


def test_transport_failure_remains_distinguishable_without_leaking_url(monkeypatch, caplog):
    def fetch(**kwargs):
        raise httpx.ConnectTimeout("https://gateway/?token=secret-test-token")

    provider = provider_with(monkeypatch, fetch)
    with pytest.raises(RuntimeError) as exc:
        provider.get_auction(["000001"], DAY, DAY, trade_dates=[DAY], strict=True)
    assert exc.value.reason == "network_timeout"
    assert "secret-test-token" not in str(exc.value)
    assert "secret-test-token" not in caplog.text


def test_missing_stock_is_not_confused_with_whole_market_empty(monkeypatch):
    provider = provider_with(monkeypatch, lambda **kw: pd.DataFrame({
        "ts_code": ["600000.SH"], "trade_date": ["20260908"], "price": [10.0],
    }))
    with pytest.raises(RuntimeError) as exc:
        provider.get_auction(["000001"], DAY, DAY, trade_dates=[DAY], strict=True)
    assert exc.value.reason == "no_matching_rows"


def test_wrong_trading_date_is_an_error_not_empty_success(monkeypatch):
    provider = provider_with(monkeypatch, lambda **kw: pd.DataFrame({
        "ts_code": ["000001.SZ"], "trade_date": ["20260907"], "price": [10.0],
    }))
    with pytest.raises(RuntimeError) as exc:
        provider.get_auction(["000001"], DAY, DAY, trade_dates=[DAY], strict=True)
    assert exc.value.reason == "invalid_response"


def test_all_gateway_failures_keep_safe_source_reasons(monkeypatch, caplog):
    from app.plugins.tushare.provider import TushareAPIFailover

    def client(status):
        def fetch(**kwargs):
            request = httpx.Request('GET', 'https://gateway/?token=private-test-value')
            response = httpx.Response(status, request=request)
            response.raise_for_status()
        return SimpleNamespace(stk_auction_o=fetch)

    api = TushareAPIFailover(client(502), client(403))
    api.last_source = 'datahub'
    provider = provider_with(monkeypatch, api.stk_auction_o)
    with pytest.raises(RuntimeError) as exc:
        provider.get_auction(['000001'], DAY, DAY, trade_dates=[DAY], strict=True)
    assert exc.value.reason == 'all_sources_failed'
    assert exc.value.source_failures == [
        {'source': 'promax', 'reason': 'gateway_rejected'},
        {'source': 'datahub', 'reason': 'permission_denied'},
    ]
    assert api.last_source is None
    assert 'private-test-value' not in str(exc.value) + caplog.text
