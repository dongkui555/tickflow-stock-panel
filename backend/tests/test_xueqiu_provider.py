from datetime import datetime

import polars as pl
import pytest

from app.config import settings
from app.data_providers import public_market
from app.plugins.xueqiu.provider import XueqiuProvider, _parse_kline_payload, _parse_quote_rows


def test_quote_rows_require_explicit_exchange_and_normalize_units():
    rows = _parse_quote_rows(
        {
            "data": [
                {
                    "symbol": "SH600000",
                    "current": 10.5,
                    "percent": 2.0,
                    "chg": 0.2,
                    "timestamp": 1790146800000,
                    "volume": 123400,
                    "amount": 1295700,
                    "open": 10.3,
                    "last_close": 10.3,
                    "high": 10.6,
                    "low": 10.2,
                },
                {"symbol": "SH000001", "current": 3000},
                {"symbol": "920001", "current": 10.0},
            ]
        },
        {"600000.SH": "stock", "000001.SH": "index", "920001.BJ": "stock"},
    )
    assert len(rows) == 1
    assert rows[0]["symbol"] == "600000.SH"
    assert rows[0]["change_pct"] == pytest.approx(0.02)
    assert rows[0]["volume"] == pytest.approx(123400)
    assert rows[0]["amount"] == pytest.approx(1295700)
    assert rows[0]["source"] == "xueqiu"


def test_kline_payload_maps_columns_and_filters_date_range():
    payload = {
        "data": {
            "symbol": "SH600000",
            "column": ["timestamp", "volume", "open", "high", "low", "close", "amount"],
            "item": [
                [1790054400000, 100, 10, 11, 9, 10.5, 1050],
                [1790140800000, 200, 10.5, 11.5, 10, 11, 2200],
            ],
        }
    }
    frame = _parse_kline_payload(
        payload,
        "600000.SH",
        start=datetime(2026, 9, 22),
        end=datetime(2026, 9, 23),
    )
    assert frame.columns == ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    assert frame.height == 2
    assert frame["symbol"].unique().to_list() == ["600000.SH"]
    assert frame["volume"].to_list() == [100.0, 200.0]


def test_provider_does_not_claim_unsupported_minute_or_moneyflow():
    provider = XueqiuProvider()
    assert set(provider.config.datasets) == {"realtime", "daily"}
    assert not provider.history_auth_configured()
    assert provider.get_minute(["600000.SH"], None, None).is_empty()
    assert provider.get_realtime(symbols=[]).is_empty()
    provider.close()


def test_xueqiu_is_last_realtime_fallback(monkeypatch):
    now = datetime(2026, 9, 9, 10, 30, tzinfo=public_market.CN_TZ)
    row = {
        "symbol": "600000.SH", "name": "浦发银行", "source": "xueqiu", "asset_type": "stock",
        "open": 10.0, "high": 11.0, "low": 9.0, "last_price": 10.5, "prev_close": 10.0,
        "volume": 1000.0, "amount": 10400.0, "timestamp": now.timestamp() * 1000,
    }
    chain = public_market.PublicMarketBackup()
    calls = []

    def fetch(source, assets):
        calls.append(source)
        return [row] if source == "xueqiu" else []

    monkeypatch.setattr(settings, "public_market_fallback_enabled", True)
    monkeypatch.setattr(chain, "_request", fetch)
    result = chain.fetch({"600000.SH": "stock"}, now=now)
    assert result and result[0]["source"] == "xueqiu"
    assert calls[-1] == "xueqiu"
    assert calls == ["tencent", "sina", "eastmoney", "xueqiu"]
