"""本地 TDX HTTP 数据源的归一化与路由契约测试。"""
from __future__ import annotations

import datetime as dt

import polars as pl

from app.plugins.tdx.provider import TdxProvider


def test_tdx_lunch_close_label_is_normalized_before_range_filter(monkeypatch):
    provider = TdxProvider()
    stamps = [dt.datetime(2026, 9, 9, 9, 31) + dt.timedelta(minutes=i) for i in range(119)]
    stamps += [dt.datetime(2026, 9, 9, 13)]
    raw = [dict(Time=t.isoformat(), Open=10000, High=10000, Low=10000,
                Close=10000, Volume=i + 1, Amount=(i + 1) * 1000000) for i, t in enumerate(stamps)]
    monkeypatch.setattr(provider, "_fetch_kline", lambda *a, **kw: raw)
    frame = provider.get_minute(["000600"], dt.datetime(2026, 9, 9, 11, 29),
                                dt.datetime(2026, 9, 9, 11, 30))
    assert frame["datetime"].to_list() == [dt.datetime(2026, 9, 9, 11, 29), dt.datetime(2026, 9, 9, 11, 30)]
    assert frame["volume"].to_list() == [119, 120]
    assert raw[-1]["Time"] == "2026-09-09T13:00:00"
    # A different convention, duplicate lunch close, or an incomplete sample is not remapped.
    raw.append(dict(raw[-1], Time="2026-09-09T11:30:00"))
    frame = provider.get_minute(["000600"], None, dt.datetime(2026, 9, 9, 13))
    assert dt.datetime(2026, 9, 9, 13) in frame["datetime"].to_list()
    raw[:] = raw[:1] + raw[-2:-1]
    frame = provider.get_minute(["000600"], None, dt.datetime(2026, 9, 9, 13))
    assert dt.datetime(2026, 9, 9, 13) in frame["datetime"].to_list()


def test_tdx_daily_and_minute_normalize_legacy_units(monkeypatch):
    provider = TdxProvider()

    def fake_request(path, params):
        assert path == "/api/kline"
        return {
            "List": [
                {
                    "Time": "2026-08-27T09:35:00",
                    "Open": 12340,
                    "High": 12500,
                    "Low": 12200,
                    "Close": 12450,
                    "Volume": 321,
                    "Amount": 3_999_000,
                },
            ],
        }

    monkeypatch.setattr(provider, "_request", fake_request)

    minute = provider.get_minute(
        ["000001.SZ"],
        dt.datetime(2026, 8, 27, 9, 30),
        dt.datetime(2026, 8, 27, 10, 0),
        freq="1m",
    )
    assert minute.columns == ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
    assert minute.height == 1
    assert minute["symbol"][0] == "000001.SZ"
    assert minute["close"][0] == 12.45
    assert minute["volume"][0] == 321
    assert minute["amount"][0] == 3999.0
    assert minute.schema["datetime"] == pl.Datetime

    daily = provider.get_daily(
        ["000001.SZ"],
        dt.datetime(2026, 8, 27),
        dt.datetime(2026, 8, 27, 23, 59),
    )
    assert daily.columns == ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    assert daily.height == 1
    assert daily["date"][0] == dt.date(2026, 8, 27)
    assert daily.schema["date"] == pl.Date


def test_tdx_realtime_returns_tickflow_schema(monkeypatch):
    provider = TdxProvider()

    def fake_request(path, params):
        assert path == "/api/quotes"
        assert params == {"codes": "000001.SZ"}
        return [{
            "Code": "000001",
            "Market": 0,
            "Name": "平安银行",
            "TotalHand": 1000,
            "Amount": 12_500_000,
            "ServerTime": 1_788_000_000,
            "K": {
                "Close": 12500,
                "Last": 12375,
                "Open": 12400,
                "High": 12600,
                "Low": 12350,
            },
        }]

    monkeypatch.setattr(provider, "_request", fake_request)
    frame = provider.get_realtime(symbols=["000001.SZ"])
    assert frame.height == 1
    row = frame.to_dicts()[0]
    assert row["symbol"] == "000001.SZ"
    assert row["name"] == "平安银行"
    assert row["last_price"] == 12.5
    assert row["prev_close"] == 12.375
    assert row["volume"] == 1000
    assert row["amount"] == 12_500.0
    assert abs(row["change_pct"] - 0.0101010101) < 1e-8


def test_tdx_realtime_falls_back_to_single_quote(monkeypatch):
    provider = TdxProvider()

    def fake_request(path, params):
        if path == "/api/quotes":
            raise RuntimeError("old tdx api")
        assert path == "/api/quote"
        return [{
            "Name": "贵州茅台",
            "TotalHand": 10,
            "Amount": 1_500_000,
            "K": {
                "Close": 1500000,
                "Last": 1490000,
            },
        }]

    monkeypatch.setattr(provider, "_request", fake_request)
    frame = provider.get_realtime(symbols=["600519.SH"])
    assert frame.height == 1
    row = frame.to_dicts()[0]
    assert row["symbol"] == "600519.SH"
    assert row["last_price"] == 1500.0
