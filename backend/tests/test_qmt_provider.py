"""QMT provider normalization and routing tests."""
from __future__ import annotations

import datetime as dt

from app.plugins.qmt.provider import QmtProvider, _qmt_symbol


def test_qmt_routes_bse_920_codes_without_exchange_suffix():
    assert _qmt_symbol("920000") == "920000.BJ"
    assert _qmt_symbol("920001.BJ") == "920001.BJ"
    assert _qmt_symbol("600000") == "600000.SH"


def test_qmt_minute_maps_60m_and_keeps_lot_units(monkeypatch):
    provider = QmtProvider()

    def fake_request(path, params):
        assert path == "/api/kline"
        assert params["freq"] == "60m"
        return [{
            "symbol": "000001.SZ",
            "datetime": "2026-09-10 10:30:00",
            "open": 11.68,
            "high": 11.82,
            "low": 11.66,
            "close": 11.77,
            "volume": 336770,
            "amount": 395668864,
        }]

    monkeypatch.setattr(provider, "_request", fake_request)
    frame = provider.get_minute(
        ["000001.SZ"], dt.datetime(2026, 9, 10, 9, 30), dt.datetime(2026, 9, 10, 15), freq="1h",
    )
    assert frame.height == 1
    assert frame["volume"][0] == 336_770
    assert frame["amount"][0] == 395_668_864


def test_qmt_realtime_keeps_five_level_order_book(monkeypatch):
    provider = QmtProvider()

    def fake_request(path, params):
        assert path == "/api/quote"
        return [{
            "symbol": "000001.SZ",
            "datetime": "2026-09-10 14:59:58",
            "lastPrice": 11.85,
            "lastClose": 11.70,
            "open": 11.68,
            "high": 11.86,
            "low": 11.66,
            "pvolume": 1_000_000,
            "amount": 1_185_000_000,
            "cumulativeVolume": 1_100_000,
            "cumulativeAmount": 1_300_000_000,
            "bidPrice": [11.84, 11.83, 11.82, 11.81, 11.80],
            "askPrice": [11.85, 11.86, 11.87, 11.88, 11.89],
            "bidVol": [128, 1648, 1782, 1103, 3429],
            "askVol": [1371, 8386, 2215, 4716, 9045],
        }]

    monkeypatch.setattr(provider, "_request", fake_request)
    row = provider.get_realtime(symbols=["000001.SZ"]).to_dicts()[0]
    assert row["source"] == "qmt"
    assert row["bid_price_1"] == 11.84
    assert row["ask_price_5"] == 11.89
    assert row["bid_volume_1"] == 128
    assert row["ask_volume_5"] == 9045
    assert row["volume"] == 1_100_000
    assert row["amount"] == 1_300_000_000
    assert abs(row["change_pct"] - (0.15 / 11.70)) < 1e-10

    depth = provider.get_depth5(["000001.SZ"])["000001.SZ"]
    assert depth["bid_prices"][0] == 11.84
    assert depth["ask_volumes"][4] == 9045
    assert depth["source"] == "qmt"


def test_qmt_index_daily_adapter(monkeypatch):
    provider = QmtProvider()

    def fake_request(path, params):
        assert path == "/api/kline"
        assert params["freq"] == "1d"
        return [{
            "symbol": "000001.SH", "datetime": "2026-09-10 00:00:00",
            "open": 3800.0, "high": 3820.0, "low": 3790.0, "close": 3810.0,
            "volume": 123.0, "amount": 456.0,
        }]

    monkeypatch.setattr(provider, "_request", fake_request)
    frame = provider.get_asset_daily(
        ["000001.SH"], dt.datetime(2026, 9, 10), dt.datetime(2026, 9, 10, 23, 59), "index",
    )
    assert frame.height == 1
    assert frame["date"][0] == dt.date(2026, 9, 10)
    assert frame["symbol"][0] == "000001.SH"


def test_qmt_auction_omits_oversized_symbol_query(monkeypatch):
    provider = QmtProvider()

    def fake_request(path, params):
        assert path == "/api/auction"
        assert params == {"date": "2026-09-10"}
        return [{
            "symbol": "000001.SZ", "date": "2026-09-10", "market": "open",
            "open": 11.8, "high": 11.8, "low": 11.8, "close": 11.8,
            "volume": 1000, "amount": 1180000,
        }]

    monkeypatch.setattr(provider, "_request", fake_request)
    symbols = [f"{index:06d}.SZ" for index in range(1, 102)]
    frame = provider.get_auction(
        symbols, dt.datetime(2026, 9, 10), dt.datetime(2026, 9, 10, 9, 25),
        trade_dates=[dt.date(2026, 9, 10)],
    )
    assert frame.height == 1
    assert frame["symbol"][0] == "000001"
