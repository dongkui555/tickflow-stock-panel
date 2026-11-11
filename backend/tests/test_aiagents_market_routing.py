from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from app.services import aiagents_bridge, aiagents_sources


@pytest.mark.parametrize("symbol,canonical,external", [
    ("920371", "920371.BJ", "bj920371"),
    ("920371.BJ", "920371.BJ", "bj920371"),
    ("833266", "833266.BJ", "bj833266"),
    ("430047", "430047.BJ", "bj430047"),
    ("600000.SH", "600000.SH", "sh600000"),
    ("000001.SZ", "000001.SZ", "sz000001"),
    ("300001", "300001.SZ", "sz300001"),
])
def test_market_identity_across_legacy_adapters(symbol, canonical, external):
    assert aiagents_sources._tickflow_symbol(symbol) == canonical
    assert aiagents_bridge._tickflow_symbol(symbol) == canonical
    assert aiagents_sources.LocalDataSourceManager._convert_to_ts_code(symbol) == canonical
    assert aiagents_sources.LocalDataSourceManager._convert_to_tx_code(symbol) == external


def test_beijing_minute_fallback_uses_beijing_symbol(monkeypatch):
    calls = []

    def unavailable(**kwargs):
        raise ConnectionError("primary unavailable")

    def fallback(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"day": ["2026-09-04 09:31:00"], "open": [7.33],
                             "high": [7.4], "low": [7.3], "close": [7.35],
                             "volume": [1000], "amount": [7350]})

    monkeypatch.setattr(aiagents_sources, "_RUNTIME_REPO", None)
    monkeypatch.setattr(aiagents_sources, "_optional_import", lambda name: SimpleNamespace(
        stock_zh_a_hist_min_em=unavailable, stock_zh_a_minute=fallback))
    manager = object.__new__(aiagents_sources.LocalDataSourceManager)
    frame = manager.get_stock_minute_data("920371.BJ", "2026-09-04", "2026-09-04", period="1")
    assert calls == [{"symbol": "bj920371", "period": "1", "adjust": ""}]
    assert frame["data_source"].tolist() == ["akshare_sina_minute"]
    assert frame["volume"].tolist() == [1000]


def test_beijing_local_minute_lookup_uses_canonical_market(monkeypatch):
    calls = []

    def minute_range(symbols, *args, **kwargs):
        calls.append(symbols)
        return pl.DataFrame()

    monkeypatch.setattr(aiagents_sources, "_RUNTIME_REPO", SimpleNamespace(get_minute_range=minute_range))
    manager = object.__new__(aiagents_sources.LocalDataSourceManager)
    assert manager._repo_minute("920371", "2026-09-04", "2026-09-04").empty
    assert calls == [["920371.BJ"]]


def test_ai_minute_aggregation_shares_session_alignment(monkeypatch):
    from datetime import datetime
    from test_minute_timeframes import session
    from app import market_time
    monkeypatch.setattr(market_time, "cn_now", lambda: datetime(2026, 9, 4, 15, 1))
    frame = session().rename({"datetime": "bar_time"}).to_pandas()
    result = aiagents_bridge._resample_minute(frame, 60)
    assert result["bar_time"].dt.strftime("%H:%M").tolist() == ["10:30", "11:30", "14:00", "15:00"]
    assert result["volume"].sum() == frame["volume"].sum()
