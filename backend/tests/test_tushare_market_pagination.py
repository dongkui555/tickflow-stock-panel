from types import SimpleNamespace

import pandas as pd
import pytest

from app.plugins.tushare.provider import TushareProvider


def rows(codes, day="20260904"):
    return pd.DataFrame({"ts_code": codes, "trade_date": [day] * len(codes),
                         "net_mf_amount": [1.5] * len(codes)})


def test_promax_moneyflow_requests_explicit_market_page(monkeypatch):
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_connection_mode", lambda: "promax")
    calls = []
    source = rows([f"{n:06d}.SZ" for n in range(5190)])

    def fetch(**kwargs):
        calls.append(kwargs)
        return source.copy() if kwargs.get("limit") == 6000 else source.head(60).copy()

    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(moneyflow=fetch))
    result = provider._moneyflow_by_trade_dates([], "20260904", "20260904", trade_dates=["20260904"])
    assert calls == [{"trade_date": "20260904", "limit": 6000, "offset": 0}]
    assert result.height == 5190
    assert result["net_mf_amount"].unique().to_list() == [15000.0]
    assert source["net_mf_amount"].iloc[0] == 1.5


def test_promax_daily_fetches_beyond_gateway_5000_row_cap(monkeypatch):
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_connection_mode", lambda: "promax")
    calls = []
    source = pd.DataFrame({
        "ts_code": [f"{n:06d}.SZ" for n in range(5001)],
        "trade_date": ["20240701"] * 5001,
        "open": [10.0] * 5001,
        "high": [10.5] * 5001,
        "low": [9.5] * 5001,
        "close": [10.2] * 5001,
        "vol": [100.0] * 5001,
        "amount": [102.0] * 5001,
    })

    def fetch(**kwargs):
        calls.append(kwargs)
        offset = kwargs["offset"]
        return source.iloc[offset:offset + kwargs["limit"]].copy()

    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(daily=fetch))
    result = provider._daily_by_trade_dates([], "20240701", "20240701", trade_dates=["20240701"])
    assert result.height == 5001
    assert result.filter(result["symbol"] == "005000").height == 1
    assert [call["offset"] for call in calls] == [0, 5000]
    assert result["volume"][0] == 10000.0


@pytest.mark.parametrize("market,method", [("o", "stk_auction_o"), ("c", "stk_auction_c")])
def test_promax_auction_requests_explicit_market_page(monkeypatch, market, method):
    provider = TushareProvider()
    monkeypatch.setattr(provider, "_connection_mode", lambda: "promax")
    calls = []
    source = rows([f"{n:06d}.SZ" for n in range(5350)]).assign(vol=100., amount=1000.)

    def fetch(**kwargs):
        calls.append(kwargs)
        return source.copy() if kwargs.get("limit") == 6000 else source.head(1).copy()

    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(**{method: fetch}))
    result = provider._auction_by_trade_dates([], "20260904", "20260904", market=market,
                                              trade_dates=["20260904"])
    assert calls == [{"trade_date": "20260904", "limit": 6000, "offset": 0}]
    assert result.height == 5350
    assert result["volume"].to_list() == [100.] * 5350
    assert result["amount"].to_list() == [1000.] * 5350


def test_pagination_reads_next_page_and_stops_on_short_page():
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return rows(["000001.SZ", "000002.SZ"] if kwargs["offset"] == 0 else ["600000.SH"])

    result = TushareProvider._promax_market_day(fetch, "20260904", page_size=2)
    assert len(result) == 3
    assert [c["offset"] for c in calls] == [0, 2]


def test_pagination_rejects_gateway_ignoring_offset():
    with pytest.raises(ValueError, match="repeated"):
        TushareProvider._promax_market_day(lambda **kw: rows(["000001.SZ", "000002.SZ"]),
                                          "20260904", page_size=2)


def test_pagination_deduplicates_identical_rows_inside_page():
    result = TushareProvider._promax_market_day(lambda **kw: rows(["000001.SZ", "000001.SZ"]), "20260904")
    assert len(result) == 1


def test_pagination_rejects_conflicting_rows_inside_page():
    source = rows(["000001.SZ", "000001.SZ"])
    source.loc[1, "net_mf_amount"] = 7.5
    with pytest.raises(ValueError, match="repeated"):
        TushareProvider._promax_market_day(lambda **kw: source, "20260904")


def test_pagination_uses_raw_count_before_deduplication():
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return rows(["000001.SZ", "000001.SZ"] if kwargs["offset"] == 0 else ["600000.SH"])

    result = TushareProvider._promax_market_day(fetch, "20260904", page_size=2)
    assert len(result) == 2
    assert [c["offset"] for c in calls] == [0, 2]


def test_pagination_rejects_another_trading_date():
    with pytest.raises(ValueError, match="date"):
        TushareProvider._promax_market_day(lambda **kw: rows(["000001.SZ"], "20260903"), "20260904")


@pytest.mark.parametrize("day", [20260904, "2026-09-04", "20260904"])
def test_pagination_accepts_gateway_date_representations(day):
    result = TushareProvider._promax_market_day(lambda **kw: rows(["000001.SZ"], day), "20260904")
    assert len(result) == 1


def test_failed_second_page_does_not_return_partial_data():
    def fetch(**kwargs):
        if kwargs["offset"]:
            raise TimeoutError("upstream")
        return rows(["000001.SZ", "000002.SZ"])

    with pytest.raises(TimeoutError):
        TushareProvider._promax_market_day(fetch, "20260904", page_size=2)


def test_pagination_is_bounded_when_endpoint_keeps_returning_full_pages():
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return rows([f"{kwargs['offset']:06d}.SZ"])

    with pytest.raises(ValueError, match="page limit"):
        TushareProvider._promax_market_day(fetch, "20260904", page_size=1, max_pages=3)
    assert len(calls) == 3
