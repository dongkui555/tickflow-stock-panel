from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from app.plugins.tushare.provider import TushareProvider


@pytest.mark.parametrize("market,method", [("o", "stk_auction_o"), ("c", "stk_auction_c")])
@pytest.mark.parametrize("connection_mode", ["official", "promax"])
@pytest.mark.parametrize("extra,expected", [({}, 17000), ({"auction_vol": 18000}, 18000),
                                             ({"volume": 19000}, 19000)])
def test_auction_volume_alias_is_preserved_once(monkeypatch, market, method, connection_mode, extra, expected):
    source = pd.DataFrame([{
        "ts_code": "000001.SZ", "trade_date": "20260904",
        "open": 10., "high": 10., "low": 10., "close": 10.,
        "vol": "17000", "amount": 170000., **extra,
    }])
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return source.copy()

    provider = TushareProvider()
    monkeypatch.setattr(provider, "_connection_mode", lambda: connection_mode)
    monkeypatch.setattr(provider, "_ensure_api", lambda: SimpleNamespace(**{method: fetch}))
    day = datetime(2026, 9, 4)
    frame = provider.get_auction(["000001"], day, day, market=market, trade_dates=[day.date()])
    expected_params = {"trade_date": "20260904"}
    if connection_mode == "promax":
        expected_params.update(limit=6000, offset=0)
    assert calls == [expected_params]
    assert frame.height == 1
    assert frame["symbol"][0] == "000001"
    assert frame["volume"][0] == expected
    assert frame["amount"][0] == 170000.
    assert "vol" not in frame.columns
    assert source["vol"].to_list() == ["17000"]
