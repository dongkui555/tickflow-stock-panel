from __future__ import annotations

import httpx
import pandas as pd
import pytest
from app.plugins.tushare.provider import DatahubTushareAPI, TushareAPIFailover
from app.services.tushare_catalog import TUSHARE_API_CATALOG


def test_catalog_contains_exactly_documented_80_endpoints():
    assert len(TUSHARE_API_CATALOG) == 80
    assert TUSHARE_API_CATALOG["daily"].http_supported is True
    assert TUSHARE_API_CATALOG["pro_bar"].http_supported is False
    assert TUSHARE_API_CATALOG["stk_nineturn"].category == "特色"


def test_datahub_client_decodes_tushare_fields_items_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "test-key"
        assert request.url.path.endswith("/daily")
        return httpx.Response(
            200,
            json={
                "code": 0,
                "msg": "",
                "data": {
                    "fields": ["ts_code", "trade_date", "close"],
                    "items": [["000001.SZ", "20260918", 12.34]],
                },
            },
        )

    client = DatahubTushareAPI("http://datahub.example/tushare", "test-key", verify=False)
    client._client.close()  # noqa: SLF001
    client._client = httpx.Client(transport=httpx.MockTransport(handler))  # noqa: SLF001
    try:
        frame = client.daily(ts_code="000001.SZ", trade_date="20260918")
    finally:
        client.close()

    assert frame.to_dict(orient="records") == [
        {"ts_code": "000001.SZ", "trade_date": "20260918", "close": 12.34}
    ]


def test_failover_uses_backup_only_when_primary_raises():
    class Primary:
        def daily(self, **_params):
            raise RuntimeError("offline")

    class Backup:
        def daily(self, **_params):
            return "backup-result"

    client = TushareAPIFailover(Primary(), Backup())
    assert client.daily(ts_code="000001.SZ") == "backup-result"
    assert client.last_source == "datahub"


def test_empty_auction_does_not_silently_switch_to_backup():
    class Primary:
        def stk_auction_o(self, **params):
            return pd.DataFrame()
    class Backup:
        def stk_auction_o(self, **params):
            raise AssertionError('An empty successful response must remain distinguishable')
    client = TushareAPIFailover(Primary(), Backup())
    assert client.stk_auction_o(trade_date='20261014').empty
    assert client.last_source == 'promax'


def test_non_auction_failures_keep_the_existing_exception_chain():
    class Primary:
        def daily(self, **params):
            raise RuntimeError('primary failed')
    failure = httpx.ConnectTimeout('unavailable')
    class Backup:
        def daily(self, **params):
            raise failure
    client = TushareAPIFailover(Primary(), Backup())
    with pytest.raises(RuntimeError) as exc:
        client.daily()
    assert exc.value.__cause__ is failure
