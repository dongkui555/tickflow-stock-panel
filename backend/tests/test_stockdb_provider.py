from datetime import date, datetime, timedelta
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
import requests

from app.plugins.stockdb import provider as sd

DAY = date(2026, 9, 7)


def hour_payload(hour, code="000001"):
    start = datetime(2026, 9, 7, 9, 30)
    stamps = [start + timedelta(minutes=i) for i in range(121)]
    stamps += [datetime(2026, 9, 7, 13) + timedelta(minutes=i) for i in range(121)]
    result = []
    for stamp in stamps:
        if stamp.hour != hour:
            continue
        empty = stamp.hour == 13 and stamp.minute == 0
        raw = {"code": code, "date": int(stamp.strftime("%Y%m%d%H%M%S")),
               "open": 10., "high": 10., "low": 10., "close": 10.,
               "volume": 0 if empty else 100., "amount": 0 if empty else 1000.}
        result.append([f"分钟k:{code}:{raw['date']}", raw])
    return result


def day_payload(code="000001"):
    return [row for hour in (9, 10, 11, 13, 14, 15) for row in hour_payload(hour, code)]


def beijing_day_payload(code="920000"):
    rows = day_payload(code)
    for minute in range(1, 31):
        stamp = datetime(2026, 9, 7, 15, minute)
        raw = {"code": code, "date": int(stamp.strftime("%Y%m%d%H%M%S")),
               "open": 10., "high": 10., "low": 10., "close": 10.,
               "volume": 100., "amount": 1000.}
        rows.append([f"分钟k:{code}:{raw['date']}", raw])
    return rows


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(sd.settings, "stockdb_base_url", "http://127.0.0.1:7899")
    monkeypatch.setattr(sd.settings, "stockdb_max_stock_days", 20)
    monkeypatch.setattr(sd, "cn_now", lambda: datetime(2026, 9, 10, 9, 50, tzinfo=ZoneInfo("Asia/Shanghai")))
    client = sd.StockDBProvider()
    monkeypatch.setattr(client, "_request", lambda query: day_payload())
    yield client
    client.close()


def test_hour_identity_units_and_lunch():
    rows = sd.parse_hour(hour_payload(13), "000001.SZ", DAY, 13)
    assert len(rows) == 59
    assert rows[0]["datetime"] == datetime(2026, 9, 7, 13, 1)
    assert rows[0]["symbol"] == "000001.SZ"
    assert rows[0]["volume"] == 1 and rows[0]["amount"] == 1000


def test_whole_day_query_uses_one_local_request(provider, monkeypatch):
    request = Mock(return_value=day_payload())
    monkeypatch.setattr(provider, "_request", request)
    frame, issues = provider.get_minute_diagnostics(
        ["000001.SZ"], datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15))
    assert frame.height == 241
    request.assert_called_once_with("分钟k:000001:20260907*")
    assert issues[0]["reason"] == "accepted"


def test_beijing_day_query_keeps_after_hours_fixed_price_bars(monkeypatch):
    monkeypatch.setattr(sd.settings, "stockdb_base_url", "http://127.0.0.1:7899")
    monkeypatch.setattr(sd.settings, "stockdb_max_stock_days", 20)
    monkeypatch.setattr(sd, "cn_now", lambda: datetime(2026, 9, 10, 9, 50, tzinfo=ZoneInfo("Asia/Shanghai")))
    client = sd.StockDBProvider()
    try:
        monkeypatch.setattr(client, "_request", lambda query: beijing_day_payload())
        frame, issues = client.get_minute_diagnostics(
            ["920000.BJ"], datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15, 30)
        )
        assert frame.height == 271
        assert frame.filter(frame["datetime"].dt.hour() == 15).height == 31
        assert issues[0]["reason"] == "accepted"
    finally:
        client.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_plugin_manifest_registration_is_opt_in(monkeypatch, enabled):
    from app.data_providers.custom import loader

    monkeypatch.setattr(sd.settings, "stockdb_enabled", enabled)
    monkeypatch.setattr(sd.settings, "stockdb_base_url", "http://stockdb-api:7899")
    monkeypatch.setattr(loader, "_PROVIDERS", {})
    monkeypatch.setattr(loader, "_PLUGIN_STATUS", {})
    loader._register_one_plugin(loader.plugin_manifest("stockdb"))
    assert loader.provider_has_dataset("stockdb", "minute") is enabled
    assert loader.list_plugins()[0]["available"] is enabled
    if enabled:
        loader.get_provider("stockdb").close()


@pytest.mark.parametrize("url", ["http://8.138.149.215:7899", "http://example.com", "http://localhost.evil.test",
                               "http://user:secret@127.0.0.1", "http://127.0.0.1/?cmd=set", "file:///tmp/data",
                               "http://169.254.169.254", "http://0.0.0.0:7899"])
def test_public_credentials_and_metadata_addresses_rejected(url):
    with pytest.raises(ValueError):
        sd.local_url(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:7899", "http://host.docker.internal:7899", "http://stockdb-api:7899",
                               "http://192.168.1.10:7899", "http://[::1]:7899"])
def test_local_urls(url):
    assert sd.local_url(url) == url


@pytest.mark.parametrize("change,reason", [({"code": "600000"}, "source_symbol_conflict"),
                                         ({"date": 20260908093000}, "source_symbol_conflict"),
                                         ({"open": 11.}, "source_invalid"),
                                         ({"amount": float("nan")}, "source_invalid"),
                                         ({"volume": -100.}, "source_invalid")])
def test_invalid_records_fail_closed(change, reason):
    payload = hour_payload(9)
    payload[0][1].update(change)
    with pytest.raises(sd.StockDBError, match=reason):
        sd.parse_hour(payload, "000001", DAY, 9)


def test_nonzero_lunch_is_not_silently_discarded():
    payload = hour_payload(13)
    payload[0][1].update(volume=100., amount=1000.)
    with pytest.raises(sd.StockDBError, match="source_time_conflict"):
        sd.parse_hour(payload, "000001", DAY, 13)


def test_whole_day_structure_and_no_implicit_adjustment(provider):
    frame, issues = provider.get_minute_diagnostics(["000001"], datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15))
    assert frame.height == 241 and frame["volume"].sum() == 241
    assert frame["open"].unique().to_list() == [10.]
    assert issues[-1]["reason"] == "accepted"


def test_partial_hour_prevents_day_acceptance(provider, monkeypatch):
    monkeypatch.setattr(provider, "_request", lambda query: day_payload()[:-1])
    frame, issues = provider.get_minute_diagnostics(["000001"], datetime(2026, 9, 7, 9, 30), datetime(2026, 9, 7, 15))
    assert frame.is_empty() and issues[-1]["reason"] == "source_incomplete"


def test_today_etf_and_higher_periods_are_not_requested(provider, monkeypatch):
    monkeypatch.setattr(provider, "_request", lambda *args: pytest.fail("unsupported request"))
    for asset, freq in (("etf", "1m"), ("index", "1m"), ("stock", "5m")):
        frame, issues = provider.get_minute_diagnostics(["000001"], datetime(2026, 9, 7), datetime(2026, 9, 7, 15), asset, freq)
        assert frame.is_empty() and issues[-1]["reason"] == "unsupported_minute_scope"
    frame, issues = provider.get_minute_diagnostics(["000001"], datetime(2026, 9, 10), datetime(2026, 9, 10, 15))
    assert frame.is_empty() and issues[-1]["reason"] == "closed_history_only"


def test_network_outage_stops_batch(provider, monkeypatch):
    calls = []

    def failed(query):
        calls.append(query)
        raise requests.Timeout("secret is not exposed")

    monkeypatch.setattr(provider, "_request", failed)
    frame, issues = provider.get_minute_diagnostics(["000001", "600000"], datetime(2026, 9, 7), datetime(2026, 9, 7, 15))
    assert frame.is_empty() and len(calls) == 1
    assert issues[-1]["reason"] == "request_timeout" and issues[-1]["batch_aborted"]
    assert "secret" not in str(issues)


def test_http_contract_and_replay_through_local_socket(monkeypatch):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from urllib.parse import parse_qs, urlsplit

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            params = parse_qs(urlsplit(self.path).query)
            seen.append(params)
            body = json.dumps(hour_payload(9)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(sd.settings, "stockdb_base_url", f"http://127.0.0.1:{server.server_port}")
    client = sd.StockDBProvider()
    try:
        assert len(client._request("分钟k:000001:2026090709*")) == 30
        assert seen == [{"cmd": ["get"], "t": ["分钟k:000001:2026090709*"], "json": ["1"]}]
        assert not client._session.trust_env
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_no_million_request_scope(provider, monkeypatch):
    monkeypatch.setattr(provider, "_request", lambda *args: pytest.fail("unbounded request"))
    frame, issues = provider.get_minute_diagnostics(["000001"], datetime(2024, 1, 1), datetime(2026, 9, 7, 15))
    assert frame.is_empty() and issues[-1]["reason"] == "request_scope_too_large"
