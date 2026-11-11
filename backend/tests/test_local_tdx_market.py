import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from app.plugins.tdx.provider import _tickflow_symbol


@pytest.fixture
def bridge(monkeypatch):
    hq = ModuleType("pytdx.hq")
    hq.TdxHq_API = object
    monkeypatch.setitem(sys.modules, "pytdx", ModuleType("pytdx"))
    monkeypatch.setitem(sys.modules, "pytdx.hq", hq)
    path = Path(__file__).resolve().parents[2] / "tools/tdx_api/app.py"
    spec = importlib.util.spec_from_file_location("test_tdx_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("symbol,expected", [
    ("920000", 2), ("920002.BJ", 2), ("BJ920002", 2), ("833266", 2),
    ("430047.BJ", 2), ("600000.SH", 1), ("900901", 1), ("000001.SZ", 0), ("510300", 1),
])
def test_market_route(bridge, symbol, expected):
    assert bridge._market(symbol) == expected


@pytest.mark.parametrize("symbol,expected", [("920002", "920002.BJ"), ("833266", "833266.BJ"),
                                            ("900901", "900901.SH"), ("000001", "000001.SZ")])
def test_canonical_symbol_keeps_beijing_market(symbol, expected):
    assert _tickflow_symbol(symbol) == expected


def test_missing_quote_does_not_shift_stock_identity(bridge, monkeypatch):
    api = SimpleNamespace(get_security_quotes=lambda query: [{"market": 2, "code": "920002", "price": 52.56}],
                          disconnect=lambda: None)
    monkeypatch.setattr(bridge, "_connect", lambda: (api, "mock"))
    result = bridge.quotes("600000.SH,920002.BJ")
    assert len(result["data"]) == 1
    row = result["data"][0]
    assert (row["Code"], row["Market"], row["K"]["Close"]) == ("920002", 2, 52560.)


def test_kline_passes_market_two_to_protocol(bridge, monkeypatch):
    calls = []
    api = SimpleNamespace(get_security_bars=lambda *args: calls.append(args) or [], disconnect=lambda: None)
    monkeypatch.setattr(bridge, "_connect", lambda **kw: (api, "mock"))
    bridge.kline(code="920002.BJ", type="1m", limit=240)
    assert calls == [(7, 2, "920002", 0, 240)]


def test_connected_empty_node_is_skipped_and_cooled_down(bridge, monkeypatch):
    from unittest.mock import Mock
    bad = SimpleNamespace(connect=Mock(return_value=True), get_security_bars=Mock(return_value=None), disconnect=Mock())
    good = SimpleNamespace(connect=Mock(return_value=True), get_security_bars=Mock(return_value=[
        dict(datetime='2026-09-09 15:00', open=10., high=10., low=10., close=10.)]), disconnect=Mock())
    clock = [100.]
    monkeypatch.setattr(bridge.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(bridge, '_parse_hosts', lambda: [('empty', 7709), ('good', 7709)])
    factory = Mock(side_effect=[bad, good, good])
    monkeypatch.setattr(bridge, 'TdxHq_API', factory)
    assert bridge._connect(require_minute=True) == (good, 'good:7709')
    bad.disconnect.assert_called_once()
    assert bridge._connect(require_minute=True) == (good, 'good:7709')
    assert bad.connect.call_count == 1


def test_minute_probe_total_budget_is_bounded(bridge, monkeypatch):
    from unittest.mock import Mock
    clock = [100.]
    def connect(*a, **kw):
        clock[0] += kw['time_out']
        raise TimeoutError()
    api = SimpleNamespace(connect=Mock(side_effect=connect), disconnect=Mock())
    monkeypatch.setattr(bridge.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(bridge, '_parse_hosts', lambda: [(str(i), 7709) for i in range(10)])
    monkeypatch.setattr(bridge, 'TdxHq_API', lambda: api)
    with pytest.raises(RuntimeError, match='minute nodes unavailable'):
        bridge._connect(require_minute=True)
    assert clock[0] <= 104.01 and api.connect.call_count == 2
