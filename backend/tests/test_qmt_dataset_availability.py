from types import SimpleNamespace

import pytest
from app.data_providers.custom import loader
from app.plugins.qmt import bridge
from app.plugins.qmt.provider import QmtProvider


@pytest.fixture
def qmt_probe(monkeypatch):
    monkeypatch.setenv("QMT_ENABLED", "1")
    monkeypatch.setenv("QMT_BASE_URL", "http://qmt-test:8186")
    monkeypatch.setattr(bridge, "_confirmed_capabilities", {}, raising=False)

    def probe(caps):
        calls = []

        def get(url, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": caps})

        monkeypatch.setattr(bridge.requests, "get", get)
        return calls

    return probe


def test_history_only_is_available_without_live_datasets(qmt_probe):
    calls = qmt_probe({"minute": True, "realtime": False, "depth5": False, "symbol_routing": False})
    assert bridge.availability()[0]
    assert calls[0][0].endswith("/api/capabilities")
    assert calls[0][1]["params"] == {"refresh": "1"}
    assert set(QmtProvider().config.datasets) == {"minute"}


def test_live_and_auction_are_independently_declared(qmt_probe):
    qmt_probe({"minute": True, "realtime": True, "depth5": False,
               "symbol_routing": True, "push_auction": True})
    assert bridge.availability()[0]
    assert set(QmtProvider().config.datasets) == {"minute", "realtime", "auction"}


def test_failed_probe_clears_previous_dataset_snapshot(qmt_probe, monkeypatch):
    qmt_probe({"minute": True})
    assert bridge.availability()[0]

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(bridge.requests, "get", fail)
    assert not bridge.availability()[0]
    assert not QmtProvider().config.datasets


def test_disabled_qmt_has_no_runtime_datasets(qmt_probe, monkeypatch):
    qmt_probe({"minute": True})
    assert bridge.availability()[0]
    monkeypatch.setenv("QMT_ENABLED", "0")
    assert not bridge.availability()[0]
    assert not QmtProvider().config.datasets


def test_loader_and_provider_route_same_available_datasets(qmt_probe, monkeypatch):
    qmt_probe({"minute": True, "realtime": False, "depth5": False, "symbol_routing": False})
    monkeypatch.setattr(loader, "_PLUGIN_STATUS", {})
    monkeypatch.setattr(loader, "_PROVIDERS", {})
    loader._register_one_plugin({"name": "qmt", "entry": "app.plugins.qmt.provider:QmtProvider",
                                 "check": "app.plugins.qmt.bridge:availability",
                                 "datasets": ["minute", "realtime", "depth5", "auction"]})
    assert loader.list_plugins()[0]["datasets"] == ["minute"]
    assert loader.provider_has_dataset("qmt", "minute")
    assert not loader.provider_has_dataset("qmt", "realtime")
    assert not loader.provider_has_dataset("qmt", "depth5")


def test_runtime_provider_cannot_expand_manifest_datasets(monkeypatch):
    provider = SimpleNamespace(config=SimpleNamespace(datasets={"minute": None, "financial": None}))
    monkeypatch.setattr(loader, "_PLUGIN_STATUS", {})
    monkeypatch.setattr(loader, "_PROVIDERS", {})
    monkeypatch.setattr(loader, "_load_entry", lambda ref: provider)
    loader._register_one_plugin({"name": "example", "entry": "x:y", "datasets": ["minute", "realtime"]})
    assert loader.list_plugins()[0]["datasets"] == ["minute"]
    assert not loader.provider_has_dataset("example", "financial")
