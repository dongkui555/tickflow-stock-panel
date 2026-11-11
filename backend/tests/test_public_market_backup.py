from datetime import datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from app.config import settings
from app.data_providers import public_market as mod
from app.market_time import CN_TZ
from app.services.quote_service import QuoteService, _PrimaryQuotes

NOW = datetime(2026, 9, 9, 10, 30, tzinfo=CN_TZ)
SYM = "002918.SZ"


def quote(symbol=SYM, source="tencent"):
    return dict(symbol=symbol, name="example", source=source, open=10., high=11., low=9.,
                last_price=10.5, prev_close=10., volume=1000., amount=10400.,
                timestamp=NOW.timestamp() * 1000, asset_type="stock",
                bid_prices=[10.5, 10.4, 10.3, 10.2, 10.1], bid_volumes=[100] * 5,
                ask_prices=[10.6, 10.7, 10.8, 10.9, 11.], ask_volumes=[200] * 5)


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "public_market_fallback_enabled", True)
    monkeypatch.setattr(mod, "cn_now", lambda: NOW)


@pytest.mark.parametrize("count", [87, 88, 91])
def test_tencent_variable_tail_and_exact_units(count):
    f = [""] * count
    for i, value in {1: "example", 2: "920002", 3: "10.5", 4: "10", 5: "10", 6: "10",
                     30: "20260909103000", 33: "11", 34: "9", 35: "10.5/10/10400",
                     38: "2.5", 57: "1.04"}.items():
        f[i] = value
    for offset, price in ((9, 10.5), (19, 10.6)):
        for i in range(5):
            f[offset + i * 2] = str(price + (i if offset == 19 else -i) / 100)
            f[offset + i * 2 + 1] = "2"
    rows = mod.parse_tencent('v_bj920002="' + "~".join(f) + '";', {"920002.BJ": "stock"})
    row = rows[0]
    assert row["amount"] == 10400
    assert row["volume"] == 1000
    assert row["bid_volumes"] == [200] * 5
    assert row["turnover_rate"] == .025
    assert mod.valid_quote(row, NOW) and mod.valid_depth(row, NOW)


@pytest.mark.parametrize("count", [33, 34, 39])
def test_sina_tail_units_and_identity(count):
    f = ["0"] * count
    f[:10] = ["example", "10", "10", "10.5", "11", "9", "10.5", "10.6", "1000", "10400"]
    for offset in (10, 20):
        for i in range(5):
            f[offset + i * 2] = "100"
            f[offset + i * 2 + 1] = str(10.5 - i / 100 if offset == 10 else 10.6 + i / 100)
    f[30:32] = ["2026-09-09", "10:30:00"]
    text = 'var hq_str_sz002918="' + ",".join(f) + '";'
    row = mod.parse_sina(text, {SYM: "stock"})[0]
    assert row["volume"] == 1000 and row["amount"] == 10400
    assert row["bid_volumes"][0] == 100
    assert mod.parse_sina(text, {"000001.SZ": "stock"}) == []


def eastmoney():
    data = {"f57": "002918", "f43": 10.5, "f60": 10., "f46": 10., "f44": 11., "f45": 9.,
            "f47": 10, "f48": 10400, "f86": NOW.timestamp(), "f137": -100, "f140": 200, "f143": -300}
    for offset, base in ((19, 10.5), (39, 10.6)):
        for i in range(5):
            data[f"f{offset - i * 2}"] = base + (i if offset == 39 else -i) / 100
            data[f"f{offset - i * 2 + 1}"] = 2
    return {"data": data}


def test_eastmoney_depth_mapping_and_net_not_gross():
    row = mod.parse_eastmoney(eastmoney(), SYM, "stock")[0]
    assert row["bid_prices"][0] == 10.5 and row["ask_prices"][0] == 10.6
    assert row["bid_volumes"] == [200] * 5
    assert row["fund_flow_snapshot"]["kind"] == "vendor_net_only"
    assert "buy_lg_amount" not in row
    assert mod.valid_depth(row, NOW)
    assert mod.parse_eastmoney(eastmoney(), "600519.SH", "stock") == []


def test_missing_depth_is_not_zero_book():
    data = eastmoney()
    data["data"].pop("f19")
    row = mod.parse_eastmoney(data, SYM, "stock")[0]
    assert mod.valid_quote(row, NOW)
    assert not mod.valid_depth(row, NOW)
    row = quote()
    row["bid_prices"] = row["ask_prices"] = [0] * 5
    row["bid_volumes"] = row["ask_volumes"] = [0] * 5
    assert not mod.valid_depth(row, NOW)


def test_index_has_no_fake_volume_or_depth():
    row = mod.parse_eastmoney(eastmoney(), SYM, "index")[0]
    assert row["volume"] is None and row["amount"] is None
    assert not mod.valid_depth(row, NOW)


@pytest.mark.parametrize("change", [
    {"timestamp": None}, {"timestamp": (NOW - timedelta(days=1)).timestamp() * 1000},
    {"timestamp": (NOW - timedelta(minutes=4)).timestamp() * 1000},
    {"timestamp": (NOW + timedelta(minutes=2)).timestamp() * 1000},
    {"last_price": 0}, {"high": 10.1}, {"amount": float("nan")}, {"volume": 10},
])
def test_invalid_quotes_rejected(change):
    assert not mod.valid_quote(quote() | change, NOW)


def test_lunch_close_freshness():
    lunch = NOW.replace(hour=12)
    assert mod.fresh_timestamp(NOW.replace(hour=11, minute=30), lunch)
    assert not mod.fresh_timestamp(NOW, lunch)
    after_close = NOW.replace(hour=19)
    assert mod.fresh_timestamp(NOW.replace(hour=15, minute=0), after_close)
    assert not mod.fresh_timestamp(NOW.replace(hour=11, minute=30), after_close)


def test_failover_partial_cache_cooldown_and_recovery(monkeypatch):
    chain = mod.PublicMarketBackup()
    calls = []
    def fetch(source, assets):
        calls.append((source, list(assets)))
        if source == "tencent":
            raise TimeoutError()
        return [quote(s, source) for s in assets]
    monkeypatch.setattr(chain, "_request", fetch)
    assert chain.fetch({SYM: "stock"})[0]["source"] == "sina"
    chain.fetch({SYM: "stock"})
    assert [c[0] for c in calls] == ["tencent", "sina"]
    chain._health["tencent"]["retry_at"] = 0
    monkeypatch.setattr(chain, "_request", lambda source, assets: [quote(s, source) for s in assets])
    assert chain.fetch({SYM: "stock"})[0]["source"] == "tencent"


def test_only_missing_symbols_go_to_next_source(monkeypatch):
    chain = mod.PublicMarketBackup()
    calls = []
    second = "600519.SH"
    def fetch(source, assets):
        calls.append((source, list(assets)))
        return [quote(SYM, source)] if source == "tencent" else [quote(s, source) for s in assets]
    monkeypatch.setattr(chain, "_request", fetch)
    rows = chain.fetch({SYM: "stock", second: "stock"})
    assert len(rows) == 2
    assert calls == [("tencent", [SYM, second]), ("sina", [second])]


def test_all_fail_is_bounded_and_backoff(monkeypatch):
    chain = mod.PublicMarketBackup()
    calls = []
    def fetch(source, assets):
        calls.append(source)
        return []
    monkeypatch.setattr(chain, "_request", fetch)
    assert not chain.fetch({f"{i:06d}.SZ": "stock" for i in range(1000)})
    assert calls.count("eastmoney") <= 3
    batch_calls = (settings.public_market_fallback_max_symbols + 49) // 50
    for source in mod.SOURCES:
        assert calls.count(source) <= (3 if source == "eastmoney" else batch_calls)
    assert len(calls) <= 3 + batch_calls * (len(mod.SOURCES) - 1)
    assert chain.status()["last"]["realtime"]["missing"] == 1000
    calls.clear()
    chain.fetch({"000000.SZ": "stock"})
    assert not calls


@pytest.mark.parametrize("symbol,expected", [
    ("920002", "920002.BJ"), ("bj920002", "920002.BJ"),
    ("920176.SH", "920176.BJ"),
    ("sh000001", "000001.SH"), ("000001", "000001.SZ"),
    ("600519.SH", "600519.SH"), ("invalid", "INVALID"),
])
def test_provider_symbol_boundary(symbol, expected):
    assert mod.qualified_symbol(symbol) == expected


def test_invalid_symbol_does_not_disable_other_quotes_or_depth(monkeypatch):
    chain = mod.PublicMarketBackup()
    monkeypatch.setattr(chain, "_request", lambda source, assets: [])
    assert chain.fetch({"000000.SZ": "stock"}) == []
    assert chain.available("tencent")
    monkeypatch.setattr(chain, "_request", lambda source, assets: [quote(s, source) for s in assets])
    assert chain.fetch({SYM: "stock"}, capability="depth5")[0]["source"] == "tencent"


def test_full_market_rejections_merge_and_preserve_local_identity(monkeypatch):
    from app.services import preferences, quote_service
    chain = mod.PublicMarketBackup()
    monkeypatch.setattr(mod, "public_market_backup", chain)
    monkeypatch.setattr(quote_service, "cn_now", lambda: NOW)
    monkeypatch.setattr(preferences, "get_realtime_data_provider", lambda: "fuyao")
    monkeypatch.setattr(preferences, "get_realtime_pull_stock", lambda: True)
    service = QuoteService()
    codes = [f"92000{i}" for i in range(10)]
    service._repo = SimpleNamespace(get_instruments=lambda: pl.DataFrame({"symbol": codes}))
    monkeypatch.setattr(service, "_backup_assets", lambda: {"920002.BJ": "stock"})
    records = [quote(s) for s in [*codes, "920999"]]
    records[0]["last_price"] = 0
    monkeypatch.setattr(service, "_bounded_primary_quotes", lambda _: _PrimaryQuotes(records))
    published = []
    monkeypatch.setattr(service, "_process_full_market_records", lambda rows, **kw: published.append((rows, kw)))
    service._fetch_full_market_quotes()
    rows, flags = published[0]
    assert flags["partial"] is False and flags["merge_existing"] is True
    assert {r["symbol"] for r in rows} == {*codes[1:], "920999"}
    assert service.is_symbol_quote_current("920002")
    assert not service.is_symbol_quote_current("920000.BJ")
    assert "covered_symbols" not in service._backup_route


def test_quote_merge_never_truncates_partition(monkeypatch):
    from app.services import quote_service
    monkeypatch.setattr(quote_service, "cn_now", lambda: NOW)
    monkeypatch.setattr(quote_service, "cn_today", lambda: NOW.date())
    service = QuoteService()
    merged = []
    service._repo = SimpleNamespace(
        get_index_symbol_set=lambda: set(), get_etf_instruments=lambda: pl.DataFrame(),
        merge_live_daily_asset=lambda asset, frame: merged.append((asset, frame.height)),
    )
    enriched_flags = []
    monkeypatch.setattr(service, "_flush_live_enriched", lambda *args, **kw: enriched_flags.append(kw))
    monkeypatch.setattr(service, "_evaluate_monitors", lambda *args: None)
    service._process_full_market_records([quote()], t0=0, now_ts=0, merge_existing=True)
    assert merged == [("stock", 1)]
    assert enriched_flags[0]["merge"] is True and enriched_flags[0]["partial_cache"] is False


def test_depth_backup_keeps_local_six_digit_codes(monkeypatch):
    from app.services.depth_service import DepthService
    service = DepthService()
    monkeypatch.setattr(service, "_get_capset", lambda: SimpleNamespace(has=lambda cap: False))
    chain = mod.PublicMarketBackup()
    monkeypatch.setattr(mod, "public_market_backup", chain)
    monkeypatch.setattr(chain, "_request", lambda source, assets: [quote(s, source) for s in assets])
    rows = service._call_depth_batch(["920002", "002918"])
    assert set(rows) == {"920002", "002918"}
    assert rows["920002"]["symbol"] == "920002"


def test_quote_service_fallback_and_primary_recovery(monkeypatch):
    from app.services import preferences, quote_service
    chain = mod.PublicMarketBackup()
    monkeypatch.setattr(mod, "public_market_backup", chain)
    monkeypatch.setattr(quote_service, "cn_now", lambda: NOW)
    monkeypatch.setattr(preferences, "get_realtime_data_provider", lambda: "fuyao")
    monkeypatch.setattr(preferences, "get_realtime_pull_stock", lambda: True)
    service = QuoteService()
    service._repo = SimpleNamespace(get_instruments=lambda: pl.DataFrame({"symbol": [SYM]}))
    monkeypatch.setattr(service, "_backup_assets", lambda: {SYM: "stock"})
    def fail(_):
        raise TimeoutError()
    monkeypatch.setattr(service, "_bounded_primary_quotes", fail)
    monkeypatch.setattr(chain, "_request", lambda source, assets: [quote(s, source) for s in assets])
    published = []
    monkeypatch.setattr(service, "_process_full_market_records", lambda rows, **kw: published.append((rows, kw)))
    service._fetch_full_market_quotes()
    assert published[-1][1]["partial"] is True
    assert published[-1][0][0]["source"] == "tencent"
    assert service._backup_route["primary"] == "fuyao"
    chain._health["primary_realtime:fuyao"]["retry_at"] = 0
    monkeypatch.setattr(service, "_bounded_primary_quotes", lambda _: _PrimaryQuotes([quote()]))
    service._fetch_full_market_quotes()
    assert published[-1][1]["partial"] is False
    assert published[-1][0][0]["source"] == "fuyao"


def test_depth_without_tickflow_subscription(monkeypatch):
    from app.services.depth_service import DepthService
    service = DepthService()
    monkeypatch.setattr(service, "_get_capset", lambda: SimpleNamespace(has=lambda cap: False))
    chain = mod.PublicMarketBackup()
    monkeypatch.setattr(mod, "public_market_backup", chain)
    monkeypatch.setattr(chain, "_request", lambda source, assets: [quote(s, source) for s in assets])
    assert service._has_capability()
    assert service._call_depth_batch([SYM])[SYM]["source"] == "tencent"


def test_depth_backup_does_not_request_unavailable_primary(monkeypatch):
    from app.services import preferences
    from app.services.depth_service import DepthService

    service = DepthService()
    monkeypatch.setattr(preferences, "get_depth5_data_provider", lambda: "tickflow")
    monkeypatch.setattr(service, "_get_capset", lambda: SimpleNamespace(has=lambda cap: False))
    monkeypatch.setattr(service, "_call_primary_depth_batch", lambda symbols:
                        pytest.fail("unsubscribed primary was requested"))
    chain = mod.PublicMarketBackup()
    monkeypatch.setattr(mod, "public_market_backup", chain)
    monkeypatch.setattr(chain, "_request", lambda source, assets: [quote(s, source) for s in assets])
    assert service._call_depth_batch([SYM])[SYM]["source"] == "tencent"


def test_primary_pending_does_not_spawn_duplicate_work():
    service = QuoteService()
    class Pending:
        def done(self):
            return False
        def result(self, timeout):
            assert timeout == 8
            raise TimeoutError()
    service._primary_quote_future = Pending()
    service._primary_quote_name = "fuyao"
    for _ in range(3):
        with pytest.raises(TimeoutError):
            service._bounded_primary_quotes("fuyao")
    assert not hasattr(service, "_primary_quote_executor")


def test_stale_primary_and_all_backups_fail_publish_nothing(monkeypatch):
    from app.services import preferences, quote_service
    chain = mod.PublicMarketBackup()
    monkeypatch.setattr(mod, "public_market_backup", chain)
    monkeypatch.setattr(quote_service, "cn_now", lambda: NOW)
    monkeypatch.setattr(preferences, "get_realtime_data_provider", lambda: "fuyao")
    service = QuoteService()
    monkeypatch.setattr(service, "_backup_assets", lambda: {SYM: "stock"})
    stale = quote() | {"timestamp": (NOW - timedelta(days=1)).timestamp() * 1000}
    monkeypatch.setattr(service, "_bounded_primary_quotes", lambda _: _PrimaryQuotes([stale], partial=True))
    monkeypatch.setattr(chain, "_request", lambda source, assets: [])
    def no_publish(*args, **kwargs):
        pytest.fail("stale quotes were published")
    monkeypatch.setattr(service, "_process_full_market_records", no_publish)
    service._fetch_full_market_quotes()
    assert service._backup_route["rejected_primary"] == 1
    assert service._backup_route["priority_missing"] == 1
    assert service.get_enriched_today()[0].is_empty()


def test_truncated_full_market_response_is_not_persisted(monkeypatch):
    from app.services import preferences, quote_service
    chain = mod.PublicMarketBackup()
    monkeypatch.setattr(mod, "public_market_backup", chain)
    monkeypatch.setattr(quote_service, "cn_now", lambda: NOW)
    monkeypatch.setattr(preferences, "get_realtime_data_provider", lambda: "fuyao")
    monkeypatch.setattr(preferences, "get_realtime_pull_stock", lambda: True)
    service = QuoteService()
    service._repo = SimpleNamespace(get_instruments=lambda: pl.DataFrame({"symbol": range(100)}))
    monkeypatch.setattr(service, "_backup_assets", lambda: {SYM: "stock"})
    monkeypatch.setattr(service, "_bounded_primary_quotes", lambda _: _PrimaryQuotes([quote()]))
    published = []
    monkeypatch.setattr(service, "_process_full_market_records", lambda rows, **kw: published.append(kw))
    service._fetch_full_market_quotes()
    assert published[0]["partial"] is True


def test_depth_route_gate_uses_backup_without_granting_tickflow(monkeypatch):
    from app.api.settings import _require_depth_route
    calls = []
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        capabilities=SimpleNamespace(require=lambda cap: calls.append(cap)))))
    _require_depth_route(request)
    assert not calls
    monkeypatch.setattr(settings, "public_market_fallback_enabled", False)
    _require_depth_route(request)
    assert len(calls) == 1


def test_stale_depth_cache_is_pending_not_sealed(monkeypatch):
    from app.services import depth_service
    service = depth_service.DepthService()
    monkeypatch.setattr(depth_service, "cn_today", lambda: NOW.date())
    service._sealed_date = NOW.date()
    service._sealed_ready = True
    service._sealed_cache = {SYM: dict(sealed_up=True, bid1_vol=500, fetched_ts=(NOW - timedelta(minutes=10)).timestamp())}
    result = service._read_from_memory(False)[SYM]
    assert result["ready"] is False and result["sealed"] is None and result["vol"] is None
    assert service.is_sealed_ready(NOW.date()) is False


def test_source_switch_resets_volume_delta_baseline(monkeypatch):
    service = QuoteService()
    service._prev_stock_volume = {SYM: (900., 9000.)}
    service._previous_quote_sources = {SYM: "fuyao"}
    observed = []
    monkeypatch.setattr(service, "_update_volume_delta", lambda *args: observed.append(dict(service._prev_stock_volume)))
    monkeypatch.setattr(service, "_build_daily", lambda rows: pl.DataFrame())
    monkeypatch.setattr(service, "_evaluate_monitors", lambda *args: None)
    service._process_full_market_records([quote()], t0=0, now_ts=0, partial=True, memory_only=True)
    assert observed == [{}]
