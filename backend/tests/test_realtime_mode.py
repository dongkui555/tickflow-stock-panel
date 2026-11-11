"""回归测试: 实时行情模式判定 — 免费档不再提供自选实时降级。

watchlist(自选前 5 只)模式已于 2026-08 移除: 自定义实时源(如 fuyao)的
全市场快照免费且更优, TickFlow 免费档不再保留降级通路。锁定判定结果,
防止该通路被无意恢复。
"""
from datetime import date

import polars as pl

from app.services.quote_service import QuoteService


def test_custom_realtime_source_is_full_market(monkeypatch):
    """自定义实时源(如 fuyao)无视 TickFlow 档位, 恒为全市场。"""
    from app.services import preferences
    import app.data_providers.custom as custom_sources

    class FullMarketProvider:
        realtime_scope = "all"

    monkeypatch.setattr(preferences, "get_realtime_data_provider", lambda: "fuyao")
    monkeypatch.setattr(custom_sources, "provider_has_dataset", lambda name, dataset: dataset == "realtime")
    monkeypatch.setattr(custom_sources, "get_provider", lambda name: FullMarketProvider())
    monkeypatch.setattr(QuoteService, "_current_tier", lambda: "free")
    assert QuoteService.realtime_mode() == "full_market"


def test_tickflow_free_has_no_realtime(monkeypatch):
    """TickFlow 免费档 = 无实时(不再降级为自选模式)。"""
    from app.services import preferences
    monkeypatch.setattr(preferences, "get_realtime_data_provider", lambda: "tickflow")
    monkeypatch.setattr(QuoteService, "_current_tier", lambda: "free")
    assert QuoteService.realtime_mode() == "none"
    assert QuoteService.is_realtime_allowed() is False


def test_tickflow_paid_is_full_market(monkeypatch):
    from app.services import preferences
    monkeypatch.setattr(preferences, "get_realtime_data_provider", lambda: "tickflow")
    monkeypatch.setattr(QuoteService, "_current_tier", lambda: "pro")
    assert QuoteService.realtime_mode() == "full_market"


def test_full_market_mode_ignores_stale_partial_enriched_cache(monkeypatch):
    """切到 Fuyao 全市场后, 旧 TDX/自选缓存不能继续遮住全市场行情。"""
    service = QuoteService()

    class _Repo:
        def __init__(self):
            self.calls = 0

        def get_enriched_latest(self):
            self.calls += 1
            return pl.DataFrame([
                {"symbol": "600519.SH", "date": date(2026, 9, 1), "close": 1480.0},
                {"symbol": "000001.SZ", "date": date(2026, 9, 1), "close": 12.3},
            ]), date(2026, 9, 1)

    repo = _Repo()
    service._repo = repo
    service._partial_enriched_cache = pl.DataFrame([
        {"symbol": "002396.SZ", "date": date(2026, 9, 1), "close": 30.5}
    ])
    service._partial_enriched_cache_date = date(2026, 9, 1)

    monkeypatch.setattr(QuoteService, "realtime_mode", classmethod(lambda cls: "full_market"))

    df, dt = service.get_enriched_today()

    assert dt == date(2026, 9, 1)
    assert repo.calls == 1
    assert df["symbol"].to_list() == ["600519.SH", "000001.SZ"]
