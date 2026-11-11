"""当前数据源的按需回源帮助函数。"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import polars as pl

logger = logging.getLogger(__name__)


def load_daily_from_active_provider(symbol: str, *, days: int = 180) -> pl.DataFrame:
    """当本地仓库还没有日线时,尝试从当前日线 provider 直接拉一份。"""
    from app.data_providers import custom as custom_sources
    from app.services import preferences

    provider_name = preferences.get_daily_data_provider()
    if provider_name == "tickflow":
        return pl.DataFrame()
    end = date.today()
    start = end - timedelta(days=max(30, days * 2))
    candidates = [provider_name]
    if provider_name != "xueqiu":
        candidates.append("xueqiu")
    for candidate in candidates:
        if not custom_sources.provider_has_dataset(candidate, "daily"):
            continue
        try:
            provider = custom_sources.get_provider(candidate)
            if candidate == "xueqiu" and not getattr(provider, "history_auth_configured", lambda: True)():
                continue
            df = provider.get_daily([symbol], start_time=start, end_time=end)
            if df is None or df.is_empty():
                continue
            if candidate != provider_name:
                logger.warning("按需回源主源 %s 无有效日K，已切换雪球末级备源", provider_name)
            return df.sort("date") if "date" in df.columns else df
        except Exception as exc:  # noqa: BLE001
            logger.warning("按需回源日线失败(%s via %s): %s", symbol, candidate, type(exc).__name__)
    return pl.DataFrame()
