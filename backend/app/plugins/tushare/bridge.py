"""Tushare 内置插件可用性检测。"""
from __future__ import annotations

import os


def _token() -> str:
    from app.config import settings

    return (os.getenv("TUSHARE_TOKEN", "").strip() or settings.tushare_token.strip())


def availability() -> tuple[bool, str]:
    mode = str(os.getenv("TUSHARE_MODE", "official") or "official").strip().lower()
    if mode == "promax":
        if not os.getenv("TUSHARE_PROMAX_URL", "").strip():
            return False, "未设置 TUSHARE_PROMAX_URL"
        if not os.getenv("TUSHARE_PROMAX_KEY", "").strip():
            return False, "未设置 TUSHARE_PROMAX_KEY"
        return True, "Tushare Promax 网关已配置"

    token = _token()
    if not token:
        return False, "未设置 TUSHARE_TOKEN"
    try:
        import tushare as ts  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        return False, f"缺少 tushare 依赖: {exc}"
    try:
        ts.set_token(token)
        ts.pro_api()
    except Exception as exc:  # noqa: BLE001
        return False, f"Tushare 初始化失败: {exc}"
    return True, "Tushare token 已配置"
