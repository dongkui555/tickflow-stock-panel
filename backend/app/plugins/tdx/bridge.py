"""本地 TDX HTTP API 可用性检测。"""
from __future__ import annotations

import os

import requests


def _enabled() -> bool:
    return str(os.getenv("TDX_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}


def availability() -> tuple[bool, str]:
    if not _enabled():
        return False, "未启用 TDX_ENABLED；不会自动连接本机接口"

    base_url = os.getenv("TDX_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return False, "未设置 TDX_BASE_URL"

    try:
        timeout = max(1.0, float(os.getenv("TDX_TIMEOUT_SECONDS", "5")))
        response = requests.get(f"{base_url}/api/health", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return False, f"TDX API 不可访问 ({base_url}): {exc}"
    if isinstance(payload, dict) and payload.get("tdx_connected") is False:
        message = payload.get("message") or "行情服务器暂不可达"
        return False, f"TDX API 已启动但行情服务器不可用 ({base_url}): {message}"
    return True, f"本地 TDX API 已就绪: {base_url}"
