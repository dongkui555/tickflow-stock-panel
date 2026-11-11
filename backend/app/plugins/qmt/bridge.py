"""QMT local HTTP bridge availability check."""
from __future__ import annotations

import os

import requests

_confirmed_capabilities: dict = {}


def available_datasets() -> list[str]:
    """Datasets confirmed by the last plugin probe; no extra network calls."""
    return [dataset for dataset, capability in (
        ("minute", "minute"), ("realtime", "realtime"),
        ("depth5", "depth5"), ("auction", "push_auction"),
    ) if _confirmed_capabilities.get(capability)]


def _enabled() -> bool:
    return str(os.getenv("QMT_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}


def availability() -> tuple[bool, str]:
    global _confirmed_capabilities
    if not _enabled():
        _confirmed_capabilities = {}
        return False, "未启用 QMT_ENABLED"
    base_url = os.getenv("QMT_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        _confirmed_capabilities = {}
        return False, "未设置 QMT_BASE_URL"
    headers = {}
    if api_key := os.getenv("QMT_BRIDGE_API_KEY", "").strip():
        headers["X-API-Key"] = api_key
    try:
        timeout = max(1.0, float(os.getenv("QMT_TIMEOUT_SECONDS", "8")))
        response = requests.get(f"{base_url}/api/capabilities", params={"refresh": "1"},
                                headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        caps = data if isinstance(data, dict) else {}
        _confirmed_capabilities = dict(caps)
        # Historical minute files are served by the local QMT kernel even when
        # the optional live quote collector / MiniQMT routing probe is not
        # running.  Do not hide the minute dataset behind the realtime check:
        # the sync pipeline can use QMT for historical minutes and fall back
        # independently for live quotes and depth.
        if caps.get("minute"):
            l2 = all(caps.get(key) for key in ("l2_quote", "l2_order", "l2_transaction"))
            return True, (
                "QMT 分钟历史可用；实时=" + ("可用" if caps.get("realtime") else "未启用")
                + f"，五档={bool(caps.get('depth5'))}，L2逐笔={l2}"
            )
        if not caps.get("symbol_routing"):
            return False, (
                "QMT 内部采集策略尚未运行，或多标的代码路由校验失败；"
                f"探测标的={caps.get('routing_probe_symbols', [])}"
            )
        if not caps.get("realtime"):
            return False, f"QMT 已连接但实时能力不可用: {caps}"
        l2 = all(caps.get(key) for key in ("l2_quote", "l2_order", "l2_transaction"))
        return True, (
            f"QMT 已就绪；分钟={bool(caps.get('minute'))}，"
            f"五档={bool(caps.get('depth5'))}，L2逐笔={l2}"
        )
    except Exception as exc:  # noqa: BLE001
        _confirmed_capabilities = {}
        return False, f"QMT bridge 不可访问 ({base_url}): {exc}"


def index_daily_availability() -> tuple[bool, str]:
    if not _enabled():
        return False, "未启用 QMT_ENABLED"
    base_url = os.getenv("QMT_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return False, "未设置 QMT_BASE_URL"
    headers = {}
    if api_key := os.getenv("QMT_BRIDGE_API_KEY", "").strip():
        headers["X-API-Key"] = api_key
    try:
        timeout = max(1.0, float(os.getenv("QMT_TIMEOUT_SECONDS", "8")))
        response = requests.get(f"{base_url}/api/capabilities", headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        caps = payload.get("data", payload) if isinstance(payload, dict) else {}
        if not caps.get("daily_index"):
            return False, f"QMT 指数日K探测未通过: {caps.get('daily_index_symbols', [])}"
        return True, "QMT 核心指数日K可用"
    except Exception as exc:  # noqa: BLE001
        return False, f"QMT 指数日K桥不可访问 ({base_url}): {exc}"
