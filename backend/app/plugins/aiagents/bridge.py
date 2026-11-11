from __future__ import annotations

from app.services.aiagents_bridge import data_health


def availability() -> tuple[bool, str]:
    status = data_health()
    if not status["provider_ready"]:
        return False, status["provider_error"] or "本地多源数据适配器初始化失败"
    if not status.get("data_root_exists", True):
        return False, f"数据目录不存在: {status.get('data_root', '')}"
    return True, "本地多源行情适配器已就绪"
