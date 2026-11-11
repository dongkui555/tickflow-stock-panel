"""Persistent sector pre-start prediction and validation records.

This layer is intentionally separate from ``sector_rotation``.  The rotation
service describes today's sector state; this service records what we would have
said at the previous close, then validates it one and two trading days later.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Any


logger = logging.getLogger(__name__)

MODEL_VERSION = "sector_prestart_v1"
STORE_FILENAME = "sector_predictions.json"
MAX_SNAPSHOTS = 180


def _store_path(data_dir: Path) -> Path:
    path = data_dir / "user_data" / STORE_FILENAME
    return path


def _read_store(data_dir: Path) -> list[dict[str, Any]]:
    path = _store_path(data_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("sector prediction store malformed: %s", exc)
    return []


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_day(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _available_dates(data_dir: Path, *, end: date | None = None) -> list[date]:
    root = data_dir / "kline_daily_enriched"
    if not root.exists():
        root = data_dir / "kline_daily"
    dates: list[date] = []
    if not root.exists():
        return dates
    for entry in root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("date="):
            continue
        try:
            day = date.fromisoformat(entry.name.removeprefix("date="))
        except ValueError:
            continue
        if end is not None and day > end:
            continue
        if (entry / "part.parquet").exists():
            dates.append(day)
    return sorted(set(dates))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float(value: Any, default: float = 0.0) -> float:
    out = _float_or_none(value)
    return default if out is None else out


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _round(value: Any, digits: int = 4) -> float | None:
    out = _float_or_none(value)
    return None if out is None else round(out, digits)


def _pct(value: Any, digits: int = 1) -> str:
    out = _float_or_none(value)
    if out is None:
        return "—"
    return f"{out * 100:.{digits}f}%"


def _amount_yi(value: Any) -> str:
    return f"{_float(value) / 100_000_000:.2f}亿"


def _parent_sector_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("-", 1)[0].strip() if "-" in text else text


def _mean_key(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_float_or_none(row.get(key)) for row in rows]
    nums = [v for v in vals if v is not None]
    return mean(nums) if nums else 0.0


def _market_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact market backdrop used by the pre-start sector model."""
    avg_ret_1d = _mean_key(rows, "ret_1d")
    avg_ret_5d = _mean_key(rows, "ret_5d")
    avg_up_ratio = _mean_key(rows, "up_ratio")
    weak = avg_ret_1d < -0.002 or avg_up_ratio < 0.48
    very_weak = avg_ret_1d < -0.008 or avg_up_ratio < 0.36
    return {
        "avg_ret_1d": round(avg_ret_1d, 6),
        "avg_ret_5d": round(avg_ret_5d, 6),
        "avg_up_ratio": round(avg_up_ratio, 6),
        "weak": weak,
        "very_weak": very_weak,
        "label": "弱市" if weak else "正常",
    }


def _parent_contexts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        parent = _parent_sector_name(row.get("sector"))
        if not parent:
            continue
        groups.setdefault(parent, []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for parent, items in groups.items():
        pre_net = sum(_float(row.get("pre_net_5d")) for row in items)
        pre_large_net = sum(_float(row.get("pre_large_net_5d")) for row in items)
        active_children = sum(
            1
            for row in items
            if _float(row.get("pre_net_5d")) > 0
            or _float(row.get("fund_layout_score"), 50.0) >= 58
            or str(row.get("phase") or "") in {"资金布局", "潜伏观察"}
        )
        avg_amount_ratio = _mean_key(items, "pre_amount_ratio_5d")
        avg_ret_1d = _mean_key(items, "ret_1d")
        avg_ret_5d = _mean_key(items, "ret_5d")
        avg_fund = _mean_key(items, "fund_layout_score")
        score = 46.0
        if pre_net > 0:
            score += 16.0
        if pre_large_net > 0:
            score += 8.0
        if active_children >= 3:
            score += 12.0
        elif active_children >= 2:
            score += 8.0
        if 1.03 <= avg_amount_ratio <= 1.75:
            score += 8.0
        elif avg_amount_ratio > 2.25:
            score -= 8.0
        if avg_fund >= 60:
            score += 8.0
        elif avg_fund >= 54:
            score += 4.0
        if avg_ret_1d > 0.04 or avg_ret_5d > 0.10:
            score -= 10.0
        out[parent] = {
            "parent_sector": parent,
            "child_count": len(items),
            "active_children": active_children,
            "pre_net_5d": round(pre_net, 2),
            "pre_large_net_5d": round(pre_large_net, 2),
            "avg_pre_amount_ratio_5d": round(avg_amount_ratio, 4),
            "avg_ret_1d": round(avg_ret_1d, 6),
            "avg_ret_5d": round(avg_ret_5d, 6),
            "avg_fund_layout_score": round(avg_fund, 1),
            "support_score": round(_clamp(score), 1),
        }
    return out


def _parent_linkage_score(row: dict[str, Any], parent_ctx: dict[str, Any] | None) -> float:
    parent = parent_ctx or {}
    support = _float(parent.get("support_score"), 50.0)
    ret_1d = _float(row.get("ret_1d"))
    ret_5d = _float(row.get("ret_5d"))
    score = support
    if _float(parent.get("pre_net_5d")) > 0 and ret_5d <= 0.06:
        score += 5.0
    if int(_float(parent.get("active_children"))) >= 2 and ret_1d <= 0.025:
        score += 4.0
    if ret_1d > 0.05 or ret_5d > 0.12:
        score -= 10.0
    return _clamp(score)


def _weak_market_defensive_score(
    row: dict[str, Any],
    market_ctx: dict[str, Any] | None,
    leader_score: float,
) -> float:
    market = market_ctx or {}
    ret_1d = _float(row.get("ret_1d"))
    ret_5d = _float(row.get("ret_5d"))
    up_ratio = _float(row.get("up_ratio"), 0.5)
    pre_net_3d = _float(row.get("pre_net_3d"))
    pre_net_5d = _float(row.get("pre_net_5d"))
    pre_amount_ratio = _float(row.get("pre_amount_ratio_5d"), 1.0)
    score = 46.0
    if market.get("weak"):
        score += 9.0
    if market.get("very_weak"):
        score += 5.0
    if ret_1d >= _float(market.get("avg_ret_1d")) - 0.002 and ret_1d <= 0.018:
        score += 14.0
    if ret_5d >= _float(market.get("avg_ret_5d")) and ret_5d <= 0.06:
        score += 10.0
    if pre_net_5d > 0 or pre_net_3d > 0:
        score += 10.0
    if 1.03 <= pre_amount_ratio <= 1.65:
        score += 8.0
    if leader_score >= 60:
        score += 6.0
    if up_ratio <= 0.45:
        score += 5.0
    if ret_1d > 0.045 or ret_5d > 0.10:
        score -= 12.0
    return _clamp(score)


def _price_position_score(row: dict[str, Any]) -> float:
    ret_1d = _float(row.get("ret_1d"))
    ret_5d = _float(row.get("ret_5d"))
    ret_10d = _float(row.get("ret_10d"))
    up_ratio = _float(row.get("up_ratio"), 0.5)
    limit_rate = _float(row.get("limit_up_rate"))
    score = 50.0
    if -0.025 <= ret_5d <= 0.04:
        score += 26.0
    elif -0.05 <= ret_5d <= 0.075:
        score += 16.0
    elif ret_5d > 0.12:
        score -= 24.0
    elif ret_5d < -0.08:
        score -= 12.0
    if -0.025 <= ret_1d <= 0.012:
        score += 14.0
    elif 0.012 < ret_1d <= 0.035:
        score += 5.0
    elif ret_1d > 0.045:
        score -= 20.0
    elif ret_1d < -0.045:
        score -= 8.0
    if -0.04 <= ret_10d <= 0.08:
        score += 8.0
    elif ret_10d > 0.14:
        score -= 12.0
    if up_ratio <= 0.25 and ret_5d > -0.04:
        score += 10.0
    elif 0.25 < up_ratio <= 0.62:
        score += 6.0
    elif up_ratio > 0.86:
        score -= 8.0
    if limit_rate > 0.10:
        score -= 12.0
    return _clamp(score)


def _flow_turn_score(row: dict[str, Any]) -> float:
    fund = _float(row.get("fund_layout_score"), 50.0)
    pre_net_3d = _float(row.get("pre_net_3d"))
    pre_net_5d = _float(row.get("pre_net_5d"))
    pre_large_net = _float(row.get("pre_large_net_5d"))
    net_5d = _float(row.get("net_5d"))
    large_net = _float(row.get("large_net_5d"))
    pre_positive = _float(row.get("pre_positive_days_5d"))
    positive = _float(row.get("positive_days_5d"))
    score = fund * 0.45 + 24.0
    if pre_net_5d > 0:
        score += 14.0
    elif pre_net_3d > 0:
        score += 9.0
    else:
        score -= 8.0
    if net_5d > 0:
        score += 5.0
    if pre_large_net > 0:
        score += 7.0
    elif large_net < 0 and pre_large_net < 0:
        score -= 7.0
    if pre_positive >= 4:
        score += 9.0
    elif pre_positive >= 2:
        score += 5.0
    if positive >= 3:
        score += 4.0
    return _clamp(score)


def _volume_wake_score(row: dict[str, Any]) -> float:
    volume = _float(row.get("volume_score"), 50.0)
    pre_amount_ratio = _float(row.get("pre_amount_ratio_5d"), 1.0)
    amount_ratio = _float(row.get("amount_ratio_5d"), 1.0)
    score = volume * 0.35 + 42.0
    for ratio in (pre_amount_ratio, amount_ratio):
        if 1.05 <= ratio <= 1.60:
            score += 9.0
        elif 1.60 < ratio <= 2.20:
            score += 3.0
        elif ratio > 2.20:
            score -= 9.0
        elif ratio < 0.72:
            score -= 6.0
    return _clamp(score)


def _stealth_setup_score(row: dict[str, Any]) -> float:
    ret_1d = _float(row.get("ret_1d"))
    ret_5d = _float(row.get("ret_5d"))
    up_ratio = _float(row.get("up_ratio"), 0.5)
    pre_amount_ratio = _float(row.get("pre_amount_ratio_5d"), 1.0)
    pre_net_3d = _float(row.get("pre_net_3d"))
    pre_net_5d = _float(row.get("pre_net_5d"))
    pre_large_net = _float(row.get("pre_large_net_5d"))
    pre_positive = _float(row.get("pre_positive_days_5d"))
    limit_rate = _float(row.get("limit_up_rate"))
    score = 44.0
    if pre_net_5d > 0 or pre_net_3d > 0:
        score += 18.0
    if -0.03 <= ret_5d <= 0.055:
        score += 16.0
    if ret_1d <= 0.012:
        score += 8.0
    if ret_1d < 0 and ret_5d > -0.035 and pre_net_5d > 0:
        score += 10.0
    if up_ratio < 0.25:
        score += 12.0
    elif up_ratio < 0.55:
        score += 7.0
    elif up_ratio > 0.86:
        score -= 10.0
    if 1.05 <= pre_amount_ratio <= 1.60:
        score += 8.0
    if pre_positive >= 2:
        score += 5.0
    if pre_large_net < 0:
        score -= 5.0
    if limit_rate > 0.08:
        score -= 8.0
    return _clamp(score)


def _leader_seed_score(row: dict[str, Any]) -> float:
    limit_up_count = _float(row.get("limit_up_count"))
    pre_limit_up_count = _float(row.get("pre_limit_up_count"))
    leaders = row.get("leader_stocks") or []
    best_ret_1d = max((_float(item.get("ret_1d")) for item in leaders if isinstance(item, dict)), default=0.0)
    best_ret_5d = max((_float(item.get("ret_5d")) for item in leaders if isinstance(item, dict)), default=0.0)
    score = 50.0
    if pre_limit_up_count > 0:
        score += 10.0
    if limit_up_count > 0:
        score += 8.0
    if 0.018 <= best_ret_1d <= 0.075:
        score += 10.0
    elif best_ret_1d > 0.095:
        score += 6.0
    if 0.025 <= best_ret_5d <= 0.12:
        score += 6.0
    elif best_ret_5d > 0.18:
        score -= 8.0
    return _clamp(score)


def _prediction_scores(
    row: dict[str, Any],
    parent_ctx: dict[str, Any] | None = None,
    market_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    setup = _float(row.get("setup_score", row.get("pre_layout_score")), 50.0)
    fund = _flow_turn_score(row)
    price = _price_position_score(row)
    stealth = _stealth_setup_score(row)
    volume = _volume_wake_score(row)
    leader = _leader_seed_score(row)
    parent = _parent_linkage_score(row, parent_ctx)
    defensive = _weak_market_defensive_score(row, market_ctx, leader)
    risk = _float(row.get("risk_score"), 60.0)
    ret_1d = _float(row.get("ret_1d"))
    ret_5d = _float(row.get("ret_5d"))
    up_ratio = _float(row.get("up_ratio"), 0.5)
    pre_net_5d = _float(row.get("pre_net_5d"))
    pre_large_net = _float(row.get("pre_large_net_5d"))
    pre_amount_ratio = _float(row.get("pre_amount_ratio_5d"), 1.0)

    base = (
        setup * 0.18
        + fund * 0.18
        + price * 0.17
        + stealth * 0.18
        + volume * 0.07
        + leader * 0.05
        + parent * 0.08
        + defensive * 0.05
        + risk * 0.04
    )
    horizon_1_raw = base
    horizon_2_raw = base
    if stealth >= 82 and price >= 85 and fund >= 58:
        horizon_1_raw += 3.0
        horizon_2_raw += 4.0
    if ret_1d < 0 and -0.035 <= ret_5d <= 0.055 and pre_net_5d > 0:
        horizon_1_raw += 5.0
        horizon_2_raw += 3.0
    if up_ratio < 0.28 and pre_amount_ratio >= 1.05:
        horizon_1_raw += 4.0
        horizon_2_raw += 6.0
    if parent >= 72 and ret_5d <= 0.07:
        horizon_1_raw += 3.0
        horizon_2_raw += 4.0
    if defensive >= 72 and (market_ctx or {}).get("weak"):
        horizon_1_raw += 4.0
        horizon_2_raw += 2.0
    if 0.012 < ret_1d <= 0.03 and up_ratio <= 0.65:
        horizon_1_raw += 4.0
    if ret_1d > 0.04:
        horizon_1_raw -= 10.0
        horizon_2_raw -= 6.0
    if ret_5d > 0.10:
        horizon_1_raw -= 9.0
        horizon_2_raw -= 8.0
    prestart_priority = 0.0
    if up_ratio <= 0.15:
        prestart_priority += 18.0
    elif up_ratio <= 0.28:
        prestart_priority += 12.0
    elif up_ratio <= 0.45:
        prestart_priority += 7.0
    elif up_ratio > 0.75:
        prestart_priority -= 8.0
    if -0.025 <= ret_1d <= 0.005:
        prestart_priority += 12.0
    elif 0.005 < ret_1d <= 0.018:
        prestart_priority += 6.0
    elif ret_1d > 0.035:
        prestart_priority -= 12.0
    if -0.02 <= ret_5d <= 0.035:
        prestart_priority += 10.0
    elif ret_5d <= 0.07:
        prestart_priority += 4.0
    elif ret_5d > 0.10:
        prestart_priority -= 15.0
    if 1.03 <= pre_amount_ratio <= 1.35:
        prestart_priority += 10.0
    elif 1.35 < pre_amount_ratio <= 1.75:
        prestart_priority += 6.0
    elif pre_amount_ratio > 2.20:
        prestart_priority -= 8.0
    if (
        up_ratio <= 0.12
        and -0.025 <= ret_1d <= 0.002
        and -0.02 <= ret_5d <= 0.04
        and pre_net_5d > 0
        and 1.03 <= pre_amount_ratio <= 1.35
    ):
        prestart_priority += 8.0
    if parent >= 72:
        prestart_priority += 8.0
    if defensive >= 72 and (market_ctx or {}).get("weak"):
        prestart_priority += 8.0
    if leader >= 58:
        prestart_priority += 4.0
    if pre_net_5d > 0:
        prestart_priority += 5.0
    if pre_large_net < 0:
        prestart_priority -= 4.0
    priority_score = max(horizon_1_raw, horizon_2_raw) + prestart_priority
    horizon_1 = _clamp(horizon_1_raw)
    horizon_2 = _clamp(horizon_2_raw)
    prediction_score = max(horizon_1, horizon_2)
    expected_horizon = 1 if horizon_1 >= horizon_2 else 2
    if prediction_score >= 72:
        level = "strong"
        label = "强预测"
    elif prediction_score >= 64:
        level = "watch"
        label = "提前观察"
    elif prediction_score >= 58:
        level = "weak_watch"
        label = "弱观察"
    else:
        level = "neutral"
        label = "普通观察"
    return {
        "prediction_score": round(prediction_score, 1),
        "priority_score": round(priority_score, 1),
        "prestart_priority_score": round(prestart_priority, 1),
        "horizon_1_score": round(horizon_1, 1),
        "horizon_2_score": round(horizon_2, 1),
        "expected_horizon_days": expected_horizon,
        "prediction_level": level,
        "prediction_label": label,
        "component_scores": {
            "setup": round(setup, 1),
            "fund_turn": round(fund, 1),
            "price_position": round(price, 1),
            "stealth": round(stealth, 1),
            "volume_wake": round(volume, 1),
            "leader_seed": round(leader, 1),
            "parent_linkage": round(parent, 1),
            "weak_defensive": round(defensive, 1),
            "risk": round(risk, 1),
        },
    }


def _prediction_reasons(
    row: dict[str, Any],
    scores: dict[str, Any],
    parent_ctx: dict[str, Any] | None = None,
    market_ctx: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    risks: list[str] = []
    pre_net_5d = _float(row.get("pre_net_5d"))
    pre_net_3d = _float(row.get("pre_net_3d"))
    pre_large_net = _float(row.get("pre_large_net_5d"))
    ret_1d = _float(row.get("ret_1d"))
    ret_5d = _float(row.get("ret_5d"))
    up_ratio = _float(row.get("up_ratio"), 0.5)
    pre_amount_ratio = _float(row.get("pre_amount_ratio_5d"), 1.0)
    amount_ratio = _float(row.get("amount_ratio_5d"), 1.0)
    positive = int(_float(row.get("pre_positive_days_5d")))
    limit_rate = _float(row.get("limit_up_rate"))
    components = scores.get("component_scores", {})
    if pre_net_5d > 0:
        reasons.append(f"昨日前5日主力净流入{_amount_yi(pre_net_5d)}")
    elif pre_net_3d > 0:
        reasons.append(f"昨日前3日主力净流入{_amount_yi(pre_net_3d)}")
    if positive >= 2:
        reasons.append(f"昨日前5日{positive}日净流入")
    if -0.03 <= ret_5d <= 0.055:
        reasons.append(f"5日涨幅{_pct(ret_5d)}，尚未明显拉升")
    if ret_1d <= 0.012:
        reasons.append(f"当日涨幅{_pct(ret_1d)}，有启动前回踩/蓄势特征")
    if up_ratio < 0.28:
        reasons.append(f"上涨家数仅{_pct(up_ratio)}，板块尚未扩散")
    if pre_amount_ratio >= 1.05:
        reasons.append(f"昨日前温和放量{pre_amount_ratio:.2f}倍")
    elif amount_ratio >= 1.05:
        reasons.append(f"近5日量能放大{amount_ratio:.2f}倍")
    if _float(components.get("stealth")) >= 72:
        reasons.append("隐蔽吸筹/启动前夜评分较高")
    if _float(components.get("weak_defensive")) >= 72 and (market_ctx or {}).get("weak"):
        reasons.append("弱市抗跌叠加小幅资金回补")
    if _float(components.get("parent_linkage")) >= 68 and parent_ctx:
        reasons.append(f"一级行业{parent_ctx.get('parent_sector')}同步出现资金/量能联动")
    if pre_large_net < 0:
        risks.append(f"大单资金仍分歧{_amount_yi(pre_large_net)}")
    if parent_ctx and _float(parent_ctx.get("support_score")) < 52:
        risks.append(f"一级行业{parent_ctx.get('parent_sector')}联动偏弱")
    if ret_5d > 0.10:
        risks.append(f"5日涨幅{_pct(ret_5d)}，后手风险上升")
    if up_ratio > 0.86:
        risks.append(f"上涨家数{_pct(up_ratio)}，可能已扩散")
    if limit_rate > 0.08:
        risks.append("涨停占比偏高，次日追高风险增加")
    if not reasons:
        reasons.append("综合评分进入观察，但缺少强触发证据")
    return reasons[:9], risks[:4]


def build_sector_prediction_snapshot(data_dir, *, as_of=None, lookback_days=70, limit=30, persist=True):
    from app.services.sector_forecast import build
    return build(data_dir, as_of=as_of, lookback_days=lookback_days, limit=limit, persist=persist)


def refresh_sector_prediction_records(data_dir, *, days=8, as_of=None, lookback_days=70, limit=30):
    from app.services.sector_forecast import refresh
    return refresh(data_dir, days=days, as_of=as_of, lookback_days=lookback_days, limit=limit)


def list_sector_prediction_records(data_dir, *, days=8, as_of=None, limit=30, origin="published"):
    from app.services.sector_forecast import records
    return records(data_dir, days=days, as_of=as_of, limit=limit, origin=origin)


def _aggregate_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [item.get("validation_summary") or {} for item in snapshots]
    top5_hits = sum(int(item.get("top5_hits") or 0) for item in summaries)
    top5_verified = sum(int(item.get("top5_verified") or 0) for item in summaries)
    top10_hits = sum(int(item.get("top10_hits") or 0) for item in summaries)
    top10_verified = sum(int(item.get("top10_verified") or 0) for item in summaries)
    strong_hits = sum(int(item.get("strong_hits") or 0) for item in summaries)
    strong_verified = sum(int(item.get("strong_verified") or 0) for item in summaries)
    actual_top10_verified = sum(int(item.get("actual_top10_verified") or 0) for item in summaries)
    actual_top10_strong_hits = sum(int(item.get("actual_top10_strong_hits") or 0) for item in summaries)
    actual_top10_watch_hits = sum(int(item.get("actual_top10_watch_hits") or 0) for item in summaries)
    actual_top10_misses = sum(int(item.get("actual_top10_misses") or 0) for item in summaries)
    excess_vals = [_float(item.get("top10_excess_1d")) for item in summaries if item.get("top10_excess_1d") is not None]
    return {
        "snapshots": len(snapshots),
        "top5_hits": top5_hits,
        "top5_verified": top5_verified,
        "top5_hit_rate": round(top5_hits / top5_verified, 4) if top5_verified else None,
        "top10_hits": top10_hits,
        "top10_verified": top10_verified,
        "top10_hit_rate": round(top10_hits / top10_verified, 4) if top10_verified else None,
        "strong_hits": strong_hits,
        "strong_verified": strong_verified,
        "strong_hit_rate": round(strong_hits / strong_verified, 4) if strong_verified else None,
        "actual_top10_verified": actual_top10_verified,
        "actual_top10_strong_hits": actual_top10_strong_hits,
        "actual_top10_watch_hits": actual_top10_watch_hits,
        "actual_top10_misses": actual_top10_misses,
        "actual_top10_strong_rate": round(actual_top10_strong_hits / actual_top10_verified, 4) if actual_top10_verified else None,
        "actual_top10_watch_rate": round(actual_top10_watch_hits / actual_top10_verified, 4) if actual_top10_verified else None,
        "actual_top10_miss_rate": round(actual_top10_misses / actual_top10_verified, 4) if actual_top10_verified else None,
        "avg_top10_excess_1d": round(mean(excess_vals), 6) if excess_vals else None,
    }
