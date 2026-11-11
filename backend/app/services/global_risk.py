"""Overnight global market risk factor for A-share short-term decisions."""
from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _quote_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("symbol") or "").upper(): item
        for item in (snapshot.get("quotes") or [])
        if isinstance(item, dict) and item.get("symbol")
    }


def _change(quotes: dict[str, dict[str, Any]], symbol: str) -> float | None:
    return _float(quotes.get(symbol.upper(), {}).get("change_pct"), None)


def _component_score(
    quotes: dict[str, dict[str, Any]],
    components: list[tuple[str, float]],
    *,
    scale: float = 12.0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_weight = 0.0
    weighted = 0.0
    for symbol, weight in components:
        item = quotes.get(symbol.upper(), {})
        change = _change(quotes, symbol)
        if change is None:
            continue
        total_weight += abs(weight)
        weighted += change * weight
        rows.append({
            "symbol": symbol,
            "name": item.get("name") or symbol,
            "change_pct": round(change, 4),
            "weight": weight,
            "contribution": round(change * weight, 4),
        })
    if total_weight <= 0:
        return {"score": 50.0, "weighted_change": 0.0, "available": 0, "components": rows}
    weighted_change = weighted / total_weight
    return {
        "score": round(_clamp(50.0 + weighted_change * scale), 2),
        "weighted_change": round(weighted_change, 4),
        "available": len(rows),
        "components": rows,
    }


def _news_signal(news: dict[str, Any]) -> dict[str, Any]:
    items = [item for item in (news.get("items") or []) if isinstance(item, dict)]
    if not items:
        return {"score": 50.0, "count": 0, "summary": "海外新闻未形成有效信号"}
    positive_words = (
        "rally", "rebounds", "gain", "gains", "surge", "beat", "optimism",
        "cut rates", "rate cut", "soft landing", "stimulus", "record high",
    )
    negative_words = (
        "selloff", "slump", "drop", "falls", "fear", "tariff", "sanction",
        "recession", "inflation", "higher rates", "crackdown", "war", "risk",
    )
    impact = 0
    headlines: list[str] = []
    for item in items[:20]:
        text = f"{item.get('title') or ''} {item.get('summary') or ''}".lower()
        pos = sum(1 for word in positive_words if word in text)
        neg = sum(1 for word in negative_words if word in text)
        impact += max(-2, min(2, pos - neg))
        title = str(item.get("title") or "").strip()
        if title and len(headlines) < 3:
            headlines.append(title[:120])
    score = _clamp(50 + impact * 3.0)
    return {
        "score": round(score, 2),
        "count": len(items),
        "headlines": headlines,
        "summary": "海外新闻偏正面" if score >= 58 else "海外新闻偏负面" if score <= 42 else "海外新闻整体中性",
    }


def _bias(score: float) -> tuple[str, str]:
    if score >= 68:
        return "进攻", "隔夜风险偏好较强，可维持正常候选筛选，但仍需竞价和资金确认"
    if score >= 56:
        return "轻试错", "外盘环境略偏正面，可小仓试错强共振标的"
    if score >= 46:
        return "中性", "外盘信号不极端，维持原有买点纪律"
    if score >= 36:
        return "谨慎", "外盘偏弱，降低追高和高弹性方向权重"
    return "防守", "外盘风险较高，盘前提高买入门槛并降低仓位"


def _recommendation_adjustment(score: float) -> dict[str, Any]:
    if score < 35:
        return {
            "score_adjustment": -8.0,
            "risk_score_adjustment": -14.0,
            "entry_cap": 62.0,
            "position_multiplier": 0.4,
            "buy_threshold_add": 8.0,
            "open_rule": "禁止开盘追涨，只看低开承接后资金回流",
        }
    if score < 45:
        return {
            "score_adjustment": -4.0,
            "risk_score_adjustment": -8.0,
            "entry_cap": 68.0,
            "position_multiplier": 0.6,
            "buy_threshold_add": 5.0,
            "open_rule": "开盘前30分钟只观察，等待分时资金确认",
        }
    if score < 55:
        return {
            "score_adjustment": -1.0,
            "risk_score_adjustment": -3.0,
            "entry_cap": 72.0,
            "position_multiplier": 0.8,
            "buy_threshold_add": 2.0,
            "open_rule": "维持谨慎，只做强资金确认标的",
        }
    if score >= 68:
        return {
            "score_adjustment": 2.0,
            "risk_score_adjustment": 3.0,
            "entry_cap": None,
            "position_multiplier": 1.1,
            "buy_threshold_add": -2.0,
            "open_rule": "可维持正常计划，但不放松止损",
        }
    return {
        "score_adjustment": 0.0,
        "risk_score_adjustment": 0.0,
        "entry_cap": None,
        "position_multiplier": 1.0,
        "buy_threshold_add": 0.0,
        "open_rule": "按原策略执行，等待集合竞价和分时资金确认",
    }


def _sector_impacts(group_scores: dict[str, dict[str, Any]], quotes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    impacts: list[dict[str, Any]] = []
    tech = group_scores.get("tech_ai", {}).get("score", 50.0)
    china = group_scores.get("china_proxy", {}).get("score", 50.0)
    broad = group_scores.get("us_broad", {}).get("score", 50.0)
    macro = group_scores.get("macro", {}).get("score", 50.0)
    if tech <= 42:
        impacts.append({
            "key": "tech_ai",
            "label": "AI/半导体/算力",
            "tone": "negative",
            "summary": "美股科技或半导体走弱，A股同方向开盘追高降级",
        })
    elif tech >= 60:
        impacts.append({
            "key": "tech_ai",
            "label": "AI/半导体/算力",
            "tone": "positive",
            "summary": "美股科技链偏强，A股同方向可等待竞价确认",
        })
    if china <= 42:
        impacts.append({
            "key": "china_proxy",
            "label": "中概/中国资产",
            "tone": "negative",
            "summary": "中概和中国资产代理偏弱，A股情绪需防低开扩散",
        })
    elif china >= 60 and broad < 50:
        impacts.append({
            "key": "china_proxy",
            "label": "中国资产相对抗跌",
            "tone": "positive",
            "summary": "美股弱但中国资产相对强，不宜简单全市场看空",
        })
    if broad <= 40:
        impacts.append({
            "key": "broad_market",
            "label": "全市场风险偏好",
            "tone": "negative",
            "summary": "美股宽基明显走弱，提高次日A股买入门槛",
        })
    if macro <= 42:
        impacts.append({
            "key": "macro",
            "label": "汇率/波动率",
            "tone": "negative",
            "summary": "美元人民币或波动率压力偏高，成长股仓位需收缩",
        })
    usdcny = _change(quotes, "USDCNY=X")
    if usdcny is not None and usdcny > 0.25:
        impacts.append({
            "key": "fx",
            "label": "人民币压力",
            "tone": "negative",
            "summary": f"美元/人民币{usdcny:+.2f}%，外资和高估值方向需降权",
        })
    return impacts[:6]


def build_global_risk(
    repo: Any | None = None,  # noqa: ARG001 - reserved for future local history storage
    *,
    force_refresh: bool = False,
    include_news: bool = True,
) -> dict[str, Any]:
    """Build a live overnight risk snapshot for A-share open decisions."""
    t0 = time.perf_counter()
    errors: list[str] = []
    try:
        from app.services.aiagents_sources import overseas_market_data

        market = overseas_market_data.get_market_snapshot(force_refresh=force_refresh)
        news = overseas_market_data.get_overseas_news(force_refresh=force_refresh, limit_per_feed=8) if include_news else {}
    except Exception as exc:  # noqa: BLE001
        market = {"data_success": False, "quotes": [], "error": str(exc)}
        news = {}
        errors.append(str(exc))

    if market.get("error"):
        errors.append(str(market.get("error")))
    if news.get("error"):
        errors.append(str(news.get("error")))

    quotes = _quote_map(market)
    group_scores = {
        "us_broad": _component_score(quotes, [
            ("^IXIC", 1.25), ("QQQ", 1.20), ("^GSPC", 1.00),
            ("SPY", 0.90), ("^DJI", 0.55), ("DIA", 0.45),
        ]),
        "tech_ai": _component_score(quotes, [
            ("SOXX", 1.30), ("QQQ", 1.05), ("NVDA", 0.90),
            ("AMD", 0.65), ("TSM", 0.65), ("MSFT", 0.40), ("AAPL", 0.35),
        ]),
        "china_proxy": _component_score(quotes, [
            ("FXI", 1.10), ("KWEB", 1.00), ("ASHR", 0.95), ("MCHI", 0.85),
            ("BABA", 0.45), ("PDD", 0.45), ("JD", 0.35),
        ]),
        "macro": _component_score(quotes, [
            ("USDCNY=X", -1.20), ("^VIX", -0.80), ("GC=F", -0.20), ("CL=F", -0.10),
        ], scale=10.0),
    }
    news_signal = _news_signal(news)
    group_scores["overseas_news"] = {
        "score": news_signal["score"],
        "weighted_change": 0.0,
        "available": news_signal["count"],
        "components": [],
    }

    appetite_score = _clamp(
        group_scores["us_broad"]["score"] * 0.28
        + group_scores["tech_ai"]["score"] * 0.24
        + group_scores["china_proxy"]["score"] * 0.26
        + group_scores["macro"]["score"] * 0.14
        + group_scores["overseas_news"]["score"] * 0.08
    )
    market_bias, guidance = _bias(appetite_score)
    adjustment = _recommendation_adjustment(appetite_score)

    quote_highlights = []
    for symbol in ("^IXIC", "QQQ", "SOXX", "FXI", "KWEB", "ASHR", "USDCNY=X", "^VIX"):
        item = quotes.get(symbol)
        change = _change(quotes, symbol)
        if item and change is not None:
            quote_highlights.append({
                "symbol": symbol,
                "name": item.get("name") or symbol,
                "change_pct": round(change, 4),
                "price": _float(item.get("price"), None),
            })

    risk_tags: list[str] = []
    if appetite_score < 45:
        risk_tags.append("隔夜外盘偏弱")
    if group_scores["tech_ai"]["score"] < 42:
        risk_tags.append("美股科技链承压")
    if group_scores["china_proxy"]["score"] < 42:
        risk_tags.append("中概/中国资产偏弱")
    if group_scores["macro"]["score"] < 42:
        risk_tags.append("汇率或波动率压力")
    if not risk_tags:
        risk_tags.append("未见极端隔夜风险")

    return {
        "as_of": datetime.now().date().isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": market.get("updated_at") or "",
        "source": market.get("source") or "AI Agents overseas market",
        "data_success": bool(market.get("data_success")),
        "cache_hit": bool(market.get("cache_hit")),
        "cache_fallback": bool(market.get("cache_fallback")),
        "quote_count": int(market.get("quote_count") or len(quotes)),
        "appetite_score": round(appetite_score, 2),
        "market_bias": market_bias,
        "guidance": guidance,
        "risk_tags": risk_tags,
        "group_scores": group_scores,
        "quote_highlights": quote_highlights,
        "sector_impacts": _sector_impacts(group_scores, quotes),
        "recommendation_adjustment": adjustment,
        "news_signal": news_signal,
        "rules": [
            "隔夜风险只作为盘前风险闸门，不单独产生买卖信号。",
            "外盘偏弱时提高买入门槛、降低仓位，等待集合竞价与分时资金确认。",
            "美股科技链弱时，A股AI、半导体、算力等高弹性方向禁止开盘追高。",
            "中国资产相对抗跌时，不简单全市场看空，保留提前布局板块的低吸观察。",
        ],
        "errors": sorted(set(errors)),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
