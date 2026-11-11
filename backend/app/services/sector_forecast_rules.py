"""Point-in-time sector start/continuation rules, shared by live and OOS research."""

from __future__ import annotations

import math

MODEL_VERSION = "sector_confluence_v2"
FEATURE_VERSION = "sector_flow_change_v2"
MIN_COVERAGE = 0.98
MAX_STRONG = 3


def number(value, default=0.0):
    try:
        n = float(value)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


def clamp(n):
    return max(0.0, min(100.0, n))


def started(row):
    if row.get("recently_started"):
        return True
    return (
        number(row.get("ret_1d")) >= 0.012
        and number(row.get("relative_return")) >= 0.006
        and number(row.get("up_ratio")) >= 0.55
    )


def classify(row: dict, market: dict, *, ablation: str | None = None) -> dict:
    ready = (
        bool(row.get("sector_data_ready", True))
        and number(row.get("flow_coverage")) >= MIN_COVERAGE
        and number(row.get("flow_history_days")) >= 15
    )
    active = started(row)
    kind = "continuation" if active else "start"
    flow = number(row.get("flow_intensity"))
    acceleration = number(row.get("flow_acceleration"))
    relative = number(row.get("relative_return"))
    ratio = number(row.get("amount_ratio_5d"), 1.0)
    leader = number(row.get("leader_strength"))
    ret5 = number(row.get("ret_5d"))
    risk_blocked = bool(market.get("very_weak") or market.get("event_risk"))
    if market.get("weak") and not row.get("event_confirmed"):
        risk_blocked = True
    evidence = {
        "flow_turn": flow > 0
        and (number(row.get("prior_flow_intensity")) <= 0 or acceleration >= 0.005),
        "flow_acceleration": acceleration > 0 and number(row.get("flow_zscore")) >= 0.5,
        "price_resilient": bool(row.get("price_resilient")) and relative >= 0.002,
        "moderate_volume": 1.03 <= ratio <= 1.8,
        "leader_first": 0.018 <= leader <= 0.09,
        "parent_linkage": number(row.get("parent_flow_acceleration")) > 0,
    }
    components = {
        "fund_turn": clamp(50 + number(row.get("flow_zscore")) * 20 + acceleration * 500)
        if flow > 0
        else 0.0,
        "relative_strength": clamp(50 + relative * 2000),
        "price_position": 100.0 if -0.02 <= ret5 <= (0.12 if active else 0.06) else 30.0,
        "volume_wake": clamp(100 - abs(ratio - 1.2) * 140),
        "leader_seed": clamp(leader / 0.035 * 100) if leader <= 0.09 else 25.0,
        "parent_linkage": clamp(50 + number(row.get("parent_flow_acceleration")) * 2000),
    }
    if ablation == "no_flow_change":
        components["fund_turn"] = 50.0
    if ablation == "no_parent":
        components["parent_linkage"] = 50.0
    component_ablation = {
        "no_resilience": "relative_strength",
        "no_volume": "volume_wake",
        "no_leader": "leader_seed",
        "no_position": "price_position",
    }
    if ablation in component_ablation:
        components[component_ablation[ablation]] = 50.0
    if ablation == "no_market_gate":
        risk_blocked = False
    keys = tuple(components)
    h1 = sum(
        components[k] * w for k, w in zip(keys, (0.30, 0.20, 0.10, 0.15, 0.15, 0.10), strict=True)
    )
    h2 = sum(
        components[k] * w for k, w in zip(keys, (0.30, 0.20, 0.10, 0.10, 0.10, 0.20), strict=True)
    )
    required_flow = evidence["flow_turn"] and evidence["flow_acceleration"]
    if ablation == "no_flow_change":
        required_flow = True
    confluence = (
        required_flow
        and (ablation == "no_resilience" or evidence["price_resilient"])
        and (ablation == "no_volume" or evidence["moderate_volume"])
        and (ablation == "no_leader" or evidence["leader_first"])
        and (ablation == "no_position" or ret5 <= (0.12 if active else 0.08))
        and number(row.get("risk_score"), 0) >= 55
    )
    score = max(h1, h2)
    level = "strong" if confluence and not risk_blocked and score >= 70 else "watch"
    if not ready:
        level = "unavailable"
    elif risk_blocked:
        level = "watch"
    labels = {
        "strong": "延续预测" if active else "启动前预测",
        "watch": "观察",
        "unavailable": "数据不足",
    }
    reasons = [
        text
        for key, text in {
            "flow_turn": "当日资金由流出转回补或强度提升",
            "flow_acceleration": "流入强度高于自身历史且继续改善",
            "price_resilient": "相对市场抗跌且价格未创新低",
            "moderate_volume": "温和放量",
            "leader_first": "龙头先行异动",
            "parent_linkage": "一级行业资金同步改善",
        }.items()
        if evidence[key]
    ]
    risks = []
    if not ready:
        risks.append("当日资金覆盖或历史窗口不足")
    if risk_blocked:
        risks.append("市场风险偏高或弱市事件确认不足, 停止强推荐")
    if not required_flow:
        risks.append("尚无资金转折与加速共振")
    return {
        **row,
        "forecast_kind": kind,
        "already_started": active,
        "prediction_score": round(score, 1),
        "priority_score": round(score, 1),
        "horizon_1_score": round(h1, 1),
        "horizon_2_score": round(h2, 1),
        "expected_horizon_days": 1 if h1 >= h2 else 2,
        "prediction_level": level,
        "prediction_label": labels[level],
        "probability": None,
        "probability_status": "uncalibrated_rule_score",
        "component_scores": {k: round(v, 2) for k, v in components.items()},
        "evidence": evidence,
        "reasons": reasons,
        "risk_flags": risks,
    }


def rank_forecasts(rows: list[dict], market: dict, *, ablation=None) -> list[dict]:
    ranked = [classify(r, market, ablation=ablation) for r in rows]
    ranked.sort(
        key=lambda r: (
            r["prediction_level"] == "strong",
            r["prediction_level"] != "unavailable",
            r["prediction_score"],
            r["sector"],
        ),
        reverse=True,
    )
    strong = 0
    for i, row in enumerate(ranked, 1):
        row["prediction_rank"] = i
        if row["prediction_level"] == "strong":
            strong += 1
            if strong > MAX_STRONG:
                row.update(prediction_level="watch", prediction_label="观察")
                row["risk_flags"].append("当日强预测名额已满")
    return ranked


def evaluate_outcome(prediction: dict, actual: dict) -> dict:
    positive = number(actual.get("ret_1d")) > 0
    relative = number(actual.get("relative_return")) >= 0.006
    launch = positive and relative and number(actual.get("ret_1d")) >= 0.012
    if prediction.get("forecast_kind") == "continuation":
        hit = positive and relative
    else:
        hit = launch and not prediction.get("already_started", False)
    return {
        "hit": bool(hit),
        "positive_return": positive,
        "relative_strength": relative,
        "new_start": launch and not prediction.get("already_started", False),
    }
