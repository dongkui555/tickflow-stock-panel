"""Short-term AI recommendation aggregation service."""
from __future__ import annotations

import copy
import json
import math
import numbers
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from app.parquet import scan_enriched_parquet
from app.market_time import cn_now
from app.services import strategy_cache
from app.services.abnormal_moves import build_intraday as build_abnormal_intraday
from app.services.abnormal_moves import build_overview as build_abnormal_overview
from app.services.ai_provider import ai_configured, current_ai_model, generate_ai_text
from app.services.fund_monitor import build_daily_fund_monitor, build_intraday_fund_monitor
from app.services.global_risk import build_global_risk
from app.services.market_gate import build_market_gate
from app.services.minute_timeframes import five_minute_confirmation, load_five_minute_confirmations
from app.services import minute_status
from app.services.sector_layout import build_sector_rotation, normalize_sector_name
from app.services.stockformer import multifactor_select


SHORT_TERM_STOP_LOSS_PCT = {"aggressive": 0.05, "balanced": 0.045, "conservative": 0.035}

PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": {
        "technical": 0.25,
        "model": 0.18,
        "fund": 0.18,
        "sector": 0.19,
        "sentiment": 0.10,
        "risk": 0.10,
    },
    "aggressive": {
        "technical": 0.24,
        "model": 0.18,
        "fund": 0.20,
        "sector": 0.20,
        "sentiment": 0.10,
        "risk": 0.08,
    },
    "conservative": {
        "technical": 0.23,
        "model": 0.18,
        "fund": 0.16,
        "sector": 0.15,
        "sentiment": 0.08,
        "risk": 0.20,
    },
}

PROFILE_LABELS = {
    "balanced": "均衡",
    "aggressive": "进攻",
    "conservative": "保守",
}

IMPORTANT_STRATEGIES = {
    "trend_breakout",
    "volume_price_surge",
    "breakout_new_high_60d",
    "ma_convergence_breakout",
    "platform_consolidation_breakout",
    "pullback_ma20_bounce",
    "pullback_to_support",
    "strong_open",
    "high_turnover_surge",
    "active_limit_gene",
    "broken_board_recovery",
}

_KLINE_QUALITY_CACHE_TTL = 300.0
_KLINE_QUALITY_CACHE_LOCK = threading.Lock()
_KLINE_QUALITY_CACHE: dict[tuple[str, str, int], tuple[float, dict[str, dict[str, Any]], dict[str, Any]]] = {}
_RECOMMENDATIONS_CACHE_TTL = 45.0
_RECOMMENDATIONS_CACHE_LOCK = threading.Lock()
_RECOMMENDATIONS_CACHE: dict[tuple[str, str, str, int, str], tuple[float, dict[str, Any]]] = {}

_AGENT_SYSTEM_PROMPT = """你是一个A股短线AI复核Agent。
任务：只基于用户提供的候选股数据，复核短线可交易性，输出研究辅助结论。
约束：
1. 不要编造未提供的行情、新闻、资金或财务数据；数据不足时明确写“数据不足”。
2. 不给确定性承诺，不使用“必涨、稳赚、满仓、梭哈”等措辞。
3. 优先考虑风险闸门：派发、跌停、异动风险、资金流出、追高风险、模型置信不足时应降级。
4. 资金判断优先识别主力建仓/吸筹阶段：低位区间、窄幅整理、温和放量、1-3日资金由弱转强、资金进场但价格未明显拉升；5/10日连续流入只能作为确认，不可单独推荐追高。
5. 市场闸门优先于个股信号：A股弱势、退潮、大跌扩散时只能观察或暂停开仓，模型高分也不能直接买入。
6. 隔夜全球风险只用于判断盘前风险偏好和仓位，不可替代A股自身资金、竞价和分时确认。
7. 输出必须是严格 JSON，不要 Markdown，不要额外解释文本。
JSON 结构：
{
  "summary": "总体复核结论",
  "market_bias": "进攻/轻试错/观望/防守",
  "stockformer_role": "说明StockFormer在本次复核中的作用",
  "recommendations": [
    {
      "symbol": "股票代码",
      "name": "股票名称",
      "agent_rating": "优先关注/观察/放弃",
      "ai_score": 0,
      "buy_point": "具体触发条件或区间",
      "sell_point": "止盈/卖出条件",
      "stop_loss": "止损条件",
      "position": "仓位建议",
      "holding_period": "预计持有周期",
      "reason": "核心理由",
      "invalidation": "信号失效条件"
    }
  ],
  "risk_controls": ["组合层面风控规则"],
  "disclaimer": "仅供研究和模拟盘验证，不构成投资建议"
}"""


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _round(value: Any, digits: int = 4) -> float | None:
    out = _float_or_none(value)
    return None if out is None else round(out, digits)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if isinstance(value, numbers.Integral):
            return int(value)
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _symbol_key(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.startswith(("SH", "SZ", "BJ")) and len(text) >= 8:
        text = text[2:]
    if "." in text:
        text = text.split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else text


def _candidate(candidates: dict[str, dict[str, Any]], symbol: str) -> dict[str, Any]:
    key = _symbol_key(symbol)
    if key not in candidates:
        candidates[key] = {
            "symbol": key,
            "name": None,
            "board": None,
            "industry": None,
            "sources": [],
            "reasons": [],
            "risk_warnings": [],
            "strategy_hits": [],
            "strategy_count": 0,
        }
    return candidates[key]


def _append_unique(row: dict[str, Any], field: str, value: Any, *, limit: int = 8) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    values = row.setdefault(field, [])
    if text not in values and len(values) < limit:
        values.append(text)


def _merge_identity(row: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("name", "board", "industry"):
        if not row.get(key) and source.get(key):
            row[key] = source.get(key)


def _is_special_treatment_name(value: Any) -> bool:
    text = str(value or "").strip().upper()
    return bool(text) and ("ST" in text or "退市" in text)


def _decision(
    final_score: float,
    risk_score: float,
    support_count: int,
    risk_warnings: list[str],
    *,
    buy_threshold_add: float = 0.0,
    min_support_count: int = 3,
) -> tuple[str, str]:
    hard_risk = any(
        "跌停" in item
        or "派发" in item
        or "异动接近" in item
        or "日K覆盖不足" in item
        or "复权价格异常" in item
        or "日K缺失" in item
        or "特别处理" in item
        or "退市" in item
        for item in risk_warnings
    )
    if risk_score < 35 or hard_risk:
        return "回避", "风险优先，不进入短线买入池"
    threshold_add = max(0.0, float(buy_threshold_add or 0.0))
    strong_support = max(3, min_support_count)
    candidate_support = max(2, min_support_count - 1)
    observe_support = max(1, min_support_count - 2)
    if final_score >= 78 + threshold_add and support_count >= strong_support and risk_score >= 55:
        return "强关注", "等计划买点出现后小仓试错"
    if final_score >= 68 + threshold_add * 0.85 and support_count >= candidate_support and risk_score >= 45:
        return "候选", "等待资金和盘口继续确认"
    if final_score >= 58 + threshold_add * 0.50 and support_count >= observe_support:
        return "观察", "信号未完全共振，暂不追高"
    return "低优先级", "只保留在观察池"


def _decision_priority(decision: str) -> int:
    return {
        "强关注": 4,
        "候选": 3,
        "观察": 2,
        "低优先级": 1,
        "回避": 0,
    }.get(decision, 0)


def _market_gate_priority(gate: Any) -> int:
    return {
        "open": 4,
        "neutral": 3,
        "strict": 2,
        "observe": 1,
        "closed": 0,
    }.get(str(gate or ""), 2)


def _confirmation_priority(status: Any) -> int:
    return {
        "confirmed": 3,
        "waiting": 2,
        "blocked": 0,
    }.get(str(status or ""), 1)


def _sector_signal_priority(signal: Any) -> int:
    text = str(signal or "")
    return {
        "pre_layout": 4,
        "early_watch": 3,
        "early_layout": 3,
        "watch_layout": 2,
        "launch_confirm": 1,
        "accelerating": 1,
        "layout_diverge": -1,
        "overheated": -2,
        "fading": -3,
    }.get(text, 0)


_GLOBAL_SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tech_ai": ("AI", "人工智能", "算力", "半导体", "芯片", "电子", "通信", "软件", "机器人"),
    "china_proxy": ("互联网", "传媒", "游戏", "电商", "消费电子", "港股", "中概"),
    "macro": ("证券", "金融", "地产", "创业板", "科创", "高估值"),
}


def _global_sector_warning(row: dict[str, Any], global_risk: dict[str, Any]) -> str | None:
    text = " ".join(
        str(row.get(field) or "")
        for field in ("name", "board", "industry", "sector")
    ).upper()
    if not text:
        return None
    for impact in global_risk.get("sector_impacts") or []:
        if not isinstance(impact, dict) or impact.get("tone") != "negative":
            continue
        key = str(impact.get("key") or "")
        keywords = _GLOBAL_SECTOR_KEYWORDS.get(key, ())
        if any(keyword.upper() in text for keyword in keywords):
            return str(impact.get("summary") or "隔夜全球风险影响相关方向，开盘追高降级")
    return None


def _risk_penalty(risk_score: float, risk_warnings: list[str]) -> float:
    penalty = 0.0
    if risk_score < 35:
        penalty += 18.0
    if any("派发" in item for item in risk_warnings):
        penalty += 20.0
    if any("跌停" in item for item in risk_warnings):
        penalty += 25.0
    if any("异动接近" in item for item in risk_warnings):
        penalty += 16.0
    if any("复权价格异常" in item for item in risk_warnings):
        penalty += 32.0
    if any("日K覆盖不足" in item or "日K缺失" in item for item in risk_warnings):
        penalty += 28.0
    if any("资金未确认" in item for item in risk_warnings):
        penalty += 16.0
    if any("特别处理" in item or "退市" in item for item in risk_warnings):
        penalty += 35.0
    return min(45.0, penalty)


def _trade_plan(row: dict[str, Any], profile: str) -> dict[str, Any]:
    close = _float_or_none(row.get("close"))
    pred_3d = _float(row.get("pred_ret_3d"), 0.0)
    pred_5d = _float(row.get("pred_ret_5d"), 0.0)
    gate = str(row.get("market_gate") or "neutral")
    gate_label = str(row.get("market_gate_label") or "市场中性")
    confirm_label = str(row.get("buy_confirmation_label") or "等待确认")
    position_multiplier = _float(row.get("combined_position_multiplier"), _float(row.get("market_position_multiplier"), 0.8))
    base_position = {"aggressive": 0.24, "balanced": 0.18, "conservative": 0.12}.get(profile, 0.18)
    planned_position = max(0.0, min(0.30, base_position * position_multiplier))
    if gate == "closed":
        position_text = "0仓位，市场闸门关闭，暂停新开仓"
    elif gate == "observe":
        position_text = "0仓位或极小观察仓，弱市不主动开仓"
    elif planned_position <= 0:
        position_text = "0仓位，等待市场闸门恢复"
    else:
        position_text = f"单票不超过{planned_position * 100:.0f}%计划仓位，首笔半仓试错"
    if close is None or close <= 0:
        return {
            "entry": f"{gate_label}，{confirm_label}后再看分时回踩或放量突破",
            "stop_loss": "按最近有效支撑下方止损",
            "take_profit": "按前高或分批止盈",
            "holding_period": "1-5个交易日",
            "position": position_text,
            "confirmation": confirm_label,
        }

    risk_pct = SHORT_TERM_STOP_LOSS_PCT.get(profile, 0.045)
    target_pct = max(0.045, min(0.12, max(pred_3d, pred_5d, 0.0) * 0.70 + 0.045))
    entry_low = close * 0.985
    entry_high = close * 1.015
    breakout = close * 1.025
    if gate in {"closed", "observe"}:
        entry = f"{gate_label}，不新开仓；只记录 {entry_low:.2f}-{entry_high:.2f} 是否出现资金承接"
    elif confirm_label != "买入确认通过":
        entry = f"等待{confirm_label}：{entry_low:.2f}-{entry_high:.2f} 回踩不破，且15分钟资金转正；或放量站上 {breakout:.2f}"
    else:
        entry = f"{entry_low:.2f}-{entry_high:.2f} 回踩不破，且分时资金继续流入；或放量站上 {breakout:.2f}"
    return {
        "entry": entry,
        "stop_loss": f"{close * (1 - risk_pct):.2f}",
        "take_profit": f"{close * (1 + target_pct):.2f}",
        "holding_period": "1-5个交易日",
        "position": position_text,
        "confirmation": confirm_label,
    }


def _buy_confirmation(
    row: dict[str, Any],
    market_gate: dict[str, Any],
    *,
    as_of: date | None,
) -> dict[str, Any]:
    adjustment = market_gate.get("recommendation_adjustment") or {}
    gate = str(market_gate.get("gate") or "neutral")
    requires_intraday = bool(adjustment.get("requires_intraday_confirmation")) and as_of is None
    requires_sector = bool(adjustment.get("requires_sector_pre_layout"))
    checks: list[dict[str, Any]] = []

    def add(label: str, passed: bool, weight: float, *, required: bool = True, evidence: str = "") -> None:
        checks.append({
            "label": label,
            "passed": bool(passed),
            "required": bool(required),
            "weight": weight,
            "evidence": evidence,
        })

    fund_signal = str(row.get("daily_fund_signal") or "")
    intraday_signal = str(row.get("intraday_fund_signal") or "")
    has_fund = bool(row.get("fund_data_complete"))
    accumulation = _float(row.get("main_accumulation_score"), 0.0)
    early_fund = _float(row.get("early_fund_setup_score"), 0.0)
    fund_score = _float(row.get("_working_fund_score"), _float(row.get("fund_score"), 0.0))
    sector_signal = str(row.get("sector_signal") or "")
    sector_score = _float(row.get("sector_score"), 50.0)
    sector_setup = _float(row.get("sector_setup_score"), 50.0)
    model_score = _float(row.get("stockformer_score"), 50.0)
    pred_1d = _float(row.get("pred_ret_1d"), 0.0)
    pred_3d = _float(row.get("pred_ret_3d"), 0.0)
    chase_risk = _float(row.get("chase_risk_score"), 0.0)
    price_position_20d = _float(row.get("price_position_20d"), 0.5)
    ret_3d = _float(row.get("ret_3d"), 0.0)
    ret_5d = _float(row.get("ret_5d"), 0.0)
    base_zone = _float(row.get("daily_base_zone_score"), _float(row.get("base_zone_score"), 50.0))
    compression = _float(row.get("daily_compression_score"), _float(row.get("price_compression_score"), 50.0))
    volume_absorption = _float(row.get("daily_volume_absorption_score"), _float(row.get("volume_absorption_score"), 50.0))
    local_market_score = _float(market_gate.get("local_score"), 50.0)
    kline_ok = row.get("kline_quality_label") == "正常"

    fund_pass = has_fund and (
        fund_signal in {"main_accumulation", "early_layout", "low_absorption"}
        or intraday_signal in {"active_inflow", "breakout_flow"}
        or accumulation >= 72
        or early_fund >= 70
        or (fund_score >= 72 and bool(row.get("moneyflow_turn_positive")))
    )
    intraday_pass = intraday_signal in {"active_inflow", "breakout_flow"}
    sector_pass = (
        sector_signal in {"pre_layout", "early_watch", "early_layout", "watch_layout", "launch_confirm"}
        or sector_setup >= 62
        or sector_score >= 65
    )
    price_pass = (
        price_position_20d <= 0.78
        and ret_3d <= 0.065
        and ret_5d <= 0.105
        and chase_risk <= 58
        and base_zone >= 50
        and compression >= 50
        and volume_absorption >= 48
    )
    model_pass = model_score >= 60 and (pred_1d > 0 or pred_3d > 0)
    risk_pass = _float(row.get("_working_risk_score"), _float(row.get("risk_score"), 50.0)) >= 55 and not any(
        "派发" in str(item) or "流出" in str(item) or "跌停" in str(item)
        for item in row.get("risk_warnings") or []
    )
    market_pass = gate not in {"closed", "observe"} and local_market_score >= 60

    add("市场闸门", market_pass, 18.0, evidence=f"{market_gate.get('gate_label') or gate} / {local_market_score:.1f}")
    add("资金确认", fund_pass, 24.0, evidence=str(row.get("fund_status_label") or ""))
    add("技术位置", price_pass, 14.0, evidence=f"位置{price_position_20d:.2f} / 后手风险{chase_risk:.1f}")
    add("板块配合", sector_pass, 14.0, required=requires_sector, evidence=str(row.get("sector_phase") or ""))
    add("盘中确认", intraday_pass, 16.0 if requires_intraday else 8.0, required=requires_intraday, evidence=str(row.get("intraday_fund_label") or ""))
    if as_of is None:
        minute_check = row.get("five_minute_confirmation") or {}
        add("5分钟确认", minute_check.get("passed") is True, 10.0,
            evidence=str(minute_check.get("label") or "缺少5分钟数据"))
    add("模型方向", model_pass, 10.0, required=False, evidence=f"模型{model_score:.1f}")
    add("风险可控", risk_pass and kline_ok, 12.0, evidence=str(row.get("kline_quality_label") or ""))

    total_weight = sum(item["weight"] for item in checks) or 1.0
    passed_weight = sum(item["weight"] for item in checks if item["passed"])
    required_failed = [item["label"] for item in checks if item["required"] and not item["passed"]]
    score = _clamp(passed_weight / total_weight * 100.0)
    if gate == "closed":
        label = "市场暂停开仓"
        status = "blocked"
    elif gate == "observe":
        label = "弱市只观察"
        status = "blocked"
    elif required_failed:
        label = "等待" + "、".join(required_failed[:2])
        status = "waiting"
    elif score >= 76:
        label = "买入确认通过"
        status = "confirmed"
    elif score >= 58:
        label = "等待确认"
        status = "waiting"
    else:
        label = "买入确认不足"
        status = "blocked"
    return {
        "score": round(score, 1),
        "label": label,
        "status": status,
        "required_failed": required_failed,
        "checks": checks,
    }


def _load_strategy_hits(
    data_dir: Path,
    *,
    engine: Any = None,
    monitor_engine: Any = None,
    as_of: str | None = None,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    cached = strategy_cache.read_cache(data_dir) or {}
    results = dict(cached.get("results") or {})
    if monitor_engine is not None and hasattr(monitor_engine, "latest_strategy_results"):
        try:
            results.update(monitor_engine.latest_strategy_results() or {})
        except Exception:
            pass

    strategy_names: dict[str, str] = {}
    if engine is not None and hasattr(engine, "list_strategies"):
        try:
            for meta in engine.list_strategies():
                sid = str(meta.get("id") or "")
                if sid:
                    strategy_names[sid] = str(meta.get("name") or sid)
        except Exception:
            pass

    hits: dict[str, list[dict[str, str]]] = {}
    stale = False
    for sid, result in results.items():
        if not isinstance(result, dict):
            continue
        result_as_of = result.get("as_of")
        if as_of and result_as_of and str(result_as_of) != str(as_of):
            stale = True
            continue
        name = strategy_names.get(str(sid), str(sid))
        for item in (result.get("rows") or [])[:300]:
            if not isinstance(item, dict):
                continue
            symbol = _symbol_key(item.get("symbol"))
            if not symbol:
                continue
            hits.setdefault(symbol, []).append({"id": str(sid), "name": name})

    meta = {
        "as_of": cached.get("as_of"),
        "updated_at": cached.get("updated_at"),
        "stale_skipped": stale,
        "strategy_count": len(results),
    }
    return hits, meta


def _parse_as_of_date(value: str | date | None) -> date | None:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _load_kline_quality(
    data_dir: Path,
    *,
    end: date | None = None,
    lookback_days: int = 220,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = data_dir / "kline_daily_enriched"
    target_end = end or date.today()
    cache_key = (str(root), target_end.isoformat(), int(lookback_days))
    now = time.time()
    with _KLINE_QUALITY_CACHE_LOCK:
        cached = _KLINE_QUALITY_CACHE.get(cache_key)
        if cached and now - cached[0] <= _KLINE_QUALITY_CACHE_TTL:
            quality, meta = cached[1], dict(cached[2])
            meta["cache_hit"] = True
            return quality, meta

    start = target_end - timedelta(days=max(30, int(lookback_days)))
    meta: dict[str, Any] = {
        "source": "kline_daily_enriched",
        "date_start": start.isoformat(),
        "date_end": target_end.isoformat(),
        "expected_days": 0,
        "rows": 0,
        "symbols": 0,
        "warning": None,
        "cache_hit": False,
    }
    if not root.exists():
        meta["warning"] = "日K增强表不存在"
        return {}, meta
    try:
        lf = scan_enriched_parquet(str(root / "**" / "*.parquet")).filter(
            (pl.col("date") >= start) & (pl.col("date") <= target_end)
        )
        schema_names = lf.collect_schema().names()
        if "symbol" not in schema_names or "date" not in schema_names or "close" not in schema_names:
            meta["warning"] = "日K增强表缺少 symbol/date/close 字段"
            return {}, meta
        select_cols = [c for c in ("symbol", "date", "close", "raw_close") if c in schema_names]
        df = lf.select(select_cols).collect()
    except Exception as exc:  # noqa: BLE001
        meta["warning"] = f"日K质量检查失败: {exc}"
        return {}, meta

    if df.is_empty():
        meta["warning"] = "最近窗口没有日K记录"
        return {}, meta
    df = df.with_columns(
        pl.col("symbol").cast(pl.Utf8, strict=False).map_elements(_symbol_key, return_dtype=pl.Utf8).alias("symbol"),
        pl.col("date").cast(pl.Date, strict=False),
        pl.col("close").cast(pl.Float64, strict=False),
    )
    if "raw_close" in df.columns:
        df = df.with_columns(pl.col("raw_close").cast(pl.Float64, strict=False))
        df = df.with_columns(
            pl.when((pl.col("raw_close") > 0) & (pl.col("close") > 0))
            .then(pl.col("close") / pl.col("raw_close"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_price_ratio")
        )
        median_ratio = pl.col("_price_ratio").median().over("symbol")
        ratio_outlier = pl.when(median_ratio > 0).then(
            (pl.col("_price_ratio") < median_ratio * 0.30)
            | (pl.col("_price_ratio") > median_ratio * 3.50)
        ).otherwise(False)
        anomaly_expr = (
            pl.col("close").is_null()
            | (pl.col("close") <= 0)
            | (
                (pl.col("raw_close") > 0)
                & (
                    (pl.col("_price_ratio") < 0.02)
                    | (pl.col("_price_ratio") > 50.0)
                    | ratio_outlier
                )
            )
        )
    else:
        anomaly_expr = pl.col("close").is_null() | (pl.col("close") <= 0)
    df = df.with_columns(anomaly_expr.fill_null(False).alias("_price_anomaly"))
    df = df.sort(["symbol", "date"]).unique(subset=["symbol", "date"], keep="last").sort(["symbol", "date"])
    expected_days = int(df.select(pl.col("date").n_unique()).item() or 0)
    if expected_days <= 0:
        meta["warning"] = "最近窗口没有可识别交易日"
        return {}, meta
    grouped = (
        df.group_by("symbol")
        .agg(
            pl.len().alias("kline_rows"),
            pl.col("date").max().alias("last_kline_date"),
            pl.col("_price_anomaly").sum().alias("price_anomaly_rows"),
        )
        .with_columns((pl.col("kline_rows") / expected_days).alias("kline_coverage_ratio"))
    )
    min_rows = max(35, min(80, int(expected_days * 0.45)))
    out: dict[str, dict[str, Any]] = {}
    for item in grouped.to_dicts():
        symbol = _symbol_key(item.get("symbol"))
        if not symbol:
            continue
        rows = int(item.get("kline_rows") or 0)
        anomalies = int(item.get("price_anomaly_rows") or 0)
        coverage = _float(item.get("kline_coverage_ratio"), 0.0)
        if anomalies > 0:
            label = "复权价格异常"
        elif rows < min_rows or coverage < 0.45:
            label = "日K覆盖不足"
        else:
            label = "正常"
        out[symbol] = {
            "kline_rows": rows,
            "last_kline_date": _json_safe(item.get("last_kline_date")),
            "price_anomaly_rows": anomalies,
            "kline_coverage_ratio": round(coverage, 4),
            "kline_quality_label": label,
        }
    meta.update({
        "expected_days": expected_days,
        "min_rows": min_rows,
        "rows": int(df.height),
        "symbols": len(out),
    })
    with _KLINE_QUALITY_CACHE_LOCK:
        _KLINE_QUALITY_CACHE[cache_key] = (time.time(), out, dict(meta))
    return out, meta


def build_short_term_recommendations(
    repo: Any,
    quote_service: Any = None,
    minute_refresh: Any = None,
    strategy_engine: Any = None,
    monitor_engine: Any = None,
    *,
    model_id: str | None = None,
    as_of: date | None = None,
    limit: int = 30,
    profile: str = "balanced",
) -> dict[str, Any]:
    """Build a combined short-term recommendation board."""
    t0 = time.perf_counter()
    limit = max(1, min(100, int(limit)))
    profile = profile if profile in PROFILE_WEIGHTS else "balanced"
    weights = PROFILE_WEIGHTS[profile]
    data_dir = repo.store.data_dir
    current_time = cn_now()
    workflow = None
    if as_of is None:
        from app.services.prediction_workflow import status as workflow_status
        current_workflow = workflow_status(data_dir)
        if current_workflow["config"]["enabled"]:
            workflow = current_workflow
    frozen = (workflow or {}).get("snapshot") or {}
    sector_store = data_dir / "user_data" / "sector_forecasts_v2.sqlite3"
    cache_key = (
        str(data_dir),
        str(model_id or ""),
        as_of.isoformat() if isinstance(as_of, date) else "",
        limit,
        profile,
        current_time.strftime("%Y%m%d%H%M") if as_of is None else "historical",
        frozen.get("id", ""),
        sector_store.stat().st_mtime_ns if sector_store.exists() else 0,
        workflow is not None,
        json.dumps((workflow or {}).get("premarket") or {}, sort_keys=True, default=str),
    )
    now_ts = time.time()
    with _RECOMMENDATIONS_CACHE_LOCK:
        cached = _RECOMMENDATIONS_CACHE.get(cache_key)
        if cached and now_ts - cached[0] <= _RECOMMENDATIONS_CACHE_TTL:
            cached_payload = copy.deepcopy(cached[1])
            cached_payload["cache_hit"] = True
            cached_payload["cache_age_s"] = round(now_ts - cached[0], 1)
            cached_payload["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return cached_payload

    errors: list[str] = []
    candidates: dict[str, dict[str, Any]] = {}

    multifactor = copy.deepcopy(frozen) if frozen else multifactor_select(
        data_dir,
        model_id=model_id,
        as_of=as_of,
        limit=max(300, limit * 8),
    )
    if multifactor.get("error"):
        errors.append(str(multifactor["error"]))
    for item in multifactor.get("rows") or []:
        if not isinstance(item, dict):
            continue
        row = _candidate(candidates, item.get("symbol"))
        _merge_identity(row, item)
        row.update({
            "close": item.get("close"),
            "model_rank": item.get("model_rank"),
            "stockformer_score": _round(item.get("stockformer_score"), 2),
            "technical_score": _round(item.get("technical_score"), 2),
            "fund_score": _round(item.get("fund_score"), 2),
            "sentiment_score": _round(item.get("sentiment_score"), 2),
            "risk_score": _round(item.get("risk_score"), 2),
            "fundamental_score": _round(item.get("fundamental_score"), 2),
            "multifactor_score": _round(item.get("final_score"), 2),
            "entry_score": _round(item.get("entry_score"), 2),
            "base_final_score": _round(item.get("entry_score") if item.get("entry_score") is not None else item.get("final_score"), 2),
            "pred_ret_1d": _round(item.get("pred_ret_1d"), 6),
            "pred_ret_3d": _round(item.get("pred_ret_3d"), 6),
            "pred_ret_5d": _round(item.get("pred_ret_5d"), 6),
            "pred_ret_10d": _round(item.get("pred_ret_10d"), 6),
            "up_prob_5d": _round(item.get("up_prob_5d"), 6),
            "trend_prob_10d": _round(item.get("trend_prob_10d"), 6),
            "confidence": _round(item.get("confidence"), 6),
            "turnover_rate": _round(item.get("turnover_rate"), 4),
            "main_net_inflow_rate": _round(item.get("main_net_inflow_rate"), 6),
            "large_net_inflow_rate": _round(item.get("large_net_inflow_rate"), 6),
            "moneyflow_strength_3d": _round(item.get("moneyflow_strength_3d"), 6),
            "moneyflow_strength_5d": _round(item.get("moneyflow_strength_5d"), 6),
            "moneyflow_acceleration_3d": _round(item.get("moneyflow_acceleration_3d"), 6),
            "moneyflow_turn_positive": bool(item.get("moneyflow_turn_positive")) if item.get("moneyflow_turn_positive") is not None else None,
            "moneyflow_early_layout_score": _round(item.get("moneyflow_early_layout_score"), 4),
            "moneyflow_chase_risk": _round(item.get("moneyflow_chase_risk"), 4),
            "moneyflow_aging_risk": _round(item.get("moneyflow_aging_risk"), 4),
            "moneyflow_positive_days_5d": _round(item.get("moneyflow_positive_days_5d"), 2),
            "price_chase_risk": _round(item.get("price_chase_risk"), 4),
            "open_strength": _round(item.get("open_strength"), 6),
            "auction_amount_ratio": _round(item.get("auction_amount_ratio"), 6),
            "auction_volume_ratio": _round(item.get("auction_volume_ratio"), 6),
            "industry_ret_1d": _round(item.get("industry_ret_1d"), 6),
            "industry_heat_rank": _round(item.get("industry_heat_rank"), 6),
            "sector_score": _round(item.get("sector_score"), 2),
            "sector_rotation_score": _round(item.get("sector_rotation_score"), 6),
            "sector_fund_layout_score": _round(item.get("sector_fund_layout_score"), 6),
            "sector_setup_score": _round(item.get("sector_setup_score"), 6),
            "sector_pre_ret_5d": _round(item.get("sector_pre_ret_5d"), 6),
            "sector_pre_up_ratio": _round(item.get("sector_pre_up_ratio"), 6),
            "sector_pre_amount_ratio": _round(item.get("sector_pre_amount_ratio"), 4),
            "sector_pre_moneyflow_strength_5d": _round(item.get("sector_pre_moneyflow_strength_5d"), 6),
            "sector_today_confirm_score": _round(item.get("sector_today_confirm_score"), 6),
            "sector_moneyflow_strength_5d": _round(item.get("sector_moneyflow_strength_5d"), 6),
            "sector_up_ratio": _round(item.get("sector_up_ratio"), 6),
            "sector_amount_ratio": _round(item.get("sector_amount_ratio"), 4),
            "sector_crowding_risk": _round(item.get("sector_crowding_risk"), 6),
            "news_sentiment_score": _round(item.get("news_sentiment_score"), 4),
            "news_count_3d": _round(item.get("news_count_3d"), 2),
            "overseas_news_sentiment_score": _round(item.get("overseas_news_sentiment_score"), 4),
            "overseas_news_count_3d": _round(item.get("overseas_news_count_3d"), 2),
        })
        _append_unique(row, "sources", "StockFormer")
        if _float(item.get("pred_ret_1d")) > 0:
            _append_unique(row, "reasons", "1日预测为正")
        if _float(item.get("pred_ret_3d")) > 0:
            _append_unique(row, "reasons", "3日预测为正")
        if _float(item.get("stockformer_score")) >= 70:
            _append_unique(row, "reasons", "模型排名靠前")

    as_of_text = str(multifactor.get("as_of") or as_of or "")
    quality_end = _parse_as_of_date(as_of_text) or as_of or date.today()
    kline_quality, kline_quality_meta = _load_kline_quality(data_dir, end=quality_end)
    minute_quality = minute_status.for_date(data_dir, candidates, quality_end)
    minute_quality_meta = {
        "date": quality_end.isoformat(),
        "source": "user_data/minute_coverage_status.json",
        "statuses": minute_status.counts(data_dir, quality_end),
        "policy": "完整分钟进入盘中确认；其他标的仅允许日K模型观察",
    }
    strategy_hits, strategy_meta = _load_strategy_hits(
        data_dir,
        engine=strategy_engine,
        monitor_engine=monitor_engine,
        as_of=as_of_text or None,
    )
    for symbol, hits in strategy_hits.items():
        row = _candidate(candidates, symbol)
        important_hits = [hit for hit in hits if hit["id"] in IMPORTANT_STRATEGIES]
        row["strategy_hits"] = hits[:8]
        row["strategy_count"] = len(hits)
        _append_unique(row, "sources", "策略")
        if important_hits:
            names = "、".join(hit["name"] for hit in important_hits[:3])
            _append_unique(row, "reasons", f"命中短线策略：{names}")

    daily_fund = build_daily_fund_monitor(
        repo,
        quote_service,
        window=20,
        limit=6000,
        min_score=0,
    )
    daily_map: dict[str, dict[str, Any]] = {}
    for item in daily_fund.get("rows") or []:
        if not isinstance(item, dict):
            continue
        symbol = _symbol_key(item.get("symbol"))
        if not symbol:
            continue
        daily_map[symbol] = item
        row = _candidate(candidates, symbol)
        _merge_identity(row, item)
        if not row.get("close"):
            row["close"] = item.get("close")
        row["daily_fund_score"] = _round(item.get("score"), 2)
        row["daily_fund_signal"] = item.get("signal")
        row["daily_fund_label"] = item.get("signal_label")
        row["net_3d"] = _round(item.get("net_3d"), 2)
        row["net_5d"] = _round(item.get("net_5d"), 2)
        row["prev_net_3d"] = _round(item.get("prev_net_3d"), 2)
        row["net_ratio_1d"] = _round(item.get("net_ratio_1d"), 6)
        row["net_ratio_3d"] = _round(item.get("net_ratio_3d"), 6)
        row["net_ratio_5d"] = _round(item.get("net_ratio_5d"), 6)
        row["prev_net_ratio_3d"] = _round(item.get("prev_net_ratio_3d"), 6)
        row["large_ratio_5d"] = _round(item.get("large_ratio_5d"), 6)
        row["daily_early_layout_score"] = _round(item.get("early_layout_score"), 2)
        row["daily_confirm_score"] = _round(item.get("confirm_score"), 2)
        row["daily_price_setup_score"] = _round(item.get("price_setup_score"), 2)
        row["daily_accumulation_score"] = _round(item.get("accumulation_score"), 2)
        row["daily_base_zone_score"] = _round(item.get("base_zone_score"), 2)
        row["daily_compression_score"] = _round(item.get("compression_score"), 2)
        row["daily_volume_absorption_score"] = _round(item.get("volume_absorption_score"), 2)
        row["absorption_efficiency"] = _round(item.get("absorption_efficiency"), 4)
        row["price_position_20d"] = _round(item.get("position_20"), 4)
        row["price_position_60d"] = _round(item.get("position_60"), 4)
        row["price_range_10d"] = _round(item.get("range_ratio_10d"), 6)
        row["ret_3d"] = _round(item.get("ret_3d"), 6)
        row["ret_5d"] = _round(item.get("ret_5d"), 6)
        row["daily_inflow_age_risk"] = _round(item.get("inflow_age_risk"), 4)
        row["daily_moneyflow_acceleration_3d"] = _round(item.get("moneyflow_acceleration_3d"), 6)
        row["positive_days_10d"] = item.get("positive_days_10d")
        if item.get("moneyflow_acceleration_3d") is not None:
            row["moneyflow_acceleration_3d"] = _round(item.get("moneyflow_acceleration_3d"), 6)
        if item.get("moneyflow_turn_positive") is not None:
            row["moneyflow_turn_positive"] = bool(item.get("moneyflow_turn_positive")) or bool(row.get("moneyflow_turn_positive"))
        _append_unique(row, "sources", "资金")
        if item.get("signal") == "main_accumulation":
            _append_unique(row, "reasons", "日级资金提示主力建仓/吸筹")
        for reason in item.get("reasons") or []:
            _append_unique(row, "reasons", reason)
        if item.get("signal") == "distribution_risk":
            _append_unique(row, "risk_warnings", "日级资金提示拉高派发")
        if item.get("signal") == "outflow":
            _append_unique(row, "risk_warnings", "日级资金流出")

    intraday_fund = build_intraday_fund_monitor(
        repo,
        quote_service,
        minute_refresh,
        limit=800,
        min_score=0,
    )
    for item in intraday_fund.get("rows") or []:
        if not isinstance(item, dict):
            continue
        symbol = _symbol_key(item.get("symbol"))
        if not symbol:
            continue
        row = _candidate(candidates, symbol)
        _merge_identity(row, item)
        if item.get("close"):
            row["close"] = item.get("close")
        row["intraday_fund_score"] = _round(item.get("score"), 2)
        row["intraday_risk_score"] = _round(item.get("risk_score"), 2)
        row["intraday_fund_signal"] = item.get("signal")
        row["intraday_fund_label"] = item.get("signal_label")
        row["est_net_15m"] = _round(item.get("est_net_15m"), 2)
        row["ret_15m"] = _round(item.get("ret_15m"), 6)
        row["amount_ratio_20d"] = _round(item.get("amount_ratio_20d"), 4)
        _append_unique(row, "sources", "盘中资金")
        if item.get("signal") in {"active_inflow", "breakout_flow"}:
            _append_unique(row, "reasons", str(item.get("signal_label") or "盘中资金确认"))
        if item.get("signal") == "distribution_risk":
            _append_unique(row, "risk_warnings", "盘中资金提示冲高派发")
        if item.get("signal") == "outflow":
            _append_unique(row, "risk_warnings", "盘中资金流出")

    abnormal = build_abnormal_overview(repo, quote_service, min_closeness=0.5, limit=1000)
    for item in abnormal.get("rows") or []:
        if not isinstance(item, dict):
            continue
        symbol = _symbol_key(item.get("symbol"))
        if not symbol:
            continue
        row = _candidate(candidates, symbol)
        _merge_identity(row, item)
        closeness = _float(item.get("max_closeness"))
        row["abnormal_closeness"] = _round(closeness, 4)
        row["abnormal_status"] = item.get("status")
        _append_unique(row, "sources", "异动")
        if closeness >= 0.7:
            _append_unique(row, "risk_warnings", "异动接近监管阈值")

    intraday_abnormal = build_abnormal_intraday(repo, quote_service, limit=1000)
    for item in intraday_abnormal.get("rows") or []:
        if not isinstance(item, dict):
            continue
        symbol = _symbol_key(item.get("symbol"))
        if not symbol:
            continue
        row = _candidate(candidates, symbol)
        _merge_identity(row, item)
        signals = [str(v) for v in item.get("signals") or []]
        row["intraday_signals"] = signals
        if signals:
            _append_unique(row, "sources", "盘中异动")
            _append_unique(row, "reasons", f"盘中异动：{'、'.join(signals[:3])}")
        if any(v in {"limit_down", "broken"} for v in signals):
            _append_unique(row, "risk_warnings", "跌停或炸板风险")

    sector_rotation = build_sector_rotation(
        data_dir,
        as_of=quality_end,
        lookback_days=70,
        level=2,
        limit=100,
    )
    global_risk = build_global_risk(repo, force_refresh=False, include_news=True)
    apply_global_risk = as_of is None and bool(global_risk.get("data_success"))
    global_risk["applied_to_recommendations"] = apply_global_risk
    global_adjustment = global_risk.get("recommendation_adjustment") or {}
    market_gate = build_market_gate(
        data_dir,
        repo=repo,
        quote_service=quote_service,
        as_of=quality_end if as_of is not None else None,
        use_overview=True,
    )
    market_gate["applied_to_recommendations"] = True
    market_adjustment = market_gate.get("recommendation_adjustment") or {}
    sector_map = {
        str(item.get("sector") or ""): item
        for item in (sector_rotation.get("rows") or [])
        if isinstance(item, dict) and item.get("sector")
    }
    from app.services.sector_forecast_store import ForecastStore
    from app.services.sector_forecast import trading_dates as sector_trading_dates
    published_forecasts = ForecastStore(data_dir).list("published")
    sector_dates = sector_trading_dates(data_dir, quality_end) if published_forecasts else []
    expected_sector_date = (str(as_of) if as_of is not None else
                            str(next((d for d in reversed(sector_dates)
                                      if d < current_time.date() or current_time.hour >= 19), "")))
    published_sector = next((s for s in published_forecasts
                             if s["as_of"] == expected_sector_date), {})
    published_map = {r["sector"]: r for r in published_sector.get("rows", [])}
    for row in candidates.values():
        sector_name = normalize_sector_name(row.get("industry"), 2)
        if not sector_name:
            continue
        info = sector_map.get(sector_name)
        if not info:
            continue
        prediction = published_map.get(sector_name, {})
        confirmed = prediction.get("prediction_level") == "strong"
        info = {**info, "score": prediction.get("prediction_score", 50.) if confirmed else 50.,
                "phase": prediction.get("prediction_label", "等待正式预测"),
                "signal": ("pre_layout" if prediction.get("forecast_kind") == "start" else "launch_confirm") if confirmed else "watch",
                "pre_layout_score": prediction.get("prediction_score", 50.) if confirmed else 50.,
                "setup_score": prediction.get("prediction_score", 50.) if confirmed else 50.,
                "up_prob_hint_3d": None,
                "reasons": prediction.get("reasons", []) if confirmed else []}
        row["sector"] = sector_name
        row["sector_rank"] = info.get("rank")
        row["sector_phase"] = info.get("phase")
        row["sector_signal"] = info.get("signal")
        row["sector_score"] = _round(info.get("score"), 2)
        row["sector_setup_score"] = _round(info.get("setup_score"), 2)
        row["sector_pre_layout_score"] = _round(info.get("pre_layout_score"), 2)
        row["sector_today_confirm_score"] = _round(info.get("today_confirm_score"), 2)
        row["sector_fund_layout_score"] = _round(info.get("fund_layout_score"), 2)
        row["sector_momentum_score"] = _round(info.get("momentum_score"), 2)
        row["sector_volume_score"] = _round(info.get("volume_score"), 2)
        row["sector_breadth_score"] = _round(info.get("breadth_score"), 2)
        row["sector_risk_score"] = _round(info.get("risk_score"), 2)
        row["industry_ret_1d"] = _round(info.get("ret_1d"), 6)
        row["sector_ret_5d"] = _round(info.get("ret_5d"), 6)
        row["sector_up_ratio"] = _round(info.get("up_ratio"), 6)
        row["sector_net_5d"] = _round(info.get("net_5d"), 2)
        row["sector_pre_ret_5d"] = _round(info.get("pre_ret_5d"), 6)
        row["sector_pre_up_ratio"] = _round(info.get("pre_up_ratio"), 6)
        row["sector_pre_amount_ratio"] = _round(info.get("pre_amount_ratio_5d"), 4)
        row["sector_pre_net_5d"] = _round(info.get("pre_net_5d"), 2)
        row["sector_up_prob_hint_3d"] = _round(info.get("up_prob_hint_3d"), 2)
        _append_unique(row, "sources", "板块轮动")
        phase = str(info.get("phase") or "")
        if phase:
            _append_unique(row, "reasons", f"所属板块{phase}")
        for reason in info.get("reasons") or []:
            _append_unique(row, "reasons", f"板块：{reason}", limit=10)
        if info.get("signal") == "overheated":
            _append_unique(row, "risk_warnings", "所属板块高位拥挤")
        if info.get("signal") == "fading":
            _append_unique(row, "risk_warnings", "所属板块退潮")
        if info.get("signal") == "layout_diverge":
            _append_unique(row, "risk_warnings", "所属板块资金分歧")

    if hasattr(repo, "get_name_map"):
        missing = [symbol for symbol, row in candidates.items() if not row.get("name")]
        try:
            name_map = repo.get_name_map(missing) if missing else {}
        except Exception:
            name_map = {}
        for symbol, name in (name_map or {}).items():
            key = _symbol_key(symbol)
            if key in candidates and not candidates[key].get("name"):
                candidates[key]["name"] = name

    for row in candidates.values():
        if _is_special_treatment_name(row.get("name")):
            row["is_special_treatment"] = True
            _append_unique(row, "risk_warnings", "特别处理或退市风险")

    for symbol, row in candidates.items():
        quality = kline_quality.get(symbol)
        if quality:
            row.update(quality)
            if quality.get("kline_quality_label") == "复权价格异常":
                _append_unique(row, "risk_warnings", "复权价格异常")
            elif quality.get("kline_quality_label") == "日K覆盖不足":
                _append_unique(row, "risk_warnings", "日K覆盖不足")
        else:
            row.update({
                "kline_rows": 0,
                "last_kline_date": None,
                "price_anomaly_rows": 0,
                "kline_coverage_ratio": 0.0,
                "kline_quality_label": "日K缺失",
            })
            _append_unique(row, "risk_warnings", "日K缺失")

    minute_checks: dict[str, Any] = {}
    if frozen:
        allowed = {_symbol_key(r.get("symbol")) for r in frozen.get("rows", [])}
        candidates = {s: r for s, r in candidates.items() if _symbol_key(s) in allowed}
    if as_of is None:
        try:
            minute_checks = load_five_minute_confirmations(repo, list(candidates), cn_now())
        except Exception:
            errors.append("本地5分钟确认不可用，暂停买入确认")
    output_rows: list[dict[str, Any]] = []
    for symbol, row in candidates.items():
        minute_item = minute_quality.get(symbol) or {
            "status": "unknown", "actual_minutes": 0, "expected_minutes": 240,
            "reason": "未找到该股票该交易日的分钟校验记录",
        }
        row["minute_coverage_status"] = str(minute_item.get("status") or "unknown")
        row["minute_coverage_minutes"] = int(minute_item.get("actual_minutes") or 0)
        row["minute_expected_minutes"] = int(minute_item.get("expected_minutes") or 240)
        row["minute_coverage_reason"] = minute_item.get("reason")
        row["minute_data_eligible"] = minute_status.is_minute_eligible(minute_item)
        row["model_data_mode"] = "minute_confirmed" if row["minute_data_eligible"] else "daily_kline_only"
        if not row["minute_data_eligible"]:
            _append_unique(row, "risk_warnings", "分钟数据未通过完整性门槛，仅使用日K模型")
        row["five_minute_confirmation"] = (
            minute_checks.get(symbol) or five_minute_confirmation(pl.DataFrame(), cn_now())
            if as_of is None else {"status": "not_applied", "passed": False,
                                   "bar_end": None, "label": "历史日级结果未执行5分钟确认"}
        )
        has_fund_confirmation = bool(
            row.get("daily_fund_label")
            or row.get("intraday_fund_label")
            or row.get("net_5d") is not None
            or row.get("est_net_15m") is not None
        )
        row["fund_data_complete"] = has_fund_confirmation
        row["fund_status_label"] = (
            row.get("daily_fund_label")
            or row.get("intraday_fund_label")
            or ("暂无资金确认" if not has_fund_confirmation else None)
        )
        if not has_fund_confirmation:
            _append_unique(row, "risk_warnings", "资金未确认")
        # Track availability separately from the score. Missing auxiliary data
        # is no longer allowed to look like a neutral, confirmed observation.
        row["minute_data_available"] = bool(row["minute_data_eligible"])
        row["moneyflow_data_available"] = bool(symbol in daily_map or has_fund_confirmation)
        row["auction_data_available"] = bool(
            row.get("auction_amount_ratio") is not None
            or row.get("auction_volume_ratio") is not None
        )
        row["news_data_available"] = bool(
            (_float(row.get("news_count_3d"), 0.0) > 0)
            or (_float(row.get("overseas_news_count_3d"), 0.0) > 0)
        )
        availability_fields = (
            "minute_data_available", "moneyflow_data_available",
            "auction_data_available", "news_data_available",
        )
        row["data_coverage_score"] = round(
            sum(1.0 for field in availability_fields if row[field]) / len(availability_fields) * 100.0,
            1,
        )
        missing_sources = [
            label for field, label in (
                ("minute_data_available", "分钟K"),
                ("moneyflow_data_available", "资金流"),
                ("auction_data_available", "集合竞价"),
                ("news_data_available", "新闻"),
            ) if not row[field]
        ]
        if missing_sources:
            _append_unique(row, "risk_warnings", f"数据覆盖不足：{'、'.join(missing_sources)}")

        strategy_count = int(row.get("strategy_count") or 0)
        strategy_bonus = min(18.0, strategy_count * 4.0)
        technical = max(
            _float(row.get("technical_score"), 50.0),
            52.0 + strategy_bonus if strategy_count else 0.0,
        )
        model = _float(row.get("stockformer_score"), 50.0)
        fund = max(
            _float(row.get("fund_score"), 50.0),
            _float(row.get("daily_fund_score"), 0.0),
            _float(row.get("intraday_fund_score"), 0.0),
        )
        sentiment = _float(row.get("sentiment_score"), 50.0)
        sector = _float(row.get("sector_score"), _float(row.get("sector_rotation_score"), 50.0))
        entry = _float(row.get("entry_score"), _float(row.get("base_final_score"), 50.0))
        risk = min(
            _float(row.get("risk_score"), 70.0),
            _float(daily_map.get(symbol, {}).get("risk_score"), 70.0),
            _float(row.get("intraday_risk_score"), 70.0),
            _float(row.get("sector_risk_score"), 70.0),
        )
        sector_today_ret = _float(row.get("industry_ret_1d"), 0.0)
        abnormal_closeness = _float(row.get("abnormal_closeness"), 0.0)
        moneyflow_early = _float(row.get("moneyflow_early_layout_score"), 0.5) * 100.0
        model_accumulation = _float(row.get("moneyflow_accumulation_score"), 0.5) * 100.0
        daily_early = _float(row.get("daily_early_layout_score"), 50.0)
        daily_accumulation = _float(row.get("daily_accumulation_score"), 50.0)
        if row.get("daily_fund_signal") == "main_accumulation":
            daily_accumulation_effective = daily_accumulation
        elif row.get("daily_fund_signal") in {"early_layout", "low_absorption"}:
            daily_accumulation_effective = min(daily_accumulation, 72.0)
        else:
            daily_accumulation_effective = min(daily_accumulation, 64.0)
        accumulation_setup = max(model_accumulation, daily_accumulation_effective)
        early_fund_setup = max(moneyflow_early, daily_early, accumulation_setup * 0.94)
        chase_risk = max(
            _float(row.get("price_chase_risk"), 0.0),
            _float(row.get("moneyflow_chase_risk"), 0.0),
            _float(row.get("moneyflow_aging_risk"), 0.0),
            _float(row.get("daily_inflow_age_risk"), 0.0),
            _float(row.get("sector_crowding_risk"), 0.0),
        )
        row["main_accumulation_score"] = round(accumulation_setup, 1)
        row["early_fund_setup_score"] = round(early_fund_setup, 1)
        row["chase_risk_score"] = round(chase_risk * 100.0, 1)
        if accumulation_setup >= 72.0 and chase_risk <= 0.62 and has_fund_confirmation:
            fund = max(fund, accumulation_setup)
            entry = max(entry, accumulation_setup * 0.62 + entry * 0.38)
            _append_unique(row, "reasons", "主力建仓/吸筹优先")
        if early_fund_setup >= 68.0 and chase_risk <= 0.72 and has_fund_confirmation:
            fund = max(fund, early_fund_setup)
            entry = max(entry, early_fund_setup * 0.55 + entry * 0.45)
            _append_unique(row, "reasons", "资金早期布局优先")
        if row.get("moneyflow_turn_positive"):
            _append_unique(row, "reasons", "个股资金由弱转强")
        if chase_risk >= 0.78:
            fund = min(fund, 58.0)
            entry = min(entry, 60.0)
            risk -= 12.0
            _append_unique(row, "risk_warnings", "短线追高风险偏高")
        elif chase_risk >= 0.62:
            risk -= 6.0
            _append_unique(row, "risk_warnings", "短线已有一定涨幅")
        if _float(row.get("daily_inflow_age_risk"), 0.0) >= 0.45 or _float(row.get("moneyflow_aging_risk"), 0.0) >= 0.45:
            entry = min(entry, 68.0)
            _append_unique(row, "risk_warnings", "资金连续流入已偏后手")
        if abnormal_closeness >= 0.7:
            risk -= 18.0
        if row.get("intraday_fund_signal") == "distribution_risk":
            risk -= 12.0
        if row.get("daily_fund_signal") == "outflow":
            risk -= 10.0
        if row.get("sector_signal") == "fading":
            risk -= 14.0
            sector = min(sector, 40.0)
        elif row.get("sector_signal") == "overheated":
            risk -= 8.0
            sector = min(sector, 58.0)
        elif row.get("sector_signal") == "layout_diverge":
            risk -= 6.0
            sector = min(sector, 55.0)
        elif row.get("sector_signal") == "launch_confirm":
            sector = min(sector, max(_float(row.get("sector_setup_score"), 50.0), 62.0))
        elif row.get("sector_signal") in {"pre_layout", "early_watch"} and sector_today_ret <= 0.025:
            sector = min(100.0, sector + 4.0)
        if _sector_signal_priority(row.get("sector_signal")) <= 0 and _float(row.get("sector_setup_score"), 50.0) < 55.0:
            entry = min(entry, 62.0)
            sector = min(sector, 55.0)
        if sector_today_ret > 0.025:
            setup_score = _float(row.get("sector_setup_score"), 50.0)
            if setup_score < 65.0:
                risk -= 10.0
                sector = min(sector, 52.0)
                _append_unique(row, "risk_warnings", "板块今日涨幅偏高且提前布局不足")
            else:
                risk -= 5.0
                sector = min(sector, max(setup_score, 60.0))
        if not has_fund_confirmation:
            fund = min(fund, 45.0)
            entry = min(entry, 56.0)
            risk -= 8.0
        if row.get("kline_quality_label") in {"日K缺失", "日K覆盖不足", "复权价格异常"}:
            technical = min(technical, 45.0)
            model = min(model, 55.0)
            risk = min(risk, 30.0)
        global_score_adjustment = 0.0
        market_score_adjustment = _float(market_adjustment.get("score_adjustment"), 0.0)
        market_risk_adjustment = _float(market_adjustment.get("risk_score_adjustment"), 0.0)
        risk += market_risk_adjustment
        market_entry_cap = market_adjustment.get("entry_cap")
        if market_entry_cap is not None:
            entry = min(entry, _float(market_entry_cap, entry))
        market_gate_name = str(market_gate.get("gate") or "neutral")
        row["market_gate"] = market_gate_name
        row["market_gate_label"] = market_gate.get("gate_label")
        row["market_local_score"] = _round(market_gate.get("local_score"), 1)
        row["market_state_label"] = market_gate.get("state_label")
        row["market_phase_label"] = market_gate.get("phase_label")
        row["market_position_multiplier"] = _round(market_adjustment.get("position_multiplier"), 2)
        row["market_buy_threshold_add"] = _round(market_adjustment.get("buy_threshold_add"), 2)
        row["market_score_adjustment"] = round(market_score_adjustment, 1)
        if market_gate_name in {"closed", "observe", "strict"}:
            _append_unique(row, "sources", "市场闸门")
            for reason in market_gate.get("reasons") or []:
                _append_unique(row, "risk_warnings", f"市场闸门：{reason}", limit=10)
        if market_gate_name == "closed":
            fund = min(fund, 55.0)
            entry = min(entry, 55.0)
            _append_unique(row, "risk_warnings", "市场闸门关闭，暂停新开仓")
        elif market_gate_name == "observe":
            fund = min(fund, 62.0)
            entry = min(entry, 62.0)
            _append_unique(row, "risk_warnings", "弱市只观察，不主动开仓")
        elif market_gate_name == "strict":
            entry = min(entry, 72.0)
            _append_unique(row, "reasons", "市场闸门要求强共振小仓试错")
        if apply_global_risk:
            global_score = _float(global_risk.get("appetite_score"), 50.0)
            global_score_adjustment = _float(global_adjustment.get("score_adjustment"), 0.0)
            risk += _float(global_adjustment.get("risk_score_adjustment"), 0.0)
            entry_cap = global_adjustment.get("entry_cap")
            if entry_cap is not None:
                entry = min(entry, _float(entry_cap, entry))
            row["global_appetite_score"] = round(global_score, 1)
            row["global_risk_label"] = global_risk.get("market_bias")
            row["global_position_multiplier"] = _round(global_adjustment.get("position_multiplier"), 2)
            row["global_buy_threshold_add"] = _round(global_adjustment.get("buy_threshold_add"), 2)
            row["global_score_adjustment"] = round(global_score_adjustment, 1)
            _append_unique(row, "sources", "隔夜风险")
            if global_score < 45:
                _append_unique(row, "risk_warnings", str(global_risk.get("guidance") or "隔夜全球风险偏弱"))
            elif global_score >= 68:
                _append_unique(row, "reasons", "隔夜全球风险偏好较强")
            sector_warning = _global_sector_warning(row, global_risk)
            if sector_warning:
                _append_unique(row, "risk_warnings", sector_warning)
        else:
            row["global_appetite_score"] = _round(global_risk.get("appetite_score"), 1)
            row["global_risk_label"] = "未应用" if as_of is not None else global_risk.get("market_bias")
            row["global_position_multiplier"] = None
            row["global_buy_threshold_add"] = None
            row["global_score_adjustment"] = 0.0
        global_position_multiplier = (
            _float(global_adjustment.get("position_multiplier"), 1.0)
            if apply_global_risk else 1.0
        )
        market_position_multiplier = _float(market_adjustment.get("position_multiplier"), 0.8)
        row["combined_position_multiplier"] = round(max(0.0, min(1.2, global_position_multiplier * market_position_multiplier)), 2)
        row["_working_fund_score"] = fund
        row["_working_risk_score"] = risk
        buy_confirmation = _buy_confirmation(row, market_gate, as_of=as_of)
        row["buy_confirmation_score"] = buy_confirmation["score"]
        row["buy_confirmation_label"] = buy_confirmation["label"]
        row["buy_confirmation_status"] = buy_confirmation["status"]
        row["buy_confirmation_failed"] = buy_confirmation["required_failed"]
        row["buy_confirmation_checks"] = buy_confirmation["checks"]
        if buy_confirmation["status"] == "confirmed":
            _append_unique(row, "reasons", "买入确认通过")
        elif buy_confirmation["status"] == "waiting":
            entry = min(entry, 68.0 if market_gate_name not in {"observe", "closed"} else 60.0)
            _append_unique(row, "risk_warnings", str(buy_confirmation["label"]))
        else:
            entry = min(entry, 58.0 if market_gate_name != "closed" else 55.0)
            risk -= 6.0
            _append_unique(row, "risk_warnings", str(buy_confirmation["label"]))
        row.pop("_working_fund_score", None)
        row.pop("_working_risk_score", None)
        risk = _clamp(risk)

        support_count = 0
        kline_usable = row.get("kline_quality_label") == "正常"
        support_count += 1 if model >= 65 and kline_usable else 0
        support_count += 1 if (technical >= 65 or strategy_count > 0) and kline_usable else 0
        support_count += 1 if accumulation_setup >= 70 and has_fund_confirmation and chase_risk <= 0.62 else 0
        support_count += 1 if early_fund_setup >= 66 and has_fund_confirmation and chase_risk <= 0.72 else 0
        support_count += 1 if fund >= 68 and has_fund_confirmation and chase_risk <= 0.80 else 0
        support_count += 1 if sector >= 65 and row.get("sector_signal") not in {"fading", "overheated"} else 0
        support_count += 1 if sentiment >= 60 else 0
        support_count += 1 if entry >= 70 and row.get("sector_signal") in {"pre_layout", "early_watch", "early_layout", "watch_layout"} else 0

        weighted_score = _clamp(
            technical * weights["technical"]
            + model * weights["model"]
            + fund * weights["fund"]
            + sector * weights["sector"]
            + sentiment * weights["sentiment"]
            + risk * weights["risk"]
        )
        raw_final_score = _clamp(weighted_score * 0.76 + entry * 0.24)
        risk_penalty = _risk_penalty(risk, row.get("risk_warnings") or [])
        final_score = _clamp(raw_final_score - risk_penalty + global_score_adjustment + market_score_adjustment)
        final_score_cap = market_adjustment.get("final_score_cap")
        if final_score_cap is not None:
            final_score = min(final_score, _float(final_score_cap, final_score))
        if not has_fund_confirmation:
            final_score = min(final_score, 60.0)
        coverage_score = _float(row.get("data_coverage_score"), 0.0)
        if coverage_score < 50.0:
            final_score = min(final_score, 64.0)
            entry = min(entry, 62.0)
        elif coverage_score < 75.0:
            final_score = min(final_score, 72.0)
            entry = min(entry, 68.0)
        decision, action = _decision(
            final_score,
            risk,
            support_count,
            row.get("risk_warnings") or [],
            buy_threshold_add=max(
                _float(market_adjustment.get("buy_threshold_add"), 0.0),
                _float(global_adjustment.get("buy_threshold_add"), 0.0) if apply_global_risk else 0.0,
            ),
            min_support_count=int(market_adjustment.get("min_support_count") or 3),
        )
        if market_gate_name == "closed":
            decision, action = "回避", "市场闸门关闭，暂停新开仓"
        elif market_gate_name == "observe" and decision in {"强关注", "候选"}:
            decision, action = "观察", "弱市只观察，等待大盘和盘中资金确认"
        elif buy_confirmation["status"] != "confirmed" and decision == "强关注":
            decision, action = "候选", "买入确认未完全通过，等待盘口和资金二次确认"
        if as_of is None and not row["minute_data_eligible"]:
            # Keep the daily model useful for observation, but never present a
            # live row without a verified minute session as high confidence.
            decision, action = "观察", "仅日K模型：分钟数据未完整，等待校验通过后再确认"
            entry = min(entry, 62.0)
            final_score = min(final_score, 68.0)
            _append_unique(row, "risk_warnings", "实时分钟覆盖不足，禁止高置信度开仓")
        row.update({
            "rank": 0,
            "technical_score": round(technical, 1),
            "stockformer_score": round(model, 1),
            "fund_score": round(fund, 1),
            "sector_score": round(sector, 1),
            "sentiment_score": round(sentiment, 1),
            "risk_score": round(risk, 1),
            "entry_score": round(entry, 1),
            "raw_final_score": round(raw_final_score, 1),
            "risk_penalty": round(risk_penalty, 1),
            "global_score_adjustment": round(global_score_adjustment, 1),
            "market_score_adjustment": round(market_score_adjustment, 1),
            "final_score": round(final_score, 1),
            "support_count": support_count,
            "decision": decision,
            "action": action,
            "trade_plan": _trade_plan(row, profile),
            "minute_coverage_status": row["minute_coverage_status"],
            "minute_coverage_minutes": row["minute_coverage_minutes"],
            "minute_expected_minutes": row["minute_expected_minutes"],
            "minute_coverage_reason": row["minute_coverage_reason"],
            "minute_data_eligible": row["minute_data_eligible"],
            "model_data_mode": row["model_data_mode"],
            "minute_data_available": row["minute_data_available"],
            "moneyflow_data_available": row["moneyflow_data_available"],
            "auction_data_available": row["auction_data_available"],
            "news_data_available": row["news_data_available"],
            "data_coverage_score": row["data_coverage_score"],
        })
        if not row.get("reasons"):
            _append_unique(row, "reasons", "进入候选池但缺少强共振")
        output_rows.append(row)

    output_rows.sort(
        key=lambda r: (
            _market_gate_priority(r.get("market_gate")),
            _confirmation_priority(r.get("buy_confirmation_status")),
            _float(r.get("buy_confirmation_score")),
            bool(r.get("fund_data_complete")),
            2 if r.get("daily_fund_signal") == "main_accumulation" else 1 if r.get("daily_fund_signal") in {"early_layout", "low_absorption"} else 0,
            _decision_priority(str(r.get("decision") or "")),
            _sector_signal_priority(r.get("sector_signal")),
            _float(r.get("final_score")),
            _float(r.get("entry_score")),
            _float(r.get("support_count")),
            _float(r.get("stockformer_score")),
            _float(r.get("main_accumulation_score")),
            _float(r.get("early_fund_setup_score")),
            -_float(r.get("chase_risk_score")),
            _float(r.get("fund_score")),
        ),
        reverse=True,
    )
    for idx, row in enumerate(output_rows, start=1):
        row["rank"] = idx

    notes = [
        "StockFormer 负责预测未来 1/3/5/10 个交易日收益和概率。",
        "短线AI荐股中心把 StockFormer 作为模型因子，再叠加策略、资金早期布局、板块轮动、情绪和风险闸门。",
        "资金因子优先识别主力建仓/吸筹：低位区间、窄幅整理、温和放量、资金转强但价格尚未拉升；5/10日连续流入只作为确认项，防止后手追高。",
        "板块轮动会优先标记资金开始布局、但短期涨幅尚未过热的行业。",
        "隔夜全球风险因子会影响盘前买入门槛、仓位系数和高弹性板块追高权重。",
        "A股市场闸门会根据市场状态、情绪周期、上涨家数、MA20上方占比、涨跌停结构自动提高买入门槛；弱市会只观察或暂停开仓。",
        "买入确认分要求市场、资金、技术位置、板块、盘中资金、模型方向和风险控制多项通过，缺少关键确认时不会升级为买入候选。",
        "实时荐股对分钟数据采用硬门槛：完整分钟进入盘中确认；缺失、部分或源不覆盖只保留日K模型观察，不伪造中性分钟特征。",
    ]
    if as_of is not None:
        notes.append("指定历史日期时，当前隔夜风险只展示不参与历史推荐评分。")

    # source_counts is rendered as scalar metrics by the frontend. Keep the
    # detailed minute quality object in minute_quality_meta instead of placing
    # it here, otherwise React attempts to render an object as a child.
    minute_status_counts = minute_quality_meta.get("statuses") or {}
    source_counts = {
        "stockformer": len(multifactor.get("rows") or []),
        "strategy_symbols": len(strategy_hits),
        "daily_fund": len(daily_fund.get("rows") or []),
        "intraday_fund": len(intraday_fund.get("rows") or []),
        "abnormal": len(abnormal.get("rows") or []),
        "intraday_abnormal": len(intraday_abnormal.get("rows") or []),
        "sector_rotation": len(sector_rotation.get("rows") or []),
        "global_risk_quotes": int(global_risk.get("quote_count") or 0),
        "minute_coverage": int(sum(int(value or 0) for value in minute_status_counts.values())),
        "minute_complete": int(minute_status_counts.get("complete") or 0),
    }
    payload = {
        "as_of": as_of_text or multifactor.get("as_of") or None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "profile": profile,
        "profile_label": PROFILE_LABELS[profile],
        "weights": weights,
        "rows": output_rows[:limit],
        "total": len(output_rows),
        "source_counts": source_counts,
        "model": multifactor.get("model"),
        "stockformer_weights": multifactor.get("weights"),
        "data_coverage": multifactor.get("data_coverage"),
        "kline_quality_meta": kline_quality_meta,
        "minute_quality_meta": minute_quality_meta,
        "cross_section": multifactor.get("cross_section"),
        "sector_rotation": {
            "as_of": sector_rotation.get("as_of"),
            "rows": (sector_rotation.get("rows") or [])[:20],
            "early_layout": sector_rotation.get("early_layout") or [],
            "meta": sector_rotation.get("meta"),
            "elapsed_ms": sector_rotation.get("elapsed_ms"),
        },
        "global_risk": global_risk,
        "market_gate": market_gate,
        "strategy_meta": strategy_meta,
        "fund_meta": {
            "daily": {
                "flow_meta": daily_fund.get("flow_meta"),
                "market_meta": daily_fund.get("market_meta"),
                "weights": daily_fund.get("weights"),
            },
            "intraday": {
                "flow_meta": intraday_fund.get("flow_meta"),
                "market_meta": intraday_fund.get("market_meta"),
                "minute_meta": intraday_fund.get("minute_meta"),
                "quote_mode": intraday_fund.get("quote_mode"),
                "quote_is_fresh": intraday_fund.get("quote_is_fresh"),
                "weights": intraday_fund.get("weights"),
            },
        },
        "errors": errors,
        "notes": notes,
        "cache_hit": False,
        "cache_age_s": 0.0,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    safe_payload = _json_safe(payload)
    if workflow is not None:
        from app.services.prediction_workflow import apply_execution_gate
        safe_payload = apply_execution_gate(safe_payload, workflow, current_time)
    with _RECOMMENDATIONS_CACHE_LOCK:
        _RECOMMENDATIONS_CACHE[cache_key] = (time.time(), copy.deepcopy(safe_payload))
        if len(_RECOMMENDATIONS_CACHE) > 24:
            oldest = sorted(_RECOMMENDATIONS_CACHE.items(), key=lambda item: item[1][0])[:8]
            for key, _ in oldest:
                _RECOMMENDATIONS_CACHE.pop(key, None)
    return safe_payload


def _compact_agent_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": row.get("rank"),
        "symbol": row.get("symbol"),
        "name": row.get("name"),
        "board": row.get("board"),
        "industry": row.get("industry"),
        "close": row.get("close"),
        "decision": row.get("decision"),
        "action": row.get("action"),
        "final_score": row.get("final_score"),
        "raw_final_score": row.get("raw_final_score"),
        "risk_penalty": row.get("risk_penalty"),
        "technical_score": row.get("technical_score"),
        "stockformer_score": row.get("stockformer_score"),
        "fund_score": row.get("fund_score"),
        "sector": row.get("sector"),
        "sector_rank": row.get("sector_rank"),
        "sector_phase": row.get("sector_phase"),
        "sector_signal": row.get("sector_signal"),
        "sector_score": row.get("sector_score"),
        "sector_setup_score": row.get("sector_setup_score"),
        "sector_pre_layout_score": row.get("sector_pre_layout_score"),
        "sector_today_confirm_score": row.get("sector_today_confirm_score"),
        "sector_fund_layout_score": row.get("sector_fund_layout_score"),
        "sector_pre_net_5d": row.get("sector_pre_net_5d"),
        "sector_pre_ret_5d": row.get("sector_pre_ret_5d"),
        "sector_pre_up_ratio": row.get("sector_pre_up_ratio"),
        "sector_net_5d": row.get("sector_net_5d"),
        "sector_ret_5d": row.get("sector_ret_5d"),
        "sector_up_ratio": row.get("sector_up_ratio"),
        "sector_up_prob_hint_3d": row.get("sector_up_prob_hint_3d"),
        "sentiment_score": row.get("sentiment_score"),
        "risk_score": row.get("risk_score"),
        "global_appetite_score": row.get("global_appetite_score"),
        "global_risk_label": row.get("global_risk_label"),
        "global_score_adjustment": row.get("global_score_adjustment"),
        "global_position_multiplier": row.get("global_position_multiplier"),
        "global_buy_threshold_add": row.get("global_buy_threshold_add"),
        "market_gate": row.get("market_gate"),
        "market_gate_label": row.get("market_gate_label"),
        "market_local_score": row.get("market_local_score"),
        "market_state_label": row.get("market_state_label"),
        "market_phase_label": row.get("market_phase_label"),
        "market_score_adjustment": row.get("market_score_adjustment"),
        "market_position_multiplier": row.get("market_position_multiplier"),
        "market_buy_threshold_add": row.get("market_buy_threshold_add"),
        "combined_position_multiplier": row.get("combined_position_multiplier"),
        "buy_confirmation_score": row.get("buy_confirmation_score"),
        "buy_confirmation_label": row.get("buy_confirmation_label"),
        "buy_confirmation_status": row.get("buy_confirmation_status"),
        "buy_confirmation_failed": row.get("buy_confirmation_failed"),
        "buy_confirmation_checks": row.get("buy_confirmation_checks"),
        "five_minute_confirmation": row.get("five_minute_confirmation"),
        "support_count": row.get("support_count"),
        "model_rank": row.get("model_rank"),
        "pred_ret_1d": row.get("pred_ret_1d"),
        "pred_ret_3d": row.get("pred_ret_3d"),
        "pred_ret_5d": row.get("pred_ret_5d"),
        "pred_ret_10d": row.get("pred_ret_10d"),
        "up_prob_5d": row.get("up_prob_5d"),
        "trend_prob_10d": row.get("trend_prob_10d"),
        "confidence": row.get("confidence"),
        "daily_fund_label": row.get("daily_fund_label"),
        "intraday_fund_label": row.get("intraday_fund_label"),
        "fund_status_label": row.get("fund_status_label"),
        "fund_data_complete": row.get("fund_data_complete"),
        "early_fund_setup_score": row.get("early_fund_setup_score"),
        "main_accumulation_score": row.get("main_accumulation_score"),
        "daily_early_layout_score": row.get("daily_early_layout_score"),
        "daily_accumulation_score": row.get("daily_accumulation_score"),
        "daily_base_zone_score": row.get("daily_base_zone_score"),
        "daily_compression_score": row.get("daily_compression_score"),
        "daily_volume_absorption_score": row.get("daily_volume_absorption_score"),
        "absorption_efficiency": row.get("absorption_efficiency"),
        "daily_confirm_score": row.get("daily_confirm_score"),
        "daily_price_setup_score": row.get("daily_price_setup_score"),
        "moneyflow_early_layout_score": row.get("moneyflow_early_layout_score"),
        "moneyflow_accumulation_score": row.get("moneyflow_accumulation_score"),
        "moneyflow_absorption_efficiency": row.get("moneyflow_absorption_efficiency"),
        "moneyflow_acceleration_3d": row.get("moneyflow_acceleration_3d"),
        "moneyflow_turn_positive": row.get("moneyflow_turn_positive"),
        "moneyflow_chase_risk": row.get("moneyflow_chase_risk"),
        "moneyflow_aging_risk": row.get("moneyflow_aging_risk"),
        "moneyflow_positive_days_5d": row.get("moneyflow_positive_days_5d"),
        "price_chase_risk": row.get("price_chase_risk"),
        "price_position_20d": row.get("price_position_20d"),
        "price_position_60d": row.get("price_position_60d"),
        "price_range_10d": row.get("price_range_10d"),
        "price_compression_score": row.get("price_compression_score"),
        "volume_absorption_score": row.get("volume_absorption_score"),
        "base_zone_score": row.get("base_zone_score"),
        "ret_3d": row.get("ret_3d"),
        "ret_5d": row.get("ret_5d"),
        "daily_inflow_age_risk": row.get("daily_inflow_age_risk"),
        "daily_moneyflow_acceleration_3d": row.get("daily_moneyflow_acceleration_3d"),
        "chase_risk_score": row.get("chase_risk_score"),
        "kline_quality_label": row.get("kline_quality_label"),
        "kline_rows": row.get("kline_rows"),
        "kline_coverage_ratio": row.get("kline_coverage_ratio"),
        "minute_coverage_status": row.get("minute_coverage_status"),
        "minute_coverage_minutes": row.get("minute_coverage_minutes"),
        "minute_expected_minutes": row.get("minute_expected_minutes"),
        "minute_coverage_reason": row.get("minute_coverage_reason"),
        "minute_data_eligible": row.get("minute_data_eligible"),
        "model_data_mode": row.get("model_data_mode"),
        "price_anomaly_rows": row.get("price_anomaly_rows"),
        "last_kline_date": row.get("last_kline_date"),
        "net_3d": row.get("net_3d"),
        "net_5d": row.get("net_5d"),
        "prev_net_3d": row.get("prev_net_3d"),
        "net_ratio_1d": row.get("net_ratio_1d"),
        "net_ratio_3d": row.get("net_ratio_3d"),
        "net_ratio_5d": row.get("net_ratio_5d"),
        "prev_net_ratio_3d": row.get("prev_net_ratio_3d"),
        "large_ratio_5d": row.get("large_ratio_5d"),
        "est_net_15m": row.get("est_net_15m"),
        "ret_15m": row.get("ret_15m"),
        "amount_ratio_20d": row.get("amount_ratio_20d"),
        "abnormal_closeness": row.get("abnormal_closeness"),
        "abnormal_status": row.get("abnormal_status"),
        "sources": (row.get("sources") or [])[:6],
        "reasons": (row.get("reasons") or [])[:6],
        "risk_warnings": (row.get("risk_warnings") or [])[:6],
        "strategy_hits": (row.get("strategy_hits") or [])[:5],
        "trade_plan": row.get("trade_plan"),
    }


def _agent_input_payload(recommendations: dict[str, Any], review_count: int) -> dict[str, Any]:
    rows = recommendations.get("rows") or []
    tradable = [row for row in rows if row.get("decision") != "回避"]
    selected = tradable[:review_count] or rows[:review_count]
    return {
        "as_of": recommendations.get("as_of"),
        "profile": recommendations.get("profile"),
        "profile_label": recommendations.get("profile_label"),
        "weights": recommendations.get("weights"),
        "model": recommendations.get("model"),
        "data_coverage": recommendations.get("data_coverage"),
        "source_counts": recommendations.get("source_counts"),
        "sector_rotation": recommendations.get("sector_rotation"),
        "global_risk": recommendations.get("global_risk"),
        "market_gate": recommendations.get("market_gate"),
        "notes": recommendations.get("notes"),
        "candidates": [_compact_agent_row(row) for row in selected],
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("AI 返回不是 JSON 对象")
    return parsed


def _normalize_agent_review(parsed: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_symbol = {str(item.get("symbol") or ""): item for item in candidates}
    recs: list[dict[str, Any]] = []
    for item in parsed.get("recommendations") or []:
        if not isinstance(item, dict):
            continue
        symbol = _symbol_key(item.get("symbol"))
        source = by_symbol.get(symbol, {})
        trade_plan = source.get("trade_plan") if isinstance(source.get("trade_plan"), dict) else {}
        score = _float(item.get("ai_score"), _float(source.get("final_score"), 0.0))
        recs.append({
            "symbol": symbol or str(item.get("symbol") or ""),
            "name": str(item.get("name") or source.get("name") or ""),
            "agent_rating": str(item.get("agent_rating") or "观察"),
            "ai_score": round(_clamp(score), 1),
            "buy_point": str(item.get("buy_point") or trade_plan.get("entry") or "等待盘口确认"),
            "sell_point": str(item.get("sell_point") or trade_plan.get("take_profit") or "按前高或分批止盈"),
            "stop_loss": str(item.get("stop_loss") or trade_plan.get("stop_loss") or "跌破关键支撑止损"),
            "position": str(item.get("position") or trade_plan.get("position") or "单票小仓试错"),
            "holding_period": str(item.get("holding_period") or trade_plan.get("holding_period") or "1-5个交易日"),
            "reason": str(item.get("reason") or "数据共振不足，保持观察"),
            "invalidation": str(item.get("invalidation") or "资金转弱或跌破止损条件"),
        })
    return {
        "summary": str(parsed.get("summary") or "AI 已完成候选股复核。"),
        "market_bias": str(parsed.get("market_bias") or "观望"),
        "stockformer_role": str(parsed.get("stockformer_role") or "StockFormer 作为未来收益概率预测因子参与复核。"),
        "recommendations": recs,
        "risk_controls": [
            str(item) for item in (parsed.get("risk_controls") or [])
            if str(item).strip()
        ][:8],
        "disclaimer": str(parsed.get("disclaimer") or "仅供研究和模拟盘验证，不构成投资建议。"),
    }


def _fallback_agent_rating(row: dict[str, Any]) -> str:
    decision = str(row.get("decision") or "")
    risk_warnings = row.get("risk_warnings") or []
    risk_score = _float(row.get("risk_score"), 0.0)
    final_score = _float(row.get("final_score"), 0.0)
    fund_score = _float(row.get("fund_score"), 0.0)
    hard_risk = any(
        "派发" in str(item)
        or "跌停" in str(item)
        or "复权价格异常" in str(item)
        or "日K缺失" in str(item)
        or "特别处理" in str(item)
        or "退市" in str(item)
        or "市场闸门关闭" in str(item)
        for item in risk_warnings
    )
    if decision == "回避" or risk_score < 35 or hard_risk:
        return "放弃"
    if decision in {"强关注", "候选"} and final_score >= 70 and fund_score >= 60:
        return "优先关注"
    return "观察"


def _fallback_agent_review(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    picks: list[dict[str, Any]] = []
    for row in candidates:
        trade_plan = row.get("trade_plan") if isinstance(row.get("trade_plan"), dict) else {}
        rating = _fallback_agent_rating(row)
        reasons = row.get("reasons") or []
        risks = row.get("risk_warnings") or []
        model_line = (
            f"StockFormer {row.get('stockformer_score')}分，"
            f"1日/3日/5日预测分别为 {row.get('pred_ret_1d')} / {row.get('pred_ret_3d')} / {row.get('pred_ret_5d')}"
        )
        reason = "；".join([model_line, *[str(x) for x in reasons[:3]]])
        if row.get("sector_phase"):
            reason += f"；板块：{row.get('sector')}处于{row.get('sector_phase')}，板块分{row.get('sector_score')}"
        if row.get("global_risk_label"):
            reason += (
                f"；隔夜风险：{row.get('global_risk_label')}"
                f"，风险偏好{row.get('global_appetite_score')}分"
            )
        if row.get("market_gate_label"):
            reason += (
                f"；市场闸门：{row.get('market_gate_label')}"
                f"，买入确认{row.get('buy_confirmation_label')}"
            )
        if risks:
            reason += "；风险：" + "、".join(str(x) for x in risks[:3])
        picks.append({
            "symbol": row.get("symbol"),
            "name": row.get("name"),
            "agent_rating": rating,
            "ai_score": round(_float(row.get("final_score"), 0.0), 1),
            "buy_point": trade_plan.get("entry") or "等待盘口和资金确认",
            "sell_point": trade_plan.get("take_profit") or "按前高或分批止盈",
            "stop_loss": trade_plan.get("stop_loss") or "跌破关键支撑止损",
            "position": trade_plan.get("position") or ("优先关注不超过单票计划仓位；观察股只保留观察，不追高" if rating != "放弃" else "不建仓"),
            "holding_period": trade_plan.get("holding_period") or "1-5个交易日",
            "reason": reason,
            "invalidation": "资金转弱、跌破止损位、或出现派发/跌停/异动风险",
        })
    tradable = sum(1 for item in picks if item["agent_rating"] != "放弃")
    return {
        "summary": f"AI 返回未能稳定解析，已使用同一批候选数据做规则复核；{tradable}/{len(picks)} 只仍可进入观察或试错。",
        "market_bias": "轻试错" if tradable else "防守",
        "stockformer_role": "StockFormer 负责预测未来收益和概率，在兜底复核中作为模型分和方向性参考，不单独决定买卖。",
        "recommendations": picks,
        "risk_controls": [
            "只操作优先关注或观察中已触发买点的标的。",
            "出现派发、跌停、资金流出或跌破止损条件时放弃。",
            "样本外表现未稳定跑赢前，只用于模拟盘或小仓验证。",
        ],
        "disclaimer": "仅供研究和模拟盘验证，不构成投资建议。",
    }


async def build_short_term_agent_review(
    repo: Any,
    quote_service: Any = None,
    minute_refresh: Any = None,
    strategy_engine: Any = None,
    monitor_engine: Any = None,
    *,
    model_id: str | None = None,
    as_of: date | None = None,
    limit: int = 30,
    profile: str = "balanced",
    review_count: int = 5,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    review_count = max(1, min(10, int(review_count)))
    recommendations = build_short_term_recommendations(
        repo,
        quote_service,
        minute_refresh,
        strategy_engine,
        monitor_engine,
        model_id=model_id,
        as_of=as_of,
        limit=max(limit, review_count),
        profile=profile,
    )
    agent_input = _agent_input_payload(recommendations, review_count)
    candidates = agent_input.get("candidates") or []
    if not candidates:
        return _json_safe({
            "configured": ai_configured(),
            "model": current_ai_model(),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "as_of": recommendations.get("as_of"),
            "profile": recommendations.get("profile"),
            "candidate_rows": [],
            "review": None,
            "error": "暂无可复核候选股",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        })
    if not ai_configured():
        return _json_safe({
            "configured": False,
            "model": current_ai_model(),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "as_of": recommendations.get("as_of"),
            "profile": recommendations.get("profile"),
            "candidate_rows": candidates,
            "review": None,
            "error": "AI 未配置，请先在「设置 → AI」配置 API Key 和模型。",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    user_prompt = (
        "请复核下面的短线AI荐股候选。StockFormer 的核心作用是预测未来收益和上涨概率，"
        "但最终结论必须结合资金确认、技术位置、板块轮动阶段、策略共振、情绪和风险闸门。\n"
        "板块处于提前布局或低位潜伏，且 sector_setup_score 高，可以作为左侧预判加分；"
        "板块处于启动确认只代表右侧确认，不能作为追高理由；高位拥挤或退潮必须降级。\n"
        "请给出最多与输入候选数量相同的 recommendations，并保持股票代码一致。\n"
        f"输入数据：\n{json.dumps(_json_safe(agent_input), ensure_ascii=False)}"
    )
    raw_text = ""
    try:
        raw_text = await generate_ai_text(
            [
                {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=None,
            timeout=180.0,
        )
        parsed = _extract_json_object(raw_text)
        review = _normalize_agent_review(parsed, candidates)
        error = ""
    except Exception as exc:  # noqa: BLE001
        review = _fallback_agent_review(candidates)
        error = f"AI 返回未能解析，已使用规则复核兜底: {exc}"

    return _json_safe({
        "configured": True,
        "model": current_ai_model(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": recommendations.get("as_of"),
        "profile": recommendations.get("profile"),
        "candidate_rows": candidates,
        "review": review,
        "raw_text": raw_text if review is None else "",
        "error": error,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    })
