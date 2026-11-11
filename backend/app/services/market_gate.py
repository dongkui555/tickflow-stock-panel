"""Market environment gate for short-term entry decisions."""
from __future__ import annotations

import math
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from app.services import regime_builder
from app.services.market_phase import PHASE_LABELS

STATE_LABELS = regime_builder.STATE_LABELS

_CACHE_TTL = 30.0
_CACHE: dict[tuple[str, str, bool], tuple[float, dict[str, Any]]] = {}


def _float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _round(value: Any, digits: int = 4) -> float | None:
    out = _float(value, None)
    return None if out is None else round(out, digits)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _as_percent(value: Any, default: float | None = 0.0) -> float | None:
    out = _float(value, None)
    if out is None:
        return default
    return out * 100.0 if abs(out) <= 1.5 else out


def _latest_regime_row(data_dir: Path, as_of: date | None) -> dict[str, Any] | None:
    df = regime_builder.load_regime_history(data_dir)
    if df.is_empty() or "date" not in df.columns:
        return None
    if df.schema.get("date") != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date, strict=False).alias("date"))
    df = df.drop_nulls("date")
    if df.is_empty():
        return None
    target = as_of or df["date"].max()
    if target is not None:
        df = df.filter(pl.col("date") <= target)
    if df.is_empty():
        return None
    rows = df.sort("date", descending=True).head(1).to_dicts()
    return rows[0] if rows else None


def _safe_overview(repo: Any, quote_service: Any, as_of: date | None) -> dict[str, Any] | None:
    if repo is None:
        return None
    try:
        from app.services.market_overview_builder import build_market_overview

        return build_market_overview(repo, quote_service, None, as_of)
    except Exception:  # noqa: BLE001
        return None


def _local_adjustment(gate: str) -> dict[str, Any]:
    if gate == "closed":
        return {
            "score_adjustment": -10.0,
            "risk_score_adjustment": -18.0,
            "entry_cap": 55.0,
            "final_score_cap": 55.0,
            "position_multiplier": 0.0,
            "buy_threshold_add": 12.0,
            "min_support_count": 99,
            "requires_intraday_confirmation": True,
            "requires_sector_pre_layout": True,
            "block_new_position": True,
            "open_rule": "大盘弱势闸门关闭，暂停新开仓，只处理已有持仓风控。",
        }
    if gate == "observe":
        return {
            "score_adjustment": -6.0,
            "risk_score_adjustment": -12.0,
            "entry_cap": 62.0,
            "final_score_cap": 62.0,
            "position_multiplier": 0.25,
            "buy_threshold_add": 8.0,
            "min_support_count": 5,
            "requires_intraday_confirmation": True,
            "requires_sector_pre_layout": True,
            "block_new_position": True,
            "open_rule": "弱市只观察，等指数止跌、上涨家数修复且盘中资金回流后再考虑。",
        }
    if gate == "strict":
        return {
            "score_adjustment": -2.0,
            "risk_score_adjustment": -5.0,
            "entry_cap": 72.0,
            "final_score_cap": 72.0,
            "position_multiplier": 0.50,
            "buy_threshold_add": 4.0,
            "min_support_count": 4,
            "requires_intraday_confirmation": True,
            "requires_sector_pre_layout": False,
            "block_new_position": False,
            "open_rule": "震荡或偏弱环境只做强共振，小仓试错，不能开盘追高。",
        }
    if gate == "open":
        return {
            "score_adjustment": 1.0,
            "risk_score_adjustment": 2.0,
            "entry_cap": None,
            "final_score_cap": None,
            "position_multiplier": 1.0,
            "buy_threshold_add": -1.0,
            "min_support_count": 3,
            "requires_intraday_confirmation": False,
            "requires_sector_pre_layout": False,
            "block_new_position": False,
            "open_rule": "市场强势，可按计划执行，但仍需个股资金和盘口确认。",
        }
    return {
        "score_adjustment": 0.0,
        "risk_score_adjustment": 0.0,
        "entry_cap": None,
        "final_score_cap": None,
        "position_multiplier": 0.8,
        "buy_threshold_add": 2.0,
        "min_support_count": 3,
        "requires_intraday_confirmation": False,
        "requires_sector_pre_layout": False,
        "block_new_position": False,
        "open_rule": "市场中性，按原买点纪律执行。",
    }


def _classify_gate(
    *,
    local_score: float,
    state: str,
    phase: str,
    up_pct: float,
    down_pct: float,
    strong_down_pct: float,
    above_ma20_pct: float,
    limit_up: int,
    limit_down: int,
    index_pct: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if state == "weak":
        reasons.append("市场状态弱势")
    elif state == "lean_weak":
        reasons.append("市场状态偏弱")
    if phase == "ebb":
        reasons.append("情绪周期退潮")
    elif phase == "climax":
        reasons.append("情绪处于高潮，追高回撤风险升高")
    if local_score < 45:
        reasons.append(f"A股环境分{local_score:.1f}低于45")
    if down_pct >= 62:
        reasons.append(f"下跌家数占比{down_pct:.1f}%")
    if strong_down_pct >= 12:
        reasons.append(f"大跌股占比{strong_down_pct:.1f}%")
    if above_ma20_pct <= 35:
        reasons.append(f"MA20上方占比{above_ma20_pct:.1f}%")
    if limit_down >= max(20, limit_up * 0.85):
        reasons.append(f"跌停结构偏弱，涨停/跌停 {limit_up}/{limit_down}")
    if index_pct <= -1.2:
        reasons.append(f"核心指数跌幅{index_pct:.2f}%")

    hard_weak = (
        state == "weak"
        or local_score < 30
        or down_pct >= 70
        or strong_down_pct >= 18
        or (limit_down >= 40 and limit_down >= limit_up)
        or index_pct <= -2.0
    )
    weak = (
        state == "lean_weak"
        or phase == "ebb"
        or local_score < 45
        or down_pct >= 62
        or strong_down_pct >= 12
        or above_ma20_pct <= 35
        or (limit_down >= max(20, limit_up * 0.85))
        or index_pct <= -1.2
    )
    strict = (
        state == "range"
        or local_score < 55
        or phase == "climax"
        or up_pct < 48
        or above_ma20_pct < 45
    )
    if hard_weak:
        return "closed", reasons
    if weak:
        return "observe", reasons
    if strict:
        return "strict", reasons or ["市场未进入顺风状态"]
    if state in {"strong", "lean_strong"} and phase in {"ignite", "rally", "repair"} and local_score >= 60:
        return "open", reasons or ["市场赚钱效应可用"]
    return "neutral", reasons or ["市场整体中性"]


def build_market_gate(
    data_dir: Path,
    *,
    repo: Any | None = None,
    quote_service: Any | None = None,
    as_of: date | None = None,
    use_overview: bool = True,
) -> dict[str, Any]:
    """Build an A-share market gate for buy confirmation and position sizing."""
    cache_key = (str(data_dir), as_of.isoformat() if as_of else "latest", bool(use_overview and as_of is None))
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] <= _CACHE_TTL:
        out = dict(cached[1])
        out["cache_hit"] = True
        return out

    regime = _latest_regime_row(data_dir, as_of)
    overview = _safe_overview(repo, quote_service, as_of) if use_overview else None

    regime_score = _float((regime or {}).get("score"), None)
    emotion_score = _float(((overview or {}).get("emotion") or {}).get("score"), None)
    if regime_score is not None and emotion_score is not None:
        local_score = _clamp(regime_score * 0.65 + emotion_score * 0.35)
    elif regime_score is not None:
        local_score = _clamp(regime_score)
    elif emotion_score is not None:
        local_score = _clamp(emotion_score)
    else:
        local_score = 50.0

    state = str((regime or {}).get("state") or "")
    phase = str((regime or {}).get("phase") or "")
    breadth = (overview or {}).get("breadth") or {}
    trend = (overview or {}).get("trend") or {}
    limit = (overview or {}).get("limit") or {}
    if not state:
        if local_score >= 70:
            state = "strong"
        elif local_score >= 55:
            state = "lean_strong"
        elif local_score >= 45:
            state = "range"
        elif local_score >= 30:
            state = "lean_weak"
        else:
            state = "weak"

    up_count = _float((regime or {}).get("up_count"), 0.0) or 0.0
    down_count = _float((regime or {}).get("down_count"), 0.0) or 0.0
    total_count = up_count + down_count
    regime_up_pct = up_count / total_count * 100.0 if total_count > 0 else None
    regime_down_pct = down_count / total_count * 100.0 if total_count > 0 else None
    up_pct = _float(breadth.get("up_pct"), regime_up_pct if regime_up_pct is not None else 50.0) or 50.0
    down_pct = _float(breadth.get("down_pct"), regime_down_pct if regime_down_pct is not None else 50.0) or 50.0
    strong_down_pct = _float(
        breadth.get("strong_down"),
        _as_percent((regime or {}).get("strong_down_pct"), 0.0),
    ) or 0.0
    overview_total = _float(breadth.get("total"), None)
    if overview_total and strong_down_pct > 100:
        strong_down_pct = strong_down_pct / overview_total * 100.0
    above_ma20_pct = _float(
        trend.get("above_ma20_pct"),
        _as_percent((regime or {}).get("above_ma20_pct"), None),
    )
    if not trend and (regime or {}).get("ma20_valid_count") == 0:
        above_ma20_pct = None
    limit_up = int(_float(limit.get("limit_up"), (regime or {}).get("limit_up", 0)) or 0)
    limit_down = int(_float(limit.get("limit_down"), (regime or {}).get("limit_down", 0)) or 0)
    index_pct = _as_percent((regime or {}).get("index_pct"), 0.0)
    if overview and overview.get("indices"):
        changes = [_float(item.get("change_pct"), None) for item in overview.get("indices") or [] if isinstance(item, dict)]
        changes = [float(v) for v in changes if v is not None]
        if changes:
            index_pct = sum(changes) / len(changes)

    gate, reasons = _classify_gate(
        local_score=local_score,
        state=state,
        phase=phase,
        up_pct=up_pct,
        down_pct=down_pct,
        strong_down_pct=strong_down_pct,
        above_ma20_pct=above_ma20_pct if above_ma20_pct is not None else 50.0,
        limit_up=limit_up,
        limit_down=limit_down,
        index_pct=index_pct,
    )
    if above_ma20_pct is None:
        gate = "closed"
        reasons.insert(0, "MA20有效样本不足，等待市场环境数据核验")
    adjustment = _local_adjustment(gate)
    gate_labels = {
        "open": "可开仓",
        "neutral": "正常观察",
        "strict": "小仓试错",
        "observe": "只观察",
        "closed": "暂停开仓",
    }
    payload = _json_safe({
        "as_of": as_of.isoformat() if as_of else (overview or {}).get("as_of") or (regime or {}).get("date"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gate": gate,
        "gate_label": gate_labels.get(gate, gate),
        "local_score": round(local_score, 1),
        "state": state,
        "state_label": STATE_LABELS.get(state, state or "未知"),
        "phase": phase,
        "phase_label": PHASE_LABELS.get(phase, phase or "未知"),
        "recommendation_adjustment": adjustment,
        "metrics": {
            "regime_score": _round(regime_score, 1),
            "emotion_score": _round(emotion_score, 1),
            "up_pct": round(up_pct, 2),
            "down_pct": round(down_pct, 2),
            "strong_down_pct": round(strong_down_pct, 2),
            "above_ma20_pct": _round(above_ma20_pct, 2),
            "limit_up": limit_up,
            "limit_down": limit_down,
            "index_pct": round(index_pct, 4),
        },
        "reasons": reasons[:8],
        "rules": [
            "弱势/退潮/大跌扩散时暂停新开仓。",
            "震荡或偏弱环境只允许强共振小仓试错。",
            "买点必须同时通过个股资金、技术位置、板块和风险确认。",
            "大盘未修复前，模型高分只作为观察名单，不直接变成买入信号。",
        ],
        "cache_hit": False,
    })
    _CACHE[cache_key] = (now, dict(payload))
    return payload


def market_gate_frame(data_dir: Path) -> pl.DataFrame:
    """Return date-level gate columns for strict historical validation."""
    df = regime_builder.load_regime_history(data_dir)
    if df.is_empty() or "date" not in df.columns:
        return pl.DataFrame()
    if df.schema.get("date") != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date, strict=False).alias("date"))
    for column, default in {
        "state": "range",
        "phase": "repair",
        "score": 50.0,
        "strong_down_pct": 0.0,
        "above_ma20_pct": None,
        "limit_up": 0.0,
        "limit_down": 0.0,
        "index_pct": 0.0,
    }.items():
        if column not in df.columns:
            df = df.with_columns(pl.lit(default).alias(column))
    strong_down = pl.col("strong_down_pct").cast(pl.Float64, strict=False).fill_null(0.0)
    above_ma20 = pl.col("above_ma20_pct").cast(pl.Float64, strict=False)
    breadth_available = above_ma20.is_finite().fill_null(False)
    if "ma20_valid_count" in df.columns:
        breadth_available = breadth_available & (
            pl.col("ma20_valid_count").is_null() | (pl.col("ma20_valid_count") > 0)
        )
    above_ma20_pct = pl.when(above_ma20.abs() <= 1.5).then(above_ma20 * 100.0).otherwise(above_ma20)
    score = pl.col("score").cast(pl.Float64, strict=False).fill_null(50.0)
    state = pl.col("state").cast(pl.Utf8, strict=False).fill_null("range")
    phase = pl.col("phase").cast(pl.Utf8, strict=False).fill_null("repair")
    limit_up = pl.col("limit_up").cast(pl.Float64, strict=False).fill_null(0.0)
    limit_down = pl.col("limit_down").cast(pl.Float64, strict=False).fill_null(0.0)
    index_pct_raw = pl.col("index_pct").cast(pl.Float64, strict=False).fill_null(0.0)
    index_pct = pl.when(index_pct_raw.abs() <= 1.5).then(index_pct_raw * 100.0).otherwise(index_pct_raw)
    hard_weak = (
        (state == "weak")
        | (score < 30.0)
        | (strong_down >= 18.0)
        | ((limit_down >= 40.0) & (limit_down >= limit_up))
        | (index_pct <= -2.0)
    )
    weak = (
        (state == "lean_weak")
        | (phase == "ebb")
        | (score < 45.0)
        | (strong_down >= 12.0)
        | (above_ma20_pct <= 35.0)
        | ((limit_down >= pl.max_horizontal(pl.lit(20.0), limit_up * 0.85)))
        | (index_pct <= -1.2)
    )
    strict = (
        (state == "range")
        | (score < 55.0)
        | (phase == "climax")
        | (above_ma20_pct < 45.0)
    )
    gate = (
        pl.when(~breadth_available | hard_weak).then(pl.lit("closed"))
        .when(weak).then(pl.lit("observe"))
        .when(strict).then(pl.lit("strict"))
        .otherwise(pl.lit("open"))
    )
    return (
        df.with_columns(
            gate.alias("market_gate"),
            score.alias("market_regime_score"),
            above_ma20_pct.alias("market_above_ma20_pct"),
            strong_down.alias("market_strong_down_pct"),
        )
        .with_columns(
            pl.when(pl.col("market_gate") == "closed").then(0.0)
            .when(pl.col("market_gate") == "observe").then(0.25)
            .when(pl.col("market_gate") == "strict").then(0.50)
            .otherwise(1.0)
            .alias("market_position_multiplier"),
            pl.when(pl.col("market_gate") == "closed").then(12.0)
            .when(pl.col("market_gate") == "observe").then(8.0)
            .when(pl.col("market_gate") == "strict").then(4.0)
            .otherwise(0.0)
            .alias("market_buy_threshold_add"),
            pl.when(pl.col("market_gate") == "closed").then(False)
            .when(pl.col("market_gate") == "observe").then(False)
            .otherwise(True)
            .alias("market_allows_open"),
        )
        .select([
            "date",
            "market_gate",
            "market_regime_score",
            "market_above_ma20_pct",
            "market_strong_down_pct",
            "market_position_multiplier",
            "market_buy_threshold_add",
            "market_allows_open",
        ])
    )
