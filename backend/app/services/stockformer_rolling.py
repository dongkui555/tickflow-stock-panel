"""Rolling 5-day StockFormer portfolio validation.

This module evaluates the frozen sample-out predictions in contiguous time
windows with the production portfolio matcher.  It is deliberately research
only: the report can recommend manual activation, but it never changes the
active model pointer.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from app.backtest.engine import BacktestEngine
from app.services import stockformer as sf
from app.services import stockformer_portfolio as portfolio
from app.tickflow.repository import DataStore, KlineRepository

DEFAULT_WINDOW_DAYS = 63
DEFAULT_MIN_WINDOW_DAYS = 30
DEFAULT_REQUIRED_POSITIVE_WINDOWS = 3
REGIME_POLICIES = (
    "strict_all",
    "relax_strict",
    "adaptive_recovery",
)


def _policy_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _write_waiting_report(
    data_dir: Path,
    *,
    model_id: str,
    meta: dict,
    policy: dict,
    available_dates: list[date],
    report_filename: str | None,
    reason: str,
) -> dict:
    """Persist an explicit waiting state without replaying the old holdout."""
    no_new_data = not available_dates
    report = sf._json_safe({
        "report_id": f"rolling_5d_waiting_{time.strftime('%Y%m%d_%H%M%S')}",
        "evaluation_kind": "real_rolling_walk_forward_portfolio_v2_frozen_policy",
        "model_id": model_id,
        "model": {
            "model_name": meta.get("model_name"),
            "training_profile": meta.get("training_profile"),
            "target_mode": meta.get("target_mode"),
            "target_definition": meta.get("target_definition"),
            "oos_start": meta.get("oos_start"),
            "oos_end": meta.get("oos_end"),
        },
        "frozen_signal_policy": policy,
        "sample_out": {
            "available_start": str(min(available_dates)) if available_dates else None,
            "available_end": str(max(available_dates)) if available_dates else None,
            "trading_days": len(available_dates),
        },
        "signal_observation": {
            "new_oos_sessions": len(available_dates),
            "evaluated": False,
            "candidate_count": 0 if no_new_data else None,
            "trade_count": 0 if no_new_data else None,
            "no_trade_reason": "no_new_oos_data" if no_new_data else "incomplete_oos_window",
            "daily_observations": [],
            "note": reason,
        },
        "rolling": {
            "windows": [],
            "stability": {
                "window_count": 0,
                "positive_windows": 0,
                "trailing_positive_run": 0,
                "required_positive_windows": DEFAULT_REQUIRED_POSITIVE_WINDOWS,
            },
            "prediction_protocol": {
                "frozen_policy": True,
                "effective_from": policy.get("effective_from"),
                "old_windows_excluded": True,
            },
        },
        "activation_decision": {
            "status": "research_only",
            "eligible": False,
            "reasons": [reason],
        },
        "promotion_status": "research_only",
        "status": "waiting_for_new_oos_window",
    })
    experiments_dir = data_dir / "stockformer" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    latest_path = experiments_dir / (report_filename or "rolling_oos_latest.json")
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(latest_path)
    return report


def build_rolling_windows(
    dates: list[date],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_window_days: int = DEFAULT_MIN_WINDOW_DAYS,
) -> list[dict]:
    """Build non-overlapping chronological test windows.

    A short remainder is excluded so that a partial tail cannot count as a
    positive confirmation window.  Dates are information dates; the matcher
    receives an additional holding-period tail for the final exits.
    """
    ordered = sorted(set(dates))
    if window_days < 1 or min_window_days < 1:
        raise ValueError("window_days and min_window_days must be positive")
    windows: list[dict] = []
    index = 0
    window_index = 1
    while index + window_days <= len(ordered):
        chunk = ordered[index:index + window_days]
        windows.append({
            "window": window_index,
            "start": chunk[0],
            "end": chunk[-1],
            "trading_days": len(chunk),
            "complete": True,
        })
        window_index += 1
        index += window_days
    remainder = ordered[index:]
    if len(remainder) >= min_window_days:
        windows.append({
            "window": window_index,
            "start": remainder[0],
            "end": remainder[-1],
            "trading_days": len(remainder),
            "complete": False,
        })
    return windows


def _positive_window(stats: dict, *, minimum_trades: int) -> bool:
    return (
        int(stats.get("trade_count", 0) or 0) >= minimum_trades
        and float(stats.get("total_return", 0.0) or 0.0) > 0.0
    )


def _window_stability(windows: list[dict], *, minimum_trades: int) -> dict:
    flags = [
        _positive_window(item.get("stats") or {}, minimum_trades=minimum_trades)
        for item in windows
    ]
    longest = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    trailing = 0
    for flag in reversed(flags):
        if not flag:
            break
        trailing += 1
    return {
        "window_count": len(windows),
        "positive_windows": sum(flags),
        "positive_flags": flags,
        "longest_positive_run": longest,
        "trailing_positive_run": trailing,
        "minimum_trades_per_window": minimum_trades,
    }


def _final_stats(report: dict) -> dict:
    """Return the full rolling OOS replay of the pre-window selected policies."""
    stats = report.get("full_oos_portfolio_stats")
    return stats if isinstance(stats, dict) else {}


def _rolling_feature_columns(family: str, feature_columns: list[str]) -> list[str]:
    from app.services import stockformer_training as training

    families = {
        "full": feature_columns,
        "compact": training.COMPACT_FEATURES,
        "short_reversal": training.SHORT_REVERSAL_FEATURES,
        "trend": training.TREND_FEATURES,
        "price_volume": training.PRICE_VOLUME_FEATURES,
        **portfolio._candidate_specs(feature_columns),
    }
    return [column for column in families.get(family, ()) if column in feature_columns]


def _regime_policy_expr(signal: pl.DataFrame, policy: str) -> pl.Expr:
    """Return a market-gate predicate for a named, regime-aware policy.

    ``strict_all`` is the production predicate.  The other policies preserve
    the hard ``closed``/``observe`` block and the score floor, and only change
    how the ``strict`` regime is handled.  This makes the ablation auditable:
    no policy can turn a defensive market state into an unrestricted entry day.
    """
    if policy not in REGIME_POLICIES:
        raise ValueError(f"unknown regime policy: {policy}")
    if signal.is_empty() or "market_gate" not in signal.columns:
        return pl.lit(policy == "strict_all")
    if policy == "strict_all":
        return pl.col("_market_gate_pass")

    gate = pl.col("market_gate").cast(pl.Utf8, strict=False).fill_null("neutral")
    score = pl.col("market_regime_score").cast(pl.Float64, strict=False).fill_null(50.0)
    strong_down = pl.col("market_strong_down_pct").cast(pl.Float64, strict=False).fill_null(100.0)
    above_ma20 = pl.col("market_above_ma20_pct").cast(pl.Float64, strict=False).fill_null(0.0)
    defensive_block = (gate == "closed") | (gate == "observe")
    score_ok = score >= 60.0
    base = (~defensive_block) & score_ok
    if policy == "relax_strict":
        return base

    # In a strict regime, relax the factor conjunction only during a measured
    # recovery: breadth must be adequate and the strong-down share contained.
    recovery = (above_ma20 >= 30.0) & (strong_down <= 10.0)
    strict_factor_pass = pl.col("_market_gate_pass")
    return base & ((gate != "strict") | strict_factor_pass | recovery)


def _select_regime_policy(
    prior_windows: list[dict],
    *,
    minimum_trades: int,
) -> dict:
    """Select a policy using only completed windows before the next window."""
    if not prior_windows:
        return {
            "policy": "strict_all",
            "reason": "no_prior_window",
            "cutoff": None,
            "history_windows": 0,
            "scores": {},
        }

    # A negative completed window is a regime warning.  De-risk the next
    # window before comparing relaxed candidates; this decision uses only the
    # just-completed real fills and prevents repeated loosening after a loss.
    latest_stats = (
        ((prior_windows[-1].get("regime_policy_results") or {})
         .get(str(prior_windows[-1].get("regime_policy") or "strict_all")) or {})
        .get("selection_stats") or {}
    )
    if not _positive_window(latest_stats, minimum_trades=minimum_trades):
        return {
            "policy": "strict_all",
            "reason": "risk_off_after_negative_window",
            "cutoff": str(prior_windows[-1].get("end")) if prior_windows[-1].get("end") is not None else None,
            "history_windows": len(prior_windows),
            "scores": {},
        }

    scores: dict[str, dict] = {}
    for policy in REGIME_POLICIES:
        stats_rows = [
            ((item.get("regime_policy_results") or {}).get(policy) or {}).get("selection_stats") or {}
            for item in prior_windows
        ]
        if not stats_rows:
            continue
        equity = 1.0
        max_drawdown = 0.0
        trades = 0
        for stats in stats_rows:
            equity *= 1.0 + float(stats.get("total_return", 0.0) or 0.0) / 100.0
            max_drawdown = min(max_drawdown, float(stats.get("max_drawdown", 0.0) or 0.0))
            trades += int(stats.get("trade_count", 0) or 0)
        compounded = (equity - 1.0) * 100.0
        enough_trades = all(
            int(stats.get("trade_count", 0) or 0) >= minimum_trades
            for stats in stats_rows
        )
        score = compounded - 0.25 * abs(max_drawdown)
        scores[policy] = {
            "compounded_return": round(compounded, 4),
            "max_drawdown": round(max_drawdown, 4),
            "trade_count": trades,
            "enough_trades": enough_trades,
            "score": round(score, 4),
        }

    eligible = {
        name: value for name, value in scores.items()
        if value["enough_trades"]
    }
    pool = eligible or scores
    # Keep strict as the deterministic tie-breaker.  A policy is never chosen
    # from a future window because ``prior_windows`` is built incrementally.
    selected = max(
        pool,
        key=lambda item: (
            pool[item]["score"],
            pool[item]["compounded_return"],
            1 if item == "strict_all" else 0,
        ),
    ) if pool else "strict_all"
    last_end = prior_windows[-1].get("end")
    return {
        "policy": selected,
        "reason": "historical_policy_score",
        "cutoff": str(last_end) if last_end is not None else None,
        "history_windows": len(prior_windows),
        "scores": scores,
    }


def activation_decision(
    rolling: dict,
    final_backtest: dict,
    *,
    required_positive_windows: int,
    minimum_final_trades: int = 20,
    maximum_final_drawdown_pct: float = 10.0,
) -> dict:
    """Apply the explicit manual activation gate to research results."""
    stability = rolling.get("stability") or {}
    final_stats = _final_stats(final_backtest)
    reasons: list[str] = []
    if not final_stats:
        reasons.append("缺少所选闸门策略的滚动样本外组合回测")
    trailing = int(stability.get("trailing_positive_run", 0) or 0)
    if trailing < required_positive_windows:
        reasons.append(
            f"最近连续成本后盈利窗口仅 {trailing} 个，要求至少 {required_positive_windows} 个"
        )
    if int(stability.get("window_count", 0) or 0) < required_positive_windows:
        reasons.append("完整滚动窗口数量不足，不能确认稳定性")
    final_return = float(final_stats.get("total_return", 0.0) or 0.0)
    final_trades = int(final_stats.get("trade_count", 0) or 0)
    if final_return <= 0:
        reasons.append(f"滚动样本外组合净收益 {final_return:.2f}% 非正")
    if final_trades < minimum_final_trades:
        reasons.append(
            f"滚动样本外组合交易数 {final_trades} 少于 {minimum_final_trades} 笔"
        )
    final_drawdown = float(final_stats.get("max_drawdown", 0.0) or 0.0)
    if final_drawdown < -maximum_final_drawdown_pct:
        reasons.append(
            f"滚动样本外组合最大回撤 {final_drawdown:.2f}% 超过 {maximum_final_drawdown_pct:.2f}% 上限"
        )
    return {
        "status": "eligible_for_manual_activation" if not reasons else "research_only",
        "eligible": not reasons,
        "reasons": reasons or ["滚动窗口和样本外整体组合均满足当前研究门槛，等待人工激活"],
        "required_positive_windows": required_positive_windows,
        "maximum_final_drawdown_pct": maximum_final_drawdown_pct,
        "final_portfolio": {
            "total_return": final_return,
            "trade_count": final_trades,
            "max_drawdown": final_drawdown,
            "profit_factor": final_stats.get("profit_factor"),
        },
    }


def run_rolling_portfolio_research(
    data_dir: Path,
    *,
    model_id: str | None = None,
    feature_family: str | None = None,
    alpha: float | None = None,
    forecast_floor: float | None = None,
    max_positions: int = 3,
    initial_capital: float = 10_000.0,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_window_days: int = DEFAULT_MIN_WINDOW_DAYS,
    required_positive_windows: int = DEFAULT_REQUIRED_POSITIVE_WINDOWS,
    minimum_trades_per_window: int = 3,
    training_lookback_months: int = 18,
    min_net_edge: float = portfolio.DEFAULT_MIN_NET_EDGE,
    turnover_penalty: float = portfolio.DEFAULT_TURNOVER_PENALTY,
    holding_lag_days: int = 1,
    refit_each_window: bool = True,
    run_final_backtest: bool = True,
    report_filename: str | None = None,
    on_progress=None,
) -> dict:
    """Run contiguous 5-day rolling OOS windows and final portfolio replay."""
    if required_positive_windows < 1:
        raise ValueError("required_positive_windows must be positive")
    if minimum_trades_per_window < 1:
        raise ValueError("minimum_trades_per_window must be positive")
    if training_lookback_months not in {12, 15, 18}:
        raise ValueError("training_lookback_months must be 12, 15, or 18")
    if min_net_edge < 0 or turnover_penalty < 0 or holding_lag_days < 0:
        raise ValueError("net edge, turnover penalty, and holding lag must be non-negative")

    def progress(stage: str, pct: int, message: str) -> None:
        if on_progress is not None:
            on_progress(stage, pct, message)

    started = time.perf_counter()
    progress("load_factors", 5, "加载5日样本外预测")
    meta = sf._model_meta(data_dir, model_id)
    if meta is None:
        return {"error": "暂无 StockFormer 模型，请先训练模型"}
    resolved_model_id = str(meta.get("model_id") or model_id or "")
    frozen_signal_policy = portfolio.load_frozen_signal_policy(data_dir, meta)
    effective_from = _policy_date(
        (frozen_signal_policy or {}).get("effective_from")
    )
    factor_path = sf._resolve_artifact_path(data_dir, meta.get("factor_file"))
    if factor_path is None or not factor_path.exists():
        return {"error": "模型缺少样本外因子文件"}
    factor = pl.read_parquet(factor_path)
    prediction_column = "pred_ret_5d"
    if factor.is_empty() or prediction_column not in factor.columns:
        return {"error": "模型缺少5日样本外预测"}
    factor_dates = sorted(factor["date"].unique().to_list())
    all_factor_dates = factor_dates
    if effective_from is not None:
        factor = factor.filter(pl.col("date") >= effective_from)
        factor_dates = sorted(factor["date"].unique().to_list())
        if not factor_dates:
            return _write_waiting_report(
                data_dir,
                model_id=resolved_model_id,
                meta=meta,
                policy=frozen_signal_policy or {},
                available_dates=[],
                report_filename=report_filename,
                reason=(
                    f"冻结策略自 {effective_from.isoformat()} 生效，模型已有样本外因子文件止于 "
                    f"{max(all_factor_dates).isoformat()}；冻结后尚无新交易日预测"
                ),
            )
        if len(factor_dates) < window_days:
            progress("waiting_for_new_oos_window", 100, "新增预测交易日不足，等待完整样本外窗口")
            return _write_waiting_report(
                data_dir,
                model_id=resolved_model_id,
                meta=meta,
                policy=frozen_signal_policy or {},
                available_dates=factor_dates,
                report_filename=report_filename,
                reason=(
                    f"冻结日后仅积累 {len(factor_dates)} 个预测交易日，"
                    f"尚未形成完整 {window_days} 交易日样本外窗口；候选和成交未评估"
                ),
            )
    symbols = [str(value) for value in factor["symbol"].unique().to_list() if value]
    start = factor_dates[0]
    end = factor_dates[-1]
    data_start, _ = sf._date_range_from_data(data_dir)
    history_start = max(data_start or (start - timedelta(days=730)), start - timedelta(days=730))
    panel_end = end + timedelta(days=window_days)

    progress("load_panel", 15, "加载滚动回测行情面板")
    repo = KlineRepository(DataStore(data_dir))
    engine = BacktestEngine(repo)
    training_profile = str(meta.get("training_profile") or "daily_baseline")
    panel = sf._load_panel(
        data_dir,
        history_start,
        panel_end,
        symbols=symbols,
        include_minute=training_profile == "minute_enhanced",
    )
    if panel.is_empty():
        return {"error": "真实组合回测缺少行情面板"}

    progress("build_features", 30, "构建生产闸门所需特征")
    from app.services import stockformer_training as training

    try:
        feature_frame, feature_coverage = portfolio._prepare_research_features(
            panel, training_profile
        )
    except ValueError as exc:
        return {"error": str(exc)}
    if feature_frame.is_empty():
        return {
            "error": (
                "分钟增强回测没有通过分钟密度覆盖门槛的股票日"
                f"（至少 {sf.MINUTE_MIN_VALID_BARS} 根，覆盖率 = 1.0）"
            )
        }
    feature_columns = [
        column for column in sf.training_feature_columns(training_profile)
        if column in feature_frame.columns
    ]
    factor = factor.join(feature_frame.select("symbol", "date"), on=["symbol", "date"], how="inner")
    if factor.is_empty():
        return {"error": "样本外预测没有通过分钟覆盖门槛的股票日"}
    factor_dates = sorted(factor["date"].unique().to_list())
    start = factor_dates[0]
    end = factor_dates[-1]
    all_windows = build_rolling_windows(
        factor_dates, window_days=window_days, min_window_days=min_window_days
    )
    windows = [item for item in all_windows if item.get("complete")]
    pending_windows = [item for item in all_windows if not item.get("complete")]
    if not windows:
        return _write_waiting_report(
            data_dir,
            model_id=resolved_model_id,
            meta=meta,
            policy=frozen_signal_policy or {},
            available_dates=factor_dates,
            report_filename=report_filename,
            reason=(
                f"冻结日后仅积累 {len(factor_dates)} 个有效交易日，"
                f"尚未形成完整 {window_days} 交易日样本外窗口"
            ),
        )
    if forecast_floor is None:
        rule = (frozen_signal_policy or {}).get("prediction_rule") or {}
        forecast_floor = float(rule.get("calibrated_floor", 0.0))
    gated = portfolio._build_gated_signal_frame(
        data_dir, factor, feature_frame, meta, horizon=5,
        forecast_floor=forecast_floor,
        signal_policy=frozen_signal_policy,
        min_net_edge=min_net_edge,
        rank_predictions=False,
    )
    if gated.is_empty():
        return {"error": "5日样本外预测经过特征和闸门构建后为空"}

    progress("rolling_replay", 40, "逐窗口执行真实成交撮合")
    # The production 5-day model family and regularization are selected before
    # the frozen holdout.  Reusing those choices while refitting at each
    # window boundary prevents later labels from influencing earlier windows.
    model_selection = ((meta.get("validation") or {}).get("horizons") or {}).get("5d") or {}
    selected_family = str(feature_family or model_selection.get("family") or "trend")
    selected_alpha = float(alpha if alpha is not None else (model_selection.get("alpha") or 100.0))
    selected_columns = _rolling_feature_columns(selected_family, feature_columns)
    if not selected_columns:
        return {"error": f"5日特征族不可用: {selected_family}"}
    target_mode = str(meta.get("target_mode") or "absolute")
    window_results: list[dict] = []
    selected_signal_frames: list[pl.DataFrame] = []
    for index, window in enumerate(windows, start=1):
        window_factor = factor.filter(
            (pl.col("date") >= window["start"]) & (pl.col("date") <= window["end"])
        )
        fit_meta = {
            "protocol": "frozen_oos_predictions" if not refit_each_window else "purged_refit_before_window",
            "fit_end_exclusive": str(window["start"]),
            "target_mode": target_mode,
            "feature_family": selected_family,
            "alpha": selected_alpha,
            "feature_columns": selected_columns,
            "minute_feature_count": sum(column in sf.MINUTE_FEATURE_COLUMNS for column in selected_columns),
            "training_lookback_months": training_lookback_months,
            "min_net_edge": min_net_edge,
            "turnover_penalty": turnover_penalty,
            "holding_lag_days": holding_lag_days,
        }
        if refit_each_window:
            fit_start = window["start"] - timedelta(days=training_lookback_months * 31)
            fit_meta["fit_start"] = str(fit_start)
            prediction = portfolio._fit_candidate_predictions(
                feature_frame,
                horizon=5,
                columns=selected_columns,
                alpha=selected_alpha,
                fit_start=fit_start,
                fit_end=window["start"],
                predict_start=window["start"],
                predict_end=window["end"] + timedelta(days=1),
                target_mode=target_mode,
            )
            pred_col = "pred_ret_5d"
            if prediction.is_empty():
                signal_window = pl.DataFrame()
                fit_meta["prediction_rows"] = 0
                fit_meta["fit_rows"] = 0
                fit_meta["fit_dates"] = 0
            else:
                window_factor = window_factor.select(
                    [column for column in window_factor.columns if column != pred_col]
                ).join(prediction, on=["symbol", "date"], how="inner")
                signal_window = portfolio._build_gated_signal_frame(
                    data_dir, window_factor, feature_frame, meta, horizon=5,
                    forecast_floor=forecast_floor,
                    signal_policy=frozen_signal_policy,
                    min_net_edge=min_net_edge,
                    rank_predictions=False,
                )
                fit_meta["prediction_rows"] = int(prediction.height)
                fit_rows = training.purged_slice(
                    feature_frame,
                    5,
                    fit_start,
                    window["start"],
                )
                fit_meta["fit_rows"] = int(fit_rows.height)
                fit_meta["fit_dates"] = int(fit_rows["date"].n_unique()) if not fit_rows.is_empty() else 0
        else:
            signal_window = gated.filter(
                (pl.col("date") >= window["start"]) & (pl.col("date") <= window["end"])
            )
            fit_meta["prediction_rows"] = int(signal_window.height)
        panel_window = panel.filter(
            (pl.col("date") >= window["start"])
            & (pl.col("date") <= window["end"] + timedelta(days=window_days))
        )
        policy_selection = _select_regime_policy(
            window_results,
            minimum_trades=minimum_trades_per_window,
        )
        policy_frames: dict[str, pl.DataFrame] = {}
        policy_results: dict[str, dict] = {}
        window_dates = [
            day for day in factor_dates
            if window["start"] <= day <= window["end"]
        ]
        selection_end = window_dates[-7] if len(window_dates) > 6 else None
        for policy in REGIME_POLICIES:
            policy_frame = signal_window.with_columns(
                _regime_policy_expr(signal_window, policy).alias("_market_gate_pass")
            )
            policy_frames[policy] = policy_frame
            policy_results[policy] = {
                "stats": portfolio._run_one(
                    panel_window,
                    policy_frame,
                    engine,
                    horizon=5,
                    forecast_floor=forecast_floor,
                    max_positions=max_positions,
                    initial_capital=initial_capital,
                    signal_frame=policy_frame,
                    turnover_penalty=turnover_penalty,
                    holding_lag_days=holding_lag_days,
                ),
                "gating": portfolio._gated_layer_stats(policy_frame, prediction_column),
            }
            if selection_end is not None:
                settled_frame = policy_frame.filter(pl.col("date") <= selection_end)
                policy_results[policy]["selection_stats"] = portfolio._run_one(
                    panel_window.filter(pl.col("date") <= window["end"]),
                    settled_frame,
                    engine,
                    horizon=5,
                    forecast_floor=forecast_floor,
                    max_positions=max_positions,
                    initial_capital=initial_capital,
                    signal_frame=settled_frame,
                    turnover_penalty=turnover_penalty,
                    holding_lag_days=holding_lag_days,
                )
                policy_results[policy]["selection_signal_end"] = str(selection_end)
        selected_policy = str(policy_selection["policy"])
        selected_signal_window = policy_frames[selected_policy]
        selected_stats = policy_results[selected_policy]["stats"]
        strict_stats = policy_results["strict_all"]["stats"]
        regime_breakdown: dict[str, dict] = {}
        if "market_gate" in signal_window.columns:
            for regime in ("closed", "observe", "strict", "open", "neutral"):
                regime_signal = selected_signal_window.filter(pl.col("market_gate") == regime)
                regime_stats = portfolio._run_one(
                    panel_window,
                    regime_signal,
                    engine,
                    horizon=5,
                    forecast_floor=forecast_floor,
                    max_positions=max_positions,
                    initial_capital=initial_capital,
                    signal_frame=regime_signal,
                    turnover_penalty=turnover_penalty,
                    holding_lag_days=holding_lag_days,
                )
                regime_rows = signal_window.filter(pl.col("market_gate") == regime)
                eligible_rows = regime_signal.filter(
                    (pl.col(prediction_column).fill_null(-1.0) > forecast_floor)
                    & pl.col("_entry_rules_pass")
                    & pl.col("_market_gate_pass")
                )
                regime_breakdown[regime] = {
                    "dates": int(regime_rows["date"].n_unique()) if not regime_rows.is_empty() else 0,
                    "signal_rows": int(regime_signal.height),
                    "eligible_rows": int(eligible_rows.height),
                    "stats": regime_stats,
                }
        # Keep the unrestricted ablation for comparison, but never use it for
        # policy selection or activation.
        relaxed_signal_window = signal_window.with_columns(
            pl.lit(True).alias("_market_gate_pass")
        )
        relaxed_stats = portfolio._run_one(
            panel_window,
            relaxed_signal_window,
            engine,
            horizon=5,
            forecast_floor=forecast_floor,
            max_positions=max_positions,
            initial_capital=initial_capital,
            signal_frame=relaxed_signal_window,
            turnover_penalty=turnover_penalty,
            holding_lag_days=holding_lag_days,
        )
        selected_signal_frames.append(selected_signal_window)
        layer_stats = portfolio._gated_layer_stats(signal_window, prediction_column)
        window_results.append({
            **window,
            "start": str(window["start"]),
            "end": str(window["end"]),
            "model_fit": fit_meta,
            "stats": selected_stats,
            "strict_baseline_stats": strict_stats,
            "relaxed_market_gate_stats": relaxed_stats,
            "gating": layer_stats,
            "regime_policy": selected_policy,
            "policy_selection": policy_selection,
            "regime_policy_results": policy_results,
            "regime_breakdown": regime_breakdown,
            "positive_after_cost": _positive_window(
                selected_stats, minimum_trades=minimum_trades_per_window
            ),
            "relaxed_market_gate_positive_after_cost": _positive_window(
                relaxed_stats, minimum_trades=minimum_trades_per_window
            ),
        })
        progress(
            "rolling_replay",
            40 + int(40 * index / max(1, len(windows))),
            f"已完成第 {index}/{len(windows)} 个滚动窗口",
        )

    stability = _window_stability(
        window_results, minimum_trades=minimum_trades_per_window
    )
    rolling = {
        "horizon": 5,
        "window_days": window_days,
        "min_window_days": min_window_days,
        "windows": window_results,
        "pending_windows": pending_windows,
        "stability": stability,
        "execution": {
            "entry_fill": "open_t+1",
            "exit_fill": "open_t+1",
            "t_plus_one": True,
            "lot_size": 100,
            "commission_pct": 0.0003,
            "stamp_tax_pct": 0.0005,
            "slippage_bps_each_side": 5.0,
            "limit_up_down_blocked": True,
        },
        "selection_policy": {
            "min_net_edge": min_net_edge,
            "turnover_penalty": turnover_penalty,
            "holding_lag_days": holding_lag_days,
            "training_lookback_months": training_lookback_months,
            "ranking_column": "net_margin_score",
        },
        "prediction_protocol": {
            "kind": "purged_refit_before_each_window" if refit_each_window else "frozen_oos_predictions",
            "refit_each_window": bool(refit_each_window),
            "feature_family": selected_family,
            "feature_columns": selected_columns,
            "minute_feature_count": sum(column in sf.MINUTE_FEATURE_COLUMNS for column in selected_columns),
            "training_profile": training_profile,
            "feature_coverage": feature_coverage,
            "alpha": selected_alpha,
            "forecast_floor": forecast_floor,
            "frozen_signal_policy": frozen_signal_policy,
            "effective_from": str(effective_from) if effective_from else None,
            "old_windows_excluded": bool(effective_from),
            "target_mode": target_mode,
            "fit_end_exclusive": "window_start",
        },
        "market_gate_ablation": {
            "activation_basis": "strict",
            "relaxed_definition": "only _market_gate_pass is forced true; prediction, entry, ranking and execution rules are unchanged",
        },
        "regime_policy": {
            "candidates": list(REGIME_POLICIES),
            "selection_protocol": "prior_completed_windows_only",
            "closed_and_observe_blocked_for_relaxed_policies": True,
            "strict_regime_recovery_conditions": {
                "market_above_ma20_pct_min": 30.0,
                "market_strong_down_pct_max": 10.0,
            },
            "activation_basis": "selected_policy_per_window",
            "selection_settlement_tail_days": 6,
            "selection_uses_settled_prefix_only": True,
        },
    }

    progress("final_backtest", 84, "执行滚动样本外整体组合回测")
    rolling_signal_start = windows[0]["start"]
    rolling_signal_end = windows[-1]["end"]
    final_panel = panel.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end + timedelta(days=window_days))
    )
    selected_signal_all = (
        pl.concat(selected_signal_frames, how="vertical_relaxed")
        if selected_signal_frames
        else pl.DataFrame()
    )
    full_oos_portfolio_stats = portfolio._run_one(
        final_panel,
        selected_signal_all,
        engine,
        horizon=5,
        forecast_floor=forecast_floor,
        max_positions=max_positions,
        initial_capital=initial_capital,
        signal_frame=selected_signal_all,
        turnover_penalty=turnover_penalty,
        holding_lag_days=holding_lag_days,
    )
    reference_backtest = (
        portfolio.run_portfolio_research(
            data_dir,
            model_id=resolved_model_id,
            max_positions=max_positions,
            initial_capital=initial_capital,
            on_progress=on_progress,
        )
        if run_final_backtest and not frozen_signal_policy
        else {
            "skipped": True,
            "reason": "frozen_policy_new_oos_only" if frozen_signal_policy else "requested",
        }
    )
    final_backtest = {
        "full_oos_portfolio_stats": full_oos_portfolio_stats,
        "full_oos_period": {
            "signal_start": str(rolling_signal_start),
            "signal_end": str(rolling_signal_end),
            "panel_end": str(final_panel["date"].max()),
        },
        "reference_backtest": reference_backtest,
        "selected_policy_schedule": [
            {
                "window": item["window"],
                "start": item["start"],
                "end": item["end"],
                "policy": item["regime_policy"],
                "selection_cutoff": (item.get("policy_selection") or {}).get("cutoff"),
            }
            for item in window_results
        ],
        "activation_basis": "full_oos_portfolio_stats",
        "reference_backtest_role": "comparison_only_different_model_and_threshold",
    }
    decision = activation_decision(
        rolling,
        final_backtest,
        required_positive_windows=required_positive_windows,
    )
    report = sf._json_safe({
        "report_id": f"rolling_5d_{time.strftime('%Y%m%d_%H%M%S')}",
        "evaluation_kind": "real_rolling_walk_forward_portfolio_v2_regime_policy",
        "model_id": resolved_model_id,
        "model": {
            "model_name": meta.get("model_name"),
            "training_profile": meta.get("training_profile"),
            "target_mode": meta.get("target_mode"),
            "target_definition": meta.get("target_definition"),
            "oos_start": meta.get("oos_start"),
            "oos_end": meta.get("oos_end"),
        },
        "frozen_signal_policy": frozen_signal_policy,
        "sample_out": {"start": str(start), "end": str(end), "trading_days": len(factor_dates)},
        "rolling": rolling,
        "final_backtest": final_backtest,
        "activation_decision": decision,
        "promotion_status": "research_only",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    })
    experiments_dir = data_dir / "stockformer" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    latest_path = experiments_dir / (report_filename or "rolling_oos_latest.json")
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    timestamped_path = experiments_dir / f"{report['report_id']}.json"
    if timestamped_path != latest_path:
        timestamped_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("done", 100, f"5日滚动样本外完成：{decision['status']}")
    report["report_path"] = str(latest_path)
    return report
