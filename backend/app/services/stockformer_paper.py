"""Frozen-policy shadow tracking for the next StockFormer OOS window."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from app.backtest.engine import BacktestEngine
from app.services import stockformer as sf
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_rolling as rolling
from app.services.stockformer_pre_oos import SETTLEMENT_DAYS, WINDOW_DAYS
from app.tickflow.repository import DataStore, KlineRepository


def _empty_layer_stats() -> dict:
    """Return the same shape as portfolio layer diagnostics for an empty day."""
    return {
        "total_rows": 0,
        "prediction_pass": {"rows": 0, "dates": 0},
        "net_margin_pass": {"rows": 0, "dates": 0},
        "entry_rules_pass": {"rows": 0, "dates": 0},
        "market_gate_pass": {"rows": 0, "dates": 0},
        "cumulative": {
            "prediction_and_entry": 0,
            "prediction_and_market": 0,
            "prediction_entry_market": 0,
            "prediction_net_margin": 0,
            "prediction_net_margin_entry_market": 0,
        },
    }


def _daily_observations(
    signal: pl.DataFrame,
    *,
    available_dates: list[date],
    pred_col: str,
    max_positions: int,
    forecast_ceiling: float | None,
) -> tuple[list[dict], dict[str, int]]:
    """Create an auditable record for every observed session, including zeroes."""
    selected_by_day: dict[date, int] = {}
    if not signal.is_empty():
        gated = signal.filter(
            pl.col("_forecast_pass")
            & pl.col("_entry_rules_pass")
            & pl.col("_market_gate_pass")
        )
        if forecast_ceiling is not None:
            gated = gated.filter(pl.col(pred_col) <= float(forecast_ceiling))
        if not gated.is_empty():
            selected = (
                gated.with_columns(
                    pl.col("net_margin_score").rank(
                        method="ordinal", descending=True,
                    ).over("date").alias("_rank")
                )
                .filter(pl.col("_rank") <= int(max_positions))
            )
            selected_by_day = {
                row["date"]: int(row["count"])
                for row in selected.group_by("date").len(name="count").to_dicts()
            }

    observations: list[dict] = []
    reason_counts: dict[str, int] = {}
    available_set = set(available_dates)
    signal_dates = set(signal["date"].unique().to_list()) if not signal.is_empty() else set()
    for day in sorted(available_set | signal_dates):
        day_frame = signal.filter(pl.col("date") == day) if not signal.is_empty() else pl.DataFrame()
        stats = (
            portfolio._gated_layer_stats(day_frame, pred_col)
            if not day_frame.is_empty()
            else _empty_layer_stats()
        )
        total = int(stats["total_rows"])
        prediction = int(stats["prediction_pass"]["rows"])
        prediction_net = int(stats["cumulative"]["prediction_net_margin"])
        prediction_net_entry = int(stats["cumulative"]["prediction_net_margin_entry_market"])
        selected = int(selected_by_day.get(day, 0))
        if total == 0:
            reason = "feature_rows_empty"
        elif prediction == 0:
            reason = "prediction_gate_empty"
        elif prediction_net == 0:
            reason = "net_margin_empty"
        elif prediction_net_entry == 0:
            prediction_net_entry_only = int(
                day_frame.filter(
                    pl.col("_prediction_pass")
                    & pl.col("_net_margin_pass")
                    & pl.col("_entry_rules_pass")
                ).height
            )
            reason = "entry_rules_empty" if prediction_net_entry_only == 0 else "market_gate_empty"
        elif forecast_ceiling is not None and selected == 0:
            reason = "forecast_ceiling_empty"
        elif selected == 0:
            reason = "candidate_selection_empty"
        else:
            reason = "candidate_selected"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        observations.append({
            "date": str(day),
            "available": day in available_set,
            "total_rows": total,
            "prediction_pass": prediction,
            "net_margin_pass": int(stats["net_margin_pass"]["rows"]),
            "entry_rules_pass": int(stats["entry_rules_pass"]["rows"]),
            "market_gate_pass": int(stats["market_gate_pass"]["rows"]),
            "prediction_net_margin_pass": prediction_net,
            "final_candidate_pass": prediction_net_entry,
            "selected_count": selected,
            "no_trade_reason": reason,
        })
    return observations, reason_counts


def _waiting_result(
    *,
    selection: dict,
    cutoff: date,
    effective_end: date,
    required: int,
) -> dict:
    """Persist a useful audit record even before the first new session exists."""
    return {
        "status": "waiting_for_new_data",
        "model_id": selection["model_id"],
        "selection_id": selection["selection_id"],
        "cutoff": str(cutoff),
        "as_of": str(effective_end),
        "available_sessions": 0,
        "required_first_window_sessions": required,
        "feature_coverage": {},
        "signal_count": 0,
        "candidate_count": 0,
        "evaluated": False,
        "positions": [],
        "daily_observations": [],
        "no_trade_reason": "no_new_oos_data",
        "no_trade_reason_counts": {"no_new_oos_data": 1},
        "layer_diagnostics": _empty_layer_stats(),
        "activation": {
            "eligible": False,
            "status": "research_only",
            "reason": "new_oos_data_not_available",
        },
        "activation_eligible": False,
    }


def run_shadow_tracking(
    data_dir: Path,
    selection_path: Path,
    output_path: Path,
    *,
    as_of: date | None = None,
) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    meta = sf._model_meta(data_dir, selection["model_id"])
    if meta is None:
        raise ValueError("frozen model metadata not found")
    cutoff = date.fromisoformat(selection["unseen_after"])
    _, latest = sf._date_range_from_data(data_dir)
    effective_end = min(as_of or latest or cutoff, latest or cutoff)
    if effective_end > cutoff:
        available_frame = sf._load_panel(
            data_dir,
            cutoff + timedelta(days=1),
            effective_end,
            symbols=[str(value) for value in meta.get("training_universe") or []],
            include_minute=str(meta.get("training_profile")) == "minute_enhanced",
        )
        available = sorted(day for day in available_frame["date"].unique().to_list() if day > cutoff)
    else:
        available = []
    required = WINDOW_DAYS + SETTLEMENT_DAYS
    if len(available) < 1:
        result = _waiting_result(
            selection=selection, cutoff=cutoff, effective_end=effective_end, required=required,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    end = available[-1]
    panel = sf._load_panel(
        data_dir, date.fromisoformat(str(meta["train_start"])), end,
        symbols=[str(value) for value in meta.get("training_universe") or []],
        include_minute=str(meta.get("training_profile")) == "minute_enhanced",
    )
    features, coverage = portfolio._prepare_research_features(
        panel, str(meta.get("training_profile") or "daily_baseline")
    )
    frozen = selection["selected"]
    policy = portfolio.load_frozen_signal_policy(data_dir, meta) or {}
    prediction_rule = policy.get("prediction_rule") or {}
    min_net_edge = float(
        prediction_rule.get("min_net_edge", frozen.get("min_net_edge", portfolio.DEFAULT_MIN_NET_EDGE))
    )
    columns = rolling._rolling_feature_columns(
        frozen["feature_family"],
        [column for column in sf.training_feature_columns(selection["training_profile"])
         if column in features.columns],
    )
    prediction = portfolio._fit_candidate_predictions(
        features, horizon=5, columns=columns, alpha=float(selection["alpha"]),
        fit_end=cutoff, predict_start=cutoff + timedelta(days=1),
        predict_end=end + timedelta(days=1), target_mode=selection["target_mode"],
    )
    signal = portfolio._build_gated_signal_frame(
        data_dir, prediction, features, meta, horizon=5,
        forecast_floor=float(selection["forecast_floor"]),
        signal_policy=policy,
        min_net_edge=min_net_edge,
        rank_predictions=True,
    )
    if not signal.is_empty():
        signal = signal.filter(pl.col("date") > cutoff).sort(
            ["date", "pred_ret_5d"], descending=[False, True],
        )
        eligible = signal.filter(
            pl.col("_forecast_pass")
            & pl.col("_entry_rules_pass") & pl.col("_market_gate_pass")
        )
    else:
        eligible = signal
    ceiling = frozen.get("forecast_ceiling")
    if ceiling is not None:
        eligible = eligible.filter(pl.col("pred_ret_5d") <= float(ceiling))
    candidate_count = int(eligible.height)
    positions = (
        eligible.with_columns(
            pl.col("net_margin_score").rank(method="ordinal", descending=True).over("date").alias("rank")
        )
        .filter(pl.col("rank") <= int(selection["max_positions"]))
        .select([column for column in (
            "symbol", "date", "pred_ret_5d", "entry_score", "market_gate",
            "market_regime_score", "minute_session_coverage", "minute_bar_count",
        ) if column in eligible.columns])
        .to_dicts()
    ) if not eligible.is_empty() else []
    daily_observations, no_trade_reason_counts = _daily_observations(
        signal,
        available_dates=available,
        pred_col="pred_ret_5d",
        max_positions=int(selection["max_positions"]),
        forecast_ceiling=ceiling,
    )
    layer_diagnostics = portfolio._gated_layer_stats(signal, "pred_ret_5d")
    if positions:
        no_trade_reason = "candidate_selected"
    elif not signal.is_empty():
        no_trade_reason = max(
            no_trade_reason_counts,
            key=lambda reason: no_trade_reason_counts[reason],
        )
    else:
        no_trade_reason = "prediction_rows_empty"
    complete = len(available) >= required
    result = sf._json_safe({
        "status": "tracking" if not complete else "first_window_ready_for_settlement",
        "model_id": selection["model_id"], "selection_id": selection["selection_id"],
        "cutoff": str(cutoff), "as_of": str(end),
        "available_sessions": len(available), "required_first_window_sessions": required,
        "feature_coverage": coverage, "frozen_policy": frozen,
        "signal_policy": policy,
        "signal_count": len(positions), "candidate_count": candidate_count,
        "evaluated": True,
        "positions": positions, "daily_observations": daily_observations,
        "no_trade_reason": no_trade_reason,
        "no_trade_reason_counts": no_trade_reason_counts,
        "layer_diagnostics": layer_diagnostics,
        "activation": {
            "eligible": False,
            "status": "research_only",
            "reason": "shadow_tracking_requires_three_complete_positive_windows",
        },
        "activation_eligible": False,
        "note": "Shadow tracking never changes the active model and does not count as OOS confirmation until the window settles.",
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
