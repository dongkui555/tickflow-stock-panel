"""Five-day feature and exit selection before a genuinely new OOS window."""
from __future__ import annotations

import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from app.backtest.engine import BacktestEngine
from app.parquet import scan_enriched_parquet
from app.services import stockformer as sf
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_rolling as rolling
from app.tickflow.repository import DataStore, KlineRepository


WINDOW_DAYS = 63
SETTLEMENT_DAYS = 6
TRAINING_FAMILIES = (
    "full", "compact", "daily_ohlcv", "daily_plus_moneyflow", "daily_plus_auction",
)
EXIT_MODES = ("max_hold", "prediction_reverse", "market_gate_close")
FORECAST_CEILINGS = (None, 0.03, 0.05)


def _three_training_windows(dates: list[date], start: date, end: date) -> list[dict]:
    available = sorted({day for day in dates if start <= day < end})
    needed = 3 * WINDOW_DAYS + SETTLEMENT_DAYS
    if len(available) < needed:
        raise ValueError(f"training validation needs {needed} sessions, found {len(available)}")
    offset = len(available) - needed
    return [
        {
            "window": index + 1,
            "start": available[offset + index * WINDOW_DAYS],
            "end": available[offset + (index + 1) * WINDOW_DAYS - 1],
            "settlement_end": available[offset + (index + 1) * WINDOW_DAYS + SETTLEMENT_DAYS - 1],
        }
        for index in range(3)
    ]


def _candidate_score(stats: list[dict]) -> tuple:
    returns = [float(row.get("total_return", 0.0)) for row in stats]
    trades = [int(row.get("trade_count", 0)) for row in stats]
    drawdown = min(float(row.get("max_drawdown", 0.0)) for row in stats)
    compound = (math.prod(1.0 + value / 100.0 for value in returns) - 1.0) * 100.0
    positive = sum(value > 0 and count >= 3 for value, count in zip(returns, trades))
    enough = all(count >= 3 for count in trades)
    return (enough, positive, round(compound - 0.35 * abs(drawdown), 4), round(compound, 4), drawdown)


def _available_future_dates(data_dir: Path, after: date) -> list[date]:
    _, latest = sf._date_range_from_data(data_dir)
    if latest is None or latest <= after:
        return []
    frame = scan_enriched_parquet(str(data_dir / "kline_daily_enriched" / "**" / "*.parquet"))
    return sorted(frame.filter(pl.col("date") > after).select("date").unique().collect()["date"].to_list())


def _load_features(data_dir: Path, meta: dict, end: date) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    start = date.fromisoformat(str(meta["train_start"]))
    symbols = [str(symbol) for symbol in meta.get("training_universe") or []]
    if not symbols:
        raise ValueError("model has no frozen training universe")
    profile = str(meta.get("training_profile") or "daily_baseline")
    panel = sf._load_panel(
        data_dir, start, end, symbols=symbols,
        include_minute=profile == "minute_enhanced",
    )
    if panel.is_empty():
        raise ValueError("no panel for frozen training universe")
    features, coverage = portfolio._prepare_research_features(panel, profile)
    if features.is_empty():
        raise ValueError("no rows pass the frozen minute coverage requirement")
    return panel, features, coverage


def select_pre_oos_policy(
    data_dir: Path,
    *,
    model_id: str,
    output_path: Path,
    forecast_floor: float = 0.004,
    max_positions: int = 3,
    initial_capital: float = 10_000.0,
    on_progress=None,
) -> dict:
    meta = sf._model_meta(data_dir, model_id)
    if meta is None:
        raise ValueError("model metadata not found")
    oos_start = date.fromisoformat(str(meta["oos_start"]))
    validation_start = date.fromisoformat(str((meta.get("validation") or {})["start"]))
    latest = sf._date_range_from_data(data_dir)[1]
    unseen_after = max(date.fromisoformat(str(meta["oos_end"])), latest or oos_start)
    panel, features, coverage = _load_features(data_dir, meta, oos_start - timedelta(days=1))
    windows = _three_training_windows(features["date"].unique().to_list(), validation_start, oos_start)
    engine = BacktestEngine(KlineRepository(DataStore(data_dir)))
    feature_columns = [
        column for column in sf.training_feature_columns(str(meta.get("training_profile")))
        if column in features.columns
    ]
    alpha = float((((meta.get("validation") or {}).get("horizons") or {}).get("5d") or {}).get("alpha") or 1.0)
    candidates = []
    for family_index, family in enumerate(TRAINING_FAMILIES):
        columns = rolling._rolling_feature_columns(family, feature_columns)
        if not columns:
            continue
        scenario_stats = {
            (mode, ceiling): []
            for mode in EXIT_MODES
            for ceiling in FORECAST_CEILINGS
        }
        for window in windows:
            prediction = portfolio._fit_candidate_predictions(
                features, horizon=5, columns=columns, alpha=alpha,
                fit_end=window["start"], predict_start=window["start"],
                predict_end=window["settlement_end"] + timedelta(days=1),
                target_mode=str(meta.get("target_mode") or "absolute"),
            )
            signal = portfolio._build_gated_signal_frame(
                data_dir, prediction, features, meta, horizon=5,
                forecast_floor=forecast_floor,
            ) if not prediction.is_empty() else pl.DataFrame()
            panel_window = panel.filter(
                (pl.col("date") >= window["start"])
                & (pl.col("date") <= window["settlement_end"])
            )
            for exit_mode in EXIT_MODES:
                for forecast_ceiling in FORECAST_CEILINGS:
                    if signal.is_empty():
                        stats = portfolio._empty_stats()
                    else:
                        stats = portfolio._run_one(
                            panel_window, signal, engine, horizon=5,
                            forecast_floor=forecast_floor, max_positions=max_positions,
                            initial_capital=initial_capital, signal_frame=signal,
                            exit_mode=exit_mode, entry_signal_end=window["end"],
                            forecast_ceiling=forecast_ceiling,
                        )
                    scenario_stats[(exit_mode, forecast_ceiling)].append(stats)
        for (exit_mode, forecast_ceiling), stats in scenario_stats.items():
            rank = _candidate_score(stats)
            candidates.append({
                "feature_family": family,
                "feature_columns": columns,
                "minute_feature_count": sum(column in sf.MINUTE_FEATURE_COLUMNS for column in columns),
                "exit_mode": exit_mode,
                "forecast_ceiling": forecast_ceiling,
                "window_stats": stats,
                "all_windows_minimum_trades": rank[0],
                "positive_windows": rank[1],
                "risk_adjusted_score": rank[2],
                "compounded_return_pct": rank[3],
                "worst_window_drawdown_pct": rank[4],
            })
        if on_progress:
            on_progress(family_index + 1, len(TRAINING_FAMILIES), family)
    if not candidates:
        raise ValueError("no train-period feature candidate can be evaluated")
    selected = max(candidates, key=lambda row: (
        row["all_windows_minimum_trades"], row["positive_windows"],
        row["risk_adjusted_score"], row["compounded_return_pct"],
        row["worst_window_drawdown_pct"],
    ))
    training_pass = (
        selected["all_windows_minimum_trades"]
        and selected["positive_windows"] >= 2
        and selected["compounded_return_pct"] > 0
        and selected["worst_window_drawdown_pct"] >= -10.0
    )
    available_future = _available_future_dates(data_dir, unseen_after)
    report = sf._json_safe({
        "selection_id": f"stockformer_5d_pre_oos_{time.strftime('%Y%m%d_%H%M%S')}",
        "model_id": str(meta["model_id"]),
        "protocol": "three_purged_63_session_train_windows_real_fills_v1",
        "selection_uses_data_before": str(oos_start),
        "validation_start": str(validation_start),
        "unseen_after": str(unseen_after),
        "training_profile": meta.get("training_profile"),
        "target_mode": meta.get("target_mode"),
        "feature_coverage": coverage,
        "market_policy": "strict_all",
        "forecast_floor": forecast_floor,
        "alpha": alpha,
        "max_positions": max_positions,
        "initial_capital": initial_capital,
        "windows": windows,
        "candidates": candidates,
        "selected": selected,
        "training_pass": training_pass,
        "new_oos": {
            "status": "pending_complete_window",
            "available_sessions_after_cutoff": len(available_future),
            "required_signal_sessions": WINDOW_DAYS,
            "required_settlement_sessions": SETTLEMENT_DAYS,
            "required_total_sessions": WINDOW_DAYS + SETTLEMENT_DAYS,
        },
        "activation_eligible": False,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_new_window(data_dir: Path, selection_path: Path, output_path: Path) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cutoff = date.fromisoformat(selection["unseen_after"])
    dates = _available_future_dates(data_dir, cutoff)
    needed = WINDOW_DAYS + SETTLEMENT_DAYS
    if len(dates) < needed:
        return {
            "status": "pending_complete_window", "available_sessions": len(dates),
            "required_sessions": needed, "unseen_after": str(cutoff),
            "selection_path": str(selection_path), "activation_eligible": False,
        }
    meta = sf._model_meta(data_dir, selection["model_id"])
    if meta is None:
        raise ValueError("frozen model metadata not found")
    panel, features, coverage = _load_features(data_dir, meta, dates[-1])
    ready_dates = sorted(features.filter(pl.col("date") > cutoff)["date"].unique().to_list())
    if len(ready_dates) < needed:
        return {
            "status": "pending_minute_coverage", "ready_sessions": len(ready_dates),
            "required_sessions": needed, "unseen_after": str(cutoff),
            "selection_path": str(selection_path), "activation_eligible": False,
        }
    window_start, window_end, settlement_end = (
        ready_dates[0], ready_dates[WINDOW_DAYS - 1], ready_dates[needed - 1]
    )
    frozen = selection["selected"]
    columns = rolling._rolling_feature_columns(
        frozen["feature_family"],
        [column for column in sf.training_feature_columns(selection["training_profile"])
         if column in features.columns],
    )
    if columns != frozen["feature_columns"]:
        raise ValueError("frozen feature columns changed since selection")
    prediction = portfolio._fit_candidate_predictions(
        features, horizon=5, columns=columns, alpha=float(selection["alpha"]),
        fit_end=window_start, predict_start=window_start,
        predict_end=settlement_end + timedelta(days=1),
        target_mode=selection["target_mode"],
    )
    signal = portfolio._build_gated_signal_frame(
        data_dir, prediction, features, meta, horizon=5,
        forecast_floor=float(selection["forecast_floor"]),
    )
    panel_window = panel.filter(
        (pl.col("date") >= window_start) & (pl.col("date") <= settlement_end)
    )
    stats = portfolio._run_one(
        panel_window, signal, BacktestEngine(KlineRepository(DataStore(data_dir))),
        horizon=5, forecast_floor=float(selection["forecast_floor"]),
        max_positions=int(selection["max_positions"]),
        initial_capital=float(selection["initial_capital"]),
        signal_frame=signal, exit_mode=frozen["exit_mode"],
        entry_signal_end=window_end,
        forecast_ceiling=frozen.get("forecast_ceiling"),
    )
    result = sf._json_safe({
        "status": "validated_new_window",
        "selection_path": str(selection_path),
        "selection_id": selection["selection_id"],
        "model_id": selection["model_id"],
        "window": {"start": window_start, "end": window_end, "settlement_end": settlement_end},
        "feature_coverage": coverage,
        "selected": frozen,
        "stats": stats,
        "positive_after_cost": stats["trade_count"] >= 3 and stats["total_return"] > 0,
        "activation_eligible": False,
        "note": "One new window is only a forward check; activation still requires three consecutive positive windows.",
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
