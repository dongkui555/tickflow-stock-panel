"""Produce read-only trade attribution for a StockFormer rolling OOS report.

The diagnostic replay intentionally uses the same gated signal builder and
BacktestEngine as the production research path.  It never promotes a model or
changes the rolling report pointer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.backtest.engine import BacktestEngine, MatcherConfig  # noqa: E402
from app.services import stockformer as sf  # noqa: E402
from app.services import stockformer_portfolio as portfolio  # noqa: E402
from app.services import stockformer_rolling as rolling  # noqa: E402
from app.tickflow.repository import DataStore, KlineRepository  # noqa: E402


MARKET_GATE_SCENARIOS = (
    "strict_all",
    "relax_strict",
    "adaptive_recovery",
    "open_all",
)

EXIT_SCENARIOS = {
    "fixed_5d": {},
    "stop_loss_4pct": {"stop_loss_pct": 0.04},
    "take_profit_8pct": {"take_profit_pct": 0.08},
    "stop_loss_4_take_profit_8": {
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.08,
    },
    "prediction_reverse": {"exit_mode": "prediction_reverse"},
    "market_gate_close": {"exit_mode": "market_gate_close"},
}


def _stats(result) -> dict:
    stats = result.stats or {}
    return {
        "total_return": round(float(stats.get("total_return", 0.0)) * 100, 4),
        "max_drawdown": round(float(stats.get("max_drawdown", 0.0)) * 100, 4),
        "trade_count": int(stats.get("n_trades", len(result.trades))),
        "win_rate": round(float(stats.get("win_rate", 0.0)) * 100, 4),
        "profit_factor": stats.get("profit_factor"),
        "avg_pnl": round(float(stats.get("avg_pnl", 0.0)) * 100, 4),
        "execution": stats.get("execution", {}),
    }


def _run_with_trades(
    panel: pl.DataFrame,
    signal_frame: pl.DataFrame,
    engine: BacktestEngine,
    *,
    horizon: int,
    forecast_floor: float,
    max_positions: int,
    initial_capital: float,
    exit_mode: str = "max_hold",
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    trailing_stop_pct: float | None = None,
) -> tuple[object, pl.DataFrame]:
    pred_col = f"pred_ret_{horizon}d"
    candidates = signal_frame.filter(
        (pl.col(pred_col).fill_null(-1.0) > forecast_floor)
        & pl.col("_entry_rules_pass")
        & pl.col("_market_gate_pass")
    )
    if candidates.is_empty() or panel.is_empty():
        return engine._empty_result(), pl.DataFrame()
    scored_panel, entries, ranked = portfolio._build_ranked_entry_orders(
        panel, candidates, pred_col=pred_col, max_positions=max_positions,
    )
    selected = ranked.select(
        [
            column
            for column in (
                "symbol",
                "date",
                pred_col,
                "target_5d",
                "entry_score",
                "market_gate",
                "market_regime_score",
                "market_above_ma20_pct",
                "market_strong_down_pct",
                "industry",
                "rank",
            )
            if column in ranked.columns
        ]
    )
    signal_rows = {
        (str(row["symbol"]), row["date"]): row
        for row in signal_frame.select(["symbol", "date", pred_col]).to_dicts()
    }
    if exit_mode == "prediction_reverse":
        exit_keys = {
            key for key, row in signal_rows.items()
            if row.get(pred_col) is not None and float(row[pred_col]) <= 0.0
        }
    elif exit_mode == "prediction_below_floor":
        exit_keys = {
            key for key, row in signal_rows.items()
            if row.get(pred_col) is not None and float(row[pred_col]) <= forecast_floor
        }
    elif exit_mode == "market_gate_close":
        exit_keys = {
            (str(row["symbol"]), row["date"])
            for row in signal_frame.filter(~pl.col("_market_gate_pass")).select(["symbol", "date"]).to_dicts()
        }
    elif exit_mode == "max_hold":
        exit_keys = set()
    else:
        raise ValueError(f"unknown exit_mode: {exit_mode}")
    exits = pl.Series(
        "_exit",
        [
            (str(row["symbol"]), row["date"]) in exit_keys
            for row in scored_panel.select(["symbol", "date"]).to_dicts()
        ],
        dtype=pl.Boolean,
    )
    config = MatcherConfig(
        matching="open_t+1",
        entry_fill="open_t+1",
        exit_fill="open_t+1",
        commission_pct=0.0003,
        stamp_tax_pct=0.0005,
        slippage_bps=5.0,
        max_hold_days=horizon,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
        max_positions=max_positions,
        initial_capital=initial_capital,
    )
    return engine.simulate_portfolio(scored_panel, entries, exits, config), selected


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _trade_rows(result, selected: pl.DataFrame, window: int, policy: str) -> list[dict]:
    if selected.is_empty():
        return []
    lookup = {
        (str(row["symbol"]), _as_date(row["date"])): row
        for row in selected.to_dicts()
    }
    rows = []
    for trade in result.trades:
        signal_date = _as_date(trade.entry_signal_date) or trade.entry_date
        signal = lookup.get((str(trade.symbol), signal_date), {})
        gross = (
            float(trade.exit_price) / float(trade.entry_price) - 1.0
            if trade.entry_price
            else 0.0
        )
        rows.append(
            {
                "window": window,
                "policy": policy,
                "symbol": str(trade.symbol),
                "industry": signal.get("industry"),
                "signal_date": str(signal_date),
                "entry_date": str(trade.entry_date),
                "exit_date": str(trade.exit_date),
                "duration": int(trade.duration),
                "pred_ret_5d": signal.get("pred_ret_5d"),
                "target_5d": signal.get("target_5d"),
                "pred_rank": signal.get("rank"),
                "entry_score": signal.get("entry_score"),
                "market_gate": signal.get("market_gate"),
                "market_regime_score": signal.get("market_regime_score"),
                "market_above_ma20_pct": signal.get("market_above_ma20_pct"),
                "market_strong_down_pct": signal.get("market_strong_down_pct"),
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price),
                "gross_pnl_pct": round(gross * 100, 6),
                "net_pnl_pct": round(float(trade.pnl_pct) * 100, 6),
                "cost_drag_pct": round((gross - float(trade.pnl_pct)) * 100, 6),
                "pnl_amount": float(trade.pnl_amount),
                "exit_reason": trade.exit_reason,
                "blocked_exit_days": int(trade.blocked_exit_days),
                "entry_signal_id": trade.entry_signal_id,
            }
        )
    return rows


def _group_stats(rows: list[dict], key: str) -> list[dict]:
    if not rows:
        return []
    frame = pl.DataFrame(rows)
    if key not in frame.columns:
        return []
    groups = []
    for value, part in frame.group_by(key, maintain_order=True):
        label = value[0] if isinstance(value, tuple) else value
        pnl = part["net_pnl_pct"].to_numpy()
        groups.append(
            {
                key: label,
                "trade_count": int(part.height),
                "win_rate": round(float(np.mean(pnl > 0) * 100), 4),
                "mean_net_pnl_pct": round(float(np.mean(pnl)), 6),
                "median_net_pnl_pct": round(float(np.median(pnl)), 6),
                "sum_net_pnl_pct": round(float(np.sum(pnl)), 6),
            }
        )
    return sorted(groups, key=lambda item: item["sum_net_pnl_pct"], reverse=True)


def _layer_stats(signal: pl.DataFrame, pred_col: str, forecast_floor: float) -> dict:
    """Count each gate and the rows rejected at each successive layer."""
    if signal.is_empty() or pred_col not in signal.columns:
        return {
            "total_rows": 0,
            "prediction_pass": {"rows": 0, "dates": 0},
            "entry_rules_pass": {"rows": 0, "dates": 0},
            "market_gate_pass": {"rows": 0, "dates": 0},
            "flow": {
                "prediction_entry_market": 0,
                "prediction_rejected": 0,
                "entry_rejected_after_prediction": 0,
                "market_rejected_after_prediction_entry": 0,
            },
        }

    prediction = pl.col(pred_col).fill_null(-1.0) > forecast_floor
    entry = pl.col("_entry_rules_pass")
    market = pl.col("_market_gate_pass")

    def count(expr: pl.Expr) -> dict:
        selected = signal.filter(expr)
        return {
            "rows": int(selected.height),
            "dates": int(selected["date"].n_unique()) if not selected.is_empty() else 0,
        }

    prediction_entry = prediction & entry
    all_pass = prediction_entry & market
    return {
        "total_rows": int(signal.height),
        "prediction_pass": count(prediction),
        "entry_rules_pass": count(entry),
        "market_gate_pass": count(market),
        "flow": {
            "prediction_entry_market": int(signal.filter(all_pass).height),
            "prediction_rejected": int(signal.filter(~prediction).height),
            "entry_rejected_after_prediction": int(signal.filter(prediction & ~entry).height),
            "market_rejected_after_prediction_entry": int(signal.filter(prediction_entry & ~market).height),
        },
    }


def _scenario_summary(
    result,
    selected: pl.DataFrame,
    *,
    label: str,
    signal: pl.DataFrame,
    forecast_floor: float,
) -> dict:
    rows = _trade_rows(result, selected, 0, label)
    return {
        "scenario": label,
        "stats": _stats(result),
        "trade_count": len(result.trades),
        "gating": _layer_stats(signal, "pred_ret_5d", forecast_floor),
        "by_regime": _group_stats(rows, "market_gate"),
        "by_exit_reason": _group_stats(rows, "exit_reason"),
    }


def _rank_ic(signal: pl.DataFrame) -> dict:
    if signal.is_empty() or "target_5d" not in signal.columns:
        return {"dates": 0, "mean_spearman": None, "positive_date_fraction": None}
    values = []
    for part in signal.partition_by("date", as_dict=False):
        part = part.select(["pred_ret_5d", "target_5d"]).drop_nulls()
        if part.height < 5:
            continue
        x = part["pred_ret_5d"].rank().to_numpy()
        y = part["target_5d"].rank().to_numpy()
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        values.append(float(np.corrcoef(x, y)[0, 1]))
    return {
        "dates": len(values),
        "mean_spearman": round(float(np.mean(values)), 6) if values else None,
        "positive_date_fraction": round(float(np.mean(np.asarray(values) > 0)), 6)
        if values
        else None,
    }


def _diagnose(data_dir: Path, model_id: str | None, report_path: Path, output_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    meta = sf._model_meta(data_dir, model_id or report.get("model_id"))
    if meta is None:
        raise RuntimeError("model metadata not found")
    factor_path = sf._resolve_artifact_path(data_dir, meta.get("factor_file"))
    if factor_path is None or not factor_path.exists():
        raise RuntimeError("model factor file not found")
    factor = pl.read_parquet(factor_path)
    dates = sorted(factor["date"].unique().to_list())
    windows = rolling.build_rolling_windows(dates)
    symbols = [str(value) for value in factor["symbol"].unique().to_list() if value]
    start, end = dates[0], dates[-1]
    data_start, _ = sf._date_range_from_data(data_dir)
    history_start = max(data_start or (start - timedelta(days=730)), start - timedelta(days=730))
    panel = sf._load_panel(
        data_dir,
        history_start,
        end + timedelta(days=rolling.DEFAULT_WINDOW_DAYS),
        symbols=symbols,
        include_minute=str(meta.get("training_profile") or "daily_baseline") == "minute_enhanced",
    )
    training_profile = str(meta.get("training_profile") or "daily_baseline")
    feature_frame, feature_coverage = portfolio._prepare_research_features(panel, training_profile)
    feature_columns = [
        column for column in sf.training_feature_columns(training_profile)
        if column in feature_frame.columns
    ]
    factor = factor.join(feature_frame.select("symbol", "date"), on=["symbol", "date"], how="inner")
    if factor.is_empty():
        raise RuntimeError("no OOS rows pass the research feature coverage requirement")
    selection = ((meta.get("validation") or {}).get("horizons") or {}).get("5d") or {}
    family = str((report.get("prediction_protocol") or {}).get("feature_family") or selection.get("family") or "trend")
    alpha = float((report.get("prediction_protocol") or {}).get("alpha") or selection.get("alpha") or 100.0)
    protocol = (report.get("rolling") or {}).get("prediction_protocol") or {}
    family = str(protocol.get("feature_family") or family)
    alpha = float(protocol.get("alpha") if protocol.get("alpha") is not None else alpha)
    columns = rolling._rolling_feature_columns(family, feature_columns)
    if not columns or (protocol.get("feature_columns") and columns != protocol["feature_columns"]):
        raise RuntimeError(f"report feature family does not match current research features: {family}")
    floor = float(protocol.get("forecast_floor") if protocol.get("forecast_floor") is not None else 0.004)
    policies = {str(item.get("window")): str(item.get("regime_policy") or "strict_all") for item in (report.get("rolling", {}).get("windows") or [])}
    engine = BacktestEngine(KlineRepository(DataStore(data_dir)))
    trades: list[dict] = []
    window_rows = []
    all_signal = []
    market_scenario_frames = {name: [] for name in MARKET_GATE_SCENARIOS}
    market_window_gating = {name: [] for name in MARKET_GATE_SCENARIOS}
    market_window_stats = {name: [] for name in MARKET_GATE_SCENARIOS}
    selected_window_frames = []
    for window in windows:
        window_factor = factor.filter((pl.col("date") >= window["start"]) & (pl.col("date") <= window["end"]))
        prediction = portfolio._fit_candidate_predictions(
            feature_frame,
            horizon=5,
            columns=columns,
            alpha=alpha,
            fit_end=window["start"],
            predict_start=window["start"],
            predict_end=window["end"] + timedelta(days=1),
            target_mode=str(meta.get("target_mode") or "absolute"),
        )
        signal = (
            portfolio._build_gated_signal_frame(
                data_dir,
                window_factor.select([c for c in window_factor.columns if c != "pred_ret_5d"]).join(
                    prediction, on=["symbol", "date"], how="inner"
                ),
                feature_frame,
                meta,
                horizon=5,
                forecast_floor=floor,
            )
            if not prediction.is_empty()
            else pl.DataFrame()
        )
        policy = policies.get(str(window["window"]), "strict_all")
        policy_frames = {
            name: signal.with_columns(
                (
                    pl.lit(True)
                    if name == "open_all"
                    else rolling._regime_policy_expr(signal, name)
                ).alias("_market_gate_pass")
            )
            for name in MARKET_GATE_SCENARIOS
        }
        selected_signal = policy_frames[policy] if policy in policy_frames else policy_frames["strict_all"]
        for name, frame in policy_frames.items():
            market_scenario_frames[name].append(frame)
            market_window_gating[name].append(
                {
                    "window": int(window["window"]),
                    "start": str(window["start"]),
                    "end": str(window["end"]),
                    "gating": _layer_stats(frame, "pred_ret_5d", floor),
                }
            )
        panel_window = panel.filter(
            (pl.col("date") >= window["start"]) & (pl.col("date") <= window["end"] + timedelta(days=rolling.DEFAULT_WINDOW_DAYS))
        )
        result, selected = _run_with_trades(
            panel_window,
            selected_signal,
            engine,
            horizon=5,
            forecast_floor=floor,
            max_positions=3,
            initial_capital=10_000.0,
        )
        rows = _trade_rows(result, selected, int(window["window"]), policy)
        trades.extend(rows)
        all_signal.append(selected_signal)
        selected_window_frames.append(selected_signal)
        for name, frame in policy_frames.items():
            window_result, _ = _run_with_trades(
                panel_window,
                frame,
                engine,
                horizon=5,
                forecast_floor=floor,
                max_positions=3,
                initial_capital=10_000.0,
            )
            market_window_stats[name].append(
                {
                    "window": int(window["window"]),
                    "start": str(window["start"]),
                    "end": str(window["end"]),
                    "stats": _stats(window_result),
                }
            )
        window_rows.append({
            "window": int(window["window"]),
            "start": str(window["start"]),
            "end": str(window["end"]),
            "policy": policy,
            "stats": _stats(result),
            "trade_count": len(rows),
            "gating": _layer_stats(selected_signal, "pred_ret_5d", floor),
        })
    signal_all = pl.concat(all_signal, how="vertical_relaxed") if all_signal else pl.DataFrame()
    tail = dates[len(windows) * rolling.DEFAULT_WINDOW_DAYS :]
    trade_frame = pl.DataFrame(trades) if trades else pl.DataFrame()
    # Re-run the selected policy schedule as one continuous account.  The
    # window reports reset cash and positions at each boundary, while this
    # replay preserves them, so its count is the correct comparison with the
    # formal full_oos portfolio result.
    continuous_panel = panel.filter(
        (pl.col("date") >= start)
        & (pl.col("date") <= end + timedelta(days=rolling.DEFAULT_WINDOW_DAYS))
    )
    continuous_result, continuous_selected = _run_with_trades(
        continuous_panel,
        signal_all,
        engine,
        horizon=5,
        forecast_floor=floor,
        max_positions=3,
        initial_capital=10_000.0,
    )
    continuous_stats = _stats(continuous_result)
    expected = ((report.get("final_backtest") or {}).get("full_oos_portfolio_stats") or {})
    replay_matches_report = bool(expected) and not (
        abs(continuous_stats["total_return"] - float(expected.get("total_return", 0))) > 0.02
        or continuous_stats["trade_count"] != int(expected.get("trade_count", 0))
    )
    continuous_trade_rows = _trade_rows(
        continuous_result,
        continuous_selected,
        0,
        "selected_policy_schedule",
    )

    # Fixed-policy market-gate ablations use the same refit predictions,
    # entry rules and real fills.  Only the market predicate changes.
    market_ablation = []
    for name in MARKET_GATE_SCENARIOS:
        frames = market_scenario_frames.get(name) or []
        scenario_signal = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
        scenario_result, scenario_selected = _run_with_trades(
            continuous_panel,
            scenario_signal,
            engine,
            horizon=5,
            forecast_floor=floor,
            max_positions=3,
            initial_capital=10_000.0,
        )
        summary = _scenario_summary(
            scenario_result,
            scenario_selected,
            label=name,
            signal=scenario_signal,
            forecast_floor=floor,
        )
        summary["window_gating"] = market_window_gating.get(name, [])
        summary["window_stats"] = market_window_stats.get(name, [])
        market_ablation.append(summary)

    # Exit ablations keep the selected per-window market policy schedule fixed
    # and vary only the exit rule in the same T+1/cost/lot matcher.
    exit_ablation = []
    for name, options in EXIT_SCENARIOS.items():
        if name == "fixed_5d":
            scenario_result, scenario_selected = continuous_result, continuous_selected
        else:
            scenario_result, scenario_selected = _run_with_trades(
                continuous_panel,
                signal_all,
                engine,
                horizon=5,
                forecast_floor=floor,
                max_positions=3,
                initial_capital=10_000.0,
                **options,
            )
        exit_summary = _scenario_summary(
            scenario_result,
            scenario_selected,
            label=name,
            signal=signal_all,
            forecast_floor=floor,
        )
        exit_summary["window_stats"] = []
        for window, frame in zip(windows, selected_window_frames):
            window_panel = panel.filter(
                (pl.col("date") >= window["start"])
                & (pl.col("date") <= window["end"] + timedelta(days=rolling.DEFAULT_WINDOW_DAYS))
            )
            window_result, _ = _run_with_trades(
                window_panel,
                frame,
                engine,
                horizon=5,
                forecast_floor=floor,
                max_positions=3,
                initial_capital=10_000.0,
                **options,
            )
            exit_summary["window_stats"].append(
                {
                    "window": int(window["window"]),
                    "start": str(window["start"]),
                    "end": str(window["end"]),
                    "stats": _stats(window_result),
                }
            )
        exit_ablation.append(exit_summary)
    result = {
        "report_id": f"stockformer_trade_diagnostics_{time.strftime('%Y%m%d_%H%M%S')}",
        "model_id": meta.get("model_id"),
        "source_report": str(report_path),
        "sample_out": {"start": str(start), "end": str(end), "trading_days": len(dates)},
        "prediction_protocol": {
            "training_profile": training_profile,
            "feature_family": family,
            "feature_columns": columns,
            "minute_feature_count": sum(column in sf.MINUTE_FEATURE_COLUMNS for column in columns),
            "feature_coverage": feature_coverage,
            "alpha": alpha,
            "forecast_floor": floor,
        },
        "coverage": {
            "windows_evaluated": len(windows),
            "window_days": rolling.DEFAULT_WINDOW_DAYS,
            "excluded_tail_dates": [str(day) for day in tail],
            "excluded_tail_trading_days": len(tail),
        },
        "windows": window_rows,
        "trade_count": len(trades),
        "trades": trades,
        "continuous_portfolio": {
            "stats": continuous_stats,
            "formal_report_stats": expected,
            "matches_formal_report": replay_matches_report,
            "trade_count": len(continuous_result.trades),
            "trades": continuous_trade_rows,
            "by_regime": _group_stats(continuous_trade_rows, "market_gate"),
            "by_exit_reason": _group_stats(continuous_trade_rows, "exit_reason"),
            "window_trade_count_sum": len(trades),
            "count_difference_vs_window_sum": len(continuous_result.trades) - len(trades),
        },
        "market_gate_ablation": {
            "same_predictions_entry_rules_and_execution": True,
            "scenarios": market_ablation,
        },
        "exit_ablation": {
            "market_policy": "selected_policy_schedule",
            "same_predictions_entry_rules_and_execution": True,
            "scenarios": exit_ablation,
        },
        "by_regime": _group_stats(trades, "market_gate"),
        "by_industry": _group_stats(trades, "industry"),
        "by_exit_reason": _group_stats(trades, "exit_reason"),
        "by_prediction_bucket": _group_stats(
            [
                {**row, "prediction_bucket": int(min(4, max(0, np.floor(float(row.get("pred_ret_5d") or 0.0) * 1000 / 2))))}
                for row in trades
            ],
            "prediction_bucket",
        ),
        "rank_ic": _rank_ic(signal_all),
        "costs": {
            "gross_mean_pnl_pct": round(float(trade_frame["gross_pnl_pct"].mean()), 6) if trades else 0.0,
            "net_mean_pnl_pct": round(float(trade_frame["net_pnl_pct"].mean()), 6) if trades else 0.0,
            "mean_cost_drag_pct": round(float(trade_frame["cost_drag_pct"].mean()), 6) if trades else 0.0,
            "total_cost_drag_pct": round(float(trade_frame["cost_drag_pct"].sum()), 6) if trades else 0.0,
        },
        "conclusion": {
            "promotion_status": "research_only",
            "activation_changed": False,
            "notes": [
                "诊断只重放逐笔成交，不改变模型、闸门或激活状态。",
                "尾部交易日未进入完整 63 日窗口，不能用于连续窗口稳定性判定。",
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sf._json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    data_dir = args.data_dir.resolve()
    report = (args.report or (data_dir / "stockformer" / "experiments" / "rolling_oos_after_auction_dedup_20260930.json")).resolve()
    output = (args.output or (data_dir / "stockformer" / "experiments" / f"{time.strftime('%Y%m%d_%H%M%S')}_trade_diagnostics.json")).resolve()
    result = _diagnose(data_dir, args.model_id, report, output)
    print(json.dumps({"report_path": str(output), "model_id": result["model_id"], "trade_count": result["trade_count"], "matches_formal_report": result["continuous_portfolio"]["matches_formal_report"], "rank_ic": result["rank_ic"], "coverage": result["coverage"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
