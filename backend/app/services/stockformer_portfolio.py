"""StockFormer horizon tuning backed by the production portfolio matcher."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from app.backtest.engine import BacktestEngine, MatcherConfig
from app.services import stockformer as sf
from app.tickflow.repository import DataStore, KlineRepository

_FORECAST_FLOORS = (0.0, 0.0025, 0.0040, 0.0060)
ROUND_TRIP_BASE_COST = 0.0021
DEFAULT_MIN_NET_EDGE = 0.0005
DEFAULT_TURNOVER_PENALTY = 0.0005


def load_frozen_signal_policy(data_dir: Path, meta: dict | None) -> dict | None:
    """Load the model-specific signal policy, if one was frozen."""
    if not meta:
        return None
    inline = meta.get("frozen_signal_policy")
    if isinstance(inline, dict) and inline.get("version"):
        return inline
    path = sf._resolve_artifact_path(data_dir, meta.get("frozen_signal_policy_file"))
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("version") else None


def _forecast_pass_frame(
    frame: pl.DataFrame,
    *,
    pred_col: str,
    forecast_floor: float,
    signal_policy: dict | None,
    rank_predictions: bool = False,
) -> pl.DataFrame:
    """Build the frozen prediction gate without using future labels.

    A calibrated positive excess-return check supplies the direction.  When a
    frozen policy requests it, the raw model score supplies the within-session
    rank so a scale shift in the calibrated tail cannot change coverage.
    """
    rule = (signal_policy or {}).get("prediction_rule") or {}
    kind = str(rule.get("kind") or "absolute_floor")
    floor = float(rule.get("calibrated_floor", forecast_floor))
    prediction = pl.col(pred_col).cast(pl.Float64, strict=False).fill_null(-1.0) > floor
    if not rank_predictions or kind == "calibrated_positive":
        return frame.with_columns(prediction.alias("_forecast_pass"))
    if kind not in {
        "calibrated_positive_and_raw_cross_sectional_rank",
        "calibrated_net_margin_cross_sectional_rank",
    }:
        return frame.with_columns(prediction.alias("_forecast_pass"))
    ranking_col = str(
        rule.get("ranking_column")
        or (
            "net_margin_score"
            if kind == "calibrated_net_margin_cross_sectional_rank"
            else f"pred_ret_raw_{pred_col.removeprefix('pred_ret_')}"
        )
    )
    if ranking_col not in frame.columns or "date" not in frame.columns:
        return frame.with_columns(prediction.alias("_forecast_pass"))
    top_quantile = min(1.0, max(0.01, float(rule.get("top_quantile", 0.20))))
    ranked = frame.with_columns(
        pl.col(ranking_col).cast(pl.Float64, strict=False).rank(
            method="average", descending=True,
        ).over("date").alias("_frozen_prediction_rank"),
        pl.col(ranking_col).cast(pl.Float64, strict=False).is_not_null()
        .cast(pl.Int64).sum().over("date").alias("_frozen_prediction_count"),
    )
    rank_pass = pl.col("_frozen_prediction_rank") <= pl.max_horizontal(
        pl.col("_frozen_prediction_count") * top_quantile,
        pl.lit(1.0),
    )
    return ranked.with_columns((prediction & rank_pass).alias("_forecast_pass")).drop(
        ["_frozen_prediction_rank", "_frozen_prediction_count"], strict=False,
    )


def _empty_stats() -> dict:
    return {
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "trade_count": 0,
        "win_rate": 0.0,
        "trading_days": 0,
    }


def _prepare_research_features(panel: pl.DataFrame, training_profile: str) -> tuple[pl.DataFrame, dict]:
    from app.services import stockformer_training as training

    features = training.execution_targets(sf._feature_frame(panel, include_targets=False), panel)
    features, coverage = sf.prepare_training_features(features, training_profile)
    if training_profile == "minute_enhanced":
        missing = [column for column in sf.MINUTE_FEATURE_COLUMNS if column not in features.columns]
        if missing:
            raise ValueError("分钟增强回测缺少分钟特征: " + ", ".join(missing))
    columns = [column for column in sf.training_feature_columns(training_profile) if column in features.columns]
    defaults = [column for column in columns if column not in sf.MINUTE_FEATURE_COLUMNS]
    return sf._fill_prediction_feature_defaults(features, defaults), coverage


def _attach_net_margin(
    frame: pl.DataFrame,
    *,
    pred_col: str,
    min_net_edge: float,
    target_mode: str = "absolute",
) -> pl.DataFrame:
    """Attach a cost and liquidity adjusted score for candidate ranking.

    The fixed round-trip cost covers the configured matcher costs.  A small
    cross-sectional liquidity surcharge keeps thin names from winning ties
    while using only information available on the signal date.
    """
    if frame.is_empty() or pred_col not in frame.columns:
        return frame.with_columns(
            pl.lit(float(min_net_edge)).alias("estimated_round_trip_cost"),
            pl.lit(-1.0).alias("net_margin_score"),
            pl.lit(False).alias("_net_margin_pass"),
        )
    normalized_target = str(target_mode or "absolute").strip().lower()
    # Net targets already subtract the fixed round-trip cost during label
    # construction.  Subtracting it again here would penalize the same signal
    # twice; only the date-level liquidity surcharge remains.
    base_cost = 0.0 if normalized_target in {"net", "cost_after", "industry_relative_net"} else ROUND_TRIP_BASE_COST
    cost = pl.lit(base_cost)
    if "amount" in frame.columns and "date" in frame.columns:
        amount = pl.col("amount").cast(pl.Float64, strict=False).fill_null(0.0).clip(lower_bound=0.0)
        amount_rank = amount.rank(method="ordinal", descending=True).over("date")
        amount_count = amount.count().over("date")
        illiquid_fraction = ((amount_rank - 1.0) / (amount_count - 1.0).clip(lower_bound=1.0)).clip(0.0, 1.0)
        cost = cost + illiquid_fraction * 0.0005
    margin = pl.col(pred_col).cast(pl.Float64, strict=False).fill_null(-1.0) - cost
    return frame.with_columns(
        cost.alias("estimated_round_trip_cost"),
        margin.alias("net_margin_score"),
        (margin >= float(min_net_edge)).alias("_net_margin_pass"),
    )


def _build_gated_signal_frame(
    data_dir: Path,
    factor: pl.DataFrame,
    feature_frame: pl.DataFrame,
    meta: dict,
    *,
    horizon: int,
    forecast_floor: float | None = None,
    signal_policy: dict | None = None,
    min_net_edge: float = DEFAULT_MIN_NET_EDGE,
    rank_predictions: bool = True,
    score_source: str = "calibrated",
    historical_identity: bool = False,
) -> pl.DataFrame:
    """Attach the production entry and market gates to research predictions.

    Portfolio research used to rank every positive forecast directly.  That
    made its return incomparable with the strict OOS report because it skipped
    the production entry rules and market gate.  Keep the feature join here so
    candidate models can reuse the same historical gate inputs without reading
    the parquet panel once per trial.
    """
    if score_source not in {"calibrated", "raw"}:
        raise ValueError("Unknown score_source")
    if factor.is_empty() or feature_frame.is_empty():
        return pl.DataFrame()
    pred_col = f"pred_ret_{horizon}d"
    if pred_col not in factor.columns:
        return pl.DataFrame()
    score_input = factor
    prediction_columns = [f"pred_ret_{h}d" for h in sf.HORIZONS if f"pred_ret_{h}d" in factor.columns]
    if score_source == "raw":
        missing = [column.replace("pred_ret_", "pred_ret_raw_") for column in prediction_columns
                   if column.replace("pred_ret_", "pred_ret_raw_") not in factor.columns]
        if missing:
            raise ValueError("Missing raw prediction scores: " + ", ".join(missing))
        # Reuse the date-scoped score calculation, then restore calibrated
        # returns for the direction and minimum-net-edge checks below.
        score_input = factor.with_columns([
            pl.col(column.replace("pred_ret_", "pred_ret_raw_")).alias(column)
            for column in prediction_columns
        ])
    df, _ = sf._add_adaptive_stockformer_score(score_input, meta)
    if score_source == "raw":
        df = df.with_columns([factor[column] for column in prediction_columns])
    # Keep the industry label with each signal so research diagnostics can
    # attribute executed trades.  It is metadata only and is not fed into the
    # prediction features or gate expressions.
    feature_cols = [
        column for column in ["amount", "industry", *sf.FEATURE_COLUMNS]
        if column in feature_frame.columns and column not in df.columns
    ]
    if feature_cols:
        df = df.join(
            feature_frame.select(["symbol", "date", *feature_cols]),
            on=["symbol", "date"],
            how="left",
        )
    if historical_identity:
        required_identity = ['symbol', 'date', 'name', '_pit_name_known', '_pit_input_ready']
        if any(column not in feature_frame.columns for column in required_identity):
            raise ValueError('Missing validated historical identity inputs')
        identity = feature_frame.select(required_identity)
        if identity.select(pl.struct('symbol', 'date').is_duplicated().any()).item():
            raise ValueError('Duplicate historical identity inputs')
        df = df.drop('name', strict=False).join(identity, on=['symbol', 'date'], how='left')
        df = sf._filter_special_treatment(df.filter(
            pl.col('_pit_name_known').fill_null(False) & pl.col('_pit_input_ready').fill_null(False)))
    else:
        df = sf._filter_special_treatment(sf._attach_instrument_names(data_dir, df))
    if df.is_empty():
        return df
    df = sf._add_multifactor_scores(df)
    df = sf._attach_market_gate_for_oos(data_dir, df)
    df = df.with_columns(
        (
            pl.col("entry_score")
            - pl.col("market_buy_threshold_add")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            * 0.55
        ).clip(0, 100).alias("entry_score"),
    )
    df = df.with_columns(
        (
            pl.col("entry_score") * 0.38
            + pl.col("stockformer_score") * 0.22
            + pl.col("fund_score") * 0.20
            + pl.col("risk_score") * 0.12
            + pl.col("technical_score") * 0.08
        ).clip(0, 100).alias("buy_confirmation_score"),
    )
    try:
        entry_expr = sf._entry_eligible_expr(df)
    except TypeError:
        entry_expr = sf._entry_eligible_expr(df)
    # Rank quality within each session so the softened entry predicate still
    # produces a stable, auditable candidate set across market regimes.
    df = df.with_columns(
        pl.col("entry_score").rank(method="average").over("date").alias("_entry_rank"),
        pl.col("entry_score").count().over("date").alias("_entry_count"),
    )
    entry_expr = entry_expr & (
        (pl.col("_entry_count") < 8)
        | (pl.col("_entry_rank") >= pl.col("_entry_count") * 0.35)
    )
    market_expr = sf._market_gate_oos_eligible_expr(df)
    if forecast_floor is None:
        forecast_floor = 0.0
    df = _attach_net_margin(
        df,
        pred_col=pred_col,
        min_net_edge=min_net_edge,
        target_mode=str(meta.get("target_mode") or "absolute"),
    )
    df = _forecast_pass_frame(df.with_columns(
        entry_expr.alias("_entry_rules_pass"),
        market_expr.alias("_market_gate_pass"),
    ), pred_col=pred_col, forecast_floor=forecast_floor,
        signal_policy=signal_policy, rank_predictions=rank_predictions)
    return df.with_columns(
        pl.col("_forecast_pass").alias("_prediction_pass"),
        (pl.col("_forecast_pass") & pl.col("_net_margin_pass")).alias("_forecast_pass"),
    )


def _gated_layer_stats(signal_frame: pl.DataFrame, pred_col: str) -> dict:
    if signal_frame.is_empty() or pred_col not in signal_frame.columns:
        return {
            "total_rows": 0,
            "prediction_pass": {"rows": 0, "dates": 0},
            "entry_rules_pass": {"rows": 0, "dates": 0},
            "market_gate_pass": {"rows": 0, "dates": 0},
            "cumulative": {
                "prediction_and_entry": 0,
                "prediction_and_market": 0,
                "prediction_entry_market": 0,
            },
        }

    def count(expr: pl.Expr) -> dict:
        selected = signal_frame.filter(expr)
        return {
            "rows": int(selected.height),
            "dates": int(selected["date"].n_unique()) if not selected.is_empty() else 0,
        }

    prediction = (
        pl.col("_prediction_pass")
        if "_prediction_pass" in signal_frame.columns
        else pl.col("_forecast_pass")
    )
    entry = pl.col("_entry_rules_pass")
    market = pl.col("_market_gate_pass")
    net_margin = pl.col("_net_margin_pass") if "_net_margin_pass" in signal_frame.columns else pl.lit(True)
    return {
        "total_rows": int(signal_frame.height),
        "prediction_pass": count(prediction),
        "net_margin_pass": count(net_margin),
        "entry_rules_pass": count(entry),
        "market_gate_pass": count(market),
        "cumulative": {
            "prediction_and_entry": int(signal_frame.filter(prediction & entry).height),
            "prediction_and_market": int(signal_frame.filter(prediction & market).height),
            "prediction_entry_market": int(signal_frame.filter(prediction & entry & market).height),
            "prediction_net_margin": int(signal_frame.filter(prediction & net_margin).height),
            "prediction_net_margin_entry_market": int(signal_frame.filter(prediction & net_margin & entry & market).height),
        },
    }


def _run_one(
    panel: pl.DataFrame,
    factor: pl.DataFrame,
    engine: BacktestEngine,
    *,
    horizon: int,
    forecast_floor: float,
    max_positions: int,
    initial_capital: float,
    signal_frame: pl.DataFrame | None = None,
    max_exposure_pct: float = 1.0,
    exit_mode: str = "max_hold",
    exit_signal_frame: pl.DataFrame | None = None,
    entry_signal_end: date | None = None,
    forecast_ceiling: float | None = None,
    turnover_penalty: float = 0.0,
    holding_lag_days: int = 0,
) -> dict:
    if exit_mode not in {"max_hold", "prediction_reverse", "market_gate_close"}:
        raise ValueError(f"unknown exit_mode: {exit_mode}")
    pred_col = f"pred_ret_{horizon}d"
    if signal_frame is not None:
        forecast_pass = (
            pl.col("_forecast_pass")
            if "_forecast_pass" in signal_frame.columns
            else pl.col(pred_col).fill_null(-1.0) > forecast_floor
        )
        candidates = signal_frame.filter(
            forecast_pass
            & (pl.col("_net_margin_pass") if "_net_margin_pass" in signal_frame.columns else pl.lit(True))
            & pl.col("_entry_rules_pass")
            & pl.col("_market_gate_pass")
        )
    else:
        candidates = factor.filter(pl.col(pred_col).fill_null(-1.0) > forecast_floor)
    if entry_signal_end is not None:
        candidates = candidates.filter(pl.col("date") <= entry_signal_end)
    if forecast_ceiling is not None:
        candidates = candidates.filter(pl.col(pred_col) <= forecast_ceiling)
    if candidates.is_empty() or panel.is_empty():
        return _empty_stats()
    scored_panel, entries, _ = _build_ranked_entry_orders(
        panel, candidates, pred_col=pred_col, max_positions=max_positions,
        score_col="net_margin_score" if "net_margin_score" in candidates.columns else None,
        turnover_penalty=turnover_penalty,
        holding_lag_days=holding_lag_days,
    )
    exit_frame = exit_signal_frame if exit_signal_frame is not None else signal_frame
    if exit_mode != "max_hold" and exit_frame is None:
        raise ValueError(f"exit signal frame required for {exit_mode}")
    exit_keys: set[tuple[str, date]] = set()
    if exit_mode == "prediction_reverse" and exit_frame is not None:
        exit_keys = {
            (str(row["symbol"]), row["date"])
            for row in exit_frame.filter(pl.col(pred_col).is_finite() & (pl.col(pred_col) <= 0.0))
            .select("symbol", "date").to_dicts()
        }
    elif exit_mode == "market_gate_close" and exit_frame is not None:
        exit_keys = {
            (str(row["symbol"]), row["date"])
            for row in exit_frame.filter(~pl.col("_market_gate_pass"))
            .select("symbol", "date").to_dicts()
        }
    exits = pl.Series(
        "_exit",
        [(str(row["symbol"]), row["date"]) in exit_keys
         for row in scored_panel.select("symbol", "date").to_dicts()],
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
        max_positions=max_positions,
        max_exposure_pct=max_exposure_pct,
        initial_capital=initial_capital,
    )
    result = engine.simulate_portfolio(scored_panel, entries, exits, config)
    stats = result.stats or {}
    return {
        "total_return": round(float(stats.get("total_return", 0.0)) * 100, 2),
        "max_drawdown": round(float(stats.get("max_drawdown", 0.0)) * 100, 2),
        "trade_count": int(stats.get("n_trades", len(result.trades))),
        "win_rate": round(float(stats.get("win_rate", 0.0)) * 100, 2),
        "profit_factor": stats.get("profit_factor"),
        "trading_days": len({t.entry_date for t in result.trades}),
        "execution": {
            "entry_fill": "open_t+1",
            "exit_fill": "open_t+1",
            "commission_pct": 0.0003,
            "stamp_tax_pct": 0.0005,
            "slippage_bps": 5.0,
        },
    }


def _build_ranked_entry_orders(
    panel: pl.DataFrame,
    candidates: pl.DataFrame,
    *,
    pred_col: str,
    max_positions: int,
    score_col: str | None = None,
    turnover_penalty: float = 0.0,
    holding_lag_days: int = 0,
) -> tuple[pl.DataFrame, pl.Series, pl.DataFrame]:
    score_col = score_col if score_col and score_col in candidates.columns else pred_col
    if turnover_penalty <= 0.0 and holding_lag_days <= 0:
        ranked = (
            candidates.sort(["date", score_col, "symbol"], descending=[False, True, False])
            .with_columns(pl.col(score_col).rank(method="ordinal", descending=True).over("date").alias("rank"))
            .filter(pl.col("rank") <= max_positions)
        )
    else:
        # Select each date sequentially so a recently selected symbol keeps a
        # small score advantage. This is a deterministic turnover penalty, not
        # a future-label lookup, and applies only to the candidate ranking.
        selected_rows: list[dict] = []
        recent_symbols: dict[str, int] = {}
        for day_index, day_frame in enumerate(candidates.sort("date").partition_by("date", maintain_order=True)):
            rows = day_frame.select("symbol", "date", pl.col(score_col).alias("_base_score")).to_dicts()
            rows.sort(
                key=lambda row: (
                    -(
                        float(row["_base_score"] or -1.0)
                        + (float(turnover_penalty) if recent_symbols.get(str(row["symbol"]), -10**9) + holding_lag_days >= day_index else 0.0)
                    ),
                    str(row["symbol"]),
                )
            )
            chosen = rows[:max_positions]
            for rank, row in enumerate(chosen, start=1):
                selected_rows.append({
                    "symbol": row["symbol"],
                    "date": row["date"],
                    "score": row["_base_score"],
                    "rank": rank,
                })
                recent_symbols[str(row["symbol"])] = day_index
        ranked = pl.DataFrame(selected_rows) if selected_rows else pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.Utf8),
            "date": pl.Series([], dtype=pl.Date),
            "score": pl.Series([], dtype=pl.Float64),
            "rank": pl.Series([], dtype=pl.Int64),
        })
    score_rows = (
        ranked.select("symbol", "date", pl.col("score").cast(pl.Float64))
        if "score" in ranked.columns
        else ranked.select("symbol", "date", pl.col(score_col).cast(pl.Float64).alias("score"))
    )
    if score_rows.select(pl.struct("symbol", "date").is_duplicated().any()).item():
        raise ValueError("duplicate prediction keys in ranked entry orders")
    # The matcher reads score from the signal day when a next-open order fills.
    scored_panel = (panel.drop("score") if "score" in panel.columns else panel).join(
        score_rows, on=["symbol", "date"], how="left"
    ).with_columns(pl.col("score").fill_null(0.0))
    keys = {(str(row["symbol"]), row["date"]) for row in ranked.select("symbol", "date").to_dicts()}
    entries = pl.Series(
        "_entry",
        [(str(row["symbol"]), row["date"]) in keys for row in scored_panel.select("symbol", "date").to_dicts()],
        dtype=pl.Boolean,
    )
    return scored_panel, entries, ranked


def _candidate_specs(feature_columns: list[str]) -> dict[str, list[str]]:
    """Return small, reproducible model candidates for the research run.

    These are deliberately limited to features available in the rebuilt
    historical panel.  A candidate is a Ridge rank model; the existing GBDT
    model remains the production baseline and is evaluated beside it.
    """
    from app.services import stockformer_training as training

    families = {
        "daily_ohlcv": (
            "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "ma20_gap",
            "vol_ratio_5d", "intraday_strength", "amplitude", "rsi_proxy",
            "risk_20d", "gap_open", "amount_ratio_5d", "price_position_20d",
            "price_range_10d",
        ),
        "daily_plus_moneyflow": (
            "ret_1d", "ret_3d", "ret_5d", "ret_20d", "ma20_gap", "vol_ratio_5d",
            "intraday_strength", "risk_20d", "amount_ratio_5d", "price_position_20d",
            "moneyflow_strength_3d", "moneyflow_strength_5d",
            "moneyflow_acceleration_3d", "moneyflow_accumulation_score",
            "moneyflow_absorption_efficiency",
        ),
        "daily_plus_auction": (
            "ret_1d", "ret_3d", "ret_5d", "ma20_gap", "vol_ratio_5d",
            "intraday_strength", "risk_20d", "amount_ratio_5d", "auction_amount_ratio",
            "auction_volume_ratio",
        ),
        "daily_plus_minute": training.COMPACT_FEATURES,
        "daily_full": tuple(feature_columns),
    }
    return {
        name: [column for column in columns if column in feature_columns]
        for name, columns in families.items()
        if any(column in feature_columns for column in columns)
    }


def _fit_candidate_predictions(
    frame: pl.DataFrame,
    *,
    horizon: int,
    columns: list[str],
    alpha: float,
    fit_start: date | None = None,
    fit_end: date,
    predict_start: date,
    predict_end: date | None = None,
    target_mode: str = "absolute",
) -> pl.DataFrame:
    """Fit on a bounded chronological history and predict later rows."""
    from app.services import stockformer_training as training

    target = training._target_column(frame, horizon, target_mode)
    if target not in frame.columns:
        target = f"target_{horizon}d"
    train = training.purged_slice(
        frame, horizon, fit_start or frame["date"].min(), fit_end
    )
    if train.is_empty():
        return pl.DataFrame({"symbol": [], "date": [], f"pred_ret_{horizon}d": []})
    train = train.with_columns(
        training._fit_target_expr(target, target_mode).alias("_fit_target")
    )
    x, y, _ = sf._prepare_xy(train, "_fit_target", columns)
    if len(y) < 100 or train["date"].n_unique() < 20:
        return pl.DataFrame({"symbol": [], "date": [], f"pred_ret_{horizon}d": []})

    # Fit the tail correction on the last chronological part of the training
    # sample, then refit the predictor on all rows before the window.  This
    # keeps the calibration independent of the future window being replayed.
    calibration = {"enabled": False, "reason": "no_calibration_slice", "rows": 0}
    train_dates = sorted(train["date"].unique().to_list())
    if len(train_dates) >= 24:
        calibration_start = train_dates[max(1, int(len(train_dates) * 0.80))]
        calibration_part = train.filter(pl.col("date") >= calibration_start)
        calibration_fit = train.filter(pl.col("date") < calibration_start)
        cx, _, c_used = sf._prepare_xy(calibration_part, target, columns)
        fx, fy, _ = sf._prepare_xy(calibration_fit, "_fit_target", columns)
        if len(fy) >= 100 and c_used.height:
            calibration_weights, calibration_scaler = sf._fit_ridge(
                fx, fy, alpha, len(columns), clip_quantiles=(1, 99),
            )
            calibration_prediction = sf._predict(cx, calibration_weights, calibration_scaler)
            calibration = sf._fit_extreme_prediction_calibration(
                calibration_prediction, c_used[target].to_numpy(),
            )
    weights, scaler = sf._fit_ridge(x, y, alpha, len(columns), clip_quantiles=(1, 99))
    expr = pl.col("date") >= predict_start
    if predict_end is not None:
        expr = expr & (pl.col("date") < predict_end)
    prediction_rows = frame.filter(expr).with_columns(pl.lit(0.0).alias("_prediction_target"))
    if prediction_rows.is_empty():
        return pl.DataFrame({"symbol": [], "date": [], f"pred_ret_{horizon}d": []})
    tx, _, used = sf._prepare_xy(prediction_rows, "_prediction_target", columns)
    raw_prediction = sf._predict(tx, weights, scaler)
    prediction = sf._apply_extreme_prediction_calibration(raw_prediction, calibration)
    return used.select(["symbol", "date"]).with_columns(
        pl.Series(f"pred_ret_raw_{horizon}d", raw_prediction),
        pl.Series(f"pred_ret_{horizon}d", prediction),
    )


def _model_score(stats: dict) -> tuple[float, float, int]:
    """Prefer net return, then smaller drawdown, while rejecting no-trade fits."""
    trades = int(stats.get("trade_count", 0) or 0)
    if trades == 0:
        return (-float("inf"), -float("inf"), 0)
    total_return = float(stats.get("total_return", 0.0) or 0.0)
    drawdown = abs(float(stats.get("max_drawdown", 0.0) or 0.0))
    return (total_return - 0.35 * drawdown, total_return, trades)


def run_portfolio_research(
    data_dir: Path,
    *,
    model_id: str | None = None,
    max_positions: int = 3,
    initial_capital: float = 10000.0,
    on_progress=None,
) -> dict:
    def progress(stage: str, pct: int, msg: str) -> None:
        if on_progress:
            on_progress(stage, pct, msg)

    progress("load_factors", 5, "加载5日/10日样本外预测")
    meta = sf._model_meta(data_dir, model_id)
    if meta is None:
        return {"error": "暂无 StockFormer 模型，请先训练模型"}
    frozen_signal_policy = load_frozen_signal_policy(data_dir, meta)
    if frozen_signal_policy:
        # Once a policy is frozen, the legacy portfolio researcher must not
        # replay the historical holdout and select another forecast floor.
        # Route callers through the frozen-policy rolling evaluator so the
        # pre-freeze windows stay excluded even when the old API is used.
        from app.services.stockformer_rolling import run_rolling_portfolio_research

        progress("frozen_policy", 10, "规则已冻结，跳过旧样本外阈值调优，检查新增窗口")
        return run_rolling_portfolio_research(
            data_dir,
            model_id=model_id,
            max_positions=max_positions,
            initial_capital=initial_capital,
            on_progress=on_progress,
        )
    path = sf._resolve_artifact_path(data_dir, meta.get("factor_file"))
    if path is None or not path.exists():
        return {"error": "模型缺少样本外因子文件"}
    factor = pl.read_parquet(path)
    if factor.is_empty() or "date" not in factor.columns:
        return {"error": "样本外因子为空"}
    symbols = [str(s) for s in factor["symbol"].unique().to_list() if s]
    start = factor["date"].min()
    factor_end = factor["date"].max()
    end = factor_end + timedelta(days=15)
    data_start, _ = sf._date_range_from_data(data_dir)
    history_start = max(data_start or (start - timedelta(days=730)), start - timedelta(days=730))
    progress("load_panel", 20, "加载调优历史与真实组合回测行情")
    repo = KlineRepository(DataStore(data_dir))
    engine = BacktestEngine(repo)
    # Load enough history to fit candidates, while retaining the exact same
    # panel for the production matcher and the feature builder.
    training_profile = str(meta.get("training_profile") or "daily_baseline")
    panel = sf._load_panel(
        data_dir, history_start, end, symbols=symbols,
        include_minute=training_profile == "minute_enhanced",
    )
    if panel.is_empty():
        return {"error": "真实组合回测缺少行情面板"}
    backtest_panel = panel.filter(pl.col("date") >= start)
    if backtest_panel.is_empty():
        return {"error": "样本外区间缺少真实组合行情"}
    progress("build_features", 35, "构建调优特征与前向收益标签")
    try:
        feature_frame, minute_coverage = _prepare_research_features(panel, training_profile)
    except ValueError as exc:
        return {"error": str(exc)}
    feature_columns = [column for column in sf.training_feature_columns(training_profile) if column in feature_frame.columns]
    if feature_frame.is_empty() or not feature_columns:
        return {"error": "模型调优缺少可用特征"}
    factor = factor.join(feature_frame.select("symbol", "date"), on=["symbol", "date"], how="inner")
    if factor.is_empty():
        return {"error": "样本外预测没有通过分钟覆盖门槛的股票日"}

    # Build production-equivalent gates once.  Candidate predictions can then
    # be swapped into the same gated rows without changing the historical
    # universe, instrument filter, or market state used by the comparison.
    gated_signals: dict[int, pl.DataFrame] = {}
    for horizon in (5, 10):
        if f"pred_ret_{horizon}d" in factor.columns:
            gated = _build_gated_signal_frame(
                data_dir,
                factor,
                feature_frame,
                meta,
                horizon=horizon,
            )
            if not gated.is_empty():
                gated_signals[horizon] = gated

    progress("tune_models", 50, "调优5日/10日模型候选")
    midpoint = start + (factor_end - start) / 2
    tune_factor = factor.filter(pl.col("date") < midpoint)
    confirm_factor = factor.filter(pl.col("date") >= midpoint)
    tune_panel = backtest_panel.filter(pl.col("date") < midpoint + timedelta(days=15))
    confirm_panel = backtest_panel.filter(pl.col("date") >= midpoint)
    candidate_specs = _candidate_specs(feature_columns)
    alpha_grid = (0.25, 1.0, 4.0, 16.0)
    candidates = {}
    model_candidates = {}

    def align_prediction(prediction: pl.DataFrame, target_factor: pl.DataFrame, horizon: int) -> pl.DataFrame:
        pred_col = f"pred_ret_{horizon}d"
        if prediction.is_empty() or target_factor.is_empty():
            return pl.DataFrame({"symbol": [], "date": [], pred_col: []})
        return target_factor.select(["symbol", "date"]).join(
            prediction.select(["symbol", "date", pred_col]),
            on=["symbol", "date"],
            how="inner",
        )

    for horizon in (5, 10):
        if f"pred_ret_{horizon}d" not in factor.columns:
            continue
        pred_col = f"pred_ret_{horizon}d"
        gated_signal = gated_signals.get(horizon)
        if gated_signal is None or gated_signal.is_empty():
            continue
        tune_gated_signal = gated_signal.filter(pl.col("date") < midpoint)
        confirm_gated_signal = gated_signal.filter(pl.col("date") >= midpoint)
        trials = []
        for floor in _FORECAST_FLOORS:
            tuning_stats = _run_one(
                tune_panel,
                tune_factor,
                engine,
                horizon=horizon,
                forecast_floor=floor,
                max_positions=max_positions,
                initial_capital=initial_capital,
                signal_frame=tune_gated_signal,
            )
            confirmation_stats = _run_one(
                confirm_panel,
                confirm_factor,
                engine,
                horizon=horizon,
                forecast_floor=floor,
                max_positions=max_positions,
                initial_capital=initial_capital,
                signal_frame=confirm_gated_signal,
            )
            trials.append({"forecast_floor": floor, "tuning_stats": tuning_stats, "stats": confirmation_stats})
        best = max(
            trials,
            key=lambda item: _model_score(item.get("tuning_stats") or {}),
        )

        candidate_trials = []
        for family, columns in candidate_specs.items():
            for alpha in alpha_grid:
                try:
                    tune_prediction = _fit_candidate_predictions(
                        feature_frame, horizon=horizon, columns=columns, alpha=alpha,
                        fit_end=start, predict_start=start, predict_end=midpoint,
                        target_mode=str(meta.get("target_mode") or "absolute"),
                    )
                    tune_candidate_factor = align_prediction(tune_prediction, tune_factor, horizon)
                    tune_candidate_signal = tune_gated_signal.drop(pred_col).join(
                        tune_candidate_factor.select(["symbol", "date", pred_col]),
                        on=["symbol", "date"],
                        how="inner",
                    )
                    tuning_stats = _run_one(
                        tune_panel, tune_candidate_factor, engine, horizon=horizon,
                        forecast_floor=0.0, max_positions=max_positions,
                        initial_capital=initial_capital,
                        signal_frame=tune_candidate_signal,
                    )
                    candidate_trials.append({
                        "family": family,
                        "alpha": alpha,
                        "feature_count": len(columns),
                        "tuning_stats": tuning_stats,
                        "status": "ok",
                    })
                except Exception as exc:  # noqa: BLE001
                    candidate_trials.append({
                        "family": family,
                        "alpha": alpha,
                        "feature_count": len(columns),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "tuning_stats": _empty_stats(),
                    })
        selected_candidate = max(
            candidate_trials,
            key=lambda item: _model_score(item.get("tuning_stats") or {}),
            default=None,
        )
        selected_model = None
        if selected_candidate and selected_candidate.get("status") == "ok":
            family = selected_candidate["family"]
            alpha = float(selected_candidate["alpha"])
            columns = candidate_specs[family]
            try:
                # Refit only after model selection, using every label available
                # before confirmation. The confirmation window is untouched.
                confirmation_prediction = _fit_candidate_predictions(
                    feature_frame, horizon=horizon, columns=columns, alpha=alpha,
                    fit_end=midpoint, predict_start=midpoint,
                    target_mode=str(meta.get("target_mode") or "absolute"),
                )
                selected_tune_prediction = _fit_candidate_predictions(
                    feature_frame, horizon=horizon, columns=columns, alpha=alpha,
                    fit_end=start, predict_start=start, predict_end=midpoint,
                    target_mode=str(meta.get("target_mode") or "absolute"),
                )
                selected_tune_factor = align_prediction(selected_tune_prediction, tune_factor, horizon)
                selected_confirm_factor = align_prediction(confirmation_prediction, confirm_factor, horizon)
                selected_tune_signal = tune_gated_signal.drop(pred_col).join(
                    selected_tune_factor.select(["symbol", "date", pred_col]),
                    on=["symbol", "date"],
                    how="inner",
                )
                selected_confirm_signal = confirm_gated_signal.drop(pred_col).join(
                    selected_confirm_factor.select(["symbol", "date", pred_col]),
                    on=["symbol", "date"],
                    how="inner",
                )
                selected_threshold_trials = []
                for floor in _FORECAST_FLOORS:
                    selected_threshold_trials.append({
                        "forecast_floor": floor,
                        "tuning_stats": _run_one(
                            tune_panel, selected_tune_factor, engine, horizon=horizon,
                            forecast_floor=floor, max_positions=max_positions,
                            initial_capital=initial_capital,
                            signal_frame=selected_tune_signal,
                        ),
                        "stats": _run_one(
                            confirm_panel, selected_confirm_factor, engine, horizon=horizon,
                            forecast_floor=floor, max_positions=max_positions,
                            initial_capital=initial_capital,
                            signal_frame=selected_confirm_signal,
                        ),
                    })
                selected_threshold = max(
                    selected_threshold_trials,
                    key=lambda item: _model_score(item.get("tuning_stats") or {}),
                )
                selected_model = {
                    "family": family,
                    "alpha": alpha,
                    "feature_count": len(columns),
                    "forecast_floor": selected_threshold["forecast_floor"],
                    "tuning_stats": selected_threshold["tuning_stats"],
                    "stats": selected_threshold["stats"],
                    "threshold_trials": selected_threshold_trials,
                }
                for item in candidate_trials:
                    if item.get("family") == family and float(item.get("alpha", -1)) == alpha:
                        item["confirmation_stats"] = selected_threshold["stats"]
                        item["selected"] = True
            except Exception as exc:  # noqa: BLE001
                selected_candidate["status"] = "refit_failed"
                selected_candidate["error"] = f"{type(exc).__name__}: {exc}"
        model_candidates[str(horizon)] = {
            "horizon": horizon,
            "feature_candidates": list(candidate_specs),
            "alpha_grid": list(alpha_grid),
            "selection_metric": "调优期真实净收益 - 0.35 * 绝对最大回撤；确认期只评估",
            "candidates": candidate_trials,
            "selected": selected_model,
        }
        candidates[str(horizon)] = {
            "horizon": horizon,
            "tuning_period": {"start": str(tune_factor["date"].min()), "end": str(tune_factor["date"].max())} if not tune_factor.is_empty() else None,
            "confirmation_period": {"start": str(confirm_factor["date"].min()), "end": str(confirm_factor["date"].max())} if not confirm_factor.is_empty() else None,
            "trials": trials,
            "selected": {"forecast_floor": best["forecast_floor"], "stats": best["stats"], "tuning_stats": best["tuning_stats"]},
            "selected_model": selected_model,
            "layer_diagnostics": _gated_layer_stats(gated_signal, f"pred_ret_{horizon}d"),
        }
    progress("confirm", 90, "生成模型确认期收益与回撤")
    progress("done", 100, "模型调优与真实组合回测完成")
    return sf._json_safe({
        "model": meta,
        "evaluation_kind": "real_portfolio_model_tuning_backtest",
        "execution_warning": "模型候选和门槛只用调优期选择；确认期只评估；所有收益来自生产组合撮合器，未自动激活模型。",
        "sample_out": {"start": str(start), "end": str(factor_end)},
        "training_profile": training_profile,
        "feature_coverage": minute_coverage,
        "training_period": {"start": str(history_start), "end": str(start)},
        "tuning_period": {"start": str(tune_factor["date"].min()), "end": str(tune_factor["date"].max())} if not tune_factor.is_empty() else None,
        "confirmation_period": {"start": str(confirm_factor["date"].min()), "end": str(confirm_factor["date"].max())} if not confirm_factor.is_empty() else None,
        "gating": {
            str(horizon): _gated_layer_stats(signal, f"pred_ret_{horizon}d")
            for horizon, signal in gated_signals.items()
        },
        "model_tuning": model_candidates,
        "horizons": candidates,
    })
