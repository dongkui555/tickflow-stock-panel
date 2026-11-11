"""Paired XGBoost/LightGBM/blend comparison for StockFormer research."""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def _calibrate_prediction(prediction: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit a positive affine calibration on a time-ordered validation slice."""
    prediction = np.asarray(prediction, dtype=float)
    actual = np.asarray(actual, dtype=float)
    valid = np.isfinite(prediction) & np.isfinite(actual)
    if int(valid.sum()) < 100 or float(np.std(prediction[valid])) < 1e-10:
        return prediction, {"status": "insufficient_validation", "slope": 1.0, "intercept": 0.0, "rows": int(valid.sum())}
    slope, intercept = np.polyfit(prediction[valid], actual[valid], 1)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0:
        return np.full_like(prediction, np.nan), {
            "status": "negative_or_unstable_slope", "slope": float(slope), "intercept": float(intercept),
            "rows": int(valid.sum()),
        }
    return prediction * float(slope) + float(intercept), {
        "status": "calibrated", "slope": float(slope), "intercept": float(intercept), "rows": int(valid.sum())
    }


def _calibrate_profit_probability(prediction: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map trade scores to P(after-cost return > 0) on a frozen validation slice."""
    prediction = np.asarray(prediction, dtype=float)
    actual = np.asarray(actual, dtype=float)
    valid = np.isfinite(prediction) & np.isfinite(actual)
    labels = (actual[valid] > 0).astype(int)
    if int(valid.sum()) < 100 or len(np.unique(labels)) < 2:
        return np.full_like(prediction, np.nan), {"status": "insufficient_validation", "rows": int(valid.sum())}
    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1.0, solver="lbfgs")
        model.fit(prediction[valid].reshape(-1, 1), labels)
        out = np.full_like(prediction, np.nan)
        out[valid] = model.predict_proba(prediction[valid].reshape(-1, 1))[:, 1]
        return out, {"status": "calibrated", "rows": int(valid.sum()), "threshold": 0.55}
    except Exception as exc:  # noqa: BLE001
        return np.full_like(prediction, np.nan), {"status": "calibration_error", "rows": int(valid.sum()), "error": str(exc)}


def _portfolio_metrics(frame: pl.DataFrame, panel: pl.DataFrame, prediction: str, *, trade_prediction: str | None = None,
                       top_n: int = 2,
                       holding_days: int = 3, initial_capital: float = 100_000.0) -> dict:
    """Small account-level replay with the same execution constraints for all models."""
    if frame.is_empty() or panel.is_empty():
        return {"strategy_return": 0.0, "benchmark_return": 0.0, "excess_return": 0.0,
        "max_drawdown": 0.0, "win_rate": 0.0, "signal_count": 0,
                "trade_count": 0, "blocked_limit_up": 0, "blocked_limit_down": 0,
                "no_signal_days": 0, "eligible_days": 0}
    fields = [c for c in ("date", "symbol", "open", "high", "low", "close", "pre_close") if c in panel.columns]
    first_signal_date = frame["date"].min()
    market = (panel.select(fields).filter(pl.col("date") >= first_signal_date)
              .unique(["date", "symbol"]).sort(["date", "symbol"]))
    signal_frame = frame
    if "industry" in signal_frame.columns:
        signal_frame = signal_frame.with_columns(
            (pl.col(prediction) - pl.col(prediction).mean().over(["date", "industry"])).alias("_rank_score")
        )
    else:
        signal_frame = signal_frame.with_columns(pl.col(prediction).alias("_rank_score"))
    join_columns = [c for c in ("date", "symbol", prediction, trade_prediction, "trade_probability", "_rank_score", "industry",
                                "market_gate", "market_allows_open", "market_regime_score", "data_coverage_score")
                    if c and c in signal_frame.columns]
    join_columns = list(dict.fromkeys(join_columns))
    rows = market.join(signal_frame.select(join_columns), on=["date", "symbol"], how="left")
    dates = sorted(rows["date"].unique().to_list())
    by_key = {(r["date"], r["symbol"]): r for r in rows.iter_rows(named=True)}
    rows_by_date = {d: [r for r in rows.filter(pl.col("date") == d).iter_rows(named=True)] for d in dates}
    symbols = sorted(rows["symbol"].unique().to_list())
    prev_close = {}
    positions: dict[str, dict] = {}
    pending_entries: list[dict] = []
    cash = initial_capital
    equity_curve = []
    trades = []
    blocked_up = blocked_down = signals = 0
    no_signal_days = eligible_days = 0
    commission, stamp, slip = 0.0002, 0.0005, 0.0005

    def valid(v):
        return v is not None and np.isfinite(float(v)) and float(v) > 0

    def locked(row, direction):
        p = row.get("pre_close") or prev_close.get(row["symbol"])
        if not valid(p) or not all(valid(row.get(c)) for c in ("open", "high", "low", "close")):
            return False
        limit = float(p) * (1.1 if direction == "up" else 0.9)
        vals = [float(row[c]) for c in ("open", "high", "low", "close")]
        return abs(vals[0] - limit) / max(limit, 0.01) < 0.012 and max(vals) - min(vals) <= max(abs(vals[0]) * 1e-4, 0.01)

    for day_index, d in enumerate(dates):
        # Mark and execute scheduled exits at today's open. T+1 is enforced by
        # requiring at least one full session after entry.
        for sym in list(positions):
            pos = positions[sym]
            row = by_key.get((d, sym))
            if row is None or day_index <= pos["entry_index"] + holding_days - 1:
                continue
            if locked(row, "down"):
                blocked_down += 1
                continue
            if not valid(row.get("open")):
                continue
            exit_price = float(row["open"])
            proceeds = pos["shares"] * exit_price * (1 - commission - stamp - slip)
            pnl = proceeds - pos["cost"]
            cash += proceeds
            trades.append(pnl / pos["cost"] if pos["cost"] else 0.0)
            del positions[sym]
        # A signal observed after today's close can only consume cash and
        # become a position at the following session's actual open.
        for order in pending_entries:
            if len(positions) >= top_n or order["symbol"] in positions:
                continue
            row = by_key.get((d, order["symbol"]))
            if row is None or not valid(row.get("open")):
                continue
            if locked(row, "up"):
                blocked_up += 1
                continue
            equity = cash + sum(
                p["shares"] * float(by_key.get((d, s), {}).get("open") or p["entry_price"])
                for s, p in positions.items()
            )
            allocation = min(cash, equity / top_n)
            entry_price = float(row["open"]) * (1 + slip)
            shares = int(allocation / (entry_price * (1 + commission)) / 100) * 100
            cost = shares * entry_price * (1 + commission)
            if shares <= 0 or cost > cash:
                continue
            cash -= cost
            positions[order["symbol"]] = {"shares": shares, "cost": cost, "entry_price": entry_price,
                                          "entry_index": day_index}
        pending_entries = []
        # Signals are ranked only; the expected-return gate is applied to the
        # absolute prediction after the target correction below.
        candidates = []
        day_rows = rows.filter(pl.col("date") == d)
        if "market_allows_open" in day_rows.columns:
            day_rows = day_rows.filter(pl.col("market_allows_open").fill_null(False))
        if "market_gate" in day_rows.columns:
            day_rows = day_rows.filter(~pl.col("market_gate").is_in(["closed", "observe", "strict"]))
        if "data_coverage_score" in day_rows.columns:
            day_rows = day_rows.filter(pl.col("data_coverage_score").fill_null(0.0) >= 0.60)
        for r in day_rows.sort("_rank_score", descending=True).iter_rows(named=True):
            p = r.get(trade_prediction or prediction)
            probability = r.get("trade_probability")
            if (probability is not None and np.isfinite(float(probability))
                    and float(probability) < 0.55) or p is None or not np.isfinite(float(p)):
                continue
            signals += 1
            if r["symbol"] not in positions:
                candidates.append(r)
        if candidates:
            eligible_days += 1
        else:
            no_signal_days += 1
        slots = max(0, top_n - len(positions))
        selected_industries = set()
        selected = []
        for r in candidates:
            industry = r.get("industry") or r["symbol"]
            if industry in selected_industries:
                continue
            selected_industries.add(industry)
            selected.append(r)
            if len(selected) >= slots:
                break
        if day_index + 1 < len(dates):
            pending_entries = [{"symbol": r["symbol"]} for r in selected]
        # Close-to-close mark-to-market for drawdown and cash-only days.
        value = cash
        for sym, pos in positions.items():
            row = by_key.get((d, sym))
            if row and valid(row.get("close")):
                value += pos["shares"] * float(row["close"])
        equity_curve.append(value)
        for sym in symbols:
            row = by_key.get((d, sym))
            if row and valid(row.get("close")):
                prev_close[sym] = float(row["close"])
    for sym, pos in list(positions.items()):
        row = by_key.get((dates[-1], sym), {})
        if valid(row.get("close")):
            proceeds = pos["shares"] * float(row["close"]) * (1 - commission - stamp - slip)
            trades.append((proceeds - pos["cost"]) / pos["cost"] if pos["cost"] else 0.0)
            cash += proceeds
    curve = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(np.r_[initial_capital, curve])
    dd = np.min(np.r_[initial_capital, curve] / peak - 1) if len(curve) else 0.0
    # Keep the old close-to-close series for audit, but do not use it as the
    # primary excess-return benchmark: the strategy enters at next open and
    # exits after a fixed holding period.
    daily = rows.group_by("date").agg(pl.col("close").mean().alias("close")).sort("date")
    bench = daily["close"].pct_change().drop_nulls().to_numpy()
    close_benchmark_return = float(np.prod(1 + np.nan_to_num(bench)) - 1) if len(bench) else 0.0

    # Matched candidate-pool benchmark: on each signal day, buy every valid
    # candidate equally at the next session's open and mark it at the same
    # holding-period open used by the account replay. This is deliberately
    # cost-free and unfiltered: it answers whether the model beats the raw
    # candidate pool, while the account replay answers whether it is tradable.
    matched_returns = []
    matched_net_returns = []
    # Use non-overlapping windows. Compounding an overlapping 3-day return
    # once per calendar day would count the same holding period three times
    # and can manufacture a wildly negative/positive benchmark.
    for day_index in range(0, max(0, len(dates) - holding_days - 1), holding_days):
        d = dates[day_index]
        entry_day = dates[day_index + 1]
        exit_index = day_index + 1 + holding_days
        exit_day = dates[exit_index] if exit_index < len(dates) else dates[-1]
        entry_rows = [r for r in rows_by_date.get(entry_day, [])
                      if valid(r.get("open")) and not locked(r, "up")]
        if not entry_rows:
            continue
        returns = []
        net_returns = []
        for entry in entry_rows:
            exit_row = by_key.get((exit_day, entry["symbol"]))
            if exit_row is None or not valid(exit_row.get("open")) or locked(exit_row, "down"):
                continue
            gross = float(exit_row["open"]) / float(entry["open"]) - 1.0
            net = (float(exit_row["open"]) * (1 - commission - stamp - slip)) / (
                float(entry["open"]) * (1 + slip) * (1 + commission)
            ) - 1.0
            returns.append(gross)
            net_returns.append(net)
        if returns:
            matched_returns.append(float(np.mean(returns)))
            matched_net_returns.append(float(np.mean(net_returns)))
    matched_benchmark_return = float(np.prod(1 + np.nan_to_num(matched_returns)) - 1) if matched_returns else 0.0
    matched_net_benchmark_return = float(np.prod(1 + np.nan_to_num(matched_net_returns)) - 1) if matched_net_returns else 0.0
    benchmark_return = matched_net_benchmark_return
    strategy_return = float(cash / initial_capital - 1)
    return {"strategy_return": round(strategy_return * 100, 4),
            "benchmark_return": round(benchmark_return * 100, 4),
            "excess_return": round((strategy_return - benchmark_return) * 100, 4),
            "matched_benchmark_return": round(matched_benchmark_return * 100, 4),
            "matched_net_benchmark_return": round(matched_net_benchmark_return * 100, 4),
            "close_to_close_benchmark_return": round(close_benchmark_return * 100, 4),
            "top2_after_cost_return": round(strategy_return * 100, 4),
            "max_drawdown": round(float(dd) * 100, 4),
            "win_rate": round(float(np.mean(np.asarray(trades) > 0) * 100), 4) if trades else 0.0,
            "signal_count": int(signals), "trade_count": int(len(trades)),
            "blocked_limit_up": int(blocked_up), "blocked_limit_down": int(blocked_down),
            "no_signal_days": int(no_signal_days), "eligible_days": int(eligible_days),
            "trading_days": len(dates)}


def _metrics(frame: pl.DataFrame, prediction: str, target: str) -> dict:
    from app.services.stockformer_training import metrics
    return metrics(frame.select(["date", "symbol"]), frame[prediction].to_numpy(), frame[target].to_numpy())


def _stratified_diagnostics(frame: pl.DataFrame, prediction: str, target: str) -> dict:
    """Expose where a signal works or fails without tuning on the holdout."""
    if frame.is_empty() or target not in frame.columns:
        return {}
    out = {}
    work = frame.with_columns(
        pl.when(pl.col("data_coverage_score") >= 0.80).then(pl.lit("high"))
        .when(pl.col("data_coverage_score") >= 0.60).then(pl.lit("medium"))
        .otherwise(pl.lit("low")).alias("coverage_bucket")
        if "data_coverage_score" in frame.columns else pl.lit("unknown").alias("coverage_bucket"),
        pl.when(pl.col("market_regime_score") >= 60.0).then(pl.lit("strong"))
        .when(pl.col("market_regime_score") <= 40.0).then(pl.lit("weak"))
        .otherwise(pl.lit("neutral")).alias("market_bucket")
        if "market_regime_score" in frame.columns else pl.lit("unknown").alias("market_bucket"),
    )
    if "amount" in work.columns:
        q = work.select(pl.col("amount").cast(pl.Float64, strict=False).quantile(0.5)).item()
        work = work.with_columns(pl.when(pl.col("amount") >= (q or 0.0)).then(pl.lit("high")).otherwise(pl.lit("low")).alias("liquidity_bucket"))
    else:
        work = work.with_columns(pl.lit("unknown").alias("liquidity_bucket"))
    for key in ("coverage_bucket", "market_bucket", "liquidity_bucket", "industry"):
        if key not in work.columns:
            continue
        grouped = (work.filter(pl.col(prediction).is_finite() & pl.col(target).is_finite())
                   .group_by(key)
                   .agg(pl.len().alias("rows"), pl.col(prediction).mean().alias("mean_score"),
                        pl.col(target).mean().alias("mean_forward_return"))
                   .sort("rows", descending=True))
        out[key] = grouped.head(30).to_dicts()
    ranked = work.filter(pl.col(prediction).is_finite() & pl.col(target).is_finite()).with_columns(
        pl.col(prediction).rank(method="ordinal", descending=True).over("date").alias("_daily_rank"),
        pl.len().over("date").alias("_daily_n"),
    )
    out["top_bottom"] = {
        "top2_mean_return": float(ranked.filter(pl.col("_daily_rank") <= 2)[target].mean() or 0.0),
        "bottom2_mean_return": float(ranked.filter(pl.col("_daily_rank") > pl.col("_daily_n") - 2)[target].mean() or 0.0),
    }
    return out


def run(data_dir: Path, *, start: date | None = None, end: date | None = None,
        max_symbols: int = 2000, sample_out_ratio: float = 0.30) -> dict:
    from app.services import stockformer as sf
    from app.services import stockformer_training as training

    t0 = time.perf_counter()
    min_date, max_date = sf._date_range_from_data(data_dir)
    if min_date is None or max_date is None:
        return {"error": "没有可用的日K数据"}
    end = end or max_date
    start = start or max(min_date, end - timedelta(days=365 * 2))
    panel = sf._load_panel(data_dir, start - timedelta(days=120), end, max_symbols=max_symbols,
                           universe_as_of=start, repair_adjusted=False)
    if panel.is_empty():
        return {"error": "实验区间没有可用行情数据"}
    features = training.execution_targets(sf._feature_frame(panel, include_targets=False), panel)
    features = sf._fill_prediction_feature_defaults(features, sf.FEATURE_COLUMNS)
    features = features.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    # Do not train on rows whose auxiliary inputs were synthesized. The gate
    # is joined before fitting so the model learns the same tradable regime
    # that the replay will permit.
    if "data_coverage_score" in features.columns:
        features = features.filter(pl.col("data_coverage_score") >= 0.60)
    gate_for_training = sf._attach_market_gate_for_oos(data_dir, features.select(["date"]).unique())
    features = features.join(gate_for_training.select(["date", "market_regime_score"]), on="date", how="left")
    features = features.filter(pl.col("market_regime_score").fill_null(50.0) >= 60.0)
    dates = sorted(features["date"].unique().to_list())
    split_index = max(1, min(len(dates) - 10, int(len(dates) * (1 - sample_out_ratio))))
    split = dates[split_index]
    columns = list(sf.FEATURE_COLUMNS)
    prediction_sets: dict[str, dict[int, pl.DataFrame]] = {"xgboost": {}, "lightgbm": {}, "blend": {}}
    calibration_meta: dict[str, dict[str, dict]] = {"xgboost": {}, "lightgbm": {}, "blend": {}}
    for horizon in (1, 3, 5):
        target = f"target_{horizon}d"
        relative_target = f"target_{horizon}d_rel"
        rank_target = relative_target if relative_target in features.columns else target
        train = training.purged_slice(features, horizon, dates[0], split).with_columns(
            pl.col(rank_target).alias("_fit_rank_target"),
            pl.col(target).alias("_fit_trade_target"),
        )
        test = training.purged_slice(features, horizon, split)
        train_dates = sorted(train["date"].unique().to_list())
        calibration_start = train_dates[max(1, int(len(train_dates) * 0.80))] if len(train_dates) >= 10 else None
        fit_part = train.filter(pl.col("date") < calibration_start) if calibration_start else train
        calibration_part = train.filter(pl.col("date") >= calibration_start) if calibration_start else train.head(0)
        x, y, _ = sf._prepare_xy(fit_part, "_fit_rank_target", columns)
        trade_x, trade_y, _ = sf._prepare_xy(fit_part, "_fit_trade_target", columns)
        tx, ty, used = sf._prepare_xy(test, target, columns)
        trade_tx, trade_ty, trade_used = sf._prepare_xy(test, target, columns)
        if not len(y) or not len(ty) or not len(trade_y) or not len(trade_ty):
            continue
        xgb = sf._fit_xgboost(x, y, n_estimators=600)
        lgbm = sf._fit_lightgbm(x, y, n_estimators=600)
        trade_xgb = sf._fit_xgboost(trade_x, trade_y, n_estimators=600)
        trade_lgbm = sf._fit_lightgbm(trade_x, trade_y, n_estimators=600)
        xgb_pred = sf._predict_xgboost(xgb, tx)
        lgb_pred = sf._predict_lightgbm(lgbm, tx)
        xgb_trade_pred = sf._predict_xgboost(trade_xgb, trade_tx)
        lgb_trade_pred = sf._predict_lightgbm(trade_lgbm, trade_tx)
        xgb_trade_prob = lgb_trade_prob = np.full_like(xgb_trade_pred, np.nan)
        if not calibration_part.is_empty():
            cx, cy, _ = sf._prepare_xy(calibration_part, "_fit_rank_target", columns)
            ctrade_x, ctrade_y, _ = sf._prepare_xy(calibration_part, "_fit_trade_target", columns)
            xgb_trade_cal, xgb_trade_meta = _calibrate_prediction(sf._predict_xgboost(trade_xgb, ctrade_x), ctrade_y)
            lgb_trade_cal, lgb_trade_meta = _calibrate_prediction(sf._predict_lightgbm(trade_lgbm, ctrade_x), ctrade_y)
            # Fit calibrators on the time-ordered validation slice only; test
            # predictions are transformed with those frozen parameters.
            def apply(meta, values):
                if meta.get("status") != "calibrated":
                    return np.full_like(values, np.nan)
                return values * meta["slope"] + meta["intercept"]
            _, xgb_rank_meta = _calibrate_prediction(sf._predict_xgboost(xgb, cx), cy)
            _, lgb_rank_meta = _calibrate_prediction(sf._predict_lightgbm(lgbm, cx), cy)
            xgb_rank_pred = apply(xgb_rank_meta, xgb_pred)
            lgb_rank_pred = apply(lgb_rank_meta, lgb_pred)
            xgb_trade_pred = apply(xgb_trade_meta, xgb_trade_pred)
            lgb_trade_pred = apply(lgb_trade_meta, lgb_trade_pred)
            _, xgb_trade_prob_meta = _calibrate_profit_probability(
                sf._predict_xgboost(trade_xgb, ctrade_x), ctrade_y)
            _, lgb_trade_prob_meta = _calibrate_profit_probability(
                sf._predict_lightgbm(trade_lgbm, ctrade_x), ctrade_y)
            def apply_probability(meta, values):
                # Refit is deliberately avoided; probability calibration is
                # represented by a logistic map captured on validation data.
                return values
            # Store raw trade scores here; probability is populated below by
            # the validation-fitted logistic coefficients when available.
            for pred, raw, meta in (("xgb", xgb_trade_pred, xgb_trade_prob_meta), ("lgb", lgb_trade_pred, lgb_trade_prob_meta)):
                pass
        else:
            xgb_rank_pred, lgb_rank_pred = xgb_pred, lgb_pred
            xgb_trade_meta = lgb_trade_meta = {"status": "no_calibration", "slope": 1.0, "intercept": 0.0, "rows": 0}
            xgb_rank_meta = lgb_rank_meta = xgb_trade_meta
            xgb_trade_prob_meta = lgb_trade_prob_meta = {"status": "no_calibration", "rows": 0, "threshold": 0.55}
        # Probability calibration is based on validation labels and applied
        # to frozen OOS trade scores through a one-dimensional logistic fit.
        def probability_from_validation(raw_values, cal_values, cal_labels):
            try:
                from sklearn.linear_model import LogisticRegression
                valid = np.isfinite(cal_values) & np.isfinite(cal_labels)
                labels = (cal_labels[valid] > 0).astype(int)
                if valid.sum() < 100 or len(np.unique(labels)) < 2:
                    return np.full_like(raw_values, np.nan)
                lr = LogisticRegression(C=1.0, solver="lbfgs").fit(cal_values[valid].reshape(-1, 1), labels)
                return lr.predict_proba(raw_values.reshape(-1, 1))[:, 1]
            except Exception:
                return np.full_like(raw_values, np.nan)
        if not calibration_part.is_empty():
            cal_x = sf._prepare_xy(calibration_part, "_fit_trade_target", columns)[0]
            cal_y = sf._prepare_xy(calibration_part, "_fit_trade_target", columns)[1]
            xgb_trade_prob = probability_from_validation(xgb_trade_pred, sf._predict_xgboost(trade_xgb, cal_x), cal_y)
            lgb_trade_prob = probability_from_validation(lgb_trade_pred, sf._predict_lightgbm(trade_lgbm, cal_x), cal_y)
        gate_source = sf._attach_market_gate_for_oos(data_dir, features.select(["date"]).unique())
        base_columns = ["date", "symbol", target, "amount"]
        if rank_target != target:
            base_columns.append(rank_target)
        for optional in ("industry", "market_gate", "market_allows_open", "market_regime_score", "data_coverage_score",
                         "minute_data_available", "moneyflow_data_available", "auction_data_available",
                         "news_data_available"):
            if optional in used.columns or optional in gate_source.columns:
                base_columns.append(optional)
        base = used.select([c for c in base_columns if c in used.columns])
        if "market_gate" not in base.columns and "market_gate" in gate_source.columns:
            base = base.join(gate_source.select([c for c in gate_source.columns if c in ("date", "market_gate", "market_allows_open", "market_regime_score")]), on="date", how="left")
        prediction_sets["xgboost"][horizon] = base.with_columns(
            pl.Series("prediction", xgb_rank_pred), pl.Series("trade_prediction", xgb_trade_pred), pl.Series("trade_probability", xgb_trade_prob)
        )
        prediction_sets["lightgbm"][horizon] = base.with_columns(
            pl.Series("prediction", lgb_rank_pred), pl.Series("trade_prediction", lgb_trade_pred), pl.Series("trade_probability", lgb_trade_prob)
        )
        prediction_sets["blend"][horizon] = base.with_columns(
            pl.Series("prediction", 0.6 * xgb_rank_pred + 0.4 * lgb_rank_pred),
            pl.Series("trade_prediction", 0.6 * xgb_trade_pred + 0.4 * lgb_trade_pred),
            pl.Series("trade_probability", 0.6 * xgb_trade_prob + 0.4 * lgb_trade_prob),
        )
        calibration_meta["xgboost"][f"{horizon}d"] = {"rank": xgb_rank_meta, "trade": xgb_trade_meta}
        calibration_meta["lightgbm"][f"{horizon}d"] = {"rank": lgb_rank_meta, "trade": lgb_trade_meta}
        calibration_meta["blend"][f"{horizon}d"] = {"rank": xgb_rank_meta, "trade": xgb_trade_meta}
    # Do not use a DataFrame as a boolean: Polars intentionally rejects
    # ambiguous truth-value checks and would fail after all model fits.
    if 3 not in prediction_sets["xgboost"]:
        return {"error": "训练或样本外样本不足"}
    recent_dates = sorted(prediction_sets["xgboost"][3]["date"].unique().to_list())[-63:]
    report = {
        "experiment_id": f"mc_{time.strftime('%Y%m%d_%H%M%S')}",
        "kind": "stockformer_model_comparison_v1",
        "protocol": "industry_relative_rank_plus_absolute_trade_calibration_same_cost_same_top2_t+1_execution_v4",
        "code_version": "coverage_features_dual_target_calibration_stratified_diagnostics_v4",
        "target_mode": "industry_relative_rank_plus_absolute_trade",
        "execution": {
            "max_positions": 2, "holding_days": 3, "entry": "next_trading_day_open",
            "t_plus_one": True, "lot_size": 100, "commission_pct": 0.0002,
            "stamp_tax_pct": 0.0005, "slippage_bps_each_side": 5,
            "limit_up_down_blocked": True, "cash_when_no_signal": True,
        },
        "train_start": str(start), "train_end": str(dates[split_index - 1]),
        "oos_start": str(split), "oos_end": str(end),
        "recent_oos_start": str(recent_dates[0]) if recent_dates else None,
        "recent_oos_end": str(recent_dates[-1]) if recent_dates else None,
        "recent_oos_trading_days": len(recent_dates),
        "models": {},
        "elapsed_ms": 0.0,
        "promotion_status": "research_only",
        "promotion_reason": "尚未完成多窗口滚动样本外准入；本报告仅用于研究，不自动进入模拟盘。",
    }
    for name, horizons in prediction_sets.items():
        all_frame = horizons[3]
        recent = all_frame.filter(pl.col("date") >= recent_dates[0]) if recent_dates else all_frame.head(0)
        prediction_metrics = {}
        recent_metrics = {}
        for horizon, frame in horizons.items():
            target = f"target_{horizon}d"
            rank_target = f"target_{horizon}d_rel" if f"target_{horizon}d_rel" in frame.columns else target
            prediction_metrics[f"{horizon}d"] = _metrics(frame, "prediction", rank_target)
            prediction_metrics[f"{horizon}d"]["target"] = "industry_relative" if rank_target != target else "absolute"
            prediction_metrics[f"{horizon}d"]["trade_model"] = _metrics(frame, "trade_prediction", target)
            recent_frame = frame.filter(pl.col("date") >= recent_dates[0]) if recent_dates else frame.head(0)
            recent_metrics[f"{horizon}d"] = _metrics(recent_frame, "prediction", rank_target) if recent_frame.height else {}
        strategy_metrics = _portfolio_metrics(all_frame, panel, "prediction", trade_prediction="trade_prediction")
        recent_strategy_metrics = _portfolio_metrics(recent, panel, "prediction", trade_prediction="trade_prediction")
        report["models"][name] = {
            "prediction_metrics": prediction_metrics,
            "recent_3m_metrics": recent_metrics,
            "recent_3m_rank_ic": recent_metrics.get("3d", {}).get("rank_ic") if recent.height else None,
            "top2_after_cost_return": strategy_metrics.get("top2_after_cost_return"),
            "strategy": strategy_metrics,
            "recent_3m_strategy": recent_strategy_metrics,
            "calibration": calibration_meta.get(name, {}),
            "stratified_diagnostics": _stratified_diagnostics(all_frame, "prediction", "target_3d"),
        }
        full_rank_ic = prediction_metrics.get("3d", {}).get("rank_ic")
        recent_rank_ic = recent_metrics.get("3d", {}).get("rank_ic")
        full_strategy = strategy_metrics.get("strategy_return", 0.0)
        full_excess = strategy_metrics.get("excess_return", 0.0)
        recent_strategy = recent_strategy_metrics.get("strategy_return", 0.0)
        recent_excess = recent_strategy_metrics.get("excess_return", 0.0)
        reasons = []
        if full_rank_ic is None or full_rank_ic <= 0:
            reasons.append("全样本3日Rank IC非正")
        if recent_rank_ic is None or recent_rank_ic <= 0:
            reasons.append("最近3个月3日Rank IC非正")
        if full_strategy <= 0 or full_excess <= 0:
            reasons.append("全样本净收益或超额收益非正")
        if recent_strategy <= 0 or recent_excess <= 0:
            reasons.append("最近3个月净收益或超额收益非正")
        report["models"][name]["promotion_check"] = {
            "allow_simulation": not reasons and strategy_metrics.get("trade_count", 0) >= 20,
            "rank_direction": "negative_or_unreliable" if reasons else "positive",
            "status": "no_signal" if reasons else "candidate",
            "reasons": reasons or ["当前窗口通过，但仍需滚动样本外确认"],
        }
    eligible_models = [name for name, value in report["models"].items()
                       if value.get("promotion_check", {}).get("allow_simulation")]
    report["activation_policy"] = {
        "mode": "no_signal_until_rolling_oos_pass",
        "eligible_models": eligible_models,
        "requires": ["全样本和最近3个月Rank IC为正", "净收益和超额收益为正", "有效交易不少于20笔", "滚动样本外多个窗口通过"],
    }
    report["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    root = data_dir / "stockformer" / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_comparison_latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def latest(data_dir: Path) -> dict:
    path = data_dir / "stockformer" / "experiments" / "model_comparison_latest.json"
    if not path.exists():
        return {"report": None}
    return {"report": json.loads(path.read_text(encoding="utf-8"))}
