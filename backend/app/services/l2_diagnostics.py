"""Offline L2 group ablation and validation-fitted market bias diagnostics.

No model selection or promotion. A previously inspected test window is a reused
holdout, not fresh out-of-sample evidence. All arms retain the full L2 mask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from app.services import l2_experiment as experiment
from app.services import stockformer

FEATURE_GROUPS = {
    "activity": ("l2_trade_count", "l2_trade_volume", "l2_trade_amount"),
    "trade_size": ("l2_large_trade_amount_ratio", "l2_average_trade_amount"),
    "direction": ("l2_buy_amount_ratio", "l2_net_buy_amount_ratio"),
    "late_flow": ("l2_late_net_buy_ratio", "l2_late_amount_ratio"),
    "order_book": (
        "l2_depth_imbalance_mean",
        "l2_depth_imbalance_positive_fraction",
        "l2_spread_bps_mean",
    ),
    "vwap_gap": ("l2_vwap_close_gap",),
}


@dataclass(frozen=True)
class DiagnosticConfig:
    min_market_stocks: int = 200
    min_market_coverage: float = 0.8
    min_calibration_dates: int = 10
    min_state_dates: int = 5
    state_shrink_dates: int = 20

    def __post_init__(self):
        for name in (
            "min_market_stocks",
            "min_calibration_dates",
            "min_state_dates",
            "state_shrink_dates",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.min_market_coverage <= 1:
            raise ValueError("min_market_coverage must be in (0, 1]")


@dataclass
class DiagnosticResult:
    report: dict
    predictions: pl.DataFrame
    market: pl.DataFrame
    reference: experiment.L2ExperimentResult


def diagnostic_plan(config: DiagnosticConfig) -> dict:
    return {
        "schema_version": 1,
        "diagnostic_config": asdict(config),
        "feature_groups": FEATURE_GROUPS,
        "variants": list(feature_variants()),
        "sample_mask": "Full 13-feature finite/availability mask shared by every arm",
        "market": "Stored SH/SZ equity universe, not the 50-stock L2 universe; valid 20-session "
        "OHLCV histories, positive volume/amount at t and t-1; no live metadata or future data",
        "state_rules": {
            "weak": "above_ma20 < 0.40 AND (up_fraction < 0.45 OR equal_weight_return < 0)",
            "strong": "above_ma20 >= 0.60 AND up_fraction >= 0.55",
            "neutral": "Other sufficiently covered dates",
            "unknown": "Below minimum history/stock coverage; global calibration fallback",
        },
        "calibration": "Fit daily equal-weight mean(target-prediction) on purged validation only. "
        "State offset = global + n/(n+shrink_dates)*(state_mean-global), with minimum state dates. "
        "Thin dates (<top_k) excluded from calibration fitting but retained in primary evaluation. "
        "Too few calibration dates => no correction. Apply fixed offsets without updating on test.",
        "interpretation": "Market offsets are constant across stocks on a date: they cannot improve "
        "within-date ranks or Top-K selections. No trading gate or position sizing is simulated.",
        "test_status": "reused_holdout_diagnostic_not_fresh_oos",
        "promotion": False,
    }


def feature_variants() -> dict[str, tuple[str, ...]]:
    base, full = experiment.BASELINE_FEATURE_COLUMNS, experiment.L2_FEATURE_COLUMNS
    variants = {"baseline": base, "full_l2": (*base, *full)}
    for name, group in FEATURE_GROUPS.items():
        variants[f"only_{name}"] = (*base, *group)
        variants[f"without_{name}"] = (*base, *(c for c in full if c not in group))
    return variants


def build_market_history(panel: pl.DataFrame, config: DiagnosticConfig) -> pl.DataFrame:
    """Causal market proxy; gaps invalidate rolling windows instead of being filled."""
    frame = experiment._keys(panel.select(experiment.PANEL_COLUMNS), "market panel")
    calendar = frame.select("date").unique().sort("date").with_row_index("session")
    frame = (
        frame.join(calendar, on="date")
        .sort(["symbol", "date"])
        .with_columns(experiment._valid_bar().alias("valid"))
    )
    frame = frame.with_columns(
        pl.col("close").rolling_mean(20, min_samples=20).over("symbol").alias("ma20"),
        pl.col("valid")
        .cast(pl.Int32)
        .rolling_sum(20, min_samples=20)
        .over("symbol")
        .alias("valid20"),
        (
            pl.col("session").cast(pl.Int64)
            - pl.col("session").cast(pl.Int64).shift(19).over("symbol")
        ).alias("session_span"),
        (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("return_1d"),
        ((pl.col("volume") > 0) & (pl.col("amount") > 0)).alias("traded"),
    ).with_columns(
        (
            (pl.col("valid20") == 20)
            & (pl.col("session_span") == 19)
            & pl.col("traded")
            & pl.col("traded").shift(1).over("symbol")
        )
        .fill_null(False)
        .alias("eligible")
    )
    market = (
        frame.group_by("date")
        .agg(
            pl.len().alias("observed_stocks"),
            pl.col("eligible").sum().alias("market_stocks"),
            (pl.col("close") > pl.col("ma20"))
            .filter(pl.col("eligible"))
            .mean()
            .alias("above_ma20"),
            (pl.col("return_1d") > 0).filter(pl.col("eligible")).mean().alias("up_fraction"),
            pl.col("return_1d").filter(pl.col("eligible")).mean().alias("equal_weight_return"),
        )
        .with_columns(
            (pl.col("market_stocks") / pl.col("observed_stocks")).alias("market_coverage")
        )
    )
    enough = (
        (pl.col("market_stocks") >= config.min_market_stocks)
        & (pl.col("market_coverage") >= config.min_market_coverage)
        & experiment._finite(["above_ma20", "up_fraction", "equal_weight_return"])
    ).fill_null(False)
    weak = (pl.col("above_ma20") < 0.40) & (
        (pl.col("up_fraction") < 0.45) | (pl.col("equal_weight_return") < 0)
    )
    strong = (pl.col("above_ma20") >= 0.60) & (pl.col("up_fraction") >= 0.55)
    return market.with_columns(
        pl.when(~enough)
        .then(pl.lit("unknown"))
        .when(weak)
        .then(pl.lit("weak"))
        .when(strong)
        .then(pl.lit("strong"))
        .otherwise(pl.lit("neutral"))
        .alias("market_state")
    ).sort("date")


def fit_calibration(
    validation: pl.DataFrame,
    prediction: str,
    config: experiment.L2ExperimentConfig,
    diagnostics: DiagnosticConfig,
) -> dict:
    if (
        validation.is_empty()
        or validation.filter(
            (pl.col("split") != "validation")
            | (pl.col("date") < config.validation_start)
            | (pl.col("date") >= config.test_start)
            | (pl.col("label_end") >= config.test_start)
            | pl.col("label_end").is_null()
        ).height
    ):
        raise ValueError("Calibration accepts only purged validation labels before test_start")
    if validation.filter(~experiment._finite([prediction, "target"])).height:
        raise ValueError("Calibration requires finite predictions and labels")
    if (
        validation.group_by("date")
        .agg(pl.col("market_state").n_unique().alias("n"))
        .filter(pl.col("n") != 1)
        .height
        or validation["market_state"].null_count()
    ):
        raise ValueError("Each date must have one nonnull market state")
    daily = (
        validation.group_by("date", "market_state")
        .agg(
            pl.len().alias("rows"), (pl.col("target") - pl.col(prediction)).mean().alias("residual")
        )
        .sort("date")
    )
    fitted = daily.filter(pl.col("rows") >= config.top_k)
    enabled = fitted.height >= diagnostics.min_calibration_dates
    global_offset = float(fitted["residual"].mean()) if enabled else 0.0
    states = {}
    for state in ("weak", "neutral", "strong", "unknown"):
        part = fitted.filter(pl.col("market_state") == state)
        n = part.height
        supported = enabled and state != "unknown" and n >= diagnostics.min_state_dates
        weight = n / (n + diagnostics.state_shrink_dates) if supported else 0.0
        state_mean = float(part["residual"].mean()) if n else None
        states[state] = {
            "dates": n,
            "supported": supported,
            "shrink_weight": weight,
            "residual_mean": state_mean,
            "offset": global_offset + weight * ((state_mean or 0.0) - global_offset),
        }
    return {
        "enabled": enabled,
        "fit_split": "validation",
        "fit_dates": fitted.height,
        "excluded_thin_dates": daily.height - fitted.height,
        "fit_first_date": fitted["date"].min(),
        "fit_last_date": fitted["date"].max(),
        "max_label_end": validation["label_end"].max(),
        "global_offset": global_offset,
        "states": states,
        "status": "fitted" if enabled else "insufficient_validation_dates_no_correction",
    }


def apply_calibration(frame: pl.DataFrame, prediction: str, model: dict) -> pl.DataFrame:
    offsets = [
        model["states"].get(s, {"offset": model["global_offset"]})["offset"]
        for s in frame["market_state"].to_list()
    ]
    return frame.with_columns(
        (pl.col(prediction) + model["global_offset"]).alias("prediction_global"),
        (pl.col(prediction) + pl.Series("offset", offsets, dtype=pl.Float64)).alias(
            "prediction_state"
        ),
    )


def _summary(frame: pl.DataFrame, prediction: str, config: experiment.L2ExperimentConfig) -> dict:
    if frame.is_empty():
        return {"rows": 0, "dates": 0}
    result = experiment._metrics(
        frame, prediction, top_k=config.top_k, cost=config.round_trip_cost_bps / 10000
    )
    pred, actual = frame[prediction].to_numpy(), frame["target"].to_numpy()
    daily = frame.group_by("date").agg(
        pl.col(prediction).mean().alias("pred"), pl.col("target").mean().alias("actual")
    )
    result.update(
        {
            "mean_prediction": float(pred.mean()),
            "mean_target": float(actual.mean()),
            "bias_prediction_minus_target": float((pred - actual).mean()),
            "predicted_positive_fraction": float((pred > 0).mean()),
            "majority_direction_reference": float(max((actual > 0).mean(), (actual <= 0).mean())),
            "date_equal_bias": float((daily["pred"] - daily["actual"]).mean()),
        }
    )
    return result


def _breakdown(frame: pl.DataFrame, prediction: str, config: experiment.L2ExperimentConfig) -> dict:
    by_state = {}
    for (state,), part in frame.partition_by("market_state", as_dict=True).items():
        values = _summary(part, prediction, config)
        values.pop("daily", None)
        by_state[state] = values
    return by_state


def _feature_health(train: pl.DataFrame, parts: dict, columns: tuple[str, ...]) -> dict:
    x = train.select(columns).to_numpy()
    means, std = x.mean(axis=0), x.std(axis=0)
    safe = np.where(std < 1e-12, 1.0, std)
    z = (x - means) / safe
    correlations = z.T @ z / len(z)
    redundancy = [
        {"a": a, "b": b, "train_correlation": float(correlations[i, j])}
        for i, a in enumerate(columns)
        for j, b in enumerate(columns)
        if i < j and abs(correlations[i, j]) >= 0.90
    ]
    drift = {}
    for split, part in parts.items():
        raw = part.select(columns).to_numpy()
        transformed = (raw - means) / safe
        drift[split] = [
            {
                "feature": column,
                "train_mean": float(means[i]),
                "train_std": float(std[i]),
                "train_constant": bool(std[i] < 1e-12),
                "mean_train_z": float(transformed[:, i].mean()),
                "std_vs_train": float(raw[:, i].std() / safe[i]),
                "fraction_abs_train_z_gt_3": float((abs(transformed[:, i]) > 3).mean()),
            }
            for i, column in enumerate(columns)
        ]
    return {"redundancy_train_abs_corr_ge_0_9": redundancy, "drift": drift}


def evaluate_l2_diagnostics(
    panel: pl.DataFrame,
    l2: pl.DataFrame,
    config: experiment.L2ExperimentConfig,
    diagnostics: DiagnosticConfig | None = None,
) -> DiagnosticResult:
    diagnostics = diagnostics or DiagnosticConfig()
    if config.l2_features != experiment.L2_FEATURE_COLUMNS:
        raise ValueError("Ablation requires the complete ordered 13-feature mask")
    # Reuse the original label/split engine verbatim, including all endpoint purges.
    reference = experiment.evaluate_l2_experiment(panel, l2, config)
    common, _, _, _, _ = experiment._common_frame(panel, l2, config)
    market = build_market_history(
        panel.filter(pl.col("date").is_between(config.history_start, config.end)), diagnostics
    )
    variants = feature_variants()
    all_features = variants["full_l2"]
    frame = reference.predictions.join(
        common.select("symbol", "date", *all_features), on=["symbol", "date"], validate="m:1"
    ).join(market, on="date", validate="m:1")
    if frame.height != reference.predictions.height or frame["market_state"].null_count():
        raise ValueError("Diagnostic join changed the fixed reference mask")
    horizons, predictions = {}, []
    for horizon in config.horizons:
        current = frame.filter(pl.col("horizon") == horizon)
        parts = {
            split: current.filter(pl.col("split") == split)
            for split in ("train", "validation", "test")
        }
        train = parts["train"]
        arms, models = {}, {}
        for arm, columns in variants.items():
            weights, scaler = stockformer._fit_ridge(
                train.select(columns).to_numpy(),
                train["target"].to_numpy(),
                config.ridge_alpha,
                n_features=len(columns),
            )
            models[arm] = {"features": columns, "weights": weights.tolist(), "scaler": scaler}
            predicted = {}
            for split, part in parts.items():
                p = stockformer._predict(part.select(columns).to_numpy(), weights, scaler)
                if not np.isfinite(p).all():
                    raise ValueError(f"Nonfinite predictions: {horizon}/{arm}/{split}")
                if arm in ("baseline", "full_l2"):
                    ref = (
                        "prediction_baseline"
                        if arm == "baseline"
                        else "prediction_baseline_plus_l2"
                    )
                    if not np.array_equal(p, part[ref].to_numpy()):
                        raise ValueError("Reference arm no longer reproduces exactly")
                predicted[split] = part.with_columns(pl.Series("prediction_raw", p))
            calibration = None
            if arm in ("baseline", "full_l2"):
                calibration = fit_calibration(
                    predicted["validation"], "prediction_raw", config, diagnostics
                )
                # Validation metrics of the fitted correction are not held-out evidence.
                for split in ("validation", "test"):
                    predicted[split] = apply_calibration(
                        predicted[split], "prediction_raw", calibration
                    )
            splits = {}
            for split, part in predicted.items():
                modes = ("raw", "global", "state") if calibration and split != "train" else ("raw",)
                results = {}
                for mode in modes:
                    column = f"prediction_{mode}"
                    thick = part.filter(pl.len().over("date") >= config.top_k)
                    results[mode] = {
                        "summary": _summary(part, column, config),
                        "by_market_state": _breakdown(part, column, config),
                        "sufficient_cross_section": _summary(thick, column, config),
                        "fit_on_this_split": bool(split == "validation" and mode != "raw"),
                    }
                    predictions.append(
                        part.select(
                            "symbol",
                            "date",
                            "entry_date",
                            "label_end",
                            "horizon",
                            "split",
                            "target",
                            "market_state",
                            pl.col(column).alias("prediction"),
                        ).with_columns(pl.lit(arm).alias("arm"), pl.lit(mode).alias("calibration"))
                    )
                splits[split] = results
            arms[arm] = {"splits": splits, "calibration_model": calibration}
        horizons[str(horizon)] = {
            "arms": arms,
            "models": models,
            "feature_health": _feature_health(train, parts, all_features),
        }
    report = {
        **diagnostic_plan(diagnostics),
        "experiment_config": asdict(config),
        "coverage": reference.report["coverage"],
        "universe": reference.report["universe"],
        "reference_arms_exact": True,
        "research_only": True,
        "executable_backtest": False,
        "limitations": [
            *experiment.LIMITATIONS,
            "Calibration is fitted on validation, NOT on test; validation calibrated metrics are in-fit.",
            "Group effects are conditional on correlated features, not causal importance or significance.",
            "Market state is a stored-universe breadth proxy, not an official index or production gate.",
            "Direction accuracy must be compared with the majority-direction reference, not just 50%.",
            "Small correlated test date counts and multiple ablations prevent promotion claims.",
        ],
        "horizons": horizons,
    }
    return DiagnosticResult(
        stockformer._json_safe(report),
        pl.concat(predictions).sort(["horizon", "arm", "calibration", "date", "symbol"]),
        market,
        reference,
    )
