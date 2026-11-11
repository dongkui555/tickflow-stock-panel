"""Leakage-safe L2 ablation, regime calibration and research-model freezing.

Candidate selection may inspect train and validation metrics only.  The test
split is evaluated after the candidate and regime policy have been frozen.
Nothing in this module promotes a production StockFormer model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from app.services import l2_experiment as experiment
from app.services import stockformer

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "all_l2": experiment.L2_FEATURE_COLUMNS,
    "trade_activity": (
        "l2_trade_count",
        "l2_trade_volume",
        "l2_trade_amount",
        "l2_average_trade_amount",
    ),
    "aggressor_flow_proxy": (
        "l2_buy_amount_ratio",
        "l2_net_buy_amount_ratio",
        "l2_large_trade_amount_ratio",
    ),
    "late_session_flow": (
        "l2_late_net_buy_ratio",
        "l2_late_amount_ratio",
    ),
    "vwap_gap": ("l2_vwap_close_gap",),
    "order_book_depth": (
        "l2_depth_imbalance_mean",
        "l2_depth_imbalance_positive_fraction",
    ),
    "spread": ("l2_spread_bps_mean",),
}

UNAVAILABLE_GROUPS = {
    "orders": "The audited archive has quote snapshots and trades, but no cross-market order event stream.",
    "cancellations": "SZ cancellation flags are not equivalent to a consistent SH/SZ cancellation feed and are excluded.",
}

FOCUS_HORIZONS = (5, 10)
REGIMES = ("rising", "sideways", "falling")


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _sign(value: float | None, tolerance: float = 1e-12) -> int:
    if value is None or not np.isfinite(value) or abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def summarize_candidate(report: dict, *, focus_horizons=FOCUS_HORIZONS) -> dict:
    """Return train/validation-only selection evidence.

    This deliberately has no code path that reads the test split.  Keeping the
    selector small and separately tested makes accidental holdout tuning visible.
    """
    evidence = []
    for horizon in focus_horizons:
        item = report["horizons"][str(horizon)]["splits"]
        train = item["train"]
        validation = item["validation"]
        row = {"horizon": horizon}
        for split_name, split in (("train", train), ("validation", validation)):
            augmented = split["baseline_plus_l2"]
            contrast = split["plus_l2_minus_baseline"]
            row.update(
                {
                    f"{split_name}_rank_ic": augmented["rank_ic"],
                    f"{split_name}_after_cost": augmented["top_k_after_cost_proxy"],
                    f"{split_name}_rank_ic_gain": contrast["rank_ic"],
                    f"{split_name}_top_k_gain": contrast["top_k_forward_return"],
                }
            )
        evidence.append(row)

    train_rank_gain = _mean([row["train_rank_ic_gain"] for row in evidence])
    validation_rank_gain = _mean([row["validation_rank_ic_gain"] for row in evidence])
    train_top_gain = _mean([row["train_top_k_gain"] for row in evidence])
    validation_top_gain = _mean([row["validation_top_k_gain"] for row in evidence])
    validation_rank_ic = _mean([row["validation_rank_ic"] for row in evidence])
    validation_after_cost = _mean([row["validation_after_cost"] for row in evidence])

    reversals = []
    for row in evidence:
        for metric in ("rank_ic_gain", "top_k_gain"):
            before = row[f"train_{metric}"]
            after = row[f"validation_{metric}"]
            if _sign(before) > 0 and _sign(after) < 0:
                reversals.append({"horizon": row["horizon"], "metric": metric})

    # A feature is retained only when it adds both rank information and head
    # return in train and validation.  This conservative gate intentionally
    # prefers the OHLCV baseline over a fragile L2 improvement.
    stable = bool(
        train_rank_gain is not None
        and validation_rank_gain is not None
        and train_top_gain is not None
        and validation_top_gain is not None
        and train_rank_gain > 0
        and validation_rank_gain > 0
        and train_top_gain > 0
        and validation_top_gain > 0
        and not reversals
    )
    objective = None
    if (
        validation_after_cost is not None
        and validation_top_gain is not None
        and validation_rank_ic is not None
    ):
        objective = validation_after_cost + 0.5 * validation_top_gain + 0.01 * validation_rank_ic
    return stockformer._json_safe(
        {
            "focus_horizons": list(focus_horizons),
            "evidence": evidence,
            "train_rank_ic_gain": train_rank_gain,
            "validation_rank_ic_gain": validation_rank_gain,
            "train_top_k_gain": train_top_gain,
            "validation_top_k_gain": validation_top_gain,
            "validation_rank_ic": validation_rank_ic,
            "validation_after_cost": validation_after_cost,
            "train_positive_validation_negative": reversals,
            "stable": stable,
            "selection_objective": objective,
        }
    )


def select_candidate(summaries: dict[str, dict]) -> str:
    """Choose using only summaries produced by :func:`summarize_candidate`."""
    stable = [
        (name, values.get("selection_objective"))
        for name, values in summaries.items()
        if values.get("stable") and values.get("selection_objective") is not None
    ]
    if not stable:
        return "baseline_only"
    stable.sort(key=lambda item: (-item[1], item[0]))
    return stable[0][0]


def build_market_regimes(
    panel: pl.DataFrame,
    symbols: list[str],
    *,
    start: date,
    end: date,
    lookback: int = 20,
    threshold: float = 0.02,
) -> pl.DataFrame:
    """Causal equal-weight market regime known at each information date."""
    clean = (
        experiment._keys(panel.select(experiment.PANEL_COLUMNS), "OHLCV regime")
        .filter(pl.col("symbol").is_in(symbols) & pl.col("date").is_between(start, end))
        .filter(experiment._valid_bar())
        .sort(["symbol", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret_1d")
        )
        .filter(pl.col("ret_1d").is_not_null() & pl.col("ret_1d").is_finite())
    )
    daily = (
        clean.group_by("date")
        .agg(
            pl.col("ret_1d").mean().alias("market_return_1d"),
            (pl.col("ret_1d") > 0).mean().alias("advance_fraction"),
            pl.col("symbol").n_unique().alias("market_symbols"),
        )
        .sort("date")
        .with_columns((pl.col("market_return_1d") + 1).cum_prod().alias("market_index"))
        .with_columns(
            (pl.col("market_index") / pl.col("market_index").shift(lookback) - 1).alias(
                "market_return_20d"
            )
        )
        .with_columns(
            pl.when(pl.col("market_return_20d").is_null())
            .then(pl.lit("unknown"))
            .when(pl.col("market_return_20d") >= threshold)
            .then(pl.lit("rising"))
            .when(pl.col("market_return_20d") <= -threshold)
            .then(pl.lit("falling"))
            .otherwise(pl.lit("sideways"))
            .alias("market_regime")
        )
    )
    return daily


def _metric_for_prediction(
    frame: pl.DataFrame,
    prediction: str,
    *,
    top_k: int,
    cost: float,
) -> dict | None:
    if frame.is_empty() or frame["date"].n_unique() < 3:
        return None
    return experiment._metrics(frame, prediction, top_k=top_k, cost=cost)


def calibrate_regimes(
    predictions: pl.DataFrame,
    regimes: pl.DataFrame,
    config: experiment.L2ExperimentConfig,
    *,
    candidate_name: str,
) -> tuple[dict, pl.DataFrame]:
    """Freeze validation-selected blends, then apply them to all splits."""
    joined = predictions.join(
        regimes.select("date", "market_regime", "market_return_20d", "advance_fraction"),
        on="date",
        how="left",
    ).with_columns(pl.col("market_regime").fill_null("unknown"))
    blend_grid = (0.0,) if candidate_name == "baseline_only" else (0.0, 0.25, 0.5, 0.75, 1.0)
    cost = config.round_trip_cost_bps / 10000.0
    policy: dict[str, dict] = {}
    output = []

    for horizon in config.horizons:
        horizon_rows = joined.filter(pl.col("horizon") == horizon)
        policy[str(horizon)] = {}
        blend_expr = pl.lit(0.0)
        for regime in REGIMES:
            validation = horizon_rows.filter(
                (pl.col("split") == "validation") & (pl.col("market_regime") == regime)
            )
            choices = []
            for blend in blend_grid:
                candidate = validation.with_columns(
                    (
                        pl.col("prediction_baseline") * (1 - blend)
                        + pl.col("prediction_baseline_plus_l2") * blend
                    ).alias("prediction_blend")
                )
                metrics = _metric_for_prediction(
                    candidate,
                    "prediction_blend",
                    top_k=config.top_k,
                    cost=cost,
                )
                objective = None
                if metrics is not None and metrics["top_k_after_cost_proxy"] is not None:
                    objective = metrics["top_k_after_cost_proxy"] + 0.01 * (
                        metrics["rank_ic"] or 0.0
                    )
                choices.append({"blend": blend, "objective": objective, "metrics": metrics})

            usable = [choice for choice in choices if choice["objective"] is not None]
            best = (
                max(usable, key=lambda choice: (choice["objective"], -choice["blend"]))
                if usable
                else choices[0]
            )
            baseline_objective = next(
                (choice["objective"] for choice in choices if choice["blend"] == 0.0), None
            )
            # Demand at least 5bp validation improvement before enabling L2 in
            # a regime.  Small gains are treated as estimation noise.
            if best["blend"] > 0 and (
                baseline_objective is None
                or best["objective"] is None
                or best["objective"] < baseline_objective + 0.0005
            ):
                best = next(choice for choice in choices if choice["blend"] == 0.0)

            net = best["metrics"]["top_k_after_cost_proxy"] if best["metrics"] else None
            excess = best["metrics"]["top_k_excess"] if best["metrics"] else None
            exposure = (
                0.0
                if net is None or net <= 0
                else (1.0 if excess is not None and excess > 0 else 0.5)
            )
            policy[str(horizon)][regime] = stockformer._json_safe(
                {
                    "blend": best["blend"],
                    "validation_dates": validation["date"].n_unique(),
                    "validation_objective": best["objective"],
                    "validation_metrics": best["metrics"],
                    "exposure_multiplier": exposure,
                }
            )
            blend_expr = (
                pl.when(pl.col("market_regime") == regime)
                .then(pl.lit(float(best["blend"])))
                .otherwise(blend_expr)
            )

        scored = horizon_rows.with_columns(blend_expr.alias("l2_blend")).with_columns(
            (
                pl.col("prediction_baseline") * (1 - pl.col("l2_blend"))
                + pl.col("prediction_baseline_plus_l2") * pl.col("l2_blend")
            ).alias("prediction_frozen")
        )
        output.append(scored)

    scored = pl.concat(output).sort(["horizon", "date", "symbol"])
    metrics: dict[str, dict] = {}
    for horizon in config.horizons:
        metrics[str(horizon)] = {}
        for split in ("train", "validation", "test"):
            metrics[str(horizon)][split] = {}
            for regime in (*REGIMES, "all"):
                rows = scored.filter(
                    (pl.col("horizon") == horizon)
                    & (pl.col("split") == split)
                    & (pl.lit(True) if regime == "all" else pl.col("market_regime") == regime)
                )
                metrics[str(horizon)][split][regime] = _metric_for_prediction(
                    rows,
                    "prediction_frozen",
                    top_k=config.top_k,
                    cost=cost,
                )
    return stockformer._json_safe({"policy": policy, "metrics": metrics}), scored


def _baseline_summary(report: dict) -> dict:
    rows = []
    for horizon in FOCUS_HORIZONS:
        split = report["horizons"][str(horizon)]["splits"]["validation"]["baseline"]
        rows.append(split)
    return {
        "validation_rank_ic": _mean([row["rank_ic"] for row in rows]),
        "validation_after_cost": _mean([row["top_k_after_cost_proxy"] for row in rows]),
        "selection_objective": _mean(
            [
                row["top_k_after_cost_proxy"] + 0.01 * (row["rank_ic"] or 0.0)
                for row in rows
                if row["top_k_after_cost_proxy"] is not None
            ]
        ),
        "stable": True,
        "selection_role": "fallback; no L2 features",
    }


def _freeze_payload(
    *,
    config: experiment.L2ExperimentConfig,
    candidate_name: str,
    candidate_result: experiment.L2ExperimentResult,
    summaries: dict[str, dict],
    regime_report: dict,
) -> dict:
    features = (
        ()
        if candidate_name == "baseline_only"
        else tuple(candidate_result.report["features"]["baseline_plus_l2"])[
            len(experiment.BASELINE_FEATURE_COLUMNS) :
        ]
    )
    models = {}
    for horizon in config.horizons:
        arm = "baseline" if candidate_name == "baseline_only" else "baseline_plus_l2"
        models[str(horizon)] = candidate_result.report["horizons"][str(horizon)]["models"][arm]
    payload = {
        "schema_version": 1,
        "status": "research_frozen",
        "production_eligible": False,
        "promotion_blocker": "Requires a genuinely unseen, sufficiently continuous 2026 L2 holdout.",
        "selection_data": ["train", "validation"],
        "test_used_for_selection": False,
        "reused_2025_test_is_final_evidence": False,
        "config": asdict(config),
        "universe": candidate_result.report["universe"],
        "candidate": candidate_name,
        "baseline_features": list(experiment.BASELINE_FEATURE_COLUMNS),
        "selected_l2_features": list(features),
        "unavailable_feature_groups": UNAVAILABLE_GROUPS,
        "candidate_summaries": summaries,
        "market_regime": {
            "method": "causal 20-session cumulative equal-weight return of the frozen universe",
            "rising_threshold": 0.02,
            "falling_threshold": -0.02,
            **regime_report,
        },
        "models": models,
    }
    canonical = json.dumps(stockformer._json_safe(payload), sort_keys=True, separators=(",", ":"))
    payload["freeze_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return stockformer._json_safe(payload)


def run_model_selection(
    panel: pl.DataFrame,
    l2: pl.DataFrame,
    config: experiment.L2ExperimentConfig,
) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """Run preset groups and singleton ablations, then freeze one candidate."""
    results: dict[str, experiment.L2ExperimentResult] = {}
    summaries: dict[str, dict] = {}
    candidate_features = dict(FEATURE_GROUPS)
    candidate_features.update(
        {f"single__{name}": (name,) for name in experiment.L2_FEATURE_COLUMNS}
    )

    for name, features in candidate_features.items():
        result = experiment.evaluate_l2_experiment(panel, l2, replace(config, l2_features=features))
        results[name] = result
        summaries[name] = {"features": list(features), **summarize_candidate(result.report)}

    stable_singletons = [
        name.removeprefix("single__")
        for name, values in summaries.items()
        if name.startswith("single__") and values["stable"]
    ]
    if stable_singletons:
        stable_result = experiment.evaluate_l2_experiment(
            panel,
            l2,
            replace(config, l2_features=tuple(stable_singletons)),
        )
        results["stable_singleton_union"] = stable_result
        summaries["stable_singleton_union"] = {
            "features": stable_singletons,
            **summarize_candidate(stable_result.report),
        }

    summaries["baseline_only"] = {
        "features": [],
        **_baseline_summary(results["all_l2"].report),
    }
    selected = select_candidate(
        {name: values for name, values in summaries.items() if name != "baseline_only"}
    )
    candidate_result = results["all_l2"] if selected == "baseline_only" else results[selected]
    symbols = candidate_result.report["universe"]["symbols"]
    regimes = build_market_regimes(
        panel,
        symbols,
        start=config.history_start,
        end=config.end,
    )
    regime_report, scored = calibrate_regimes(
        candidate_result.predictions,
        regimes,
        config,
        candidate_name=selected,
    )
    frozen = _freeze_payload(
        config=config,
        candidate_name=selected,
        candidate_result=candidate_result,
        summaries=summaries,
        regime_report=regime_report,
    )
    return frozen, scored, regimes


def audit_2026_archives(
    archive_root: Path,
    market_dates: list[date],
    *,
    start: date,
    end: date,
) -> dict:
    """Describe fresh-holdout readiness without reading multi-GB archives."""
    by_day: dict[date, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in sorted(archive_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".7z", ".zip"}:
            continue
        digits = "".join(character for character in path.stem if character.isdigit())[:8]
        if len(digits) != 8:
            continue
        try:
            day = date.fromisoformat(f"{digits[:4]}-{digits[4:6]}-{digits[6:]}")
        except ValueError:
            continue
        if not start <= day <= end:
            continue
        if day in by_day:
            duplicates.setdefault(str(day), [str(by_day[day])]).append(str(path))
            continue
        by_day[day] = path

    calendar = sorted(day for day in set(market_dates) if start <= day <= end)
    available = sorted(set(by_day) & set(calendar))
    missing = sorted(set(calendar) - set(available))
    off_calendar = sorted(set(by_day) - set(calendar))
    position = {day: index for index, day in enumerate(calendar)}
    streaks, current = [], []
    for day in available:
        if current and position[day] != position[current[-1]] + 1:
            streaks.append(current)
            current = []
        current.append(day)
    if current:
        streaks.append(current)
    longest = max(streaks, key=len) if streaks else []
    return {
        "schema_version": 1,
        "archive_root": str(archive_root.resolve()),
        "requested_start": str(start),
        "requested_end": str(end),
        "market_days": len(calendar),
        "archive_market_days": len(available),
        "coverage": len(available) / len(calendar) if calendar else 0.0,
        "ready_for_fresh_oos": len(longest) >= 60,
        "minimum_continuous_days": 60,
        "preferred_continuous_days": 100,
        "longest_continuous_market_day_streak": len(longest),
        "longest_streak_start": str(longest[0]) if longest else None,
        "longest_streak_end": str(longest[-1]) if longest else None,
        "available_dates": [str(day) for day in available],
        "missing_dates": [str(day) for day in missing],
        "off_calendar_archive_dates": [str(day) for day in off_calendar],
        "duplicate_archives": duplicates,
        "archive_bytes": sum(path.stat().st_size for path in by_day.values()),
    }
