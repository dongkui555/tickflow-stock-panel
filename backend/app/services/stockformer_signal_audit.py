"""Read-only attribution of five-day forecasts on previously inspected OOS data."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_rolling as rolling
from app.tickflow.repository import DataStore, KlineRepository


PRED_COL = "pred_ret_5d"
TARGET_COL = "target_5d_net"
SCORE_BANDS = (
    ("negative", None, 0.0),
    ("0-0.4%", 0.0, 0.004),
    ("0.4-1%", 0.004, 0.01),
    ("1-2%", 0.01, 0.02),
    ("2-3%", 0.02, 0.03),
    ("3-5%", 0.03, 0.05),
    (">=5%", 0.05, None),
)


def _rank_ic(frame: pl.DataFrame) -> dict:
    values = []
    for part in frame.partition_by("date"):
        sample = part.select(PRED_COL, TARGET_COL).drop_nulls()
        if sample.height < 5:
            continue
        prediction = sample[PRED_COL].rank().to_numpy()
        actual = sample[TARGET_COL].rank().to_numpy()
        if np.std(prediction) and np.std(actual):
            values.append(float(np.corrcoef(prediction, actual)[0, 1]))
    return {
        "dates": len(values),
        "mean_rank_ic": round(float(np.mean(values)), 6) if values else None,
        "positive_date_fraction": round(float(np.mean(np.asarray(values) > 0)), 6)
        if values else None,
    }


def _outcome(frame: pl.DataFrame) -> dict:
    known = frame.filter(pl.col(TARGET_COL).is_finite())
    return {
        "rows": known.height,
        "dates": known["date"].n_unique() if not known.is_empty() else 0,
        "mean_predicted_net_pct": round(float(known[PRED_COL].mean()) * 100, 4)
        if not known.is_empty() else None,
        "mean_realized_net_pct": round(float(known[TARGET_COL].mean()) * 100, 4)
        if not known.is_empty() else None,
        "positive_realized_fraction": round(float((known[TARGET_COL] > 0).mean()), 4)
        if not known.is_empty() else None,
    }


def _score_bands(frame: pl.DataFrame) -> list[dict]:
    rows = []
    for label, lower, upper in SCORE_BANDS:
        expr = pl.lit(True)
        if lower is not None:
            expr = expr & (pl.col(PRED_COL) >= lower)
        if upper is not None:
            expr = expr & (pl.col(PRED_COL) < upper)
        rows.append({"band": label, **_outcome(frame.filter(expr))})
    return rows


def _top_three(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return (
        frame.sort(["date", PRED_COL, "symbol"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("date").alias("_rank"))
        .filter(pl.col("_rank") < 3)
    )


def _family_summary(frame: pl.DataFrame, forecast_floor: float) -> dict:
    known = frame.filter(pl.col(TARGET_COL).is_finite() & pl.col(PRED_COL).is_finite())
    pred_pass = pl.col(PRED_COL) > forecast_floor
    entry_pass = pl.col("_entry_rules_pass")
    market_pass = pl.col("_market_gate_pass")
    predicted = known.filter(pred_pass)
    after_entry = predicted.filter(entry_pass)
    after_market = after_entry.filter(market_pass)
    regimes = []
    for regime in sorted(str(value) for value in known["market_gate"].unique().to_list() if value):
        part = known.filter(pl.col("market_gate") == regime)
        eligible = part.filter(pred_pass & entry_pass & market_pass)
        regimes.append({
            "regime": regime,
            "all": {**_outcome(part), **_rank_ic(part)},
            "eligible": _outcome(eligible),
            "top_three_eligible": _outcome(_top_three(eligible)),
        })
    return {
        "all": {**_outcome(known), **_rank_ic(known)},
        "forecast_pass": _outcome(predicted),
        "entry_pass_after_forecast": _outcome(after_entry),
        "market_pass_after_entry": _outcome(after_market),
        "top_three_after_entry_and_market": _outcome(_top_three(after_market)),
        "entry_rejected_after_forecast": _outcome(predicted.filter(~entry_pass)),
        "market_rejected_after_entry": _outcome(after_entry.filter(~market_pass)),
        "score_bands": _score_bands(known),
        "regimes": regimes,
    }


def audit_signal_decay(data_dir: Path, model_id: str, output_path: Path, on_progress=None) -> dict:
    meta = sf._model_meta(data_dir, model_id)
    if meta is None:
        raise ValueError("model metadata not found")
    factor_path = sf._resolve_artifact_path(data_dir, meta.get("factor_file"))
    if factor_path is None or not factor_path.exists():
        raise ValueError("model OOS factor file not found")
    factor = pl.read_parquet(factor_path).select("symbol", "date").unique()
    factor_dates = sorted(factor["date"].unique().to_list())
    windows = rolling.build_rolling_windows(factor_dates)
    if len(windows) < 3:
        raise ValueError("fewer than three complete OOS windows")
    start, end = factor_dates[0], factor_dates[-1]
    data_start, _ = sf._date_range_from_data(data_dir)
    history_start = max(data_start or start - timedelta(days=730), start - timedelta(days=730))
    panel = sf._load_panel(
        data_dir, history_start, end + timedelta(days=10),
        symbols=[str(value) for value in factor["symbol"].unique().to_list()],
        include_minute=str(meta.get("training_profile")) == "minute_enhanced",
    )
    features, coverage = portfolio._prepare_research_features(
        panel, str(meta.get("training_profile") or "daily_baseline")
    )
    available = [column for column in sf.training_feature_columns(meta["training_profile"])
                 if column in features.columns]
    selected = ((meta.get("validation") or {}).get("horizons") or {}).get("5d") or {}
    alpha = float(selected.get("alpha") or 1.0)
    full = rolling._rolling_feature_columns("full", available)
    families = {
        "full": full,
        "full_without_minute": [column for column in full if column not in sf.MINUTE_FEATURE_COLUMNS],
        "compact": rolling._rolling_feature_columns("compact", available),
    }
    floor = 0.004
    candidate_validation = next(
        (
            item.get("validation") or {}
            for item in selected.get("candidates") or []
            if item.get("family") == selected.get("family")
            and float(item.get("alpha") or 0.0) == alpha
        ),
        {},
    )
    results = {}
    for name, columns in families.items():
        frames = []
        for window in windows[:3]:
            prediction = portfolio._fit_candidate_predictions(
                features, horizon=5, columns=columns, alpha=alpha,
                fit_end=window["start"], predict_start=window["start"],
                predict_end=window["end"] + timedelta(days=1),
                target_mode=str(meta.get("target_mode") or "absolute"),
            )
            prediction = prediction.join(factor, on=["symbol", "date"], how="inner")
            signal = portfolio._build_gated_signal_frame(
                data_dir, prediction, features, meta, horizon=5,
                forecast_floor=floor,
            )
            if not signal.is_empty():
                signal = signal.join(
                    features.select("symbol", "date", TARGET_COL),
                    on=["symbol", "date"], how="left",
                )
                frames.append(signal)
        combined = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
        results[name] = {
            "feature_count": len(columns),
            "minute_feature_count": sum(column in sf.MINUTE_FEATURE_COLUMNS for column in columns),
            "summary": _family_summary(combined, floor) if not combined.is_empty() else {},
        }
        if on_progress:
            on_progress(name)
    report = sf._json_safe({
        "model_id": model_id,
        "purpose": "diagnosis_only_previously_inspected_oos_not_for_selection",
        "sample_out": {"start": str(windows[0]["start"]), "end": str(windows[2]["end"])},
        "training_rank_ic": candidate_validation.get("rank_ic"),
        "training_validation_positive_rank_ic_folds": candidate_validation.get("positive_rank_ic_folds"),
        "coverage": coverage,
        "forecast_floor": floor,
        "families": results,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
