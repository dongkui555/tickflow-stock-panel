"""Diagnose StockFormer score calibration and sample-out distribution drift.

This command is deliberately descriptive. It does not search thresholds on the
old rolling windows. It writes one model-specific frozen signal policy so the
next rolling run can wait for new post-freeze observations.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services import stockformer as sf
from app.services.stockformer_rolling import build_rolling_windows


def _q(values: np.ndarray, probability: float) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.quantile(values, probability)) if values.size else None


def _summary(frame: pl.DataFrame, column: str) -> dict:
    if column not in frame.columns:
        return {"column": column, "rows": 0}
    values = frame[column].cast(pl.Float64, strict=False).to_numpy()
    values = values[np.isfinite(values)]
    if not values.size:
        return {"column": column, "rows": 0}
    return {
        "column": column,
        "rows": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q01": _q(values, 0.01),
        "q10": _q(values, 0.10),
        "q50": _q(values, 0.50),
        "q90": _q(values, 0.90),
        "q99": _q(values, 0.99),
    }


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return None
    a, b = left[valid], right[valid]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return None
    a = pl.Series(left[valid]).rank(method="average").to_numpy()
    b = pl.Series(right[valid]).rank(method="average").to_numpy()
    return _corr(a, b)


def _date_diagnostics(frame: pl.DataFrame, *, prediction: str, target: str) -> dict:
    rows: list[dict] = []
    if frame.is_empty() or prediction not in frame.columns or target not in frame.columns:
        return {"dates": 0, "rank_ic_mean": None, "rank_ic_median": None, "positive_rank_ic_ratio": None}
    for day, group in frame.group_by("date", maintain_order=True):
        raw_day = group[prediction].cast(pl.Float64, strict=False).to_numpy()
        target_day = group[target].cast(pl.Float64, strict=False).to_numpy()
        rank_ic = _rank_corr(raw_day, target_day)
        if rank_ic is not None:
            rows.append({"date": str(day[0]), "rank_ic": rank_ic, "rows": int(group.height)})
    values = np.asarray([item["rank_ic"] for item in rows], dtype=float)
    return {
        "dates": len(rows),
        "rank_ic_mean": float(np.mean(values)) if values.size else None,
        "rank_ic_median": float(np.median(values)) if values.size else None,
        "positive_rank_ic_ratio": float(np.mean(values > 0.0)) if values.size else None,
        "daily": rows,
    }


def _tail_evidence(frame: pl.DataFrame, *, raw: str, calibrated: str, target: str, floor: float) -> dict:
    result = {
        "raw_floor_coverage": 0.0,
        "calibrated_floor_coverage": 0.0,
        "raw_q90_actual_median": None,
        "calibrated_q90_actual_median": None,
    }
    if frame.is_empty():
        return result
    raw_values = frame[raw].cast(pl.Float64, strict=False).to_numpy()
    calibrated_values = frame[calibrated].cast(pl.Float64, strict=False).to_numpy()
    result["raw_floor_coverage"] = float(
        np.mean(np.isfinite(raw_values) & (raw_values > floor))
    )
    result["calibrated_floor_coverage"] = float(
        np.mean(np.isfinite(calibrated_values) & (calibrated_values > floor))
    )
    raw_q90 = _q(raw_values, 0.90)
    calibrated_q90 = _q(calibrated_values, 0.90)
    if raw_q90 is not None:
        tail = frame.filter(pl.col(raw) >= raw_q90)
        result["raw_q90_actual_median"] = float(tail[target].median()) if tail.height else None
    if calibrated_q90 is not None:
        tail = frame.filter(pl.col(calibrated) >= calibrated_q90)
        result["calibrated_q90_actual_median"] = float(tail[target].median()) if tail.height else None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--window-days", type=int, default=63)
    parser.add_argument("--report-name", default=None)
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    meta = sf._model_meta(data_dir, args.model_id)
    if meta is None:
        print(json.dumps({"error": "model not found"}, ensure_ascii=False))
        return 1
    model_id = str(meta.get("model_id") or args.model_id or "")
    factor_path = sf._resolve_artifact_path(data_dir, meta.get("factor_file"))
    if factor_path is None or not factor_path.exists():
        print(json.dumps({"error": "factor file not found"}, ensure_ascii=False))
        return 1
    factors = pl.read_parquet(factor_path)
    raw_col, calibrated_col, target_col = "pred_ret_raw_5d", "pred_ret_5d", "target_5d"
    if any(column not in factors.columns for column in (raw_col, calibrated_col, target_col)):
        print(json.dumps({"error": "5-day raw/calibrated/target columns are required"}, ensure_ascii=False))
        return 1

    calibration = (meta.get("prediction_calibration") or {}).get("5") or {}
    floor = 0.0
    factor_dates = sorted(factors["date"].unique().to_list())
    windows = build_rolling_windows(factor_dates, window_days=args.window_days, min_window_days=args.window_days)
    complete_windows = [item for item in windows if item.get("complete")]
    window_reports = []
    for item in complete_windows:
        sample = factors.filter(
            (pl.col("date") >= item["start"]) & (pl.col("date") <= item["end"])
        )
        window_reports.append({
            **item,
            "start": str(item["start"]),
            "end": str(item["end"]),
            "raw": _summary(sample, raw_col),
            "calibrated": _summary(sample, calibrated_col),
            "target": _summary(sample, target_col),
            "rank_ic": _date_diagnostics(sample, prediction=raw_col, target=target_col),
            "tail": _tail_evidence(sample, raw=raw_col, calibrated=calibrated_col, target=target_col, floor=floor),
        })

    diagnostics = {
        "model_id": model_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(factor_path),
        "target_definition": meta.get("target_definition"),
        "target_mode": meta.get("target_mode"),
        "calibration": calibration,
        "all_oos": {
            "start": str(min(factor_dates)) if factor_dates else None,
            "end": str(max(factor_dates)) if factor_dates else None,
            "trading_days": len(factor_dates),
            "raw": _summary(factors, raw_col),
            "calibrated": _summary(factors, calibrated_col),
            "target": _summary(factors, target_col),
            "rank_ic": _date_diagnostics(factors, prediction=raw_col, target=target_col),
            "tail": _tail_evidence(factors, raw=raw_col, calibrated=calibrated_col, target=target_col, floor=floor),
        },
        "complete_windows": window_reports,
        "interpretation": {
            "high_tail_calibrated_below_zero": bool(
                calibration.get("high_tail_actual_median", 0.0) < 0.0
            ),
            "calibration_can_reverse_extreme_order": bool(
                calibration.get("high_slope", 1.0) <= 0.0
                or calibration.get("low_slope", 1.0) <= 0.0
            ),
            "fixed_004_coverage_is_sparse": bool(
                (_summary(factors, calibrated_col).get("q90") or 0.0) < 0.004
            ),
        },
    }

    effective_from = None
    for key in ("live_refit_through", "oos_end", "trained_at"):
        value = meta.get(key)
        if value:
            try:
                effective_from = date.fromisoformat(str(value)[:10]) + timedelta(days=1)
                break
            except ValueError:
                continue
    if effective_from is None:
        effective_from = (max(factor_dates) + timedelta(days=1)) if factor_dates else date.today()
    policy = {
        "version": "stockformer_signal_policy_v1",
        "model_id": model_id,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "effective_from": effective_from.isoformat(),
        "status": "research_only",
        "prediction_rule": {
            "kind": "calibrated_net_margin_cross_sectional_rank",
            "calibrated_floor": 0.0,
            "ranking_column": "net_margin_score",
            "min_net_edge": 0.0005,
            "top_quantile": 0.20,
            "rank_scope": "date",
            "rank_direction": "descending",
        },
        "calibration_source": "model.prediction_calibration.5",
        "diagnostic_report": f"stockformer/experiments/calibration_drift_{model_id}.json",
        "freeze_reason": "rank calibrated net margin after estimated costs; reject non-positive extreme tails",
    }
    diagnostics["frozen_policy"] = policy

    experiments = data_dir / "stockformer" / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    report_path = experiments / (args.report_name or f"calibration_drift_{model_id}.json")
    report_path.write_text(json.dumps(sf._json_safe(diagnostics), ensure_ascii=False, indent=2), encoding="utf-8")

    model_dir = data_dir / "stockformer" / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    policy_path = model_dir / "frozen_signal_policy.json"
    policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    rel_policy = sf._store_relpath(data_dir, policy_path)
    for metadata_path in (model_dir / "model.json", data_dir / "stockformer" / "models.json"):
        if not metadata_path.exists():
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else [payload]
        changed = False
        for item in items:
            if isinstance(item, dict) and item.get("model_id") == model_id:
                item["frozen_signal_policy_file"] = rel_policy
                item["frozen_signal_policy_version"] = policy["version"]
                item["promotion_status"] = "research_only"
                changed = True
        if changed:
            metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "model_id": model_id,
        "report": str(report_path),
        "policy": str(policy_path),
        "effective_from": policy["effective_from"],
        "complete_old_windows": len(complete_windows),
        "status": "frozen_research_only",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
