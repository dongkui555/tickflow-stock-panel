"""Frozen close forecasts. Historical replay never enters prospective accuracy."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from statistics import mean

from app.market_time import cn_now
from app.services import sector_layout as rotation
from app.services.sector_forecast_rules import (
    FEATURE_VERSION,
    MODEL_VERSION,
    evaluate_outcome,
    number,
    rank_forecasts,
)
from app.services.sector_forecast_store import ForecastStore, encode


def trading_dates(data_dir, end=None):
    return sorted(
        set(
            day
            for table in ("kline_index_daily", "kline_daily", "kline_daily_enriched")
            for day in rotation._partition_dates(data_dir / table, end=end)
        )
    )


def enrich_context(rows):
    from app.services.sector_prediction import _market_context

    market = _market_context(rows)
    parents = {}
    for row in rows:
        parents.setdefault(row["sector"].split("-")[0], []).append(row)
    for row in rows:
        siblings = [
            r
            for r in parents[row["sector"].split("-")[0]]
            if r["sector"] != row["sector"]
            and number(r.get("flow_coverage")) >= 0.98
            and r.get("flow_acceleration") is not None
        ]
        row["parent_flow_acceleration"] = (
            mean(number(r.get("flow_acceleration")) for r in siblings) if siblings else 0.0
        )
        row["price_resilient"] = number(row.get("resilient_ratio")) >= 0.65
        row["leader_strength"] = max(
            (number(r.get("ret_1d")) for r in row.get("leader_stocks", [])), default=0.0
        )
        row["sector_data_ready"] = (
            number(row.get("limit_coverage")) >= 0.98
            and number(row.get("price_coverage")) >= 0.98
            and row.get("flow_zscore") is not None
            and row.get("prior_flow_intensity") is not None
        )
    return rows, market


def build(data_dir: Path, *, as_of=None, lookback_days=70, limit=30, persist=True, origin=None):
    now = cn_now()
    target = as_of or now.date()
    origin = origin or ("published" if target == now.date() else "replay")
    if origin not in {"published", "replay"}:
        raise ValueError("Invalid forecast origin")
    blocked = {
        "status": "waiting_data",
        "as_of": str(target),
        "model_version": MODEL_VERSION,
        "origin": origin,
        "rows": [],
        "summary": {},
    }
    if target > now.date() or (target == now.date() and now.hour < 19):
        blocked["summary"]["warning"] = "等待当日19点后收盘数据校验"
        return blocked
    if origin == "published" and target != now.date():
        raise ValueError("Past forecasts must be marked as replay")
    if origin == "published":
        stored = next(
            (
                s
                for s in ForecastStore(data_dir).list(origin)
                if s["as_of"] == str(target) and s["model_version"] == MODEL_VERSION
            ),
            None,
        )
        if stored:
            return merge_validations(data_dir, stored)
    from app.services.prediction_workflow import readiness

    ready = readiness(data_dir, target)
    if not ready["ready"]:
        blocked.update(readiness=ready)
        blocked["summary"]["warning"] = ready["reason"]
        return blocked
    dates = trading_dates(data_dir, target)[-max(25, min(160, lookback_days)) :]
    before = rotation.input_signature(data_dir, dates)
    payload = rotation.build_sector_rotation(
        data_dir, as_of=target, lookback_days=lookback_days, limit=100
    )
    if payload.get("as_of") != str(target):
        blocked["summary"]["warning"] = "日K未覆盖目标日期"
        return blocked
    rows, market = enrich_context(payload.get("rows") or [])
    rows = rank_forecasts(rows, market)
    after = rotation.input_signature(data_dir, dates)
    if before != after:
        blocked["summary"]["warning"] = "计算期间数据发生变化, 等待重试"
        return blocked
    if not rows or all(r["prediction_level"] == "unavailable" for r in rows):
        blocked["summary"]["warning"] = "板块资金覆盖、涨停口径或有效历史窗口不足"
        return blocked
    digest = hashlib.sha256(encode(before).encode()).hexdigest()
    identity = f"{target}_{MODEL_VERSION}_{origin}"
    if origin == "replay":
        identity += "_" + digest[:16]
    snapshot = {
        "id": identity,
        "as_of": str(target),
        "origin": origin,
        "generated_at": now.isoformat(),
        "data_cutoff": f"{target}T15:00:00+08:00",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "input_version": digest,
        "input_manifest": before,
        "status": "ready",
        "readiness": ready,
        "rows": rows,
        "lookback_days": lookback_days,
        "limit": limit,
        "source_meta": {
            **payload.get("meta", {}),
            "prediction_market_context": market,
            "industry_membership": "current_map_not_point_in_time",
            "fund_definition": "large_plus_extra_large_net_yuan",
        },
        "summary": {
            "total_candidates": len(rows),
            "stored_candidates": len(rows),
            "strong_count": sum(r["prediction_level"] == "strong" for r in rows),
            "watch_count": sum(r["prediction_level"] == "watch" for r in rows),
        },
    }
    if persist:
        snapshot = ForecastStore(data_dir).save(snapshot)
    return snapshot


def validate(data_dir: Path, *, recheck_recent_days=8):
    store = ForecastStore(data_dir)
    snapshots = store.list()
    if not snapshots:
        return
    now = cn_now()
    dates = trading_dates(data_dir, now.date())
    if now.hour < 19:
        dates = [d for d in dates if d < now.date()]
    actual_cache = {}
    from app.services.prediction_workflow import readiness

    for snapshot in snapshots:
        existing = {event["horizon_days"]: event for event in store.validations(snapshot["id"])}
        target = date.fromisoformat(snapshot["as_of"])
        if target not in dates:
            continue
        idx = dates.index(target)
        for h in (1, 2):
            if idx + h >= len(dates):
                continue
            day = dates[idx + h]
            prior = existing.get(h)
            if (
                prior
                and day not in dates[-recheck_recent_days:]
                and not any(r.get("status") == "missing" for r in prior.get("rows", []))
            ):
                continue
            if day not in actual_cache:
                ready = readiness(data_dir, day)
                if not all(
                    ready["checks"].get(k)
                    for k in ("日K", "技术特征", "核心指数", "复权历史一致性")
                ):
                    actual_cache[day] = None
                else:
                    manifest = rotation.input_signature(
                        data_dir, trading_dates(data_dir, day)[-70:]
                    )
                    payload = rotation.build_sector_rotation(
                        data_dir, as_of=day, lookback_days=70, limit=100
                    )
                    if manifest != rotation.input_signature(
                        data_dir, trading_dates(data_dir, day)[-70:]
                    ):
                        actual_cache[day] = None
                        continue
                    actual_rows = sorted(
                        payload.get("rows", []), key=lambda r: number(r.get("ret_1d")), reverse=True
                    )
                    for i, row in enumerate(actual_rows, 1):
                        row["actual_rank"] = i
                    actual_cache[day] = (
                        actual_rows,
                        hashlib.sha256(encode(manifest).encode()).hexdigest(),
                    )
            cached = actual_cache[day]
            if cached is None:
                continue
            actual_rows, digest = cached
            actual_map = {r["sector"]: r for r in actual_rows}
            results = []
            for prediction in snapshot["rows"]:
                actual = actual_map.get(prediction["sector"])
                if (
                    actual is None
                    or actual.get("ret_1d") is None
                    or number(actual.get("price_coverage")) < 0.98
                ):
                    results.append(
                        {
                            "sector": prediction["sector"],
                            "status": "missing",
                            "status_label": "缺数据",
                        }
                    )
                    continue
                outcome = evaluate_outcome(prediction, actual)
                results.append(
                    {
                        "sector": prediction["sector"],
                        "status": "hit" if outcome["hit"] else "miss",
                        "status_label": "命中" if outcome["hit"] else "未命中",
                        "actual_rank": actual["actual_rank"],
                        "actual_ret_1d": actual["ret_1d"],
                        "actual_up_ratio": actual["up_ratio"],
                        "actual_limit_up_count": actual["limit_up_count"],
                        **outcome,
                    }
                )
            store.append_validation(
                snapshot["id"],
                h,
                {
                    "verify_date": str(day),
                    "validated_at": now.isoformat(),
                    "input_version": digest,
                    "rows": results,
                    "actual_top10": actual_rows[:10],
                    "all_sector_avg": mean(number(r.get("ret_1d")) for r in actual_rows)
                    if actual_rows
                    else None,
                },
            )


def merge_validations(data_dir, snapshot):
    snapshot = json.loads(encode(snapshot))
    latest = {}
    for event in ForecastStore(data_dir).validations(snapshot["id"]):
        latest[event["horizon_days"]] = event
    for row in snapshot["rows"]:
        values = []
        for h in (1, 2):
            event = latest.get(h, {})
            actual = next((r for r in event.get("rows", []) if r["sector"] == row["sector"]), None)
            values.append(
                {
                    "horizon_days": h,
                    "verify_date": event.get("verify_date"),
                    **(actual or {"status": "pending", "status_label": "待验证"}),
                }
            )
        row["validations"] = values
        # The headline is T+1, never a best-of-two result with early wins only.
        row["validation_status"] = values[0]["status"]
        row["validation_label"] = values[0]["status_label"]
        row["best_validation"] = values[0] if values[0]["status"] in {"hit", "miss"} else None
    summary = {"horizons": {}}
    for h in (1, 2):
        values = [(r, r["validations"][h - 1]) for r in snapshot["rows"]]
        hs = {}
        for name, subset in (
            ("top5", values[:5]),
            ("top10", values[:10]),
            ("strong", [(r, v) for r, v in values if r["prediction_level"] == "strong"]),
        ):
            subset = [(r, v) for r, v in subset if v["status"] in {"hit", "miss"}]
            hs[f"{name}_hits"] = sum(v["status"] == "hit" for _, v in subset)
            hs[f"{name}_verified"] = len(subset)
        for kind in ("start", "continuation"):
            subset = [
                (r, v)
                for r, v in values
                if r["forecast_kind"] == kind
                and r["prediction_level"] == "strong"
                and v["status"] in {"hit", "miss"}
            ]
            hs[f"{kind}_hits"] = sum(v["status"] == "hit" for _, v in subset)
            hs[f"{kind}_verified"] = len(subset)
        summary["horizons"][str(h)] = hs
    summary.update(summary["horizons"]["1"])
    one = latest.get(1, {})
    preds = {r["sector"]: r for r in snapshot["rows"]}
    actual_top = []
    for a in one.get("actual_top10", []):
        p = preds.get(a["sector"], {})
        target_hit = bool(p) and evaluate_outcome(p, a)["hit"]
        verdict = (
            "强预测命中"
            if target_hit and p.get("prediction_level") == "strong"
            else (
                "观察命中"
                if target_hit
                and p.get("prediction_level") == "watch"
                and p.get("prediction_rank", 999) <= 10
                else "漏判"
            )
        )
        actual_top.append(
            {
                "sector": a["sector"],
                "actual_rank": a["actual_rank"],
                "actual_ret_1d": a["ret_1d"],
                "actual_up_ratio": a["up_ratio"],
                "actual_limit_up_count": a["limit_up_count"],
                "verdict": verdict,
                "prediction_rank": p.get("prediction_rank"),
                "prediction_score": p.get("prediction_score"),
            }
        )
    next_returns = [
        r["validations"][0]["actual_ret_1d"]
        for r in snapshot["rows"][:10]
        if r["validations"][0].get("actual_ret_1d") is not None
    ]
    avg = mean(next_returns) if next_returns else None
    baseline = one.get("all_sector_avg")
    summary.update(
        verify_1d_date=one.get("verify_date"),
        actual_top10_1d=actual_top,
        actual_top10_verified=len(actual_top),
        actual_top10_strong_hits=sum(a["verdict"] == "强预测命中" for a in actual_top),
        actual_top10_watch_hits=sum(a["verdict"] == "观察命中" for a in actual_top),
        actual_top10_misses=sum(a["verdict"] == "漏判" for a in actual_top),
        missed_actual_top10_1d=[a for a in actual_top if a["verdict"] == "漏判"],
        top10_next_1d_avg=avg,
        all_sector_next_1d_avg=baseline,
        top10_excess_1d=avg - baseline if avg is not None and baseline is not None else None,
    )
    snapshot["validation_summary"] = summary
    return snapshot


def records(data_dir, *, days=8, as_of=None, limit=30, origin="published"):
    from app.services.sector_prediction import _aggregate_snapshots, _read_store

    if origin == "legacy":
        snapshots = [
            {**s, "origin": "legacy", "status": "historical_replay"} for s in _read_store(data_dir)
        ]
    else:
        snapshots = ForecastStore(data_dir).list(origin)
    snapshots = [s for s in snapshots if as_of is None or s["as_of"] <= str(as_of)][:days]
    if origin != "legacy":
        snapshots = [merge_validations(data_dir, s) for s in snapshots]
    aggregate = _aggregate_snapshots(snapshots)
    aggregate["horizons"] = {
        str(h): {
            key: sum(
                s.get("validation_summary", {}).get("horizons", {}).get(str(h), {}).get(key, 0)
                for s in snapshots
            )
            for key in (
                "top5_hits",
                "top5_verified",
                "top10_hits",
                "top10_verified",
                "strong_hits",
                "strong_verified",
                "start_hits",
                "start_verified",
                "continuation_hits",
                "continuation_verified",
            )
        }
        for h in (1, 2)
    }
    for s in snapshots:
        s["rows"] = s["rows"][:limit]
    return {
        "model_version": MODEL_VERSION,
        "origin": origin,
        "snapshots": snapshots,
        "aggregate": aggregate,
        "store_file": str(ForecastStore(data_dir).path),
    }


def refresh(data_dir, *, days=8, as_of=None, lookback_days=70, limit=30):
    # Only one new forecast; old predictions get validation events, never a rebuild.
    snapshot = build(data_dir, as_of=as_of, lookback_days=lookback_days, limit=limit)
    validate(data_dir)
    payload = records(data_dir, days=days, as_of=as_of, limit=limit, origin=snapshot["origin"])
    payload["generation"] = {
        k: snapshot.get(k) for k in ("status", "as_of", "summary", "readiness")
    }
    payload["generated_dates"] = [snapshot["as_of"]] if snapshot.get("id") else []
    return payload
