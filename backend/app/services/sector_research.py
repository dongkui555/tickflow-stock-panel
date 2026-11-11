"""Date-blocked walk-forward comparisons using the production portfolio matcher."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path
from statistics import mean

import polars as pl

from app.market_time import cn_now
from app.services import sector_layout as rotation
from app.services.sector_forecast import enrich_context, trading_dates
from app.services.sector_forecast_rules import (
    MODEL_VERSION,
    classify,
    evaluate_outcome,
    number,
    rank_forecasts,
)
from app.services.sector_forecast_store import encode

VARIANTS = (
    "v2",
    "no_flow_change",
    "no_resilience",
    "no_volume",
    "no_leader",
    "no_position",
    "no_parent",
    "no_market_gate",
    "legacy",
    "relative_strength",
    "random_matched",
)
EXPERIMENTS = ("baseline", "early_layout", "stock_entry")


def result_pointer(experiment):
    return {
        "baseline": "latest.json",
        "early_layout": "latest-early-layout.json",
        "stock_entry": "latest-stock-entry.json",
    }[experiment]


def assess(prediction, actual, horizon):
    """Exact-session confirmation and complete-window launch are distinct labels."""
    if horizon in (1, 2):
        outcome = evaluate_outcome(prediction, actual)
        return {
            **outcome,
            "period_return": number(actual.get("ret_1d")),
            "period_excess": number(actual.get("relative_return")),
            "first_start_day": horizon if outcome["new_start"] else None,
            "late_start": False,
        }
    window = actual.get("window", [])
    if horizon not in (3, 5) or len(window) != horizon:
        return None
    wealth = benchmark = peak = 1.0
    drawdown = minimum_return = 0.0
    first = None
    eligible = prediction.get("forecast_kind") == "start" and not prediction.get("already_started")
    for index, row in enumerate(window, 1):
        if not row or number(row.get("price_coverage")) < 0.98:
            return None
        ret, excess = number(row.get("ret_1d"), None), number(row.get("relative_return"), None)
        if ret is None or excess is None or ret <= -1 or ret - excess <= -1:
            return None
        wealth *= 1 + ret
        benchmark *= 1 + ret - excess
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1)
        minimum_return = min(minimum_return, wealth - 1)
        if first is None and eligible and evaluate_outcome(prediction, row)["new_start"]:
            first = index
    return {
        "hit": first is not None,
        "new_start": first is not None,
        "positive_return": wealth > 1,
        "relative_strength": wealth > benchmark,
        "period_return": wealth - 1,
        "benchmark_return": benchmark - 1,
        "period_excess": wealth - benchmark,
        "max_drawdown": drawdown,
        "minimum_return": minimum_return,
        "first_start_day": first,
        "first_start_date": window[first - 1].get("as_of") if first else None,
        "late_start": first is not None and first > 2,
    }


def build_outcomes(days, source, horizons):
    by_day = {d: {r["sector"]: r for r in source.get(d, ([], {}))[0]} for d in days}
    outcomes = {}
    for index, day in enumerate(days):
        for h in horizons:
            if index + h >= len(days):
                continue
            for name, row in by_day[days[index + h]].items():
                if h <= 2:
                    if (
                        number(row.get("ret_1d"), None) is not None
                        and number(row.get("price_coverage")) >= 0.98
                    ):
                        outcomes[(day, h, name)] = row
                else:
                    actual = {
                        "window": [by_day[d].get(name) for d in days[index + 1 : index + h + 1]]
                    }
                    if assess({"forecast_kind": "start"}, actual, h) is not None:
                        outcomes[(day, h, name)] = actual
    return outcomes


def _score(row, horizon):
    return row[f"horizon_{horizon}_score"] if horizon in (1, 2) else row["prediction_score"]


def walk_forward_folds(dates, *, train_days=80, test_days=20, holdout_days=40, gap=2):
    end = len(dates) - holdout_days
    folds = []
    cursor = train_days + gap
    while cursor < end:
        stop = min(cursor + test_days, end)
        folds.append(
            {
                "train": dates[cursor - gap - train_days : cursor - gap],
                "test": dates[cursor:stop],
                "kind": "walk_forward",
            }
        )
        cursor = stop
    if end >= train_days + gap:
        folds.append(
            {
                "train": dates[end - gap - train_days : end - gap],
                "test": dates[end:],
                "kind": "holdout",
            }
        )
    return folds


def calibration(training, outcomes, horizon):
    bins = {}
    for day, rows in training.items():
        for row in rows:
            actual = outcomes.get((day, horizon, row["sector"]))
            if actual is None or row["prediction_level"] == "unavailable":
                continue
            key = (row["forecast_kind"], int(_score(row, horizon) // 10))
            bins.setdefault(key, []).append(assess(row, actual, horizon)["hit"])
    return {
        f"{kind}:{band}": {"n": len(values), "p": (sum(values) + 1) / (len(values) + 2)}
        for (kind, band), values in bins.items()
    }


def calibrated_probability(row, bins, horizon, minimum_samples=20):
    key = f"{row['forecast_kind']}:{int(_score(row, horizon) // 10)}"
    estimate = bins.get(key, {})
    return estimate.get("p") if estimate.get("n", 0) >= minimum_samples else None


def select_variant(rows, market, variant, count):
    if variant == "v2" or variant.startswith("no_"):
        return [
            r
            for r in rank_forecasts(rows, market, ablation=None if variant == "v2" else variant)
            if r["prediction_level"] == "strong"
        ][:3]
    ready = [r for r in rows if classify(r, market)["prediction_level"] != "unavailable"]
    if variant == "legacy":
        from app.services.sector_prediction import _parent_contexts, _prediction_scores

        parents = _parent_contexts(rows)
        ranked = sorted(
            ready,
            key=lambda r: _prediction_scores(r, parents.get(r["sector"].split("-")[0]), market)[
                "priority_score"
            ],
            reverse=True,
        )
    elif variant == "relative_strength":
        ranked = sorted(ready, key=lambda r: number(r.get("ret_5d")), reverse=True)
    else:
        ranked = sorted(
            ready,
            key=lambda r: hashlib.sha256(
                f"sector-v2-fixed-seed:{r['as_of']}:{r['sector']}".encode()
            ).hexdigest(),
        )
    return [classify(r, market) for r in ranked[: count if variant == "random_matched" else 3]]


def trade_comparison(
    bars, selections, dates, horizon, capital, *, stop_loss_pct=None, opening_bounds=None
):
    from app.backtest.engine import BacktestEngine, MatcherConfig, SimulationOptions
    from app.backtest.matrix import build_market_matrix

    if not dates or not selections:
        return {"total_return": 0.0, "n_trades": 0, "execution": {}, "equity_curve": []}
    marks = []
    for day, rows in selections.items():
        for row in rows:
            # Leader membership/rank is frozen at signal close, never picked from future winners.
            leaders = row.get("leader_stocks") or []
            if leaders:
                marks.append(
                    {
                        "date": day,
                        "symbol": leaders[0]["symbol"],
                        "score": row["prediction_score"],
                        "entry": True,
                    }
                )
    if not marks:
        return {"total_return": 0.0, "n_trades": 0, "execution": {}, "equity_curve": []}
    marks = pl.DataFrame(marks).unique(["date", "symbol"])
    names = marks["symbol"].unique().to_list()
    panel = bars.filter(pl.col("symbol").is_in(names) & pl.col("date").is_in(dates))
    panel = panel.drop([c for c in ("score", "entry") if c in panel.columns]).join(
        marks, on=["date", "symbol"], how="left"
    )
    panel = panel.with_columns(
        pl.col("entry").fill_null(False),
        pl.col("score").fill_null(0.0),
        (pl.col("consecutive_limit_ups").fill_null(0) > 0).alias("signal_limit_up"),
    )
    if "consecutive_limit_downs" in panel.columns:
        panel = panel.with_columns(
            (pl.col("consecutive_limit_downs").fill_null(0) > 0).alias("signal_limit_down")
        )
    panel = panel.sort(["symbol", "date"])
    matrix = build_market_matrix(panel, panel["entry"], None, entry_delay_bars=1, exit_delay_bars=0)
    opening_rejections = []
    if opening_bounds is not None:
        from app.services.sector_entry import apply_opening_guards

        allowed_entries, opening_rejections = apply_opening_guards(matrix, opening_bounds)
        matrix = replace(matrix, entry=allowed_entries)
    config = MatcherConfig(
        entry_fill="open_t+1",
        exit_fill="open_t+1",
        commission_pct=0.0003,
        stamp_tax_pct=0.0005,
        slippage_bps=10,
        max_hold_days=horizon,
        max_positions=3,
        max_exposure_pct=0.6,
        initial_capital=capital,
        stop_loss_pct=stop_loss_pct,
    )
    result = BacktestEngine(repo=None).simulate_market_matrix(
        matrix, config, options=SimulationOptions(include_monte_carlo=False)
    )
    return {
        "n_trades": len(result.trades),
        **result.stats,
        "equity_curve": result.equity_curve,
        "trades": [asdict(t) for t in result.trades],
        "turnover": sum(t.entry_value for t in result.trades) / capital,
        "execution_model": "next_open_stock_leader_Tplus1_fees_slippage_limits_lots",
        "fee_caveat": "按比例佣金, 尚未单独计入最低佣金及冲击成本",
        "stop_loss_pct": stop_loss_pct,
        "opening_rejections": opening_rejections,
    }


def run(
    data_dir: Path,
    *,
    lookback_days=252,
    train_days=80,
    test_days=20,
    holdout_days=40,
    capital=100000.0,
    on_progress=None,
    experiment="baseline",
):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown sector research experiment")
    early = experiment != "baseline"
    stock_entry = experiment == "stock_entry"
    horizons = (1, 2, 3, 5) if early else (1, 2)
    variants = ("v2", "legacy", "relative_strength", "random_matched") if early else VARIANTS
    gap = max(horizons)
    stop_loss = None
    if early:
        from app.services.short_term_ai import SHORT_TERM_STOP_LOSS_PCT

        stop_loss = SHORT_TERM_STOP_LOSS_PCT["balanced"]
    now = cn_now()
    days = [d for d in trading_dates(data_dir, now.date()) if d < now.date() or now.hour >= 19][
        -lookback_days:
    ]
    # Keep labels plus next-open liquidation outside the last signal window.
    folds = walk_forward_folds(
        days[25 : -(gap + 1 + (2 if stock_entry else 0))],
        train_days=train_days,
        test_days=test_days,
        holdout_days=holdout_days,
        gap=gap,
    )
    if not folds:
        raise ValueError("已收盘交易日不足以完成训练、隔离期和独立留出验证")
    progress = on_progress or (lambda *args: None)
    progress("sector_research", 2, "读取日K、资金及行业数据")
    manifest = rotation.input_signature(data_dir, days)
    name_map = rotation._load_name_map(data_dir) if stock_entry else {}
    sector, meta, bars = rotation._sector_daily_frame(data_dir, days, 2)
    source = {}
    models = {}
    for idx, day in enumerate(days):
        if idx < 25:
            continue
        rows = rotation.rows_for_date(data_dir, sector, bars, day)
        rows, market = enrich_context(rows)
        for row in rows:
            row["as_of"] = str(day)
        source[day] = (rows, market)
        models[day] = rank_forecasts(rows, market)
        if idx % 10 == 0:
            progress(
                "sector_features", 5 + int(idx / len(days) * 40), f"板块特征 {idx}/{len(days)}"
            )
    outcomes = build_outcomes(days, source, horizons)
    results = []
    fold_meta = []
    calibration_reports = []
    oos_probabilities = {}
    decisions = {v: {} for v in variants}
    forecast_records = []
    for index, fold in enumerate(folds):
        train = {day: models.get(day, []) for day in fold["train"]}
        cals = {h: calibration(train, outcomes, h) for h in horizons}
        fold_meta.append(
            {
                "fold": index + 1,
                "kind": fold["kind"],
                "train_start": str(fold["train"][0]),
                "train_end": str(fold["train"][-1]),
                "test_start": str(fold["test"][0]),
                "test_end": str(fold["test"][-1]),
                "gap_sessions": gap,
            }
        )
        calibration_reports.append({"fold": index + 1, "bins": cals})
        for day in fold["test"]:
            rows, market = source.get(day, ([], {}))
            for row in models.get(day, []):
                for h in horizons:
                    oos_probabilities[(day, h, row["sector"])] = calibrated_probability(
                        row, cals[h], h
                    )
            candidates = rows
            if early:
                candidates = [r for r in rows if not classify(r, market)["already_started"]]
            count = len(
                [
                    r
                    for r in models.get(day, [])
                    if r["prediction_level"] == "strong" and (not early or not r["already_started"])
                ]
            )
            for variant in variants:
                # Freeze the production v2 selection; do not promote rank-four observers.
                picked = (
                    [r for r in models.get(day, []) if r["prediction_level"] == "strong"]
                    if early and variant == "v2"
                    else select_variant(candidates, market, variant, count)
                )
                decisions[variant][day] = [
                    r for r in picked if not early or not r["already_started"]
                ]
            if early:
                for row in decisions["v2"][day]:
                    forecast_records.append(
                        {
                            "signal_id": f"{day}:{row['sector']}",
                            "signal_date": str(day),
                            "phase": fold["kind"],
                            "fold": index + 1,
                            "sector": row["sector"],
                            "prediction_score": row["prediction_score"],
                            "prediction_level": row["prediction_level"],
                            "leader_stocks": row.get("leader_stocks", [])[:1],
                            "validations": {
                                str(h): (
                                    {
                                        "status": "verified",
                                        **assess(row, outcomes[(day, h, row["sector"])], h),
                                    }
                                    if (day, h, row["sector"]) in outcomes
                                    else {"status": "missing"}
                                )
                                for h in horizons
                            },
                        }
                    )
        progress(
            "sector_folds",
            45 + int((index + 1) / len(folds) * 20),
            f"样本外折 {index + 1}/{len(folds)}",
        )
    for phase_idx, phase in enumerate(("walk_forward", "holdout")):
        test_dates = sorted(d for f in folds if f["kind"] == phase for d in f["test"])
        if not test_dates:
            continue
        trade_dates = days[days.index(test_dates[0]) : days.index(test_dates[-1]) + gap + 2]
        for v_idx, variant in enumerate(variants):
            selected = {d: decisions[variant][d] for d in test_dates}
            for h in horizons:
                pairs = [
                    (r, outcomes[(day, h, r["sector"])])
                    for day, rows in selected.items()
                    for r in rows
                    if (day, h, r["sector"]) in outcomes
                ]
                start_actual = {
                    (day, r["sector"])
                    for day in test_dates
                    for r in models.get(day, [])
                    if not r["already_started"]
                    and (day, h, r["sector"]) in outcomes
                    and assess(r, outcomes[(day, h, r["sector"])], h)["new_start"]
                }
                detected = {
                    (day, r["sector"])
                    for day, rows in selected.items()
                    for r in rows
                    if (day, r["sector"]) in start_actual
                }
                groups = {}
                for kind in ("start", "continuation"):
                    group = [(r, a) for r, a in pairs if r["forecast_kind"] == kind]
                    groups[kind] = {
                        "selected": len(group),
                        "hits": sum(assess(r, a, h)["hit"] for r, a in group),
                        "precision": mean(assess(r, a, h)["hit"] for r, a in group)
                        if group
                        else None,
                    }
                probabilities = []
                if variant == "v2":
                    for day, rows in selected.items():
                        for row in rows:
                            key = (day, h, row["sector"])
                            p = oos_probabilities.get(key)
                            if p is not None and key in outcomes:
                                probabilities.append((p, assess(row, outcomes[key], h)["hit"]))
                trading = trade_comparison(
                    bars, selected, trade_dates, h, capital, stop_loss_pct=stop_loss
                )
                assessed = [assess(r, a, h) for r, a in pairs]
                waiting = [a["first_start_day"] for a in assessed if a.get("first_start_day")]
                results.append(
                    {
                        "phase": phase,
                        "variant": variant,
                        "horizon": h,
                        "selected": len(pairs),
                        "precision": mean(a["hit"] for a in assessed) if pairs else None,
                        "new_start_recall": len(detected) / len(start_actual)
                        if start_actual
                        else None,
                        "missed_starts": len(start_actual) - len(detected),
                        "forecast_kinds": groups,
                        "calibrated_samples": len(probabilities),
                        "brier_score": mean((p - float(y)) ** 2 for p, y in probabilities)
                        if probabilities
                        else None,
                        "positive_rate": mean(a["period_return"] > 0 for a in assessed)
                        if pairs
                        else None,
                        "mean_sector_return": mean(a["period_return"] for a in assessed)
                        if pairs
                        else None,
                        "mean_sector_excess": mean(a["period_excess"] for a in assessed)
                        if pairs
                        else None,
                        "abstention_days": sum(not r for r in selected.values()),
                        "test_days": len(test_dates),
                        "trading": trading,
                        "issued": sum(len(r) for r in selected.values()),
                        "missing_labels": sum(len(r) for r in selected.values()) - len(pairs),
                        "mean_start_day": mean(waiting) if waiting else None,
                        "late_hits": sum(a["late_start"] for a in assessed),
                        "mean_window_drawdown": mean(a["max_drawdown"] for a in assessed)
                        if h > 2 and assessed
                        else None,
                        "label_kind": "window_launch" if h > 2 else "session_confirmation",
                    }
                )
            progress(
                "sector_matching",
                65 + int((phase_idx * len(variants) + v_idx + 1) / (2 * len(variants)) * 30),
                f"{phase} 对照撮合 {variant}",
            )
    entry_comparisons, entry_records = [], []
    if stock_entry:
        from app.services.sector_entry import DIAGNOSTIC_VARIANTS, run_comparison

        entry_comparisons, entry_records = run_comparison(
            bars,
            decisions["v2"],
            source,
            days,
            folds,
            capital,
            stop_loss,
            trade_comparison,
            progress,
            name_map=name_map,
            variants=DIAGNOSTIC_VARIANTS,
        )
    if manifest != rotation.input_signature(data_dir, days):
        raise ValueError("实验期间本地数据发生变化, 本次结果不发布; 请在同步结束后重试")
    if stock_entry and name_map != rotation._load_name_map(data_dir):
        raise ValueError("实验期间股票名称维表发生变化, 本次结果不发布")
    report = {
        "id": f"{experiment + '_' if early else ''}{now:%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}",
        "generated_at": now.isoformat(),
        "model_version": MODEL_VERSION,
        "status": "research_only",
        "experiment": experiment,
        "origin": "historical_replay",
        "forecast_records": forecast_records,
        "entry_comparisons": entry_comparisons,
        "entry_records": entry_records,
        "folds": fold_meta,
        "comparisons": results,
        "calibration": calibration_reports,
        "source_meta": meta,
        "input_version": hashlib.sha256(
            encode([manifest, name_map] if stock_entry else manifest).encode()
        ).hexdigest(),
        "parameters": {
            "lookback_days": lookback_days,
            "train_days": train_days,
            "test_days": test_days,
            "holdout_days": holdout_days,
            "initial_capital": capital,
            "horizons": list(horizons),
            "gap_sessions": gap,
            "stop_loss_pct": stop_loss,
        },
        "limitations": [
            "当前行业成分映射缺乏历史时点版本, 仍存在幸存者/成分修订偏差",
            "历史分区没有当时下载留痕, 回放不能冒充正式向前预测",
            "使用信号日板块龙头股票模拟, 非行业指数或ETF收益",
            "T+1预测涨幅不等于T+1可卖收益, 股票交易遵守至少隔日卖出",
            "规则权重固定; 概率分箱只用训练期, 不自动将实验结果切换到线上",
        ],
    }
    if early:
        report["limitations"] += [
            "T+1/T+2为目标日确认, T+3/T+5为完整窗口内首次启动; 窗口命中不等于持有盈利",
            "四个期限使用同一批启动前候选及固定4.5%止损; 盘后信号次日开盘成交, 不放宽风控",
            "板块窗口回撤基于收盘收益, 非盘中最低价; 股票收益由独立撮合计算",
            "此历史区间已用于此前研究, 留出段仅作诊断, 不能作为新的独立上线验收",
        ]
    if stock_entry:
        from app.services.sector_entry import VERSION

        report["parameters"].update(
            entry_rule_version=VERSION,
            confirmation_sessions=2,
            entry_holding_sessions=[3, 5],
            max_open_gap_pct=0.02,
            max_open_drop_pct=0.03,
            entry_variant_count=len(DIAGNOSTIC_VARIANTS),
        )
        report["limitations"] += [
            "原四组保持基准; 新增单条件消融及相对强势/分形态确认实验, 固定4.5%止损及60%总仓位, 不调整线上规则",
            "标的排名冻结于板块信号收盘; 等待T+1/T+2收盘确认后次日开盘, 最早T+2买入, 不是分钟精确确认",
            "确认组持有3/5日从实际买入日计起; 板块窗口标签仍从原始预测日计起, 两者不可混用",
            "个股排序使用量价技术字段; 未补造个股资金、财务和事件证据, 现有行业映射及名称仍有时点偏差",
        ]
    report = clean_json(report)
    root = data_dir / "research" / "sector_forecasts"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{report['id']}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, allow_nan=False)
    temp = root / f"latest-{uuid.uuid4().hex}.tmp"
    temp.write_text(encode({"id": report["id"]}), encoding="utf-8")
    os.replace(temp, root / result_pointer(experiment))
    progress("sector_research", 100, "滚动样本外对照完成, 实验结果未切换至线上")
    return report


def clean_json(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def latest(data_dir, *, experiment="baseline"):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown sector research experiment")
    root = data_dir / "research" / "sector_forecasts"
    pointer = root / result_pointer(experiment)
    if not pointer.exists():
        return {"status": "empty", "comparisons": []}
    identity = json.loads(pointer.read_text(encoding="utf-8"))["id"]
    if Path(identity).name != identity or any(c in identity for c in ("/", "\\", ":")):
        raise ValueError("Invalid research id")
    return json.loads((root / f"{identity}.json").read_text(encoding="utf-8"))
