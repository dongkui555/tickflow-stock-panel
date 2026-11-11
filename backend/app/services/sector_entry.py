"""Frozen sector members and causal daily entry confirmation for research only."""

from __future__ import annotations

from collections import Counter
from statistics import mean

import numpy as np
import polars as pl

from app.price_limits import is_risk_warning_name
from app.services.sector_forecast_rules import clamp, number

VERSION = "sector_stock_entry_v2_diagnostics"
VARIANTS = ("leader_open", "ranked_open", "leader_confirm", "ranked_confirm")
DIAGNOSTIC_VARIANTS = (
    *VARIANTS,
    "ranked_no_pattern",
    "ranked_no_candle",
    "ranked_no_location",
    "ranked_no_volume",
    "strength_open",
    "strength_confirm",
    "strength_adaptive",
)


def condition(key, label, passed, observed, required, *, hard=False, active=True):
    return dict(
        id=key,
        label=label,
        status="missing" if passed is None else "passed" if passed else "failed",
        observed=observed,
        required=required,
        hard=hard,
        active=active,
    )


def prepare_features(bars, days, *, name_map=None):
    if bars.is_empty():
        return bars
    columns = [
        "date",
        "symbol",
        "sector",
        "name",
        "open",
        "high",
        "low",
        "close",
        "amount",
        "ret_3d",
        "ret_5d",
        "amount_ratio_5d",
        "turnover_rate",
        "price_valid",
        "limit_up_flag",
        "limit_known",
    ]
    df = bars.select([c for c in columns if c in bars.columns]).sort(["symbol", "date"])
    for c in columns:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    if name_map:
        df = df.with_columns(
            pl.coalesce(
                pl.col("name"),
                pl.col("symbol").replace_strict(name_map, default=None, return_dtype=pl.String),
            ).alias("name")
        )
    df = df.join(pl.DataFrame({"date": days, "_session": range(len(days))}), on="date", how="inner")
    df = (
        df.with_columns(
            pl.col("close").shift(1).over("symbol").alias("prior_close"),
            pl.col("high").shift(1).over("symbol").alias("prior_high"),
            pl.col("close").rolling_mean(5).over("symbol").alias("ma5"),
            pl.col("close").rolling_mean(20).over("symbol").alias("ma20"),
            pl.col("low").shift(1).rolling_min(10).over("symbol").alias("low10"),
            pl.col("low").rolling_min(20).over("symbol").alias("_low20"),
            pl.col("high").rolling_max(20).over("symbol").alias("_high20"),
            (
                (pl.col("_session") - pl.col("_session").shift(19).over("symbol") == 19)
                & (
                    pl.col("price_valid")
                    .fill_null(False)
                    .cast(pl.Int32)
                    .rolling_sum(20)
                    .over("symbol")
                    == 20
                )
            )
            .fill_null(False)
            .alias("history_ok"),
        )
        .with_columns(
            pl.col("ma20").shift(5).over("symbol").alias("ma20_prior"),
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("prior_close")).abs(),
                (pl.col("low") - pl.col("prior_close")).abs(),
            ).alias("_tr"),
            ((pl.col("close") - pl.col("_low20")) / (pl.col("_high20") - pl.col("_low20"))).alias(
                "position20"
            ),
            ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias(
                "close_location"
            ),
            (
                pl.col("ret_5d")
                - pl.when(pl.col("history_ok"))
                .then(pl.col("ret_5d"))
                .otherwise(None)
                .mean()
                .over(["date", "sector"])
            ).alias("relative5"),
        )
        .with_columns(
            (pl.col("_tr").rolling_mean(14).over("symbol") / pl.col("close")).alias("atr_pct")
        )
    )
    return df.drop([c for c in df.columns if c.startswith("_")])


def support(row):
    return max(number(row.get("low10")), number(row.get("ma20")) * 0.985)


def evaluate_stock(row, *, profile="legacy"):
    if profile not in {"legacy", "strength"}:
        raise ValueError("Unknown stock selection profile")
    required = (
        "close",
        "ma20",
        "ma20_prior",
        "low10",
        "position20",
        "atr_pct",
        "ret_3d",
        "ret_5d",
        "relative5",
        "amount",
        "turnover_rate",
        "close_location",
    )
    if (
        not row.get("history_ok")
        or not row.get("price_valid")
        or not row.get("name")
        or any(
            number(row.get(c), None) is None
            for c in (*required, *(("ma5",) if profile == "strength" else ()))
        )
        or number(row.get("limit_known")) < 1
    ):
        return {
            **row,
            "eligible": False,
            "score": None,
            "reasons": ["历史或价格字段不足"],
            "components": {},
            "conditions": [
                condition(
                    "data",
                    "历史或价格字段不足",
                    None,
                    {c: number(row.get(c), None) for c in required},
                    "连续有效历史、名称、涨停口径和全部选股字段",
                    hard=True,
                )
            ],
        }
    price = row["close"]
    if price <= 0 or row["ma20"] <= 0 or row["ma20_prior"] <= 0:
        return {
            **row,
            "eligible": False,
            "score": None,
            "reasons": ["价格无效"],
            "components": {},
            "conditions": [
                condition("data", "价格无效", False, {"close": price}, "价格及均线为正", hard=True)
            ],
        }
    position_hot = (
        row["position20"] > 0.78
        if profile == "legacy"
        else (row["position20"] > 0.9 and (row["ret_3d"] > 0.04 or price / row["ma20"] - 1 > 0.03))
    )
    checks = [
        condition(
            "data",
            "数据完整",
            True,
            {"history_ok": True},
            "连续有效历史、名称和涨停口径",
            hard=True,
        ),
        condition(
            "risk_name",
            "风险标记或当日涨停",
            not (is_risk_warning_name(row.get("name")) or number(row.get("limit_up_flag")) > 0),
            {"name": row["name"], "limit_up": row.get("limit_up_flag")},
            "非风险警示、非当日涨停",
            hard=True,
        ),
        condition(
            "liquidity",
            "流动性或换手不合格",
            row["amount"] >= 50_000_000 and 0.2 <= row["turnover_rate"] <= 20,
            {"amount_yuan": row["amount"], "turnover_pct": row["turnover_rate"]},
            "成交额≥5000万元; 换手0.2%-20%",
            hard=True,
        ),
        condition(
            "heat",
            "追涨位置",
            row["ret_3d"] <= 0.065 and row["ret_5d"] <= 0.105 and not position_hot,
            {
                "ret_3d": row["ret_3d"],
                "ret_5d": row["ret_5d"],
                "position20": row["position20"],
                "ma20_distance": price / row["ma20"] - 1,
            },
            "3日≤6.5%、5日≤10.5%; 位置≤0.78"
            if profile == "legacy"
            else "涨幅上限不变; 位置>0.9且满足3日>4%或MA20偏离>3%时排除",
        ),
        condition(
            "trend",
            "趋势尚未企稳",
            price >= row["ma20"] and row["ma20"] >= row["ma20_prior"] * 0.99,
            {"close": price, "ma20": row["ma20"], "ma20_prior": row["ma20_prior"]},
            "收盘≥MA20; MA20五日降幅≤1%",
        ),
        condition(
            "risk_distance",
            "波动或支撑距离过大",
            row["atr_pct"] <= 0.045 and price / support(row) - 1 <= 0.045,
            {"atr_pct": row["atr_pct"], "support_distance": price / support(row) - 1},
            "ATR与支撑距离均≤4.5%",
            hard=True,
        ),
    ]
    if profile == "strength":
        checks += [
            condition(
                "relative_strength",
                "板块内弱势跟随",
                row["relative5"] >= 0,
                {"relative5": row["relative5"]},
                "5日收益不低于同板块均值",
            ),
            condition(
                "recovery",
                "短线未收复",
                price >= row["ma5"] and row["ret_3d"] >= -0.02 and row["close_location"] >= 0.5,
                {
                    "close": price,
                    "ma5": row["ma5"],
                    "ret_3d": row["ret_3d"],
                    "close_location": row["close_location"],
                },
                "收盘≥MA5; 3日跌幅≤2%; 收盘处于日内上半区",
            ),
        ]
    reasons = [c["label"] for c in checks if c["status"] != "passed"]
    components = {
        "趋势": clamp(50 + (row["ma20"] / row["ma20_prior"] - 1) * 1500),
        "位置": clamp(100 - abs(row["position20"] - 0.5) * 160),
        "波动": clamp(100 - row["atr_pct"] / 0.045 * 100),
        "相对强度": clamp(50 + row["relative5"] * 1000),
        "流动性": clamp(row["amount"] / 200_000_000 * 100),
        "收盘强度": clamp(row["close_location"] * 100),
    }
    weights = (
        (0.2, 0.25, 0.2, 0.15, 0.1, 0.1)
        if profile == "legacy"
        else (0.25, 0.1, 0.1, 0.3, 0.1, 0.15)
    )
    score = sum(v * w for v, w in zip(components.values(), weights, strict=True))
    return {
        **row,
        "eligible": not reasons,
        "score": round(score, 2),
        "components": components,
        "reasons": reasons,
        "support": support(row),
        "conditions": checks,
    }


def confirm(frozen, current, sector, market, *, mode="strict"):
    if mode not in {"strict", "no_pattern", "no_candle", "no_location", "no_volume", "adaptive"}:
        raise ValueError("Unknown confirmation mode; hard gates cannot be disabled")
    if (
        not current
        or not current.get("history_ok")
        or not current.get("price_valid")
        or any(
            number(current.get(c), None) is None
            for c in (
                "close",
                "open",
                "low",
                "ma5",
                "prior_high",
                "prior_close",
                "amount_ratio_5d",
                "close_location",
            )
        )
        or not sector
        or number(sector.get("price_coverage")) < 0.98
        or number(sector.get("flow_coverage")) < 0.98
    ):
        return {
            "status": "missing_data",
            "reasons": ["确认日量价或板块资金覆盖不足"],
            "conditions": [
                condition(
                    "data",
                    "确认日量价或板块资金覆盖不足",
                    None,
                    {
                        "stock_available": bool(current),
                        "price_coverage": (sector or {}).get("price_coverage"),
                        "flow_coverage": (sector or {}).get("flow_coverage"),
                    },
                    "有效确认日量价; 板块价格及资金覆盖≥98%",
                    hard=True,
                )
            ],
        }
    breakout = current["close"] >= current["prior_high"]
    reclaim = (
        current["low"] <= current["ma5"] * 1.01
        and current["close"] > current["prior_close"]
        and current["close"] >= current["ma5"]
    )
    volume_ok = 1.05 <= current["amount_ratio_5d"] <= 2
    adaptive_reclaim = reclaim and 0.8 <= current["amount_ratio_5d"] <= 1.3
    if mode == "adaptive":
        volume_ok = (breakout and volume_ok) or adaptive_reclaim
    checks = [
        condition(
            "data",
            "数据完整",
            True,
            {"price_coverage": sector["price_coverage"], "flow_coverage": sector["flow_coverage"]},
            "有效量价; 板块覆盖≥98%",
            hard=True,
        ),
        condition(
            "support",
            "跌破冻结支撑",
            current["close"] >= support(frozen),
            {"close": current["close"], "support": support(frozen)},
            "收盘不破信号日冻结支撑",
            hard=True,
        ),
        condition(
            "advance",
            "确认前涨幅过大",
            current["close"] <= frozen["close"] * 1.04,
            {"return_since_signal": current["close"] / frozen["close"] - 1},
            "确认前涨幅≤4%",
            hard=True,
        ),
        condition(
            "market",
            "市场风险过滤",
            market is not None
            and not any(market.get(k) for k in ("weak", "very_weak", "event_risk")),
            {k: (market or {}).get(k) for k in ("weak", "very_weak", "event_risk")},
            "市场状态已知; 非弱势、非高事件风险",
            hard=True,
        ),
        condition(
            "limits",
            "涨停或涨停口径缺失",
            number(current.get("limit_up_flag")) <= 0 and number(current.get("limit_known")) >= 1,
            {"limit_up": current.get("limit_up_flag"), "limit_known": current.get("limit_known")},
            "涨停口径已知且未涨停",
            hard=True,
        ),
        condition(
            "pattern",
            "突破或回踩回收未确认",
            breakout or reclaim,
            {"breakout": breakout, "reclaim": reclaim},
            "突破前日高点, 或回踩MA5后收复且较昨收上涨",
            active=mode != "no_pattern",
        ),
        condition(
            "candle",
            "收阳未确认",
            current["close"] > current["prior_close"]
            if mode == "adaptive"
            else current["close"] > current["open"],
            {
                "close": current["close"],
                "open": current["open"],
                "prior_close": current["prior_close"],
            },
            "收盘高于昨收" if mode == "adaptive" else "收盘高于开盘",
            active=mode != "no_candle",
        ),
        condition(
            "location",
            "强收盘未确认",
            current["close_location"] >= 0.65,
            {"close_location": current["close_location"]},
            "收盘位于当日振幅上部35%",
            active=mode != "no_location",
        ),
        condition(
            "volume",
            "温和放量未确认",
            volume_ok,
            {"amount_ratio_5d": current["amount_ratio_5d"]},
            "突破1.05-2倍; 回踩收复0.8-1.3倍" if mode == "adaptive" else "成交额/前5日均额1.05-2倍",
            active=mode != "no_volume",
        ),
    ]
    failed = [c["label"] for c in checks if c["active"] and c["status"] != "passed"]
    invalid = any(c["id"] in {"support", "advance"} and c["status"] != "passed" for c in checks)
    return {
        "status": "invalidated" if invalid else "waiting" if failed else "confirmed",
        "reasons": failed,
        "pattern": "回踩回收"
        if mode == "adaptive" and adaptive_reclaim
        else "突破"
        if breakout
        else "回踩回收",
        "conditions": checks,
    }


def summarize_conditions(groups):
    rows = {}
    for group in groups:
        blockers = [c for c in group if c["active"] and c["status"] != "passed"]
        alive = True
        for check in group:
            row = rows.setdefault(
                check["id"],
                {k: check[k] for k in ("id", "label", "hard", "required")}
                | {
                    "tested": 0,
                    "passed": 0,
                    "failed": 0,
                    "missing": 0,
                    "disabled": 0,
                    "sole_blocker": 0,
                    "survivors": 0,
                },
            )
            row["tested"] += 1
            row[check["status"]] += 1
            row["disabled"] += not check["active"]
            row["sole_blocker"] += len(blockers) == 1 and blockers[0]["id"] == check["id"]
            alive = alive and (not check["active"] or check["status"] == "passed")
            row["survivors"] += alive
    return list(rows.values())


def combine_summaries(groups):
    rows = {}
    for group in groups:
        for check in group:
            if check["id"] not in rows:
                rows[check["id"]] = dict(check)
            else:
                for k in (
                    "tested",
                    "passed",
                    "failed",
                    "missing",
                    "disabled",
                    "sole_blocker",
                    "survivors",
                ):
                    rows[check["id"]][k] += check[k]
    return list(rows.values())


def plan_entries(features, signals, source, days, variant):
    if variant not in DIAGNOSTIC_VARIANTS:
        raise ValueError("Unknown entry variant")
    profile = "strength" if variant.startswith("strength") else "legacy"
    needs_confirmation = not variant.endswith("open")
    mode = (
        variant.removeprefix("ranked_")
        if variant.startswith("ranked_no_")
        else "adaptive"
        if variant == "strength_adaptive"
        else "strict"
    )
    planned, audit, bounds = {}, [], {}
    daily = features.partition_by("date", as_dict=True) if not features.is_empty() else {}
    for day, sectors in signals.items():
        frame = daily.get((day,), pl.DataFrame())
        for sector in sectors:
            record = dict(
                signal_date=str(day),
                sector=sector["sector"],
                variant=variant,
                status="no_candidate",
                symbol=None,
                name=None,
                score=None,
                checks=[],
            )
            members = (
                frame.filter(pl.col("sector") == sector["sector"]).to_dicts()
                if not frame.is_empty()
                else []
            )
            evaluated = [evaluate_stock(r, profile=profile) for r in members]
            ranked = sorted(
                [r for r in evaluated if r["eligible"]], key=lambda r: (-r["score"], r["symbol"])
            )
            record["filter_reasons"] = dict(
                Counter(reason for r in evaluated for reason in r["reasons"])
            )
            record["selection_conditions"] = (
                summarize_conditions([r["conditions"] for r in evaluated])
                if not variant.startswith("leader")
                else []
            )
            record["shortlist"] = [
                {
                    k: r.get(k)
                    for k in (
                        "symbol",
                        "name",
                        "score",
                        "components",
                        "ret_5d",
                        "position20",
                        "atr_pct",
                        "support",
                        "relative5",
                        "conditions",
                    )
                }
                for r in ranked[:3]
            ]
            record["rejected"] = [
                {k: r.get(k) for k in ("symbol", "name", "score", "conditions", "reasons")}
                for r in sorted(
                    [r for r in evaluated if not r["eligible"]],
                    key=lambda r: (len(r["reasons"]), -(r["score"] or 0), r["symbol"]),
                )[:3]
            ]
            leader = (sector.get("leader_stocks") or [{}])[0]
            frozen = (
                (ranked[0] if ranked else None)
                if not variant.startswith("leader")
                else next((r for r in members if r["symbol"] == leader.get("symbol")), None)
            )
            if frozen is None:
                record["reasons"] = [
                    "没有合格标的" if not variant.startswith("leader") else "原龙头缺少行情"
                ]
                audit.append(record)
                continue
            record.update(
                symbol=frozen["symbol"],
                name=frozen.get("name")
                or (leader.get("name") if frozen["symbol"] == leader.get("symbol") else None),
                score=evaluate_stock(frozen, profile=profile)["score"],
                original_symbol=leader.get("symbol"),
            )
            decision_day = day
            if needs_confirmation:
                record["status"] = "expired"
                if not frozen.get("history_ok") or support(frozen) <= 0:
                    record.update(status="missing_data", reasons=["信号日支撑数据不足"])
                    audit.append(record)
                    continue
                decision_day = None
                for target in days[days.index(day) + 1 : days.index(day) + 3]:
                    current_frame = daily.get((target,), pl.DataFrame())
                    stock_rows = (
                        current_frame.filter(pl.col("symbol") == frozen["symbol"]).to_dicts()
                        if not current_frame.is_empty()
                        else []
                    )
                    rows, market = source.get(target, ([], None))
                    context = next((r for r in rows if r["sector"] == sector["sector"]), None)
                    checked = confirm(
                        frozen, stock_rows[0] if stock_rows else None, context, market, mode=mode
                    )
                    record["checks"].append({"date": str(target), **checked})
                    record["reasons"] = checked["reasons"]
                    if checked["status"] == "confirmed":
                        decision_day = target
                        record["confirmation_date"] = str(target)
                        break
                    if checked["status"] == "invalidated":
                        record["status"] = "invalidated"
                        break
                if decision_day is None:
                    if record["status"] != "invalidated" and any(
                        c["status"] == "missing_data" for c in record["checks"]
                    ):
                        record["status"] = "missing_data"
                    audit.append(record)
                    continue
            next_index = days.index(decision_day) + 1
            if next_index >= len(days):
                record.update(status="missing_data", reasons=["没有下一个交易日"])
                audit.append(record)
                continue
            chosen = {
                **sector,
                "leader_stocks": [{"symbol": frozen["symbol"], "name": record["name"]}],
            }
            record.update(
                status="scheduled",
                decision_date=str(decision_day),
                scheduled_entry_date=str(days[next_index]),
            )
            if (str(decision_day), frozen["symbol"]) in bounds:
                record.update(status="merged", reasons=["同日同股计划合并"])
                audit.append(record)
                continue
            planned.setdefault(decision_day, []).append(chosen)
            # All arms reject a suspended/missing next session instead of silently delaying entry.
            bound = {"execution_date": str(days[next_index])}
            if needs_confirmation:
                current = (
                    daily[(decision_day,)]
                    .filter(pl.col("symbol") == frozen["symbol"])
                    .to_dicts()[0]
                )
                bound.update(
                    floor=max(support(frozen), current["close"] * 0.97),
                    ceiling=current["close"] * 1.02,
                )
            bounds[(str(decision_day), frozen["symbol"])] = bound
            audit.append(record)
    return planned, audit, bounds


def apply_opening_guards(matrix, bounds):
    rejected = []
    allowed = matrix.entry.copy()
    for time_id, asset_id in zip(*np.nonzero(matrix.entry), strict=True):
        signal_time = int(matrix.entry_signal_time[time_id, asset_id])
        key = (matrix.timestamp_labels[signal_time][:10], matrix.symbols[asset_id])
        bound = bounds.get(key)
        actual_day = matrix.timestamp_labels[time_id][:10]
        price = number(matrix.open[time_id, asset_id], None)
        reason = None
        if bound is None or actual_day != bound["execution_date"]:
            reason = "次日缺行情或停牌"
        elif price is None or price <= 0:
            reason = "开盘价无效"
        elif ("floor" in bound and price < bound["floor"]) or (
            "ceiling" in bound and price > bound["ceiling"]
        ):
            reason = "开盘超出入场区间"
        if reason:
            allowed[time_id, asset_id] = False
            rejected.append(
                dict(signal_date=key[0], symbol=key[1], date=actual_day, open=price, reason=reason)
            )
    allowed.setflags(write=False)
    return allowed, rejected


def run_comparison(
    bars,
    signals,
    source,
    days,
    folds,
    capital,
    stop_loss,
    matcher,
    progress,
    *,
    name_map=None,
    variants=VARIANTS,
):
    features = prepare_features(bars, days, name_map=name_map)
    comparisons, records = [], []
    for phase in ("walk_forward", "holdout"):
        test_dates = sorted(d for f in folds if f["kind"] == phase for d in f["test"])
        if not test_dates:
            continue
        selected = {d: signals[d] for d in test_dates}
        dates = days[days.index(test_dates[0]) : days.index(test_dates[-1]) + 9]
        for variant in variants:
            planned, audit, bounds = plan_entries(features, selected, source, days, variant)
            for horizon in (3, 5):
                result = matcher(
                    bars,
                    planned,
                    dates,
                    horizon,
                    capital,
                    stop_loss_pct=stop_loss,
                    opening_bounds=bounds,
                )
                trades = result.get("trades", [])
                returns = [t["pnl_pct"] for t in trades]
                for record in audit:
                    if record["status"] != "scheduled":
                        continue
                    trade = next(
                        (
                            t
                            for t in trades
                            if t["symbol"] == record["symbol"]
                            and str(t.get("entry_signal_date"))[:10] == record["decision_date"]
                        ),
                        None,
                    )
                    blocked = next(
                        (
                            r
                            for r in result.get("opening_rejections", [])
                            if r["symbol"] == record["symbol"]
                            and r["signal_date"] == record["decision_date"]
                        ),
                        None,
                    )
                    outcome = (
                        {"status": "closed", **trade}
                        if trade
                        else {"status": "opening_blocked", **blocked}
                        if blocked
                        else {"status": "not_filled", "reason": "未成交或未平仓, 查看撮合约束"}
                    )
                    record.setdefault("executions", {})[str(horizon)] = outcome
                comparisons.append(
                    dict(
                        phase=phase,
                        variant=variant,
                        horizon=horizon,
                        issued=len(audit),
                        scheduled=sum(r["status"] == "scheduled" for r in audit),
                        status_counts=dict(Counter(r["status"] for r in audit)),
                        opening_rejections=len(result.get("opening_rejections", [])),
                        stop_count=sum(t["exit_reason"] == "stop_loss" for t in trades),
                        win_rate=mean(r > 0 for r in returns) if returns else None,
                        mean_trade_return=mean(returns) if returns else None,
                        condition_diagnostics={
                            "selection": combine_summaries(
                                [r.get("selection_conditions", []) for r in audit]
                            ),
                            "confirmation": summarize_conditions(
                                [c.get("conditions", []) for r in audit for c in r["checks"]]
                            ),
                        },
                        trading=result,
                    )
                )
            records.extend({**r, "phase": phase} for r in audit)
            progress("sector_entry", 96, f"选股与入场撮合 {phase} {variant}")
    return comparisons, records
