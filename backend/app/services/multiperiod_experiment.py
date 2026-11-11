"""Paired daily/60m/15m research. No production-model promotion or orders."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from app.parquet import scan_enriched_parquet
from app.services import stockformer
from app.services.l2_experiment import BASELINE_FEATURE_COLUMNS, PANEL_COLUMNS, _finite, _metrics, _valid_bar, prepare_daily_research
from app.services.minute_timeframes import aggregate_minutes

VERSION = "daily_60m_15m_research_v1"
MINUTE_FEATURES = tuple(f"m{p}_{f}" for p in (60, 15) for f in ("last_return", "range_position", "realized_vol", "late_volume_ratio"))


@dataclass(frozen=True)
class MultiPeriodConfig:
    history_start: date
    train_start: date
    validation_start: date
    test_start: date
    end: date
    max_symbols: int = 50
    min_history_days: int = 60
    symbols: tuple[str, ...] = ()
    ridge_alpha: float = 1.0
    top_k: int = 5
    horizons: tuple[int, ...] = (1, 3, 5, 10)

    def __post_init__(self):
        if not self.history_start < self.train_start < self.validation_start < self.test_start < self.end:
            raise ValueError("训练、验证、测试日期必须依次递增")
        if not 20 <= self.max_symbols <= 200 or self.ridge_alpha <= 0 or not np.isfinite(self.ridge_alpha):
            raise ValueError("股票数须为20-200且正则强度必须为正数")


def daily_minute_features(frame: pl.DataFrame, day: date) -> pl.DataFrame:
    """Within-session ratios only: no raw-minute/adjusted-daily price mixing."""
    parts = []
    for period in (60, 15):
        bars = aggregate_minutes(frame, period, datetime.combine(day, time(15)))
        if bars.is_empty():
            return pl.DataFrame()
        bars = bars.with_columns((pl.col("close") / pl.col("open") - 1).alias("_return"))
        features = bars.group_by("symbol").agg(
            pl.col("complete").all().alias("_complete"), pl.len().alias("_n"),
            pl.col("_return").last().alias(f"m{period}_last_return"),
            ((pl.col("close").last() - pl.col("low").min()) /
             (pl.col("high").max() - pl.col("low").min()).clip(lower_bound=1e-8)).alias(f"m{period}_range_position"),
            (pl.col("_return").pow(2).sum().sqrt()).alias(f"m{period}_realized_vol"),
            (pl.col("volume").last() / pl.col("volume").mean().clip(lower_bound=1.)).alias(f"m{period}_late_volume_ratio"),
        ).filter(pl.col("_complete") & (pl.col("_n") == 240 // period)).drop("_complete", "_n")
        parts.append(features)
    return parts[0].join(parts[1], on="symbol", how="inner").with_columns(pl.lit(day).alias("date"))


def evaluate_multiperiod(panel: pl.DataFrame, minute_features: pl.DataFrame, config: MultiPeriodConfig):
    baseline, selected, calendar, _, _, symbols, _ = prepare_daily_research(panel, config)
    if not {"symbol", "date", *MINUTE_FEATURES}.issubset(minute_features.columns):
        raise ValueError("缺少通过校验的60/15分钟特征")
    if minute_features.select(pl.struct("symbol", "date").is_duplicated().any()).item():
        raise ValueError("重复的股票日分钟特征")
    common = baseline.join(minute_features, on=["symbol", "date"], how="inner").filter(_finite(MINUTE_FEATURES))
    arms = {"daily": BASELINE_FEATURE_COLUMNS,
            "daily_60m": (*BASELINE_FEATURE_COLUMNS, *[c for c in MINUTE_FEATURES if c.startswith("m60_")]),
            "daily_15m": (*BASELINE_FEATURE_COLUMNS, *[c for c in MINUTE_FEATURES if c.startswith("m15_")]),
            "daily_60m_15m": (*BASELINE_FEATURE_COLUMNS, *MINUTE_FEATURES)}
    prices = selected.filter(_valid_bar()).select("symbol", "date", "open")
    common = common.join(calendar.with_columns(pl.col("date").shift(-1).alias("entry_date")), on="date")
    common = common.join(prices.rename({"date": "entry_date", "open": "entry_open"}), on=["symbol", "entry_date"], how="left")
    reports, predictions = {}, []
    for horizon in config.horizons:
        frame = common.join(calendar.with_columns(pl.col("date").shift(-(horizon + 1)).alias("label_end")), on="date")
        frame = frame.join(prices.rename({"date": "label_end", "open": "exit_open"}), on=["symbol", "label_end"], how="left")
        frame = frame.with_columns((pl.col("exit_open") / pl.col("entry_open") - 1).alias("target"))
        splits, counts = {}, {}
        for name, start, end in (("train", config.train_start, config.validation_start),
                                 ("validation", config.validation_start, config.test_start),
                                 ("test", config.test_start, config.end + timedelta(days=1))):
            candidate = frame.filter((pl.col("date") >= start) & (pl.col("date") < end))
            part = candidate.filter((pl.col("label_end") < end) & _finite(["target", "entry_open", "exit_open"]))
            counts[name] = {"candidate_rows": candidate.height, "usable_rows": part.height,
                            "excluded_boundary_or_endpoint": candidate.height - part.height,
                            "dates": part["date"].n_unique()}
            if part["date"].n_unique() < (30 if name == "train" else 10) or part["symbol"].n_unique() < 3:
                raise ValueError(f"{horizon}日 {name} 有效样本不足: {counts[name]}")
            splits[name] = part
        models = {}
        for arm, columns in arms.items():
            training = splits["train"]
            w, scaler = stockformer._fit_ridge(training.select(columns).to_numpy(), training["target"].to_numpy(), config.ridge_alpha, len(columns))
            if not np.isfinite(w).all():
                raise ValueError("模型系数无效")
            models[arm] = {"weights": w.tolist(), "scaler": scaler}
            for name, part in splits.items():
                predicted = stockformer._predict(part.select(columns).to_numpy(), w, scaler)
                if not np.isfinite(predicted).all():
                    raise ValueError("模型输出无效")
                splits[name] = part.with_columns(pl.Series(f"prediction_{arm}", predicted))
        metrics = {}
        for name, part in splits.items():
            metrics[name] = {arm: _metrics(part, f"prediction_{arm}", top_k=config.top_k, cost=.001) for arm in arms}
            predictions.append(part.select("symbol", "date", "entry_date", "label_end", "target", *[f"prediction_{a}" for a in arms])
                               .with_columns(pl.lit(name).alias("split"), pl.lit(horizon).alias("horizon")))
        reports[str(horizon)] = {"counts": counts, "metrics": metrics, "models": models}
    report = {"version": VERSION, "research_only": True, "production_model_changed": False,
              "config": asdict(config), "symbols": symbols, "features": arms,
              "coverage": {"baseline_rows": baseline.height, "common_rows": common.height,
                           "common_ratio": common.height / baseline.height if baseline.height else 0},
              "label": "次日开盘至h个交易日后开盘收益；按全市场交易日对齐，跨分界标签剔除",
              "limitations": ["日K基线为StockFormer的OHLCV子集，并非当前生产模型全部因子",
                              "同样本对照；固定训练前股票池，仍可能存在存活及分钟覆盖偏差",
                              "使用当前存储的复权版本，并非历史时点复权档案",
                              "Top5为重叠远期收益均值，不是组合净值或可成交回测；费用为单次10bp假设",
                              "验证和测试均不调参，不自动替换生产模型"], "horizons": reports}
    return stockformer._json_safe(report), pl.concat(predictions)


def run_multiperiod(repo, *, start: date, end: date, max_symbols: int = 50, on_progress=None) -> dict:
    def emit(stage, pct, msg):
        if on_progress:
            on_progress(stage, pct, msg)
    emit("multiperiod_daily", 1, "读取本地日K，固定训练前股票池")
    history_start = start - timedelta(days=180)
    panel = scan_enriched_parquet(str(repo.store.data_dir / "kline_daily_enriched" / "**" / "*.parquet"))
    panel = panel.filter(pl.col("date").is_between(history_start, end) & pl.col("symbol").str.contains(r"^(?:(?:00|30|60|68)\d{4}(?:\.(?:SH|SZ))?|(?:SH|SZ)(?:00|30|60|68)\d{4})$"))
    panel = panel.select(PANEL_COLUMNS).collect(engine="streaming")
    dates = sorted(panel.filter(pl.col("date").is_between(start, end))["date"].unique())
    if len(dates) < 100:
        raise ValueError("实验至少需要100个有效交易日，并另有60个交易日预热历史")
    config = MultiPeriodConfig(history_start, dates[0], dates[int(len(dates) * .6)], dates[int(len(dates) * .8)], dates[-1], max_symbols=max_symbols)
    _, _, _, _, _, _, eligible = prepare_daily_research(panel, config)
    # Fixed-seed sample uses pre-training eligibility only, not test-period turnover or returns.
    universe = eligible["symbol"].to_list()
    symbols = sorted(np.random.default_rng(20260907).choice(universe, min(max_symbols, len(universe)), replace=False).tolist())
    config = replace(config, symbols=tuple(symbols))
    features, exclusions = [], []
    for i, day in enumerate(dates):
        emit("multiperiod_minutes", 5 + int(i / len(dates) * 80), f"校验并聚合分钟K {day} ({i + 1}/{len(dates)}日，{len(symbols)}只)")
        valid, issues = repo.get_minute_quality(symbols, [day])
        exclusions.extend(issues)
        if not valid.is_empty():
            part = daily_minute_features(stockformer._normalize_symbol_column(valid), day)
            if not part.is_empty():
                features.append(part)
    if not features:
        raise ValueError("没有通过完整分钟及日K量价校验的样本")
    emit("multiperiod_fit", 88, "拟合四组模型并评估独立样本外")
    report, predictions = evaluate_multiperiod(panel, pl.concat(features), config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    root = repo.store.data_dir / "research" / "multiperiod" / run_id
    root.mkdir(parents=True, exist_ok=False)
    report.update({"run_id": run_id, "created_at": datetime.now().isoformat(), "excluded_stock_days": len(exclusions),
                   "sampling_seed": 20260907, "sampling": "训练前合格股票池固定种子抽样"})
    (root / "exclusions.json").write_text(json.dumps(stockformer._json_safe(exclusions), ensure_ascii=False), encoding="utf-8")
    predictions.write_parquet(root / "predictions.parquet")
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    emit("done", 100, f"多周期实验完成，保存于 research/multiperiod/{run_id}")
    return {"multiperiod_run_id": run_id, "coverage": report["coverage"], "production_model_changed": False}


def latest_report(data_dir: Path) -> dict:
    root = data_dir / "research" / "multiperiod"
    reports = sorted(root.glob("*/report.json"), reverse=True)
    if not reports:
        return {}
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    for item in report["horizons"].values():
        item.pop("models", None)
        for metrics in item["metrics"].values():
            for values in metrics.values():
                values.pop("daily", None)
    return report
