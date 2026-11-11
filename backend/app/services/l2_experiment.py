"""Offline, paired OHLCV vs OHLCV+L2 factor research, never a trading backtest.

Information date t uses OHLCV(t) and validated L2(t). Decision time is 09:25
China naive on the next global trading session; entry is that session's open,
exit is the open h sessions later (h >= 1). Endpoints must exist for the stock.
Splits are defined on information dates, with actual exit dates strictly before
the next split boundary. No production models, caches, APIs or files are changed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from app.parquet import scan_enriched_parquet
from app.services import stockformer

# Pin the reader's contract independently of an extractor upgrade.
L2_FEATURE_VERSION = "l2_daily_research_v1"
L2_SOURCE_CONTRACT = "wind_csv_sh_sz_research_v1"
BASELINE_HISTORY_BARS = 21

L2_FEATURE_COLUMNS = (
    "l2_trade_count",
    "l2_trade_volume",
    "l2_trade_amount",
    "l2_buy_amount_ratio",
    "l2_net_buy_amount_ratio",
    "l2_large_trade_amount_ratio",
    "l2_late_net_buy_ratio",
    "l2_late_amount_ratio",
    "l2_average_trade_amount",
    "l2_vwap_close_gap",
    "l2_depth_imbalance_mean",
    "l2_depth_imbalance_positive_fraction",
    "l2_spread_bps_mean",
)
BASELINE_FEATURE_COLUMNS = (
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ma20_gap",
    "vol_ratio_5d",
    "intraday_strength",
    "amplitude",
    "rsi_proxy",
    "risk_20d",
    "gap_open",
    "amount_ratio_5d",
    "price_position_20d",
    "price_range_10d",
)
PANEL_COLUMNS = ("symbol", "date", "open", "high", "low", "close", "volume", "amount")
LIMITATIONS = (
    "Pure forward-factor-return research, not an executable backtest or production model promotion.",
    "Baseline is the causal OHLCV subset of StockFormer features, NOT its production full-feature model; "
    "no moneyflow/news/industry/latest metadata or historical turnover fallbacks are used.",
    "Both labels and baseline OHLC use stored enriched forward-adjusted prices, never a mixture "
    "with raw prices. Current-vintage adjustments are not a point-in-time corporate-action archive. "
    "No full-window price repair is applied. L2 ratios retain their extractor definitions.",
    "Open-to-open labels allow overnight h=1 holding but do not simulate T+1 settlement, "
    "auction fills, price limits, suspensions during holding, delistings, lot sizes or capacity.",
    "The same fixed round-trip cost proxy is subtracted once per selected label in both models; "
    "it is not measured turnover, commissions or slippage. Gross returns are the training labels.",
    "Top-K figures are equal-day arithmetic averages of overlapping forward labels, not NAV, "
    "compounded returns, Sharpe or statistical significance. IC/RankIC are mean daily cross-sections.",
    "Universe is fixed using pre-sample history only; current stored coverage can still have "
    "survivorship bias. A missing whole-market session cannot be inferred from stored dates.",
    "Complete-case L2 filtering may introduce coverage bias; results only describe the common mask. "
    "Supplier B/S flags are a side proxy, not verified aggressor identity. Large executed trades "
    "use a fixed CNY 200000 threshold; amount ratios divide by all executed amount. "
    "Regular-session quote aggregates are snapshot-weighted, not time-weighted.",
    "Availability is the extractor's conservative next-calendar-day midnight convention, not "
    "verified historical delivery. Validation and test are diagnostics; no tuning or refit is performed.",
)


@dataclass(frozen=True)
class L2ExperimentConfig:
    """All date boundaries are information dates; end also caps label exit dates.

    Optional symbols are a predeclared candidate list, not an ex-post performance
    filter. Each must qualify using history_start <= date < train_start. Otherwise
    qualifying stocks are selected lexicographically, never by future liquidity.
    """

    history_start: date
    train_start: date
    validation_start: date
    test_start: date
    end: date
    horizons: tuple[int, ...] = (1, 3, 5, 10)
    max_symbols: int = 20
    min_history_days: int = 60
    symbols: tuple[str, ...] = ()
    l2_features: tuple[str, ...] = L2_FEATURE_COLUMNS
    ridge_alpha: float = 1.0
    top_k: int = 5
    round_trip_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        dates = (
            self.history_start,
            self.train_start,
            self.validation_start,
            self.test_start,
            self.end,
        )
        if not all(type(d) is date for d in dates) or not all(a < b for a, b in pairwise(dates)):
            raise ValueError(
                "Require chronological history_start < train < validation < test < end"
            )
        if (
            not self.horizons
            or len(set(self.horizons)) != len(self.horizons)
            or any(type(h) is not int or h not in (1, 3, 5, 10) for h in self.horizons)
        ):
            raise ValueError("horizons must be a unique subset of 1, 3, 5, 10")
        if (
            not self.l2_features
            or len(set(self.l2_features)) != len(self.l2_features)
            or any(name not in L2_FEATURE_COLUMNS for name in self.l2_features)
        ):
            raise ValueError(
                "l2_features must be a unique nonempty subset of the explicit whitelist"
            )
        for name in ("max_symbols", "min_history_days", "top_k"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be finite and positive")
        if not math.isfinite(self.round_trip_cost_bps) or self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must be finite and nonnegative")


@dataclass
class L2ExperimentResult:
    report: dict
    predictions: pl.DataFrame


def _keys(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    if not {"symbol", "date"}.issubset(frame.columns):
        raise ValueError(f"{name}: missing symbol/date")
    if frame.schema["date"] == pl.String:
        frame = frame.with_columns(pl.col("date").str.to_date("%Y-%m-%d", strict=True))
    if frame.schema["date"] != pl.Date or frame.schema["symbol"] != pl.String:
        raise ValueError(f"{name}: date must be Date or ISO date; symbol must be String")
    # Check exchange identity before the shared helper discards the exchange.
    # 000001.SH is an index, never an alias of the equity 000001.SZ. This L2
    # contract covers SH/SZ equities only, not BJ, ETFs or indices.
    valid_symbol = pl.col("symbol").str.contains(
        r"^(?:(?:00|30)\d{4}(?:\.SZ)?|(?:60|68)\d{4}(?:\.SH)?|"
        r"SZ(?:00|30)\d{4}|SH(?:60|68)\d{4})$"
    )
    if frame.filter(~valid_symbol.fill_null(False) | pl.col("date").is_null()).height:
        raise ValueError(f"{name}: invalid SH/SZ A-equity symbol/date or exchange identity")
    frame = stockformer._normalize_symbol_column(frame)
    if frame.select(pl.struct("symbol", "date").is_duplicated().any()).item():
        raise ValueError(f"{name}: duplicate symbol/date")
    return frame.sort(["symbol", "date"])


def _finite(columns: tuple[str, ...] | list[str]) -> pl.Expr:
    return pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in columns])


def _valid_bar() -> pl.Expr:
    return (
        _finite(PANEL_COLUMNS[2:])
        & pl.all_horizontal([pl.col(c) > 0 for c in ("open", "high", "low", "close")])
        & (pl.col("high") >= pl.max_horizontal("open", "close", "low"))
        & (pl.col("low") <= pl.min_horizontal("open", "close", "high"))
        & (pl.col("volume") >= 0)
        & (pl.col("amount") >= 0)
    ).fill_null(False)


def _l2_schema(schema: pl.Schema, features: tuple[str, ...]) -> None:
    required = {"symbol", "date", "quality_status", "available_at", *features}
    missing = required - set(schema.names())
    if missing:
        raise ValueError(f"L2: missing columns {sorted(missing)}")
    if any(schema[name] != pl.Float64 for name in features):
        raise ValueError("L2: whitelisted features must be Float64")
    dtype = schema["available_at"]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone is not None:
        raise ValueError("L2: available_at must be China-naive Datetime")
    if schema["quality_status"] != pl.String:
        raise ValueError("L2: quality_status must be String")


def _manifest_snapshot(path: Path, day: date) -> tuple[dict, bytes]:
    manifest_path = path.parents[2] / "extraction" / f"{day}.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"L2 manifest missing or unreadable for {day}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or manifest.get("date") != str(day)
    ):
        raise ValueError(f"L2 manifest must be complete and match partition date {day}")
    signature = manifest.get("signature")
    if not isinstance(signature, dict) or signature.get("feature_version") != L2_FEATURE_VERSION:
        raise ValueError(f"L2 manifest has unsupported feature_version for {day}")
    # Original manifests have no source_contract in signature; the hashed row
    # metadata is mandatory in either format. Reject an explicit incompatible one.
    if "source_contract" in signature and signature["source_contract"] != L2_SOURCE_CONTRACT:
        raise ValueError(f"L2 manifest has unsupported source_contract for {day}")
    symbols = signature.get("symbols")
    if not isinstance(symbols, list) or not symbols or any(not isinstance(s, str) for s in symbols):
        raise ValueError(f"L2 manifest has invalid signature symbols for {day}")
    if type(manifest.get("accepted")) is not int or manifest["accepted"] <= 0:
        raise ValueError(f"L2 manifest has no accepted rows for partition {day}")
    # Daily aggregates are small. Scan the exact verified bytes, not a path that
    # could be replaced between hashing and collection; never load raw archives.
    payload = path.read_bytes()
    if manifest.get("output_sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"L2 manifest SHA-256 mismatch for {day}")
    if manifest.get("output_bytes") != len(payload):
        raise ValueError(f"L2 manifest output size mismatch for {day}")
    return manifest, payload


def load_l2_daily(
    data_dir: Path, config: L2ExperimentConfig, *, research_dir: Path | None = None
) -> pl.DataFrame:
    """Read compact daily partitions one at a time; never read raw L2 archives.

    Strict feature projection excludes all quality/provenance numeric columns.
    A complete extraction/<day>.json manifest and pinned row provenance are
    mandatory. Orphans, changed bytes, dates and unsupported versions fail closed.
    SHA-256 provides integrity relative to the manifest, not source authentication.
    """
    parts = []
    root = (research_dir or Path(data_dir) / "l2_research") / "daily"
    for path in sorted(root.glob("date=*/part.parquet")):
        day = date.fromisoformat(path.parent.name.removeprefix("date="))
        if not config.train_start <= day <= config.end:
            continue
        manifest, payload = _manifest_snapshot(path, day)
        lazy = pl.scan_parquet(payload, hive_partitioning=False)
        schema = lazy.collect_schema()
        _l2_schema(schema, config.l2_features)
        if any(schema.get(c) != pl.String for c in ("feature_version", "source_contract")):
            raise ValueError(f"L2 provenance columns missing or invalid for {day}")
        part = lazy.select(
            "symbol",
            "date",
            "quality_status",
            "available_at",
            *config.l2_features,
            "feature_version",
            "source_contract",
        ).collect(engine="streaming")
        part = _keys(part, "L2")
        if part.filter(pl.col("date") != day).height:
            raise ValueError("L2 partition date differs from row date")
        valid_provenance = (
            (pl.col("feature_version") == L2_FEATURE_VERSION)
            & (pl.col("source_contract") == L2_SOURCE_CONTRACT)
            & (pl.col("quality_status") == "validated")
        ).fill_null(False)
        if part.filter(~valid_provenance).height:
            raise ValueError(f"L2 provenance version or validation status mismatch for {day}")
        manifest_symbols = _keys(
            pl.DataFrame(
                {
                    "symbol": manifest["signature"]["symbols"],
                    "date": [day] * len(manifest["signature"]["symbols"]),
                }
            ),
            "L2 manifest symbols",
        )["symbol"]
        if part.height != manifest["accepted"] or not set(part["symbol"]).issubset(
            manifest_symbols
        ):
            raise ValueError(f"L2 manifest row count or symbols mismatch for {day}")
        parts.append(
            part.drop("feature_version", "source_contract").with_columns(
                pl.col("available_at").cast(pl.Datetime("us"))
            )
        )
    if not parts:
        raise ValueError("No L2 daily partitions in the requested information-date sample")
    return pl.concat(parts)


def prepare_daily_research(panel: pl.DataFrame, config):
    """Shared causal OHLCV baseline and pre-sample universe for paired experiments."""
    missing = set(PANEL_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"OHLCV: missing columns {sorted(missing)}")
    panel = _keys(panel.select(PANEL_COLUMNS), "OHLCV").filter(
        pl.col("date").is_between(config.history_start, config.end)
    )
    numeric = list(PANEL_COLUMNS[2:])
    if any(not panel.schema[c].is_numeric() for c in numeric):
        raise ValueError("OHLCV: prices/volume/amount must be numerical")
    panel = panel.with_columns(pl.col(numeric).cast(pl.Float64))
    calendar = panel.select("date").unique().sort("date")
    if calendar.is_empty():
        raise ValueError("No OHLCV history in requested dates")
    valid_bar = _valid_bar()
    history = panel.filter((pl.col("date") < config.train_start) & valid_bar)
    eligible = (
        history.group_by("symbol")
        .len()
        .filter(pl.col("len") >= config.min_history_days)
        .sort("symbol")
    )
    if config.symbols:
        candidates = _keys(
            pl.DataFrame(
                {
                    "symbol": list(config.symbols),
                    "date": [config.history_start] * len(config.symbols),
                }
            ),
            "universe",
        )["symbol"].to_list()
        if len(candidates) > config.max_symbols or not set(candidates).issubset(eligible["symbol"]):
            raise ValueError("Explicit universe exceeds max_symbols or lacks pre-sample history")
        symbols = sorted(candidates)
    else:
        symbols = eligible.head(config.max_symbols)["symbol"].to_list()
    if not symbols:
        raise ValueError("No fixed-universe stocks with sufficient pre-sample history")
    selected = panel.filter(pl.col("symbol").is_in(symbols))
    # Explicit null sessions keep both rolling features and labels off compressed
    # per-stock calendars. Never forward-fill missing prices or L2 observations.
    aligned = (
        pl.DataFrame({"symbol": symbols})
        .join(calendar, how="cross")
        .join(
            selected,
            on=["symbol", "date"],
            how="left",
        )
        .sort(["symbol", "date"])
        .with_columns(valid_bar.alias("_valid_bar"))
        .with_columns(
            (
                pl.col("_valid_bar")
                .cast(pl.UInt32)
                .rolling_sum(BASELINE_HISTORY_BARS, min_samples=BASELINE_HISTORY_BARS)
                .over("symbol")
                == BASELINE_HISTORY_BARS
            )
            .fill_null(False)
            .alias("_valid_history")
        )
    )
    clean = aligned.with_columns(
        [pl.when(valid_bar).then(pl.col(c)).otherwise(None).alias(c) for c in numeric]
    )
    baseline = (
        stockformer._feature_frame(clean, include_targets=False)
        .select("symbol", "date", *BASELINE_FEATURE_COLUMNS)
        .join(
            aligned.select("symbol", "date", "_valid_history"),
            on=["symbol", "date"],
            how="left",
        )
        .filter(pl.col("date").is_between(config.train_start, config.end))
    )
    baseline_rows = baseline.height
    history_rows = baseline.filter(pl.col("_valid_history")).height
    baseline = baseline.filter(pl.col("_valid_history") & _finite(BASELINE_FEATURE_COLUMNS))
    return baseline, selected, calendar, baseline_rows, history_rows, symbols, eligible


def _common_frame(panel: pl.DataFrame, l2: pl.DataFrame, config: L2ExperimentConfig):
    baseline, selected, calendar, baseline_rows, history_rows, symbols, eligible = prepare_daily_research(panel, config)
    _l2_schema(l2.schema, config.l2_features)
    l2 = _keys(l2, "L2").filter(pl.col("date").is_between(config.train_start, config.end))
    l2 = l2.filter(pl.col("symbol").is_in(symbols))
    l2_input_rows = l2.height
    validated = l2.filter(pl.col("quality_status") == "validated")
    midnight = pl.col("date").cast(pl.Datetime("us")) + pl.duration(days=1)
    if validated.filter(
        pl.col("available_at").is_null() | (pl.col("available_at") < midnight)
    ).height:
        raise ValueError("L2: validated available_at must be at least next-calendar-day 00:00")
    valid_l2 = validated.filter(_finite(config.l2_features)).select(
        "symbol",
        "date",
        "available_at",
        *config.l2_features,
    )
    timing = calendar.with_columns(pl.col("date").shift(-1).alias("entry_date"))
    frame = (
        baseline.join(timing, on="date", how="left")
        .with_columns(
            (pl.col("entry_date").cast(pl.Datetime("us")) + pl.duration(hours=9, minutes=25)).alias(
                "decision_at"
            ),
        )
        .join(valid_l2, on=["symbol", "date"], how="inner")
    )
    late_rows = frame.filter(pl.col("available_at") > pl.col("decision_at")).height
    frame = (
        frame.filter(pl.col("available_at") <= pl.col("decision_at"))
        .with_columns(pl.col("date").alias("l2_date"))
        .sort(["date", "symbol"])
    )
    coverage = {
        "panel_rows": panel.height,
        "candidate_rows": baseline_rows,
        "valid_history_rows": history_rows,
        "invalid_or_incomplete_history_rows": baseline_rows - history_rows,
        "baseline_feature_rows": baseline.height,
        "l2_rows": l2_input_rows,
        "l2_validated_rows": validated.height,
        "l2_finite_rows": valid_l2.height,
        "late_l2_rows": late_rows,
        "common_feature_rows": frame.height,
        "common_feature_coverage": frame.height / baseline.height if baseline.height else 0.0,
    }
    universe = {
        "symbols": symbols,
        "selection": "pre-sample history, explicit list or symbol ascending",
        "history_start": config.history_start,
        "history_end_exclusive": config.train_start,
        "min_history_days": config.min_history_days,
        "history_counts": dict(eligible.filter(pl.col("symbol").is_in(symbols)).iter_rows()),
    }
    return frame, selected, calendar, coverage, universe


def _metrics(frame: pl.DataFrame, prediction: str, *, top_k: int, cost: float) -> dict:
    """Daily IC averages and top-K label contrasts; no portfolio simulation."""
    pred, actual = frame[prediction].to_numpy(), frame["target"].to_numpy()
    daily = []
    for (day,), cross in frame.partition_by("date", as_dict=True, maintain_order=True).items():
        p, y = cross[prediction].to_numpy(), cross["target"].to_numpy()
        ranked = cross.sort([prediction, "symbol"], descending=[True, False])
        k = min(top_k, cross.height)
        top = float(ranked.head(k)["target"].mean())
        universe = float(cross["target"].mean())
        daily.append(
            {
                "date": day,
                "rows": cross.height,
                "k": k,
                "ic": stockformer._corr(p, y),
                "rank_ic": stockformer._corr(p, y, rank=True),
                "top_k_forward_return": top,
                "universe_forward_return": universe,
                "top_k_excess": top - universe,
                "top_k_after_cost_proxy": top - cost,
            }
        )

    def mean_metric(name):
        values = [row[name] for row in daily if row[name] is not None]
        return float(np.mean(values)) if values else None

    return {
        "rows": frame.height,
        "dates": len(daily),
        "symbols": frame["symbol"].n_unique(),
        "ic": mean_metric("ic"),
        "rank_ic": mean_metric("rank_ic"),
        "ic_days": sum(row["ic"] is not None for row in daily),
        "rank_ic_days": sum(row["rank_ic"] is not None for row in daily),
        "mse": float(np.mean((pred - actual) ** 2)),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "mae": float(np.mean(np.abs(pred - actual))),
        "direction_accuracy": float(np.mean((pred > 0) == (actual > 0))),
        "actual_positive_fraction": float(np.mean(actual > 0)),
        **{
            name: mean_metric(name)
            for name in (
                "top_k_forward_return",
                "universe_forward_return",
                "top_k_excess",
                "top_k_after_cost_proxy",
            )
        },
        "daily": daily,
    }


def evaluate_l2_experiment(
    panel: pl.DataFrame,
    l2: pl.DataFrame,
    config: L2ExperimentConfig,
) -> L2ExperimentResult:
    """Evaluate in memory. panel must contain the complete market date calendar.

    Future/held-out rows never select the universe or fit scalers. The only label
    mask is shared by the two arms. This function performs no I/O or model writes.
    """
    common, selected, calendar, coverage, universe = _common_frame(panel, l2, config)
    features = {
        "baseline": BASELINE_FEATURE_COLUMNS,
        "baseline_plus_l2": (*BASELINE_FEATURE_COLUMNS, *config.l2_features),
    }
    prices = selected.filter(_valid_bar()).select(
        "symbol",
        "date",
        "open",
    )
    common = common.join(
        prices.rename({"date": "entry_date", "open": "entry_open"}),
        on=["symbol", "entry_date"],
        how="left",
    )
    horizons, predictions, diagnostics = {}, [], []
    if len(universe["symbols"]) < 20:
        diagnostics.append("Weak sample: fewer than 20 fixed-universe stocks; IC may be unstable.")
    for horizon in config.horizons:
        exits = calendar.with_columns(pl.col("date").shift(-(horizon + 1)).alias("label_end"))
        frame = (
            common.join(exits, on="date", how="left")
            .join(
                prices.rename({"date": "label_end", "open": "exit_open"}),
                on=["symbol", "label_end"],
                how="left",
            )
            .with_columns((pl.col("exit_open") / pl.col("entry_open") - 1).alias("target"))
        )
        parts, counts = {}, {}
        for split, start, upper in (
            ("train", config.train_start, config.validation_start),
            ("validation", config.validation_start, config.test_start),
            ("test", config.test_start, None),
        ):
            candidate = frame.filter(
                (pl.col("date") >= start)
                & (pl.col("date") < upper if upper else pl.col("date") <= config.end)
            )
            # Count boundary purges even when the exit price is missing.
            boundary_ok = (
                pl.col("label_end") < upper if upper else pl.col("label_end") <= config.end
            )
            purged = candidate.filter((~boundary_ok).fill_null(True))
            within = candidate.filter(boundary_ok)
            usable = within.filter(_finite(["entry_open", "exit_open", "target"]))
            counts[split] = {
                "common_feature_rows": candidate.height,
                "purged_rows": purged.height,
                "missing_endpoint_rows": within.height - usable.height,
                "usable_rows": usable.height,
            }
            if usable.is_empty() or (split == "train" and usable.height < 2):
                raise ValueError(
                    f"Insufficient common samples for {horizon}d {split}: {counts[split]}; "
                    "extend dates/validated coverage, shorten horizon or provide more pre-sample history"
                )
            parts[split] = usable.with_columns(pl.lit(split).alias("split"))
            minimum = 60 if split == "train" else 20
            if usable["date"].n_unique() < minimum:
                diagnostics.append(
                    f"Weak sample: {horizon}d {split} has fewer than {minimum} dates."
                )
            if usable.height <= len(features["baseline_plus_l2"]) and split == "train":
                diagnostics.append(
                    f"Weak sample: {horizon}d training rows <= augmented feature count."
                )
        models = {}
        train = parts["train"]
        for arm, columns in features.items():
            # Do not use _prepare_xy: its percentiles are fitted on each input split.
            weights, scaler = stockformer._fit_ridge(
                train.select(columns).to_numpy(),
                train["target"].to_numpy(),
                config.ridge_alpha,
                n_features=len(columns),
            )
            if not all(
                np.isfinite(values).all()
                for values in (
                    weights,
                    scaler["mean"],
                    scaler["std"],
                )
            ):
                raise ValueError("Nonfinite ridge fit/scaler; review input magnitude/quality")
            models[arm] = {"weights": weights.tolist(), "scaler": scaler}
            for split, part in parts.items():
                pred = stockformer._predict(part.select(columns).to_numpy(), weights, scaler)
                if not np.isfinite(pred).all():
                    raise ValueError("Nonfinite model predictions; review input magnitude/quality")
                parts[split] = part.with_columns(pl.Series(f"prediction_{arm}", pred))
        splits = {}
        for split, part in parts.items():
            metrics = {
                arm: _metrics(
                    part,
                    f"prediction_{arm}",
                    top_k=config.top_k,
                    cost=config.round_trip_cost_bps / 10000.0,
                )
                for arm in features
            }
            for arm, values in metrics.items():
                if not values["ic_days"] or not values["rank_ic_days"]:
                    diagnostics.append(
                        f"Weak sample: {horizon}d {split} {arm} IC/RankIC unavailable; "
                        "cross-sections require >=3 stocks and nonconstant predictions/labels."
                    )
            contrast = {}
            for name in (
                "ic",
                "rank_ic",
                "mse",
                "rmse",
                "mae",
                "direction_accuracy",
                "top_k_forward_return",
                "top_k_after_cost_proxy",
                "top_k_excess",
            ):
                a, b = metrics["baseline"][name], metrics["baseline_plus_l2"][name]
                contrast[name] = b - a if a is not None and b is not None else None
            splits[split] = {
                "samples": counts[split],
                **metrics,
                "plus_l2_minus_baseline": contrast,
            }
            predictions.append(
                part.select(
                    "symbol",
                    "date",
                    "l2_date",
                    "available_at",
                    "decision_at",
                    "entry_date",
                    "label_end",
                    "entry_open",
                    "exit_open",
                    "target",
                    "split",
                    "prediction_baseline",
                    "prediction_baseline_plus_l2",
                ).with_columns(pl.lit(horizon).alias("horizon"))
            )
        horizons[str(horizon)] = {"models": models, "splits": splits}
    report = {
        "schema_version": 1,
        "research_only": True,
        "executable_backtest": False,
        "config": asdict(config),
        "universe": universe,
        "coverage": coverage,
        "features": features,
        "stockformer_feature_version": stockformer.FEATURE_VERSION,
        "l2_version_contract": {
            "feature_version": L2_FEATURE_VERSION,
            "source_contract": L2_SOURCE_CONTRACT,
        },
        "baseline_eligibility": "21 consecutive valid global-session OHLCV bars including t; "
        "gate computed before feature defaults; entry/exit require valid OHLCV",
        "label": "adjusted_open[t+h+1] / adjusted_open[t+1] - 1; global market sessions",
        "decision": "information date t; next global session 09:25 Asia/Shanghai naive",
        "purge": "actual label_end < next information-date split boundary; test label_end <= end",
        "normalization": "mean/std fitted only on purged common training rows; no quantile clipping",
        "model": "same StockFormer ridge alpha in both arms; train only; no tuning/refit",
        "direction": "prediction > 0 equals gross target > 0; zero is nonpositive",
        "contrast": "baseline_plus_l2 minus baseline; negative error deltas are improvements",
        "cost": {
            "round_trip_bps": config.round_trip_cost_bps,
            "unit": "decimal return",
            "training_labels": "gross; identical in both arms",
        },
        "diagnostics": diagnostics,
        "limitations": LIMITATIONS,
        "horizons": horizons,
    }
    return L2ExperimentResult(
        stockformer._json_safe(report),
        pl.concat(predictions).sort(["horizon", "date", "symbol"]),
    )


def load_l2_experiment_inputs(
    data_dir: Path,
    config: L2ExperimentConfig,
    *,
    research_dir: Path | None = None,
    extra_research_dirs: tuple[Path, ...] = (),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read only local standardized data. No extraction, providers or production writes."""
    # _load_panel repairs using full-window medians and attaches auxiliary latest
    # metadata. Use its existing storage utility without those noncausal steps.
    panel = (
        scan_enriched_parquet(str(Path(data_dir) / "kline_daily_enriched" / "**" / "*.parquet"))
        .filter(pl.col("date").is_between(config.history_start, config.end))
        # The shared daily store also contains BJ equities. They are outside
        # this SH/SZ L2 experiment, not malformed SH/SZ aliases. Keep wrong
        # exchange identities in scope so _keys rejects them explicitly.
        .filter(
            ~pl.col("symbol").str.contains(
                r"^(?:(?:43|83|87|88|92)\d{4}(?:\.BJ)?|BJ(?:43|83|87|88|92)\d{4})$"
            )
        )
        .select(PANEL_COLUMNS)
        .collect(engine="streaming")
    )
    l2 = pl.concat(
        [
            load_l2_daily(data_dir, config, research_dir=root)
            for root in (research_dir, *extra_research_dirs)
        ]
    )
    return panel, l2


def run_l2_experiment(
    data_dir: Path,
    config: L2ExperimentConfig,
    *,
    research_dir: Path | None = None,
    extra_research_dirs: tuple[Path, ...] = (),
) -> L2ExperimentResult:
    """Read local inputs and run the unchanged paired experiment."""
    panel, l2 = load_l2_experiment_inputs(
        data_dir, config, research_dir=research_dir, extra_research_dirs=extra_research_dirs
    )
    return evaluate_l2_experiment(panel, l2, config)
