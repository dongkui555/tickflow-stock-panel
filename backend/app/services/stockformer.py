"""StockFormer-style model management and factor research service.

This module deliberately exposes a stable StockFormer integration contract while
using a lightweight local linear rank model for the first implementation. The
saved model artifact can later be replaced by a real Transformer trainer without
changing API consumers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from app.parquet import scan_enriched_parquet, scan_parquet_compat

logger = logging.getLogger(__name__)

HORIZONS = (1, 3, 5, 10)
FEATURE_VERSION = "sfv8_minute_daily_features_v1"
MODEL_DATA_MODE = "daily_kline_plus_minute_aggregates"
MODEL_KIND = "stockformer_linear_rank_v2"
XGBOOST_MODEL_KIND = "stockformer_xgboost_rank_v1"
GBDT_ENSEMBLE_MODEL_KIND = "stockformer_gbdt_ensemble_v1"
FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ma20_gap",
    "vol_ratio_5d",
    "turnover_rate",
    "intraday_strength",
    "amplitude",
    "rsi_proxy",
    "risk_20d",
    "gap_open",
    "open_strength",
    "turnover_ratio_5d",
    "amount_ratio_5d",
    "limit_up_flag",
    "limit_down_flag",
    "broken_limit_up_flag",
    "limit_down_recovery_flag",
    "limit_up_streak",
    "limit_down_streak",
    "main_net_inflow_rate",
    "large_net_inflow_rate",
    "moneyflow_strength_3d",
    "moneyflow_strength_5d",
    "moneyflow_acceleration_3d",
    "moneyflow_turn_positive",
    "moneyflow_early_layout_score",
    "moneyflow_accumulation_score",
    "moneyflow_absorption_efficiency",
    "moneyflow_chase_risk",
    "moneyflow_positive_days_5d",
    "moneyflow_aging_risk",
    "price_chase_risk",
    "price_position_20d",
    "price_position_60d",
    "price_range_10d",
    "price_compression_score",
    "volume_absorption_score",
    "base_zone_score",
    "auction_amount_ratio",
    "auction_volume_ratio",
    "industry_ret_1d",
    "industry_heat_rank",
    "industry_limit_up_rate",
    "sector_ret_3d",
    "sector_ret_5d",
    "sector_up_ratio",
    "sector_amount_ratio",
    "sector_moneyflow_strength_5d",
    "sector_fund_layout_score",
    "sector_setup_score",
    "sector_pre_ret_5d",
    "sector_pre_up_ratio",
    "sector_pre_amount_ratio",
    "sector_pre_moneyflow_strength_5d",
    "sector_today_confirm_score",
    "sector_rotation_score",
    "sector_crowding_risk",
    "news_sentiment_score",
    "news_count_3d",
    "overseas_news_sentiment_score",
    "overseas_news_count_3d",
    "minute_session_coverage",
    "minute_bar_count",
    "minute_intraday_return",
    "minute_early_return",
    "minute_open_strength",
    "minute_intraday_volatility",
    "minute_early_amount_ratio",
    "minute_late_amount_ratio",
    "minute_volume_concentration",
    "minute_vwap_gap",
    "minute_close_position",
    # Explicit availability features. A missing source is different from a
    # real neutral observation and must be visible to both training and rank.
    "minute_data_available",
    "moneyflow_data_available",
    "auction_data_available",
    "news_data_available",
    "data_coverage_score",
]
TRAINING_PROFILES = ("daily_baseline", "minute_enhanced")
MINUTE_FEATURE_COLUMNS = tuple(column for column in FEATURE_COLUMNS if column.startswith("minute_"))
MINUTE_COVERAGE_COLUMNS = (
    "minute_data_available",
    "minute_bar_count",
    "minute_session_coverage",
)
# Retain the old floor only for explicitly reproduced historical experiments.
# New training, calibration and inference share the complete-session contract.
MINUTE_RESEARCH_MIN_VALID_BARS = 228
MINUTE_LIVE_MIN_VALID_BARS = 240
MINUTE_MIN_VALID_BARS = MINUTE_LIVE_MIN_VALID_BARS
MINUTE_AVAILABILITY_POLICY = {
    'version': 'complete_session_240_v1',
    'minimum_minute_bars': 240,
    'minimum_minute_coverage': 1.0,
    'required_available': 1.0,
    'missing_policy': 'exclude_without_imputation',
}

# The baseline deliberately stays on daily/enriched observations.  External
# minute, auction, money-flow and news fields are excluded so a missing
# auxiliary source cannot turn into a synthetic neutral observation in the
# baseline model.
DAILY_BASELINE_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_COLUMNS
    if not column.startswith("minute_")
    and column not in {
        "moneyflow_data_available", "auction_data_available", "news_data_available",
        "data_coverage_score", "main_net_inflow_rate", "large_net_inflow_rate",
        "moneyflow_strength_3d", "moneyflow_strength_5d", "moneyflow_acceleration_3d",
        "moneyflow_turn_positive", "moneyflow_early_layout_score", "moneyflow_accumulation_score",
        "moneyflow_absorption_efficiency", "moneyflow_chase_risk", "moneyflow_positive_days_5d",
        "moneyflow_aging_risk", "auction_amount_ratio", "auction_volume_ratio",
        "news_sentiment_score", "news_count_3d", "overseas_news_sentiment_score",
        "overseas_news_count_3d",
    }
)
DATA_PROVENANCE_COLUMNS = [
    "_minute_latest_date",
    "_moneyflow_latest_date",
    "_auction_latest_date",
    "_news_latest_date",
]
OPTIONAL_FEATURE_DEFAULTS = {
    "gap_open": 0.0,
    "open_strength": 0.0,
    "turnover_ratio_5d": 1.0,
    "amount_ratio_5d": 1.0,
    "limit_up_flag": 0.0,
    "limit_down_flag": 0.0,
    "broken_limit_up_flag": 0.0,
    "limit_down_recovery_flag": 0.0,
    "limit_up_streak": 0.0,
    "limit_down_streak": 0.0,
    "main_net_inflow_rate": 0.0,
    "large_net_inflow_rate": 0.0,
    "moneyflow_strength_3d": 0.0,
    "moneyflow_strength_5d": 0.0,
    "moneyflow_acceleration_3d": 0.0,
    "moneyflow_turn_positive": 0.0,
    "moneyflow_early_layout_score": 0.5,
    "moneyflow_accumulation_score": 0.5,
    "moneyflow_absorption_efficiency": 0.0,
    "moneyflow_chase_risk": 0.5,
    "moneyflow_positive_days_5d": 0.0,
    "moneyflow_aging_risk": 0.0,
    "price_chase_risk": 0.5,
    "price_position_20d": 0.5,
    "price_position_60d": 0.5,
    "price_range_10d": 0.0,
    "price_compression_score": 0.5,
    "volume_absorption_score": 0.5,
    "base_zone_score": 0.5,
    "auction_amount_ratio": 0.0,
    "auction_volume_ratio": 0.0,
    "industry_ret_1d": 0.0,
    "industry_heat_rank": 0.5,
    "industry_limit_up_rate": 0.0,
    "sector_ret_3d": 0.0,
    "sector_ret_5d": 0.0,
    "sector_up_ratio": 0.5,
    "sector_amount_ratio": 1.0,
    "sector_moneyflow_strength_5d": 0.0,
    "sector_fund_layout_score": 0.5,
    "sector_setup_score": 0.5,
    "sector_pre_ret_5d": 0.0,
    "sector_pre_up_ratio": 0.5,
    "sector_pre_amount_ratio": 1.0,
    "sector_pre_moneyflow_strength_5d": 0.0,
    "sector_today_confirm_score": 0.5,
    "sector_rotation_score": 0.5,
    "sector_crowding_risk": 0.5,
    "news_sentiment_score": 0.0,
    "news_count_3d": 0.0,
    "overseas_news_sentiment_score": 0.0,
    "overseas_news_count_3d": 0.0,
    "minute_session_coverage": 0.0,
    "minute_bar_count": 0.0,
    "minute_intraday_return": 0.0,
    "minute_early_return": 0.0,
    "minute_open_strength": 0.0,
    "minute_intraday_volatility": 0.0,
    "minute_early_amount_ratio": 0.0,
    "minute_late_amount_ratio": 0.0,
    "minute_volume_concentration": 0.0,
    "minute_vwap_gap": 0.0,
    "minute_close_position": 0.5,
    "minute_data_available": 0.0,
    "moneyflow_data_available": 0.0,
    "auction_data_available": 0.0,
    "news_data_available": 0.0,
    "data_coverage_score": 0.0,
}
SCORE_FEATURE_DEFAULTS = {
    "amount": 0.0,
    "ma20_gap": 0.0,
    "ret_20d": 0.0,
    "ret_3d": 0.0,
    "vol_ratio_5d": 1.0,
    "intraday_strength": 0.0,
    "risk_20d": 0.0,
    "amplitude": 0.0,
    "turnover_rate": 0.0,
    **OPTIONAL_FEATURE_DEFAULTS,
}
PREDICTION_FEATURE_DEFAULTS = {
    "turnover_rate": 0.0,
    **OPTIONAL_FEATURE_DEFAULTS,
}
MULTIFACTOR_WEIGHTS = {
    "technical_score": 0.25,
    "stockformer_score": 0.20,
    "fund_score": 0.20,
    "sector_score": 0.15,
    "sentiment_score": 0.10,
    "risk_score": 0.07,
    "fundamental_score": 0.03,
}


def _env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        value = int(str(os.getenv(name, "")).strip() or default)
    except Exception:
        value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


@dataclass
class StockFormerTrainConfig:
    start: date | None = None
    end: date | None = None
    sample_out_ratio: float = 0.30
    max_symbols: int = 1000
    ridge_alpha: float = 1.0
    # The model ranks stocks by forward return relative to their industry.
    # Gross industry-relative labels stay unshifted; net labels carry an
    # explicit cost version and remain distinct from portfolio returns.
    target_mode: str = "industry_relative"
    training_profile: str = "daily_baseline"
    # Explicit research opt-in; historical/default labels retain their version.
    execution_costs: dict | None = None


def training_feature_columns(profile: str) -> list[str]:
    """Return the immutable feature contract for a training profile."""
    normalized = str(profile or "").strip().lower()
    if normalized == "daily_baseline":
        return list(DAILY_BASELINE_FEATURE_COLUMNS)
    if normalized == "minute_enhanced":
        return list(FEATURE_COLUMNS)
    raise ValueError(f"training_profile 只支持: {', '.join(TRAINING_PROFILES)}")


def prepare_training_features(
    features: pl.DataFrame,
    profile: str,
    *,
    minimum_minute_bars: int | None = None,
    minimum_minute_coverage: float | None = None,
) -> tuple[pl.DataFrame, dict[str, int | float | str]]:
    """Apply a profile gate without filling missing minute observations.

    All new stages default to the same complete session. Explicit lower bounds
    exist only to reproduce recorded historical research, not paired models.
    """
    normalized = str(profile or "").strip().lower()
    if normalized not in TRAINING_PROFILES:
        raise ValueError(f"training_profile 只支持: {', '.join(TRAINING_PROFILES)}")
    total = features.height
    if normalized == "daily_baseline":
        return features, {
            "profile": normalized,
            "total_rows": total,
            "ready_rows": total,
            "incomplete_rows": 0,
            "missing_rows": 0,
        }
    work = features
    for column in MINUTE_COVERAGE_COLUMNS:
        if column not in work.columns:
            work = work.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    available = pl.col("minute_data_available").cast(pl.Float64, strict=False).fill_null(0.0)
    bars = pl.col("minute_bar_count").cast(pl.Float64, strict=False).fill_null(0.0)
    coverage = pl.col("minute_session_coverage").cast(pl.Float64, strict=False).fill_null(0.0)
    required_bars = int(
        minimum_minute_bars
        if minimum_minute_bars is not None
        else MINUTE_AVAILABILITY_POLICY['minimum_minute_bars']
    )
    required_coverage = float(
        minimum_minute_coverage
        if minimum_minute_coverage is not None
        else MINUTE_AVAILABILITY_POLICY['minimum_minute_coverage']
    )
    ready = (available.is_finite() & (available == 1.0) & bars.is_finite()
             & (bars >= required_bars) & coverage.is_finite()
             & coverage.is_between(required_coverage, 1.0))
    present = bars > 0
    return work.filter(ready), {
        "profile": normalized,
        "total_rows": total,
        "ready_rows": int(work.filter(ready).height),
        "incomplete_rows": int(work.filter(present & ~ready).height),
        "missing_rows": int(work.filter(~present).height),
        "minimum_minute_bars": required_bars,
        "minimum_minute_coverage": required_coverage,
        "minute_policy_version": (
            MINUTE_AVAILABILITY_POLICY['version']
            if required_bars == 240 and required_coverage == 1.0 else 'explicit_legacy_research_bounds'
        ),
    }


@dataclass
class StockFormerModelMeta:
    model_id: str
    model_name: str
    model_kind: str
    feature_version: str
    trained_at: str
    train_start: str
    train_end: str
    oos_start: str
    oos_end: str
    horizons: list[int]
    feature_columns: list[str]
    model_file: str
    factor_file: str
    prediction_file: str
    oos_metrics: dict
    config: dict
    n_symbols: int
    n_rows_train: int
    n_rows_oos: int
    elapsed_ms: float


def _stockformer_dir(data_dir: Path) -> Path:
    path = data_dir / "stockformer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _models_dir(data_dir: Path) -> Path:
    path = _stockformer_dir(data_dir) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_path(data_dir: Path) -> Path:
    return _stockformer_dir(data_dir) / "models.json"


def _latest_path(data_dir: Path) -> Path:
    return _stockformer_dir(data_dir) / "latest_model.txt"


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _json_safe(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if isinstance(value, numbers.Integral):
            return int(value)
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value, key=lambda x: str(x))]
    return value


def _pct_rank_expr(name: str) -> pl.Expr:
    return (pl.col(name).rank(method="average").over("date") / pl.len().over("date")).fill_null(0.5)


def _score_from_feature(name: str, *, invert: bool = False) -> pl.Expr:
    rank = _pct_rank_expr(name)
    if invert:
        rank = 1.0 - rank
    return (rank * 100).clip(0, 100)


def _compact_symbol_text(value) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    for prefix in ("SH", "SZ", "BJ"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return digits[-6:].zfill(6)
    return text


def _compact_symbol_set(symbols: list[str] | None) -> set[str]:
    return {code for code in (_compact_symbol_text(symbol) for symbol in (symbols or [])) if code}


def _symbol_in_filter(symbols: list[str] | None) -> pl.Expr:
    wanted = sorted(_compact_symbol_set(symbols))
    if not wanted:
        return pl.lit(False)
    symbol = pl.col("symbol").cast(pl.Utf8, strict=False)
    return symbol.is_in(wanted) | symbol.map_elements(_compact_symbol_text, return_dtype=pl.Utf8).is_in(wanted)


def _infer_exchange_text(code: str) -> str:
    if code.startswith("6"):
        return "SH"
    if code.startswith(("0", "3")):
        return "SZ"
    if code.startswith(("4", "8")):
        return "BJ"
    return ""


def _symbol_aliases(symbol: object, code: object = None, exchange: object = None) -> list[str]:
    aliases: list[str] = []

    def add(value: object) -> None:
        text = str(value or "").strip().upper()
        if text and text not in aliases:
            aliases.append(text)

    raw_exchange = str(exchange or "").strip().upper()
    exchange_text = raw_exchange.split(".", 1)[-1] if raw_exchange else ""
    for raw in (symbol, code):
        text = str(raw or "").strip().upper()
        if not text:
            continue
        add(text)
        compact = _compact_symbol_text(text)
        if not compact:
            continue
        add(compact)
        suffix = exchange_text
        if not suffix and "." in text:
            suffix = text.rsplit(".", 1)[-1]
        if not suffix:
            suffix = _infer_exchange_text(compact)
        if suffix:
            add(f"{compact}.{suffix}")
            add(f"{suffix}{compact}")
    return aliases


def _parquet_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [str(path) for path in sorted(root.rglob("*.parquet")) if path.is_file()]


def _parquet_schema_union(files: list[str]) -> tuple[list[str], dict[str, pl.DataType]]:
    """Return readable files and the union schema for additive partitions."""
    readable: list[str] = []
    schema: dict[str, pl.DataType] = {}
    for path in files:
        try:
            current = pl.read_parquet_schema(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("parquet schema read failed for %s: %s", path, exc)
            continue
        readable.append(path)
        for name, dtype in current.items():
            schema.setdefault(name, dtype)
    return readable, schema


def _read_filtered_parquet_root(
    root: Path,
    *,
    start: date | None = None,
    end: date | None = None,
    symbols: list[str] | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Read an auxiliary parquet tree with push-down filters.

    Research endpoints often need only a model universe and a bounded date
    range. Reading every partition into a materialized frame first can use
    many gigabytes when a user opens StockFormer. Keep the filter in the
    parquet scan so only the requested slice reaches Python memory.
    """
    files = _parquet_files(root)
    if not files:
        return pl.DataFrame()
    try:
        files, union_schema = _parquet_schema_union(files)
        if not files:
            return pl.DataFrame()
        try:
            lf = scan_parquet_compat(files, hive_partitioning=True, schema=union_schema)
        except TypeError:
            lf = scan_parquet_compat(files, schema=union_schema)
        schema_names = lf.collect_schema().names()
        date_col = next((name for name in ("date", "trade_date") if name in schema_names), None)
        symbol_col = next((name for name in ("symbol", "ts_code", "code") if name in schema_names), None)

        wanted = {
            _compact_symbol_text(value)
            for value in (symbols or [])
            if _compact_symbol_text(value)
        }
        if wanted and symbol_col:
            raw_symbol = pl.col(symbol_col).cast(pl.Utf8, strict=False).str.to_uppercase().str.strip_chars()
            symbol_key = (
                raw_symbol
                .str.replace(r"^(SZ|SH|BJ|HK)", "")
                .str.replace(r"\.(SZ|SH|BJ|HK)$", "")
            )
            lf = lf.filter(symbol_key.is_in(sorted(wanted)))

        if date_col and (start is not None or end is not None):
            date_expr = pl.col(date_col).cast(pl.Date, strict=False)
            if start is not None:
                lf = lf.filter(date_expr >= start)
            if end is not None:
                lf = lf.filter(date_expr <= end)

        selected = list(dict.fromkeys(columns or schema_names))
        if date_col and date_col not in selected:
            selected.append(date_col)
        selected = [name for name in selected if name in schema_names]
        frame = lf.select(selected).collect() if selected else lf.collect()
        frame = _normalize_symbol_column(frame)
        frame = _normalize_date_column(frame)
        if wanted and not frame.is_empty() and "symbol" in frame.columns:
            frame = frame.filter(
                pl.col("symbol")
                .map_elements(_compact_symbol_text, return_dtype=pl.Utf8)
                .is_in(sorted(wanted))
            )
        return frame
    except Exception as exc:  # noqa: BLE001
        logger.warning("filtered parquet read failed for %s: %s", root, exc)
        return pl.DataFrame()


def _latest_unavailable_reasons(data_dir: Path) -> dict[str, list[str]]:
    job_dir = data_dir / "job_store"
    if not job_dir.exists():
        return {}
    aliases = {
        "auction": "auction_open",
        "auction_open": "auction_open",
        "moneyflow": "moneyflow",
        "news": "news",
        "news_domestic": "news",
        "news_overseas": "news",
        "news_overseas_history": "news",
    }
    reasons: dict[str, list[str]] = {}
    try:
        paths = sorted(
            job_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:30]
    except Exception:
        return reasons
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            continue
        items = []
        for field in ("unavailable_datasets", "news_unavailable"):
            values = result.get(field)
            if isinstance(values, list):
                items.extend(values)
        for item in items:
            if not isinstance(item, dict):
                continue
            key = aliases.get(str(item.get("dataset") or ""))
            reason = str(item.get("reason") or "").strip()
            if key and reason:
                bucket = reasons.setdefault(key, [])
                if reason not in bucket:
                    bucket.append(reason)
        if all(reasons.get(key) for key in ("moneyflow", "auction_open", "news")):
            break
    return reasons


def _coverage_config_reasons(key: str) -> list[str]:
    reasons: list[str] = []
    if key in {"moneyflow", "auction_open"}:
        mode = str(os.getenv("TUSHARE_MODE", "official") or "official").strip().lower()
        has_promax = bool(os.getenv("TUSHARE_PROMAX_URL", "").strip() and os.getenv("TUSHARE_PROMAX_KEY", "").strip())
        if has_promax and mode != "promax":
            reasons.append("已配置 Promax 网关，但 TUSHARE_MODE 不是 promax，当前不会使用聚合网关。")
    if key == "news":
        if not os.getenv("TUSHARE_NEWS_SRC", "").strip():
            reasons.append("未显式配置 TUSHARE_NEWS_SRC，国内新闻将按 cls、sina、10jqka 回退链尝试。")
        if not os.getenv("ALPHAVANTAGE_API_KEY", "").strip():
            reasons.append("未配置 ALPHAVANTAGE_API_KEY，海外/美股历史新闻无法覆盖两年。")
    return reasons


def _coverage_file_stats(root: Path) -> dict:
    files = _parquet_files(root)
    stats: dict[str, Any] = {"files": len(files)}
    if not files:
        return stats
    try:
        files, union_schema = _parquet_schema_union(files)
        if not files:
            return stats
        try:
            lf = scan_parquet_compat(files, hive_partitioning=True, schema=union_schema)
        except TypeError:
            lf = scan_parquet_compat(files, schema=union_schema)
        columns = set(lf.collect_schema().names())
        exprs: list[pl.Expr] = [pl.len().alias("rows")]
        if "date" in columns:
            date_expr = pl.col("date").cast(pl.Date, strict=False)
            exprs.extend([
                date_expr.min().alias("earliest_date"),
                date_expr.max().alias("latest_date"),
                date_expr.n_unique().alias("trading_days"),
            ])
        if "symbol" in columns:
            exprs.append(pl.col("symbol").cast(pl.Utf8, strict=False).n_unique().alias("symbols_covered"))
        row = lf.select(exprs).collect().to_dicts()[0]
        for key, value in row.items():
            if isinstance(value, (date, datetime)):
                stats[key] = value.isoformat()
            else:
                stats[key] = value
    except Exception as exc:  # noqa: BLE001
        logger.debug("stockformer coverage stats failed for %s: %s", root, exc)
    return stats


def _external_feature_coverage(data_dir: Path) -> dict[str, dict]:
    specs = {
        "moneyflow": ("资金流", data_dir / "moneyflow"),
        "auction_open": ("集合竞价", data_dir / "auction_open"),
        "news": ("国内外新闻", data_dir / "news"),
        "industry": ("行业映射", data_dir / "ext_data" / "ext_hy_ths"),
    }
    latest_reasons = _latest_unavailable_reasons(data_dir)
    coverage = {}
    for key, (label, root) in specs.items():
        stats = _coverage_file_stats(root)
        files = int(stats.get("files") or 0)
        reasons: list[str] = []
        if not files:
            for reason in [*latest_reasons.get(key, []), *_coverage_config_reasons(key)]:
                if reason and reason not in reasons:
                    reasons.append(reason)
        coverage[key] = {
            "label": label,
            "available": files > 0,
            "reasons": reasons[:5],
            **stats,
        }
    return coverage


def _read_parquet_root(root: Path) -> pl.DataFrame:
    files = _parquet_files(root)
    if not files:
        return pl.DataFrame()
    try:
        return pl.read_parquet(files, hive_partitioning=True)
    except TypeError:
        try:
            return pl.read_parquet(files)
        except Exception:
            pass
    except Exception:
        pass
    frames = []
    for file in files:
        try:
            frames.append(pl.read_parquet(file))
        except Exception:
            continue
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _instrument_name_map(data_dir: Path) -> dict[str, str]:
    name_map: dict[str, str] = {}
    for root in (
        data_dir / "instruments",
        data_dir / "instruments_etf",
        data_dir / "instruments_index",
    ):
        inst = _read_parquet_root(root)
        if inst.is_empty() or "name" not in inst.columns:
            continue
        cols = [c for c in ("symbol", "name", "code", "exchange") if c in inst.columns]
        if "symbol" not in cols:
            continue
        for row in inst.select(cols).iter_rows(named=True):
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            for alias in _symbol_aliases(row.get("symbol"), row.get("code"), row.get("exchange")):
                name_map.setdefault(alias, name)
    return name_map


def _attach_instrument_names(data_dir: Path, df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "symbol" not in df.columns:
        return df
    name_map = _instrument_name_map(data_dir)
    if not name_map:
        return df

    def lookup(value) -> str | None:
        for alias in _symbol_aliases(value):
            hit = name_map.get(alias)
            if hit:
                return hit
        return None

    if "name" in df.columns:
        df = df.drop("name")
    return df.with_columns(
        pl.col("symbol").map_elements(lookup, return_dtype=pl.Utf8).alias("name")
    )


def _special_treatment_expr() -> pl.Expr:
    return (
        pl.col("name")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.to_uppercase()
        .str.contains(r"(^|\*)ST|退市")
        .fill_null(False)
    )


def _filter_special_treatment(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "name" not in df.columns:
        return df
    return df.filter(~_special_treatment_expr())


def _normalize_symbol_column(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    if "symbol" not in df.columns:
        for candidate in ("ts_code", "code", "股票代码", "证券代码"):
            if candidate in df.columns:
                df = df.rename({candidate: "symbol"})
                break
    if "symbol" not in df.columns:
        return df
    return df.with_columns(
        pl.col("symbol").map_elements(_compact_symbol_text, return_dtype=pl.Utf8).alias("symbol")
    )


def _normalize_date_column(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    if "date" not in df.columns:
        for candidate in ("trade_date", "日期", "day", "published_date"):
            if candidate in df.columns:
                df = df.rename({candidate: "date"})
                break
    if "date" not in df.columns:
        return df
    dtype = df.schema.get("date")
    if dtype == pl.Date:
        return df
    if dtype == pl.Datetime:
        return df.with_columns(pl.col("date").dt.date().alias("date"))
    date_text = pl.col("date").cast(pl.Utf8).str.strip_chars()
    return df.with_columns(
        pl.coalesce(
            [
                date_text.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                date_text.str.strptime(pl.Date, "%Y%m%d", strict=False),
                date_text.str.strptime(pl.Date, "%Y/%m/%d", strict=False),
                date_text.str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            ]
        ).alias("date")
    )


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _sum_columns(columns: list[str], names: list[str]) -> pl.Expr:
    expr: pl.Expr | None = None
    for name in names:
        if name not in columns:
            continue
        item = pl.col(name).cast(pl.Float64, strict=False).fill_null(0.0)
        expr = item if expr is None else expr + item
    return expr if expr is not None else pl.lit(0.0)


def _safe_ratio(numerator: str | pl.Expr, denominator: str | pl.Expr) -> pl.Expr:
    num = pl.col(numerator) if isinstance(numerator, str) else numerator
    denom = pl.col(denominator) if isinstance(denominator, str) else denominator
    denom = denom.cast(pl.Float64, strict=False)
    return (
        pl.when(denom.abs() > 1e-9)
        .then(num.cast(pl.Float64, strict=False) / denom)
        .otherwise(0.0)
        .fill_null(0.0)
    )


def _attach_moneyflow_features(data_dir: Path, df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    flow = _read_filtered_parquet_root(
        data_dir / "moneyflow",
        start=start - timedelta(days=7),
        end=end,
        symbols=df["symbol"].unique().to_list() if "symbol" in df.columns else None,
        columns=[
            "symbol", "date", "trade_date", "net_mf_amount", "main_net_inflow", "main_net",
            "net_amount", "主力净流入", "large_net_inflow", "large_net", "net_lg_amount",
            "大单净流入", "buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount",
        ],
    )
    if flow.is_empty():
        return df
    flow = _normalize_date_column(_normalize_symbol_column(flow))
    if "symbol" not in flow.columns or "date" not in flow.columns:
        return df
    flow = flow.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    if flow.is_empty():
        return df
    columns = flow.columns
    net_col = _first_existing(columns, ["net_mf_amount", "main_net_inflow", "main_net", "net_amount", "主力净流入"])
    large_col = _first_existing(columns, ["large_net_inflow", "large_net", "net_lg_amount", "大单净流入"])
    main_expr = (
        pl.col(net_col).cast(pl.Float64, strict=False).fill_null(0.0)
        if net_col
        else (
            _sum_columns(columns, ["buy_lg_amount", "buy_elg_amount"])
            - _sum_columns(columns, ["sell_lg_amount", "sell_elg_amount"])
        )
    )
    large_expr = (
        pl.col(large_col).cast(pl.Float64, strict=False).fill_null(0.0)
        if large_col
        else _sum_columns(columns, ["buy_lg_amount"]) - _sum_columns(columns, ["sell_lg_amount"])
    )
    flow = (
        flow.with_columns(
            main_expr.alias("_main_net_flow"),
            large_expr.alias("_large_net_flow"),
        )
        .group_by(["symbol", "date"])
        .agg(
            pl.sum("_main_net_flow").alias("_main_net_flow"),
            pl.sum("_large_net_flow").alias("_large_net_flow"),
        )
    )
    joined = df.join(flow, on=["symbol", "date"], how="left")
    latest = (
        flow
        .with_columns(pl.col("date").alias("_moneyflow_latest_date"))
        .rename({
            "_main_net_flow": "_main_net_flow_latest",
            "_large_net_flow": "_large_net_flow_latest",
        })
        .sort(["symbol", "date"])
    )
    try:
        return (
            joined
            .sort(["symbol", "date"])
            .join_asof(
                latest,
                on="date",
                by="symbol",
                strategy="backward",
                tolerance=timedelta(days=7),
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("moneyflow asof attach failed: %s", exc)
        return joined


def _attach_auction_features(data_dir: Path, df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    auction = _read_filtered_parquet_root(
        data_dir / "auction_open",
        start=start,
        end=end,
        symbols=df["symbol"].unique().to_list() if "symbol" in df.columns else None,
        columns=[
            "symbol", "date", "trade_date", "market", "open", "close", "auction_price",
            "price", "vwap", "amount", "volume",
        ],
    )
    if auction.is_empty():
        return df
    auction = _normalize_date_column(_normalize_symbol_column(auction))
    if "symbol" not in auction.columns or "date" not in auction.columns:
        return df
    auction = auction.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    if "market" in auction.columns:
        auction = auction.filter(pl.col("market").cast(pl.Utf8).str.to_lowercase().fill_null("") != "close")
    if auction.is_empty():
        return df
    # A close-then-repair sync can leave the same auction row twice under
    # qualified and compact symbol identities.  Deduplicate only exact core
    # observations; conflicting values remain visible to the aggregation
    # below instead of being silently discarded.
    dedupe_columns = [
        column for column in [
            "symbol", "date", "market", "open", "close", "auction_price",
            "price", "vwap", "amount", "volume",
        ] if column in auction.columns
    ]
    if dedupe_columns:
        auction = auction.unique(subset=dedupe_columns, keep="first", maintain_order=True)
    price_col = _first_existing(auction.columns, ["open", "close", "auction_price", "price", "vwap"])
    exprs = []
    if price_col:
        exprs.append(pl.col(price_col).cast(pl.Float64, strict=False).drop_nulls().first().alias("_auction_open_price"))
    if "amount" in auction.columns:
        exprs.append(pl.col("amount").cast(pl.Float64, strict=False).sum().alias("_auction_amount"))
    if "volume" in auction.columns:
        exprs.append(pl.col("volume").cast(pl.Float64, strict=False).sum().alias("_auction_volume"))
    if not exprs:
        return df
    exprs.append(pl.col("date").first().alias("_auction_latest_date"))
    auction = auction.group_by(["symbol", "date"]).agg(exprs)
    return df.join(auction, on=["symbol", "date"], how="left")


def _attach_industry_features(data_dir: Path, df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "industry" in df.columns:
        return df
    industry_path = data_dir / "ext_data" / "ext_hy_ths"
    industry = _read_parquet_root(industry_path)
    if industry.is_empty():
        return df
    industry = _normalize_symbol_column(industry)
    field = _first_existing(
        industry.columns,
        ["industry", "sector", "所属同花顺行业", "所属行业", "申万行业", "行业"],
    )
    if "symbol" not in industry.columns or not field:
        return df
    industry = (
        industry.select(["symbol", field])
        .rename({field: "industry"})
        .with_columns(pl.col("industry").cast(pl.Utf8).str.strip_chars().alias("industry"))
        .filter((pl.col("symbol") != "") & pl.col("industry").is_not_null() & (pl.col("industry") != ""))
        .unique(subset=["symbol"], keep="last")
    )
    if industry.is_empty():
        return df
    return df.join(industry, on="symbol", how="left")


def _sentiment_label_score(value) -> float | None:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return None
    positive = ("positive", "bullish", "利好", "正面", "乐观", "上涨", "看多")
    negative = ("negative", "bearish", "利空", "负面", "悲观", "下跌", "看空")
    neutral = ("neutral", "中性", "持平")
    if any(word in text for word in positive):
        return 1.0
    if any(word in text for word in negative):
        return -1.0
    if any(word in text for word in neutral):
        return 0.0
    return None


def _keyword_sentiment_score(value) -> float:
    text = str(value or "").lower()
    if not text:
        return 0.0
    positive = (
        "利好", "增长", "大涨", "新高", "突破", "回购", "中标", "盈利", "上调", "超预期",
        "beat", "surge", "rally", "upgrade", "record high", "profit",
    )
    negative = (
        "利空", "下跌", "暴跌", "减持", "亏损", "调查", "处罚", "风险", "下调", "不及预期",
        "miss", "plunge", "downgrade", "loss", "probe", "sanction",
    )
    pos = sum(1 for word in positive if word in text)
    neg = sum(1 for word in negative if word in text)
    if pos == neg:
        return 0.0
    return float(max(-1, min(1, pos - neg)))


def _attach_news_features(data_dir: Path, df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    news = _read_filtered_parquet_root(
        data_dir / "news",
        start=start,
        end=end,
        columns=[
            "date", "trade_date", "title", "summary", "content", "sentiment", "market",
            "market_scope", "source", "symbols",
        ],
    )
    if news.is_empty():
        return df
    news = _normalize_date_column(news)
    if "date" not in news.columns:
        return df
    news = news.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    if news.is_empty():
        return df
    text_cols = [c for c in ["title", "summary", "content"] if c in news.columns]
    text_expr = (
        pl.concat_str([pl.col(c).cast(pl.Utf8).fill_null("") for c in text_cols], separator=" ")
        if text_cols
        else pl.lit("")
    )
    label_expr = (
        pl.col("sentiment").map_elements(_sentiment_label_score, return_dtype=pl.Float64)
        if "sentiment" in news.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    market_text = (
        pl.concat_str(
            [pl.col(c).cast(pl.Utf8).fill_null("") for c in ["market", "market_scope", "source"] if c in news.columns],
            separator=" ",
        ).str.to_lowercase()
        if any(c in news.columns for c in ["market", "market_scope", "source"])
        else pl.lit("")
    )
    news = news.with_columns(
        pl.coalesce(
            [
                label_expr,
                text_expr.map_elements(_keyword_sentiment_score, return_dtype=pl.Float64),
                pl.lit(0.0),
            ]
        ).alias("_news_score"),
        market_text.str.contains("overseas|global|us|usa|nasdaq|nyse|美股|海外|港股").fill_null(False).alias("_is_overseas"),
    )
    daily = news.group_by("date").agg(
        pl.mean("_news_score").alias("_market_news_sentiment"),
        pl.len().alias("_market_news_count"),
        pl.col("_news_score").filter(pl.col("_is_overseas")).mean().alias("_overseas_news_sentiment"),
        pl.col("_is_overseas").cast(pl.Int64).sum().alias("_overseas_news_count"),
        pl.col("date").first().alias("_news_latest_date"),
    )
    df = df.join(daily, on="date", how="left")
    if "symbols" not in news.columns:
        return df
    symbol_news = (
        news.select(["date", "symbols", "_news_score"])
        .filter(pl.col("symbols").is_not_null() & (pl.col("symbols").cast(pl.Utf8).str.strip_chars() != ""))
        .with_columns(
            pl.col("symbols")
            .cast(pl.Utf8)
            .str.replace_all("[;；|]", ",")
            .str.split(",")
            .alias("symbol")
        )
        .explode("symbol")
        .with_columns(pl.col("symbol").map_elements(_compact_symbol_text, return_dtype=pl.Utf8).alias("symbol"))
        .filter(pl.col("symbol").str.len_chars() == 6)
        .group_by(["symbol", "date"])
        .agg(
            pl.mean("_news_score").alias("_symbol_news_sentiment"),
            pl.len().alias("_symbol_news_count"),
        )
    )
    if symbol_news.is_empty():
        return df
    return df.join(symbol_news, on=["symbol", "date"], how="left")


def _attach_auxiliary_features(
    data_dir: Path, df: pl.DataFrame, start: date, end: date,
    *, include_minute: bool = True, on_minute_progress=None,
) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = _attach_moneyflow_features(data_dir, df, start, end)
    df = _attach_auction_features(data_dir, df, start, end)
    df = _attach_industry_features(data_dir, df)
    df = _attach_news_features(data_dir, df, start, end)
    from app.services.minute_features import build_daily_features
    minute = (
        build_daily_features(data_dir, df["symbol"].unique().to_list(), start, end,
                             on_progress=on_minute_progress)
        if include_minute else pl.DataFrame()
    )
    if not minute.is_empty():
        minute = minute.with_columns(pl.col("date").alias("_minute_latest_date"))
        df = df.join(minute, on=["symbol", "date"], how="left")
    # Availability is calculated before neutral defaults are applied. The
    # latest-date fields are retained for provenance and debugging but are not
    # used as numeric model inputs.
    availability = []
    if "minute_session_coverage" in df.columns:
        availability.append((pl.col("minute_session_coverage").fill_null(0.0) >= 0.95).cast(pl.Float64).alias("minute_data_available"))
    else:
        availability.append(pl.lit(0.0).alias("minute_data_available"))
    if "_moneyflow_latest_date" in df.columns:
        availability.append((pl.col("_moneyflow_latest_date") == pl.col("date")).cast(pl.Float64).alias("moneyflow_data_available"))
    else:
        availability.append(pl.lit(0.0).alias("moneyflow_data_available"))
    if "_auction_open_price" in df.columns:
        availability.append((pl.col("_auction_open_price").cast(pl.Float64, strict=False).is_not_null()).cast(pl.Float64).alias("auction_data_available"))
    else:
        availability.append(pl.lit(0.0).alias("auction_data_available"))
    if "_news_latest_date" in df.columns:
        availability.append((pl.col("_news_latest_date") == pl.col("date")).cast(pl.Float64).alias("news_data_available"))
    else:
        availability.append(pl.lit(0.0).alias("news_data_available"))
    df = df.with_columns(availability).with_columns(
        (pl.col("minute_data_available") * 0.30
         + pl.col("moneyflow_data_available") * 0.25
         + pl.col("auction_data_available") * 0.15
         + pl.col("news_data_available") * 0.15
         + pl.lit(0.15)).alias("data_coverage_score")
    )
    for provenance in DATA_PROVENANCE_COLUMNS:
        if provenance not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Date).alias(provenance))
    return df


def _read_models(data_dir: Path) -> list[dict]:
    path = _metadata_path(data_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_models(data_dir: Path, models: list[dict]) -> None:
    path = _metadata_path(data_dir)
    path.write_text(
        json.dumps(models, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _store_relpath(data_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(data_dir).as_posix()
    except Exception:
        return path.as_posix()


def _resolve_artifact_path(data_dir: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    raw = Path(str(value))
    if raw.exists():
        return raw
    if not raw.is_absolute():
        candidate = data_dir / raw
        if candidate.exists():
            return candidate
    parts = raw.parts
    for marker in ("stockformer",):
        if marker in parts:
            candidate = data_dir / Path(*parts[parts.index(marker):])
            if candidate.exists():
                return candidate
    if "data" in parts:
        candidate = data_dir / Path(*parts[parts.index("data") + 1 :])
        if candidate.exists():
            return candidate
    return None


def _normalize_model_meta(data_dir: Path, meta: dict) -> dict:
    item = dict(meta)
    for field in ("model_file", "factor_file", "prediction_file"):
        resolved = _resolve_artifact_path(data_dir, item.get(field))
        if resolved is not None:
            item[field] = resolved.as_posix()
    return item


def _latest_model_id(data_dir: Path) -> str | None:
    path = _latest_path(data_dir)
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        return value or None
    models = _read_models(data_dir)
    return models[0].get("model_id") if models else None


def _model_meta(data_dir: Path, model_id: str | None = None) -> dict | None:
    target = model_id or _latest_model_id(data_dir)
    if not target:
        return None
    for item in _read_models(data_dir):
        if item.get("model_id") == target:
            return _normalize_model_meta(data_dir, item)
    return None


def _load_model_payload(model_file: str | Path) -> dict:
    return json.loads(Path(model_file).read_text(encoding="utf-8"))


def _date_range_from_data(data_dir: Path) -> tuple[date | None, date | None]:
    try:
        frame = scan_enriched_parquet(str(data_dir / "kline_daily_enriched" / "**" / "*.parquet")).select(
            pl.min("date").alias("min_date"),
            pl.max("date").alias("max_date"),
        ).collect()
        if frame.is_empty():
            return None, None
        return frame[0, "min_date"], frame[0, "max_date"]
    except Exception:
        return None, None


def _repair_corrupt_adjusted_panel(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "close" not in df.columns or "raw_close" not in df.columns:
        return df
    try:
        columns = set(df.columns)
        numeric_cols = [
            c for c in ("open", "high", "low", "close", "raw_close", "raw_high", "raw_low")
            if c in columns
        ]
        df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in numeric_cols])
        df = df.with_columns(
            pl.when((pl.col("raw_close") > 0) & (pl.col("close") > 0))
            .then(pl.col("close") / pl.col("raw_close"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("_repair_price_ratio")
        )
        median_ratio = (
            pl.col("_repair_price_ratio").median().over("symbol")
            if "symbol" in columns
            else pl.col("_repair_price_ratio").median()
        )
        ratio_outlier = pl.when(median_ratio > 0).then(
            (pl.col("_repair_price_ratio") < median_ratio * 0.30)
            | (pl.col("_repair_price_ratio") > median_ratio * 3.50)
        ).otherwise(False)
        bad_expr = (
            (pl.col("raw_close") > 0)
            & (
                pl.col("close").is_null()
                | (pl.col("close") <= 0)
                | (pl.col("_repair_price_ratio") < 0.02)
                | (pl.col("_repair_price_ratio") > 50.0)
                | ratio_outlier
            )
        ).fill_null(False)
        df = df.with_columns(
            bad_expr.alias("_repair_price_bad"),
            pl.when((pl.col("close").is_not_null()) & (pl.col("close") > 0))
            .then(pl.col("raw_close") / pl.col("close"))
            .otherwise(1.0)
            .alias("_repair_price_scale"),
        )
        if not bool(df.select(pl.col("_repair_price_bad").any()).item()):
            return df.drop([
                c for c in ("_repair_price_ratio", "_repair_price_bad", "_repair_price_scale")
                if c in df.columns
            ])
        exprs: list[pl.Expr] = []
        if "open" in columns:
            exprs.append(
                pl.when(pl.col("_repair_price_bad") & (pl.col("open") > 0))
                .then(pl.col("open") * pl.col("_repair_price_scale"))
                .when(pl.col("_repair_price_bad"))
                .then(pl.col("raw_close"))
                .otherwise(pl.col("open"))
                .alias("open")
            )
        if "high" in columns:
            if "raw_high" in columns:
                repaired_high = (
                    pl.when(pl.col("_repair_price_bad") & (pl.col("raw_high") > 0))
                    .then(pl.col("raw_high"))
                    .when(pl.col("_repair_price_bad") & (pl.col("high") > 0))
                    .then(pl.col("high") * pl.col("_repair_price_scale"))
                    .when(pl.col("_repair_price_bad"))
                    .then(pl.col("raw_close"))
                )
            else:
                repaired_high = (
                    pl.when(pl.col("_repair_price_bad") & (pl.col("high") > 0))
                    .then(pl.col("high") * pl.col("_repair_price_scale"))
                    .when(pl.col("_repair_price_bad"))
                    .then(pl.col("raw_close"))
                )
            exprs.append(
                repaired_high.otherwise(pl.col("high")).alias("high")
            )
        if "low" in columns:
            if "raw_low" in columns:
                repaired_low = (
                    pl.when(pl.col("_repair_price_bad") & (pl.col("raw_low") > 0))
                    .then(pl.col("raw_low"))
                    .when(pl.col("_repair_price_bad") & (pl.col("low") > 0))
                    .then(pl.col("low") * pl.col("_repair_price_scale"))
                    .when(pl.col("_repair_price_bad"))
                    .then(pl.col("raw_close"))
                )
            else:
                repaired_low = (
                    pl.when(pl.col("_repair_price_bad") & (pl.col("low") > 0))
                    .then(pl.col("low") * pl.col("_repair_price_scale"))
                    .when(pl.col("_repair_price_bad"))
                    .then(pl.col("raw_close"))
                )
            exprs.append(
                repaired_low.otherwise(pl.col("low")).alias("low")
            )
        exprs.append(
            pl.when(pl.col("_repair_price_bad"))
            .then(pl.col("raw_close"))
            .otherwise(pl.col("close"))
            .alias("close")
        )
        return df.with_columns(exprs).drop([
            c for c in ("_repair_price_ratio", "_repair_price_bad", "_repair_price_scale")
            if c in df.columns
        ])
    except Exception:
        return df


def _prediction_cross_section_threshold() -> int:
    return _env_int("STOCKFORMER_MIN_CROSS_SECTION_SYMBOLS", 500, min_value=20)


def _prediction_cross_section_lookback_days() -> int:
    return _env_int("STOCKFORMER_CROSS_SECTION_LOOKBACK_DAYS", 45, min_value=7, max_value=180)


def _fill_prediction_feature_defaults(df: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    if df.is_empty():
        return df
    exprs: list[pl.Expr] = []
    for column in feature_columns:
        default = PREDICTION_FEATURE_DEFAULTS.get(column)
        if column not in df.columns:
            exprs.append(pl.lit(default if default is not None else 0.0).alias(column))
        elif default is not None:
            exprs.append(
                pl.col(column)
                .cast(pl.Float64, strict=False)
                .fill_null(default)
                .alias(column)
            )
    return df.with_columns(exprs) if exprs else df


def _resolve_prediction_target_date(
    data_dir: Path,
    max_date: date,
    as_of: date | None,
) -> tuple[date, dict[str, Any]]:
    min_symbols = _prediction_cross_section_threshold()
    meta: dict[str, Any] = {
        "latest_data_date": max_date,
        "min_cross_section_symbols": min_symbols,
        "requested_as_of": as_of,
        "date_policy": "requested" if as_of else "latest_full_cross_section",
    }
    if as_of is not None:
        return as_of, meta

    try:
        lookback_start = max_date - timedelta(days=_prediction_cross_section_lookback_days())
        counts = (
            scan_enriched_parquet(str(data_dir / "kline_daily_enriched" / "**" / "*.parquet"))
            .filter((pl.col("date") >= lookback_start) & (pl.col("date") <= max_date))
            .group_by("date")
            .agg(
                pl.col("symbol").n_unique().alias("cross_section_symbols"),
                pl.len().alias("cross_section_rows"),
            )
            .sort("date", descending=True)
            .collect()
        )
    except Exception:
        return max_date, meta

    if counts.is_empty():
        return max_date, meta
    complete = counts.filter(pl.col("cross_section_symbols") >= min_symbols)
    if not complete.is_empty():
        row = complete.row(0, named=True)
        target = row["date"]
        meta.update({
            "cross_section_symbols": int(row.get("cross_section_symbols") or 0),
            "cross_section_rows": int(row.get("cross_section_rows") or 0),
        })
        skipped = counts.filter(pl.col("date") > target).to_dicts()
        if skipped:
            meta["skipped_sparse_dates"] = [
                {
                    "date": str(item.get("date")),
                    "symbols": int(item.get("cross_section_symbols") or 0),
                    "rows": int(item.get("cross_section_rows") or 0),
                }
                for item in skipped[:5]
            ]
        return target, meta

    row = counts.row(0, named=True)
    target = row["date"]
    meta.update({
        "cross_section_symbols": int(row.get("cross_section_symbols") or 0),
        "cross_section_rows": int(row.get("cross_section_rows") or 0),
        "warning": f"最近 {_prediction_cross_section_lookback_days()} 天没有达到 {min_symbols} 只股票的完整截面",
    })
    return target, meta


def _load_panel(
    data_dir: Path,
    start: date,
    end: date,
    max_symbols: int | None = None,
    symbols: list[str] | None = None,
    universe_as_of: date | None = None,
    repair_adjusted: bool = True,
    include_minute: bool = True,
    on_minute_progress=None,
) -> pl.DataFrame:
    path = data_dir / "kline_daily_enriched" / "**" / "*.parquet"
    columns = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover_rate",
        "raw_close",
        "signal_limit_up",
        "signal_limit_down",
        "signal_limit_down_recovery",
        "signal_broken_limit_up",
        "consecutive_limit_ups",
        "consecutive_limit_downs",
    ]
    try:
        lf = scan_enriched_parquet(str(path)).filter((pl.col("date") >= start) & (pl.col("date") <= end))
        if symbols:
            lf = lf.filter(_symbol_in_filter(symbols))
        available = [c for c in columns if c in lf.collect_schema().names()]
        df = lf.select(available).collect()
    except Exception:
        return pl.DataFrame()
    if df.is_empty():
        return df
    df = _normalize_symbol_column(df).sort(["symbol", "date"])
    if "symbol" in df.columns and "date" in df.columns:
        df = df.unique(subset=["symbol", "date"], keep="last").sort(["symbol", "date"])
    if repair_adjusted:
        df = _repair_corrupt_adjusted_panel(df)
    if max_symbols and max_symbols > 0 and universe_as_of is not None:
        from app.services.stockformer_training import select_universe
        selected = select_universe(df, universe_as_of, max_symbols)
        df = df.filter(pl.col('symbol').is_in(selected))
    elif max_symbols and max_symbols > 0:
        agg_exprs = [pl.len().alias("n")]
        if "amount" in df.columns:
            agg_exprs.append(pl.col("amount").fill_null(0).mean().alias("avg_amount"))
        ranked = df.group_by("symbol").agg(agg_exprs)
        sort_cols = ["n"]
        descending = [True]
        if "avg_amount" in ranked.columns:
            sort_cols.append("avg_amount")
            descending.append(True)
        sort_cols.append("symbol")
        descending.append(False)
        symbols = ranked.sort(sort_cols, descending=descending).head(max_symbols)["symbol"].to_list()
        df = df.filter(pl.col("symbol").is_in(symbols))
    return _attach_auxiliary_features(data_dir, df, start, end,
                                      include_minute=include_minute,
                                      on_minute_progress=on_minute_progress)


def _feature_frame(panel: pl.DataFrame, include_targets: bool = True) -> pl.DataFrame:
    if panel.is_empty():
        return panel
    df = panel.sort(["symbol", "date"])
    if "close" not in df.columns:
        return pl.DataFrame()
    if "open" not in df.columns:
        df = df.with_columns(pl.col("close").alias("open"))
    if "high" not in df.columns:
        df = df.with_columns(pl.col("close").alias("high"))
    if "low" not in df.columns:
        df = df.with_columns(pl.col("close").alias("low"))
    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("volume"))
    if "turnover_rate" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("turnover_rate"))
    if "amount" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("amount"))
    for column in ("_main_net_flow", "_large_net_flow", "_auction_amount", "_auction_volume", "_auction_open_price"):
        if column not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(column))
    for source, fallback in (
        ("_main_net_flow", "_main_net_flow_latest"),
        ("_large_net_flow", "_large_net_flow_latest"),
    ):
        if fallback not in df.columns:
            df = df.with_columns(pl.col(source).alias(fallback))
    for column in ("_market_news_sentiment", "_overseas_news_sentiment", "_symbol_news_sentiment"):
        if column not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(column))
    for column in ("_market_news_count", "_overseas_news_count", "_symbol_news_count"):
        if column not in df.columns:
            df = df.with_columns(pl.lit(0).alias(column))
    for column in ("signal_limit_up", "signal_limit_down", "signal_limit_down_recovery", "signal_broken_limit_up"):
        if column not in df.columns:
            df = df.with_columns(pl.lit(False).alias(column))
    for column in ("consecutive_limit_ups", "consecutive_limit_downs"):
        if column not in df.columns:
            df = df.with_columns(pl.lit(0).alias(column))
    prev_close = pl.col("close").shift(1).over("symbol")
    df = df.with_columns(
        (pl.col("close") / prev_close - 1).alias("ret_1d"),
        (pl.col("close") / pl.col("close").shift(3).over("symbol") - 1).alias("ret_3d"),
        (pl.col("close") / pl.col("close").shift(5).over("symbol") - 1).alias("ret_5d"),
        (pl.col("close") / pl.col("close").shift(10).over("symbol") - 1).alias("ret_10d"),
        (pl.col("close") / pl.col("close").shift(20).over("symbol") - 1).alias("ret_20d"),
        (pl.col("close").rolling_mean(20).over("symbol")).alias("_ma20"),
        (pl.col("volume").rolling_mean(5).over("symbol")).alias("_vol5"),
        (pl.col("turnover_rate").rolling_mean(5).over("symbol")).alias("_turnover5"),
        (pl.col("amount").rolling_mean(5).over("symbol")).alias("_amount5"),
        (pl.col("amount").rolling_mean(20).over("symbol")).alias("_amount20"),
        (pl.col("amount").rolling_sum(3).over("symbol")).alias("_amount_sum3"),
        (pl.col("amount").rolling_sum(5).over("symbol")).alias("_amount_sum5"),
        (pl.col("high").rolling_max(10).over("symbol")).alias("_high10"),
        (pl.col("low").rolling_min(10).over("symbol")).alias("_low10"),
        (pl.col("high").rolling_max(20).over("symbol")).alias("_high20"),
        (pl.col("low").rolling_min(20).over("symbol")).alias("_low20"),
        (pl.col("high").rolling_max(60).over("symbol")).alias("_high60"),
        (pl.col("low").rolling_min(60).over("symbol")).alias("_low60"),
        (pl.col("_main_net_flow").rolling_sum(3).over("symbol")).alias("_main_net_sum3"),
        (pl.col("_main_net_flow").rolling_sum(5).over("symbol")).alias("_main_net_sum5"),
        (pl.col("amount").shift(1).rolling_sum(3).over("symbol")).alias("_amount_prev3"),
        (pl.col("amount").shift(1).rolling_sum(5).over("symbol")).alias("_amount_prev5"),
        (pl.col("_main_net_flow").shift(1).rolling_sum(3).over("symbol")).alias("_main_net_prev3"),
        (pl.col("_main_net_flow").shift(1).rolling_sum(5).over("symbol")).alias("_main_net_prev5"),
        (
            (pl.col("_main_net_flow").cast(pl.Float64, strict=False).fill_null(0.0) > 0)
            .cast(pl.Float64)
            .rolling_sum(5)
            .over("symbol")
        ).fill_null(0.0).alias("moneyflow_positive_days_5d"),
    )
    df = df.with_columns(
        (pl.col("close") / pl.col("_ma20") - 1).alias("ma20_gap"),
        (pl.col("volume") / pl.col("_vol5")).alias("vol_ratio_5d"),
        _safe_ratio("_amount5", "_amount20").alias("_amount5_to_20"),
        _safe_ratio(pl.col("close") - pl.col("_low20"), pl.col("_high20") - pl.col("_low20")).clip(0, 1).alias("price_position_20d"),
        _safe_ratio(pl.col("close") - pl.col("_low60"), pl.col("_high60") - pl.col("_low60")).clip(0, 1).alias("price_position_60d"),
        _safe_ratio(pl.col("_high10") - pl.col("_low10"), "close").clip(0, 1).alias("price_range_10d"),
        pl.col("ret_1d").rolling_std(10).over("symbol").fill_null(0.0).alias("_price_volatility_10d"),
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("intraday_strength"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("amplitude"),
        ((pl.col("ret_1d").rolling_mean(6).over("symbol") / pl.col("ret_1d").rolling_std(6).over("symbol")).clip(-3, 3)).alias("rsi_proxy"),
        (pl.col("ret_1d").rolling_std(20).over("symbol")).alias("risk_20d"),
        (pl.col("open") / prev_close - 1).alias("gap_open"),
        _safe_ratio("turnover_rate", "_turnover5").alias("turnover_ratio_5d"),
        _safe_ratio("amount", "_amount5").alias("amount_ratio_5d"),
        pl.col("signal_limit_up").cast(pl.Boolean, strict=False).fill_null(False).cast(pl.Float64).alias("limit_up_flag"),
        pl.col("signal_limit_down").cast(pl.Boolean, strict=False).fill_null(False).cast(pl.Float64).alias("limit_down_flag"),
        pl.col("signal_broken_limit_up").cast(pl.Boolean, strict=False).fill_null(False).cast(pl.Float64).alias("broken_limit_up_flag"),
        pl.col("signal_limit_down_recovery").cast(pl.Boolean, strict=False).fill_null(False).cast(pl.Float64).alias("limit_down_recovery_flag"),
        pl.col("consecutive_limit_ups").cast(pl.Float64, strict=False).fill_null(0.0).clip(0, 10).alias("limit_up_streak"),
        pl.col("consecutive_limit_downs").cast(pl.Float64, strict=False).fill_null(0.0).clip(0, 10).alias("limit_down_streak"),
        (
            pl.coalesce([
                pl.col("_main_net_flow").cast(pl.Float64, strict=False),
                pl.col("_main_net_flow_latest").cast(pl.Float64, strict=False),
                pl.lit(0.0),
            ])
            / pl.when(pl.col("amount").cast(pl.Float64, strict=False).abs() > 1e-9)
            .then(pl.col("amount").cast(pl.Float64, strict=False))
            .otherwise(1.0)
        ).fill_null(0.0).clip(-1, 1).alias("main_net_inflow_rate"),
        (
            pl.coalesce([
                pl.col("_large_net_flow").cast(pl.Float64, strict=False),
                pl.col("_large_net_flow_latest").cast(pl.Float64, strict=False),
                pl.lit(0.0),
            ])
            / pl.when(pl.col("amount").cast(pl.Float64, strict=False).abs() > 1e-9)
            .then(pl.col("amount").cast(pl.Float64, strict=False))
            .otherwise(1.0)
        ).fill_null(0.0).clip(-1, 1).alias("large_net_inflow_rate"),
        _safe_ratio("_auction_amount", "amount").clip(0, 1).alias("auction_amount_ratio"),
        _safe_ratio("_auction_volume", "volume").clip(0, 1).alias("auction_volume_ratio"),
    )
    df = df.with_columns(
        _safe_ratio("_main_net_sum3", "_amount_sum3").clip(-1, 1).alias("moneyflow_strength_3d"),
        _safe_ratio("_main_net_sum5", "_amount_sum5").clip(-1, 1).alias("moneyflow_strength_5d"),
        _safe_ratio("_main_net_prev3", "_amount_prev3").clip(-1, 1).alias("_moneyflow_strength_prev3"),
        _safe_ratio("_main_net_prev5", "_amount_prev5").clip(-1, 1).alias("_moneyflow_strength_prev5"),
        (
            pl.col("gap_open").fill_null(0.0) * 0.55
            + (
                pl.when((prev_close.abs() > 1e-9) & (pl.col("_auction_open_price").cast(pl.Float64, strict=False) > 0))
                .then(pl.col("_auction_open_price").cast(pl.Float64, strict=False) / prev_close - 1)
                .otherwise(0.0)
                .fill_null(0.0)
                * 0.25
            )
            + pl.col("auction_amount_ratio").fill_null(0.0) * 0.20
        ).alias("open_strength"),
    )
    base_zone_expr = (
        (
            pl.when((pl.col("price_position_20d") >= 0.18) & (pl.col("price_position_20d") <= 0.60))
            .then(0.92)
            .when((pl.col("price_position_20d") > 0.60) & (pl.col("price_position_20d") <= 0.72))
            .then(0.68)
            .when(pl.col("price_position_20d") < 0.10)
            .then(0.36)
            .when(pl.col("price_position_20d") > 0.82)
            .then(0.24)
            .otherwise(0.56)
            * 0.58
        )
        + (
            pl.when((pl.col("price_position_60d") >= 0.16) & (pl.col("price_position_60d") <= 0.55))
            .then(0.88)
            .when((pl.col("price_position_60d") > 0.55) & (pl.col("price_position_60d") <= 0.68))
            .then(0.64)
            .when(pl.col("price_position_60d") > 0.78)
            .then(0.28)
            .otherwise(0.50)
            * 0.42
        )
    ).clip(0.0, 1.0)
    price_compression_expr = (
        (
            pl.when((pl.col("price_range_10d") >= 0.035) & (pl.col("price_range_10d") <= 0.13))
            .then(0.90)
            .when(pl.col("price_range_10d") < 0.035)
            .then(0.58)
            .when(pl.col("price_range_10d") <= 0.18)
            .then(0.70)
            .otherwise((0.70 - (pl.col("price_range_10d") - 0.18) * 3.00).clip(0.18, 0.70))
            * 0.42
        )
        + (
            pl.when(pl.col("_price_volatility_10d") <= 0.025)
            .then(0.86)
            .when(pl.col("_price_volatility_10d") <= 0.045)
            .then(0.68)
            .otherwise(0.35)
            * 0.22
        )
        + (
            pl.when((pl.col("ret_5d").fill_null(0.0) >= -0.045) & (pl.col("ret_5d").fill_null(0.0) <= 0.055))
            .then(0.88)
            .when(pl.col("ret_5d").fill_null(0.0) <= 0.090)
            .then(0.56)
            .otherwise(0.25)
            * 0.36
        )
    ).clip(0.0, 1.0)
    volume_absorption_expr = (
        (
            pl.when((pl.col("amount_ratio_5d") >= 0.75) & (pl.col("amount_ratio_5d") <= 2.15))
            .then(0.86)
            .when((pl.col("amount_ratio_5d") > 2.15) & (pl.col("amount_ratio_5d") <= 2.80))
            .then(0.62)
            .when(pl.col("amount_ratio_5d") > 3.20)
            .then(0.28)
            .otherwise(0.46)
            * 0.50
        )
        + (
            pl.when((pl.col("_amount5_to_20") >= 0.82) & (pl.col("_amount5_to_20") <= 1.65))
            .then(0.82)
            .when(pl.col("_amount5_to_20") > 2.20)
            .then(0.35)
            .otherwise(0.52)
            * 0.30
        )
        + (
            pl.when(pl.col("turnover_rate").fill_null(0.0) <= 12)
            .then(0.74)
            .when(pl.col("turnover_rate").fill_null(0.0) >= 22)
            .then(0.30)
            .otherwise(0.56)
            * 0.20
        )
    ).clip(0.0, 1.0)
    price_setup_expr = (
        pl.when(pl.col("ret_5d").fill_null(0.0) <= -0.08)
        .then(0.34)
        .when(pl.col("ret_5d").fill_null(0.0) <= 0.045)
        .then(0.92 - (pl.col("ret_5d").fill_null(0.0) - 0.012).abs() * 3.20)
        .when(pl.col("ret_5d").fill_null(0.0) <= 0.095)
        .then(0.74 - (pl.col("ret_5d").fill_null(0.0) - 0.045) * 5.60)
        .otherwise(0.28)
        .clip(0.0, 1.0)
    )
    price_chase_risk_expr = (
        pl.max_horizontal(pl.col("ret_3d").fill_null(0.0) - 0.055, pl.lit(0.0)) * 4.8
        + pl.max_horizontal(pl.col("ret_5d").fill_null(0.0) - 0.095, pl.lit(0.0)) * 3.8
        + pl.max_horizontal(pl.col("ma20_gap").fill_null(0.0) - 0.16, pl.lit(0.0)) * 2.2
        + pl.col("limit_up_streak").fill_null(0.0).clip(0.0, 6.0) * 0.08
        + pl.max_horizontal(pl.col("amount_ratio_5d").fill_null(1.0) - 3.0, pl.lit(0.0)) * 0.16
    ).clip(0.0, 1.0)
    df = df.with_columns(
        (pl.col("moneyflow_strength_3d") - pl.col("_moneyflow_strength_prev3")).clip(-1, 1).alias("moneyflow_acceleration_3d"),
        (
            (pl.col("main_net_inflow_rate").fill_null(0.0) > 0)
            & (pl.col("_moneyflow_strength_prev3").fill_null(0.0) <= 0)
        ).cast(pl.Float64).alias("moneyflow_turn_positive"),
        price_chase_risk_expr.alias("price_chase_risk"),
    )
    df = df.with_columns(
        base_zone_expr.alias("base_zone_score"),
        price_compression_expr.alias("price_compression_score"),
        volume_absorption_expr.alias("volume_absorption_score"),
        (
            pl.max_horizontal(pl.col("moneyflow_strength_3d").fill_null(0.0), pl.lit(0.0))
            / pl.max_horizontal(pl.col("ret_3d").abs().fill_null(0.0), pl.lit(0.008))
        ).clip(0.0, 1.0).alias("moneyflow_absorption_efficiency"),
    )
    df = df.with_columns(
        (
            pl.max_horizontal(pl.col("moneyflow_positive_days_5d").fill_null(0.0) - 3.0, pl.lit(0.0)) * 0.20
            + pl.col("price_chase_risk").fill_null(0.0) * 0.45
            + pl.max_horizontal(pl.col("moneyflow_strength_5d").fill_null(0.0) - 0.08, pl.lit(0.0)) * 0.90
        ).clip(0.0, 1.0).alias("moneyflow_aging_risk")
    )
    df = df.with_columns(
        (
            _pct_rank_expr("moneyflow_acceleration_3d") * 0.34
            + _pct_rank_expr("main_net_inflow_rate") * 0.22
            + _pct_rank_expr("large_net_inflow_rate") * 0.16
            + _pct_rank_expr("amount_ratio_5d") * 0.10
            + pl.col("moneyflow_turn_positive").fill_null(0.0) * 0.10
            + pl.col("moneyflow_absorption_efficiency").fill_null(0.0) * 0.08
            + pl.col("base_zone_score").fill_null(0.5) * 0.08
            + price_setup_expr * 0.18
            - pl.col("price_chase_risk").fill_null(0.5) * 0.12
            - pl.col("moneyflow_aging_risk").fill_null(0.0) * 0.18
        ).clip(0.0, 1.0).alias("moneyflow_early_layout_score"),
        (
            pl.col("base_zone_score").fill_null(0.5) * 0.22
            + pl.col("price_compression_score").fill_null(0.5) * 0.22
            + pl.col("volume_absorption_score").fill_null(0.5) * 0.15
            + _pct_rank_expr("moneyflow_acceleration_3d") * 0.14
            + _pct_rank_expr("main_net_inflow_rate") * 0.10
            + _pct_rank_expr("large_net_inflow_rate") * 0.08
            + pl.col("moneyflow_absorption_efficiency").fill_null(0.0) * 0.09
            + pl.col("moneyflow_turn_positive").fill_null(0.0) * 0.06
            - pl.col("moneyflow_aging_risk").fill_null(0.0) * 0.16
            - pl.col("price_chase_risk").fill_null(0.0) * 0.10
        ).clip(0.0, 1.0).alias("moneyflow_accumulation_score"),
        (
            pl.col("price_chase_risk").fill_null(0.0) * 0.58
            + pl.col("moneyflow_aging_risk").fill_null(0.0) * 0.28
            + pl.max_horizontal(pl.col("moneyflow_strength_5d").fill_null(0.0) - 0.08, pl.lit(0.0)) * 1.20
            + pl.max_horizontal(pl.col("moneyflow_strength_3d").fill_null(0.0) - 0.11, pl.lit(0.0)) * 0.80
        ).clip(0.0, 1.0).alias("moneyflow_chase_risk"),
    )
    df = df.with_columns(
        pl.when((pl.col("price_position_20d") > 0.82) | (pl.col("price_position_60d") > 0.78))
        .then(pl.min_horizontal(pl.col("moneyflow_accumulation_score"), pl.lit(0.66)))
        .when(pl.col("ret_3d").fill_null(0.0) > 0.055)
        .then(pl.min_horizontal(pl.col("moneyflow_accumulation_score"), pl.lit(0.70)))
        .when((pl.col("ret_5d").fill_null(0.0) > 0.08) | (pl.col("ret_10d").fill_null(0.0) > 0.14))
        .then(pl.min_horizontal(pl.col("moneyflow_accumulation_score"), pl.lit(0.68)))
        .when(pl.col("base_zone_score") < 0.45)
        .then(pl.min_horizontal(pl.col("moneyflow_accumulation_score"), pl.lit(0.62)))
        .otherwise(pl.col("moneyflow_accumulation_score"))
        .alias("moneyflow_accumulation_score")
    )
    if "industry" in df.columns:
        df = df.with_columns(
            pl.mean("ret_1d").over(["date", "industry"]).alias("industry_ret_1d"),
            pl.mean("limit_up_flag").over(["date", "industry"]).alias("industry_limit_up_rate"),
            pl.mean("amount_ratio_5d").over(["date", "industry"]).alias("_industry_amount_heat"),
            pl.mean("ret_3d").over(["date", "industry"]).alias("sector_ret_3d"),
            pl.mean("ret_5d").over(["date", "industry"]).alias("sector_ret_5d"),
            (pl.col("ret_1d").fill_null(0.0) > 0).cast(pl.Float64).mean().over(["date", "industry"]).alias("sector_up_ratio"),
            pl.mean("amount_ratio_5d").over(["date", "industry"]).alias("sector_amount_ratio"),
            pl.mean("moneyflow_strength_5d").over(["date", "industry"]).alias("sector_moneyflow_strength_5d"),
        )
        sector_daily = (
            df.select([
                "date",
                "industry",
                "sector_ret_5d",
                "sector_up_ratio",
                "sector_amount_ratio",
                "sector_moneyflow_strength_5d",
            ])
            .unique(subset=["date", "industry"], keep="last")
            .sort(["industry", "date"])
            .with_columns(
                pl.col("sector_ret_5d").shift(1).over("industry").alias("sector_pre_ret_5d"),
                pl.col("sector_up_ratio").shift(1).over("industry").alias("sector_pre_up_ratio"),
                pl.col("sector_amount_ratio").shift(1).over("industry").alias("sector_pre_amount_ratio"),
                pl.col("sector_moneyflow_strength_5d")
                .shift(1)
                .over("industry")
                .alias("sector_pre_moneyflow_strength_5d"),
            )
            .select([
                "date",
                "industry",
                "sector_pre_ret_5d",
                "sector_pre_up_ratio",
                "sector_pre_amount_ratio",
                "sector_pre_moneyflow_strength_5d",
            ])
        )
        df = df.join(sector_daily, on=["date", "industry"], how="left")
        df = df.with_columns(
            (
                pl.col("industry_ret_1d").fill_null(0.0) * 0.50
                + pl.col("industry_limit_up_rate").fill_null(0.0) * 0.25
                + pl.col("_industry_amount_heat").fill_null(1.0) * 0.25
            ).alias("_industry_heat_raw")
        )
        sector_early_position = (
            pl.when(pl.col("sector_ret_5d").fill_null(0.0) <= -0.06)
            .then(0.40)
            .when(pl.col("sector_ret_5d").fill_null(0.0) <= 0.08)
            .then(0.88 - (pl.col("sector_ret_5d").fill_null(0.0) - 0.025).abs() * 2.80)
            .when(pl.col("sector_ret_5d").fill_null(0.0) <= 0.16)
            .then(0.65 - (pl.col("sector_ret_5d").fill_null(0.0) - 0.08) * 3.50)
            .otherwise(0.28)
            .clip(0.0, 1.0)
        )
        sector_pre_position = (
            pl.when(pl.col("sector_pre_ret_5d").fill_null(0.0) <= -0.08)
            .then(0.42)
            .when(pl.col("sector_pre_ret_5d").fill_null(0.0) <= -0.035)
            .then(0.66 + (pl.col("sector_pre_ret_5d").fill_null(0.0) + 0.08) * 3.60)
            .when(pl.col("sector_pre_ret_5d").fill_null(0.0) <= 0.045)
            .then(0.92 - (pl.col("sector_pre_ret_5d").fill_null(0.0) - 0.012).abs() * 2.60)
            .when(pl.col("sector_pre_ret_5d").fill_null(0.0) <= 0.085)
            .then(0.76 - (pl.col("sector_pre_ret_5d").fill_null(0.0) - 0.045) * 5.20)
            .otherwise(0.38)
            .clip(0.0, 1.0)
        )
        today_heat_penalty = (
            pl.max_horizontal(pl.col("industry_ret_1d").fill_null(0.0) - 0.018, pl.lit(0.0)) * 8.0
            + pl.max_horizontal(pl.col("sector_up_ratio").fill_null(0.5) - 0.78, pl.lit(0.0)) * 1.25
            + pl.col("industry_limit_up_rate").fill_null(0.0) * 1.10
        ).clip(0.0, 1.0)
        sector_today_confirm = (
            pl.lit(0.50)
            + pl.max_horizontal(pl.col("industry_ret_1d").fill_null(0.0), pl.lit(0.0)) * 5.2
            + pl.max_horizontal(pl.col("sector_up_ratio").fill_null(0.5) - 0.55, pl.lit(0.0)) * 0.55
            + pl.max_horizontal(pl.col("sector_amount_ratio").fill_null(1.0) - 1.05, pl.lit(0.0)) * 0.18
            + pl.col("industry_limit_up_rate").fill_null(0.0) * 0.35
        ).clip(0.0, 1.0)
        sector_crowding = (
            pl.max_horizontal(pl.col("sector_ret_5d").fill_null(0.0), pl.lit(0.0)) * 1.60
            + pl.col("industry_limit_up_rate").fill_null(0.0) * 1.20
            + pl.max_horizontal(pl.col("sector_up_ratio").fill_null(0.5) - 0.70, pl.lit(0.0)) * 0.90
        ).clip(0.0, 1.0)
        df = df.with_columns(
            _pct_rank_expr("_industry_heat_raw").alias("industry_heat_rank"),
            (
                _pct_rank_expr("sector_moneyflow_strength_5d") * 0.38
                + _pct_rank_expr("sector_amount_ratio") * 0.22
                + _pct_rank_expr("sector_up_ratio") * 0.14
                + _pct_rank_expr("industry_limit_up_rate") * 0.08
                + sector_early_position * 0.18
            ).alias("sector_fund_layout_score"),
            (
                _pct_rank_expr("sector_pre_moneyflow_strength_5d") * 0.34
                + _pct_rank_expr("sector_pre_amount_ratio") * 0.16
                + _pct_rank_expr("sector_pre_up_ratio") * 0.12
                + sector_pre_position * 0.28
                + (1.0 - today_heat_penalty) * 0.10
            ).clip(0.0, 1.0).alias("sector_setup_score"),
            sector_today_confirm.alias("sector_today_confirm_score"),
            sector_crowding.alias("sector_crowding_risk"),
        )
        df = df.with_columns(
            (
                pl.col("sector_setup_score").fill_null(0.5) * 0.34
                + pl.col("sector_fund_layout_score").fill_null(0.5) * 0.22
                + pl.col("sector_today_confirm_score").fill_null(0.5) * 0.12
                + _pct_rank_expr("sector_ret_3d") * 0.12
                + _pct_rank_expr("sector_ret_5d") * 0.08
                + _pct_rank_expr("sector_amount_ratio") * 0.06
                + _pct_rank_expr("sector_up_ratio") * 0.02
                + (1.0 - pl.col("sector_crowding_risk").fill_null(0.5)) * 0.04
            ).clip(0.0, 1.0).alias("sector_rotation_score")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("industry_ret_1d"),
            pl.lit(0.5).alias("industry_heat_rank"),
            pl.lit(0.0).alias("industry_limit_up_rate"),
            pl.lit(0.0).alias("sector_ret_3d"),
            pl.lit(0.0).alias("sector_ret_5d"),
            pl.lit(0.5).alias("sector_up_ratio"),
            pl.lit(1.0).alias("sector_amount_ratio"),
            pl.lit(0.0).alias("sector_moneyflow_strength_5d"),
            pl.lit(0.5).alias("sector_fund_layout_score"),
            pl.lit(0.5).alias("sector_setup_score"),
            pl.lit(0.0).alias("sector_pre_ret_5d"),
            pl.lit(0.5).alias("sector_pre_up_ratio"),
            pl.lit(1.0).alias("sector_pre_amount_ratio"),
            pl.lit(0.0).alias("sector_pre_moneyflow_strength_5d"),
            pl.lit(0.5).alias("sector_today_confirm_score"),
            pl.lit(0.5).alias("sector_rotation_score"),
            pl.lit(0.5).alias("sector_crowding_risk"),
        )
    df = df.with_columns(
        (
            pl.col("_symbol_news_sentiment").fill_null(pl.col("_market_news_sentiment")).fill_null(0.0) * 0.65
            + pl.col("_market_news_sentiment").fill_null(0.0) * 0.25
            + pl.col("_overseas_news_sentiment").fill_null(0.0) * 0.10
        ).alias("news_sentiment_score"),
        (pl.col("_symbol_news_count").fill_null(0) + pl.col("_market_news_count").fill_null(0))
        .rolling_sum(3)
        .over("symbol")
        .log1p()
        .alias("news_count_3d"),
        pl.col("_overseas_news_sentiment").fill_null(0.0).alias("overseas_news_sentiment_score"),
        pl.col("_overseas_news_count").fill_null(0).rolling_sum(3).over("symbol").log1p().alias("overseas_news_count_3d"),
    )
    finite_columns = [c for c in [*FEATURE_COLUMNS, "amount"] if c in df.columns]
    if finite_columns:
        df = df.with_columns(
            [
                pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
                for c in finite_columns
            ]
        )
    default_exprs = [
        pl.col(column).fill_null(default).alias(column)
        for column, default in OPTIONAL_FEATURE_DEFAULTS.items()
        if column in df.columns
    ]
    if default_exprs:
        df = df.with_columns(default_exprs)
    if include_targets:
        for horizon in HORIZONS:
            df = df.with_columns(
                (pl.col("close").shift(-horizon).over("symbol") / pl.col("close") - 1).alias(f"target_{horizon}d")
            )
    keep = ["symbol", "date", "close", "amount"]
    if "industry" in df.columns:
        keep.append("industry")
    keep.extend(FEATURE_COLUMNS)
    keep.extend([c for c in DATA_PROVENANCE_COLUMNS if c in df.columns])
    if include_targets:
        keep.extend([f"target_{horizon}d" for horizon in HORIZONS])
    return df.select([c for c in keep if c in df.columns])


def _prepare_xy(
    df: pl.DataFrame,
    target_col: str,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    feature_columns = feature_columns or FEATURE_COLUMNS
    usable = df.filter(pl.col(target_col).is_not_null() & pl.col(target_col).is_finite())
    for col in feature_columns:
        usable = usable.filter(pl.col(col).is_not_null() & pl.col(col).is_finite())
    if usable.is_empty():
        return np.empty((0, len(feature_columns))), np.empty((0,)), usable
    x = usable.select(feature_columns).to_numpy().astype(float)
    y = usable[target_col].to_numpy().astype(float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y, usable


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float, n_features: int | None = None,
               *, clip_quantiles: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
    if x.size == 0:
        return np.zeros((n_features or len(FEATURE_COLUMNS)) + 1), {"mean": [], "std": []}
    bounds = {}
    if clip_quantiles is not None:
        lo, hi = np.nanpercentile(x, clip_quantiles, axis=0)
        x = np.clip(x, lo, hi)
        bounds = {'clip_low': lo.tolist(), 'clip_high': hi.tolist()}
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    xs = (x - mean) / std
    xb = np.column_stack([np.ones(xs.shape[0]), xs])
    reg = np.eye(xb.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    weights = np.linalg.pinv(xb.T @ xb + reg) @ xb.T @ y
    return weights, {"mean": mean.tolist(), "std": std.tolist(), **bounds}


def _predict(x: np.ndarray, weights: np.ndarray, scaler: dict) -> np.ndarray:
    if x.size == 0:
        return np.empty((0,))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = np.asarray(scaler.get("mean") or [0] * x.shape[1], dtype=float)
    std = np.asarray(scaler.get("std") or [1] * x.shape[1], dtype=float)
    n_features = max(0, len(weights) - 1)
    if n_features and x.shape[1] != n_features:
        if x.shape[1] > n_features:
            x = x[:, :n_features]
        else:
            pad = np.zeros((x.shape[0], n_features - x.shape[1]))
            x = np.column_stack([x, pad])
    if mean.size != x.shape[1]:
        mean = np.resize(mean, x.shape[1]) if mean.size else np.zeros(x.shape[1])
    if std.size != x.shape[1]:
        std = np.resize(std, x.shape[1]) if std.size else np.ones(x.shape[1])
    if 'clip_low' in scaler:
        x = np.clip(x, np.asarray(scaler['clip_low']), np.asarray(scaler['clip_high']))
    std = np.where(std < 1e-8, 1.0, std)
    xs = (x - mean) / std
    xb = np.column_stack([np.ones(xs.shape[0]), xs])
    return xb @ weights


def _fit_xgboost(x: np.ndarray, y: np.ndarray, *, seed: int = 42,
                 eval_x: np.ndarray | None = None, eval_y: np.ndarray | None = None,
                 n_estimators: int = 600):
    """Fit the production tabular model without importing xgboost at app boot."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("XGBoost 未安装，请重新构建 Docker 镜像") from exc
    if x.size == 0:
        raise ValueError("XGBoost 训练数据为空")
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_alpha=0.05,
        reg_lambda=2.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        verbosity=0,
    )
    fit_kwargs = {}
    if eval_x is not None and eval_y is not None and len(eval_y) > 0:
        fit_kwargs["eval_set"] = [(np.nan_to_num(eval_x), np.nan_to_num(eval_y))]
        fit_kwargs["verbose"] = False
        fit_kwargs["early_stopping_rounds"] = 50
    try:
        model.fit(np.nan_to_num(x), np.nan_to_num(y), **fit_kwargs)
    except TypeError:
        fit_kwargs.pop("early_stopping_rounds", None)
        model.fit(np.nan_to_num(x), np.nan_to_num(y), **fit_kwargs)
    return model


def _predict_xgboost(model, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(np.nan_to_num(x)), dtype=float)


def _fit_lightgbm(x: np.ndarray, y: np.ndarray, *, seed: int = 42,
                  eval_x: np.ndarray | None = None, eval_y: np.ndarray | None = None,
                  n_estimators: int = 600):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("LightGBM 未安装，请重新构建 Docker 镜像") from exc
    if x.size == 0:
        raise ValueError("LightGBM 训练数据为空")
    model = LGBMRegressor(
        n_estimators=n_estimators, num_leaves=20, max_depth=5, learning_rate=0.025,
        min_child_samples=30, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=2.0, objective="regression",
        n_jobs=2, random_state=seed, verbosity=-1,
    )
    fit_kwargs = {}
    if eval_x is not None and eval_y is not None and len(eval_y) > 0:
        try:
            from lightgbm import early_stopping
            fit_kwargs["eval_set"] = [(np.nan_to_num(eval_x), np.nan_to_num(eval_y))]
            fit_kwargs["callbacks"] = [early_stopping(50, verbose=False)]
        except ImportError:
            pass
    model.fit(np.nan_to_num(x), np.nan_to_num(y), **fit_kwargs)
    return model


def _predict_lightgbm(model, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(np.nan_to_num(x)), dtype=float)


def _fit_extreme_prediction_calibration(
    prediction: np.ndarray,
    actual: np.ndarray,
    *,
    tail_quantile: float = 0.90,
    min_tail_rows: int = 20,
) -> dict:
    """Fit a train-only reliability correction for extreme predictions.

    The production rank model has repeatedly produced overly optimistic upper
    tails.  Fit the correction on the chronological calibration slice only and
    keep the middle of the score distribution untouched.  A robust linear fit
    in each tail preserves a little ordering while anchoring the tail to the
    realized target level.  The returned JSON contains no model objects and is
    safe to persist with the model artifact.
    """
    pred = np.asarray(prediction, dtype=float)
    y = np.asarray(actual, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(y)
    pred, y = pred[valid], y[valid]
    q = float(np.clip(tail_quantile, 0.75, 0.98))
    if len(pred) < min_tail_rows * 2:
        return {"enabled": False, "reason": "insufficient_calibration_rows", "rows": int(len(pred))}
    low_q, high_q = np.quantile(pred, [1.0 - q, q])
    low_mask = pred <= low_q
    high_mask = pred >= high_q
    if int(low_mask.sum()) < min_tail_rows or int(high_mask.sum()) < min_tail_rows:
        return {"enabled": False, "reason": "insufficient_tail_rows", "rows": int(len(pred))}

    def fit_tail(mask: np.ndarray) -> tuple[float, float]:
        x = pred[mask]
        target = y[mask]
        if len(x) < 2 or np.std(x) < 1e-10:
            return 0.0, float(np.median(target))
        # Keep only a small positive slope so calibration corrects the level of
        # an extreme tail without amplifying unstable extrapolation or letting
        # a noisy tail reverse the ranking of every extreme observation.
        slope = 0.10
        intercept = float(np.median(target) - slope * np.median(x))
        return slope, intercept

    low_slope, low_intercept = fit_tail(low_mask)
    high_slope, high_intercept = fit_tail(high_mask)
    y_low, y_high = np.quantile(y, [0.02, 0.98])
    return {
        "enabled": True,
        "version": "extreme_tail_piecewise_v1",
        "rows": int(len(pred)),
        "tail_quantile": q,
        "prediction_low_quantile": float(low_q),
        "prediction_high_quantile": float(high_q),
        "low_slope": low_slope,
        "low_intercept": low_intercept,
        "high_slope": high_slope,
        "high_intercept": high_intercept,
        "actual_clip_low": float(y_low),
        "actual_clip_high": float(y_high),
        "low_tail_actual_median": float(np.median(y[low_mask])),
        "high_tail_actual_median": float(np.median(y[high_mask])),
    }


def _apply_extreme_prediction_calibration(
    prediction: np.ndarray,
    calibration: dict | None,
) -> np.ndarray:
    """Apply a previously frozen train-only tail calibration."""
    pred = np.asarray(prediction, dtype=float).copy()
    if not calibration or not calibration.get("enabled") or pred.size == 0:
        return pred
    try:
        low_q = float(calibration["prediction_low_quantile"])
        high_q = float(calibration["prediction_high_quantile"])
        low_slope = float(calibration["low_slope"])
        low_intercept = float(calibration["low_intercept"])
        high_slope = float(calibration["high_slope"])
        high_intercept = float(calibration["high_intercept"])
        clip_low = float(calibration["actual_clip_low"])
        clip_high = float(calibration["actual_clip_high"])
    except (KeyError, TypeError, ValueError):
        return pred
    low = np.isfinite(pred) & (pred <= low_q)
    high = np.isfinite(pred) & (pred >= high_q)
    if low.any():
        pred[low] = low_slope * pred[low] + low_intercept
    if high.any():
        pred[high] = high_slope * pred[high] + high_intercept
    return np.clip(pred, clip_low, clip_high)


def _corr(a: np.ndarray, b: np.ndarray, rank: bool = False) -> float | None:
    if len(a) < 3 or len(b) < 3:
        return None
    if rank:
        a = np.asarray(pl.Series(a).rank(method="average").to_list(), dtype=float)
        b = np.asarray(pl.Series(b).rank(method="average").to_list(), dtype=float)
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else None


def _oos_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    valid = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[valid]
    actual = actual[valid]
    if len(pred) == 0:
        return {"rows": 0}
    ic = _corr(pred, actual, rank=False)
    rank_ic = _corr(pred, actual, rank=True)
    direction_acc = float(np.mean((pred > 0) == (actual > 0)))
    top_cut = max(1, int(len(pred) * 0.2))
    order = np.argsort(pred)
    top = actual[order[-top_cut:]]
    bottom = actual[order[:top_cut]]
    long_short = float(np.nanmean(top) - np.nanmean(bottom)) if len(bottom) else None
    return {
        "rows": int(len(pred)),
        "ic": round(ic, 4) if ic is not None else None,
        "rank_ic": round(rank_ic, 4) if rank_ic is not None else None,
        "direction_acc": round(direction_acc, 4),
        "top_quantile_return": round(float(np.nanmean(top)), 4) if len(top) else None,
        "bottom_quantile_return": round(float(np.nanmean(bottom)), 4) if len(bottom) else None,
        "long_short_return": round(long_short, 4) if long_short is not None else None,
    }


def _nearest_horizon(days: int) -> int:
    target = max(1, int(days or 1))
    return min(HORIZONS, key=lambda h: abs(h - target))


def _active_horizon_weights(meta: dict | None, available: list[int] | None = None) -> dict[int, float]:
    candidates = [h for h in HORIZONS if available is None or h in available]
    if not candidates:
        return {}
    frozen_present = 'signal_weights' in (meta or {})
    frozen = (meta or {}).get('signal_weights') or {}
    role_weights = (meta or {}).get("horizon_role_weights") or {}
    explicit = {h: _safe_float(frozen.get(f'{h}d'), 0.) or 0. for h in candidates}
    total = sum(max(0., v) for v in explicit.values())
    if total > 0:
        # Frozen weights already encode training/validation selection.
        return {h: round(v / total, 6) for h, v in explicit.items() if v > 0}
    # New research models explicitly freeze an empty map when no horizon clears
    # validation quality gates. Do not silently reactivate a failed model.
    if frozen_present:
        return {}
    if role_weights:
        selected = {h: _safe_float(role_weights.get(f"{h}d"), 0.0) or 0.0 for h in candidates}
        total = sum(max(0.0, value) for value in selected.values())
        if total > 0:
            return {h: round(max(0.0, value) / total, 6) for h, value in selected.items() if value > 0}
    # Historical holdout scores are diagnostics, never tuning inputs.
    metrics = ((meta or {}).get('validation') or {}).get('metrics') or {}
    raw: dict[int, float] = {}
    for horizon in candidates:
        item = metrics.get(f"{horizon}d") or {}
        rank_ic = _safe_float(item.get("rank_ic"), 0.0) or 0.0
        long_short = _safe_float(item.get("long_short_return"), 0.0) or 0.0
        top_return = _safe_float(item.get("top_quantile_return"), 0.0) or 0.0
        quality = max(rank_ic, 0.0) + max(long_short, 0.0) * 2.0 + max(top_return, 0.0)
        if quality > 0:
            raw[horizon] = quality
    if not raw:
        fallback = {1: 0.20, 3: 0.30, 5: 0.30, 10: 0.20}
        raw = {h: fallback.get(h, 1.0) for h in candidates}
    total = sum(raw.values())
    return {h: round(v / total, 6) for h, v in raw.items() if total > 0}


def _add_adaptive_stockformer_score(df: pl.DataFrame, meta: dict | None) -> tuple[pl.DataFrame, dict[int, float]]:
    available = [h for h in HORIZONS if f"pred_ret_{h}d" in df.columns]
    weights = _active_horizon_weights(meta, available)
    if not weights:
        return df.with_columns(
            pl.lit(0.0).alias("stockformer_blend"),
            pl.lit(50.0).alias("stockformer_score"),
        ), {}
    blend = pl.lit(0.0)
    # Rolling OOS frames contain multiple signal dates.  Calibrate each
    # session against its own cross-section so a later prediction distribution
    # cannot change an earlier window's score.
    has_date_scope = "date" in df.columns
    for horizon, weight in weights.items():
        column = f"pred_ret_{horizon}d"
        scope = (lambda expr: expr.over("date")) if has_date_scope else (lambda expr: expr)
        q25 = scope(pl.col(column).quantile(0.25))
        q50 = scope(pl.col(column).quantile(0.50))
        q75 = scope(pl.col(column).quantile(0.75))
        calibrated = ((pl.col(column).fill_null(q50) - q50) / (q75 - q25).clip(lower_bound=1e-6))
        df = df.with_columns(calibrated.clip(-4.0, 4.0).alias(f"pred_calibrated_{horizon}d"))
        blend = blend + pl.col(f"pred_calibrated_{horizon}d").fill_null(0.0) * weight
    df = df.with_columns(blend.alias("stockformer_blend"))
    df = df.with_columns((_pct_rank_expr("stockformer_blend") * 100).alias("stockformer_score"))
    return df, weights


def list_models(data_dir: Path) -> dict:
    models = [_normalize_model_meta(data_dir, item) for item in _read_models(data_dir)]
    return _json_safe({"models": models, "latest_model_id": _latest_model_id(data_dir)})


def train_model(
    data_dir: Path,
    config: StockFormerTrainConfig,
    *,
    activate: bool = False,
    on_progress=None,
) -> dict:
    from app.services import stockformer_training as training
    t0 = time.perf_counter()

    def progress(stage: str, pct: int, message: str) -> None:
        if on_progress is not None:
            on_progress(stage, pct, message)

    try:
        if config.execution_costs is not None:
            if activate:
                raise ValueError("真实模拟费用标签目前仅允许研究训练，不能直接激活")
            if config.target_mode not in {"net", "industry_relative_net"}:
                raise ValueError("模拟费用训练必须使用 net 或 industry_relative_net 目标")
        training_profile = str(config.training_profile or "daily_baseline").strip().lower()
        feature_columns = training_feature_columns(training_profile)
        progress("prepare", 8, "读取训练日期和本地行情")
        min_date, max_date = _date_range_from_data(data_dir)
        if min_date is None or max_date is None:
            return {"error": "本地 kline_daily_enriched 为空，请先同步并重算日线数据"}
        end = config.end or max_date
        start = config.start or max(min_date, end - timedelta(days=365 * 2))
        if start >= end:
            return {"error": "训练开始日期必须早于结束日期"}
        panel = _load_panel(data_dir, start - timedelta(days=120), end, max_symbols=config.max_symbols,
                            universe_as_of=start, repair_adjusted=True,
                            include_minute=training_profile == "minute_enhanced",
                            on_minute_progress=lambda done, total, day: progress(
                                "prepare", 9 + int(12 * done / max(1, total)),
                                f"分钟特征逐日聚合 {done}/{total}，已完成 {day}",
                            ))
        if panel.is_empty():
            return {"error": "训练区间无可用行情数据"}
        progress("prepare", 22, f"已加载 {panel['symbol'].n_unique()} 只股票的日K")
        base_features = _feature_frame(panel, include_targets=False)
        if config.execution_costs is not None:
            if config.target_mode == 'industry_relative_net' and 'industry' not in base_features.columns:
                raise ValueError('行业净收益目标缺少行业分类，拒绝退回绝对收益')
            raw_prices, conflicts = training.load_paper_raw_prices(
                data_dir, panel['symbol'].unique().to_list(), start=panel['date'].min(), end=end,
            )
            if conflicts.height:
                logger.warning("paper-cost research excludes %s conflicting raw-price keys", conflicts.height)
            all_features = training.paper_execution_targets(
                base_features, panel, raw_prices, execution=config.execution_costs,
            )
        else:
            all_features = training.execution_targets(base_features, panel)
        all_features = all_features.filter((pl.col('date') >= start) & (pl.col('date') <= end))
        dates = sorted(all_features["date"].unique().to_list())
        if len(dates) < 80:
            return {"error": "有效交易日不足，至少需要约 80 个交易日"}
        progress("prepare", 38, f"特征准备完成，共 {len(dates)} 个交易日")
        features, coverage = prepare_training_features(all_features, training_profile)
        if training_profile == "minute_enhanced":
            missing_minute_columns = [column for column in MINUTE_FEATURE_COLUMNS if column not in features.columns]
            if missing_minute_columns:
                return {
                    "error": "分钟增强模型缺少分钟特征列，不能用默认值补齐: " + ", ".join(missing_minute_columns),
                    "training_coverage": coverage,
                }
            if features.is_empty():
                return {
                    "error": (
                        "分钟增强模型没有通过分钟密度及量价覆盖门槛的股票日"
                        f"（至少 {MINUTE_MIN_VALID_BARS} 根，覆盖率 = 1.0）"
                    ),
                    "training_coverage": coverage,
                }
            ready_dates = sorted(features["date"].unique().to_list())
            if len(ready_dates) < 80:
                return {
                    "error": (
                        f"分钟增强模型有效交易日仅 {len(ready_dates)} 天，至少需要约 80 天；"
                        "请先补齐分钟历史，不能用中性值替代"
                    ),
                    "training_coverage": {**coverage, "ready_dates": len(ready_dates)},
                }
        # Fill only non-minute auxiliary defaults. Enhanced rows have already
        # passed the minute gate; missing minute values must remain missing and
        # be rejected by _prepare_xy instead of becoming neutral observations.
        features = _fill_prediction_feature_defaults(
            features,
            [column for column in feature_columns if column not in MINUTE_FEATURE_COLUMNS],
        )
        split_index = max(1, min(len(dates) - 10, int(len(dates) * (1 - config.sample_out_ratio))))
        split_date = dates[split_index]
        train_df = features.filter(pl.col("date") < split_date)
        oos_df = features.filter(pl.col("date") >= split_date)
        if train_df.is_empty() or oos_df.is_empty():
            return {
                "error": "样本内或样本外为空，请调整日期或补齐对应训练配置的数据覆盖",
                "training_profile": training_profile,
                "training_coverage": coverage,
                "ready_dates": int(features["date"].n_unique()) if "date" in features.columns else 0,
            }

        progress("fit", 52, "开始滚动验证和特征族选择")

        # Use the latter half of the pre-holdout history for expanding
        # walk-forward selection. A short final validation window can belong
        # to one market regime and produce unstable horizon/feature choices.
        validation_index = max(20, int(split_index * .5))
        validation_index = min(validation_index, split_index - 12)
        validation_date = dates[validation_index]
        target_mode = str(config.target_mode or "absolute").strip().lower()
        if target_mode not in {"absolute", "industry_relative", "net", "industry_relative_net"}:
            return {"error": "target_mode 只支持 absolute、industry_relative、net 或 industry_relative_net"}
        input_manifest = training.input_manifest(panel, features)
        input_manifest.update(
            captured_at=datetime.now().isoformat(), split_date=str(split_date),
            training_profile=training_profile, target_mode=target_mode,
            feature_columns=feature_columns, validation_start=str(validation_date),
            live_refit_through=str(end),
            execution_costs=config.execution_costs,
            label_cost_version=training.PAPER_COST_TARGET_VERSION if config.execution_costs is not None else 'fixed_20bp_v1',
            minute_availability_policy=dict(MINUTE_AVAILABILITY_POLICY) if training_profile == 'minute_enhanced' else None,
        )
        fitted = training.fit_candidates(
            features, feature_columns, validation_date, split_date, config.ridge_alpha,
            target_mode=target_mode,
        )
        rolling_ablation = training.rolling_ablation(
            features, feature_columns, split_date, target_mode=target_mode,
        )
        progress("fit", 70, "滚动验证完成，开始生成模型文件")
        models = fitted['models']
        oos_metrics = fitted['oos_metrics']
        prediction_frames = fitted['prediction_frames']

        model_id = f"sf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        model_dir = _models_dir(data_dir) / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "model.json"
        factor_file = model_dir / "oos_factors.parquet"
        prediction_file = model_dir / "latest_predictions.parquet"
        manifest_file = model_dir / "training_inputs.json"
        manifest_file.write_text(json.dumps(input_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

        # XGBoost is the production tabular model.  Keep the selected linear
        # models in the artifact so every training run has an apples-to-apples
        # baseline for later diagnostics.
        xgb_models = {}
        xgb_oos_metrics = {}
        xgb_oos_prediction_frames = {}
        xgb_calibrations = {}
        for horizon in HORIZONS:
            ridge_model = fitted['models'].get(str(horizon))
            if not ridge_model:
                continue
            columns = ridge_model.get('feature_columns') or feature_columns
            target = training._target_column(features, horizon, target_mode)
            refit_part = training.purged_slice(features, horizon, features['date'].min(), split_date)
            refit_frame = refit_part.with_columns(
                training._fit_target_expr(target, target_mode).alias('_fit_target')
            )
            inner_train, inner_eval, inner_cut = training.purged_inner_split(refit_frame, horizon)
            rx, ry, _ = _prepare_xy(inner_train, '_fit_target', columns)
            ex, ey, _ = _prepare_xy(inner_eval, '_fit_target', columns) if not inner_eval.is_empty() else (None, None, None)
            tuning_xgb = _fit_xgboost(rx, ry, eval_x=ex, eval_y=ey) if ex is not None and len(ey) else _fit_xgboost(rx, ry)
            tuning_lgbm = _fit_lightgbm(rx, ry, eval_x=ex, eval_y=ey) if ex is not None and len(ey) else _fit_lightgbm(rx, ry)
            blend_weight = 0.6
            prediction_calibration = {
                "enabled": False,
                "reason": "no_chronological_calibration_slice",
                "rows": 0,
            }
            if not inner_eval.is_empty():
                vx, vy, vused = _prepare_xy(inner_eval, target, columns)
                if len(vy) and vused.height:
                    px = _predict_xgboost(tuning_xgb, vx)
                    plgb = _predict_lightgbm(tuning_lgbm, vx)
                    blend_candidates = []
                    for xgb_weight in np.arange(0.0, 1.01, 0.1):
                        score = training.metrics(
                            vused, xgb_weight * px + (1.0 - xgb_weight) * plgb, vy,
                            target_mode=target_mode,
                        )
                        blend_candidates.append((
                            int(score.get("positive_rank_ic_folds", 0)),
                            _safe_float(score.get("rank_ic"), -999.0) or -999.0,
                            _safe_float(score.get("long_short_return"), -999.0) or -999.0,
                            float(xgb_weight),
                        ))
                    if blend_candidates:
                        blend_weight = max(blend_candidates)[3]
                    validation_prediction = blend_weight * px + (1.0 - blend_weight) * plgb
                    prediction_calibration = _fit_extreme_prediction_calibration(
                        validation_prediction, vy,
                    )
            prediction_calibration.update(
                split_policy='purged_before_calibration_v1',
                calibration_start=str(inner_cut) if inner_cut else None,
                inner_train_max_label_end=str(inner_train[f'label_end_{horizon}d'].max()) if not inner_train.is_empty() else None,
            )
            xgb_rounds = int(getattr(tuning_xgb, "best_iteration", 599) or 599) + 1
            lgb_rounds = int(getattr(tuning_lgbm, "best_iteration_", 599) or 599) + 1
            selected_rounds = max(40, min(600, round((xgb_rounds + lgb_rounds) / 2)))
            # First evaluate the frozen pre-holdout model.  These predictions
            # are the only ones allowed to produce the OOS diagnostics.
            rx, ry, _ = _prepare_xy(refit_frame, '_fit_target', columns)
            holdout_booster = _fit_xgboost(rx, ry, n_estimators=selected_rounds)
            holdout_lgbm = _fit_lightgbm(rx, ry, n_estimators=selected_rounds)
            test_part = training.purged_slice(features, horizon, split_date)
            tx, ty, used = _prepare_xy(test_part, target, columns)
            blended_raw = blend_weight * _predict_xgboost(holdout_booster, tx) + (1.0 - blend_weight) * _predict_lightgbm(holdout_lgbm, tx)
            blended = _apply_extreme_prediction_calibration(
                blended_raw, prediction_calibration,
            )
            xgb_oos_metrics[f'{horizon}d'] = training.metrics(used, blended, ty, target_mode=target_mode)
            xgb_oos_prediction_frames[horizon] = used.select(['symbol', 'date']).with_columns(
                pl.Series('_prediction', blended),
                pl.Series('_prediction_raw', blended_raw),
                pl.Series('_actual', ty),
            )
            xgb_calibrations[str(horizon)] = prediction_calibration

            # Each horizon needs its own live artifact; these values are
            # horizon-specific and cannot be reused after the loop.
            live_part = features.filter(pl.col('date') < end).with_columns(
                training._fit_target_expr(target, target_mode).alias('_fit_target')
            )
            lx, ly, _ = _prepare_xy(live_part, '_fit_target', columns)
            booster = _fit_xgboost(lx, ly, n_estimators=selected_rounds)
            lgbm = _fit_lightgbm(lx, ly, n_estimators=selected_rounds)
            xgb_file = model_dir / f'xgboost_{horizon}d.json'
            booster.save_model(str(xgb_file))
            lgbm_file = model_dir / f'lightgbm_{horizon}d.txt'
            lgbm.booster_.save_model(str(lgbm_file))
            xgb_models[str(horizon)] = {
                'model_type': 'gbdt_ensemble',
                'xgboost_file': _store_relpath(data_dir, xgb_file),
                'lightgbm_file': _store_relpath(data_dir, lgbm_file),
                'weights': {'xgboost': round(blend_weight, 2), 'lightgbm': round(1.0 - blend_weight, 2)},
                'n_estimators': selected_rounds,
                'feature_columns': columns,
                'prediction_calibration': prediction_calibration,
            }

        # The executable OOS artifact must match the production GBDT model.
        # The linear models remain in ``linear_baseline_models`` for research,
        # but using their factors for a real-fill replay made calibration and
        # model selection silently diverge from live predictions.
        if xgb_oos_prediction_frames:
            prediction_frames = []
            for horizon, frame in xgb_oos_prediction_frames.items():
                prediction_frames.append(
                    frame.select(['symbol', 'date', '_prediction', '_prediction_raw', '_actual'])
                    .rename({
                        '_prediction': f'pred_ret_{horizon}d',
                        '_prediction_raw': f'pred_ret_raw_{horizon}d',
                        '_actual': f'target_{horizon}d',
                    })
                )

        recent_validation = {}
        recent_dates = sorted(oos_df['date'].unique().to_list())[-63:]
        if recent_dates:
            recent_start = recent_dates[0]
            for horizon, frame in xgb_oos_prediction_frames.items():
                recent = frame.filter(pl.col('date') >= recent_start)
                if recent.height:
                    recent_validation[f'{horizon}d'] = training.metrics(
                        recent.select(['date', 'symbol']),
                        recent['_prediction'].to_numpy(),
                        recent['_actual'].to_numpy(),
                        target_mode=target_mode,
                    )


        payload = {
            "model_id": model_id,
            "model_kind": GBDT_ENSEMBLE_MODEL_KIND,
            "feature_version": FEATURE_VERSION,
            "feature_columns": feature_columns,
            "horizons": list(HORIZONS),
            "models": xgb_models,
            "linear_baseline_models": models,
            'training_protocol': training.PROTOCOL,
            'target_mode': target_mode,
            'target_definition': (
                'industry_cross_sectional_excess_return_next_session_open_to_open_h_sessions_later'
                if target_mode == 'industry_relative'
                else 'industry_neutral_net_of_round_trip_cost_next_session_open_to_open'
                if target_mode == 'industry_relative_net'
                else 'net_of_round_trip_cost_next_session_open_to_open'
                if target_mode in {'net', 'cost_after'}
                else 'industry_neutral_next_session_open_to_open_h_sessions_later'
                if target_mode == 'industry_relative'
                else 'next_session_open_to_open_h_sessions_later'
            ),
            'oos_model_training_through': str(split_date),
            'live_refit_through': str(end),
            'model_data_mode': (
                'daily_kline_baseline'
                if training_profile == 'daily_baseline'
                else 'daily_kline_plus_minute_strict'
            ),
            'training_profile': training_profile,
            'training_coverage': coverage,
            'training_universe': sorted({str(value) for value in features['symbol'].unique().to_list()}),
            'prediction_universe_policy': '训练股票池; 显式指定 symbols 时按指定股票池',
            'horizon_role_weights': {'1d': 0.40, '3d': 0.35, '5d': 0.15, '10d': 0.10},
            'rolling_oos_protocol': {
                'enabled': True,
                'windows': [
                    {'train_months': 12, 'test_months': 3},
                    {'train_months': 15, 'test_months': 3},
                    {'train_months': 18, 'test_months': 3},
                    {'train_months': 'recent', 'test_months': 3},
                ],
                'selection_rule': '只用训练期和隔离验证期选择; 样本外只追加评估',
            },
            'rolling_ablation': rolling_ablation,
            'recent_oos_validation': {
                'test_trading_days': len(recent_dates),
                'start': str(recent_dates[0]) if recent_dates else None,
                'end': str(recent_dates[-1]) if recent_dates else None,
                'metrics': recent_validation,
                'selection_rule': '仅用于最新市场状态验证，不参与模型调参或自动激活',
            },
            'minute_coverage_policy': (
                'daily baseline excludes minute features; minute gaps do not affect this model'
                if training_profile == 'daily_baseline'
                else (
                    f'minute-incomplete, conflicting, or fewer than {MINUTE_MIN_VALID_BARS} '
                    'rows are excluded; no minute default or neutral imputation'
                )
            ),
            'prediction_calibration': xgb_calibrations,
            'minute_availability_policy': dict(MINUTE_AVAILABILITY_POLICY) if training_profile == 'minute_enhanced' else None,
            'training_input_manifest_file': _store_relpath(data_dir, manifest_file),
            'training_input_manifest_sha256': manifest_sha256,
            'cost_metrics_version': training.COST_METRICS_VERSION,
            'execution_costs': config.execution_costs,
            'label_cost_version': input_manifest['label_cost_version'],
        }
        model_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if prediction_frames:
            # 不直接连续 outer join: Polars 会为每次连接生成重复的
            # symbol_right/date_right 列, 第二次连接容易触发 DuplicateError。
            # 先建立唯一键表, 再逐个左连接每个 horizon 的预测和标签。
            keys = (
                pl.concat(
                    [frame.select(["symbol", "date"]) for frame in prediction_frames],
                    how="vertical",
                )
                .unique(subset=["symbol", "date"])
                .sort(["date", "symbol"])
            )
            factor = keys
            for frame in prediction_frames:
                factor = factor.join(frame, on=["symbol", "date"], how="left")
            factor.write_parquet(factor_file)
        else:
            pl.DataFrame(
                {
                    "symbol": pl.Series([], dtype=pl.Utf8),
                    "date": pl.Series([], dtype=pl.Date),
                }
            ).write_parquet(factor_file)

        config_dict = asdict(config)
        config_dict["start"] = str(config.start) if config.start else None
        config_dict["end"] = str(config.end) if config.end else None

        meta = StockFormerModelMeta(
            model_id=model_id,
            model_name=f"StockFormer {model_id[-6:]}",
            model_kind=GBDT_ENSEMBLE_MODEL_KIND,
            feature_version=FEATURE_VERSION,
            trained_at=datetime.now().isoformat(timespec="seconds"),
            train_start=str(start),
            train_end=str(dates[split_index - 1]),
            oos_start=str(split_date),
            oos_end=str(end),
            horizons=list(HORIZONS),
            feature_columns=feature_columns,
            model_file=_store_relpath(data_dir, model_file),
            factor_file=_store_relpath(data_dir, factor_file),
            prediction_file=_store_relpath(data_dir, prediction_file),
            oos_metrics=xgb_oos_metrics,
            config=config_dict,
            n_symbols=features["symbol"].n_unique(),
            n_rows_train=max(item['refit']['rows'] for item in fitted['validation']['horizons'].values()),
            n_rows_oos=oos_df.height,
            elapsed_ms=0.0,
        )
        existing = [m for m in _read_models(data_dir) if m.get("model_id") != model_id]
        meta_dict = _json_safe(asdict(meta))
        meta_dict.update(
            training_protocol=training.PROTOCOL,
            training_input_manifest_file=payload['training_input_manifest_file'],
            training_input_manifest_sha256=manifest_sha256,
            cost_metrics_version=training.COST_METRICS_VERSION,
            execution_costs=config.execution_costs,
            label_cost_version=payload['label_cost_version'],
            target_definition=payload['target_definition'],
            model_data_mode=payload['model_data_mode'],
            training_profile=training_profile,
            training_coverage=coverage,
            target_mode=payload['target_mode'],
            oos_model_training_through=payload['oos_model_training_through'],
            live_refit_through=payload['live_refit_through'],
            minute_coverage_policy=payload['minute_coverage_policy'],
            minute_availability_policy=payload['minute_availability_policy'],
            validation=fitted['validation'], signal_weights=fitted['signal_weights'],
            horizon_role_weights=payload['horizon_role_weights'],
            rolling_oos_protocol=payload['rolling_oos_protocol'],
            rolling_ablation=payload['rolling_ablation'],
            recent_oos_validation=payload['recent_oos_validation'],
            prediction_calibration=payload['prediction_calibration'],
            training_universe=payload['training_universe'],
            prediction_universe_policy=payload['prediction_universe_policy'],
            holdout_comparison=xgb_oos_metrics,
            promotion_status='active' if activate else 'research_only',
            training_notes=[
                '训练参数、截尾边界和周期权重仅使用训练集及隔离后的验证集, 留出集不参与选择。',
                ('收益标签按独立模拟账户的整手、实际开盘、最低佣金、税和滑点计费；条件成交标签不是组合收益。'
                 if config.execution_costs is not None else
                 '收益标签为下一交易日开盘至持有期后开盘; Top2扣费为单次20基点成本假设, 不是组合收益。'),
                '当前历史区间已多次查看, 只能用于留出诊断, 仍需要新增时间段的前向验证。',
                '当前复权版本、行业分类及新闻历史可得时间并非完整时点档案, 仍有数据版本偏差风险。',
                '预测分的极端尾部只使用训练期末段做分段校准；上涨概率与置信度仍是辅助数值, 不等于实际命中率。',
                '当前生产模型为 XGBoost 60% + LightGBM 40% 的表格模型集成；线性排序模型保存在 linear_baseline_models 中用于对照。',
                f'训练配置: {training_profile}; 两个配置必须使用相同日期切分和股票池进行对照。',
            ],
        )
        _write_models(data_dir, [meta_dict, *existing])
        if activate:
            _latest_path(data_dir).write_text(model_id, encoding="utf-8")

        progress("predict", 88, "模型文件已保存，生成最新预测")
        latest_predictions = predict(data_dir, model_id=model_id, limit=5000, write_artifact=False)
        if latest_predictions.get("rows"):
            pl.DataFrame(latest_predictions["rows"]).write_parquet(prediction_file)
        else:
            pl.DataFrame(
                {
                    "symbol": pl.Series([], dtype=pl.Utf8),
                    "date": pl.Series([], dtype=pl.Date),
                    "close": pl.Series([], dtype=pl.Float64),
                }
            ).write_parquet(prediction_file)

        meta_dict["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        _write_models(data_dir, [meta_dict, *existing])
        progress("done", 100, f"训练完成：{meta_dict['model_name']}")
        return _json_safe({"model": meta_dict, "predictions": latest_predictions})
    except Exception as exc:  # noqa: BLE001
        return {"error": f"StockFormer 训练失败: {type(exc).__name__}: {exc}", "predictions": {"rows": []}}


def _load_prediction_base(
    data_dir: Path,
    as_of: date | None = None,
    *,
    required_feature_columns: list[str] | None = None,
    strict_minute: bool = False,
    symbols: list[str] | None = None,
) -> tuple[pl.DataFrame, date | None, dict[str, Any]]:
    min_date, max_date = _date_range_from_data(data_dir)
    if max_date is None:
        return pl.DataFrame(), None, {}
    target, meta = _resolve_prediction_target_date(data_dir, max_date, as_of)
    start = target - timedelta(days=160)
    # Inference only needs the requested universe. Loading the whole market's
    # 160-day feature window here made a 200-stock training run read all 5,569
    # stocks, then get killed while the model artifact was already on disk.
    include_minute = strict_minute or any(
        column.startswith("minute_") for column in (required_feature_columns or FEATURE_COLUMNS)
    )
    panel = _load_panel(data_dir, start, target, max_symbols=None, symbols=symbols,
                        include_minute=include_minute)
    if panel.is_empty():
        return panel, target, meta
    latest_date = panel.filter(pl.col("date") <= target)["date"].max()
    if latest_date is None:
        return pl.DataFrame(), target, meta
    raw_features = _feature_frame(panel, include_targets=False)
    ready_columns = required_feature_columns or FEATURE_COLUMNS
    if strict_minute:
        features_all, _ = prepare_training_features(
            raw_features,
            "minute_enhanced",
        )
        for column in ready_columns:
            if column not in features_all.columns:
                features_all = features_all.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
        features_all = _fill_prediction_feature_defaults(
            features_all,
            [column for column in ready_columns if column not in MINUTE_FEATURE_COLUMNS],
        )
        if not features_all.is_empty():
            latest_date = features_all["date"].max()
    else:
        features_all = _fill_prediction_feature_defaults(raw_features, FEATURE_COLUMNS)
    if as_of is None and not features_all.is_empty():
        ready_counts = (
            features_all
            .drop_nulls(ready_columns)
            .group_by("date")
            .agg(pl.col("symbol").n_unique().alias("feature_ready_symbols"))
            .sort("date", descending=True)
        )
        ready = ready_counts.filter(pl.col("feature_ready_symbols") >= _prediction_cross_section_threshold())
        if not ready.is_empty():
            row = ready.row(0, named=True)
            ready_date = row["date"]
            meta["feature_ready_symbols"] = int(row.get("feature_ready_symbols") or 0)
            if ready_date != latest_date:
                all_counts = (
                    features_all
                    .group_by("date")
                    .agg(pl.col("symbol").n_unique().alias("cross_section_symbols"))
                    .join(ready_counts, on="date", how="left")
                    .filter(pl.col("date") > ready_date)
                    .sort("date", descending=True)
                    .to_dicts()
                )
                meta["skipped_unready_dates"] = [
                    {
                        "date": str(item.get("date")),
                        "symbols": int(item.get("cross_section_symbols") or 0),
                        "feature_ready_symbols": int(item.get("feature_ready_symbols") or 0),
                    }
                    for item in all_counts[:8]
                ]
                latest_date = ready_date
        else:
            meta["warning"] = (
                meta.get("warning")
                or f"最近 {_prediction_cross_section_lookback_days()} 天没有达到 "
                f"{_prediction_cross_section_threshold()} 只股票的可预测特征截面"
            )
    features = features_all.filter(pl.col("date") == latest_date)
    meta.update({
        "as_of": latest_date,
        "raw_universe_total": int(features["symbol"].n_unique()) if "symbol" in features.columns else int(features.height),
    })
    return features, latest_date, meta


def predict(
    data_dir: Path,
    *,
    model_id: str | None = None,
    symbols: list[str] | None = None,
    as_of: date | None = None,
    limit: int = 100,
    write_artifact: bool = True,
) -> dict:
    meta = _model_meta(data_dir, model_id)
    if meta is None:
        return {"error": "暂无 StockFormer 模型，请先训练模型", "rows": []}
    model_path = _resolve_artifact_path(data_dir, meta.get("model_file"))
    if model_path is None or not model_path.exists():
        return {"error": f"模型文件不存在: {meta.get('model_file')}", "rows": []}
    payload = _load_model_payload(model_path)
    feature_columns = payload.get("feature_columns") if isinstance(payload.get("feature_columns"), list) else FEATURE_COLUMNS
    feature_columns = [str(col) for col in feature_columns if str(col)]
    training_profile = str(payload.get("training_profile") or meta.get("training_profile") or "daily_baseline")
    strict_minute = training_profile == "minute_enhanced"
    minute_policy = payload.get('minute_availability_policy')
    if strict_minute and minute_policy is not None and minute_policy != MINUTE_AVAILABILITY_POLICY:
        return {'error': '模型分钟可用性口径与当前推理不一致', 'rows': []}
    training_universe = payload.get("training_universe") or meta.get("training_universe") or []
    prediction_universe = symbols or [str(value) for value in training_universe] or None
    base, latest_date, base_meta = _load_prediction_base(
        data_dir,
        as_of,
        required_feature_columns=feature_columns,
        strict_minute=strict_minute,
        symbols=prediction_universe,
    )
    if as_of is not None and latest_date != as_of:
        return {
            "error": (
                f"请求日期 {as_of.isoformat()} 没有达到"
                "分钟增强模型可预测门槛"
                if strict_minute
                else f"请求日期 {as_of.isoformat()} 没有可预测的本地行情特征"
            ),
            "requested_as_of": as_of.isoformat(),
            "as_of": latest_date.isoformat() if latest_date else None,
            "rows": [],
            "total": 0,
            "universe_total": 0,
            "cross_section": base_meta,
            "training_profile": training_profile,
        }
    if base.is_empty():
        return {"error": "没有可预测的本地行情特征", "rows": []}
    if symbols:
        base = base.filter(_symbol_in_filter(symbols))
    else:
        # Keep inference on the same liquidity/availability universe used at
        # training time. An explicit symbol request remains an intentional
        # override for research and single-stock analysis.
        if training_universe:
            base = base.filter(_symbol_in_filter([str(v) for v in training_universe]))
    for col in feature_columns:
        if col not in base.columns and (not strict_minute or col not in MINUTE_FEATURE_COLUMNS):
            base = base.with_columns(pl.lit(PREDICTION_FEATURE_DEFAULTS.get(col, 0.0)).alias(col))
    base = _fill_prediction_feature_defaults(
        base,
        [column for column in feature_columns if not strict_minute or column not in MINUTE_FEATURE_COLUMNS],
    )
    usable = base.drop_nulls(feature_columns)
    if usable.is_empty():
        return {"error": "预测特征为空", "rows": []}
    universe_total = int(usable["symbol"].n_unique()) if "symbol" in usable.columns else int(usable.height)
    base_meta["universe_total"] = universe_total
    x = usable.select(feature_columns).to_numpy().astype(float)
    out = usable.select(["symbol", "date", "close"])
    payload_models = payload.get("models") or {}
    model_horizons = [horizon for horizon in HORIZONS if str(horizon) in payload_models]
    if not model_horizons:
        return {"error": "模型文件缺少可用预测期限", "rows": []}
    for horizon in model_horizons:
        model = payload_models[str(horizon)]
        horizon_columns = model.get('feature_columns') or feature_columns
        horizon_x = usable.select(horizon_columns).to_numpy().astype(float) if horizon_columns != feature_columns else x
        if model.get('model_type') in ('xgboost', 'gbdt_ensemble'):
            xgb_artifact = _resolve_artifact_path(data_dir, model.get('xgboost_file') or model.get('model_file'))
            if xgb_artifact is None or not xgb_artifact.exists():
                return {"error": f"XGBoost 模型文件不存在: {model.get('xgboost_file') or model.get('model_file')}", "rows": []}
            try:
                from xgboost import XGBRegressor
                xgb_model = XGBRegressor()
                xgb_model.load_model(str(xgb_artifact))
                xgb_pred = _predict_xgboost(xgb_model, horizon_x)
                if model.get('model_type') == 'gbdt_ensemble':
                    lgbm_artifact = _resolve_artifact_path(data_dir, model.get('lightgbm_file'))
                    if lgbm_artifact is None or not lgbm_artifact.exists():
                        return {"error": f"LightGBM 模型文件不存在: {model.get('lightgbm_file')}", "rows": []}
                    from lightgbm import Booster
                    lgbm_model = Booster(model_file=str(lgbm_artifact))
                    weights = model.get('weights') or {}
                    pred = (float(weights.get('xgboost', 0.6)) * xgb_pred
                            + float(weights.get('lightgbm', 0.4)) * _predict_lightgbm(lgbm_model, horizon_x))
                else:
                    pred = xgb_pred
            except Exception as exc:  # noqa: BLE001
                return {"error": f"树模型加载失败: {type(exc).__name__}: {exc}", "rows": []}
        else:
            pred = _predict(horizon_x, np.asarray(model["weights"], dtype=float), model["scaler"])
        # Raw predictions are retained for diagnostics.  Trading and score
        # construction use the frozen calibration fitted on the chronological
        # training validation slice, including its extreme-tail correction.
        calibrated = _apply_extreme_prediction_calibration(
            pred, model.get("prediction_calibration"),
        )
        out = out.with_columns(
            pl.Series(f"pred_ret_raw_{horizon}d", pred),
            pl.Series(f"pred_ret_{horizon}d", calibrated),
        )
    out, signal_weights = _add_adaptive_stockformer_score(out, meta)
    signal_status = "active" if signal_weights else "no_signal"
    signal_label = "预测周期权重有效" if signal_weights else "无信号：没有可用的预测周期权重"
    out = out.with_columns(
        (1 / (1 + (-pl.col("stockformer_blend") * 100).exp())).alias("up_prob_5d"),
        (1 / (1 + (-pl.col("stockformer_blend") * 80).exp())).alias("trend_prob_10d"),
    )
    out = out.with_columns(
        (
            (pl.col("up_prob_5d") * 0.45 + pl.col("trend_prob_10d") * 0.35)
            + (pl.col("stockformer_score") / 100 * 0.20)
        ).clip(0, 1).alias("confidence"),
    ).sort("stockformer_score", descending=True).with_row_index("model_rank", offset=1)
    out = _attach_instrument_names(data_dir, out)
    # StockFormer currently trains on daily/enriched features.  Make that
    # contract explicit so a missing minute session cannot be mistaken for a
    # silently filled intraday feature.
    from app.services import minute_status
    coverage = minute_status.for_date(data_dir, out["symbol"].to_list(), latest_date)
    out = out.with_columns(
        pl.Series("minute_coverage_status", [
            str((coverage.get(symbol) or {}).get("status") or "unknown")
            for symbol in out["symbol"].to_list()
        ]),
        pl.Series("minute_coverage_minutes", [
            int((coverage.get(symbol) or {}).get("actual_minutes") or 0)
            for symbol in out["symbol"].to_list()
        ]),
        pl.lit(payload.get("model_data_mode") or meta.get("model_data_mode") or MODEL_DATA_MODE).alias("model_data_mode"),
        pl.lit(signal_status).alias("model_signal_status"),
        pl.lit(signal_label).alias("model_signal_label"),
    )
    if limit and limit > 0:
        out = out.head(limit)
    raw_rows = out.to_dicts()
    rows = _json_safe(raw_rows)
    prediction_path = _resolve_artifact_path(data_dir, meta.get("prediction_file"))
    if write_artifact and prediction_path is not None and raw_rows:
        pl.DataFrame(raw_rows).write_parquet(prediction_path)
    return _json_safe({
        "model": meta,
        "as_of": str(latest_date) if latest_date else None,
        "rows": rows,
        "total": len(rows),
        "universe_total": universe_total,
        "selection_scope": "指定股票池" if symbols else "全市场有效截面",
        "model_data_mode": payload.get("model_data_mode") or meta.get("model_data_mode") or MODEL_DATA_MODE,
        "training_profile": training_profile,
        "minute_coverage_policy": (
            f"分钟增强模型实时只预测完整{MINUTE_LIVE_MIN_VALID_BARS}根、覆盖率>=1.0且量价校验通过的股票日；稀疏分钟仅用于研究"
            if strict_minute
            else "日K基线模型不依赖分钟数据；分钟覆盖仅作为单独的盘中确认状态"
        ),
        "minute_coverage": {
            "date": str(latest_date) if latest_date else None,
            "statuses": minute_status.counts(data_dir, latest_date),
        },
        "cross_section": base_meta,
        "signal_weights": {f"{horizon}d": weight for horizon, weight in signal_weights.items()},
        "signal_status": signal_status,
        "signal_label": signal_label,
    })


def _with_score_defaults(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for column, default in SCORE_FEATURE_DEFAULTS.items():
        if column not in df.columns:
            exprs.append(pl.lit(default).alias(column))
        else:
            exprs.append(
                pl.col(column)
                .cast(pl.Float64, strict=False)
                .fill_null(default)
                .alias(column)
            )
    if "stockformer_score" not in df.columns:
        exprs.append(pl.lit(50.0).alias("stockformer_score"))
    return df.with_columns(exprs) if exprs else df


def _add_multifactor_scores(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    df = _with_score_defaults(df)
    amount_score = _score_from_feature("amount")
    df = df.with_columns(
        pl.col("gap_open").abs().alias("_abs_gap_open"),
        (pl.col("turnover_rate") - pl.col("turnover_rate").median().over("date")).abs().alias("_turnover_extreme"),
    )
    df = df.with_columns(
        (
            _score_from_feature("ma20_gap") * 0.22
            + _score_from_feature("ret_20d") * 0.18
            + _score_from_feature("vol_ratio_5d") * 0.15
            + _score_from_feature("intraday_strength") * 0.15
            + _score_from_feature("open_strength") * 0.12
            + _score_from_feature("amount_ratio_5d") * 0.10
            + _score_from_feature("limit_down_recovery_flag") * 0.04
            + _score_from_feature("limit_up_streak") * 0.04
        ).alias("technical_score"),
        (
            _score_from_feature("moneyflow_accumulation_score") * 0.30
            + _score_from_feature("moneyflow_early_layout_score") * 0.24
            + _score_from_feature("moneyflow_acceleration_3d") * 0.16
            + _score_from_feature("moneyflow_absorption_efficiency") * 0.10
            + _score_from_feature("main_net_inflow_rate") * 0.08
            + _score_from_feature("large_net_inflow_rate") * 0.06
            + _score_from_feature("moneyflow_chase_risk", invert=True) * 0.04
            + _score_from_feature("moneyflow_aging_risk", invert=True) * 0.02
        ).alias("fund_score"),
        (
            _score_from_feature("risk_20d", invert=True) * 0.22
            + _score_from_feature("amplitude", invert=True) * 0.13
            + _score_from_feature("limit_down_streak", invert=True) * 0.13
            + _score_from_feature("broken_limit_up_flag", invert=True) * 0.10
            + _score_from_feature("_abs_gap_open", invert=True) * 0.08
            + _score_from_feature("_turnover_extreme", invert=True) * 0.08
            + _score_from_feature("limit_down_flag", invert=True) * 0.08
            + _score_from_feature("sector_crowding_risk", invert=True) * 0.04
            + _score_from_feature("price_chase_risk", invert=True) * 0.12
            + _score_from_feature("moneyflow_aging_risk", invert=True) * 0.02
        ).alias("risk_score"),
        (
            _score_from_feature("sector_setup_score") * 0.36
            + _score_from_feature("sector_rotation_score") * 0.20
            + _score_from_feature("sector_fund_layout_score") * 0.16
            + _score_from_feature("sector_pre_moneyflow_strength_5d") * 0.12
            + _score_from_feature("sector_pre_amount_ratio") * 0.06
            + _score_from_feature("sector_today_confirm_score") * 0.06
            + _score_from_feature("sector_crowding_risk", invert=True) * 0.04
        ).alias("sector_score"),
        (
            _score_from_feature("industry_heat_rank") * 0.22
            + _score_from_feature("industry_ret_1d") * 0.12
            + _score_from_feature("industry_limit_up_rate") * 0.10
            + _score_from_feature("news_sentiment_score") * 0.17
            + _score_from_feature("overseas_news_sentiment_score") * 0.08
            + _score_from_feature("news_count_3d") * 0.07
            + _score_from_feature("ret_3d") * 0.04
            + _score_from_feature("sector_setup_score") * 0.12
            + _score_from_feature("sector_rotation_score") * 0.08
        ).alias("sentiment_score"),
        pl.lit(50.0).alias("fundamental_score"),
    )
    df = df.with_columns(
        (
            pl.col("technical_score") * MULTIFACTOR_WEIGHTS["technical_score"]
            + pl.col("stockformer_score") * MULTIFACTOR_WEIGHTS["stockformer_score"]
            + pl.col("fund_score") * MULTIFACTOR_WEIGHTS["fund_score"]
            + pl.col("sector_score") * MULTIFACTOR_WEIGHTS["sector_score"]
            + pl.col("sentiment_score") * MULTIFACTOR_WEIGHTS["sentiment_score"]
            + pl.col("risk_score") * MULTIFACTOR_WEIGHTS["risk_score"]
            + pl.col("fundamental_score") * MULTIFACTOR_WEIGHTS["fundamental_score"]
        ).alias("final_score")
    )
    return df.with_columns(
        (
            pl.col("final_score") * 0.52
            + _score_from_feature("moneyflow_accumulation_score") * 0.14
            + _score_from_feature("moneyflow_early_layout_score") * 0.10
            + _score_from_feature("sector_setup_score") * 0.14
            + pl.col("risk_score") * 0.08
            + pl.col("technical_score") * 0.08
        ).clip(0, 100).alias("entry_score")
    )


def _amplitude_gate(df: pl.DataFrame, quantile: float = 0.88) -> pl.Expr:
    if "date" in df.columns:
        return pl.col("amplitude") <= pl.col("amplitude").quantile(quantile).over("date")
    return pl.col("amplitude") <= pl.col("amplitude").quantile(quantile)


def _entry_eligible_expr(df: pl.DataFrame, min_entry_score: float = 62) -> pl.Expr:
    # Entry confirmation is deliberately a soft cross-sectional filter.  The
    # previous conjunction of every factor threshold made a single missing or
    # regime-shifted factor eliminate the entire OOS universe.  Keep only hard
    # execution/risk exclusions here; quality is enforced by entry_score and
    # the per-day rank in the portfolio selector.
    return (
        (pl.col("entry_score") >= min_entry_score)
        & (pl.col("limit_down_flag") < 0.5)
        & (pl.col("broken_limit_up_flag") < 0.5)
        & (pl.col("price_chase_risk").fill_null(0.0) <= 0.82)
        & (pl.col("moneyflow_chase_risk").fill_null(0.0) <= 0.88)
        & (pl.col("risk_score").fill_null(50.0) >= 38)
        & _amplitude_gate(df, 0.95)
    )


def _entry_eligible_without_sector_expr(df: pl.DataFrame) -> pl.Expr:
    return (
        (pl.col("entry_score_without_sector") >= 55)
        & (pl.col("limit_down_flag") < 0.5)
        & (pl.col("broken_limit_up_flag") < 0.5)
        & (pl.col("price_chase_risk").fill_null(0.0) <= 0.82)
        & _amplitude_gate(df, 0.95)
    )


def _attach_market_gate_for_oos(data_dir: Path, df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "date" not in df.columns:
        return df
    try:
        from app.services.market_gate import market_gate_frame

        gate_df = market_gate_frame(data_dir)
    except Exception:  # noqa: BLE001
        gate_df = pl.DataFrame()
    if gate_df.is_empty():
        return df.with_columns(
            pl.lit("neutral").alias("market_gate"),
            pl.lit(50.0).alias("market_regime_score"),
            pl.lit(50.0).alias("market_above_ma20_pct"),
            pl.lit(0.0).alias("market_strong_down_pct"),
            pl.lit(0.8).alias("market_position_multiplier"),
            pl.lit(2.0).alias("market_buy_threshold_add"),
            pl.lit(True).alias("market_allows_open"),
        )
    out = df.join(gate_df, on="date", how="left")
    defaults = {
        "market_gate": "neutral",
        "market_regime_score": 50.0,
        "market_above_ma20_pct": 50.0,
        "market_strong_down_pct": 0.0,
        "market_position_multiplier": 0.8,
        "market_buy_threshold_add": 2.0,
        "market_allows_open": True,
    }
    exprs = []
    for column, default in defaults.items():
        if column not in out.columns:
            exprs.append(pl.lit(default).alias(column))
        else:
            exprs.append(pl.col(column).fill_null(default).alias(column))
    return out.with_columns(exprs)


def _market_gate_oos_eligible_expr(df: pl.DataFrame) -> pl.Expr:
    if "market_gate" not in df.columns:
        return pl.lit(True)
    gate = pl.col("market_gate").cast(pl.Utf8, strict=False).fill_null("neutral")
    regime_ok = pl.col("market_regime_score").cast(pl.Float64, strict=False).fill_null(50.0) >= 60.0
    strict_ok = (
        (pl.col("entry_score") >= 60)
        & (pl.col("fund_score") >= 58)
        & (pl.col("risk_score") >= 50)
        & (pl.col("stockformer_score") >= 50)
        & (
            (pl.col("moneyflow_accumulation_score") >= 0.52)
            | (pl.col("moneyflow_early_layout_score") >= 0.58)
            | (pl.col("sector_setup_score") >= 0.60)
        )
    )
    return (
        (gate != "closed")
        & (gate != "observe")
        & regime_ok
        & ((gate != "strict") | strict_ok)
    )


def _market_gate_oos_relaxed_expr(df: pl.DataFrame) -> pl.Expr:
    """Ablation gate: keep all non-market rules unchanged."""
    return pl.lit(True)


def multifactor_select(
    data_dir: Path,
    *,
    model_id: str | None = None,
    as_of: date | None = None,
    limit: int = 100,
    prediction_result: dict | None = None,
) -> dict:
    pred = (prediction_result if prediction_result is not None else
            predict(data_dir, model_id=model_id, as_of=as_of, limit=0, write_artifact=False))
    if pred.get("error"):
        return pred
    if pred.get("signal_status") == "no_signal":
        return {
            "model": pred.get("model"),
            "as_of": pred.get("as_of"),
            "rows": [],
            "total": 0,
            "universe_total": pred.get("universe_total"),
            "selection_scope": pred.get("selection_scope"),
            "signal_weights": {},
            "signal_status": "no_signal",
            "signal_label": pred.get("signal_label"),
            "data_coverage": _external_feature_coverage(data_dir),
            "weights": MULTIFACTOR_WEIGHTS,
        }
    rows = pred.get("rows") or []
    if not rows:
        return {"rows": [], "total": 0, "as_of": pred.get("as_of"), "model": pred.get("model")}
    df = pl.DataFrame(rows)
    base, _, _ = _load_prediction_base(data_dir, date.fromisoformat(pred["as_of"]) if pred.get("as_of") else as_of)
    if not base.is_empty():
        base_cols = ["symbol", *[c for c in ["amount", *FEATURE_COLUMNS] if c in base.columns]]
        df = df.join(base.select(base_cols), on="symbol", how="left")
    df = _add_multifactor_scores(df)
    df = _filter_special_treatment(_attach_instrument_names(data_dir, df))
    eligible_df = df.filter(_entry_eligible_expr(df))
    if not eligible_df.is_empty():
        df = eligible_df
    sort_score = "entry_score" if "entry_score" in df.columns else "final_score"
    df = df.sort(sort_score, descending=True).with_row_index("rank", offset=1)
    if limit and limit > 0:
        df = df.head(limit)
    public_df = df.drop([column for column in df.columns if column.startswith("_")])
    return _json_safe({
        "model": pred.get("model"),
        "as_of": pred.get("as_of"),
        "rows": public_df.to_dicts(),
        "total": public_df.height,
        "universe_total": pred.get("universe_total"),
        "selection_scope": pred.get("selection_scope"),
        "cross_section": pred.get("cross_section"),
        "signal_weights": pred.get("signal_weights"),
        "data_coverage": _external_feature_coverage(data_dir),
        "weights": MULTIFACTOR_WEIGHTS,
    })


def factor_analysis(data_dir: Path, *, model_id: str | None = None, horizon: int = 5, n_groups: int = 5) -> dict:
    meta = _model_meta(data_dir, model_id)
    if meta is None:
        return {"error": "暂无 StockFormer 模型，请先训练模型"}
    path = _resolve_artifact_path(data_dir, meta.get("factor_file"))
    if path is None or not path.exists():
        return {"error": f"模型缺少样本外因子文件: {meta.get('factor_file')}"}
    df = pl.read_parquet(path)
    pred_col = f"pred_ret_{horizon}d"
    target_col = f"target_{horizon}d"
    if df.is_empty() or pred_col not in df.columns or target_col not in df.columns:
        return {"error": f"样本外因子文件缺少 {pred_col}/{target_col}"}
    df = df.filter(pl.col(pred_col).is_not_null() & pl.col(target_col).is_not_null())
    if df.is_empty():
        return {"error": "样本外因子有效行为空"}
    ic_df = (
        df.group_by("date")
        .agg(
            pl.corr(pl.col(pred_col), pl.col(target_col)).alias("ic"),
            pl.corr(pl.col(pred_col).rank(method="average"), pl.col(target_col).rank(method="average")).alias("rank_ic"),
        )
        .sort("date")
    )
    ic_values = [x for x in ic_df["ic"].to_list() if x is not None and math.isfinite(float(x))]
    rank_ic_values = [x for x in ic_df["rank_ic"].to_list() if x is not None and math.isfinite(float(x))]
    grouped = (
        df.with_columns(
            ((pl.col(pred_col).rank(method="ordinal").over("date") - 1) * n_groups / pl.len().over("date"))
            .floor()
            .clip(0, n_groups - 1)
            .cast(pl.Int32)
            .alias("group")
        )
        .group_by(["date", "group"])
        .agg(pl.mean(target_col).alias("ret"))
        .sort(["date", "group"])
    )
    group_stats = (
        grouped.group_by("group")
        .agg(
            pl.mean("ret").alias("avg_return"),
            pl.sum("ret").alias("cum_return"),
            pl.len().alias("periods"),
        )
        .sort("group")
        .to_dicts()
    )
    top = grouped.filter(pl.col("group") == n_groups - 1).select(["date", pl.col("ret").alias("top")])
    bottom = grouped.filter(pl.col("group") == 0).select(["date", pl.col("ret").alias("bottom")])
    ls = top.join(bottom, on="date", how="inner").with_columns((pl.col("top") - pl.col("bottom")).alias("long_short"))
    ls_nav = (
        ls.with_columns((1 + pl.col("long_short")).cum_prod().alias("nav"))
        .select(["date", "long_short", "nav"])
        .to_dicts()
    )
    decay = []
    for h in HORIZONS:
        pc = f"pred_ret_{h}d"
        tc = f"target_{h}d"
        if pc in df.columns and tc in df.columns:
            vals = (
                df.group_by("date")
                .agg(pl.corr(pl.col(pc).rank(method="average"), pl.col(tc).rank(method="average")).alias("rank_ic"))
                ["rank_ic"]
                .drop_nulls()
                .to_list()
            )
            decay.append({"horizon": h, "rank_ic": round(float(np.mean(vals)), 4) if vals else None})
    return _json_safe({
        "model": meta,
        "horizon": horizon,
        "ic": round(float(np.mean(ic_values)), 4) if ic_values else None,
        "rank_ic": round(float(np.mean(rank_ic_values)), 4) if rank_ic_values else None,
        "icir": round(float(np.mean(ic_values) / np.std(ic_values)), 4) if len(ic_values) > 1 and np.std(ic_values) > 1e-8 else None,
        "ic_series": [{"date": str(r["date"]), "ic": r["ic"], "rank_ic": r["rank_ic"]} for r in ic_df.to_dicts()],
        "quantile_returns": group_stats,
        "decay_curve": decay,
        "long_short_nav": [{"date": str(r["date"]), "long_short": r["long_short"], "nav": r["nav"]} for r in ls_nav],
    })


def _load_factor_feature_frame(data_dir: Path, factor: pl.DataFrame) -> pl.DataFrame:
    if factor.is_empty() or "symbol" not in factor.columns or "date" not in factor.columns:
        return pl.DataFrame()
    start = factor["date"].min()
    end = factor["date"].max()
    if start is None or end is None:
        return pl.DataFrame()
    symbols = [str(s) for s in factor["symbol"].unique().to_list() if s]
    panel = _load_panel(
        data_dir,
        start - timedelta(days=90),
        end,
        max_symbols=None,
        symbols=symbols,
    )
    if panel.is_empty():
        return pl.DataFrame()
    return _feature_frame(panel, include_targets=False).filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )


def _legacy_strict_oos_backtest(
    data_dir: Path,
    *,
    model_id: str | None = None,
    max_positions: int = 3,
    holding_days: int = 5,
    initial_capital: float = 10000.0,
    candidate_symbols: list[str] | None = None,
) -> dict:
    from app.services.stockformer_training import EVALUATION_LIMITATION
    meta = _model_meta(data_dir, model_id)
    if meta is None:
        return {"error": "暂无 StockFormer 模型，请先训练模型"}
    factor_path = _resolve_artifact_path(data_dir, meta.get("factor_file"))
    if factor_path is None or not factor_path.exists():
        return {"error": f"模型缺少样本外因子文件: {meta.get('factor_file')}"}
    factor = pl.read_parquet(factor_path)
    target_horizon = _nearest_horizon(holding_days)
    target_col = f"target_{target_horizon}d"
    available_preds = [h for h in HORIZONS if f"pred_ret_{h}d" in factor.columns]
    if factor.is_empty() or not available_preds or target_col not in factor.columns:
        return {"error": f"模型缺少样本外预测因子或 {target_col}"}
    df, signal_weights = _add_adaptive_stockformer_score(factor, meta)
    df = df.filter(pl.col("stockformer_score").is_not_null() & pl.col(target_col).is_not_null())
    if candidate_symbols:
        df = df.filter(_symbol_in_filter(candidate_symbols))
    if df.is_empty():
        return {"error": "样本外有效交易为空"}
    features = _load_factor_feature_frame(data_dir, df)
    if not features.is_empty():
        feature_cols = ["symbol", "date", *[c for c in ["amount", *FEATURE_COLUMNS] if c in features.columns]]
        df = df.join(features.select(feature_cols), on=["symbol", "date"], how="left")
    df = _filter_special_treatment(_attach_instrument_names(data_dir, df))
    if df.is_empty():
        return {"error": "样本外正常交易标的为空，请检查个股维表或放宽过滤条件"}
    df = _add_multifactor_scores(df)
    df = _attach_market_gate_for_oos(data_dir, df)
    df = df.with_columns(
        (
            pl.col("entry_score")
            - pl.col("market_buy_threshold_add").cast(pl.Float64, strict=False).fill_null(0.0) * 0.55
        ).clip(0, 100).alias("entry_score")
    )
    df = df.with_columns(
        (
            pl.col("entry_score") * 0.38
            + pl.col("stockformer_score") * 0.22
            + pl.col("fund_score") * 0.20
            + pl.col("risk_score") * 0.12
            + pl.col("technical_score") * 0.08
        ).clip(0, 100).alias("buy_confirmation_score")
    )
    df = df.with_columns(
        (
            pl.col("technical_score") * 0.25
            + pl.col("stockformer_score") * 0.20
            + pl.col("fund_score") * 0.25
            + pl.col("sentiment_score") * 0.15
            + pl.col("risk_score") * 0.10
            + pl.col("fundamental_score") * 0.05
        ).alias("final_score_without_sector")
    )
    df = df.with_columns(
        (
            pl.col("final_score_without_sector") * 0.70
            + pl.col("risk_score") * 0.15
            + pl.col("technical_score") * 0.15
        ).clip(0, 100).alias("entry_score_without_sector")
    )
    df = df.with_columns(
        (
            pl.col("entry_score_without_sector")
            - pl.col("market_buy_threshold_add").cast(pl.Float64, strict=False).fill_null(0.0) * 0.55
        ).clip(0, 100).alias("entry_score_without_sector")
    )
    df = df.with_columns(
        (
            pl.col("entry_score_without_sector") * 0.38
            + pl.col("stockformer_score") * 0.22
            + pl.col("fund_score") * 0.20
            + pl.col("risk_score") * 0.12
            + pl.col("technical_score") * 0.08
        ).clip(0, 100).alias("buy_confirmation_score_without_sector")
    )
    # Ranking alone is not an entry signal.  A cross-section can rank every
    # stock highly even when the model's absolute forecast is negative.  For
    # the strict OOS replay, require the forecast for the tested horizon to
    # clear a small round-trip cost proxy before allowing a position.
    forecast_column = f"pred_ret_{target_horizon}d"
    forecast_floor = 0.0025 if target_horizon <= 3 else 0.0040
    if forecast_column in df.columns:
        df = df.with_columns(
            (pl.col(forecast_column).fill_null(-1.0) > forecast_floor).alias("_forecast_pass")
        )
    else:
        df = df.with_columns(pl.lit(False).alias("_forecast_pass"))
    market_gate_expr = _market_gate_oos_eligible_expr(df)
    df = df.with_columns(
        (_entry_eligible_expr(df) & market_gate_expr & pl.col("_forecast_pass")).alias("_eligible")
    )
    eligible = df.filter(pl.col("_eligible"))
    if eligible.is_empty():
        return {"error": "样本外通过买入确认和市场闸门过滤的交易为空，请放宽确认条件或重新训练模型"}
    return_scale = float(max(target_horizon, 1))
    daily = (
        eligible.sort(["date", "buy_confirmation_score", "risk_score", "stockformer_score"], descending=[False, True, True, True])
        .with_columns(pl.col("buy_confirmation_score").rank(method="ordinal", descending=True).over("date").alias("rank"))
        .filter(pl.col("rank") <= max_positions)
        .group_by("date")
        .agg(
            ((pl.mean(target_col) / return_scale) * pl.mean("market_position_multiplier")).alias("strategy_ret"),
            pl.len().alias("positions"),
            pl.mean("market_position_multiplier").alias("avg_position_multiplier"),
            pl.mean("buy_confirmation_score").alias("avg_buy_confirmation_score"),
        )
        .sort("date")
    )
    bench = (
        df.filter(pl.col("date").is_in(daily["date"].to_list()))
        .group_by("date")
        .agg((pl.mean(target_col) / return_scale).alias("benchmark_ret"))
        .sort("date")
    )
    nav = (
        daily.join(bench, on="date", how="left")
        .with_columns(
            (1 + pl.col("strategy_ret")).cum_prod().alias("strategy_nav"),
            (1 + pl.col("benchmark_ret")).cum_prod().alias("benchmark_nav"),
        )
        .with_columns((pl.col("strategy_nav") / pl.col("benchmark_nav") - 1).alias("excess_return"))
    )
    rets = nav["strategy_ret"].to_numpy()
    nav_values = nav["strategy_nav"].to_numpy()
    equity_path = np.concatenate([np.array([1.0]), nav_values]) if len(nav_values) else np.array([])
    peak = np.maximum.accumulate(equity_path) if len(equity_path) else np.array([])
    dd = equity_path / peak - 1 if len(equity_path) else np.array([])
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    total_return = float(nav_values[-1] - 1) if len(nav_values) else 0.0
    benchmark_return = float(nav["benchmark_nav"][-1] - 1) if nav.height else 0.0
    stats = {
        "strategy_return": round(total_return * 100, 2),
        "benchmark_return": round(benchmark_return * 100, 2),
        "excess_return": round((total_return - benchmark_return) * 100, 2),
        "max_drawdown": round(float(np.min(dd)) * 100, 2) if len(dd) else 0.0,
        "win_rate": round(float(len(wins) / len(rets)) * 100, 2) if len(rets) else 0.0,
        "profit_loss_ratio": round(float(np.mean(wins) / abs(np.mean(losses))), 3) if len(wins) and len(losses) else None,
        "trade_count": int(daily["positions"].sum()) if daily.height else 0,
        "turnover_rate": round(100 / max(holding_days, 1), 2),
        "trading_days": int(daily.height),
            "eligible_rows": int(eligible.height),
            "forecast_pass_rows": int(df.filter(pl.col("_forecast_pass")).height),
            "forecast_floor": forecast_floor,
        "avg_position_multiplier": round(float(daily["avg_position_multiplier"].mean()), 3) if daily.height and "avg_position_multiplier" in daily.columns else 1.0,
        "avg_buy_confirmation_score": round(float(daily["avg_buy_confirmation_score"].mean()), 2) if daily.height and "avg_buy_confirmation_score" in daily.columns else None,
        "market_closed_days": int(df.filter(pl.col("market_gate").is_in(["closed", "observe"])).select(pl.col("date").n_unique()).item() or 0) if "market_gate" in df.columns else 0,
    }

    legacy_eligible = df.filter(
        _entry_eligible_without_sector_expr(df) & market_gate_expr & pl.col("_forecast_pass")
    )
    ablation: dict[str, Any] | None = None
    if not legacy_eligible.is_empty():
        legacy_daily = (
            legacy_eligible.sort(
                ["date", "buy_confirmation_score_without_sector", "risk_score", "stockformer_score"],
                descending=[False, True, True, True],
            )
            .with_columns(pl.col("buy_confirmation_score_without_sector").rank(method="ordinal", descending=True).over("date").alias("rank"))
            .filter(pl.col("rank") <= max_positions)
            .group_by("date")
            .agg(
                ((pl.mean(target_col) / return_scale) * pl.mean("market_position_multiplier")).alias("strategy_ret"),
                pl.len().alias("positions"),
                pl.mean("market_position_multiplier").alias("avg_position_multiplier"),
            )
            .sort("date")
        )
        legacy_bench = (
            df.filter(pl.col("date").is_in(legacy_daily["date"].to_list()))
            .group_by("date")
            .agg((pl.mean(target_col) / return_scale).alias("benchmark_ret"))
            .sort("date")
        )
        legacy_nav = (
            legacy_daily.join(legacy_bench, on="date", how="left")
            .with_columns(
                (1 + pl.col("strategy_ret")).cum_prod().alias("strategy_nav"),
                (1 + pl.col("benchmark_ret")).cum_prod().alias("benchmark_nav"),
            )
            .with_columns((pl.col("strategy_nav") / pl.col("benchmark_nav") - 1).alias("excess_return"))
        )
        legacy_rets = legacy_nav["strategy_ret"].to_numpy()
        legacy_nav_values = legacy_nav["strategy_nav"].to_numpy()
        legacy_equity_path = np.concatenate([np.array([1.0]), legacy_nav_values]) if len(legacy_nav_values) else np.array([])
        legacy_peak = np.maximum.accumulate(legacy_equity_path) if len(legacy_equity_path) else np.array([])
        legacy_dd = legacy_equity_path / legacy_peak - 1 if len(legacy_equity_path) else np.array([])
        legacy_wins = legacy_rets[legacy_rets > 0]
        legacy_losses = legacy_rets[legacy_rets < 0]
        legacy_total_return = float(legacy_nav_values[-1] - 1) if len(legacy_nav_values) else 0.0
        legacy_benchmark_return = float(legacy_nav["benchmark_nav"][-1] - 1) if legacy_nav.height else 0.0
        legacy_stats = {
            "strategy_return": round(legacy_total_return * 100, 2),
            "benchmark_return": round(legacy_benchmark_return * 100, 2),
            "excess_return": round((legacy_total_return - legacy_benchmark_return) * 100, 2),
            "max_drawdown": round(float(np.min(legacy_dd)) * 100, 2) if len(legacy_dd) else 0.0,
            "win_rate": round(float(len(legacy_wins) / len(legacy_rets)) * 100, 2) if len(legacy_rets) else 0.0,
            "profit_loss_ratio": round(float(np.mean(legacy_wins) / abs(np.mean(legacy_losses))), 3) if len(legacy_wins) and len(legacy_losses) else None,
            "trade_count": int(legacy_daily["positions"].sum()) if legacy_daily.height else 0,
            "turnover_rate": round(100 / max(holding_days, 1), 2),
            "trading_days": int(legacy_daily.height),
            "eligible_rows": int(legacy_eligible.height),
        }
        ablation = {
            "with_sector": stats,
            "without_sector": legacy_stats,
            "delta": {
                "strategy_return": round(stats["strategy_return"] - legacy_stats["strategy_return"], 2),
                "excess_return": round(stats["excess_return"] - legacy_stats["excess_return"], 2),
                "max_drawdown": round(stats["max_drawdown"] - legacy_stats["max_drawdown"], 2),
                "win_rate": round(stats["win_rate"] - legacy_stats["win_rate"], 2),
                "trade_count": int(stats["trade_count"] - legacy_stats["trade_count"]),
            },
            "note": "with_sector 使用短线入场分、昨日前板块布局、今日不过热和市场闸门；without_sector 使用同一市场闸门，但不使用板块提前布局。",
        }
    checks = [
        {
            'label': '完整成交回测', 'passed': None,
            'value': '仅收益标签估算', 'criteria': '需独立组合成交验证',
        },
        {
            "label": "样本外收益为正",
            "passed": stats["strategy_return"] > 0,
            "value": f'{stats["strategy_return"]}%',
            "criteria": "> 0%",
        },
        {
            "label": "样本外超额收益为正",
            "passed": stats["excess_return"] > 0,
            "value": f'{stats["excess_return"]}%',
            "criteria": "> 0%",
        },
        {
            "label": "候选信号不少于20条",
            "passed": stats["trade_count"] >= 20,
            "value": str(stats["trade_count"]),
            "criteria": ">= 20",
        },
        {
            "label": "最大回撤可接受",
            "passed": abs(stats["max_drawdown"]) <= 15,
            "value": f'{stats["max_drawdown"]}%',
            "criteria": ">= -15%",
        },
        {
            "label": "盈亏比不低于1",
            "passed": stats["profit_loss_ratio"] is not None and stats["profit_loss_ratio"] >= 1,
            "value": str(stats["profit_loss_ratio"] if stats["profit_loss_ratio"] is not None else "--"),
            "criteria": ">= 1",
        },
    ]
    return _json_safe({
        "model": meta,
        'evaluation_kind': 'forward_label_proxy',
        'evaluation_warning': EVALUATION_LIMITATION,
        "sample_in": {"start": meta["train_start"], "end": meta["train_end"]},
        "sample_out": {"start": meta["oos_start"], "end": meta["oos_end"]},
        "target_horizon": target_horizon,
        "signal_weights": {f"{horizon}d": weight for horizon, weight in signal_weights.items()},
        "data_coverage": _external_feature_coverage(data_dir),
        "stats": stats,
        "sector_ablation": ablation,
        "nav": [{"date": str(r["date"]), "strategy_nav": r["strategy_nav"], "benchmark_nav": r["benchmark_nav"], "excess_return": r["excess_return"]} for r in nav.to_dicts()],
        "diagnosis": {
            "passed": all(item["passed"] for item in checks),
            "checks": checks,
            "rules": [
                "样本外收益为正",
                "样本外超额收益为正",
                "候选信号不少于20条, 不代表真实平仓交易数",
                "盈亏比不低于1",
                "最大回撤默认不超过15%",
                "买入确认要求入场分、StockFormer、资金、技术、风险同时达标，优先低位压缩、吸筹效率和资金由弱转强",
                "过滤资金连续流入后手、短期涨幅过大、价格脱离低位和极端振幅，避免追已经拉升的票",
                "板块要求昨日前已有布局迹象，今日涨幅不超过2.5%，避免追已经加速的板块",
                "市场环境闸门过滤弱势/退潮/大跌扩散日期，市场分低于60时停止开仓，震荡日按仓位系数折算收益",
                "严格样本外交易必须满足对应持有周期预测收益超过成本代理阈值，避免只按横截面排名买入负预测股票",
                f"持有 {holding_days} 日按 {target_horizon} 日收益标签验证",
                "使用综合分排序并过滤高风险、跌停、炸板、极端振幅",
                EVALUATION_LIMITATION,
            ],
        },
    })


def _oos_empty_stats(*, forecast_floor: float, eligible_rows: int = 0) -> dict:
    return {
        "strategy_return": 0.0,
        "benchmark_return": 0.0,
        "excess_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_loss_ratio": None,
        "trade_count": 0,
        "turnover_rate": 0.0,
        "trading_days": 0,
        "eligible_rows": int(eligible_rows),
        "forecast_floor": forecast_floor,
    }


def _evaluate_oos_candidates(
    df: pl.DataFrame,
    eligible_expr: pl.Expr,
    *,
    target_col: str,
    target_horizon: int,
    max_positions: int,
    holding_days: int,
    forecast_floor: float,
) -> tuple[dict, list[dict]]:
    eligible = df.filter(eligible_expr)
    if eligible.is_empty():
        return _oos_empty_stats(forecast_floor=forecast_floor), []
    return_scale = float(max(target_horizon, 1))
    daily = (
        eligible.sort(
            ["date", "buy_confirmation_score", "risk_score", "stockformer_score"],
            descending=[False, True, True, True],
        )
        .with_columns(
            pl.col("buy_confirmation_score")
            .rank(method="ordinal", descending=True)
            .over("date")
            .alias("rank")
        )
        .filter(pl.col("rank") <= max_positions)
        .group_by("date")
        .agg(
            ((pl.mean(target_col) / return_scale) * pl.mean("market_position_multiplier")).alias("strategy_ret"),
            pl.len().alias("positions"),
            pl.mean("market_position_multiplier").alias("avg_position_multiplier"),
            pl.mean("buy_confirmation_score").alias("avg_buy_confirmation_score"),
        )
        .sort("date")
    )
    bench = (
        df.filter(pl.col("date").is_in(daily["date"].to_list()))
        .group_by("date")
        .agg((pl.mean(target_col) / return_scale).alias("benchmark_ret"))
        .sort("date")
    )
    nav = (
        daily.join(bench, on="date", how="left")
        .with_columns(
            (1 + pl.col("strategy_ret")).cum_prod().alias("strategy_nav"),
            (1 + pl.col("benchmark_ret")).cum_prod().alias("benchmark_nav"),
        )
        .with_columns((pl.col("strategy_nav") / pl.col("benchmark_nav") - 1).alias("excess_return"))
    )
    rets = nav["strategy_ret"].to_numpy()
    nav_values = nav["strategy_nav"].to_numpy()
    equity_path = np.concatenate([np.array([1.0]), nav_values])
    peak = np.maximum.accumulate(equity_path)
    dd = equity_path / peak - 1
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    total_return = float(nav_values[-1] - 1) if len(nav_values) else 0.0
    benchmark_return = float(nav["benchmark_nav"][-1] - 1) if nav.height else 0.0
    stats = {
        "strategy_return": round(total_return * 100, 2),
        "benchmark_return": round(benchmark_return * 100, 2),
        "excess_return": round((total_return - benchmark_return) * 100, 2),
        "max_drawdown": round(float(np.min(dd)) * 100, 2) if len(dd) else 0.0,
        "win_rate": round(float(len(wins) / len(rets)) * 100, 2) if len(rets) else 0.0,
        "profit_loss_ratio": round(float(np.mean(wins) / abs(np.mean(losses))), 3) if len(wins) and len(losses) else None,
        "trade_count": int(daily["positions"].sum()) if daily.height else 0,
        "turnover_rate": round(100 / max(holding_days, 1), 2),
        "trading_days": int(daily.height),
        "eligible_rows": int(eligible.height),
        "forecast_floor": forecast_floor,
        "avg_position_multiplier": round(float(daily["avg_position_multiplier"].mean()), 3) if daily.height else 1.0,
        "avg_buy_confirmation_score": round(float(daily["avg_buy_confirmation_score"].mean()), 2) if daily.height else None,
        "market_closed_days": int(df.filter(pl.col("market_gate").is_in(["closed", "observe"])).select(pl.col("date").n_unique()).item() or 0),
    }
    nav_rows = [
        {
            "date": str(row["date"]),
            "strategy_nav": row["strategy_nav"],
            "benchmark_nav": row["benchmark_nav"],
            "excess_return": row["excess_return"],
        }
        for row in nav.to_dicts()
    ]
    return stats, nav_rows


def strict_oos_backtest(
    data_dir: Path,
    *,
    model_id: str | None = None,
    max_positions: int = 3,
    holding_days: int = 5,
    initial_capital: float = 10000.0,
    candidate_symbols: list[str] | None = None,
    on_progress=None,
) -> dict:
    """Strict OOS research report with a filter funnel and one-variable ablations."""
    from app.services.stockformer_training import EVALUATION_LIMITATION

    def progress(stage: str, pct: int, msg: str) -> None:
        if on_progress is not None:
            on_progress(stage, pct, msg)

    progress("load_factors", 5, "加载样本外因子")
    meta = _model_meta(data_dir, model_id)
    if meta is None:
        return {"error": "暂无 StockFormer 模型，请先训练模型"}
    from app.services.stockformer_portfolio import load_frozen_signal_policy

    if load_frozen_signal_policy(data_dir, meta):
        from app.services.stockformer_rolling import run_rolling_portfolio_research

        return run_rolling_portfolio_research(
            data_dir,
            model_id=str(meta["model_id"]),
            max_positions=max_positions,
            initial_capital=initial_capital,
            on_progress=on_progress,
        )
    factor_path = _resolve_artifact_path(data_dir, meta.get("factor_file"))
    if factor_path is None or not factor_path.exists():
        return {"error": f"模型缺少样本外因子文件: {meta.get('factor_file')}"}
    factor = pl.read_parquet(factor_path)
    target_horizon = _nearest_horizon(holding_days)
    target_col = f"target_{target_horizon}d"
    available_preds = [h for h in HORIZONS if f"pred_ret_{h}d" in factor.columns]
    if factor.is_empty() or not available_preds or target_col not in factor.columns:
        return {"error": f"模型缺少样本外预测因子或 {target_col}"}
    df, signal_weights = _add_adaptive_stockformer_score(factor, meta)
    df = df.filter(pl.col("stockformer_score").is_not_null() & pl.col(target_col).is_not_null())
    if candidate_symbols:
        df = df.filter(_symbol_in_filter(candidate_symbols))
    if df.is_empty():
        return {"error": "样本外有效交易为空"}

    progress("load_features", 25, "加载历史特征")
    features = _load_factor_feature_frame(data_dir, df)
    if not features.is_empty():
        feature_cols = ["symbol", "date", *[c for c in ["amount", *FEATURE_COLUMNS] if c in features.columns]]
        df = df.join(features.select(feature_cols), on=["symbol", "date"], how="left")
    df = _filter_special_treatment(_attach_instrument_names(data_dir, df))
    if df.is_empty():
        return {"error": "样本外正常交易标的为空，请检查个股维表或放宽过滤条件"}

    progress("compute_gates", 55, "计算入场规则和市场闸门")
    df = _add_multifactor_scores(df)
    df = _attach_market_gate_for_oos(data_dir, df)
    df = df.with_columns(
        (pl.col("entry_score") - pl.col("market_buy_threshold_add").cast(pl.Float64, strict=False).fill_null(0.0) * 0.55).clip(0, 100).alias("entry_score"),
    )
    df = df.with_columns(
        (pl.col("entry_score") * 0.38 + pl.col("stockformer_score") * 0.22 + pl.col("fund_score") * 0.20 + pl.col("risk_score") * 0.12 + pl.col("technical_score") * 0.08).clip(0, 100).alias("buy_confirmation_score"),
    )
    forecast_column = f"pred_ret_{target_horizon}d"
    forecast_floor = 0.0025 if target_horizon <= 3 else 0.0040
    df = df.with_columns((pl.col(forecast_column).fill_null(-1.0) > forecast_floor).alias("_forecast_pass"))
    def entry_expr_for(min_score: float) -> pl.Expr:
        # Keep compatibility with research tests/plugins that monkeypatch the
        # historical one-argument predicate.
        try:
            return _entry_eligible_expr(df, min_score)
        except TypeError:
            return _entry_eligible_expr(df)

    # The production entry rule is intentionally permissive enough to produce
    # an observable OOS candidate set.  Ranking and the forecast floor still
    # provide the quality/absolute-return controls.
    entry_expr = entry_expr_for(55)
    market_expr = _market_gate_oos_eligible_expr(df)
    forecast_expr = pl.col("_forecast_pass")
    funnel = {
        "total_rows": int(df.height),
        "prediction_pass": {"rows": int(df.filter(forecast_expr).height), "dates": int(df.filter(forecast_expr).select(pl.col("date").n_unique()).item() or 0)},
        "entry_rules_pass": {"rows": int(df.filter(entry_expr).height), "dates": int(df.filter(entry_expr).select(pl.col("date").n_unique()).item() or 0)},
        "market_gate_pass": {"rows": int(df.filter(market_expr).height), "dates": int(df.filter(market_expr).select(pl.col("date").n_unique()).item() or 0)},
        "cumulative": {
            "prediction_and_entry": int(df.filter(forecast_expr & entry_expr).height),
            "prediction_and_market": int(df.filter(forecast_expr & market_expr).height),
            "prediction_entry_market": int(df.filter(forecast_expr & entry_expr & market_expr).height),
        },
    }
    progress("evaluate", 75, "生成收益与回撤")
    variants = {
        "base": (forecast_floor, 55.0, market_expr, "strict", "当前阈值"),
        "relax_forecast": (0.0, 55.0, market_expr, "strict", "仅放宽预测门槛"),
        "relax_market_gate": (forecast_floor, 55.0, _market_gate_oos_relaxed_expr(df), "relaxed", "仅放宽市场闸门"),
        "relax_entry_score": (forecast_floor, 45.0, market_expr, "strict", "仅放宽入场分"),
    }
    ablation: dict[str, dict] = {}
    for key, (floor, min_score, gate, mode, label) in variants.items():
        forecast = pl.col(forecast_column).fill_null(-1.0) > floor
        entry = entry_expr_for(min_score)
        stats, nav = _evaluate_oos_candidates(
            df,
            forecast & entry & gate,
            target_col=target_col,
            target_horizon=target_horizon,
            max_positions=max_positions,
            holding_days=holding_days,
            forecast_floor=floor,
        )
        ablation[key] = {
            "label": label,
            "forecast_floor": floor,
            "entry_score_min": min_score,
            "market_gate_mode": mode,
            "stats": stats,
            "nav": nav,
        }
    stats = ablation["base"]["stats"]
    nav = ablation["base"]["nav"]
    checks = [
        {"label": "完整成交回测", "passed": False, "value": "仅收益标签估算", "criteria": "需独立组合成交验证"},
        {"label": "样本外收益为正", "passed": stats["strategy_return"] > 0, "value": f'{stats["strategy_return"]}%', "criteria": "> 0%"},
        {"label": "样本外超额收益为正", "passed": stats["excess_return"] > 0, "value": f'{stats["excess_return"]}%', "criteria": "> 0%"},
        {"label": "候选信号不少于20条", "passed": stats["trade_count"] >= 20, "value": str(stats["trade_count"]), "criteria": ">= 20"},
        {"label": "最大回撤可接受", "passed": abs(stats["max_drawdown"]) <= 15, "value": f'{stats["max_drawdown"]}%', "criteria": ">= -15%"},
    ]
    progress("done", 100, "样本外评估完成")
    return _json_safe({
        "model": meta,
        "evaluation_kind": "forward_label_proxy",
        "evaluation_warning": EVALUATION_LIMITATION,
        "sample_in": {"start": meta["train_start"], "end": meta["train_end"]},
        "sample_out": {"start": meta["oos_start"], "end": meta["oos_end"]},
        "target_horizon": target_horizon,
        "signal_weights": {f"{horizon}d": weight for horizon, weight in signal_weights.items()},
        "data_coverage": _external_feature_coverage(data_dir),
        "stats": stats,
        "nav": nav,
        "layer_diagnostics": funnel,
        "threshold_ablation": ablation,
        "diagnosis": {
            "passed": all(item["passed"] for item in checks),
            "checks": checks,
            "rules": ["预测门槛、入场规则、市场闸门按层统计", "消融每次只改变一个阈值", EVALUATION_LIMITATION],
        },
    })
