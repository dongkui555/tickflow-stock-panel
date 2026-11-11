"""Sector rotation prediction and early fund-layout detection."""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from app.parquet import read_parquet_safe

logger = logging.getLogger(__name__)

_CACHE_TTL = 45.0
_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}

_BAR_NUMERIC_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "raw_close",
    "raw_high",
    "raw_low",
    "consecutive_limit_ups",
    "consecutive_limit_downs",
]

_FLOW_NUMERIC_COLS = [
    "buy_sm_amount",
    "sell_sm_amount",
    "buy_md_amount",
    "sell_md_amount",
    "buy_lg_amount",
    "sell_lg_amount",
    "buy_elg_amount",
    "sell_elg_amount",
    "net_mf_amount",
]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float(value: Any, default: float = 0.0) -> float:
    out = _float_or_none(value)
    return default if out is None else out


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _round(value: Any, digits: int = 4) -> float | None:
    out = _float_or_none(value)
    return None if out is None else round(out, digits)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _symbol_key(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.startswith(("SH", "SZ", "BJ")) and len(text) >= 8:
        text = text[2:]
    if "." in text:
        text = text.split(".", 1)[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def normalize_sector_name(value: Any, level: int = 2) -> str:
    """Normalize THS industry text to a stable sector bucket."""
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace("－", "-").replace("—", "-").replace("/", "-")
    parts = [part.strip() for part in text.split("-") if part.strip()]
    if not parts:
        return text
    if level and level > 0:
        parts = parts[: max(1, int(level))]
    if len(parts) >= 2 and parts[0] == parts[1]:
        return parts[0]
    return "-".join(parts)


def _parse_partition_date(entry: Path) -> date | None:
    if not entry.is_dir() or not entry.name.startswith("date="):
        return None
    try:
        return date.fromisoformat(entry.name.removeprefix("date="))
    except ValueError:
        return None


def _partition_dates(root: Path, *, end: date | None = None) -> list[date]:
    if not root.exists():
        return []
    dates: list[date] = []
    for entry in root.iterdir():
        value = _parse_partition_date(entry)
        if value is None:
            continue
        if end is not None and value > end:
            continue
        if (entry / "part.parquet").exists():
            dates.append(value)
    dates.sort()
    return dates


def _read_partitions(root: Path, dates: list[date]) -> tuple[pl.DataFrame, int]:
    frames: list[pl.DataFrame] = []
    skipped = 0
    for day in dates:
        part = root / f"date={day.isoformat()}" / "part.parquet"
        df = read_parquet_safe(part)
        if df is None:
            skipped += 1
            continue
        if df.is_empty():
            continue
        if "date" not in df.columns:
            df = df.with_columns(pl.lit(day).cast(pl.Date).alias("date"))
        frames.append(df)
    if not frames:
        return pl.DataFrame(), skipped
    try:
        return pl.concat(frames, how="diagonal_relaxed"), skipped
    except Exception as exc:  # noqa: BLE001
        logger.warning("sector rotation partition concat failed: %s", exc)
        return pl.DataFrame(), skipped + len(frames)


def _parse_date_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8, strict=False).str.replace_all("/", "-")
    return (
        pl.when(text.str.contains(r"^\d{8}$"))
        .then(text.str.strptime(pl.Date, "%Y%m%d", strict=False))
        .otherwise(text.str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False))
    )


def _normalize_date_column(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    if "date" not in df.columns and "trade_date" in df.columns:
        df = df.with_columns(pl.col("trade_date").alias("date"))
    if "date" not in df.columns:
        return df
    dtype = df.schema.get("date")
    if dtype == pl.Date:
        return df
    if dtype == pl.Datetime:
        return df.with_columns(pl.col("date").dt.date().alias("date"))
    return df.with_columns(_parse_date_expr("date").alias("date"))


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
        pl.col("symbol").map_elements(_symbol_key, return_dtype=pl.Utf8).alias("symbol")
    ).filter(pl.col("symbol") != "")


def _cast_numeric(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    exprs = [
        pl.col(col).cast(pl.Float64, strict=False).alias(col)
        for col in columns
        if col in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def _sum_columns(columns: list[str], names: list[str]) -> pl.Expr:
    expr: pl.Expr | None = None
    for name in names:
        if name not in columns:
            continue
        item = pl.col(name).cast(pl.Float64, strict=False).fill_null(0.0)
        expr = item if expr is None else expr + item
    return expr if expr is not None else pl.lit(0.0)


def _pct_rank_expr(name: str) -> pl.Expr:
    return (pl.col(name).rank(method="average").over("date") / pl.len().over("date")).fill_null(0.5)


def _read_parquet_root(root: Path) -> pl.DataFrame:
    files = [path for path in sorted(root.rglob("*.parquet")) if path.is_file()] if root.exists() else []
    frames: list[pl.DataFrame] = []
    for file in files:
        df = read_parquet_safe(file)
        if df is not None and not df.is_empty():
            frames.append(df)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _load_industry_map(data_dir: Path, *, level: int = 2) -> pl.DataFrame:
    industry = _read_parquet_root(data_dir / "ext_data" / "ext_hy_ths")
    if industry.is_empty():
        return pl.DataFrame(schema={"symbol": pl.Utf8, "sector": pl.Utf8, "industry_raw": pl.Utf8})
    industry = _normalize_symbol_column(industry)
    if "symbol" not in industry.columns:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "sector": pl.Utf8, "industry_raw": pl.Utf8})
    field = None
    for candidate in ("industry", "sector", "所属同花顺行业", "所属行业", "申万行业", "行业"):
        if candidate in industry.columns:
            field = candidate
            break
    if not field:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "sector": pl.Utf8, "industry_raw": pl.Utf8})
    return (
        industry.select(["symbol", field])
        .rename({field: "industry_raw"})
        .with_columns(
            pl.col("industry_raw").cast(pl.Utf8, strict=False).str.strip_chars().alias("industry_raw"),
            pl.col("industry_raw")
            .map_elements(lambda value: normalize_sector_name(value, level), return_dtype=pl.Utf8)
            .alias("sector"),
        )
        .filter((pl.col("symbol") != "") & (pl.col("sector") != ""))
        .unique(subset=["symbol"], keep="last")
    )


def _load_name_map(data_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for root in (data_dir / "instruments", data_dir / "instruments_etf", data_dir / "instruments_index"):
        df = _read_parquet_root(root)
        if df.is_empty() or "name" not in df.columns:
            continue
        df = _normalize_symbol_column(df)
        if "symbol" not in df.columns:
            continue
        for item in df.select(["symbol", "name"]).to_dicts():
            symbol = _symbol_key(item.get("symbol"))
            name = str(item.get("name") or "").strip()
            if symbol and name:
                out.setdefault(symbol, name)
    return out


def _repair_price_panel(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "close" not in df.columns or "raw_close" not in df.columns:
        return df
    try:
        df = _cast_numeric(df, ["open", "high", "low", "close", "raw_close", "raw_high", "raw_low"])
        df = df.with_columns(
            pl.when((pl.col("raw_close") > 0) & (pl.col("close") > 0))
            .then(pl.col("close") / pl.col("raw_close"))
            .otherwise(None)
            .alias("_price_ratio")
        )
        bad = (
            (pl.col("raw_close") > 0)
            & (
                pl.col("close").is_null()
                | (pl.col("close") <= 0)
                | (pl.col("_price_ratio") < 0.02)
                | (pl.col("_price_ratio") > 50.0)
            )
        ).fill_null(False)
        exprs = [
            pl.when(bad).then(pl.col("raw_close")).otherwise(pl.col("close")).alias("close"),
        ]
        if "open" in df.columns:
            exprs.append(pl.when(bad).then(pl.col("raw_close")).otherwise(pl.col("open")).alias("open"))
        if "high" in df.columns:
            replacement = pl.col("raw_high") if "raw_high" in df.columns else pl.col("raw_close")
            exprs.append(pl.when(bad).then(replacement).otherwise(pl.col("high")).alias("high"))
        if "low" in df.columns:
            replacement = pl.col("raw_low") if "raw_low" in df.columns else pl.col("raw_close")
            exprs.append(pl.when(bad).then(replacement).otherwise(pl.col("low")).alias("low"))
        return df.with_columns(exprs).drop([c for c in ("_price_ratio",) if c in df.columns])
    except Exception:
        return df


def _load_daily_panel(
    data_dir: Path,
    selected_dates: list[date],
    industry_map: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    root = data_dir / "kline_daily_enriched"
    source = "kline_daily_enriched"
    if not root.exists():
        root = data_dir / "kline_daily"
        source = "kline_daily"
    df, skipped = _read_partitions(root, selected_dates)
    meta = {
        "source": source,
        "rows": int(df.height) if not df.is_empty() else 0,
        "skipped_partitions": skipped,
    }
    if df.is_empty() or industry_map.is_empty():
        return pl.DataFrame(), meta
    df = _normalize_date_column(_normalize_symbol_column(df))
    df = _cast_numeric(df, _BAR_NUMERIC_COLS)
    df = _repair_price_panel(df)
    required = {"symbol", "date", "close"}
    if not required.issubset(set(df.columns)):
        return pl.DataFrame(), meta
    if "open" not in df.columns:
        df = df.with_columns(pl.col("close").alias("open"))
    if "high" not in df.columns:
        df = df.with_columns(pl.col("close").alias("high"))
    if "low" not in df.columns:
        df = df.with_columns(pl.col("close").alias("low"))
    if "amount" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("amount"))
    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("volume"))
    if "turnover_rate" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("turnover_rate"))
    if "consecutive_limit_ups" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("consecutive_limit_ups"))
    df = (
        df.join(industry_map, on="symbol", how="inner")
        .unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
    )
    prev_close = pl.col("close").shift(1).over("symbol")
    calendar = pl.DataFrame({"date": selected_dates}).with_columns(pl.col("date").shift(1).alias("_expected_previous"))
    df = df.join(calendar, on="date", how="left").with_columns(
        ((pl.col("date").shift(1).over("symbol") == pl.col("_expected_previous"))
         & pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in ("open", "high", "low", "close")])
         & (prev_close > 0) & (pl.col("high") >= pl.max_horizontal("open", "close", "low"))
         & (pl.col("low") <= pl.min_horizontal("open", "close"))).fill_null(False).alias("price_valid")
    )
    df = df.with_columns(
        pl.when(pl.col("price_valid")).then(pl.col("close") / prev_close - 1).otherwise(None).alias("ret_1d"),
        (pl.col("close") / pl.col("close").shift(3).over("symbol") - 1).alias("ret_3d"),
        (pl.col("close") / pl.col("close").shift(5).over("symbol") - 1).alias("ret_5d"),
        (pl.col("close") / pl.col("close").shift(10).over("symbol") - 1).alias("ret_10d"),
        pl.col("amount").shift(1).rolling_mean(5).over("symbol").alias("_amount_ma5_prev"),
    )
    df = df.with_columns(
        pl.when(pl.col("_amount_ma5_prev").abs() > 1e-9)
        .then(pl.col("amount") / pl.col("_amount_ma5_prev"))
        .otherwise(1.0)
        .fill_null(1.0)
        .alias("amount_ratio_5d"),
        # The enriched pipeline already applies raw-price, board and ST rules.
        (pl.col("consecutive_limit_ups").fill_null(0.0) > 0).cast(pl.Float64).alias("limit_up_flag"),
        pl.col("consecutive_limit_ups").is_not_null().cast(pl.Float64).alias("limit_known"),
        (pl.col("low") >= pl.col("low").shift(1).rolling_min(5).over("symbol"))
        .cast(pl.Float64).alias("not_new_low"),
        (pl.col("ret_1d").fill_null(0.0) > 0).cast(pl.Float64).alias("up_flag"),
    )
    for column in ("ret_1d", "ret_3d", "ret_5d", "ret_10d"):
        df = df.with_columns(
            pl.when(pl.col(column).is_finite()).then(pl.col(column)).otherwise(None).alias(column)
        )
    return df, meta


def _load_flow_panel(
    data_dir: Path,
    selected_dates: list[date],
    industry_map: pl.DataFrame,
    eligible_bars: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    root = data_dir / "moneyflow"
    flow_dates = _partition_dates(root, end=selected_dates[-1] if selected_dates else None)
    wanted = [day for day in flow_dates if day in set(selected_dates)]
    df, skipped = _read_partitions(root, wanted)
    meta = {
        "source": "moneyflow",
        "date_start": wanted[0].isoformat() if wanted else None,
        "date_end": wanted[-1].isoformat() if wanted else None,
        "trading_days": len(wanted),
        "rows": int(df.height) if not df.is_empty() else 0,
        "skipped_partitions": skipped,
    }
    if df.is_empty() or industry_map.is_empty():
        return pl.DataFrame(), meta
    df = _normalize_date_column(_normalize_symbol_column(df))
    df = _cast_numeric(df, _FLOW_NUMERIC_COLS)
    if "symbol" not in df.columns or "date" not in df.columns:
        return pl.DataFrame(), meta
    columns = df.columns
    buy_total = _sum_columns(columns, ["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"])
    sell_total = _sum_columns(columns, ["sell_sm_amount", "sell_md_amount", "sell_lg_amount", "sell_elg_amount"])
    large_net = _sum_columns(columns, ["buy_lg_amount"]) - _sum_columns(columns, ["sell_lg_amount"])
    extra_large_net = _sum_columns(columns, ["buy_elg_amount"]) - _sum_columns(columns, ["sell_elg_amount"])
    required = ["buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"]
    if not set(required).issubset(columns):
        meta["warning"] = "资金分单字段缺失"
        return pl.DataFrame(), meta
    df = df.filter(pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) >= 0) for c in required]))
    df = df.unique(subset=["symbol", "date"], keep="last")
    if eligible_bars is not None:
        if eligible_bars.is_empty():
            return pl.DataFrame(), meta
        df = df.join(eligible_bars.select("symbol", "date").unique(), on=["symbol", "date"], how="semi")
    main_net = large_net + extra_large_net
    active_net = pl.col("net_mf_amount") if "net_mf_amount" in columns else pl.lit(None, dtype=pl.Float64)
    flow = (
        df.with_columns(
            main_net.alias("net_1d"),
            active_net.alias("active_net_1d"),
            large_net.alias("large_net_1d"),
            extra_large_net.alias("extra_large_net_1d"),
            pl.max_horizontal(buy_total, sell_total, main_net.abs()).alias("flow_turnover_proxy"),
        )
        .join(industry_map.select(["symbol", "sector"]), on="symbol", how="inner")
        .group_by(["sector", "date"])
        .agg(
            pl.sum("net_1d").alias("net_1d"),
            pl.sum("active_net_1d").alias("active_net_1d"),
            pl.col("symbol").n_unique().alias("flow_member_count"),
            pl.sum("large_net_1d").alias("large_net_1d"),
            pl.sum("extra_large_net_1d").alias("extra_large_net_1d"),
            pl.sum("flow_turnover_proxy").alias("flow_turnover_proxy"),
        )
    )
    return flow, meta


def _sector_daily_frame(data_dir: Path, selected_dates: list[date], level: int) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame]:
    industry_map = _load_industry_map(data_dir, level=level)
    meta: dict[str, Any] = {
        "industry_members": int(industry_map.height) if not industry_map.is_empty() else 0,
        "industry_level": level,
    }
    bars, bar_meta = _load_daily_panel(data_dir, selected_dates, industry_map)
    meta["daily"] = bar_meta
    flow, flow_meta = _load_flow_panel(data_dir, selected_dates, industry_map, bars)
    meta["moneyflow"] = flow_meta
    if bars.is_empty():
        return pl.DataFrame(), meta, bars

    sector = (
        bars.group_by(["sector", "date"])
        .agg(
            pl.col("symbol").n_unique().alias("member_count"),
            pl.mean("ret_1d").alias("ret_1d"),
            pl.mean("ret_3d").alias("ret_3d"),
            pl.mean("ret_5d").alias("ret_5d"),
            pl.mean("ret_10d").alias("ret_10d"),
            pl.mean("up_flag").alias("up_ratio"),
            pl.sum("limit_up_flag").alias("limit_up_count"),
            pl.sum("amount").alias("amount"),
            pl.mean("amount_ratio_5d").alias("amount_ratio_5d"),
            pl.mean("turnover_rate").alias("turnover_rate"),
            pl.mean("limit_known").alias("limit_coverage"),
            pl.mean("price_valid").alias("price_coverage"),
            pl.mean("not_new_low").alias("resilient_ratio"),
        )
        .filter(pl.col("member_count") >= 5)
    )
    if not flow.is_empty():
        sector = sector.join(flow, on=["sector", "date"], how="left")
    if "flow_member_count" not in sector.columns:
        sector = sector.with_columns(pl.lit(0).alias("flow_member_count"))
    sector = sector.with_columns(
        (pl.col("flow_member_count").fill_null(0) / pl.col("member_count")).clip(0, 1).alias("flow_coverage")
    )
    for column in ("net_1d", "large_net_1d", "extra_large_net_1d", "flow_turnover_proxy"):
        if column not in sector.columns:
            sector = sector.with_columns(pl.lit(0.0).alias(column))
    sector = sector.with_columns([
        pl.col(column).fill_null(0.0).alias(column)
        for column in ("net_1d", "large_net_1d", "extra_large_net_1d", "flow_turnover_proxy")
    ])
    sector = sector.sort(["sector", "date"])
    sector = sector.with_columns(
        pl.when((pl.col("flow_coverage") >= .98) & (pl.col("amount") > 0))
        .then(pl.col("net_1d") / pl.col("amount")).otherwise(None).alias("flow_intensity"),
        (pl.col("ret_1d") - pl.col("ret_1d").mean().over("date")).alias("relative_return"),
    )
    sector = sector.with_columns(
        ((pl.col("ret_1d") >= .012) & (pl.col("relative_return") >= .006) & (pl.col("up_ratio") >= .55))
        .fill_null(False).cast(pl.Int32).rolling_max(3, min_samples=1).over("sector").alias("recently_started")
    )
    sector = sector.with_columns(
        pl.col("flow_intensity").shift(1).over("sector").alias("prior_flow_intensity"),
        pl.col("flow_intensity").shift(1).rolling_mean(20, min_samples=15).over("sector").alias("_flow_mean"),
        pl.col("flow_intensity").shift(1).rolling_std(20, min_samples=15).over("sector").alias("_flow_std"),
        pl.col("flow_intensity").is_not_null().cast(pl.Int32).rolling_sum(20, min_samples=1).over("sector").alias("flow_history_days"),
    ).with_columns(
        (pl.col("flow_intensity") - pl.col("prior_flow_intensity")).alias("flow_acceleration"),
        pl.when(pl.col("_flow_std") > 1e-8)
        .then((pl.col("flow_intensity") - pl.col("_flow_mean")) / pl.col("_flow_std"))
        .otherwise(None).alias("flow_zscore"),
    )
    sector = sector.with_columns(
        pl.col("net_1d").rolling_sum(3).over("sector").alias("net_3d"),
        pl.col("net_1d").rolling_sum(5).over("sector").alias("net_5d"),
        pl.col("net_1d").rolling_sum(10).over("sector").alias("net_10d"),
        pl.col("net_1d").rolling_sum(20).over("sector").alias("net_20d"),
        (pl.col("large_net_1d") + pl.col("extra_large_net_1d")).rolling_sum(5).over("sector").alias("large_net_5d"),
        (pl.col("net_1d") > 0).cast(pl.Float64).rolling_sum(5).over("sector").alias("positive_days_5d"),
        pl.col("flow_turnover_proxy").rolling_sum(5).over("sector").alias("flow_turnover_5d"),
    )
    sector = sector.with_columns(
        (
            pl.when(pl.col("flow_turnover_5d").abs() > 1e-9)
            .then(pl.col("net_5d") / pl.col("flow_turnover_5d"))
            .otherwise(0.0)
            .clip(-1, 1)
            .alias("net_ratio_5d")
        ),
        (
            pl.when(pl.col("flow_turnover_5d").abs() > 1e-9)
            .then(pl.col("large_net_5d") / pl.col("flow_turnover_5d"))
            .otherwise(0.0)
            .clip(-1, 1)
            .alias("large_ratio_5d")
        ),
    )
    sector = sector.with_columns(
        pl.col("ret_1d").shift(1).over("sector").alias("pre_ret_1d"),
        pl.col("ret_3d").shift(1).over("sector").alias("pre_ret_3d"),
        pl.col("ret_5d").shift(1).over("sector").alias("pre_ret_5d"),
        pl.col("ret_10d").shift(1).over("sector").alias("pre_ret_10d"),
        pl.col("up_ratio").shift(1).over("sector").alias("pre_up_ratio"),
        pl.col("limit_up_count").shift(1).over("sector").alias("pre_limit_up_count"),
        pl.col("amount_ratio_5d").shift(1).over("sector").alias("pre_amount_ratio_5d"),
        pl.col("turnover_rate").shift(1).over("sector").alias("pre_turnover_rate"),
        pl.col("net_3d").shift(1).over("sector").alias("pre_net_3d"),
        pl.col("net_5d").shift(1).over("sector").alias("pre_net_5d"),
        pl.col("large_net_5d").shift(1).over("sector").alias("pre_large_net_5d"),
        pl.col("positive_days_5d").shift(1).over("sector").alias("pre_positive_days_5d"),
    )
    sector = sector.with_columns(
        (
            _pct_rank_expr("net_3d") * 0.22
            + _pct_rank_expr("net_5d") * 0.25
            + _pct_rank_expr("large_net_5d") * 0.18
            + _pct_rank_expr("positive_days_5d") * 0.15
            + _pct_rank_expr("amount_ratio_5d") * 0.12
            + _pct_rank_expr("up_ratio") * 0.08
        ).alias("_fund_layout_rank"),
        (
            _pct_rank_expr("ret_3d") * 0.30
            + _pct_rank_expr("ret_5d") * 0.30
            + _pct_rank_expr("ret_10d") * 0.15
            + _pct_rank_expr("up_ratio") * 0.15
            + _pct_rank_expr("limit_up_count") * 0.10
        ).alias("_momentum_rank"),
        (
            _pct_rank_expr("amount_ratio_5d") * 0.55
            + _pct_rank_expr("amount") * 0.30
            + _pct_rank_expr("turnover_rate") * 0.15
        ).alias("_volume_rank"),
        (
            _pct_rank_expr("pre_net_3d") * 0.28
            + _pct_rank_expr("pre_net_5d") * 0.22
            + _pct_rank_expr("pre_large_net_5d") * 0.16
            + _pct_rank_expr("pre_positive_days_5d") * 0.12
            + _pct_rank_expr("pre_amount_ratio_5d") * 0.12
            + _pct_rank_expr("pre_up_ratio") * 0.10
        ).alias("_setup_rank"),
    )
    return sector, meta, bars


def _early_position_score(ret_5d: float, up_ratio: float, limit_rate: float) -> float:
    if ret_5d <= -0.06:
        price_score = 40.0
    elif ret_5d <= 0.08:
        price_score = 88.0 - abs(ret_5d - 0.025) * 280.0
    elif ret_5d <= 0.16:
        price_score = 65.0 - (ret_5d - 0.08) * 350.0
    else:
        price_score = 28.0
    if up_ratio < 0.38:
        price_score -= 12.0
    elif up_ratio > 0.82:
        price_score -= 8.0
    if limit_rate > 0.14:
        price_score -= 15.0
    elif limit_rate > 0.08:
        price_score -= 8.0
    return _clamp(price_score)


def _pre_layout_position_score(pre_ret_5d: float, pre_up_ratio: float, ret_1d: float, limit_rate: float) -> float:
    if pre_ret_5d <= -0.08:
        score = 42.0
    elif pre_ret_5d <= -0.035:
        score = 66.0 + (pre_ret_5d + 0.08) * 360.0
    elif pre_ret_5d <= 0.045:
        score = 92.0 - abs(pre_ret_5d - 0.012) * 260.0
    elif pre_ret_5d <= 0.085:
        score = 76.0 - (pre_ret_5d - 0.045) * 520.0
    else:
        score = 38.0
    if 0.42 <= pre_up_ratio <= 0.72:
        score += 6.0
    elif pre_up_ratio < 0.32:
        score -= 10.0
    elif pre_up_ratio > 0.82:
        score -= 12.0
    if ret_1d > 0.055:
        score -= 28.0
    elif ret_1d > 0.032:
        score -= 16.0
    elif ret_1d > 0.018:
        score -= 8.0
    elif ret_1d < -0.045:
        score -= 8.0
    if limit_rate > 0.10:
        score -= 12.0
    elif limit_rate > 0.055:
        score -= 6.0
    return _clamp(score)


def _today_confirm_score(ret_1d: float, up_ratio: float, amount_ratio: float, limit_rate: float, net_3d: float) -> float:
    score = 50.0
    if ret_1d > 0:
        score += min(ret_1d * 520.0, 22.0)
    if up_ratio >= 0.55:
        score += min((up_ratio - 0.55) * 55.0, 18.0)
    if amount_ratio >= 1.05:
        score += min((amount_ratio - 1.05) * 18.0, 14.0)
    if net_3d > 0:
        score += 8.0
    if limit_rate > 0.08:
        score += 5.0
    return _clamp(score)


def _risk_score(ret_5d: float, up_ratio: float, limit_rate: float, turnover_rate: float, net_5d: float) -> float:
    score = 76.0
    if ret_5d > 0.18:
        score -= 28.0
    elif ret_5d > 0.12:
        score -= 16.0
    if up_ratio > 0.86:
        score -= 12.0
    if limit_rate > 0.15:
        score -= 18.0
    elif limit_rate > 0.08:
        score -= 10.0
    if turnover_rate > 18:
        score -= 10.0
    if net_5d < 0 and ret_5d > 0:
        score -= 18.0
    return _clamp(score)


def _phase_of(
    score: float,
    setup: float,
    fund_layout: float,
    risk: float,
    ret_1d: float,
    ret_5d: float,
    up_ratio: float,
    net_5d: float,
    pre_net_3d: float,
    pre_net_5d: float,
    pre_ret_5d: float,
) -> tuple[str, str]:
    if ret_5d > 0.12 and risk < 55:
        return "高位拥挤", "overheated"
    if net_5d < 0 and score < 45:
        return "退潮", "fading"
    if setup >= 65 and pre_net_5d > 0 and net_5d < 0:
        return "资金分歧", "layout_diverge"
    if (
        setup >= 72
        and (pre_net_3d > 0 or pre_net_5d > 0)
        and -0.05 <= pre_ret_5d <= 0.065
        and ret_1d <= 0.025
        and risk >= 55
    ):
        return "提前布局", "pre_layout"
    if (
        setup >= 65
        and (pre_net_3d > 0 or pre_net_5d > 0)
        and -0.07 <= pre_ret_5d <= 0.085
        and ret_1d <= 0.04
        and risk >= 52
    ):
        return "低位潜伏", "early_watch"
    if setup >= 68 and fund_layout >= 65 and ret_1d > 0.025 and ret_5d <= 0.10:
        return "启动确认", "launch_confirm"
    if fund_layout >= 72 and -0.04 <= ret_5d <= 0.09 and up_ratio >= 0.42:
        return "资金布局", "early_layout"
    if score >= 76 and ret_5d > 0.035 and up_ratio >= 0.55:
        return "加速主升", "accelerating"
    if fund_layout >= 62 and risk >= 55:
        return "潜伏观察", "watch_layout"
    return "观察", "watch"


def _build_rotation_evidence(sector_df: pl.DataFrame) -> dict[str, Any]:
    if sector_df.is_empty():
        return {}
    days = sorted(sector_df["date"].unique().to_list())
    recent = days[-20:]
    spreads: list[float] = []
    top_sets: list[set[str]] = []
    for day in recent:
        daily = sector_df.filter(pl.col("date") == day).sort("ret_1d", descending=True)
        if daily.is_empty():
            continue
        vals = [_float(v) for v in daily["ret_1d"].drop_nulls().to_list()]
        if vals:
            spreads.append(max(vals) - min(vals))
        top_sets.append(set(str(x) for x in daily.head(5)["sector"].to_list()))
    jaccards: list[float] = []
    for prev, curr in zip(top_sets, top_sets[1:], strict=False):
        union = prev | curr
        if union:
            jaccards.append(len(prev & curr) / len(union))
    avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else None
    avg_spread = sum(spreads) / len(spreads) if spreads else None
    turnover = 1.0 - avg_jaccard if avg_jaccard is not None else None
    if turnover is not None and turnover >= 0.70 and (avg_spread or 0) >= 0.035:
        regime = "快速轮动"
    elif turnover is not None and turnover <= 0.45:
        regime = "主线延续"
    else:
        regime = "混合轮动"
    return {
        "regime": regime,
        "top5_jaccard_avg_20d": round(avg_jaccard, 4) if avg_jaccard is not None else None,
        "top5_turnover_avg_20d": round(turnover, 4) if turnover is not None else None,
        "industry_spread_avg_20d": round(avg_spread, 6) if avg_spread is not None else None,
        "rotation_days": len(recent),
    }


def _leader_stocks(bars: pl.DataFrame, target: date, sectors: set[str], *, limit_each: int = 3) -> dict[str, list[dict[str, Any]]]:
    if bars.is_empty() or not sectors:
        return {}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    latest = bars.filter(pl.col("date") == target)
    if latest.is_empty():
        return {}
    latest = latest.sort(["sector", "ret_5d", "ret_1d", "amount"], descending=[False, True, True, True])
    for row in latest.to_dicts():
        sector = str(row.get("sector") or "")
        if sector not in sectors or len(out[sector]) >= limit_each:
            continue
        symbol = _symbol_key(row.get("symbol"))
        out[sector].append({
            "symbol": symbol,
            "name": None,
            "ret_1d": _round(row.get("ret_1d"), 6),
            "ret_5d": _round(row.get("ret_5d"), 6),
            "amount": _round(row.get("amount"), 2),
        })
    return dict(out)


def rows_for_date(data_dir, sector_df, bars, target, limit=100):
    latest = sector_df.filter(pl.col("date") == target)
    latest = latest.with_columns(
        (
            pl.col("limit_up_count").fill_null(0.0)
            / pl.max_horizontal(pl.col("member_count").cast(pl.Float64), pl.lit(1.0))
        ).alias("limit_up_rate")
    )

    rows: list[dict[str, Any]] = []
    for item in latest.to_dicts():
        sector = str(item.get("sector") or "")
        if not sector:
            continue
        member_count = int(item.get("member_count") or 0)
        ret_1d = _float(item.get("ret_1d"))
        ret_3d = _float(item.get("ret_3d"))
        ret_5d = _float(item.get("ret_5d"))
        ret_10d = _float(item.get("ret_10d"))
        up_ratio = _float(item.get("up_ratio"))
        limit_up_count = _float(item.get("limit_up_count"))
        limit_rate = _float(item.get("limit_up_rate"))
        turnover_rate = _float(item.get("turnover_rate"))
        net_3d = _float(item.get("net_3d"))
        net_5d = _float(item.get("net_5d"))
        net_10d = _float(item.get("net_10d"))
        net_20d = _float(item.get("net_20d"))
        large_net_5d = _float(item.get("large_net_5d"))
        amount_ratio = _float(item.get("amount_ratio_5d"), 1.0)
        positive_days = int(_float(item.get("positive_days_5d")))
        pre_ret_1d = _float(item.get("pre_ret_1d"))
        pre_ret_3d = _float(item.get("pre_ret_3d"))
        pre_ret_5d = _float(item.get("pre_ret_5d"))
        pre_ret_10d = _float(item.get("pre_ret_10d"))
        pre_up_ratio = _float(item.get("pre_up_ratio"), 0.5)
        pre_limit_up_count = _float(item.get("pre_limit_up_count"))
        pre_amount_ratio = _float(item.get("pre_amount_ratio_5d"), 1.0)
        pre_turnover_rate = _float(item.get("pre_turnover_rate"))
        pre_net_3d = _float(item.get("pre_net_3d"))
        pre_net_5d = _float(item.get("pre_net_5d"))
        pre_large_net_5d = _float(item.get("pre_large_net_5d"))
        pre_positive_days = int(_float(item.get("pre_positive_days_5d")))

        early_score = _early_position_score(ret_5d, up_ratio, limit_rate)
        pre_position = _pre_layout_position_score(pre_ret_5d, pre_up_ratio, ret_1d, limit_rate)
        setup = _clamp(_float(item.get("_setup_rank"), 0.5) * 100.0 * 0.82 + pre_position * 0.18)
        today_confirm = _today_confirm_score(ret_1d, up_ratio, amount_ratio, limit_rate, net_3d)
        fund_layout = _clamp(_float(item.get("_fund_layout_rank"), 0.5) * 100.0 * 0.78 + early_score * 0.22)
        momentum = _clamp(_float(item.get("_momentum_rank"), 0.5) * 100.0)
        volume = _clamp(_float(item.get("_volume_rank"), 0.5) * 100.0)
        breadth = _clamp(up_ratio * 100.0)
        risk = _risk_score(ret_5d, up_ratio, limit_rate, turnover_rate, net_5d)
        score = _clamp(
            setup * 0.34
            + fund_layout * 0.22
            + today_confirm * 0.14
            + momentum * 0.10
            + volume * 0.08
            + breadth * 0.04
            + risk * 0.08
        )
        phase, signal = _phase_of(
            score,
            setup,
            fund_layout,
            risk,
            ret_1d,
            ret_5d,
            up_ratio,
            net_5d,
            pre_net_3d,
            pre_net_5d,
            pre_ret_5d,
        )
        reasons: list[str] = []
        if pre_net_5d > 0:
            reasons.append(f"昨日前5日净流入{pre_net_5d / 100_000_000:.2f}亿")
        elif pre_net_3d > 0:
            reasons.append(f"昨日前3日净流入{pre_net_3d / 100_000_000:.2f}亿")
        if pre_positive_days:
            reasons.append(f"昨日前5日{pre_positive_days}日净流入")
        if pre_amount_ratio >= 1.08:
            reasons.append(f"昨日前量能放大{pre_amount_ratio:.2f}倍")
        if -0.05 <= pre_ret_5d <= 0.065:
            reasons.append("昨日前涨幅未过热")
        if net_5d > 0:
            reasons.append(f"5日主力净流入{net_5d / 100_000_000:.2f}亿")
        elif net_5d < 0:
            reasons.append(f"5日主力净流出{abs(net_5d) / 100_000_000:.2f}亿")
        if positive_days:
            reasons.append(f"近5日{positive_days}日净流入")
        if amount_ratio >= 1.15:
            reasons.append(f"成交额放大{amount_ratio:.2f}倍")
        if -0.04 <= ret_5d <= 0.09:
            reasons.append("5日涨幅未过热")
        if up_ratio >= 0.55:
            reasons.append(f"上涨家数占比{up_ratio * 100:.1f}%")
        if ret_1d > 0.025:
            reasons.append(f"今日已涨{ret_1d * 100:.1f}%，偏确认信号")
        rows.append({
            **{key: _round(item.get(key), 6) for key in (
                "net_1d", "flow_coverage", "flow_history_days", "flow_intensity", "flow_acceleration",
                "flow_zscore", "prior_flow_intensity", "relative_return", "resilient_ratio", "limit_coverage",
            )},
            "sector": sector,
            "phase": phase,
            "signal": signal,
            "score": round(score, 1),
            "setup_score": round(setup, 1),
            "pre_layout_score": round(setup, 1),
            "today_confirm_score": round(today_confirm, 1),
            "fund_layout_score": round(fund_layout, 1),
            "momentum_score": round(momentum, 1),
            "volume_score": round(volume, 1),
            "breadth_score": round(breadth, 1),
            "risk_score": round(risk, 1),
            "member_count": member_count,
            "ret_1d": _round(item.get("ret_1d"), 6),
            "price_coverage": _round(item.get("price_coverage"), 6),
            "recently_started": bool(item.get("recently_started")),
            "ret_3d": _round(ret_3d, 6),
            "ret_5d": _round(ret_5d, 6),
            "ret_10d": _round(ret_10d, 6),
            "up_ratio": _round(up_ratio, 6),
            "limit_up_count": int(limit_up_count),
            "limit_up_rate": _round(limit_rate, 6),
            "turnover_rate": _round(turnover_rate, 4),
            "amount": _round(item.get("amount"), 2),
            "amount_ratio_5d": _round(amount_ratio, 4),
            "pre_ret_1d": _round(pre_ret_1d, 6),
            "pre_ret_3d": _round(pre_ret_3d, 6),
            "pre_ret_5d": _round(pre_ret_5d, 6),
            "pre_ret_10d": _round(pre_ret_10d, 6),
            "pre_up_ratio": _round(pre_up_ratio, 6),
            "pre_limit_up_count": int(pre_limit_up_count),
            "pre_amount_ratio_5d": _round(pre_amount_ratio, 4),
            "pre_turnover_rate": _round(pre_turnover_rate, 4),
            "pre_net_3d": _round(pre_net_3d, 2),
            "pre_net_5d": _round(pre_net_5d, 2),
            "pre_large_net_5d": _round(pre_large_net_5d, 2),
            "pre_positive_days_5d": pre_positive_days,
            "net_3d": _round(net_3d, 2),
            "net_5d": _round(net_5d, 2),
            "net_10d": _round(net_10d, 2),
            "net_20d": _round(net_20d, 2),
            "large_net_5d": _round(large_net_5d, 2),
            "positive_days_5d": positive_days,
            "up_prob_hint_3d": round(_clamp(45.0 + (score - 50.0) * 0.38 + (setup - 50.0) * 0.24 - max(ret_5d - 0.10, 0.0) * 140.0, 20.0, 85.0), 1),
            "reasons": reasons[:7],
        })

    rows.sort(
        key=lambda row: (
            row.get("signal") in {"pre_layout", "early_watch"},
            _float(row.get("setup_score")),
            _float(row.get("score")),
            _float(row.get("fund_layout_score")),
        ),
        reverse=True,
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    top_rows = rows[:limit]
    sector_set = {str(row["sector"]) for row in top_rows}
    leaders = _leader_stocks(bars, target, sector_set)
    name_map = _load_name_map(data_dir)
    for members in leaders.values():
        for member in members:
            if not member.get("name"):
                member["name"] = name_map.get(str(member.get("symbol") or ""))
    for row in top_rows:
        row["leader_stocks"] = leaders.get(str(row.get("sector") or ""), [])

    return rows


def input_signature(data_dir, days):
    entries = []
    for table in ("kline_daily", "kline_daily_enriched", "moneyflow", "kline_index_daily"):
        for day in days:
            for path in sorted((data_dir / table / f"date={day}").glob("*.parquet")):
                stat = path.stat()
                entries.append((table, str(day), path.name, stat.st_size, stat.st_mtime_ns))
    for path in sorted((data_dir / "ext_data" / "ext_hy_ths").rglob("*.parquet")):
        stat = path.stat()
        entries.append(("industry", "", path.name, stat.st_size, stat.st_mtime_ns))
    return entries


def build_sector_rotation(
    data_dir: Path,
    *,
    as_of: date | None = None,
    lookback_days: int = 60,
    level: int = 2,
    limit: int = 30,
) -> dict[str, Any]:
    """Build latest industry rotation candidates and early fund-layout signals."""
    t0 = time.perf_counter()
    level = max(1, min(3, int(level)))
    limit = max(1, min(100, int(limit)))
    root = data_dir / "kline_daily_enriched"
    if not root.exists():
        root = data_dir / "kline_daily"
    all_dates = _partition_dates(root, end=as_of)
    selected_dates = all_dates[-max(25, min(160, int(lookback_days))):]
    cache_key = (
        str(data_dir),
        as_of.isoformat() if as_of else None,
        selected_dates[0].isoformat() if selected_dates else None,
        selected_dates[-1].isoformat() if selected_dates else None,
        len(selected_dates),
        level,
        limit,
        tuple(input_signature(data_dir, selected_dates)),
    )
    cached = _CACHE.get(cache_key)
    now = time.monotonic()
    if cached and now - cached[0] <= _CACHE_TTL:
        return _json_safe(cached[1])

    if not selected_dates:
        payload = {
            "as_of": None,
            "rows": [],
            "leaders": [],
            "early_layout": [],
            "meta": {"warning": "本地日K为空，无法计算板块轮动。"},
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        _CACHE[cache_key] = (now, payload)
        return _json_safe(payload)

    sector_df, meta, bars = _sector_daily_frame(data_dir, selected_dates, level)
    if sector_df.is_empty():
        payload = {
            "as_of": selected_dates[-1],
            "rows": [],
            "leaders": [],
            "early_layout": [],
            "meta": {**meta, "warning": "行业映射或日K数据不足，无法计算板块轮动。"},
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        _CACHE[cache_key] = (now, payload)
        return _json_safe(payload)

    target = sector_df["date"].max()
    rows = rows_for_date(data_dir, sector_df, bars, target)
    top_rows = rows[:limit]

    evidence = _build_rotation_evidence(sector_df)
    flow_latest = meta.get("moneyflow", {}).get("date_end")
    lag_days = None
    try:
        if flow_latest and isinstance(target, date):
            lag_days = (target - date.fromisoformat(str(flow_latest))).days
    except ValueError:
        lag_days = None
    phase_counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        phase_counts[str(row.get("phase") or "观察")] += 1
    payload = {
        "as_of": target,
        "rows": top_rows,
        "leaders": rows[:10],
        "early_layout": [
            row
            for row in rows
            if row.get("signal") in {"pre_layout", "early_watch", "early_layout", "watch_layout"}
        ][:12],
        "meta": {
            **meta,
            **evidence,
            "flow_latest_date": flow_latest,
            "flow_lag_days": lag_days,
            "sector_count": len(rows),
            "phase_counts": dict(phase_counts),
            "method": "昨日以前资金布局 + 温和放量 + 宽度扩散 + 今日不过热；今日上涨只作为确认信号，不作为提前布局主因。",
        },
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    _CACHE[cache_key] = (now, payload)
    return _json_safe(payload)
