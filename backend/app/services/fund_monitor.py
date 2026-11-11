"""Fund monitoring service.

Two boards are exposed:
- daily: confirmed Tushare moneyflow based ranking.
- intraday: realtime quote + minute bar based approximation.

The intraday board deliberately uses "estimated" wording because the current
system does not have tick-by-tick Level2 order flow.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from app.market_time import cn_today, trading_minutes_elapsed
from app.parquet import read_parquet_safe

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_FLOW_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, dict[str, Any]], dict[str, Any]]] = {}
_MARKET_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, dict[str, Any]], dict[str, Any]]] = {}
_MINUTE_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, dict[str, Any]], dict[str, Any]]] = {}
_FLOW_TTL = 45.0
_MARKET_TTL = 20.0
_MINUTE_TTL = 5.0


DAILY_SIGNAL_LABELS: dict[str, str] = {
    "main_accumulation": "主力吸筹",
    "early_layout": "资金初动",
    "low_absorption": "低位吸筹",
    "strong_inflow": "持续强流入",
    "inflow_confirm": "流入确认",
    "watch": "观察",
    "distribution_risk": "拉高派发风险",
    "outflow": "资金流出",
}

INTRADAY_SIGNAL_LABELS: dict[str, str] = {
    "active_inflow": "盘中主动流入",
    "breakout_flow": "放量突破",
    "watch": "观察",
    "distribution_risk": "冲高派发风险",
    "outflow": "盘中流出",
}

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
    "buy_sm_vol",
    "sell_sm_vol",
    "buy_md_vol",
    "sell_md_vol",
    "buy_lg_vol",
    "sell_lg_vol",
    "buy_elg_vol",
    "sell_elg_vol",
    "net_mf_vol",
]

_BAR_NUMERIC_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "prev_close",
    "change_pct",
    "turnover_rate",
    "consecutive_limit_ups",
    "consecutive_limit_downs",
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


def _safe_div(num: float | None, den: float | None, default: float = 0.0) -> float:
    if num is None or den is None or abs(den) <= 1e-9:
        return default
    return num / den


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: Any, digits: int = 4) -> float | None:
    out = _float_or_none(value)
    return None if out is None else round(out, digits)


def _symbol_key(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text:
        return ""
    if text.startswith(("SH", "SZ", "BJ")) and len(text) >= 8:
        text = text[2:]
    if "." in text:
        text = text.split(".", 1)[0]
    if text.isdigit() and len(text) < 6:
        text = text.zfill(6)
    return text


def _symbol_aliases(symbol: str) -> list[str]:
    key = _symbol_key(symbol)
    if not key:
        return []
    from app.services.security_identity import exchange_for_code
    suffix = exchange_for_code(key, "stock")
    if suffix is None:
        return [key]
    return [key, f"{key}.{suffix}", f"{suffix}{key}"]


def _board_of(symbol: str) -> str:
    code = _symbol_key(symbol)
    if code.startswith(("43", "83", "87", "92")):
        return "北交所"
    if code.startswith("68"):
        return "科创板"
    if code.startswith(("30", "301")):
        return "创业板"
    return "主板"


def _date_to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


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
        if df is None or df.is_empty():
            if df is None:
                skipped += 1
            continue
        if "date" not in df.columns:
            df = df.with_columns(pl.lit(day).cast(pl.Date).alias("date"))
        frames.append(df)
    if not frames:
        return pl.DataFrame(), skipped
    try:
        return pl.concat(frames, how="diagonal_relaxed"), skipped
    except Exception as exc:  # noqa: BLE001
        logger.warning("concat partition frames failed: %s", exc)
        return pl.DataFrame(), skipped + len(frames)


def _parse_date_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8, strict=False).str.replace_all("/", "-")
    return (
        pl.when(text.str.contains(r"^\d{8}$"))
        .then(text.str.strptime(pl.Date, "%Y%m%d", strict=False))
        .otherwise(text.str.strptime(pl.Date, "%Y-%m-%d", strict=False))
    )


def _normalize_symbol_column(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "symbol" not in df.columns:
        return df
    return df.with_columns(
        pl.col("symbol")
        .map_elements(_symbol_key, return_dtype=pl.Utf8)
        .alias("symbol")
    ).filter(pl.col("symbol") != "")


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


def _normalize_datetime_column(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "datetime" not in df.columns:
        return df
    dtype = df.schema.get("datetime")
    if dtype == pl.Datetime:
        return df
    if dtype == pl.Date:
        return df.with_columns(pl.col("datetime").cast(pl.Datetime).alias("datetime"))
    text = pl.col("datetime").cast(pl.Utf8, strict=False).str.replace_all("/", "-")
    return df.with_columns(text.str.strptime(pl.Datetime, strict=False).alias("datetime"))


def _cast_numeric(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    exprs = [
        pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0).alias(col)
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


def _pct_decimal(value: Any) -> float | None:
    out = _float_or_none(value)
    if out is None:
        return None
    # Most enriched fields use decimal values. Some provider payloads use
    # percentage points; normalize only clearly percentage-like values.
    return out / 100.0 if abs(out) > 1.0 else out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _clone_metrics(value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(k): dict(v) for k, v in value.items()}


def _cache_get(
    cache: dict[tuple[Any, ...], tuple[float, dict[str, dict[str, Any]], dict[str, Any]]],
    key: tuple[Any, ...],
    ttl: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | None:
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = cache.get(key)
    if cached is None or now - cached[0] > ttl:
        return None
    return _clone_metrics(cached[1]), dict(cached[2])


def _cache_put(
    cache: dict[tuple[Any, ...], tuple[float, dict[str, dict[str, Any]], dict[str, Any]]],
    key: tuple[Any, ...],
    metrics: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> None:
    with _CACHE_LOCK:
        cache[key] = (time.monotonic(), _clone_metrics(metrics), dict(meta))


def _load_moneyflow_metrics(
    data_dir: Path,
    *,
    window: int = 20,
    end: date | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = data_dir / "moneyflow"
    all_dates = _partition_dates(root, end=end)
    selected_dates = all_dates[-max(1, min(120, int(window))):]
    cache_key = (
        "moneyflow",
        str(root),
        int(window),
        selected_dates[0].isoformat() if selected_dates else None,
        selected_dates[-1].isoformat() if selected_dates else None,
        len(selected_dates),
    )
    cached = _cache_get(_FLOW_CACHE, cache_key, _FLOW_TTL)
    if cached is not None:
        return cached
    meta: dict[str, Any] = {
        "source": "tushare_moneyflow_local",
        "table": "moneyflow",
        "requested_window": window,
        "date_start": selected_dates[0].isoformat() if selected_dates else None,
        "date_end": selected_dates[-1].isoformat() if selected_dates else None,
        "trading_days": len(selected_dates),
        "rows": 0,
        "symbols": 0,
        "skipped_partitions": 0,
    }
    if not selected_dates:
        meta["note"] = "本地 moneyflow 分区为空，请先同步 Tushare 主力资金流。"
        return {}, meta

    df, skipped = _read_partitions(root, selected_dates)
    meta["skipped_partitions"] = skipped
    if df.is_empty():
        meta["note"] = "moneyflow 分区存在，但没有可读取记录。"
        return {}, meta

    df = _normalize_date_column(_normalize_symbol_column(df))
    if "symbol" not in df.columns or "date" not in df.columns:
        meta["note"] = "moneyflow 缺少 symbol/date 字段。"
        return {}, meta

    df = _cast_numeric(df, _FLOW_NUMERIC_COLS)
    columns = df.columns
    gross_buy = _sum_columns(columns, ["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount"])
    gross_sell = _sum_columns(columns, ["sell_sm_amount", "sell_md_amount", "sell_lg_amount", "sell_elg_amount"])
    large_net = _sum_columns(columns, ["buy_lg_amount"]) - _sum_columns(columns, ["sell_lg_amount"])
    extra_large_net = _sum_columns(columns, ["buy_elg_amount"]) - _sum_columns(columns, ["sell_elg_amount"])
    main_net = (
        pl.col("net_mf_amount").cast(pl.Float64, strict=False).fill_null(0.0)
        if "net_mf_amount" in columns
        else large_net + extra_large_net
    )
    daily = (
        df.with_columns(
            main_net.alias("_main_net"),
            large_net.alias("_large_net"),
            extra_large_net.alias("_extra_large_net"),
            pl.max_horizontal(gross_buy, gross_sell, main_net.abs()).alias("_turnover_proxy"),
        )
        .group_by(["symbol", "date"])
        .agg(
            pl.sum("_main_net").alias("main_net"),
            pl.sum("_large_net").alias("large_net"),
            pl.sum("_extra_large_net").alias("extra_large_net"),
            pl.sum("_turnover_proxy").alias("turnover_proxy"),
            pl.len().alias("row_count"),
        )
        .sort(["symbol", "date"])
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily.to_dicts():
        symbol = _symbol_key(row.get("symbol"))
        day = row.get("date")
        if not symbol or not isinstance(day, date):
            continue
        grouped[symbol].append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for symbol, items in grouped.items():
        items.sort(key=lambda r: r["date"])
        last_items = items[-window:]

        def sum_rows(rows: list[dict[str, Any]], field: str) -> float:
            return sum(_float(row.get(field)) for row in rows)

        def sum_last(field: str, n: int) -> float:
            return sum_rows(last_items[-n:], field)

        net_1d = sum_last("main_net", 1)
        net_3d = sum_last("main_net", 3)
        net_5d = sum_last("main_net", 5)
        net_10d = sum_last("main_net", 10)
        net_window = sum_last("main_net", window)
        prev3_items = last_items[-6:-3] if len(last_items) >= 6 else last_items[:-3]
        prev5_items = last_items[-10:-5] if len(last_items) >= 10 else last_items[:-5]
        prev_net_3d = sum_rows(prev3_items, "main_net")
        prev_net_5d = sum_rows(prev5_items, "main_net")
        large_5d = sum_last("large_net", 5)
        extra_large_5d = sum_last("extra_large_net", 5)
        turnover_1d = max(1.0, sum_last("turnover_proxy", 1))
        turnover_3d = max(1.0, sum_last("turnover_proxy", 3))
        turnover_5d = max(1.0, sum_last("turnover_proxy", 5))
        turnover_window = max(1.0, sum_last("turnover_proxy", window))
        prev_turnover_3d = max(1.0, sum_rows(prev3_items, "turnover_proxy"))
        net_ratio_1d = _safe_div(net_1d, turnover_1d)
        net_ratio_3d = _safe_div(net_3d, turnover_3d)
        prev_net_ratio_3d = _safe_div(prev_net_3d, prev_turnover_3d)
        moneyflow_acceleration_3d = net_ratio_3d - prev_net_ratio_3d
        last5 = last_items[-5:]
        last10 = last_items[-10:]
        positive_days_5d = sum(1 for row in last5 if _float(row.get("main_net")) > 0)
        positive_days_10d = sum(1 for row in last10 if _float(row.get("main_net")) > 0)
        positive_streak = 0
        for row in reversed(last_items):
            if _float(row.get("main_net")) <= 0:
                break
            positive_streak += 1
        metrics[symbol] = {
            "symbol": symbol,
            "last_flow_date": items[-1]["date"],
            "flow_days": len(last_items),
            "net_1d": net_1d,
            "net_3d": net_3d,
            "net_5d": net_5d,
            "net_10d": net_10d,
            "net_window": net_window,
            "prev_net_3d": prev_net_3d,
            "prev_net_5d": prev_net_5d,
            "large_net_5d": large_5d,
            "extra_large_net_5d": extra_large_5d,
            "turnover_1d": turnover_1d,
            "turnover_3d": turnover_3d,
            "turnover_5d": turnover_5d,
            "turnover_window": turnover_window,
            "net_ratio_1d": net_ratio_1d,
            "net_ratio_3d": net_ratio_3d,
            "net_ratio_5d": _safe_div(net_5d, turnover_5d),
            "net_ratio_window": _safe_div(net_window, turnover_window),
            "prev_net_ratio_3d": prev_net_ratio_3d,
            "large_ratio_5d": _safe_div(large_5d + extra_large_5d, turnover_5d),
            "moneyflow_acceleration_3d": moneyflow_acceleration_3d,
            "moneyflow_turn_positive": net_1d > 0 and prev_net_3d <= 0,
            "positive_days_5d": positive_days_5d,
            "positive_days_10d": positive_days_10d,
            "positive_streak": positive_streak,
        }

    meta["rows"] = int(df.height)
    meta["symbols"] = len(metrics)
    _cache_put(_FLOW_CACHE, cache_key, metrics, meta)
    return metrics, meta


def _load_industry_map(data_dir: Path, symbols: set[str]) -> dict[str, str]:
    if not symbols:
        return {}
    path = data_dir / "ext_data" / "ext_hy_ths" / "part.parquet"
    df = read_parquet_safe(path)
    if df is None or df.is_empty():
        return {}
    out: dict[str, str] = {}
    for row in df.to_dicts():
        key = _symbol_key(row.get("symbol") or row.get("code"))
        if not key or key not in symbols:
            continue
        for col, value in row.items():
            if col in {"symbol", "code"} or value is None:
                continue
            text = str(value).strip()
            if not text or text.upper() == key:
                continue
            if "-" in text or "行业" in str(col):
                out[key] = text
                break
    return out


def _load_name_map(repo: Any, symbols: set[str]) -> dict[str, str]:
    if not symbols or repo is None or not hasattr(repo, "get_name_map"):
        return {}
    candidates: list[str] = []
    for symbol in symbols:
        candidates.extend(_symbol_aliases(symbol))
    try:
        raw = repo.get_name_map(candidates) or {}
    except Exception:
        return {}
    out: dict[str, str] = {}
    for symbol in symbols:
        for alias in _symbol_aliases(symbol):
            name = raw.get(alias)
            if name:
                out[symbol] = str(name)
                break
    return out


def _load_market_metrics(
    repo: Any,
    quote_service: Any = None,
    *,
    history_days: int = 45,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    data_dir = repo.store.data_dir
    root = data_dir / "kline_daily_enriched"
    if not root.exists():
        root = data_dir / "kline_daily"
    all_dates = _partition_dates(root)
    selected_dates = all_dates[-max(2, min(80, int(history_days))):]
    cache_key = (
        "market",
        str(root),
        selected_dates[0].isoformat() if selected_dates else None,
        selected_dates[-1].isoformat() if selected_dates else None,
        len(selected_dates),
    )
    cached = _cache_get(_MARKET_CACHE, cache_key, _MARKET_TTL)
    if cached is not None:
        metrics, meta = cached
        if quote_service is not None:
            try:
                meta["quote_status"] = quote_service.status() or {}
            except Exception:
                meta["quote_status"] = {}
            _overlay_live_quotes(metrics, quote_service, meta)
        return metrics, meta
    meta: dict[str, Any] = {
        "source": "local_enriched",
        "date_start": selected_dates[0].isoformat() if selected_dates else None,
        "date_end": selected_dates[-1].isoformat() if selected_dates else None,
        "trading_days": len(selected_dates),
        "rows": 0,
        "quote_status": {},
        "quote_date": None,
    }
    df, skipped = _read_partitions(root, selected_dates)
    meta["skipped_partitions"] = skipped
    if df.is_empty():
        return {}, meta

    df = _normalize_date_column(_normalize_symbol_column(df))
    df = _cast_numeric(df, _BAR_NUMERIC_COLS)
    repair_prices = getattr(repo, "_repair_corrupt_adjusted_prices", None)
    if callable(repair_prices):
        df = repair_prices(df)
    if "symbol" not in df.columns or "date" not in df.columns or "close" not in df.columns:
        return {}, meta

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in df.sort(["symbol", "date"]).to_dicts():
        symbol = _symbol_key(row.get("symbol"))
        day = row.get("date")
        close = _float_or_none(row.get("close"))
        if not symbol or not isinstance(day, date) or close is None or close <= 0:
            continue
        grouped[symbol].append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for symbol, items in grouped.items():
        items.sort(key=lambda r: r["date"])
        latest = items[-1]
        prev = items[-2] if len(items) >= 2 else {}
        closes = [_float(row.get("close")) for row in items if _float(row.get("close")) > 0]
        highs = [_float(row.get("high")) for row in items if _float(row.get("high")) > 0]
        lows = [_float(row.get("low")) for row in items if _float(row.get("low")) > 0]
        amounts = [_float(row.get("amount")) for row in items if _float(row.get("amount")) > 0]

        def avg_last(values: list[float], n: int) -> float | None:
            vals = values[-n:]
            return sum(vals) / len(vals) if vals else None

        def return_over(days: int) -> float | None:
            if len(closes) <= days:
                return None
            base = closes[-days - 1]
            return close / base - 1 if base > 0 else None

        def std_last_returns(days: int) -> float | None:
            if len(closes) <= 1:
                return None
            start = max(1, len(closes) - days)
            rets = [
                closes[idx] / closes[idx - 1] - 1
                for idx in range(start, len(closes))
                if closes[idx - 1] > 0
            ]
            if len(rets) < 2:
                return None
            avg = sum(rets) / len(rets)
            return math.sqrt(sum((ret - avg) ** 2 for ret in rets) / (len(rets) - 1))

        close = _float(latest.get("close"))
        prev_close = _float_or_none(latest.get("prev_close")) or _float_or_none(prev.get("close"))
        change_pct = _pct_decimal(latest.get("change_pct"))
        if change_pct is None and prev_close and prev_close > 0:
            change_pct = close / prev_close - 1
        high20 = max(highs[-20:]) if highs[-20:] else None
        low20 = min(lows[-20:]) if lows[-20:] else None
        high60 = max(highs[-60:]) if highs[-60:] else high20
        low60 = min(lows[-60:]) if lows[-60:] else low20
        high10 = max(highs[-10:]) if highs[-10:] else high20
        low10 = min(lows[-10:]) if lows[-10:] else low20
        position_20 = _safe_div(close - low20, high20 - low20, 0.5) if high20 is not None and low20 is not None else 0.5
        position_60 = _safe_div(close - low60, high60 - low60, 0.5) if high60 is not None and low60 is not None else position_20
        avg_amount20 = avg_last(amounts[:-1] or amounts, 20)
        avg_amount5 = avg_last(amounts[:-1] or amounts, 5)
        amount = _float_or_none(latest.get("amount"))
        metrics[symbol] = {
            "symbol": symbol,
            "market_date": latest["date"],
            "open": _float_or_none(latest.get("open")),
            "high": _float_or_none(latest.get("high")),
            "low": _float_or_none(latest.get("low")),
            "close": close,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "volume": _float_or_none(latest.get("volume")),
            "amount": amount,
            "turnover_rate": _float_or_none(latest.get("turnover_rate")),
            "ma5": avg_last(closes, 5),
            "ma10": avg_last(closes, 10),
            "ma20": avg_last(closes, 20),
            "avg_amount5": avg_amount5,
            "avg_amount20": avg_amount20,
            "amount_ratio_20d": _safe_div(amount, avg_amount20, None) if amount is not None else None,
            "amount_ratio_5d_20d": _safe_div(avg_amount5, avg_amount20, None) if avg_amount5 is not None else None,
            "high20": high20,
            "low20": low20,
            "high60": high60,
            "low60": low60,
            "high10": high10,
            "low10": low10,
            "position_20": _clamp(position_20, 0.0, 1.0),
            "position_60": _clamp(position_60, 0.0, 1.0),
            "range_ratio_10d": _safe_div(high10 - low10, close, None) if high10 is not None and low10 is not None and close > 0 else None,
            "ret_3d": return_over(3),
            "ret_5d": return_over(5),
            "ret_10d": return_over(10),
            "ret_20d": return_over(20),
            "volatility_10d": std_last_returns(10),
            "consecutive_limit_ups": _float_or_none(latest.get("consecutive_limit_ups")),
            "consecutive_limit_downs": _float_or_none(latest.get("consecutive_limit_downs")),
        }

    meta["rows"] = int(df.height)

    symbols = set(metrics.keys())
    names = _load_name_map(repo, symbols)
    industries = _load_industry_map(data_dir, symbols)
    for symbol, item in metrics.items():
        item["name"] = names.get(symbol)
        item["industry"] = industries.get(symbol)
        item["board"] = _board_of(symbol)
    _cache_put(_MARKET_CACHE, cache_key, metrics, meta)
    if quote_service is not None:
        try:
            meta["quote_status"] = quote_service.status() or {}
        except Exception:
            meta["quote_status"] = {}
        _overlay_live_quotes(metrics, quote_service, meta)
    return metrics, meta


def _overlay_live_quotes(
    metrics: dict[str, dict[str, Any]],
    quote_service: Any,
    meta: dict[str, Any],
) -> None:
    frames: list[pl.DataFrame] = []
    live_date: date | None = None
    try:
        enriched, live_date = quote_service.get_enriched_today()
        if enriched is not None and not enriched.is_empty():
            frames.append(enriched)
    except Exception as exc:  # noqa: BLE001
        logger.debug("live enriched overlay skipped: %s", exc)
    try:
        compat = quote_service.get_quotes_compat()
        if compat is not None and not compat.is_empty():
            frames.append(compat)
    except Exception as exc:  # noqa: BLE001
        logger.debug("quote compat overlay skipped: %s", exc)

    meta["quote_date"] = live_date.isoformat() if isinstance(live_date, date) else _date_to_iso(live_date)
    for frame in frames:
        frame = _normalize_symbol_column(frame)
        frame = _cast_numeric(frame, _BAR_NUMERIC_COLS + ["last_price"])
        if frame.is_empty() or "symbol" not in frame.columns:
            continue
        keep = [c for c in [
            "symbol",
            "name",
            "open",
            "high",
            "low",
            "close",
            "last_price",
            "prev_close",
            "change_pct",
            "volume",
            "amount",
            "turnover_rate",
            "consecutive_limit_ups",
            "consecutive_limit_downs",
        ] if c in frame.columns]
        for row in frame.select(keep).to_dicts():
            symbol = _symbol_key(row.get("symbol"))
            if not symbol:
                continue
            item = metrics.setdefault(symbol, {"symbol": symbol, "board": _board_of(symbol)})
            price = _float_or_none(row.get("last_price")) or _float_or_none(row.get("close"))
            prev_close = _float_or_none(row.get("prev_close")) or _float_or_none(item.get("prev_close"))
            if price is not None:
                item["close"] = price
            for col in ("open", "high", "low", "volume", "amount", "turnover_rate",
                        "consecutive_limit_ups", "consecutive_limit_downs"):
                value = _float_or_none(row.get(col))
                if value is not None:
                    item[col] = value
            change_pct = _pct_decimal(row.get("change_pct"))
            if change_pct is None and price is not None and prev_close and prev_close > 0:
                change_pct = price / prev_close - 1
            if change_pct is not None:
                item["change_pct"] = change_pct
            if prev_close is not None:
                item["prev_close"] = prev_close
            if row.get("name"):
                item["name"] = str(row["name"])


def _daily_scores(flow: dict[str, Any], market: dict[str, Any]) -> tuple[float, float, float, float, str, list[str], dict[str, Any]]:
    net_ratio_1d = _float(flow.get("net_ratio_1d"))
    net_ratio_3d = _float(flow.get("net_ratio_3d"))
    net_ratio_5d = _float(flow.get("net_ratio_5d"))
    net_ratio_window = _float(flow.get("net_ratio_window"))
    large_ratio_5d = _float(flow.get("large_ratio_5d"))
    positive_days_5d = int(flow.get("positive_days_5d") or 0)
    positive_days_10d = int(flow.get("positive_days_10d") or 0)
    positive_streak = int(flow.get("positive_streak") or 0)
    acceleration = _float(flow.get("moneyflow_acceleration_3d"))
    turn_positive = bool(flow.get("moneyflow_turn_positive"))
    net_5d = _float(flow.get("net_5d"))
    net_3d = _float(flow.get("net_3d"))
    net_1d = _float(flow.get("net_1d"))

    close = _float_or_none(market.get("close"))
    ma5 = _float_or_none(market.get("ma5"))
    ma10 = _float_or_none(market.get("ma10"))
    ma20 = _float_or_none(market.get("ma20"))
    change_pct = _float_or_none(market.get("change_pct")) or 0.0
    turnover_rate = _float_or_none(market.get("turnover_rate")) or 0.0
    position_20 = _float_or_none(market.get("position_20"))
    position_60 = _float_or_none(market.get("position_60"))
    amount_ratio = _float_or_none(market.get("amount_ratio_20d")) or 1.0
    amount_ratio_5d_20d = _float_or_none(market.get("amount_ratio_5d_20d")) or amount_ratio
    range_ratio_10d = _float_or_none(market.get("range_ratio_10d"))
    volatility_10d = _float_or_none(market.get("volatility_10d"))
    ret_3d = _float_or_none(market.get("ret_3d")) or change_pct
    ret_5d = _float_or_none(market.get("ret_5d")) or change_pct
    ret_10d = _float_or_none(market.get("ret_10d")) or ret_5d
    limit_ups = _float_or_none(market.get("consecutive_limit_ups")) or 0.0
    name_text = str(market.get("name") or "").upper()
    is_special_treatment = "ST" in name_text or "退市" in name_text

    price_setup = 50.0
    if position_20 is not None:
        if 0.18 <= position_20 <= 0.68:
            price_setup += 18.0
        elif 0.68 < position_20 <= 0.82:
            price_setup += 6.0
        elif position_20 > 0.90:
            price_setup -= 22.0
        elif position_20 < 0.12:
            price_setup -= 8.0
    if -0.018 <= change_pct <= 0.018:
        price_setup += 8.0
    elif 0.018 < change_pct <= 0.035:
        price_setup -= 4.0
    elif 0.035 < change_pct <= 0.055:
        price_setup -= 14.0
    elif change_pct > 0.055:
        price_setup -= 28.0
    if 1.08 <= amount_ratio <= 2.30:
        price_setup += min(12.0, (amount_ratio - 1.0) * 9.0)
    elif amount_ratio >= 3.20:
        price_setup -= 10.0
    price_setup = _clamp(price_setup, 0.0, 100.0)

    base_zone_score = 50.0
    if position_20 is not None:
        if 0.18 <= position_20 <= 0.60:
            base_zone_score += 22.0
        elif 0.60 < position_20 <= 0.72:
            base_zone_score += 10.0
        elif position_20 > 0.82:
            base_zone_score -= 24.0
        elif position_20 < 0.10:
            base_zone_score -= 8.0
    if position_60 is not None:
        if 0.16 <= position_60 <= 0.55:
            base_zone_score += 14.0
        elif 0.55 < position_60 <= 0.68:
            base_zone_score += 5.0
        elif position_60 > 0.78:
            base_zone_score -= 14.0
    base_zone_score = _clamp(base_zone_score, 0.0, 100.0)

    compression_score = 50.0
    if range_ratio_10d is not None:
        if 0.035 <= range_ratio_10d <= 0.13:
            compression_score += 24.0
        elif range_ratio_10d < 0.035:
            compression_score += 8.0
        elif range_ratio_10d <= 0.18:
            compression_score += 8.0
        else:
            compression_score -= min(24.0, (range_ratio_10d - 0.18) * 130.0)
    if volatility_10d is not None:
        if volatility_10d <= 0.025:
            compression_score += 12.0
        elif volatility_10d >= 0.055:
            compression_score -= 12.0
    if -0.045 <= ret_5d <= 0.055:
        compression_score += 12.0
    elif ret_5d > 0.09:
        compression_score -= 22.0
    if ret_10d > 0.16:
        compression_score -= 12.0
    compression_score = _clamp(compression_score, 0.0, 100.0)

    volume_absorption_score = 50.0
    if 0.75 <= amount_ratio <= 2.15:
        volume_absorption_score += 18.0
    elif 2.15 < amount_ratio <= 2.80:
        volume_absorption_score += 5.0
    elif amount_ratio > 3.20:
        volume_absorption_score -= 18.0
    elif amount_ratio < 0.55:
        volume_absorption_score -= 8.0
    if 0.82 <= amount_ratio_5d_20d <= 1.65:
        volume_absorption_score += 12.0
    elif amount_ratio_5d_20d > 2.20:
        volume_absorption_score -= 10.0
    if turnover_rate and turnover_rate <= 12:
        volume_absorption_score += 6.0
    elif turnover_rate >= 22:
        volume_absorption_score -= 16.0
    volume_absorption_score = _clamp(volume_absorption_score, 0.0, 100.0)

    inflow_age_risk = 0.0
    if positive_streak > 3:
        inflow_age_risk += _clamp((positive_streak - 3) / 4.0, 0.0, 1.0) * 0.55
    if positive_days_10d >= 7:
        inflow_age_risk += 0.16
    if position_20 is not None and position_20 > 0.76 and positive_streak >= 4:
        inflow_age_risk += 0.28
    if change_pct > 0.045:
        inflow_age_risk += 0.16
    if amount_ratio > 2.8 and positive_streak >= 4:
        inflow_age_risk += 0.12
    inflow_age_risk = _clamp(inflow_age_risk, 0.0, 1.0)

    absorption_efficiency = 0.0
    if net_ratio_3d > 0:
        absorption_efficiency = _clamp(net_ratio_3d / max(abs(ret_3d), 0.008), 0.0, 1.0)
    quiet_price = ret_3d <= 0.055 and ret_5d <= 0.065 and ret_10d <= 0.12 and change_pct <= 0.035
    accumulation_score = 30.0
    accumulation_score += (base_zone_score - 50.0) * 0.22
    accumulation_score += (compression_score - 50.0) * 0.22
    accumulation_score += (volume_absorption_score - 50.0) * 0.14
    accumulation_score += _clamp(net_ratio_3d * 210.0, -10.0, 10.0)
    accumulation_score += _clamp(net_ratio_1d * 160.0, -7.0, 7.0)
    accumulation_score += _clamp(large_ratio_5d * 120.0, -6.0, 6.0)
    accumulation_score += _clamp(acceleration * 170.0, -7.0, 8.0)
    accumulation_score += absorption_efficiency * 5.0
    accumulation_score += 3.0 if turn_positive else 0.0
    accumulation_score += 3.0 if quiet_price else -10.0
    if positive_streak > 4:
        accumulation_score -= (positive_streak - 4) * 5.0
    if inflow_age_risk >= 0.45:
        accumulation_score -= 18.0
    if limit_ups >= 1:
        accumulation_score -= min(18.0, limit_ups * 8.0)
    if base_zone_score < 45.0:
        accumulation_score = min(accumulation_score, 62.0)
    elif base_zone_score < 58.0:
        accumulation_score = min(accumulation_score, 74.0)
    if position_60 is not None and position_60 > 0.78:
        accumulation_score = min(accumulation_score, 66.0)
    if position_20 is not None and position_20 > 0.82:
        accumulation_score = min(accumulation_score, 66.0)
    if ret_5d > 0.08 or ret_10d > 0.14:
        accumulation_score = min(accumulation_score, 68.0)
    if is_special_treatment:
        accumulation_score = min(accumulation_score, 45.0)
    accumulation_score = _clamp(accumulation_score, 0.0, 100.0)

    early_layout_score = 50.0
    early_layout_score += _clamp(acceleration * 620.0, -22.0, 28.0)
    early_layout_score += _clamp(net_ratio_1d * 480.0, -20.0, 24.0)
    early_layout_score += _clamp(net_ratio_3d * 260.0, -14.0, 18.0)
    early_layout_score += _clamp(large_ratio_5d * 150.0, -8.0, 10.0)
    early_layout_score += 8.0 if turn_positive else 0.0
    early_layout_score += (price_setup - 50.0) * 0.42
    early_layout_score -= inflow_age_risk * 26.0
    if positive_streak >= 5:
        cap = 62.0 if (position_20 or 0.5) > 0.72 else 68.0
        early_layout_score = min(early_layout_score, cap)
    elif positive_streak >= 4 and (position_20 or 0.5) > 0.76:
        early_layout_score = min(early_layout_score - 8.0, 68.0)
    early_layout_score = _clamp(early_layout_score, 0.0, 100.0)

    confirm_score = 50.0
    confirm_score += _clamp(net_ratio_3d * 280.0, -16.0, 20.0)
    confirm_score += _clamp(net_ratio_5d * 220.0, -14.0, 16.0)
    confirm_score += _clamp(net_ratio_window * 120.0, -6.0, 6.0)
    confirm_score += _clamp(large_ratio_5d * 190.0, -10.0, 10.0)
    confirm_score += min(positive_days_5d, 3) * 2.0
    confirm_score += min(positive_streak, 3) * 1.2
    confirm_score += _clamp(net_1d / 100_000_000.0 * 1.4, -5.0, 5.0)
    if positive_streak >= 6:
        confirm_score -= 7.0
    confirm_score = _clamp(confirm_score, 0.0, 100.0)

    fund_score = _clamp(accumulation_score * 0.44 + early_layout_score * 0.36 + confirm_score * 0.20, 0.0, 100.0)

    trend_score = 50.0
    if close is not None and ma5 is not None:
        trend_score += 10.0 if close >= ma5 else -7.0
    if ma5 is not None and ma10 is not None:
        trend_score += 9.0 if ma5 >= ma10 else -7.0
    if close is not None and ma20 is not None:
        trend_score += 8.0 if close >= ma20 else -6.0
    if position_20 is not None:
        if 0.30 <= position_20 <= 0.82:
            trend_score += 9.0
        elif position_20 > 0.92:
            trend_score -= 9.0
        elif position_20 < 0.15:
            trend_score -= 7.0
    trend_score += _clamp(change_pct * 150.0, -10.0, 8.0)
    if 1.15 <= amount_ratio <= 2.4:
        trend_score += min(9.0, (amount_ratio - 1.0) * 5.0)
    trend_score = _clamp(trend_score, 0.0, 100.0)

    risk_score = 74.0
    if abs(change_pct) >= 0.085:
        risk_score -= 12.0
    elif change_pct >= 0.055:
        risk_score -= 8.0
    if turnover_rate >= 25:
        risk_score -= 18.0
    elif turnover_rate >= 15:
        risk_score -= 9.0
    if limit_ups >= 2:
        risk_score -= 10.0
    if position_20 is not None and position_20 > 0.88:
        risk_score -= 12.0 if net_3d > 0 else 18.0
    if positive_streak >= 6:
        risk_score -= 8.0
    risk_score -= inflow_age_risk * 14.0
    if net_5d < 0:
        risk_score -= 6.0
    risk_score = _clamp(risk_score, 0.0, 100.0)

    score = _clamp(accumulation_score * 0.30 + fund_score * 0.25 + trend_score * 0.25 + risk_score * 0.20, 0.0, 100.0)
    early_age_ok = positive_streak <= 3 or (positive_streak <= 4 and (position_20 is None or position_20 <= 0.68))

    if net_5d < 0 and change_pct > 0.015:
        signal = "distribution_risk"
    elif (
        accumulation_score >= 76
        and net_3d > 0
        and (large_ratio_5d > 0 or net_ratio_3d >= 0.012)
        and quiet_price
        and positive_streak <= 4
        and (position_20 is None or position_20 <= 0.72)
        and (position_60 is None or position_60 <= 0.68)
        and inflow_age_risk < 0.45
        and not is_special_treatment
    ):
        signal = "main_accumulation"
    elif early_layout_score >= 74 and early_age_ok and net_1d > 0 and (turn_positive or acceleration > 0.018) and (position_20 is None or position_20 <= 0.82) and change_pct <= 0.035:
        signal = "early_layout"
    elif max(early_layout_score, accumulation_score) >= 68 and positive_streak <= 4 and net_3d > 0 and (position_20 is None or position_20 <= 0.72) and change_pct <= 0.025:
        signal = "low_absorption"
    elif score >= 78 and net_5d > 0 and positive_days_5d >= 3 and (position_20 is None or position_20 <= 0.82) and positive_streak <= 5:
        signal = "strong_inflow"
    elif score >= 64 and net_3d > 0:
        signal = "inflow_confirm"
    elif net_5d < 0:
        signal = "outflow"
    else:
        signal = "watch"

    reasons: list[str] = []
    if accumulation_score >= 72:
        reasons.append("疑似主力建仓/吸筹")
    if turn_positive:
        reasons.append("资金由弱转强")
    if acceleration > 0:
        reasons.append(f"3日资金加速度{acceleration * 100:.2f}%")
    if net_3d > 0:
        reasons.append(f"3日主力净流入{net_3d / 100_000_000:.2f}亿")
    elif net_5d > 0:
        reasons.append(f"5日主力净流入{net_5d / 100_000_000:.2f}亿")
    else:
        reasons.append(f"5日主力净流出{abs(net_5d) / 100_000_000:.2f}亿")
    if positive_streak >= 5:
        reasons.append(f"连续流入{positive_streak}日，注意追高")
    elif positive_days_5d >= 3:
        reasons.append(f"近5日{positive_days_5d}日净流入")
    if inflow_age_risk >= 0.45:
        reasons.append("资金连续流入已进入确认期")
    if net_ratio_3d > 0 and ret_3d <= 0.035:
        reasons.append(f"资金流入但3日涨幅{ret_3d * 100:.2f}%")
    if range_ratio_10d is not None and range_ratio_10d <= 0.14:
        reasons.append(f"10日振幅收敛{range_ratio_10d * 100:.1f}%")
    if is_special_treatment:
        reasons.append("ST/退市风险标的降级")
    if close is not None and ma10 is not None:
        reasons.append("价格站上MA10" if close >= ma10 else "价格未站上MA10")
    if turnover_rate >= 15:
        reasons.append(f"换手率偏高{turnover_rate:.1f}%")
    if position_20 is not None and position_20 > 0.9:
        reasons.append("接近20日高位")
    extras = {
        "early_layout_score": early_layout_score,
        "confirm_score": confirm_score,
        "price_setup_score": price_setup,
        "accumulation_score": accumulation_score,
        "base_zone_score": base_zone_score,
        "compression_score": compression_score,
        "volume_absorption_score": volume_absorption_score,
        "absorption_efficiency": absorption_efficiency,
        "moneyflow_acceleration_3d": acceleration,
        "moneyflow_turn_positive": turn_positive,
        "inflow_age_risk": inflow_age_risk,
    }
    return score, fund_score, trend_score, risk_score, signal, reasons[:5], extras


def build_daily_fund_monitor(
    repo: Any,
    quote_service: Any = None,
    *,
    window: int = 20,
    limit: int = 100,
    min_score: float = 0.0,
    signal: str | None = None,
) -> dict[str, Any]:
    """Build the confirmed daily moneyflow board."""
    window = max(1, min(60, int(window)))
    limit = max(1, min(6000, int(limit)))
    data_dir = repo.store.data_dir
    flow_metrics, flow_meta = _load_moneyflow_metrics(data_dir, window=window)
    market_metrics, market_meta = _load_market_metrics(repo, quote_service, history_days=45)

    rows: list[dict[str, Any]] = []
    for symbol, flow in flow_metrics.items():
        market = market_metrics.get(symbol, {"symbol": symbol, "board": _board_of(symbol)})
        score, fund_score, trend_score, risk_score, sig, reasons, extras = _daily_scores(flow, market)
        if score < min_score:
            continue
        if signal and signal != "all" and sig != signal:
            continue
        rows.append({
            "symbol": symbol,
            "name": market.get("name"),
            "board": market.get("board") or _board_of(symbol),
            "industry": market.get("industry"),
            "close": _round(market.get("close"), 3),
            "change_pct": _round(market.get("change_pct"), 5),
            "turnover_rate": _round(market.get("turnover_rate"), 4),
            "amount": _round(market.get("amount"), 2),
            "ma5": _round(market.get("ma5"), 3),
            "ma10": _round(market.get("ma10"), 3),
            "ma20": _round(market.get("ma20"), 3),
            "position_20": _round(market.get("position_20"), 4),
            "position_60": _round(market.get("position_60"), 4),
            "ret_3d": _round(market.get("ret_3d"), 6),
            "ret_5d": _round(market.get("ret_5d"), 6),
            "ret_10d": _round(market.get("ret_10d"), 6),
            "range_ratio_10d": _round(market.get("range_ratio_10d"), 6),
            "volatility_10d": _round(market.get("volatility_10d"), 6),
            "amount_ratio_20d": _round(market.get("amount_ratio_20d"), 4),
            "amount_ratio_5d_20d": _round(market.get("amount_ratio_5d_20d"), 4),
            "last_flow_date": _date_to_iso(flow.get("last_flow_date")),
            "flow_days": flow.get("flow_days"),
            "net_1d": _round(flow.get("net_1d"), 2),
            "net_3d": _round(flow.get("net_3d"), 2),
            "net_5d": _round(flow.get("net_5d"), 2),
            "net_10d": _round(flow.get("net_10d"), 2),
            "net_window": _round(flow.get("net_window"), 2),
            "prev_net_3d": _round(flow.get("prev_net_3d"), 2),
            "prev_net_5d": _round(flow.get("prev_net_5d"), 2),
            "large_net_5d": _round(flow.get("large_net_5d"), 2),
            "extra_large_net_5d": _round(flow.get("extra_large_net_5d"), 2),
            "net_ratio_1d": _round(flow.get("net_ratio_1d"), 6),
            "net_ratio_3d": _round(flow.get("net_ratio_3d"), 6),
            "net_ratio_5d": _round(flow.get("net_ratio_5d"), 6),
            "prev_net_ratio_3d": _round(flow.get("prev_net_ratio_3d"), 6),
            "large_ratio_5d": _round(flow.get("large_ratio_5d"), 6),
            "moneyflow_acceleration_3d": _round(extras.get("moneyflow_acceleration_3d"), 6),
            "moneyflow_turn_positive": extras.get("moneyflow_turn_positive"),
            "early_layout_score": round(_float(extras.get("early_layout_score")), 1),
            "confirm_score": round(_float(extras.get("confirm_score")), 1),
            "price_setup_score": round(_float(extras.get("price_setup_score")), 1),
            "accumulation_score": round(_float(extras.get("accumulation_score")), 1),
            "base_zone_score": round(_float(extras.get("base_zone_score")), 1),
            "compression_score": round(_float(extras.get("compression_score")), 1),
            "volume_absorption_score": round(_float(extras.get("volume_absorption_score")), 1),
            "absorption_efficiency": _round(extras.get("absorption_efficiency"), 4),
            "positive_days_5d": flow.get("positive_days_5d"),
            "positive_days_10d": flow.get("positive_days_10d"),
            "positive_streak": flow.get("positive_streak"),
            "inflow_age_risk": _round(extras.get("inflow_age_risk"), 4),
            "score": round(score, 1),
            "fund_score": round(fund_score, 1),
            "trend_score": round(trend_score, 1),
            "risk_score": round(risk_score, 1),
            "signal": sig,
            "signal_label": DAILY_SIGNAL_LABELS.get(sig, sig),
            "reasons": reasons,
        })

    rows.sort(
        key=lambda r: (
            r.get("signal") == "main_accumulation",
            r.get("signal") in {"early_layout", "low_absorption"},
            _float(r.get("accumulation_score")),
            _float(r.get("early_layout_score")),
            _float(r.get("score")),
            _float(r.get("moneyflow_acceleration_3d")),
        ),
        reverse=True,
    )
    rows = rows[:limit]
    counts = Counter(row["signal"] for row in rows)
    payload = {
        "asof": time.time(),
        "mode": "daily_confirmed",
        "source_label": "Tushare moneyflow（日级确认）",
        "data_note": "日级榜使用本地已落库的 Tushare moneyflow 主力资金流；优先识别低位窄幅整理、温和放量、资金由弱转强且价格尚未拉升的主力建仓/吸筹阶段，连续流入只作为确认项。",
        "flow_meta": flow_meta,
        "market_meta": market_meta,
        "weights": {"accumulation": 0.30, "early_fund_setup": 0.25, "trend": 0.25, "risk": 0.20},
        "counts": dict(counts),
        "rows": rows,
    }
    return _jsonable(payload)


def _load_minute_features(repo: Any, *, target_date: date | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    data_dir = repo.store.data_dir
    root = data_dir / "kline_minute"
    dates = _partition_dates(root, end=target_date)
    selected = dates[-1:] if dates else []
    part_mtime = None
    if selected:
        part = root / f"date={selected[0].isoformat()}" / "part.parquet"
        try:
            part_mtime = part.stat().st_mtime_ns
        except OSError:
            part_mtime = None
    cache_key = (
        "minute",
        str(root),
        selected[0].isoformat() if selected else None,
        part_mtime,
    )
    cached = _cache_get(_MINUTE_CACHE, cache_key, _MINUTE_TTL)
    if cached is not None:
        return cached
    meta: dict[str, Any] = {
        "source": "local_minute_bar",
        "date": selected[0].isoformat() if selected else None,
        "latest_minute": None,
        "rows": 0,
        "symbols": 0,
        "skipped_partitions": 0,
        "note": None,
    }
    if not selected:
        meta["note"] = "本地分钟K为空，盘中资金强度将退化为行情成交额估算。"
        return {}, meta
    df, skipped = _read_partitions(root, selected)
    meta["skipped_partitions"] = skipped
    if df.is_empty():
        meta["note"] = "分钟K分区存在，但没有可读取记录。"
        return {}, meta
    df = _normalize_datetime_column(_normalize_symbol_column(df))
    df = _cast_numeric(df, ["open", "high", "low", "close", "volume", "amount"])
    if "symbol" not in df.columns or "datetime" not in df.columns or "close" not in df.columns:
        meta["note"] = "分钟K缺少 symbol/datetime/close 字段。"
        return {}, meta
    if "amount" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("amount"))

    df = (
        df.drop_nulls(["datetime", "symbol", "close"])
        .sort(["symbol", "datetime"])
        .with_columns((pl.col("close") - pl.col("close").shift(1).over("symbol")).alias("_delta"))
        .with_columns(
            (
                pl.col("amount").fill_null(0.0)
                * pl.when(pl.col("_delta") > 0)
                .then(1.0)
                .when(pl.col("_delta") < 0)
                .then(-1.0)
                .otherwise(0.0)
            ).alias("_signed_amount")
        )
    )
    if df.is_empty():
        meta["note"] = "分钟K没有有效分钟记录。"
        return {}, meta

    latest_dt = df["datetime"].max()
    if not isinstance(latest_dt, datetime):
        meta["note"] = "分钟K最新时间无法识别。"
        return {}, meta

    base = df.group_by("symbol").agg(
        pl.col("datetime").max().alias("latest_minute"),
        pl.col("open").first().alias("minute_open"),
        pl.col("high").max().alias("minute_high"),
        pl.col("low").min().alias("minute_low"),
        pl.col("close").last().alias("minute_close"),
        pl.sum("amount").alias("minute_amount"),
        pl.sum("_signed_amount").alias("est_net_day"),
        ((pl.col("close").last() / pl.col("close").first()) - 1.0).alias("ret_day_minutes"),
    )
    for minutes in (5, 15, 30):
        cutoff = latest_dt - timedelta(minutes=minutes)
        sub = df.filter(pl.col("datetime") >= cutoff)
        if sub.is_empty():
            continue
        win = sub.group_by("symbol").agg(
            pl.sum("amount").alias(f"amount_{minutes}m"),
            pl.sum("_signed_amount").alias(f"est_net_{minutes}m"),
            ((pl.col("close").last() / pl.col("close").first()) - 1.0).alias(f"ret_{minutes}m"),
        )
        base = base.join(win, on="symbol", how="left")

    features: dict[str, dict[str, Any]] = {}
    for row in base.to_dicts():
        symbol = _symbol_key(row.get("symbol"))
        if not symbol:
            continue
        row["latest_minute"] = _date_to_iso(row.get("latest_minute"))
        features[symbol] = row

    meta["latest_minute"] = latest_dt.isoformat(sep=" ")
    meta["rows"] = int(df.height)
    meta["symbols"] = len(features)
    _cache_put(_MINUTE_CACHE, cache_key, features, meta)
    return features, meta


def _intraday_scores(
    market: dict[str, Any],
    minute: dict[str, Any],
    flow: dict[str, Any] | None,
    *,
    quote_is_fresh: bool,
) -> tuple[float, float, float, float, str, list[str]]:
    close = _float_or_none(market.get("close"))
    high = _float_or_none(market.get("high"))
    open_price = _float_or_none(market.get("open"))
    change_pct = _float_or_none(market.get("change_pct")) or 0.0
    turnover_rate = _float_or_none(market.get("turnover_rate")) or 0.0
    amount = _float_or_none(market.get("amount")) or 0.0
    avg_amount20 = _float_or_none(market.get("avg_amount20"))
    daily_confirm = _float(flow.get("net_ratio_5d")) if flow else 0.0
    daily_net5 = _float(flow.get("net_5d")) if flow else 0.0

    elapsed_ratio = _clamp(trading_minutes_elapsed() / 240.0, 0.08, 1.0)
    projected_amount = amount / elapsed_ratio if quote_is_fresh else amount
    amount_ratio = _safe_div(projected_amount, avg_amount20, 1.0) if avg_amount20 else 1.0

    amount_15m = _float_or_none(minute.get("amount_15m"))
    net_15m = _float_or_none(minute.get("est_net_15m"))
    amount_5m = _float_or_none(minute.get("amount_5m"))
    net_5m = _float_or_none(minute.get("est_net_5m"))
    strength_15m = _safe_div(net_15m, amount_15m, 0.0)
    strength_5m = _safe_div(net_5m, amount_5m, 0.0)
    if amount_15m is None and amount > 0:
        # Fallback: signed amount approximation from quote direction.
        strength_15m = _clamp(change_pct / 0.06, -1.0, 1.0) * 0.25
        net_15m = amount * strength_15m
    ret_5m = _float_or_none(minute.get("ret_5m")) or 0.0
    ret_15m = _float_or_none(minute.get("ret_15m")) or 0.0
    retreat_from_high = 0.0
    if close is not None and high is not None and high > 0:
        retreat_from_high = close / high - 1.0
    open_strength = 0.0
    if close is not None and open_price is not None and open_price > 0:
        open_strength = close / open_price - 1.0

    fund_score = 50.0
    fund_score += _clamp(strength_15m * 85.0, -28.0, 28.0)
    fund_score += _clamp(strength_5m * 30.0, -8.0, 8.0)
    fund_score += _clamp((amount_ratio - 1.0) * 12.0, -10.0, 24.0)
    fund_score += _clamp(daily_confirm * 220.0, -12.0, 12.0)
    fund_score = _clamp(fund_score, 0.0, 100.0)

    price_score = 50.0
    price_score += _clamp(change_pct * 360.0, -18.0, 22.0)
    price_score += _clamp(ret_15m * 260.0, -12.0, 12.0)
    price_score += _clamp(open_strength * 180.0, -8.0, 8.0)
    if close is not None and market.get("ma10") is not None:
        price_score += 7.0 if close >= _float(market.get("ma10")) else -6.0
    price_score = _clamp(price_score, 0.0, 100.0)

    risk_score = 74.0
    if turnover_rate >= 25:
        risk_score -= 16.0
    elif turnover_rate >= 15:
        risk_score -= 8.0
    if retreat_from_high <= -0.035:
        risk_score -= 16.0
    if change_pct >= 0.085:
        risk_score -= 10.0
    if daily_net5 < 0 and change_pct > 0:
        risk_score -= 8.0
    risk_score = _clamp(risk_score, 0.0, 100.0)

    score = _clamp(fund_score * 0.55 + price_score * 0.30 + risk_score * 0.15, 0.0, 100.0)

    if change_pct > 0.012 and (strength_15m < -0.06 or retreat_from_high <= -0.03 or daily_net5 < 0):
        signal = "distribution_risk"
    elif score >= 78 and change_pct > 0.018 and amount_ratio >= 1.35 and strength_15m > 0.04:
        signal = "active_inflow"
    elif score >= 72 and change_pct > 0.035 and amount_ratio >= 1.7:
        signal = "breakout_flow"
    elif change_pct < 0 and strength_15m < -0.03:
        signal = "outflow"
    else:
        signal = "watch"

    reasons: list[str] = []
    if amount_ratio != 1.0:
        reasons.append(f"预计成交额约{amount_ratio:.1f}倍20日均额")
    if net_15m is not None:
        direction = "净流入" if net_15m >= 0 else "净流出"
        reasons.append(f"15分钟估算{direction}{abs(net_15m) / 100_000_000:.2f}亿")
    if daily_net5 > 0:
        reasons.append("日级资金为正")
    elif daily_net5 < 0:
        reasons.append("日级资金为负")
    if retreat_from_high <= -0.03:
        reasons.append("冲高回落超过3%")
    return score, fund_score, price_score, risk_score, signal, reasons[:4]


def build_intraday_fund_monitor(
    repo: Any,
    quote_service: Any = None,
    minute_refresh: Any = None,
    *,
    limit: int = 100,
    min_score: float = 0.0,
    signal: str | None = None,
) -> dict[str, Any]:
    """Build the realtime-ish fund radar board."""
    limit = max(1, min(1000, int(limit)))
    data_dir = repo.store.data_dir
    market_metrics, market_meta = _load_market_metrics(repo, quote_service, history_days=45)
    flow_metrics, flow_meta = _load_moneyflow_metrics(data_dir, window=10)
    minute_features, minute_meta = _load_minute_features(repo, target_date=cn_today())
    quote_status = market_meta.get("quote_status") or {}
    quote_age = _float_or_none(quote_status.get("quote_age_ms"))
    quote_mode = str(quote_status.get("mode") or "none")
    quote_is_fresh = bool(quote_status.get("enabled")) and (quote_age is None or quote_age <= 60_000)
    backup_route = quote_status.get("backup_route") or {}
    is_symbol_current = getattr(quote_service, "is_symbol_quote_current", None)

    rows: list[dict[str, Any]] = []
    for symbol, market in market_metrics.items():
        close = _float_or_none(market.get("close"))
        if close is None or close <= 0:
            continue
        minute = minute_features.get(symbol, {})
        flow = flow_metrics.get(symbol)
        symbol_is_fresh = quote_is_fresh and (
            is_symbol_current(symbol) if callable(is_symbol_current) else not backup_route.get("degraded")
        )
        score, fund_score, price_score, risk_score, sig, reasons = _intraday_scores(
            market,
            minute,
            flow,
            quote_is_fresh=symbol_is_fresh,
        )
        if score < min_score:
            continue
        if signal and signal != "all" and sig != signal:
            continue
        amount = _float_or_none(market.get("amount")) or 0.0
        change_pct = _float_or_none(market.get("change_pct")) or 0.0
        elapsed_ratio = _clamp(trading_minutes_elapsed() / 240.0, 0.08, 1.0)
        projected_amount = amount / elapsed_ratio if symbol_is_fresh else amount
        avg_amount20 = _float_or_none(market.get("avg_amount20"))
        est_net_15m = _float_or_none(minute.get("est_net_15m"))
        estimate_source = "minute"
        if est_net_15m is None and amount > 0:
            est_net_15m = amount * _clamp(change_pct / 0.06, -1.0, 1.0) * 0.25
            estimate_source = "quote_fallback"
        rows.append({
            "symbol": symbol,
            "name": market.get("name"),
            "board": market.get("board") or _board_of(symbol),
            "industry": market.get("industry"),
            "close": _round(close, 3),
            "change_pct": _round(market.get("change_pct"), 5),
            "turnover_rate": _round(market.get("turnover_rate"), 4),
            "amount": _round(amount, 2),
            "projected_amount": _round(projected_amount, 2),
            "amount_ratio_20d": _round(_safe_div(projected_amount, avg_amount20, 1.0) if avg_amount20 else 1.0, 4),
            "minute_latest": minute.get("latest_minute"),
            "est_net_5m": _round(minute.get("est_net_5m"), 2),
            "est_net_15m": _round(est_net_15m, 2),
            "est_net_30m": _round(minute.get("est_net_30m"), 2),
            "estimate_source": estimate_source,
            "ret_5m": _round(minute.get("ret_5m"), 5),
            "ret_15m": _round(minute.get("ret_15m"), 5),
            "ret_30m": _round(minute.get("ret_30m"), 5),
            "daily_net_5d": _round(flow.get("net_5d"), 2) if flow else None,
            "daily_net_ratio_5d": _round(flow.get("net_ratio_5d"), 6) if flow else None,
            "score": round(score, 1),
            "fund_score": round(fund_score, 1),
            "price_score": round(price_score, 1),
            "risk_score": round(risk_score, 1),
            "signal": sig,
            "signal_label": INTRADAY_SIGNAL_LABELS.get(sig, sig),
            "reasons": reasons,
        })

    rows.sort(key=lambda r: (_float(r.get("score")), _float(r.get("amount_ratio_20d"))), reverse=True)
    rows = rows[:limit]
    counts = Counter(row["signal"] for row in rows)
    refresh_status = {}
    if minute_refresh is not None and hasattr(minute_refresh, "status"):
        try:
            refresh_status = minute_refresh.status() or {}
        except Exception:
            refresh_status = {}
    payload = {
        "asof": time.time(),
        "mode": "intraday_estimated",
        "source_label": "Fuyao/实时行情 + 本地分钟K（盘中估算）",
        "data_note": "盘中资金雷达按实时成交额、分钟K价格方向和日级 moneyflow 做近似估算；不是 Level2 逐笔主力资金，适合盯盘预警，不应单独作为买入依据。",
        "quote_status": quote_status,
        "quote_mode": quote_mode,
        "quote_is_fresh": quote_is_fresh,
        "market_meta": market_meta,
        "minute_meta": minute_meta,
        "minute_refresh": refresh_status,
        "flow_meta": flow_meta,
        "weights": {"intraday_fund": 0.55, "price_action": 0.30, "risk": 0.15},
        "counts": dict(counts),
        "rows": rows,
    }
    return _jsonable(payload)
