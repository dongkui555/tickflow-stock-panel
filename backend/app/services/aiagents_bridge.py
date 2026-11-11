"""AI Agents 增强能力适配层。

TickFlow 负责底层行情、缓存、选股、回测和监控；本模块把多源行情、
资金流、海外新闻、持仓优化和模拟盘能力以 API 服务的形式接入，
不依赖外部 Streamlit 项目。
"""
from __future__ import annotations

import logging
import hashlib
import json
import math
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd
import polars as pl

from app.config import settings
from app.services.aiagents_sources import (
    build_unified_news_context,
    data_root as aiagents_data_root,
    data_source_manager,
    fund_flow_fetcher,
    overseas_market_data,
)

logger = logging.getLogger(__name__)

_LEGACY_LOCK = RLock()
_LEGACY_MANAGER = data_source_manager
_LEGACY_ERROR = getattr(data_source_manager, "init_error", "")


def data_root() -> Path:
    return aiagents_data_root()


def legacy_root() -> Path:
    return data_root()


def _ensure_legacy_path() -> Path:
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _legacy_manager():
    return _LEGACY_MANAGER


def _import_legacy(name: str):
    raise RuntimeError("legacy imports are disabled; use local aiagents_sources helpers")


def _compact_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if "." in value:
        value = value.split(".", 1)[0]
    return value.zfill(6) if value.isdigit() else value


def _tickflow_symbol(symbol: str) -> str:
    code = _compact_symbol(symbol)
    if not code.isdigit():
        return code
    from app.services.security_identity import exchange_for_code
    market = exchange_for_code(code, "stock")
    return f"{code}.{market}" if market else code


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _runtime_repo() -> Any | None:
    try:
        from app.services import aiagents_sources

        return aiagents_sources._repo()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return None


def _parse_cache_date_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8)
    return (
        pl.when(text.str.contains(r"^\d{8}$"))
        .then(text.str.strptime(pl.Date, "%Y%m%d", strict=False))
        .otherwise(text.str.strptime(pl.Date, strict=False))
    )


def _append_partitioned_cache(
    root: Path,
    df: pl.DataFrame,
    *,
    date_col: str = "date",
    unique_subset: list[str] | None = None,
    sort_cols: list[str] | None = None,
) -> int:
    if df.is_empty() or date_col not in df.columns:
        return 0
    root.mkdir(parents=True, exist_ok=True)
    if df.schema.get(date_col) == pl.Date:
        df = df.with_columns(pl.col(date_col).alias("_cache_date"))
    else:
        df = df.with_columns(_parse_cache_date_expr(date_col).alias("_cache_date"))
    df = df.filter(pl.col("_cache_date").is_not_null())
    if df.is_empty():
        return 0

    written = 0
    for date_df in df.partition_by("_cache_date"):
        dt = date_df["_cache_date"][0]
        ds = dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
        out = root / f"date={ds}" / "part.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        date_df = date_df.drop("_cache_date")
        if out.exists():
            try:
                existing = pl.read_parquet(out)
                date_df = pl.concat([existing, date_df], how="diagonal_relaxed")
            except Exception as exc:  # noqa: BLE001
                logger.warning("AI 本地缓存分区读取失败，将重写 %s: %s", out, exc)
        subset = [col for col in (unique_subset or []) if col in date_df.columns]
        if subset:
            date_df = date_df.unique(subset=subset, keep="last", maintain_order=True)
        order = [col for col in (sort_cols or []) if col in date_df.columns]
        if order:
            date_df = date_df.sort(order)
        tmp = out.with_name(out.name + ".tmp")
        date_df.write_parquet(tmp)
        tmp.replace(out)
        written += date_df.height
    return written


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(value) if isinstance(value, int) else number
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _frame_records(frame: pd.DataFrame | None, limit: int = 200) -> list[dict]:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    rows = frame.tail(limit).copy()
    return [_json_value(row) for row in rows.to_dict(orient="records")]


def _standard_daily(frame: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    df = frame.copy()
    aliases = {
        "日期": "date",
        "交易日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    if "date" not in df.columns:
        for candidate in ("trade_date", "datetime"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = _tickflow_symbol(symbol)
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def _technical_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"data_success": False, "error": "未获取到有效日线"}
    close = df["close"].astype(float)
    latest = float(close.iloc[-1])
    previous = float(close.iloc[-2]) if len(close) > 1 else latest
    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma10 = float(close.rolling(10).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi14 = float((100 - (100 / (1 + rs))).iloc[-1]) if len(close) >= 15 else None
    recent = close.tail(min(60, len(close)))
    high_20 = float(close.tail(min(20, len(close))).max())
    low_20 = float(close.tail(min(20, len(close))).min())
    high_60 = float(recent.max())
    low_60 = float(recent.min())
    volume_ratio = None
    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        avg_volume = volume.rolling(20).mean().iloc[-1]
        if _safe_float(avg_volume, 0):
            volume_ratio = float(volume.iloc[-1] / avg_volume)
    trend = "up" if ma20 and latest > ma20 and (ma60 is None or ma20 >= ma60) else "down" if ma20 and latest < ma20 else "range"
    support = min(low_20, low_60)
    resistance = max(high_20, high_60)
    return {
        "data_success": True,
        "latest_close": latest,
        "previous_close": previous,
        "change_pct": (latest / previous - 1) * 100 if previous else None,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "rsi14": rsi14,
        "volume_ratio": volume_ratio,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "high_20": high_20,
        "low_20": low_20,
        "high_60": high_60,
        "low_60": low_60,
        "bars": len(df),
        "latest_date": str(df["date"].iloc[-1].date()),
    }


def _get_quote(symbol: str) -> dict:
    local_quote = _quote_from_local(symbol)
    if local_quote.get("data_success"):
        return local_quote
    manager = _legacy_manager()
    try:
        quote = manager.get_realtime_quotes(_compact_symbol(symbol)) or {}
        quote = dict(quote)
        quote["symbol"] = _tickflow_symbol(symbol)
        quote["data_success"] = bool(quote.get("price") or quote.get("last_price"))
        quote["fetched_at"] = datetime.now().isoformat(timespec="seconds")
        return _json_value(quote)
    except Exception as exc:  # noqa: BLE001
        return {"symbol": _tickflow_symbol(symbol), "data_success": False, "error": str(exc)}


def _parse_query_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _partition_dates(root: Path) -> list[date]:
    dates: list[date] = []
    if not root.exists():
        return dates
    for item in root.iterdir():
        if not item.is_dir() or not item.name.startswith("date="):
            continue
        try:
            dates.append(date.fromisoformat(item.name.split("=", 1)[1]))
        except ValueError:
            continue
    return sorted(set(dates))


def _daily_from_local(symbol: str, start: str | None, end: str | None, days: int) -> pd.DataFrame:
    code = _compact_symbol(symbol)
    start_date = _parse_query_date(start)
    end_date = _parse_query_date(end)
    for root_name in ("kline_daily_enriched", "kline_daily", "ai_cache/daily"):
        root = settings.data_dir / root_name
        all_dates = _partition_dates(root)
        if not all_dates:
            continue
        bounded_dates = [
            day for day in all_dates
            if (start_date is None or day >= start_date) and (end_date is None or day <= end_date)
        ]
        selected_dates = bounded_dates[-max(30, min(600, int(days))):]
        if not selected_dates:
            continue
        frames: list[pl.DataFrame] = []
        for day in selected_dates:
            part = root / f"date={day.isoformat()}" / "part.parquet"
            if not part.exists():
                continue
            try:
                frames.append(pl.read_parquet(str(part)))
            except Exception as exc:  # noqa: BLE001
                logger.debug("本地日线分区读取失败(%s): %s", part, exc)
        if not frames:
            continue
        df = pl.concat(frames, how="diagonal_relaxed")
        if "symbol" not in df.columns or "date" not in df.columns or "close" not in df.columns:
            continue
        if df.schema.get("date") != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias("date"))
        df = (
            df.with_columns(
                pl.col("symbol")
                .cast(pl.Utf8)
                .str.extract(r"(\d{6})", 1)
                .fill_null(pl.col("symbol").cast(pl.Utf8))
                .alias("symbol")
            )
            .filter(pl.col("symbol") == code)
            .sort("date")
            .tail(max(1, int(days)))
        )
        if df.is_empty():
            continue
        keep_cols = [col for col in ("symbol", "date", "open", "high", "low", "close", "volume", "amount") if col in df.columns]
        out = pd.DataFrame(df.select(keep_cols).to_dicts())
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["symbol"] = _tickflow_symbol(symbol)
        return out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return pd.DataFrame()


def _persist_daily_cache(frame: pd.DataFrame, symbol: str, source: str) -> None:
    df = _standard_daily(frame, symbol)
    if df.empty:
        return
    try:
        df = df.copy()
        df["symbol"] = _compact_symbol(symbol)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["data_source"] = source
        df["fetched_at"] = datetime.now().isoformat(timespec="seconds")
        keep = [
            col for col in (
                "symbol", "date", "open", "high", "low", "close",
                "volume", "amount", "data_source", "fetched_at",
            )
            if col in df.columns
        ]
        _append_partitioned_cache(
            settings.data_dir / "ai_cache" / "daily",
            pl.from_pandas(df[keep], include_index=False),
            date_col="date",
            unique_subset=["symbol", "date"],
            sort_cols=["symbol", "date"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("AI 日线缓存写入失败(%s): %s", symbol, exc)


def _quote_from_local(symbol: str) -> dict:
    code = _compact_symbol(symbol)
    for root_name in ("kline_daily_enriched", "kline_daily", "ai_cache/daily"):
        root = settings.data_dir / root_name
        all_dates = _partition_dates(root)
        for day in reversed(all_dates[-10:]):
            part = root / f"date={day.isoformat()}" / "part.parquet"
            if not part.exists():
                continue
            try:
                df = pl.read_parquet(str(part))
            except Exception as exc:  # noqa: BLE001
                logger.debug("本地 quote 分区读取失败(%s): %s", part, exc)
                continue
            if "symbol" not in df.columns or "close" not in df.columns:
                continue
            df = (
                df.with_columns(
                    pl.col("symbol")
                    .cast(pl.Utf8)
                    .str.extract(r"(\d{6})", 1)
                    .fill_null(pl.col("symbol").cast(pl.Utf8))
                    .alias("symbol")
                )
                .filter(pl.col("symbol") == code)
            )
            if df.is_empty():
                continue
            row = df.tail(1).to_dicts()[0]
            close = _safe_float(row.get("close"))
            previous_close = _safe_float(row.get("pre_close") or row.get("previous_close") or row.get("raw_close"))
            change_pct = _safe_float(row.get("change_pct"))
            if change_pct is None and close is not None and previous_close:
                change_pct = (close / previous_close - 1) * 100
            return _json_value({
                "symbol": _tickflow_symbol(symbol),
                "price": close,
                "last_price": close,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "volume": row.get("volume"),
                "amount": row.get("amount"),
                "change_pct": change_pct,
                "trade_date": row.get("date") or day.isoformat(),
                "source": f"local_{root_name}",
                "data_success": close is not None,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
            })
    return {"symbol": _tickflow_symbol(symbol), "data_success": False, "error": "本地最新行情不可用"}


def get_daily(symbol: str, start: str | None = None, end: str | None = None, days: int = 180) -> pd.DataFrame:
    local = _daily_from_local(symbol, start, end, days)
    if not local.empty:
        return local
    manager = _legacy_manager()
    end_value = end or datetime.now().strftime("%Y%m%d")
    start_value = start or (datetime.now() - timedelta(days=max(30, days * 2))).strftime("%Y%m%d")
    try:
        frame = _standard_daily(
            manager.get_stock_hist_data(_compact_symbol(symbol), start_value, end_value, adjust="qfq"),
            symbol,
        )
        if not frame.empty:
            _persist_daily_cache(frame, symbol, "aiagents_remote_daily")
        return frame
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI Agents 日线获取失败(%s): %s", symbol, exc)
        return pd.DataFrame()


def _minute_period(period: str | int | None) -> int:
    text = "".join(ch for ch in str(period or "1") if ch.isdigit())
    try:
        return max(1, int(text or "1"))
    except ValueError:
        return 1


def _standard_minute(frame: Any, symbol: str, source: str = "") -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pl.DataFrame):
        if frame.is_empty():
            return pd.DataFrame()
        df = frame.to_pandas()
    elif isinstance(frame, pd.DataFrame):
        if frame.empty:
            return pd.DataFrame()
        df = frame.copy()
    else:
        try:
            df = pd.DataFrame(frame)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
        if df.empty:
            return pd.DataFrame()
    rename = {
        "datetime": "bar_time",
        "trade_time": "bar_time",
        "trade_date": "bar_time",
        "date": "bar_time",
        "时间": "bar_time",
        "日期": "bar_time",
        "day": "bar_time",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_rate",
    }
    df = df.rename(columns={key: value for key, value in rename.items() if key in df.columns})
    if "bar_time" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df["bar_time"] = pd.to_datetime(df["bar_time"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover_rate"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["bar_time", "close"]).sort_values("bar_time")
    if df.empty:
        return pd.DataFrame()
    df["symbol"] = _tickflow_symbol(symbol)
    if source:
        df["data_source"] = source
    elif "data_source" not in df.columns:
        df["data_source"] = "unknown"
    keep = [
        col for col in (
            "symbol", "bar_time", "open", "high", "low", "close",
            "volume", "amount", "turnover_rate", "data_source", "fetched_at",
        )
        if col in df.columns
    ]
    return df[keep].reset_index(drop=True)


def _resample_minute(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    if frame.empty or period <= 1 or "bar_time" not in frame.columns:
        return frame
    from app.market_time import cn_now
    from app.services.minute_timeframes import aggregate_minutes
    df = frame.copy()
    df["datetime"] = pd.to_datetime(df["bar_time"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    if df.empty:
        return pd.DataFrame()
    if "symbol" not in df:
        df["symbol"] = ""
    bars = aggregate_minutes(pl.from_pandas(df), period, cn_now())
    if bars.is_empty():
        return pd.DataFrame()
    out = bars.filter(pl.col("complete")).rename({"datetime": "bar_time"}).to_pandas()
    if out.empty:
        return pd.DataFrame()
    out["symbol"] = frame["symbol"].iloc[-1] if "symbol" in frame.columns else ""
    out["data_source"] = frame["data_source"].iloc[-1] if "data_source" in frame.columns else "local_kline_minute"
    cols = [
        col for col in (
            "symbol", "bar_time", "open", "high", "low", "close",
            "volume", "amount", "data_source",
        )
        if col in out.columns
    ]
    return out[cols].sort_values("bar_time").reset_index(drop=True)


def _minute_from_local(symbol: str, start: str | None, end: str | None, period: str | int) -> pd.DataFrame:
    code = _compact_symbol(symbol)
    start_date = _parse_query_date(start)
    end_date = _parse_query_date(end)
    period_int = _minute_period(period)
    roots = [
        (settings.data_dir / "kline_minute", "local_kline_minute", True),
        (settings.data_dir / "ai_cache" / "minute" / f"period={period_int}", "local_ai_minute_cache", False),
    ]
    for root, source, can_resample in roots:
        all_dates = _partition_dates(root)
        if not all_dates:
            continue
        bounded_dates = [
            day for day in all_dates
            if (start_date is None or day >= start_date) and (end_date is None or day <= end_date)
        ]
        selected_dates = bounded_dates[-60:] if (start_date or end_date) else bounded_dates[-10:]
        if not selected_dates:
            continue
        frames: list[pl.DataFrame] = []
        for day in selected_dates:
            part = root / f"date={day.isoformat()}" / "part.parquet"
            if not part.exists():
                continue
            try:
                frames.append(pl.read_parquet(str(part)))
            except Exception as exc:  # noqa: BLE001
                logger.debug("本地分钟K分区读取失败(%s): %s", part, exc)
        if not frames:
            continue
        df = pl.concat(frames, how="diagonal_relaxed")
        if "symbol" not in df.columns or "datetime" not in df.columns or "close" not in df.columns:
            continue
        if not isinstance(df.schema.get("datetime"), pl.Datetime):
            df = df.with_columns(pl.col("datetime").cast(pl.Datetime("us"), strict=False).alias("datetime"))
        df = (
            df.with_columns(
                pl.col("symbol")
                .cast(pl.Utf8)
                .str.extract(r"(\d{6})", 1)
                .fill_null(pl.col("symbol").cast(pl.Utf8))
                .alias("_code")
            )
            .filter((pl.col("_code") == code) & pl.col("datetime").is_not_null())
            .sort("datetime")
        )
        if df.is_empty():
            continue
        keep_cols = [
            col for col in ("symbol", "datetime", "open", "high", "low", "close", "volume", "amount", "turnover_rate")
            if col in df.columns
        ]
        out = _standard_minute(
            pd.DataFrame(df.select(keep_cols).to_dicts()).rename(columns={"datetime": "bar_time"}),
            symbol,
            source,
        )
        if can_resample and period_int > 1:
            out = _resample_minute(out, period_int)
        if not out.empty:
            return out
    return pd.DataFrame()


def _persist_minute_cache(frame: pd.DataFrame, symbol: str, period: str | int, source: str) -> None:
    df = _standard_minute(frame, symbol, source)
    if df.empty:
        return
    try:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["bar_time"], errors="coerce")
        df["date"] = df["datetime"].dt.date
        df["symbol"] = _compact_symbol(symbol)
        df["data_source"] = source
        df["fetched_at"] = datetime.now().isoformat(timespec="seconds")
        keep = [
            col for col in (
                "symbol", "date", "datetime", "open", "high", "low", "close",
                "volume", "amount", "turnover_rate", "data_source", "fetched_at",
            )
            if col in df.columns
        ]
        _append_partitioned_cache(
            settings.data_dir / "ai_cache" / "minute" / f"period={_minute_period(period)}",
            pl.from_pandas(df[keep], include_index=False),
            date_col="date",
            unique_subset=["symbol", "datetime"],
            sort_cols=["symbol", "datetime"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("AI 分钟K缓存写入失败(%s): %s", symbol, exc)


def get_minute(symbol: str, start: str | None = None, end: str | None = None, period: str = "5") -> pd.DataFrame:
    local = _minute_from_local(symbol, start, end, period)
    if not local.empty:
        return local
    manager = _legacy_manager()
    try:
        frame = _standard_minute(
            manager.get_stock_minute_data(_compact_symbol(symbol), start, end, period=period, adjust=""),
            symbol,
            "aiagents_remote_minute",
        )
        if not frame.empty:
            _persist_minute_cache(frame, symbol, period, "aiagents_remote_minute")
        return frame
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI Agents 分钟线获取失败(%s): %s", symbol, exc)
        return pd.DataFrame()


def data_health() -> dict:
    root = legacy_root()
    manager = _legacy_manager()
    return {
        "enabled": bool(os.getenv("AIAGENTS_ENABLE", "1").lower() not in {"0", "false", "no"}),
        "data_root": str(root),
        "data_root_exists": root.exists(),
        "legacy_root": str(root),
        "legacy_root_exists": root.exists(),
        "provider_ready": manager is not None,
        "provider_error": _LEGACY_ERROR,
        "tushare_available": bool(getattr(manager, "tushare_available", False)) if manager else False,
        "cache_dir": str(settings.data_dir),
        "data_policy": {
            "mode": "local_first",
            "external_fallback": True,
            "archive_external_before_reuse": True,
            "daily_cache": str(settings.data_dir / "ai_cache" / "daily"),
            "minute_cache": str(settings.data_dir / "ai_cache" / "minute"),
            "news_archive": str(settings.data_dir / "news"),
        },
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "capabilities": {
            "realtime_quote": True,
            "daily": True,
            "minute": True,
            "fund_flow": True,
            "overseas_news": True,
            "paper_trading": True,
            "holdings_optimizer": True,
        },
    }


def _moneyflow_partition_dates(root: Path) -> list[date]:
    dates: list[date] = []
    if not root.exists():
        return dates
    for item in root.iterdir():
        if not item.is_dir() or not item.name.startswith("date="):
            continue
        try:
            dates.append(date.fromisoformat(item.name.split("=", 1)[1]))
        except ValueError:
            continue
    return sorted(set(dates))


def _read_moneyflow_partitions(root: Path, days: int) -> tuple[pl.DataFrame, int, str | None, str | None]:
    all_dates = _moneyflow_partition_dates(root)
    if not all_dates:
        return pl.DataFrame(), 0, None, None
    selected_dates = all_dates[-max(5, min(180, int(days))):]
    frames: list[pl.DataFrame] = []
    skipped = 0
    for day in selected_dates:
        part = root / f"date={day.isoformat()}" / "part.parquet"
        if not part.exists():
            skipped += 1
            continue
        try:
            frames.append(pl.read_parquet(str(part)))
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.debug("本地 moneyflow 分区读取失败(%s): %s", part, exc)
    if not frames:
        return pl.DataFrame(), skipped, selected_dates[0].isoformat(), selected_dates[-1].isoformat()
    return (
        pl.concat(frames, how="diagonal_relaxed"),
        skipped,
        selected_dates[0].isoformat(),
        selected_dates[-1].isoformat(),
    )


def _local_moneyflow_amount(row: dict, *columns: str) -> float:
    return sum(_safe_float(row.get(col), 0.0) or 0.0 for col in columns)


def _local_moneyflow_ratio(value: float, denominator: float) -> float:
    if abs(denominator) <= 1e-9:
        return 0.0
    return value / denominator * 100


def _format_local_moneyflow_row(row: dict) -> dict:
    main_net = _safe_float(row.get("net_mf_amount"), None)
    extra_large_net = _local_moneyflow_amount(row, "buy_elg_amount") - _local_moneyflow_amount(row, "sell_elg_amount")
    large_net = _local_moneyflow_amount(row, "buy_lg_amount") - _local_moneyflow_amount(row, "sell_lg_amount")
    mid_net = _local_moneyflow_amount(row, "buy_md_amount") - _local_moneyflow_amount(row, "sell_md_amount")
    small_net = _local_moneyflow_amount(row, "buy_sm_amount") - _local_moneyflow_amount(row, "sell_sm_amount")
    if main_net is None:
        main_net = extra_large_net + large_net
    gross_buy = _local_moneyflow_amount(
        row,
        "buy_sm_amount",
        "buy_md_amount",
        "buy_lg_amount",
        "buy_elg_amount",
    )
    gross_sell = _local_moneyflow_amount(
        row,
        "sell_sm_amount",
        "sell_md_amount",
        "sell_lg_amount",
        "sell_elg_amount",
    )
    turnover_proxy = max(abs(gross_buy), abs(gross_sell), abs(main_net), 1.0)
    trade_date = row.get("date") or row.get("trade_date")
    if isinstance(trade_date, pd.Timestamp):
        trade_date = trade_date.date().isoformat()
    elif isinstance(trade_date, datetime):
        trade_date = trade_date.date().isoformat()
    elif isinstance(trade_date, date):
        trade_date = trade_date.isoformat()
    return {
        "日期": str(trade_date or ""),
        "主力净流入-净额": main_net,
        "主力净流入-净占比": _local_moneyflow_ratio(main_net, turnover_proxy),
        "超大单净流入-净额": extra_large_net,
        "超大单净流入-净占比": _local_moneyflow_ratio(extra_large_net, turnover_proxy),
        "大单净流入-净额": large_net,
        "大单净流入-净占比": _local_moneyflow_ratio(large_net, turnover_proxy),
        "中单净流入-净额": mid_net,
        "中单净流入-净占比": _local_moneyflow_ratio(mid_net, turnover_proxy),
        "小单净流入-净额": small_net,
        "小单净流入-净占比": _local_moneyflow_ratio(small_net, turnover_proxy),
    }


def _fund_flow_from_local(symbol: str, days: int = 60) -> dict:
    root = settings.data_dir / "moneyflow"
    if not root.exists():
        return {
            "data_success": False,
            "source": "local_moneyflow",
            "days": 0,
            "rows": [],
            "error": "本地 moneyflow 目录不存在，请先同步主力资金流。",
        }
    code = _compact_symbol(symbol)
    try:
        df, skipped, start_text, end_text = _read_moneyflow_partitions(root, days)
        if df.is_empty():
            return {
                "data_success": False,
                "source": "local_moneyflow",
                "days": 0,
                "rows": [],
                "date_start": start_text,
                "date_end": end_text,
                "skipped_partitions": skipped,
                "error": "本地 moneyflow 分区为空或无法读取。",
            }
        if "symbol" not in df.columns:
            return {
                "data_success": False,
                "source": "local_moneyflow",
                "days": 0,
                "rows": [],
                "date_start": start_text,
                "date_end": end_text,
                "skipped_partitions": skipped,
                "error": "本地 moneyflow 缺少 symbol 字段。",
            }
        if "date" not in df.columns and "trade_date" in df.columns:
            df = df.rename({"trade_date": "date"})
        if "date" not in df.columns:
            return {
                "data_success": False,
                "source": "local_moneyflow",
                "days": 0,
                "rows": [],
                "date_start": start_text,
                "date_end": end_text,
                "skipped_partitions": skipped,
                "error": "本地 moneyflow 缺少 date/trade_date 字段。",
            }
        if df.schema.get("date") != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias("date"))
        df = (
            df.with_columns(
                pl.col("symbol")
                .cast(pl.Utf8)
                .str.extract(r"(\d{6})", 1)
                .fill_null(pl.col("symbol").cast(pl.Utf8))
                .alias("symbol")
            )
            .filter(pl.col("symbol") == code)
            .sort("date")
        )
        if df.is_empty():
            return {
                "data_success": False,
                "source": "local_moneyflow",
                "days": 0,
                "rows": [],
                "date_start": start_text,
                "date_end": end_text,
                "skipped_partitions": skipped,
                "error": f"本地 moneyflow 未找到 {code} 的记录。",
            }
        numeric_cols = [
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
        existing_numeric = [col for col in numeric_cols if col in df.columns]
        if existing_numeric:
            df = df.with_columns([
                pl.col(col).cast(pl.Float64, strict=False).fill_null(0.0)
                for col in existing_numeric
            ])
        records = [_format_local_moneyflow_row(row) for row in df.tail(max(1, days)).to_dicts()]
        main_values = [_safe_float(item.get("主力净流入-净额"), 0.0) or 0.0 for item in records]
        latest = records[-1] if records else {}
        return {
            "data_success": bool(records),
            "source": "local_moneyflow_tushare",
            "days": len(records),
            "date_start": records[0].get("日期") if records else start_text,
            "date_end": latest.get("日期") if latest else end_text,
            "query_time": datetime.now().isoformat(timespec="seconds"),
            "latest": _json_value(latest),
            "main_net_total": sum(main_values),
            "main_net_5d": sum(main_values[-5:]),
            "main_net_20d": sum(main_values[-20:]),
            "main_net_60d": sum(main_values[-60:]),
            "positive_days_20": sum(1 for value in main_values[-20:] if value > 0),
            "positive_days_60": sum(1 for value in main_values[-60:] if value > 0),
            "rows": _json_value(records[-20:]),
            "skipped_partitions": skipped,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("本地 moneyflow 读取失败(%s): %s", symbol, exc)
        return {
            "data_success": False,
            "source": "local_moneyflow",
            "days": 0,
            "rows": [],
            "error": f"本地 moneyflow 读取失败：{exc}",
        }


def _fund_flow(symbol: str, days: int = 60) -> dict:
    local_result = _fund_flow_from_local(symbol, days=days)
    if local_result.get("data_success"):
        return local_result
    try:
        fetcher = fund_flow_fetcher
        result = fetcher.get_fund_flow_data(_compact_symbol(symbol), days=days) or {}
        payload = result.get("fund_flow_data") or {}
        rows = payload.get("data") or []
        main_values = [
            _safe_float(item.get("主力净流入-净额"), 0.0) or 0.0 for item in rows
        ]
        latest = rows[-1] if rows else {}
        return {
            "data_success": bool(result.get("data_success") and rows),
            "source": result.get("source") or payload.get("source", ""),
            "days": len(rows),
            "query_time": payload.get("query_time", ""),
            "latest": _json_value(latest),
            "main_net_total": sum(main_values),
            "main_net_5d": sum(main_values[-5:]),
            "main_net_20d": sum(main_values[-20:]),
            "positive_days_20": sum(1 for value in main_values[-20:] if value > 0),
            "rows": _json_value(rows[-20:]),
            "error": result.get("error", ""),
        }
    except Exception as exc:  # noqa: BLE001
        if local_result:
            local_result["remote_error"] = str(exc)
            return local_result
        return {
            "data_success": False,
            "days": 0,
            "main_net_total": 0,
            "main_net_5d": 0,
            "main_net_20d": 0,
            "positive_days_20": 0,
            "rows": [],
            "error": str(exc),
        }


def _annotate_fund_freshness(fund: dict, latest_trade_date: str | None) -> dict:
    """按本地日K最新交易日标记资金流新鲜度。"""
    out = dict(fund or {})
    reference = _parse_query_date(latest_trade_date)
    flow_end = _parse_query_date(out.get("date_end") or (out.get("latest") or {}).get("日期"))
    out["latest_trade_date"] = reference.isoformat() if reference else None
    if not out.get("data_success"):
        out["freshness_status"] = "unavailable"
        out["lag_days"] = None
        return out
    if reference is None or flow_end is None:
        out["freshness_status"] = "unknown"
        out["lag_days"] = None
        return out
    lag_days = max(0, (reference - flow_end).days)
    out["lag_days"] = lag_days
    if lag_days <= 0:
        out["freshness_status"] = "fresh"
        out["freshness_note"] = "资金流已覆盖日K最新交易日"
    elif lag_days <= 1:
        out["freshness_status"] = "lagging_1d"
        out["freshness_note"] = "资金流可用但比日K最新交易日滞后1日，按盘后确认数据降权参考"
    else:
        out["freshness_status"] = "stale"
        out["freshness_note"] = f"资金流比日K最新交易日滞后{lag_days}日，需要先同步增强尾部数据"
    return out


_NEWS_COLUMNS = [
    "id", "date", "published_at", "market", "market_scope", "source",
    "title", "summary", "content", "channels", "symbols", "url",
    "language", "sentiment", "raw",
]


def _news_text(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(_json_value(value), ensure_ascii=False)
    text = str(value).strip()
    return text[:limit] if limit else text


def _news_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return datetime.now()
    if isinstance(parsed, pd.Timestamp):
        if parsed.tzinfo is not None:
            parsed = parsed.tz_convert("Asia/Shanghai").tz_localize(None)
        return parsed.to_pydatetime()
    return datetime.now()


def _news_fingerprint(*parts: Any) -> str:
    text = "|".join(_news_text(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _news_item_from_archive(item: dict[str, Any], default_scope: str) -> dict:
    return {
        "title": _news_text(item.get("title")),
        "date": _news_text(item.get("published_at") or item.get("date")),
        "source": _news_text(item.get("source") or "本地新闻库"),
        "content": _news_text(item.get("summary") or item.get("content")),
        "url": _news_text(item.get("url")),
        "topic": _news_text(item.get("channels") or default_scope),
        "market_scope": _news_text(item.get("market_scope") or default_scope),
        "symbols": _news_text(item.get("symbols")),
        "sentiment": _news_text(item.get("sentiment")),
    }


def _dedupe_news_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for item in items:
        key = _news_text(item.get("url")) or "|".join((
            _news_text(item.get("date")),
            _news_text(item.get("source")),
            _news_text(item.get("title")),
        ))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _news_from_local(symbol: str, name: str = "") -> dict:
    try:
        from app.services.news_sync import read_local_news

        code = _compact_symbol(symbol)
        domestic = read_local_news(settings.data_dir, symbol=code, market="domestic", limit=30)
        overseas = read_local_news(
            settings.data_dir,
            symbol="" if code.isdigit() else symbol,
            market="overseas",
            limit=20,
        )
        items = [
            *(_news_item_from_archive(item, "A股/国内财经") for item in domestic),
            *(_news_item_from_archive(item, "美股/海外财经") for item in overseas),
        ]
        items = _dedupe_news_items(items)[:45]
        dates = [_news_datetime(item.get("date")).date().isoformat() for item in items if item.get("date")]
        return {
            "symbol": code,
            "stock_name": name,
            "data_success": bool(items),
            "source": "local_news_archive",
            "local_first": True,
            "domestic_count": len(domestic),
            "overseas_count": len(overseas),
            "news_data": {
                "items": items,
                "count": len(items),
                "query_time": datetime.now().isoformat(timespec="seconds"),
                "date_range": f"{min(dates)}~{max(dates)}" if dates else "",
            },
            "error": "" if items else "本地新闻归档暂无可用记录",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "symbol": _compact_symbol(symbol),
            "data_success": False,
            "source": "local_news_archive",
            "local_first": True,
            "news_data": {"items": [], "count": 0},
            "error": f"本地新闻读取失败：{exc}",
        }


def _news_frame_from_items(symbol: str, items: list[dict]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    code = _compact_symbol(symbol)
    for item in items:
        if not isinstance(item, dict):
            continue
        title = _news_text(item.get("title") or item.get("标题") or item.get("新闻标题"))
        content = _news_text(
            item.get("content") or item.get("summary") or item.get("内容") or item.get("摘要"),
            5000,
        )
        if not title and not content:
            continue
        published = _news_datetime(item.get("published_at") or item.get("date") or item.get("时间") or item.get("日期"))
        source = _news_text(item.get("source") or item.get("feed") or "AI Agents 新闻源", 200)
        scope_text = " ".join(
            _news_text(item.get(key))
            for key in ("market", "market_scope", "topic", "channels", "source", "language")
        )
        scope_lower = scope_text.lower()
        is_overseas = (
            "海外" in scope_text
            or "美股" in scope_text
            or "rss/" in scope_lower
            or "alpha vantage" in scope_lower
            or "google news" in scope_lower
            or "nasdaq" in scope_lower
            or "nyse" in scope_lower
            or _news_text(item.get("language")).lower() == "en"
        )
        market = "overseas" if is_overseas else "domestic"
        default_scope = "美股/海外财经" if is_overseas else "A股/国内财经"
        symbols = _news_text(item.get("symbols"))
        if not symbols and not is_overseas and code:
            symbols = ",".join(sorted({code, _tickflow_symbol(code)}))
        rows.append({
            "id": _news_fingerprint(source, published.isoformat(), title, item.get("url"), content[:600]),
            "date": published.date(),
            "published_at": published.strftime("%Y-%m-%d %H:%M:%S"),
            "market": market,
            "market_scope": _news_text(item.get("market_scope") or default_scope),
            "source": source,
            "title": title or content[:80],
            "summary": content[:1200],
            "content": content,
            "channels": _news_text(item.get("topic") or item.get("channels")),
            "symbols": symbols,
            "url": _news_text(item.get("url")),
            "language": _news_text(item.get("language") or ("en" if is_overseas else "zh")),
            "sentiment": _news_text(item.get("sentiment")),
            "raw": _news_text(item),
        })
    if not rows:
        return pl.DataFrame(schema={**{col: pl.Utf8 for col in _NEWS_COLUMNS}, "date": pl.Date})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).select(_NEWS_COLUMNS)


def _persist_news_items(symbol: str, items: list[dict]) -> int:
    frame = _news_frame_from_items(symbol, items)
    if frame.is_empty():
        return 0
    repo = _runtime_repo()
    try:
        if repo is not None and hasattr(repo, "append_news"):
            repo.append_news(frame)
            return frame.height
    except Exception as exc:  # noqa: BLE001
        logger.debug("repo 新闻归档失败，降级直接写分区: %s", exc)
    return _append_partitioned_cache(
        settings.data_dir / "news",
        frame,
        date_col="date",
        unique_subset=["id"],
        sort_cols=["date", "published_at", "source"],
    )


def _persist_news_context(symbol: str, context: dict) -> int:
    news_data = context.get("news_data") if isinstance(context, dict) else {}
    return _persist_news_items(symbol, list((news_data or {}).get("items") or []))


def _news(symbol: str, name: str = "") -> dict:
    local = _news_from_local(symbol, name)
    local_count = int((local.get("news_data") or {}).get("count") or 0)
    if local_count >= 5:
        return _json_value(local)
    try:
        result = build_unified_news_context(_compact_symbol(symbol), name)
        archived = _persist_news_context(symbol, result)
        result = dict(result or {})
        result["local_first"] = True
        result["archived_rows"] = archived
        if not result.get("data_success") and local_count:
            local["remote_error"] = result.get("error") or "外部新闻源未返回有效数据"
            return _json_value(local)
        return _json_value(result)
    except Exception as exc:  # noqa: BLE001
        if local_count:
            local["remote_error"] = str(exc)
            return _json_value(local)
        return {"data_success": False, "news_data": {"items": [], "count": 0}, "error": str(exc)}


def _basic_info_from_local(symbol: str) -> dict:
    code = _compact_symbol(symbol)
    out: dict[str, Any] = {}
    instruments_path = settings.data_dir / "instruments" / "instruments.parquet"
    if instruments_path.exists():
        try:
            df = pl.read_parquet(str(instruments_path))
            if "symbol" in df.columns:
                df = (
                    df.with_columns(
                        pl.col("symbol")
                        .cast(pl.Utf8)
                        .str.extract(r"(\d{6})", 1)
                        .fill_null(pl.col("symbol").cast(pl.Utf8))
                        .alias("symbol")
                    )
                    .filter(pl.col("symbol") == code)
                )
                if not df.is_empty():
                    row = df.head(1).to_dicts()[0]
                    out.update({
                        "symbol": _tickflow_symbol(code),
                        "code": code,
                        "name": row.get("name"),
                        "exchange": row.get("exchange"),
                        "region": row.get("region"),
                        "type": row.get("type"),
                        "listing_date": row.get("listing_date"),
                        "total_shares": row.get("total_shares"),
                        "float_shares": row.get("float_shares"),
                        "limit_up": row.get("limit_up"),
                        "limit_down": row.get("limit_down"),
                        "as_of": row.get("as_of"),
                        "source": "local_instruments",
                    })
        except Exception as exc:  # noqa: BLE001
            logger.debug("本地 instruments 读取失败(%s): %s", symbol, exc)

    hy_path = settings.data_dir / "ext_data" / "ext_hy_ths" / "part.parquet"
    if hy_path.exists():
        try:
            df = pl.read_parquet(str(hy_path))
            key = "code" if "code" in df.columns else "symbol" if "symbol" in df.columns else ""
            if key:
                df = (
                    df.with_columns(
                        pl.col(key)
                        .cast(pl.Utf8)
                        .str.extract(r"(\d{6})", 1)
                        .fill_null(pl.col(key).cast(pl.Utf8))
                        .alias("_code")
                    )
                    .filter(pl.col("_code") == code)
                )
                if not df.is_empty():
                    row = df.head(1).to_dicts()[0]
                    out["industry"] = row.get("所属同花顺行业") or row.get("industry")
        except Exception as exc:  # noqa: BLE001
            logger.debug("本地同花顺行业读取失败(%s): %s", symbol, exc)

    gn_path = settings.data_dir / "ext_data" / "ext_gn_ths" / "part.parquet"
    if gn_path.exists():
        try:
            df = pl.read_parquet(str(gn_path))
            key = "code" if "code" in df.columns else "symbol" if "symbol" in df.columns else ""
            if key:
                df = (
                    df.with_columns(
                        pl.col(key)
                        .cast(pl.Utf8)
                        .str.extract(r"(\d{6})", 1)
                        .fill_null(pl.col(key).cast(pl.Utf8))
                        .alias("_code")
                    )
                    .filter(pl.col("_code") == code)
                )
                if not df.is_empty():
                    row = df.head(1).to_dicts()[0]
                    out["concepts"] = row.get("所属概念") or row.get("concepts")
        except Exception as exc:  # noqa: BLE001
            logger.debug("本地同花顺概念读取失败(%s): %s", symbol, exc)

    return _json_value(out)


def _basic_info(symbol: str) -> dict:
    local = _basic_info_from_local(symbol)
    if local:
        return local
    manager = _legacy_manager()
    if manager is None:
        return {}
    try:
        return _json_value(manager.get_stock_basic_info(_compact_symbol(symbol)) or {})
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _overseas() -> dict:
    try:
        payload = overseas_market_data.get_context_for_analysis(force_refresh=False)
        items: list[dict] = []
        refreshed = False
        for key in ("global_news", "symbol_news"):
            news = (payload or {}).get(key) or {}
            if news.get("data_success") and not news.get("cache_hit"):
                refreshed = True
            for item in news.get("items") or []:
                if isinstance(item, dict):
                    row = dict(item)
                    row.setdefault("market", "overseas")
                    row.setdefault("market_scope", "美股/海外财经")
                    row.setdefault("language", "en")
                    items.append(row)
        local_overseas_count = int((_news_from_local("", "").get("overseas_count") or 0))
        archived = _persist_news_items("", items) if (refreshed or local_overseas_count < 5) else 0
        payload = dict(payload or {})
        payload["local_first"] = True
        payload["archived_news_rows"] = archived
        return _json_value(payload)
    except Exception as exc:  # noqa: BLE001
        return {"data_success": False, "market": {"quotes": []}, "global_news": {"items": []}, "error": str(exc)}


def market_context(
    symbol: str,
    include_fund_flow: bool = True,
    include_news: bool = True,
    include_overseas: bool = True,
    fund_flow_days: int = 60,
) -> dict:
    code = _compact_symbol(symbol)
    daily = get_daily(code, days=180)
    quote = _get_quote(code)
    technical = _technical_summary(daily)
    basic = _basic_info(code)
    fund_flow = _fund_flow(code, fund_flow_days) if include_fund_flow else {"skipped": True}
    if include_fund_flow:
        fund_flow = _annotate_fund_freshness(fund_flow, technical.get("latest_date"))
    news = _news(code, basic.get("name", "")) if include_news else {"skipped": True}
    overseas = _overseas() if include_overseas else {"skipped": True}
    return _json_value({
        "symbol": _tickflow_symbol(code),
        "compact_symbol": code,
        "quote": quote,
        "technical": technical,
        "basic": basic,
        "fund_flow": fund_flow,
        "news": news,
        "overseas": overseas,
        "daily": _frame_records(daily, limit=120),
        "data_policy": {
            "mode": "local_first",
            "external_fallback": True,
            "archive_external_before_reuse": True,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "disclaimer": "公开数据增强分析，仅供研究，不构成投资建议。",
    })


def _score_context(context: dict) -> dict:
    technical = context.get("technical") or {}
    fund = context.get("fund_flow") or {}
    overseas = context.get("overseas") or {}
    score = 50.0
    evidence: list[str] = []
    risks: list[str] = []
    trend = technical.get("trend")
    if trend == "up":
        score += 15
        evidence.append("收盘价位于MA20上方，短中期趋势偏强")
    elif trend == "down":
        score -= 15
        risks.append("收盘价位于MA20下方，趋势偏弱")
    else:
        evidence.append("价格处于震荡区间，趋势确认不足")
    rsi = _safe_float(technical.get("rsi14"))
    if rsi is not None:
        if 45 <= rsi <= 70:
            score += 8
            evidence.append(f"RSI14={rsi:.1f}，未处于明显极端区")
        elif rsi > 78:
            score -= 8
            risks.append(f"RSI14={rsi:.1f}，短线偏热")
        elif rsi < 30:
            evidence.append(f"RSI14={rsi:.1f}，处于超卖区，需等待反转确认")
    flow_5 = _safe_float(fund.get("main_net_5d"), 0) or 0
    flow_20 = _safe_float(fund.get("main_net_20d"), 0) or 0
    if flow_5 > 0:
        score += 10
        evidence.append("近5日主力资金净流入")
    elif flow_5 < 0:
        score -= 10
        risks.append("近5日主力资金净流出")
    if flow_20 > 0:
        score += 5
    elif flow_20 < 0:
        score -= 5
    if not fund.get("data_success"):
        risks.append("资金流向数据不可用，资金面结论降级")
    elif fund.get("freshness_status") == "stale":
        score -= 5
        risks.append(str(fund.get("freshness_note") or "资金流数据明显滞后"))
    elif fund.get("freshness_status") == "lagging_1d":
        risks.append(str(fund.get("freshness_note") or "资金流数据滞后1日，按盘后确认数据降权参考"))
    market = overseas.get("market") or {}
    impact = _safe_float((overseas.get("impact") or {}).get("score"))
    if impact is not None:
        score += max(-5, min(5, (impact - 50) / 10))
    if not market.get("data_success") and context.get("overseas", {}).get("data_success") is False:
        risks.append("海外行情或新闻不可用")
    score = max(0, min(100, score))
    rating = "强势观察" if score >= 70 else "中性观察" if score >= 50 else "谨慎观察"
    support = _safe_float(technical.get("support"), 0) or 0
    resistance = _safe_float(technical.get("resistance"), 0) or 0
    close = _safe_float(technical.get("latest_close"), 0) or 0
    stop = support * 0.98 if support > 0 else close * 0.95
    return {
        "score": round(score, 2),
        "rating": rating,
        "evidence": evidence,
        "risks": risks,
        "support": support,
        "resistance": resistance,
        "reference_entry": [round(support, 2), round(max(support, _safe_float(technical.get("ma20"), support) or support), 2)],
        "reference_stop": round(stop, 2),
        "reference_take_profit": round(resistance, 2),
        "holding_period": "1-5个交易日观察，信号失效即退出",
    }


def analyze(symbol: str, include_fund_flow: bool = True, include_news: bool = True) -> dict:
    context = market_context(
        symbol,
        include_fund_flow=include_fund_flow,
        include_news=include_news,
        include_overseas=True,
    )
    decision = _score_context(context)
    technical = context.get("technical") or {}
    fund = context.get("fund_flow") or {}
    basic = context.get("basic") or {}
    news = context.get("news") or {}
    return {
        "symbol": context["symbol"],
        "generated_at": context["generated_at"],
        "mode": "structured_multi_agent",
        "decision": decision,
        "agents": {
            "technical": {
                "status": "ok" if technical.get("data_success") else "degraded",
                "summary": f"趋势={technical.get('trend', 'unknown')}，支撑={technical.get('support', '--')}，压力={technical.get('resistance', '--')}，RSI14={technical.get('rsi14', '--')}",
                "data": technical,
            },
            "fundamental": {
                "status": "ok" if basic else "degraded",
                "summary": f"名称={basic.get('name', '--')}，行业={basic.get('industry', '--')}，市值={basic.get('market_cap', '--')}；估值字段需财务数据源进一步补齐",
                "data": basic,
            },
            "fund_flow": {
                "status": "ok" if fund.get("data_success") else "degraded",
                "summary": f"近5日主力净流入={fund.get('main_net_5d', 0):,.2f}，近20日={fund.get('main_net_20d', 0):,.2f}",
                "data": fund,
            },
            "risk": {
                "status": "ok",
                "summary": "；".join(decision["risks"]) or "当前规则未发现明显风险项",
                "data": {"risks": decision["risks"]},
            },
            "sentiment_news": {
                "status": "ok" if news.get("data_success") else "degraded",
                "summary": f"新闻条数={((news.get('news_data') or {}).get('count', 0))}，海外市场数据已纳入上下文",
                "data": {"news": news, "overseas": context.get("overseas")},
            },
        },
        "context": context,
        "disclaimer": context["disclaimer"],
    }


def legacy_holdings() -> dict:
    snapshot = paper_store.snapshot()
    rows = snapshot.get("positions") or []
    enriched = []
    for row in rows:
        code = _compact_symbol(row.get("symbol") or row.get("code") or "")
        if not code:
            continue
        context = market_context(code, include_fund_flow=True, include_news=True, include_overseas=False)
        enriched.append(
            {
                **_json_value(row),
                "code": code,
                "symbol": _tickflow_symbol(code),
                "quote": context.get("quote"),
                "technical": context.get("technical"),
                "fund_flow": context.get("fund_flow"),
                "optimization": _score_context(context),
            }
        )
    return {"rows": enriched, "count": len(enriched), "source": snapshot.get("db_path", str(paper_store.path))}


class PaperStore:
    """TickFlow 自己维护的轻量模拟盘，全部落在本地数据目录。"""

    def __init__(self, path: Path | None = None):
        self.path = path or settings.data_dir / "aiagents_paper.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    initial_cash REAL NOT NULL,
                    cash REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    account_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    quantity INTEGER NOT NULL,
                    available_quantity INTEGER NOT NULL,
                    cost_price REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(account_id, symbol)
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    reason TEXT DEFAULT ''
                );
                """
            )

    def account(self, name: str = "TickFlow模拟账户", initial_cash: float = 100000.0) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM accounts WHERE name=?", (name,)).fetchone()
            if row:
                return dict(row)
            now = datetime.now().isoformat(timespec="seconds")
            cur = conn.execute(
                "INSERT INTO accounts(name, initial_cash, cash, created_at) VALUES(?,?,?,?)",
                (name, float(initial_cash), float(initial_cash), now),
            )
            return dict(conn.execute("SELECT * FROM accounts WHERE id=?", (cur.lastrowid,)).fetchone())

    def snapshot(self, name: str = "TickFlow模拟账户") -> dict:
        account = self.account(name)
        with self._conn() as conn:
            positions = [dict(row) for row in conn.execute(
                "SELECT * FROM positions WHERE account_id=? ORDER BY symbol", (account["id"],)
            ).fetchall()]
            trades = [dict(row) for row in conn.execute(
                "SELECT * FROM trades WHERE account_id=? ORDER BY id DESC LIMIT 100", (account["id"],)
            ).fetchall()]
        market_value = 0.0
        unrealized = 0.0
        for position in positions:
            quote = _get_quote(position["symbol"])
            price = _safe_float(quote.get("price") or quote.get("last_price"), position["cost_price"]) or position["cost_price"]
            position["current_price"] = price
            position["market_value"] = price * position["quantity"]
            position["unrealized_pnl"] = (price - position["cost_price"]) * position["quantity"]
            position["unrealized_pnl_pct"] = (price / position["cost_price"] - 1) * 100 if position["cost_price"] else 0
            market_value += position["market_value"]
            unrealized += position["unrealized_pnl"]
        total = float(account["cash"]) + market_value
        return {
            "account": {**account, "market_value": market_value, "total_asset": total, "unrealized_pnl": unrealized,
                        "return_pct": (total / account["initial_cash"] - 1) * 100 if account["initial_cash"] else 0},
            "positions": positions,
            "trades": trades,
            "db_path": str(self.path),
        }

    def order(self, symbol: str, side: str, quantity: int, price: float | None = None, reason: str = "") -> dict:
        code = _compact_symbol(symbol)
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side 只能是 BUY 或 SELL")
        quantity = int(quantity)
        if quantity <= 0 or quantity % 100 != 0:
            raise ValueError("A股模拟交易数量必须是100股的整数倍")
        account = self.account()
        quote = _get_quote(code)
        market_price = _safe_float(quote.get("price") or quote.get("last_price"))
        trade_price = float(price or market_price or 0)
        if trade_price <= 0:
            raise ValueError("没有可用成交价")
        with self._conn() as conn:
            position = conn.execute(
                "SELECT * FROM positions WHERE account_id=? AND symbol=?", (account["id"], _tickflow_symbol(code))
            ).fetchone()
            now = datetime.now().isoformat(timespec="seconds")
            amount = trade_price * quantity
            fee = max(5.0, amount * 0.0003)
            if side == "BUY":
                if amount + fee > float(account["cash"]):
                    raise ValueError("模拟账户现金不足")
                old_qty = int(position["quantity"]) if position else 0
                old_cost = float(position["cost_price"]) if position else 0.0
                new_qty = old_qty + quantity
                new_cost = ((old_cost * old_qty) + amount + fee) / new_qty
                conn.execute("UPDATE accounts SET cash=cash-? WHERE id=?", (amount + fee, account["id"]))
                conn.execute(
                    """
                    INSERT INTO positions(account_id,symbol,name,quantity,available_quantity,cost_price,updated_at)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(account_id,symbol) DO UPDATE SET
                      quantity=excluded.quantity, available_quantity=excluded.available_quantity,
                      cost_price=excluded.cost_price, updated_at=excluded.updated_at
                    """,
                    (account["id"], _tickflow_symbol(code), quote.get("name", ""), new_qty, old_qty + quantity, new_cost, now),
                )
            else:
                if not position or int(position["available_quantity"]) < quantity:
                    raise ValueError("可卖持仓不足，或尚未满足T+1")
                remaining = int(position["quantity"]) - quantity
                conn.execute("UPDATE accounts SET cash=cash+? WHERE id=?", (amount - fee, account["id"]))
                if remaining == 0:
                    conn.execute("DELETE FROM positions WHERE account_id=? AND symbol=?", (account["id"], _tickflow_symbol(code)))
                else:
                    conn.execute(
                        "UPDATE positions SET quantity=?, available_quantity=?, updated_at=? WHERE account_id=? AND symbol=?",
                        (remaining, remaining, now, account["id"], _tickflow_symbol(code)),
                    )
            conn.execute(
                "INSERT INTO trades(account_id,symbol,side,quantity,price,amount,created_at,reason) VALUES(?,?,?,?,?,?,?,?)",
                (account["id"], _tickflow_symbol(code), side, quantity, trade_price, amount, now, reason),
            )
        return self.snapshot()


paper_store = PaperStore()
