"""历史行情文件导入主行情库。"""
from __future__ import annotations

import csv
import logging
import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable, Literal

import polars as pl

from app.data_providers.normalizer import normalize_daily
from app.indicators.pipeline import compute_enriched, run_pipeline
from app.services.ext_data import ensure_utf8_csv
from app.services.security_identity import IdentityBatchError, build_instrument_lookup, parse_identity, validate_frame_identity
from app.tickflow.repository import KlineRepository

logger = logging.getLogger(__name__)

AssetTypeHint = Literal["auto", "stock", "index", "etf"]


class HistoryCsvError(ValueError):
    """历史 CSV 格式错误。"""


class HistoryCsvMinuteError(HistoryCsvError):
    """当前导入器仅支持日线 CSV。"""


_ALIASES: dict[str, list[str]] = {
    "symbol": ["ts_code", "证券代码", "股票代码", "标的代码", "代码", "code"],
    "date": ["trade_date", "交易日期", "日期", "datetime", "时间"],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价"],
    "volume": ["vol", "volume", "成交量", "手数"],
    "amount": ["amt", "amount", "成交额"],
}

_MINUTE_COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]

_MINUTE_ALIASES: dict[str, list[str]] = {
    "symbol": ["ts_code", "证券代码", "股票代码", "标的代码", "代码", "code"],
    "datetime": [
        "__minute_datetime",
        "trade_time",
        "datetime",
        "bar_time",
        "timestamp",
        "日期时间",
        "交易时间",
        "时间",
        "date",
        "日期",
    ],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价"],
    "volume": ["vol", "volume", "成交量", "手数"],
    "amount": ["amt", "amount", "成交额", "成交金额"],
}


def _progress(cb: Callable[[str, int, str], None] | None, stage: str, pct: int, msg: str) -> None:
    if cb is not None:
        cb(stage, pct, msg)


def _build_code_lookup(repo: KlineRepository) -> dict[str, str]:
    """从股票 / 指数 / ETF 维表构建 code → symbol 映射。"""
    return build_instrument_lookup((
        repo.get_instruments(),
        repo.get_index_instruments(),
        repo.get_etf_instruments(),
    ))


def _detect_separator(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:  # noqa: BLE001
        counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        return best if counts.get(best, 0) > 0 else ","


def _read_csv(file_path: Path) -> pl.DataFrame:
    utf8_path = ensure_utf8_csv(file_path)
    sample = utf8_path.read_text(encoding="utf-8", errors="ignore")[:8192]
    separator = _detect_separator(sample)
    return pl.read_csv(
        utf8_path,
        separator=separator,
        infer_schema_length=10_000,
        ignore_errors=True,
    )


def _read_table(file_path: Path) -> pl.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(file_path)
    if suffix == ".parquet":
        try:
            return pl.read_parquet(file_path)
        except Exception as exc:  # noqa: BLE001
            raise HistoryCsvError(f"无法读取 Parquet 文件 {file_path.name}: {exc}") from exc
    raise HistoryCsvError(f"不支持的历史文件类型: {file_path.suffix}")


def _infer_symbol_from_filename(file_path: Path, lookup: dict[str, str]) -> str | None:
    stem = file_path.stem.upper()
    match = re.search(r"(\d{6}\.[A-Z]{2})", stem)
    if match:
        return match.group(1)
    match = re.search(r"(\d{6})", stem)
    if not match:
        return None
    code = match.group(1)
    mapped = lookup.get(code)
    if mapped:
        return mapped
    # Unknown bare codes are isolated. Never guess SZ for an unrecognised file.
    return None


def _rename_alias_columns_from(df: pl.DataFrame, aliases: dict[str, list[str]]) -> pl.DataFrame:
    rename_map: dict[str, str] = {}
    drop_cols: list[str] = []
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    for target, sources in aliases.items():
        if target in df.columns:
            drop_cols.extend([
                lower_cols[source.strip().lower()]
                for source in sources
                if source.strip().lower() in lower_cols and lower_cols[source.strip().lower()] != target
            ])
            continue
        chosen: str | None = None
        for source in sources:
            actual = lower_cols.get(source.strip().lower())
            if actual in df.columns:
                chosen = actual
                break
        if chosen is not None:
            rename_map[chosen] = target
            drop_cols.extend([
                lower_cols[source.strip().lower()]
                for source in sources
                if source.strip().lower() in lower_cols and lower_cols[source.strip().lower()] != chosen
            ])
    if rename_map:
        df = df.rename(rename_map)
    if drop_cols:
        keep = [c for c in df.columns if c not in set(drop_cols)]
        df = df.select(keep)
    return df


def _rename_alias_columns(df: pl.DataFrame) -> pl.DataFrame:
    return _rename_alias_columns_from(df, _ALIASES)


def _combine_minute_datetime_columns(df: pl.DataFrame) -> pl.DataFrame:
    date_candidates = ["trade_date", "date", "日期"]
    time_candidates = ["trade_time", "time", "minute", "时间"]
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    for date_col in date_candidates:
        actual_date_col = lower_cols.get(date_col.strip().lower())
        if actual_date_col not in df.columns:
            continue
        for time_col in time_candidates:
            actual_time_col = lower_cols.get(time_col.strip().lower())
            if actual_time_col not in df.columns or actual_time_col == actual_date_col:
                continue
            # Some daily-partitioned Parquet files carry both `date` and a
            # complete `trade_time` (YYYY-MM-DD HH:MM:SS). Do not prepend the
            # date a second time; that would make every timestamp invalid.
            sample = df.get_column(actual_time_col).drop_nulls().head(1)
            if sample.len() and re.search(r"(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}", str(sample.item())):
                return df.with_columns(pl.col(actual_time_col).alias("__minute_datetime"))
            return df.with_columns(
                (
                    pl.col(actual_date_col).cast(pl.Utf8).str.strip_chars()
                    + pl.lit(" ")
                    + pl.col(actual_time_col).cast(pl.Utf8).str.strip_chars()
                ).alias("__minute_datetime")
            )
    return df


def _parse_daily_value(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.time() != time(0, 0):
            raise HistoryCsvMinuteError("当前只支持日线 CSV，请先转成日线后再导入")
        return value.date()
    if isinstance(value, date):
        return value

    s = str(value).strip()
    if not s:
        return None
    s = s.replace("/", "-")
    if "T" in s or " " in s:
        try:
            dt = datetime.fromisoformat(s.replace("Z", ""))
            if dt.time() != time(0, 0):
                raise HistoryCsvMinuteError("当前只支持日线 CSV，请先转成日线后再导入")
            return dt.date()
        except ValueError:
            pass

    digits = re.sub(r"\D", "", s)
    if len(digits) >= 8:
        if len(digits) > 8 and any(ch != "0" for ch in digits[8:]):
            raise HistoryCsvMinuteError("当前只支持日线 CSV，请先转成日线后再导入")
        try:
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        except ValueError as exc:  # noqa: BLE001
            raise HistoryCsvError(f"无法解析日期: {value!r}") from exc

    try:
        return date.fromisoformat(s[:10])
    except ValueError as exc:  # noqa: BLE001
        raise HistoryCsvError(f"无法解析日期: {value!r}") from exc


def _coerce_daily_date_column(df: pl.DataFrame) -> pl.DataFrame:
    if "date" not in df.columns:
        raise HistoryCsvError("未找到日期列，请提供 trade_date / date / datetime")
    parsed = df["date"].map_elements(_parse_daily_value, return_dtype=pl.Date)
    return df.with_columns(parsed.alias("date"))


def _ensure_required_columns(df: pl.DataFrame) -> pl.DataFrame:
    required = ["symbol", "date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HistoryCsvError(f"缺少必要列: {', '.join(missing)}")

    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("volume"))
    if "amount" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("amount"))
    return df


def _select_storage_columns(df: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in ["symbol", "date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    return df.select(cols)


def _validate_lifecycle_rows(
    df: pl.DataFrame,
    repo: KlineRepository,
    date_column: str,
    *,
    lifecycle_index: dict | None = None,
) -> None:
    """Reject historical rows whose security did not exist under that identity."""
    if df.is_empty() or not {"symbol", date_column}.issubset(df.columns):
        return
    from collections import Counter
    from app.services.security_lifecycle import load_lifecycle_index, validate_security_lifecycle

    index = lifecycle_index if lifecycle_index is not None else load_lifecycle_index(repo.store.data_dir)
    # Minimal test/import directories may not have a dimension yet.  Identity
    # validation still runs separately; do not invent lifecycle metadata.
    if not index:
        return
    failures = Counter()
    examples: dict[str, str] = {}
    for row in df.select(["symbol", date_column]).unique().iter_rows(named=True):
        decision = validate_security_lifecycle(
            row["symbol"], row[date_column], repo.store.data_dir, index=index,
        )
        if not decision.ok:
            failures[decision.reason] += 1
            examples.setdefault(decision.reason, decision.symbol)
    if failures:
        detail = ", ".join(
            f"{reason}={count}(示例:{examples.get(reason, '')})"
            for reason, count in sorted(failures.items())
        )
        raise HistoryCsvError(f"生命周期校验失败，未写入数据: {detail}")


def _normalize_history_df(
    file_path: Path,
    repo: KlineRepository,
    lifecycle_index: dict | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    raw = _read_csv(file_path)
    meta = {"source_rows": raw.height, "dropped_rows": 0}
    if raw.is_empty():
        return raw, meta

    df = _rename_alias_columns(raw)
    lookup = {str(k): str(v) for k, v in _build_code_lookup(repo).items()}

    if "symbol" not in df.columns:
        inferred = _infer_symbol_from_filename(file_path, lookup)
        if inferred:
            df = df.with_columns(pl.lit(inferred).alias("symbol"))

    if "symbol" not in df.columns:
        raise HistoryCsvError("未找到标的列，请提供 ts_code / symbol / code，或把文件命名为 000001.csv 之类")

    df = _coerce_daily_date_column(df)
    df = normalize_daily(df)

    df = _ensure_required_columns(df)
    try:
        validate_frame_identity(
            df,
            asset_type="stock",
            lookup=lookup,
            require_lookup=True,
        )
    except IdentityBatchError as exc:
        raise HistoryCsvError(str(exc)) from exc
    df = df.with_columns(
        pl.col("symbol").cast(pl.Utf8).map_elements(
            lambda value: parse_identity(value, asset_type="stock", lookup=lookup, require_lookup=True).symbol,
            return_dtype=pl.Utf8,
        ).alias("symbol")
    )
    _validate_lifecycle_rows(df, repo, "date", lifecycle_index=lifecycle_index)

    # 丢弃不完整行，避免脏数据污染主库。
    before = df.height
    df = df.drop_nulls(subset=["symbol", "date", "open", "high", "low", "close"])
    meta["dropped_rows"] = before - df.height
    if df.is_empty():
        raise HistoryCsvError("CSV 中没有可导入的有效行情行")

    return df.sort(["symbol", "date"]).unique(subset=["symbol", "date"], keep="last"), meta


def _parse_minute_value(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value.astimezone().replace(tzinfo=None)
        return value.replace(second=0, microsecond=0)
    if isinstance(value, date):
        return datetime.combine(value, time(0, 0))

    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "nan", "nat"}:
        return None
    s = s.replace("/", "-").replace("T", " ").replace("Z", "").strip()
    digits = re.sub(r"\D", "", s)

    if digits and len(digits) >= 14 and digits.startswith(("19", "20")):
        try:
            return datetime(
                int(digits[:4]),
                int(digits[4:6]),
                int(digits[6:8]),
                int(digits[8:10]),
                int(digits[10:12]),
                int(digits[12:14]),
            ).replace(second=0, microsecond=0)
        except ValueError as exc:
            raise HistoryCsvError(f"无法解析分钟时间: {value!r}") from exc
    if digits and len(digits) == 12 and digits.startswith(("19", "20")):
        try:
            return datetime(
                int(digits[:4]),
                int(digits[4:6]),
                int(digits[6:8]),
                int(digits[8:10]),
                int(digits[10:12]),
            )
        except ValueError as exc:
            raise HistoryCsvError(f"无法解析分钟时间: {value!r}") from exc
    if digits and len(digits) == 8 and digits.startswith(("19", "20")):
        try:
            return datetime(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        except ValueError as exc:
            raise HistoryCsvError(f"无法解析分钟时间: {value!r}") from exc
    if digits and len(digits) in {10, 13} and not digits.startswith(("19", "20")):
        try:
            seconds = int(digits) / (1000 if len(digits) == 13 else 1)
            return datetime.fromtimestamp(seconds).replace(second=0, microsecond=0)
        except Exception as exc:  # noqa: BLE001
            raise HistoryCsvError(f"无法解析分钟时间戳: {value!r}") from exc

    try:
        parsed = datetime.fromisoformat(s)
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed.replace(second=0, microsecond=0)
    except ValueError as exc:
        raise HistoryCsvError(f"无法解析分钟时间: {value!r}") from exc


def _ensure_required_minute_columns(df: pl.DataFrame) -> pl.DataFrame:
    required = ["symbol", "datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HistoryCsvError(f"缺少分钟K必要列: {', '.join(missing)}")
    if "volume" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("volume"))
    if "amount" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("amount"))
    return df


def _filter_regular_minutes(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "datetime" not in df.columns:
        return df
    return (
        df.with_columns(pl.col("datetime").dt.strftime("%H:%M").alias("_hm"))
        .filter(
            ((pl.col("_hm") >= "09:31") & (pl.col("_hm") <= "11:30"))
            | ((pl.col("_hm") >= "13:01") & (pl.col("_hm") <= "15:00"))
        )
        .drop("_hm")
    )


def _normalize_minute_history_df(
    file_path: Path,
    repo: KlineRepository,
    lookup: dict[str, str] | None = None,
    lifecycle_index: dict | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    raw = _read_table(file_path)
    meta = {"source_rows": raw.height, "dropped_rows": 0}
    if raw.is_empty():
        return raw, meta

    df = _rename_alias_columns_from(_combine_minute_datetime_columns(raw), _MINUTE_ALIASES)
    if lookup is None:
        lookup = {str(k): str(v) for k, v in _build_code_lookup(repo).items()}

    if "symbol" not in df.columns:
        inferred = _infer_symbol_from_filename(file_path, lookup)
        if inferred:
            df = df.with_columns(pl.lit(inferred).alias("symbol"))
    if "symbol" not in df.columns:
        raise HistoryCsvError("未找到标的列，请提供 ts_code / symbol / code，或把文件命名为 000001.csv 之类")
    if "datetime" not in df.columns:
        raise HistoryCsvError("未找到分钟时间列，请提供 trade_time / datetime / 日期时间，或 date + time 两列")

    raw_datetime = df["datetime"]
    # 常见离线包是 YYYY-MM-DD HH:MM，使用 Polars 向量化解析；其他来源保留
    # 原有逐值解析，兼容时间戳、紧凑日期和带时区格式。
    sample_datetime = raw_datetime.head(1).item() if raw_datetime.len() else None
    if isinstance(sample_datetime, str) and re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", sample_datetime.strip()):
        # Let Polars infer optional seconds/fractional seconds. Fixed
        # `%Y-%m-%d %H:%M` parsing rejects common `HH:MM:SS` exports.
        datetime_expr = pl.col("datetime").cast(pl.Utf8).str.to_datetime(strict=False)
    else:
        datetime_expr = df["datetime"].map_elements(
            _parse_minute_value, return_dtype=pl.Datetime("us"),
        )
    df = df.with_columns(datetime_expr.alias("datetime"))
    df = _ensure_required_minute_columns(df)
    try:
        validate_frame_identity(
            df,
            asset_type="stock",
            lookup=lookup,
            require_lookup=True,
        )
    except IdentityBatchError as exc:
        raise HistoryCsvError(str(exc)) from exc
    df = df.with_columns(
        pl.col("symbol").cast(pl.Utf8).map_elements(
            lambda value: parse_identity(value, asset_type="stock", lookup=lookup, require_lookup=True).symbol,
            return_dtype=pl.Utf8,
        ).alias("symbol")
    )
    _validate_lifecycle_rows(df, repo, "datetime", lifecycle_index=lifecycle_index)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        df = df.with_columns(pl.col(column).cast(pl.Float64, strict=False).alias(column))

    before = df.height
    df = df.drop_nulls(subset=["symbol", "datetime", "open", "high", "low", "close"])
    df = _filter_regular_minutes(df)
    meta["dropped_rows"] = before - df.height
    if df.is_empty():
        raise HistoryCsvError("文件中没有可导入的有效连续竞价分钟K行")

    return (
        df.select(_MINUTE_COLS)
        .sort(["symbol", "datetime"])
        .unique(subset=["symbol", "datetime"], keep="last", maintain_order=True),
        meta,
    )


def _finalize_history_import(
    repo: KlineRepository,
    stock_symbols: list[str],
    on_progress: Callable[[str, int, str], None] | None = None,
) -> int:
    wrote_enriched = 0
    if stock_symbols:
        _progress(on_progress, "rebuild_stock", 60, "重算股票指标 …")

        def _on_batch_done(cur: int, total: int) -> None:
            pct = 60 + int(30 * cur / max(total, 1))
            _progress(on_progress, "rebuild_stock", pct, f"重算股票指标 {cur}/{total}")

        wrote_enriched += run_pipeline(
            data_dir=repo.store.data_dir,
            symbols=sorted(set(stock_symbols)),
            on_batch_done=_on_batch_done,
        )

    _progress(on_progress, "refresh", 95, "刷新视图 …")
    repo.rebuild_views()
    repo.clear_cache()
    return wrote_enriched


def _import_history_csv_single(
    repo: KlineRepository,
    file_path: Path,
    asset_type_hint: AssetTypeHint = "auto",
    on_progress: Callable[[str, int, str], None] | None = None,
    *,
    finalize: bool = True,
) -> tuple[dict, list[str]]:
    """导入单个历史 CSV。finalize=False 时仅写入原始/分资产指标，不做最终股票重算。"""
    if asset_type_hint not in ("auto", "stock", "index", "etf"):
        raise HistoryCsvError(f"不支持的资产类型: {asset_type_hint}")

    _progress(on_progress, "parse_csv", 5, "读取并识别 CSV …")
    df, meta = _normalize_history_df(file_path, repo)
    _progress(on_progress, "parse_csv", 20, f"解析完成, {df.height} 行有效数据")

    if asset_type_hint == "auto":
        df = df.with_columns(
            pl.col("symbol").map_elements(repo.resolve_asset_type, return_dtype=pl.Utf8).alias("asset_type")
        )
    else:
        df = df.with_columns(pl.lit(asset_type_hint).alias("asset_type"))

    groups = []
    stock_symbols: list[str] = []
    wrote_raw = 0
    wrote_enriched = 0

    for asset_type in ("stock", "index", "etf"):
        part = df.filter(pl.col("asset_type") == asset_type)
        if part.is_empty():
            continue
        symbols = sorted(set(part["symbol"].cast(pl.Utf8).to_list()))
        groups.append({
            "asset_type": asset_type,
            "rows": part.height,
            "symbols": len(symbols),
        })
        _progress(on_progress, f"write_{asset_type}", 30, f"写入 {asset_type} 原始日线 …")
        repo.append_daily_asset(asset_type, _select_storage_columns(part))
        wrote_raw += part.height

        if asset_type == "stock":
            stock_symbols.extend(symbols)
        elif asset_type == "index":
            _progress(on_progress, "rebuild_index", 60, "重算指数指标 …")
            enriched = compute_enriched(_select_storage_columns(part), factors=None, instruments=None)
            repo.append_index_enriched(enriched)
            wrote_enriched += enriched.height
        elif asset_type == "etf":
            factor_path = repo.store.data_dir / "adj_factor_etf" / "all.parquet"
            factors = pl.read_parquet(factor_path) if factor_path.exists() else pl.DataFrame()
            if not factors.is_empty() and "symbol" in factors.columns:
                factors = factors.filter(pl.col("symbol").is_in(symbols))
            _progress(on_progress, "rebuild_etf", 60, "重算 ETF 指标 …")
            enriched = compute_enriched(_select_storage_columns(part), factors=factors, instruments=None)
            repo.append_etf_enriched(enriched)
            wrote_enriched += enriched.height

    if finalize:
        wrote_enriched += _finalize_history_import(repo, stock_symbols, on_progress)

    result = {
        "status": "ok",
        "file": file_path.name,
        "asset_type": asset_type_hint,
        "source_rows": meta["source_rows"],
        "dropped_rows": meta["dropped_rows"],
        "written_rows": wrote_raw,
        "recomputed_rows": wrote_enriched,
        "symbols": len(set(df["symbol"].cast(pl.Utf8).to_list())),
        "groups": groups,
        "finalized": finalize,
    }
    if finalize:
        _progress(on_progress, "done", 100, f"导入完成, 原始 {wrote_raw} 行, 重算 {wrote_enriched} 行")
    return result, stock_symbols


def import_history_csv(
    repo: KlineRepository,
    file_path: Path,
    asset_type_hint: AssetTypeHint = "auto",
    on_progress: Callable[[str, int, str], None] | None = None,
    finalize: bool = True,
) -> dict:
    """导入历史日线 CSV 到主行情库。"""
    result, _ = _import_history_csv_single(
        repo,
        file_path,
        asset_type_hint=asset_type_hint,
        on_progress=on_progress,
        finalize=finalize,
    )
    return result


def import_history_csv_batch(
    repo: KlineRepository,
    file_paths: list[Path],
    asset_type_hint: AssetTypeHint = "auto",
    on_progress: Callable[[str, int, str], None] | None = None,
) -> dict:
    """批量导入多个历史 CSV。股票重算与最终刷新仅执行一次。"""
    if asset_type_hint not in ("auto", "stock", "index", "etf"):
        raise HistoryCsvError(f"不支持的资产类型: {asset_type_hint}")
    if not file_paths:
        raise HistoryCsvError("未提供可导入的 CSV 文件")

    total_written = 0
    total_recomputed = 0
    total_source_rows = 0
    total_dropped_rows = 0
    total_symbols = 0
    files: list[dict] = []
    stock_symbols: list[str] = []

    for idx, file_path in enumerate(file_paths, start=1):
        _progress(on_progress, "parse_csv", max(5, int(20 * idx / max(len(file_paths), 1))), f"读取文件 {idx}/{len(file_paths)}: {file_path.name}")
        result, symbols = _import_history_csv_single(
            repo,
            file_path,
            asset_type_hint=asset_type_hint,
            on_progress=on_progress,
            finalize=False,
        )
        files.append(result)
        stock_symbols.extend(symbols)
        total_written += int(result.get("written_rows", 0))
        total_recomputed += int(result.get("recomputed_rows", 0))
        total_source_rows += int(result.get("source_rows", 0))
        total_dropped_rows += int(result.get("dropped_rows", 0))
        total_symbols += int(result.get("symbols", 0))

    total_recomputed += _finalize_history_import(repo, stock_symbols, on_progress)

    result = {
        "status": "ok",
        "file_count": len(file_paths),
        "asset_type": asset_type_hint,
        "source_rows": total_source_rows,
        "dropped_rows": total_dropped_rows,
        "written_rows": total_written,
        "recomputed_rows": total_recomputed,
        "symbols": total_symbols,
        "files": files,
        "finalized": True,
    }
    _progress(on_progress, "done", 100, f"批量导入完成, 文件 {len(file_paths)} 个, 原始 {total_written} 行, 重算 {total_recomputed} 行")
    return result


def import_history_minute_batch(
    repo: KlineRepository,
    file_paths: list[Path],
    on_progress: Callable[[str, int, str], None] | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
) -> dict:
    """批量导入历史分钟K CSV / Parquet 到 kline_minute。

    start_date/end_date 用于离线包的窗口导入，避免每次把单股文件中的全历史
    重新解析和写回。边界按中国交易日的自然日处理，缺失交易日不会伪造数据。
    """
    if not file_paths:
        raise HistoryCsvError("未提供可导入的分钟K文件")

    def _bound(value: str | date | None, label: str) -> date | None:
        if value is None or str(value).strip() == "":
            return None
        if isinstance(value, date):
            return value
        try:
            return date.fromisoformat(str(value).strip()[:10])
        except ValueError as exc:
            raise HistoryCsvError(f"{label} 必须是 YYYY-MM-DD") from exc

    start_bound = _bound(start_date, "start_date")
    end_bound = _bound(end_date, "end_date")
    if start_bound is not None and end_bound is not None and start_bound > end_bound:
        raise HistoryCsvError("start_date 不能晚于 end_date")
    start_dt = datetime.combine(start_bound, time.min) if start_bound is not None else None
    end_dt = datetime.combine(end_bound, time.max) if end_bound is not None else None

    from app.services.kline_sync import _write_minute_partition

    minute_dir = repo.store.data_dir / "kline_minute"
    # 单股 CSV 只携带代码时需要维表映射。映射在整批开始时构建一次，
    # 避免 5,000+ 个文件各自重复扫描维表和 Parquet。
    code_lookup = {str(k): str(v) for k, v in _build_code_lookup(repo).items()}
    from app.services.security_lifecycle import load_lifecycle_index
    lifecycle_index = load_lifecycle_index(repo.store.data_dir)
    total_source_rows = 0
    total_dropped_rows = 0
    total_import_rows = 0
    total_partition_rows = 0
    symbols: set[str] = set()
    files: list[dict] = []
    pending_frames: list[pl.DataFrame] = []
    pending_rows = 0
    batch_rows_target = 1_000_000

    def _flush_pending() -> None:
        nonlocal total_partition_rows, pending_rows
        if not pending_frames:
            return
        batch = pl.concat(pending_frames, how="vertical_relaxed", rechunk=True)
        with repo._write_lock:
            total_partition_rows += _write_minute_partition(
                batch,
                minute_dir,
                asset_type="stock",
                identity_lookup=code_lookup,
            )
        pending_frames.clear()
        pending_rows = 0

    for idx, file_path in enumerate(file_paths, start=1):
        pct = max(5, int(90 * idx / max(len(file_paths), 1)))
        _progress(on_progress, "parse_minute", pct, f"读取分钟文件 {idx}/{len(file_paths)}: {file_path.name}")
        df, meta = _normalize_minute_history_df(file_path, repo, code_lookup, lifecycle_index)
        if start_dt is not None:
            df = df.filter(pl.col("datetime") >= pl.lit(start_dt))
        if end_dt is not None:
            df = df.filter(pl.col("datetime") <= pl.lit(end_dt))
        imported_rows = df.height
        total_source_rows += int(meta["source_rows"])
        total_dropped_rows += int(meta["dropped_rows"])
        total_import_rows += imported_rows
        if not df.is_empty():
            symbols.update(str(s) for s in df["symbol"].cast(pl.Utf8).unique().to_list())
            pending_frames.append(df)
            pending_rows += imported_rows
            if pending_rows >= batch_rows_target:
                _flush_pending()
        files.append({
            "file": file_path.name,
            "source_rows": int(meta["source_rows"]),
            "dropped_rows": int(meta["dropped_rows"]),
            "import_rows": imported_rows,
        })

    _flush_pending()

    _progress(on_progress, "refresh", 95, "刷新分钟K视图 …")
    try:
        d = repo.store.data_dir.as_posix()
        repo.db.execute(
            f"""CREATE OR REPLACE VIEW kline_minute AS
                SELECT * FROM read_parquet('{d}/kline_minute/**/*.parquet', union_by_name=true)"""
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("refresh kline_minute view after minute import failed: %s", exc)
    repo.clear_cache()

    result = {
        "status": "ok",
        "data_kind": "minute",
        "file_count": len(file_paths),
        "source_rows": total_source_rows,
        "dropped_rows": total_dropped_rows,
        "import_rows": total_import_rows,
        "written_rows": total_partition_rows,
        "symbols": len(symbols),
        "files": files,
        "finalized": True,
    }
    _progress(
        on_progress,
        "done",
        100,
        f"分钟K导入完成, 文件 {len(file_paths)} 个, 有效 {total_import_rows} 行, 标的 {len(symbols)} 只",
    )
    return result
