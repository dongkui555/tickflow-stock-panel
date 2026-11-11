from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl


CANONICAL_COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]

COLUMN_ALIASES = {
    "date": ["date", "trade_date", "日期"],
    "time": ["time", "trade_time", "时间"],
    "datetime": ["datetime", "trade_time", "日期时间", "交易时间"],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价"],
    "volume": ["vol", "volume", "成交量", "手数"],
    "amount": ["amt", "amount", "成交额", "成交金额"],
}


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _normalize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    rename: dict[str, str] = {}
    for col in df.columns:
        clean = str(col).strip().lstrip("\ufeff")
        if clean and clean != col:
            rename[col] = clean
    return df.rename(rename) if rename else df


def _pick(df: pl.DataFrame, target: str) -> str | None:
    lower = {str(col).strip().lower(): col for col in df.columns}
    for name in COLUMN_ALIASES.get(target, []):
        actual = lower.get(name.lower())
        if actual in df.columns:
            return actual
    return None


def _pick_from_names(names: list[str], target: str) -> str | None:
    lower = {str(col).strip().lstrip("\ufeff").lower(): col for col in names}
    for name in COLUMN_ALIASES.get(target, []):
        actual = lower.get(name.lower())
        if actual:
            return actual
    return None


def _symbol_from_filename(path: Path) -> str | None:
    stem = path.stem.strip().lower()
    match = re.search(r"(sh|sz|bj)?(\d{6})", stem)
    if not match:
        return None
    prefix, code = match.groups()
    structural = (
        "BJ" if code.startswith(("43", "83", "87", "88", "92"))
        else "SH" if code.startswith(("60", "68"))
        else "SZ" if code.startswith(("00", "30"))
        else None
    )
    if structural is None:
        return None
    if prefix and prefix.upper() != structural.lower():
        return None
    return f"{code}.{structural}"


def _symbol_expr_from_path(path_col: str = "__file_path") -> pl.Expr:
    token = (
        pl.col(path_col)
        .cast(pl.Utf8)
        .str.extract(r"(?i)(?:^|[/\\])((?:sh|sz|bj)?\d{6})\.csv$", 1)
        .str.to_lowercase()
    )
    code = token.str.extract(r"(\d{6})", 1)
    prefix = token.str.extract(r"^(sh|sz|bj)", 1)
    structural = (
        pl.when(code.str.starts_with("43") | code.str.starts_with("83") | code.str.starts_with("87") | code.str.starts_with("88") | code.str.starts_with("92"))
        .then(pl.lit("BJ"))
        .when(code.str.starts_with("60") | code.str.starts_with("68"))
        .then(pl.lit("SH"))
        .when(code.str.starts_with("00") | code.str.starts_with("30"))
        .then(pl.lit("SZ"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    return (
        pl.when(prefix == "sh")
        .then(pl.when(structural == "SH").then(code + pl.lit(".SH")))
        .when(prefix == "sz")
        .then(pl.when(structural == "SZ").then(code + pl.lit(".SZ")))
        .when(prefix == "bj")
        .then(pl.when(structural == "BJ").then(code + pl.lit(".BJ")))
        .otherwise(pl.when(structural.is_not_null()).then(code + pl.lit(".") + structural))
        .alias("symbol")
    )


def _read_one(path: Path, expected_date: date | None) -> pl.DataFrame | None:
    symbol = _symbol_from_filename(path)
    if not symbol:
        return None
    try:
        df = pl.read_csv(path, encoding="utf8-lossy", infer_schema_length=1000, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN read failed: {path.name}: {exc}", file=sys.stderr)
        return None
    if df.is_empty():
        return None
    df = _normalize_column_names(df)

    datetime_col = _pick(df, "datetime")
    date_col = _pick(df, "date")
    time_col = _pick(df, "time")
    if datetime_col:
        datetime_expr = pl.col(datetime_col).cast(pl.Utf8).str.strip_chars()
    elif date_col and time_col:
        datetime_expr = (
            pl.col(date_col).cast(pl.Utf8).str.strip_chars()
            + pl.lit(" ")
            + pl.col(time_col).cast(pl.Utf8).str.strip_chars()
        )
    else:
        return None

    required = {name: _pick(df, name) for name in ("open", "high", "low", "close")}
    if any(value is None for value in required.values()):
        return None
    volume_col = _pick(df, "volume")
    amount_col = _pick(df, "amount")

    out = df.with_columns(
        [
            pl.lit(symbol).alias("symbol"),
            datetime_expr.str.to_datetime(strict=False).cast(pl.Datetime("us")).alias("datetime"),
            pl.col(required["open"]).cast(pl.Float64, strict=False).alias("open"),
            pl.col(required["high"]).cast(pl.Float64, strict=False).alias("high"),
            pl.col(required["low"]).cast(pl.Float64, strict=False).alias("low"),
            pl.col(required["close"]).cast(pl.Float64, strict=False).alias("close"),
            (
                pl.col(volume_col).cast(pl.Float64, strict=False)
                if volume_col
                else pl.lit(None, dtype=pl.Float64)
            ).alias("volume"),
            (
                pl.col(amount_col).cast(pl.Float64, strict=False)
                if amount_col
                else pl.lit(None, dtype=pl.Float64)
            ).alias("amount"),
        ]
    ).select(CANONICAL_COLS)

    out = out.drop_nulls(subset=["symbol", "datetime", "open", "high", "low", "close"])
    if expected_date is not None:
        out = out.filter(pl.col("datetime").dt.date() == expected_date)
    if out.is_empty():
        return None
    return out


def _load_day_fast(
    source: Path, expected_date: date | None, files_total: int,
    needed_pairs: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, dict]:
    lf = pl.scan_csv(
        str(source / "*.csv"),
        encoding="utf8-lossy",
        infer_schema_length=1000,
        ignore_errors=needed_pairs is None,
        include_file_paths="__file_path",
    )
    names = lf.collect_schema().names()
    date_col = _pick_from_names(names, "date")
    time_col = _pick_from_names(names, "time")
    datetime_col = _pick_from_names(names, "datetime")
    open_col = _pick_from_names(names, "open")
    high_col = _pick_from_names(names, "high")
    low_col = _pick_from_names(names, "low")
    close_col = _pick_from_names(names, "close")
    volume_col = _pick_from_names(names, "volume")
    amount_col = _pick_from_names(names, "amount")
    if not (open_col and high_col and low_col and close_col and ((date_col and time_col) or datetime_col)):
        raise ValueError("fast path could not identify required minute columns")

    if expected_date is not None and date_col:
        lf = lf.filter(pl.col(date_col).cast(pl.Utf8).str.slice(0, 10) == expected_date.isoformat())
    if datetime_col:
        datetime_expr = pl.col(datetime_col).cast(pl.Utf8).str.strip_chars()
    else:
        datetime_expr = (
            pl.col(date_col).cast(pl.Utf8).str.strip_chars()
            + pl.lit(" ")
            + pl.col(time_col).cast(pl.Utf8).str.strip_chars()
        )
    if needed_pairs is not None:
        # Filter stock-days before parsing timestamps or retaining history in memory.
        day_expr = pl.col(date_col).cast(pl.Utf8) if date_col else datetime_expr
        lf = lf.with_columns(
            _symbol_expr_from_path(),
            day_expr.str.slice(0, 10).alias("_trade_date"),
        ).join(needed_pairs.lazy(), on=["symbol", "_trade_date"], how="semi")
    lf = lf.select(
        [
            _symbol_expr_from_path(),
            datetime_expr.str.to_datetime(strict=False).cast(pl.Datetime("us")).alias("datetime"),
            pl.col(open_col).cast(pl.Float64, strict=False).alias("open"),
            pl.col(high_col).cast(pl.Float64, strict=False).alias("high"),
            pl.col(low_col).cast(pl.Float64, strict=False).alias("low"),
            pl.col(close_col).cast(pl.Float64, strict=False).alias("close"),
            (
                pl.col(volume_col).cast(pl.Float64, strict=False)
                if volume_col
                else pl.lit(None, dtype=pl.Float64)
            ).alias("volume"),
            (
                pl.col(amount_col).cast(pl.Float64, strict=False)
                if amount_col
                else pl.lit(None, dtype=pl.Float64)
            ).alias("amount"),
        ]
    )
    df = (
        lf.drop_nulls(subset=["symbol", "datetime", "open", "high", "low", "close"])
        .unique(subset=["symbol", "datetime"], keep="last", maintain_order=True)
        .sort(["symbol", "datetime"])
        .collect(engine="streaming")
    )
    if df.is_empty():
        raise ValueError(f"No valid minute rows found under {source}")
    read_files = int(df.get_column("symbol").n_unique())
    rows_before_unique = df.height
    meta = {
        "files_total": files_total,
        "read_files": read_files,
        "skipped_files": max(0, files_total - read_files),
        "rows_before_unique": rows_before_unique,
        "duplicate_rows": 0,
        "rows": df.height,
        "fast_path": True,
    }
    return df, meta


def _atomic_write(df: pl.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, out)


def _load_day(source: Path, expected_date: date | None, batch_size: int) -> tuple[pl.DataFrame, dict]:
    files = sorted(source.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv files found under {source}")
    try:
        return _load_day_fast(source, expected_date, len(files))
    except Exception as exc:  # noqa: BLE001
        print(f"WARN fast scan failed, fallback to per-file read: {exc}", file=sys.stderr)

    frames: list[pl.DataFrame] = []
    read_files = 0
    skipped_files = 0
    source_rows = 0
    started = time.perf_counter()
    for idx, path in enumerate(files, start=1):
        frame = _read_one(path, expected_date)
        if frame is None or frame.is_empty():
            skipped_files += 1
            continue
        read_files += 1
        source_rows += frame.height
        frames.append(frame)
        if idx % batch_size == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "stage": "read",
                        "files_seen": idx,
                        "files_total": len(files),
                        "read_files": read_files,
                        "rows": source_rows,
                        "elapsed_s": round(elapsed, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if not frames:
        raise ValueError(f"No valid minute rows found under {source}")

    df = pl.concat(frames, how="vertical_relaxed")
    before_unique = df.height
    df = df.unique(subset=["symbol", "datetime"], keep="last", maintain_order=True)
    df = df.sort(["symbol", "datetime"])
    meta = {
        "files_total": len(files),
        "read_files": read_files,
        "skipped_files": skipped_files,
        "rows_before_unique": before_unique,
        "duplicate_rows": before_unique - df.height,
        "rows": df.height,
    }
    return df, meta


def _validate(df: pl.DataFrame) -> dict:
    if df.is_empty():
        return {"ok": False, "reason": "empty"}
    high_floor = pl.max_horizontal(["open", "low", "close"])
    low_ceiling = pl.min_horizontal(["open", "high", "close"])
    invalid_ohlc = df.filter((pl.col("high") < high_floor) | (pl.col("low") > low_ceiling)).height
    non_positive_price = df.filter(
        (pl.col("open") <= 0) | (pl.col("high") <= 0) | (pl.col("low") <= 0) | (pl.col("close") <= 0)
    ).height
    count_stats = (
        df.group_by(["symbol", pl.col("datetime").dt.date().alias("date")])
        .len()
        .select(
            [
                pl.col("len").min().alias("min_bars_per_symbol_day"),
                pl.col("len").median().alias("median_bars_per_symbol_day"),
                pl.col("len").max().alias("max_bars_per_symbol_day"),
            ]
        )
        .to_dicts()[0]
    )
    return {
        "ok": invalid_ohlc == 0 and non_positive_price == 0,
        "invalid_ohlc": invalid_ohlc,
        "non_positive_price": non_positive_price,
        "symbols": int(df.get_column("symbol").n_unique()),
        "rows": df.height,
        "min_datetime": df.get_column("datetime").min(),
        "max_datetime": df.get_column("datetime").max(),
        **count_stats,
    }


def _drop_invalid_price_rows(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    if df.is_empty():
        return df, {"dropped_invalid_price_rows": 0, "invalid_price_samples": []}
    high_floor = pl.max_horizontal(["open", "low", "close"])
    low_ceiling = pl.min_horizontal(["open", "high", "close"])
    valid_expr = (
        (pl.col("open") > 0)
        & (pl.col("high") > 0)
        & (pl.col("low") > 0)
        & (pl.col("close") > 0)
        & (pl.col("high") >= high_floor)
        & (pl.col("low") <= low_ceiling)
    )
    invalid = df.filter(~valid_expr)
    if invalid.is_empty():
        return df, {"dropped_invalid_price_rows": 0, "invalid_price_samples": []}
    samples = invalid.head(5).to_dicts()
    return df.filter(valid_expr), {
        "dropped_invalid_price_rows": invalid.height,
        "invalid_price_samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Import extracted offline 1-minute CSV files into kline_minute.")
    parser.add_argument("--source", required=True, help="Directory containing extracted 1min/*.csv files.")
    parser.add_argument("--data-dir", default="/app/data", help="TickFlow data directory.")
    parser.add_argument("--expected-date", default="", help="Only import this trading date, YYYY-MM-DD.")
    parser.add_argument("--mode", choices=["replace", "merge"], default="replace")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    source = Path(args.source)
    if source.name != "1min" and (source / "1min").exists():
        source = source / "1min"
    expected = date.fromisoformat(args.expected_date) if args.expected_date else None

    started = time.perf_counter()
    df, meta = _load_day(source, expected, max(1, args.batch_size))
    df, clean_meta = _drop_invalid_price_rows(df)
    meta.update(clean_meta)
    validation = _validate(df)
    if not validation.get("ok"):
        print(json.dumps({"status": "failed", "validation": validation}, default=_json_default, ensure_ascii=False))
        return 2

    data_dir = Path(args.data_dir)
    minute_dir = data_dir / "kline_minute"
    written_rows = 0
    written_dates: list[str] = []
    partitioned = df.with_columns(pl.col("datetime").dt.date().alias("_trade_date"))
    for day_df in partitioned.partition_by("_trade_date", maintain_order=True):
        trade_date = day_df.get_column("_trade_date")[0]
        out = minute_dir / f"date={trade_date.isoformat()}" / "part.parquet"
        day_df = day_df.drop("_trade_date").select(CANONICAL_COLS)
        if args.mode == "merge" and out.exists():
            existing = pl.read_parquet(out).select(CANONICAL_COLS)
            day_df = (
                pl.concat([existing, day_df], how="vertical_relaxed")
                .unique(subset=["symbol", "datetime"], keep="last", maintain_order=True)
                .sort(["symbol", "datetime"])
            )
        _atomic_write(day_df, out)
        written_rows += day_df.height
        written_dates.append(trade_date.isoformat())

    elapsed = time.perf_counter() - started
    result = {
        "status": "ok",
        "source": str(source),
        "mode": args.mode,
        "expected_date": args.expected_date,
        "written_dates": written_dates,
        "written_rows": written_rows,
        "meta": meta,
        "validation": validation,
        "elapsed_s": round(elapsed, 2),
    }
    print(json.dumps(result, default=_json_default, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
