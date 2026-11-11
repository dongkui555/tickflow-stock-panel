from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl


CANONICAL_COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _symbol_expr(col: str) -> pl.Expr:
    raw = pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    code = raw.str.extract(r"(\d{6})", 1)
    suffix = raw.str.extract(r"\.(SH|SZ|BJ)$", 1)
    prefix = raw.str.extract(r"^(SH|SZ|BJ)", 1)
    structural = (
        pl.when(code.str.starts_with("43") | code.str.starts_with("83") | code.str.starts_with("87") | code.str.starts_with("88") | code.str.starts_with("92"))
        .then(pl.lit("BJ"))
        .when(code.str.starts_with("6"))
        .then(pl.lit("SH"))
        .when(code.str.starts_with("0") | code.str.starts_with("3"))
        .then(pl.lit("SZ"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )
    declared = pl.coalesce([
        suffix,
        pl.when(prefix.is_in(["SH", "SZ", "BJ"])).then(prefix),
    ])
    return (
        pl.when(code.is_null() | structural.is_null() | (declared.is_not_null() & (declared != structural)))
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(code + pl.lit(".") + structural)
        .alias("symbol")
    )


def _existing_status(out: Path) -> dict[str, Any]:
    if not out.exists():
        return {"exists": False, "rows": 0, "symbols": 0}
    try:
        stat = (
            pl.scan_parquet(str(out))
            .select(pl.len().alias("rows"), pl.col("symbol").n_unique().alias("symbols"))
            .collect()
        )
        return {
            "exists": True,
            "rows": int(stat["rows"][0]),
            "symbols": int(stat["symbols"][0]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "rows": -1, "symbols": -1, "error": str(exc)[:200]}


def _is_full(status: dict[str, Any], min_rows: int, min_symbols: int) -> bool:
    return bool(status.get("exists")) and int(status.get("rows") or 0) >= min_rows and int(status.get("symbols") or 0) >= min_symbols


def _read_source(path: Path, expected: date) -> tuple[pl.DataFrame, dict[str, Any]]:
    schema = pl.read_parquet_schema(str(path))
    names = {name.lower(): name for name in schema}
    symbol_col = names.get("symbol") or names.get("ts_code") or names.get("code")
    datetime_col = names.get("datetime") or names.get("trade_time") or names.get("time")
    if not symbol_col:
        raise ValueError("missing symbol/code column")
    if not datetime_col:
        raise ValueError("missing datetime/trade_time column")

    column = lambda *keys: next((names[k] for k in keys if k in names), None)
    open_col = column("open")
    high_col = column("high")
    low_col = column("low")
    close_col = column("close")
    if not all([open_col, high_col, low_col, close_col]):
        raise ValueError("missing required OHLC columns")
    volume_col = column("volume", "vol")
    amount_col = column("amount", "amt")

    dt_expr = pl.col(datetime_col)
    if schema[datetime_col] == pl.Utf8:
        dt_expr = dt_expr.str.to_datetime(strict=False)
    else:
        dt_expr = dt_expr.cast(pl.Datetime("us"), strict=False)

    lf = pl.scan_parquet(str(path))
    df = (
        lf.select(
            [
                _symbol_expr(symbol_col),
                dt_expr.cast(pl.Datetime("us")).alias("datetime"),
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
        .drop_nulls(subset=["symbol", "datetime", "open", "high", "low", "close"])
        .with_columns(
            [
                pl.col("datetime").dt.date().alias("_trade_date"),
                pl.col("datetime").dt.strftime("%H:%M").alias("_hm"),
            ]
        )
        .filter(
            (pl.col("_trade_date") == expected)
            & (
                ((pl.col("_hm") >= "09:30") & (pl.col("_hm") <= "11:30"))
                | ((pl.col("_hm") >= "13:01") & (pl.col("_hm") <= "15:00"))
            )
        )
        .drop(["_trade_date", "_hm"])
        .collect(engine="streaming")
    )
    source_rows = int(pl.scan_parquet(str(path)).select(pl.len()).collect().item())
    before_invalid = df.height
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
    if not invalid.is_empty():
        df = df.filter(valid_expr)
    unit_ratio = None
    volume_unit = "unknown"
    if "volume" in df.columns and "amount" in df.columns and not df.is_empty():
        unit_sample = df.filter((pl.col("volume") > 0) & (pl.col("amount") > 0) & (pl.col("close") > 0))
        if not unit_sample.is_empty():
            unit_ratio = unit_sample.select((pl.col("amount") / (pl.col("close") * pl.col("volume"))).median()).item()
            if unit_ratio is not None and 0.5 <= float(unit_ratio) <= 2.0:
                df = df.with_columns((pl.col("volume") / 100.0).alias("volume"))
                volume_unit = "shares_to_lots"
            elif unit_ratio is not None and 50 <= float(unit_ratio) <= 150:
                volume_unit = "lots"
    before_unique = df.height
    df = df.unique(subset=["symbol", "datetime"], keep="last", maintain_order=True).sort(["symbol", "datetime"])
    return df.select(CANONICAL_COLS), {
        "source_rows": source_rows,
        "filtered_rows": before_invalid,
        "dropped_invalid_rows": int(invalid.height),
        "duplicate_rows": before_unique - df.height,
        "volume_unit": volume_unit,
        "amount_close_volume_median": round(float(unit_ratio), 6) if unit_ratio is not None else None,
        "invalid_samples": invalid.head(3).to_dicts() if not invalid.is_empty() else [],
    }


def _atomic_write(df: pl.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, out)


def _date_from_stem(path: Path) -> date | None:
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", path.stem)
    if not match:
        return None
    year, month, day = match.groups()
    return date(int(year), int(month), int(day))


def _iter_files(source: Path) -> list[Path]:
    return sorted(
        p for p in source.rglob("*.parquet")
        if "__minute_import" not in str(p) and _date_from_stem(p) is not None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Import offline daily minute parquet files into TickFlow kline_minute.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-dir", default="/app/data")
    parser.add_argument("--mode", choices=["fill", "replace", "merge"], default="fill")
    parser.add_argument("--min-full-rows", type=int, default=1_000_000)
    parser.add_argument("--min-full-symbols", type=int, default=4000)
    parser.add_argument("--start-date", default="", help="Only import files on/after this date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="", help="Only import files on/before this date, YYYY-MM-DD.")
    parser.add_argument("--summary-only", action="store_true", help="Print compact final output.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source_root)
    data_dir = Path(args.data_dir)
    minute_dir = data_dir / "kline_minute"
    files = _iter_files(source)
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None
    if start_date or end_date:
        files = [
            p for p in files
            if (start_date is None or (_date_from_stem(p) or date.min) >= start_date)
            and (end_date is None or (_date_from_stem(p) or date.max) <= end_date)
        ]
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(json.dumps({"status": "failed", "reason": "no parquet files found"}, ensure_ascii=False))
        return 2

    started = time.perf_counter()
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    written_rows = 0
    total_source_rows = 0
    dropped_invalid_rows = 0

    for idx, path in enumerate(files, start=1):
        expected = _date_from_stem(path)
        assert expected is not None
        out = minute_dir / f"date={expected.isoformat()}" / "part.parquet"
        existing = _existing_status(out)
        if args.mode == "fill" and _is_full(existing, args.min_full_rows, args.min_full_symbols):
            skipped.append({"file": path.name, "date": expected.isoformat(), "reason": "target_full", **existing})
            continue
        try:
            df, meta = _read_source(path, expected)
            total_source_rows += int(meta["source_rows"])
            dropped_invalid_rows += int(meta["dropped_invalid_rows"])
            if df.is_empty():
                skipped.append({"file": path.name, "date": expected.isoformat(), "reason": "empty_after_filter", **meta})
                continue
            if args.mode == "merge" and out.exists():
                old = pl.read_parquet(out).select(CANONICAL_COLS)
                df = pl.concat([old, df], how="vertical_relaxed").unique(
                    subset=["symbol", "datetime"], keep="last", maintain_order=True
                ).sort(["symbol", "datetime"])
            _atomic_write(df, out)
            stat = {
                "file": path.name,
                "date": expected.isoformat(),
                "rows": df.height,
                "symbols": int(df["symbol"].n_unique()),
                **meta,
            }
            imported.append(stat)
            written_rows += df.height
        except Exception as exc:  # noqa: BLE001
            failed.append({"file": path.name, "date": expected.isoformat(), "error": f"{type(exc).__name__}: {exc}"})
        if idx % 10 == 0 or idx == len(files):
            progress = {
                "stage": "progress",
                "seen": idx,
                "total": len(files),
                "imported": len(imported),
                "skipped": len(skipped),
                "failed": len(failed),
                "elapsed_s": round(time.perf_counter() - started, 2),
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)

    result = {
        "status": "ok" if imported or skipped else "failed",
        "mode": args.mode,
        "source_root": str(source),
        "files_total": len(files),
        "imported_files": len(imported),
        "skipped_files": len(skipped),
        "failed_files": len(failed),
        "written_rows": written_rows,
        "source_rows_read": total_source_rows,
        "dropped_invalid_rows": dropped_invalid_rows,
        "failed": failed,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    if args.summary_only:
        result["imported_sample"] = imported[:3] + imported[-3:] if len(imported) > 6 else imported
        result["skipped_sample"] = skipped[:3] + skipped[-3:] if len(skipped) > 6 else skipped
    else:
        result["imported"] = imported
        result["skipped"] = skipped
    print(json.dumps(result, default=_json_default, ensure_ascii=False), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
