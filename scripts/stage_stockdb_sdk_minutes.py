"""Batch-extract missing stock-days from the locally running StockDB SDK.

This script is read-only with respect to the application's market data. It
stages raw source rows for a second, app-side validation/import pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import polars as pl


COLUMNS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]


def _codes(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("symbol").cast(pl.Utf8).str.extract(r"(\d{6})", 1).alias("code"))


def _candidates(
    root: Path,
    day: str,
    requested_codes: set[str] | None = None,
) -> tuple[list[str], dict[str, str], dict]:
    daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
    minute_path = root / "kline_minute" / f"date={day}" / "part.parquet"
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily reference missing: {daily_path}")
    daily = _codes(pl.read_parquet(daily_path))
    daily = daily.filter(
        pl.col("code").str.contains(r"^(00|30|60|68|43|83|87|88|92)")
        & pl.all_horizontal(pl.col(col).is_finite() & (pl.col(col) > 0)
                            for col in ("open", "high", "low", "close", "volume", "amount"))
    )
    daily = daily.filter(pl.len().over("code") == 1)
    if minute_path.exists():
        minute = _codes(pl.read_parquet(minute_path, columns=["symbol", "datetime"]))
        minute_time = pl.col("datetime").dt.strftime("%H:%M")
        minute = minute.filter(
            minute_time.is_between(pl.lit("09:31"), pl.lit("11:30"))
            | minute_time.is_between(pl.lit("13:01"), pl.lit("15:00"))
        )
        counts = minute.group_by("code").agg(pl.col("datetime").n_unique().alias("bars"))
    else:
        counts = pl.DataFrame(schema={"code": pl.Utf8, "bars": pl.UInt32})
    candidates = daily.join(counts, on="code", how="left")
    if requested_codes is None:
        candidates = candidates.filter(pl.col("bars").fill_null(0) < 240)
    else:
        candidates = candidates.filter(pl.col("code").is_in(sorted(requested_codes)))
    code_map = dict(candidates.select("code", "symbol").iter_rows())
    return sorted(code_map), code_map, {
        "daily_stock_days": daily.height,
        "already_full_stock_days": daily.height - len(code_map),
        "requested_stock_days": len(code_map),
    }


def stage_day(
    root: Path,
    output: Path,
    day: str,
    batch_size: int,
    sdk,
    requested_codes: set[str] | None = None,
) -> dict:
    codes, code_map, stats = _candidates(root, day, requested_codes)
    frames: list[pl.DataFrame] = []
    errors: list[dict] = []
    source_codes: set[str] = set()
    started = time.monotonic()

    def fetch(batch: list[str]) -> None:
        try:
            raw = sdk.get_data(batch, start=day.replace("-", ""), end=day.replace("-", ""),
                               frequency="1m", fq=None, as_df=True)
        except Exception as exc:
            if len(batch) > 1:
                half = len(batch) // 2
                fetch(batch[:half])
                fetch(batch[half:])
                return
            errors.append({"code": batch[0], "error": str(exc)[:240]})
            return
        if raw is None or raw.empty:
            # StockDB SDK may return an empty frame for an oversized batch
            # without raising. Split it so a source-side batch limit cannot be
            # mistaken for missing historical data.
            if len(batch) > 1:
                half = len(batch) // 2
                fetch(batch[:half])
                fetch(batch[half:])
            return
        frame = pl.from_pandas(raw)
        frame = frame.with_columns(
            pl.col("code").cast(pl.Utf8).str.extract(r"(\d{6})", 1).alias("code"),
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Datetime("us"), "%Y%m%d%H%M%S", strict=False).alias("datetime"),
        ).filter(pl.col("code").is_in(batch) & (pl.col("datetime").dt.date() == date.fromisoformat(day)))
        if frame.is_empty():
            return
        source_codes.update(frame["code"].unique().to_list())
        frame = frame.with_columns(
            pl.col("code").replace_strict(code_map).alias("symbol"),
            (pl.col("volume").cast(pl.Float64, strict=False) / 100.0).alias("volume"),
            pl.col("amount").cast(pl.Float64, strict=False).alias("amount"),
            *[pl.col(col).cast(pl.Float64, strict=False).alias(col)
              for col in ("open", "high", "low", "close")],
        ).select(COLUMNS)
        frames.append(frame)

    for index in range(0, len(codes), batch_size):
        fetch(codes[index:index + batch_size])
        print(json.dumps({"date": day, "checked": min(index + batch_size, len(codes)),
                          "total": len(codes), "source_codes": len(source_codes),
                          "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    if frames:
        staged = pl.concat(frames, rechunk=True).sort(["symbol", "datetime"])
        path = output / f"{day}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        staged.write_parquet(temporary, compression="zstd")
        os.replace(temporary, path)
        if pl.read_parquet(path).height != staged.height:
            raise RuntimeError(f"Staging readback failed: {path}")
    else:
        staged = pl.DataFrame(schema={col: pl.Utf8 for col in COLUMNS})
        path = None
    return {"date": day, **stats, "source_codes": len(source_codes),
            "source_rows": staged.height, "missing_source_codes": len(codes) - len(source_codes),
            "source_error_count": len(errors), "source_errors": errors[:100],
            "file": path.name if path else None, "elapsed_s": round(time.monotonic() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--conflict-manifest",
        type=Path,
        help="CSV from audit_market_minute_coverage.py; stage only complete_conflict stock-days.",
    )
    parser.add_argument("dates", nargs="+", help="Closed dates in YYYY-MM-DD format")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 500:
        raise ValueError("batch size must be 1..500")
    root = args.data_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.sdk_dir.resolve()))
    from stock_sdk import rd

    requested_by_day: dict[str, set[str]] = {}
    if args.conflict_manifest:
        manifest = pl.read_csv(args.conflict_manifest.resolve())
        required = {"date", "code", "status"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"conflict manifest requires columns: {sorted(required)}")
        manifest = manifest.filter(pl.col("status") == "complete_conflict")
        requested_by_day = {}
        for key, group in manifest.group_by("date"):
            day = key[0] if isinstance(key, tuple) else key
            requested_by_day[str(day)] = {
                str(code).zfill(6) for code in group["code"].to_list()
            }

    reports = []
    for day in args.dates:
        date.fromisoformat(day)
        report = stage_day(
            root,
            output,
            day,
            args.batch_size,
            rd,
            requested_codes=requested_by_day.get(day) if args.conflict_manifest else None,
        )
        reports.append(report)
        (output / "stage-report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
