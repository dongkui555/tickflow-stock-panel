from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl


CANONICAL_COLS = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]


@dataclass(frozen=True)
class ZipParquetEntry:
    zip_path: Path
    member_name: str
    trade_date: date
    file_size: int

    @property
    def label(self) -> str:
        return f"{self.zip_path.name}!{self.member_name}"


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _date_from_name(name: str) -> date | None:
    stem = Path(name).stem
    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", stem)
    if not match:
        return None
    year, month, day = match.groups()
    return date(int(year), int(month), int(day))


def _iter_entries(source: Path) -> list[ZipParquetEntry]:
    zip_paths = [source] if source.is_file() else sorted(source.rglob("*.zip"))
    entries: list[ZipParquetEntry] = []
    for zip_path in zip_paths:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".parquet"):
                        continue
                    trade_date = _date_from_name(info.filename)
                    if trade_date is None:
                        continue
                    entries.append(
                        ZipParquetEntry(
                            zip_path=zip_path,
                            member_name=info.filename,
                            trade_date=trade_date,
                            file_size=info.file_size,
                        )
                    )
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Bad zip file: {zip_path}") from exc
    return sorted(entries, key=lambda e: (e.trade_date, e.zip_path.name, e.member_name))


def _symbol_expr(col: str) -> pl.Expr:
    raw = pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    code = raw.str.extract(r"(\d{6})", 1)
    suffix = raw.str.extract(r"\.(SH|SZ|BJ)$", 1)
    prefix = raw.str.extract(r"^(SH|SZ|BJ)", 1)
    structural = (
        pl.when(
            code.str.starts_with("43")
            | code.str.starts_with("83")
            | code.str.starts_with("87")
            | code.str.starts_with("88")
            | code.str.starts_with("92")
        )
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


def _first_existing(names: dict[str, str], *keys: str) -> str | None:
    return next((names[key] for key in keys if key in names), None)


def _datetime_expr(schema: dict[str, pl.DataType], names: dict[str, str], datetime_col: str) -> pl.Expr:
    dt_col = pl.col(datetime_col)
    if schema[datetime_col] == pl.Utf8:
        return dt_col.str.to_datetime(strict=False)
    return dt_col.cast(pl.Datetime("us"), strict=False)


def _read_entry(entry: ZipParquetEntry, *, volume_unit: str = "auto", markets: set[str] | None = None) -> tuple[pl.DataFrame, dict[str, Any]]:
    with zipfile.ZipFile(entry.zip_path) as zf:
        raw_bytes = io.BytesIO(zf.read(entry.member_name))
    raw = pl.read_parquet(raw_bytes)
    schema = raw.schema
    names = {name.lower(): name for name in schema}

    symbol_col = _first_existing(names, "symbol", "ts_code", "code")
    datetime_col = _first_existing(names, "datetime", "trade_time", "time")
    if not symbol_col:
        raise ValueError("missing symbol/code column")
    if not datetime_col:
        raise ValueError("missing datetime/trade_time column")

    open_col = _first_existing(names, "open")
    high_col = _first_existing(names, "high")
    low_col = _first_existing(names, "low")
    close_col = _first_existing(names, "close")
    if not all([open_col, high_col, low_col, close_col]):
        raise ValueError("missing required OHLC columns")
    volume_col = _first_existing(names, "volume", "vol")
    amount_col = _first_existing(names, "amount", "amt")

    df = (
        raw.lazy()
        .select(
            [
                _symbol_expr(symbol_col),
                _datetime_expr(schema, names, datetime_col).cast(pl.Datetime("us")).alias("datetime"),
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
        .filter(pl.lit(markets is None) | pl.col("symbol").str.extract(r"\.(SH|SZ|BJ)$", 1).is_in(sorted(markets or [])))
        .with_columns(
            [
                pl.col("datetime").dt.date().alias("_trade_date"),
                pl.col("datetime").dt.strftime("%H:%M").alias("_hm"),
            ]
        )
        .filter(
            (pl.col("_trade_date") == entry.trade_date)
            & (
                ((pl.col("_hm") >= "09:30") & (pl.col("_hm") <= "11:30"))
                | ((pl.col("_hm") >= "13:01") & (pl.col("_hm") <= "15:00"))
            )
        )
        .drop(["_trade_date", "_hm"])
        .collect(engine="streaming")
    )

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
    detected_volume_unit = "unknown"
    if "volume" in df.columns and "amount" in df.columns and not df.is_empty():
        unit_sample = df.filter((pl.col("volume") > 0) & (pl.col("amount") > 0) & (pl.col("close") > 0))
        if not unit_sample.is_empty():
            unit_ratio = unit_sample.select((pl.col("amount") / (pl.col("close") * pl.col("volume"))).median()).item()
            if volume_unit == "shares":
                detected_volume_unit = "shares"
            elif volume_unit == "lots":
                detected_volume_unit = "lots"
            elif unit_ratio is not None and 0.5 <= float(unit_ratio) <= 2.0:
                df = df.with_columns((pl.col("volume") / 100.0).alias("volume"))
                detected_volume_unit = "shares_to_lots"
            elif unit_ratio is not None and 50 <= float(unit_ratio) <= 150:
                detected_volume_unit = "lots"

    before_unique = df.height
    df = df.unique(subset=["symbol", "datetime"], keep="last", maintain_order=True).sort(["symbol", "datetime"])
    return df.select(CANONICAL_COLS), {
        "source_rows": raw.height,
        "filtered_rows": before_invalid,
        "dropped_invalid_rows": int(invalid.height),
        "duplicate_rows": before_unique - df.height,
        "volume_unit": detected_volume_unit,
        "amount_close_volume_median": round(float(unit_ratio), 6) if unit_ratio is not None else None,
        "invalid_samples": invalid.head(3).to_dicts() if not invalid.is_empty() else [],
    }


def _atomic_write(df: pl.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, out)


def _verified_missing_rows(
    incoming: pl.DataFrame, old: pl.DataFrame, daily: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    incoming = incoming.with_columns(pl.col("symbol").str.extract(r"(\d{6})", 1).alias("_code"))
    old = old.with_columns(pl.col("symbol").str.extract(r"(\d{6})", 1).alias("_code"))
    daily = daily.with_columns(pl.col("symbol").str.extract(r"(\d{6})", 1).alias("_code"))
    daily = daily.filter(pl.col("_code").is_not_null())
    missing = incoming.join(old.select("_code", "datetime"), on=["_code", "datetime"], how="anti")
    meta: dict[str, Any] = {
        "existing_keys_preserved": incoming.height - missing.height,
        "missing_keys": missing.height, "added_rows": 0, "excluded_rows": 0,
        "symbol_checks": [],
    }
    if missing.is_empty():
        return missing, meta

    bars = missing.sort(["_code", "datetime"]).group_by("_code").agg(
        pl.col("symbol").first().alias("symbol"),
        pl.len().alias("rows"),
        pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
        pl.col("close").last(), pl.col("volume").sum(), pl.col("amount").sum(),
        pl.all_horizontal([pl.col(c).is_finite() for c in CANONICAL_COLS[2:]])
        .fill_null(False).all().alias("finite"),
        ((pl.col("volume") >= 0) & (pl.col("amount") >= 0)).fill_null(False).all().alias("nonnegative"),
        pl.col("datetime").filter(
            (pl.col("datetime").dt.second() == 0)
            & (pl.col("datetime").dt.microsecond() == 0)
            & (
                pl.col("datetime").dt.strftime("%H:%M").is_between(pl.lit("09:31"), pl.lit("11:30"))
                | pl.col("datetime").dt.strftime("%H:%M").is_between(pl.lit("13:01"), pl.lit("15:00"))
            )
        ).dt.strftime("%H:%M").n_unique().alias("session_minutes"),
    )
    if daily["_code"].n_unique() != daily.height:
        raise ValueError("Daily reference has duplicate stock codes; refusing verified fill")
    checks = bars.join(
        daily.select("_code", *CANONICAL_COLS[2:]), on="_code", how="left", suffix="_daily",
    )
    existing_codes = set(old["_code"].unique().to_list())
    accepted_symbols = []
    for row in checks.iter_rows(named=True):
        reason = "accepted"
        if row["_code"] in existing_codes:
            reason = "partial_existing_symbol_requires_source_alignment"
        elif not row["finite"] or not row["nonnegative"]:
            reason = "invalid_numeric_values"
        elif row["volume"] <= 0 or row["amount"] <= 0:
            reason = "zero_trade_day"
        elif row["rows"] not in (240, 241) or row["session_minutes"] != 240:
            reason = "incomplete_minute_day"
        elif any(row[f"{c}_daily"] is None for c in CANONICAL_COLS[2:]):
            reason = "no_daily_reference"
        elif any(
            not abs(row[c] - row[f"{c}_daily"]) <= max(0.011, abs(row[f"{c}_daily"]) * 1e-6)
            for c in ["open", "high", "low", "close"]
        ):
            reason = "daily_price_mismatch"
        elif any(
            not abs(row[c] - row[f"{c}_daily"]) <= max(1.0, abs(row[f"{c}_daily"]) * 0.001)
            for c in ["volume", "amount"]
        ):
            reason = "daily_turnover_mismatch"
        meta["symbol_checks"].append({"symbol": row["symbol"], "rows": row["rows"], "reason": reason})
        if reason == "accepted":
            accepted_symbols.append(row["symbol"])
    accepted = missing.filter(pl.col("symbol").is_in(accepted_symbols))
    meta["added_rows"] = accepted.height
    meta["excluded_rows"] = missing.height - accepted.height
    return accepted.select(CANONICAL_COLS), meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Import zipped daily minute parquet files into TickFlow kline_minute.")
    parser.add_argument("--source-root", required=True, help="Directory containing *.zip files, or one zip file.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mode", choices=["fill", "replace", "merge", "verified-fill"], default="fill")
    parser.add_argument("--report", default="", help="Write an import audit JSON to a new file.")
    parser.add_argument("--min-full-rows", type=int, default=1_000_000)
    parser.add_argument("--min-full-symbols", type=int, default=4000)
    parser.add_argument("--start-date", default="", help="Only import files on/after this date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="", help="Only import files on/before this date, YYYY-MM-DD.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--markets", default="", help="Optional comma-separated exchange suffixes, e.g. BJ,SH.")
    parser.add_argument("--source-volume-unit", choices=["auto", "shares", "lots"], default="auto")
    args = parser.parse_args()
    markets = {part.strip().upper() for part in args.markets.split(",") if part.strip()} or None
    if markets is not None and not markets <= {"BJ", "SH", "SZ"}:
        parser.error("--markets only accepts BJ,SH,SZ")

    source = Path(args.source_root)
    data_dir = Path(args.data_dir)
    minute_dir = data_dir / "kline_minute"
    report = Path(args.report).resolve() if args.report else None
    if report is not None and report.exists():
        raise ValueError(f"Report already exists: {report}")
    start_date = date.fromisoformat(args.start_date) if args.start_date else None
    end_date = date.fromisoformat(args.end_date) if args.end_date else None
    entries = [
        e for e in _iter_entries(source)
        if (start_date is None or e.trade_date >= start_date)
        and (end_date is None or e.trade_date <= end_date)
    ]
    if args.limit > 0:
        entries = entries[: args.limit]
    if not entries:
        print(json.dumps({"status": "failed", "reason": "no parquet entries found"}, ensure_ascii=False))
        return 2

    started = time.perf_counter()
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    written_rows = 0
    total_source_rows = 0
    dropped_invalid_rows = 0

    for idx, entry in enumerate(entries, start=1):
        out = minute_dir / f"date={entry.trade_date.isoformat()}" / "part.parquet"
        existing = _existing_status(out)
        if args.mode == "fill" and _is_full(existing, args.min_full_rows, args.min_full_symbols):
            skipped.append({"entry": entry.label, "date": entry.trade_date.isoformat(), "reason": "target_full", **existing})
            continue
        try:
            df, meta = _read_entry(entry, volume_unit=args.source_volume_unit, markets=markets)
            total_source_rows += int(meta["source_rows"])
            dropped_invalid_rows += int(meta["dropped_invalid_rows"])
            if df.is_empty():
                skipped.append({"entry": entry.label, "date": entry.trade_date.isoformat(), "reason": "empty_after_filter", **meta})
                continue
            if args.mode == "verified-fill":
                old = pl.read_parquet(out).select(CANONICAL_COLS) if out.exists() else df.clear()
                daily_path = data_dir / "kline_daily" / f"date={entry.trade_date}" / "part.parquet"
                daily = pl.read_parquet(daily_path) if daily_path.exists() else df.clear()
                additions, audit = _verified_missing_rows(df, old, daily)
                meta.update(audit)
                if additions.is_empty():
                    skipped.append({
                        "entry": entry.label, "date": entry.trade_date.isoformat(),
                        "reason": "no_verified_new_records", **meta,
                    })
                    if idx % 10 == 0 or idx == len(entries):
                        print(json.dumps({"stage": "verified_fill", "seen": idx, "total": len(entries),
                                          "imported": len(imported), "skipped": len(skipped)}), flush=True)
                    continue
                df = pl.concat([old, additions], how="vertical_relaxed").sort(["symbol", "datetime"])
                if out.exists():
                    backup = data_dir / "import_backups" / f"minute_{time.time_ns()}" / out.parent.name / out.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(out, backup)
                    meta["backup"] = str(backup)
            if args.mode == "merge" and out.exists():
                old = pl.read_parquet(out).select(CANONICAL_COLS)
                df = pl.concat([old, df], how="vertical_relaxed").unique(
                    subset=["symbol", "datetime"], keep="last", maintain_order=True
                ).sort(["symbol", "datetime"])
            _atomic_write(df, out)
            if args.mode == "verified-fill":
                persisted = pl.read_parquet(out).select(CANONICAL_COLS)
                if not persisted.equals(df) or persisted.unique(subset=["symbol", "datetime"]).height != persisted.height:
                    raise ValueError("Post-write readback or unique-key validation failed")
                if not persisted.join(old.select("symbol", "datetime"), on=["symbol", "datetime"], how="semi").sort(
                    ["symbol", "datetime"]
                ).equals(old.sort(["symbol", "datetime"])):
                    raise ValueError("Existing records changed during verified fill")
                meta["readback_verified"] = True
            stat = {
                "entry": entry.label,
                "date": entry.trade_date.isoformat(),
                "rows": df.height,
                "symbols": int(df["symbol"].n_unique()),
                **meta,
            }
            imported.append(stat)
            written_rows += int(meta.get("added_rows", df.height))
        except Exception as exc:  # noqa: BLE001
            failed.append({"entry": entry.label, "date": entry.trade_date.isoformat(), "error": f"{type(exc).__name__}: {exc}"})

        if idx % 10 == 0 or idx == len(entries):
            print(
                json.dumps(
                    {
                        "stage": "progress",
                        "seen": idx,
                        "total": len(entries),
                        "imported": len(imported),
                        "skipped": len(skipped),
                        "failed": len(failed),
                        "elapsed_s": round(time.perf_counter() - started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    result = {
        "status": "ok" if imported or skipped else "failed",
        "mode": args.mode,
        "source_root": str(source),
        "entries_total": len(entries),
        "imported_files": len(imported),
        "skipped_files": len(skipped),
        "failed_files": len(failed),
        "written_rows": written_rows,
        "source_rows_read": total_source_rows,
        "dropped_invalid_rows": dropped_invalid_rows,
        "failed": failed,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    if args.mode == "verified-fill":
        result["added_rows"] = sum(row.get("added_rows", 0) for row in imported)
        result["excluded_rows"] = sum(row.get("excluded_rows", 0) for row in imported + skipped)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8") as fh:
            json.dump({**result, "imported": imported, "skipped": skipped}, fh, default=_json_default, ensure_ascii=False, indent=2)
    if args.summary_only:
        for label, records in [("imported_sample", imported), ("skipped_sample", skipped)]:
            sample = records[:3] + records[-3:] if len(records) > 6 else records
            result[label] = [{key: value for key, value in row.items() if key != "symbol_checks"} for row in sample]
    else:
        result["imported"] = imported
        result["skipped"] = skipped
    print(json.dumps(result, default=_json_default, ensure_ascii=False), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
