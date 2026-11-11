from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")

import polars as pl
from check_minute_coverage import _partition_map
from audit_history_semantics import audit_symbol_expr as _symbol_expr

PRICE = ["open", "high", "low", "close"]
NUMERIC = [*PRICE, "volume", "amount"]
STOCK_PATTERN = r"^(?:(?:60|68)\d{4}\.SH|(?:00|30)\d{4}\.SZ|(?:4\d{5}|8\d{5}|92\d{4})\.BJ)$"


def summarize_minutes(frame: pl.DataFrame, day: date) -> pl.DataFrame:
    stamp = pl.col("datetime")
    clock = stamp.dt.hour().cast(pl.Int32) * 60 + stamp.dt.minute().cast(pl.Int32)
    aligned = (
        (stamp.dt.date() == day) & (stamp.dt.second() == 0)
        & (stamp.dt.microsecond() == 0)
    ).fill_null(False)
    regular = aligned & (clock.is_between(571, 690) | clock.is_between(781, 900))
    valid_time = regular | (aligned & (clock == 570))
    price_ok = pl.all_horizontal([
        (pl.col(c).is_finite() & (pl.col(c) > 0)) for c in PRICE
    ]) & (pl.col("high") >= pl.max_horizontal(PRICE)) & (
        pl.col("low") <= pl.min_horizontal(PRICE)
    )
    turnover_ok = pl.all_horizontal([
        pl.col(c).is_finite() & (pl.col(c) >= 0) for c in ["volume", "amount"]
    ])
    return frame.sort(["symbol", "datetime"]).group_by("symbol").agg(
        pl.len().alias("rows"),
        (pl.len() - stamp.n_unique()).alias("duplicate_rows"),
        stamp.filter(regular).n_unique().alias("session_minutes"),
        (~valid_time).sum().alias("bad_time_rows"),
        (~price_ok.fill_null(False)).sum().alias("bad_price_rows"),
        (~turnover_ok.fill_null(False)).sum().alias("bad_turnover_rows"),
        stamp.min().alias("first"), stamp.max().alias("last"),
        pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
        pl.col("close").last(), pl.col("volume").sum(), pl.col("amount").sum(),
    )


def compare_daily(summary: pl.DataFrame, daily: pl.DataFrame, *, data_dir=None) -> tuple[pl.DataFrame, dict]:
    minute_reference = summary.with_columns(
        pl.col("symbol").str.contains(STOCK_PATTERN).fill_null(False).alias("_native"),
        _symbol_expr("symbol"),
    ).filter(pl.col("symbol").str.contains(STOCK_PATTERN).fill_null(False))
    minute_reference = minute_reference.filter(
        pl.col("_native") == pl.col("_native").max().over("symbol")
    )
    ambiguous_minute = minute_reference.group_by("symbol").len().filter(pl.col("len") > 1)
    minute_reference = minute_reference.join(ambiguous_minute.select("symbol"), on="symbol", how="anti").drop("_native")
    native_count = daily.filter(pl.col("symbol").str.contains(STOCK_PATTERN).fill_null(False)).height
    canonical = daily.with_columns(
        pl.col("symbol").str.contains(STOCK_PATTERN).fill_null(False).alias("_native"),
        _symbol_expr("symbol"),
    ).filter(pl.col("symbol").str.contains(STOCK_PATTERN).fill_null(False))
    # Legacy daily files use bare codes. Normalize only this audit's join keys;
    # retain explicit canonical records over aliases without changing stored data.
    preferred = canonical.filter(pl.col("_native") == pl.col("_native").max().over("symbol"))
    duplicate_symbols = preferred.group_by("symbol").len().filter(pl.col("len") > 1)
    reference = preferred.join(duplicate_symbols.select("symbol"), on="symbol", how="anti")
    reference = reference.filter(pl.all_horizontal([
        pl.col(c).is_finite() & (pl.col(c) > 0) for c in NUMERIC
    ]))
    adjusted = excluded = 0
    if data_dir is not None and (Path(data_dir) / 'minute_reference_evidence').is_dir():
        from app.services.minute_reference import apply_reference_evidence
        refs = {r['symbol'][:6]: r for r in reference.iter_rows(named=True)}
        reviewed = apply_reference_evidence(refs, data_dir)
        adjusted = sum(r.get('_volume_scope') == 'regular_session' for r in reviewed.values())
        excluded = len(refs) - len(reviewed)
        if adjusted or excluded:
            reference = pl.DataFrame([{c: r[c] for c in ['symbol', *NUMERIC]} for r in reviewed.values()]) if reviewed else reference.clear()
    joined = reference.select("symbol", *NUMERIC).join(
        minute_reference, on="symbol", how="left", suffix="_minute",
    )
    joined = joined.with_columns(
        pl.col("rows").is_null().alias("missing"),
        (pl.col("session_minutes").fill_null(0) < 240).alias("incomplete"),
        pl.any_horizontal([
            (pl.col(c) - pl.col(c + "_minute")).abs()
            > pl.max_horizontal(pl.lit(0.011), pl.col(c).abs() * 1e-6)
            for c in PRICE
        ]).fill_null(False).alias("price_mismatch"),
        (
            ((pl.col("amount") - pl.col("amount_minute")).abs()
             > pl.max_horizontal(pl.lit(1.0), pl.col("amount").abs() * 0.002))
            | (
                ((pl.col("volume") - pl.col("volume_minute") * 100).abs()
                 > pl.max_horizontal(pl.lit(1.0), pl.col("volume").abs() * 0.002))
                & ((pl.col("volume") - pl.col("volume_minute")).abs()
                   > pl.max_horizontal(pl.lit(1.0), pl.col("volume").abs() * 0.002))
            )
        ).fill_null(False).alias("turnover_mismatch"),
    )
    return joined, {
        "daily_rows": daily.height,
        "ambiguous_minute_symbols": ambiguous_minute.height,
        "daily_noncanonical_rows": daily.height - native_count,
        "daily_unresolved_rows": daily.height - canonical.height,
        "daily_duplicate_symbols": duplicate_symbols.height,
        "daily_excluded_inactive_or_invalid": canonical.height - reference.height,
        "expected_active_symbols": reference.height,
        "regular_reference_adjusted": adjusted,
        "regular_reference_excluded": excluded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only minute coverage and integrity audit.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", required=True, help="New audit-only output directory.")
    parser.add_argument("--dates-file", help="Optional CSV manifest; audit only its distinct dates.")
    args = parser.parse_args()
    root = Path(args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start, end = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    now = datetime.now(timezone(timedelta(hours=8)))
    maps = {key: _partition_map(root / key) for key in [
        "kline_daily", "kline_minute", "kline_etf_daily", "kline_index_daily",
    ]}
    days = sorted(d for d in set().union(*(set(m) for m in maps.values())) if start <= d <= end)
    if args.dates_file:
        selected = {date.fromisoformat(d) for d in pl.read_csv(args.dates_file)["date"].unique()}
        days = sorted(d for d in selected if start <= d <= end)
    summaries, failures = [], []
    began = time.perf_counter()
    with (output / "stock_day_issues.csv").open("x", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "symbol", "issue", "rows", "session_minutes"])
        for index, day in enumerate(days, 1):
            item = {"date": str(day), "in_progress": day == now.date() and now.hour < 15}
            path = maps["kline_minute"].get(day)
            try:
                if path is None:
                    item["status"] = "missing_partition"
                    summaries.append(item)
                    continue
                before = path.stat()
                frame = pl.read_parquet(path, columns=["symbol", "datetime", *NUMERIC])
                after = path.stat()
                item["changed_during_read"] = (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)
                item["timezone"] = str(frame.schema["datetime"])
                summary = summarize_minutes(frame, day)
                item.update({
                    "status": "readable", "rows": frame.height,
                    "symbols": summary.height,
                    "first": str(frame["datetime"].min()), "last": str(frame["datetime"].max()),
                    "full_session_symbols": summary.filter(pl.col("session_minutes") == 240).height,
                    "partial_session_symbols": summary.filter(pl.col("session_minutes") < 240).height,
                    "zero_turnover_symbols": summary.filter((pl.col("volume") == 0) | (pl.col("amount") == 0)).height,
                    "noncanonical_rows": frame.filter(~pl.col("symbol").str.contains(STOCK_PATTERN).fill_null(False)).height,
                })
                del frame
                for key in ["duplicate_rows", "bad_time_rows", "bad_price_rows", "bad_turnover_rows"]:
                    item[key] = int(summary[key].sum())
                    for row in summary.filter(pl.col(key) > 0).iter_rows(named=True):
                        writer.writerow([day, row["symbol"], key, row["rows"], row["session_minutes"]])
                for row in summary.filter(pl.col("session_minutes") < 240).iter_rows(named=True):
                    writer.writerow([day, row["symbol"], "partial_session", row["rows"], row["session_minutes"]])
                daily_path = maps["kline_daily"].get(day)
                if daily_path is not None:
                    daily = pl.read_parquet(daily_path)
                    compared, meta = compare_daily(summary, daily.with_columns(pl.lit(day).alias('date')), data_dir=args.data_dir)
                    item.update(meta)
                    item["missing_active_symbols"] = int(compared["missing"].sum())
                    item["partial_active_symbols"] = compared.filter(~pl.col("missing") & pl.col("incomplete")).height
                    item["full_active_symbols"] = compared.filter(~pl.col("incomplete")).height
                    item["active_full_coverage_pct"] = round(item["full_active_symbols"] / compared.height * 100, 4) if compared.height else None
                    for flag in ["missing", "price_mismatch", "turnover_mismatch"]:
                        affected = compared.filter(pl.col(flag))
                        item[flag + "_stock_days"] = affected.height
                        item[flag + "_samples"] = affected["symbol"].head(5).to_list()
                        for row in affected.iter_rows(named=True):
                            writer.writerow([day, row["symbol"], flag, row["rows"], row["session_minutes"]])
                    item["full_session_price_mismatches"] = compared.filter(~pl.col("incomplete") & pl.col("price_mismatch")).height
                    item["full_session_turnover_mismatches"] = compared.filter(~pl.col("incomplete") & pl.col("turnover_mismatch")).height
                else:
                    item["daily_reference_missing"] = True
                summaries.append(item)
            except Exception as exc:
                item.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                summaries.append(item)
                failures.append(item)
            if index % 20 == 0 or index == len(days):
                fh.flush()
                print(json.dumps({"checked": index, "total": len(days), "date": str(day),
                                  "errors": len(failures), "elapsed_s": round(time.perf_counter() - began, 1)}), flush=True)
    completed = [s for s in summaries if not s["in_progress"]]
    result = {
        "audit_time_cn": now.isoformat(), "start": start, "end": end,
        "dates_manifest": args.dates_file,
        "calendar_basis": "Union of local stock/ETF/index daily and minute partitions; not an independent exchange calendar",
        "bar_basis": "240 unique end-labeled regular minutes; optional 09:30 auction; current session reported separately",
        "daily_comparison_basis": "A-share daily rows with finite positive OHLC/volume/amount; bare codes normalized only for audit joins; ambiguous duplicates excluded",
        "price_comparison_note": "Cross-source discrepancy, not proof that minute rather than daily data is wrong",
        "partition_days": len(days), "completed_partition_days": len(completed),
        "total_rows": sum(s.get("rows", 0) for s in summaries),
        "errors": failures,
        "missing_minute_dates": [s["date"] for s in summaries if s["status"] == "missing_partition"],
        "missing_daily_reference_dates": [s["date"] for s in summaries if s.get("daily_reference_missing")],
        "totals_completed_days": {key: sum(s.get(key, 0) for s in completed) for key in [
            "rows", "symbols", "duplicate_rows", "bad_time_rows", "bad_price_rows", "bad_turnover_rows",
            "expected_active_symbols", "full_active_symbols", "missing_active_symbols", "partial_active_symbols",
            "full_session_price_mismatches", "full_session_turnover_mismatches",
        ]},
        "days": summaries,
        "elapsed_s": round(time.perf_counter() - began, 2),
    }
    with (output / "summary.json").open("x", encoding="utf-8") as fh:
        json.dump(result, fh, default=str, ensure_ascii=False, indent=2)
    scalar_rows = [{k: v for k, v in s.items() if not isinstance(v, list)} for s in summaries]
    pl.DataFrame(scalar_rows, infer_schema_length=None).write_csv(output / "daily_coverage.csv")
    print(json.dumps({"report": str(output / "summary.json"), "days": len(days),
                      "rows": result["total_rows"], "errors": len(failures),
                      "elapsed_s": result["elapsed_s"]}), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
