"""Stage complete per-stock factor histories with coverage and anchor checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from app.plugins.tushare.provider import TushareProvider, minute_error_details


def validate_factors(frame, symbol, expected_dates, existing):
    required = {"ts_code", "trade_date", "adj_factor"}
    if not required <= set(frame.columns) or frame.is_empty():
        raise ValueError("missing_factor_fields")
    if set(frame["ts_code"].to_list()) != {symbol}:
        raise ValueError("unexpected_factor_symbol")
    result = frame.select(
        pl.col("ts_code").str.slice(0, 6).alias("symbol"),
        pl.col("trade_date").cast(pl.String).str.replace_all("-", "").str.to_date("%Y%m%d"),
        pl.col("adj_factor").cast(pl.Float64).alias("ex_factor"),
    ).sort("trade_date")
    if result.select(pl.struct("symbol", "trade_date").n_unique()).item() != result.height:
        raise ValueError("duplicate_factor_dates")
    if result.filter(pl.col("trade_date").is_null() | pl.col("ex_factor").is_null()
                     | ~pl.col("ex_factor").is_finite() | (pl.col("ex_factor") <= 0)).height:
        raise ValueError("invalid_factor_values")
    missing = sorted(set(expected_dates) - set(result["trade_date"].to_list()))
    overlap = result.join(existing, on=["symbol", "trade_date"], how="inner", suffix="_old")
    if overlap.is_empty():
        raise ValueError("factor_anchor_missing")
    if overlap.filter((pl.col("ex_factor") - pl.col("ex_factor_old")).abs()
                      > pl.col("ex_factor_old").abs().clip(lower_bound=1) * 1e-8).height:
        raise ValueError("factor_anchor_conflict")
    return result, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symbols", default="000001.SZ,600000.SH,002945.SZ")
    parser.add_argument("--official-once", action="store_true")
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output.resolve()
    if root == output or output.is_relative_to(root):
        raise ValueError("Staging must be outside production data")
    output.mkdir(parents=True, exist_ok=True)
    symbols = list(dict.fromkeys(args.symbols.split(",")))
    if any(len(s) != 9 or not s[:6].isdigit() or s[6:] not in {".SZ", ".SH", ".BJ"} for s in symbols):
        raise ValueError("Expected full stock codes")
    expected = {s[:6]: set() for s in symbols}
    for path in sorted((root / "kline_daily").glob("date=*/part.parquet")):
        day = date.fromisoformat(path.parent.name[5:])
        if day < date(2024, 1, 1):
            continue
        frame = pl.read_parquet(path, columns=["symbol", "close"]).with_columns(
            pl.col("symbol").cast(pl.String).str.extract(r"^(\d{1,6})(?:\.(?:SH|SZ|BJ))?$", 1)
            .str.pad_start(6, "0").alias("symbol"))
        for code in frame.filter((pl.col("close") > 0) & pl.col("symbol").is_in(list(expected)))["symbol"].unique():
            expected[code].add(day)
    existing = pl.read_parquet(root / "adj_factor/all.parquet")
    from app.config import settings
    settings.data_dir = output / "provider_journal"
    logging.disable(logging.CRITICAL)
    pro = TushareProvider()
    api = None if args.official_once else pro._ensure_api()
    if api is not None:
        api._client.timeout = httpx.Timeout(30, connect=10)
    reports, failures = [], 0
    try:
        for symbol in symbols:
            dates = expected[symbol[:6]]
            if not dates:
                continue
            report = {"symbol": symbol, "expected_dates": len(dates)}
            path = output / f"{symbol}-raw.parquet"
            try:
                if path.exists():
                    raw = pl.read_parquet(path)
                elif args.official_once:
                    # The official token has a separate quota. Never forward it to a gateway.
                    token = os.environ.get("TUSHARE_TOKEN", "").strip()
                    if not token:
                        raise ValueError("Official token missing")
                    response = httpx.post("https://api.tushare.pro", json={
                        "api_name": "adj_factor", "token": token,
                        "params": {"ts_code": symbol, "start_date": min(dates).strftime("%Y%m%d"),
                                   "end_date": max(dates).strftime("%Y%m%d")},
                        "fields": "ts_code,trade_date,adj_factor",
                    }, timeout=httpx.Timeout(30, connect=10))
                    report["http_status"] = response.status_code
                    response.raise_for_status()
                    payload = response.json()
                    report["api_code"] = payload.get("code")
                    if payload.get("code") != 0:
                        report.update(status="failed", reason="official_api_rejected")
                        reports.append(report)
                        (output / "stock-factor-stage.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
                        print(json.dumps(report), flush=True)
                        break
                    data = payload["data"]
                    raw = pl.DataFrame(data["items"], schema=data["fields"], orient="row")
                    raw.write_parquet(path)
                else:
                    raw = pl.from_pandas(api.adj_factor(ts_code=symbol,
                        start_date=min(dates).strftime("%Y%m%d"), end_date=max(dates).strftime("%Y%m%d")))
                    raw.write_parquet(path)
                frame, missing = validate_factors(raw, symbol, dates, existing)
                report.update(rows=frame.height, missing_dates=[str(d) for d in missing],
                              status="partial" if missing else "validated",
                              raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if not missing:
                    target = output / f"{symbol}-validated.parquet"
                    frame.write_parquet(target)
                    if not pl.read_parquet(target).equals(frame):
                        raise RuntimeError("factor_readback_failed")
                failures = 0
            except Exception as exc:
                detail = {"reason": str(exc)} if isinstance(exc, ValueError) and str(exc) in {
                    "missing_factor_fields", "unexpected_factor_symbol", "duplicate_factor_dates", "invalid_factor_values",
                    "factor_anchor_missing", "factor_anchor_conflict"} else minute_error_details(exc)
                report.update(status="failed", **detail)
                failures += 1
            reports.append(report)
            (output / "stock-factor-stage.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
            print(json.dumps(report), flush=True)
            if failures >= 3 or args.official_once:
                break
            time.sleep(1)
    finally:
        if api is not None:
            api.close()


if __name__ == "__main__":
    main()
