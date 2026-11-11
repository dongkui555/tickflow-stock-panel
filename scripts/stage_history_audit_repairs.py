"""Stage audited history repairs. Source data is never written by this command."""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from app.plugins.tushare.provider import TushareProvider, minute_error_details


def stage_factors(root: Path, output: Path, provider, limit: int = 0, workers: int = 2) -> dict:
    existing = pl.read_parquet(root / "adj_factor/all.parquet")
    first = existing["trade_date"].min()
    dates = sorted(date.fromisoformat(p.parent.name[5:]) for p in
                   (root / "kline_daily").glob("date=*/part.parquet"))
    dates = [d for d in dates if date(2024, 1, 1) <= d < first]
    if limit:
        dates = dates[:limit]
    target = output / "factors"
    target.mkdir(parents=True, exist_ok=True)
    api = provider._ensure_api()
    results = []
    def fetch(day):
        path = target / f"{day}.parquet"
        daily = pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet")
        codes = set(daily.filter((pl.col("volume") > 0) & (pl.col("close") > 0))
                    ["symbol"].cast(pl.String).str.extract(r"^(\d{1,6})(?:\.(?:SH|SZ|BJ))?$", 1)
                    .str.pad_start(6, "0").drop_nulls().to_list())
        codes = {s for s in codes if s.startswith(("00", "30", "60", "68", "4", "8", "92"))}
        if path.exists():
            cached = pl.read_parquet(path)
            if set(cached["trade_date"].to_list()) != {day}:
                raise ValueError("Staged factor date mismatch")
            if not codes - set(cached["symbol"].to_list()):
                return {"date": str(day), "status": "staged", "rows": cached.height}
        result = {"date": str(day)}
        for attempt in range(3):
            try:
                frame = provider._promax_market_day(api.adj_factor, day.strftime("%Y%m%d"))
                if frame.empty or not {"ts_code", "trade_date", "adj_factor"} <= set(frame.columns):
                    raise ValueError("missing_factor_fields")
                staged = pl.from_pandas(frame).select(
                    pl.col("ts_code").str.extract(r"^(\d{6})\.(?:SH|SZ|BJ)$", 1).alias("symbol"),
                    pl.col("trade_date").cast(pl.String).str.replace_all("-", "").str.to_date("%Y%m%d"),
                    pl.col("adj_factor").cast(pl.Float64).alias("ex_factor"),
                )
                if staged.filter(pl.col("symbol").is_null() | pl.col("ex_factor").is_null()
                                 | ~pl.col("ex_factor").is_finite() | (pl.col("ex_factor") <= 0)).height:
                    raise ValueError("invalid_factor_values")
                missing = sorted(codes - set(staged["symbol"].to_list()))
                result = {"date": str(day)}
                # Partial responses remain explicit; they cannot certify complete coverage.
                result.update(rows=staged.height, missing_active_symbols=missing,
                              status="staged" if not missing else "partial")
                tmp = path.with_suffix(".tmp")
                staged.sort(["symbol", "trade_date"]).write_parquet(tmp)
                if pl.read_parquet(tmp).height != staged.height:
                    raise RuntimeError("factor_readback_failed")
                tmp.replace(path)
                break
            except Exception as exc:
                details = minute_error_details(exc)
                result.update(status="failed", **details)
                if details.get("reason") == "permission_denied" or attempt == 2:
                    break
                time.sleep(2 ** (attempt + 1))
        time.sleep(0.6)
        return result

    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index in range(0, len(dates), workers):
            for future in as_completed([pool.submit(fetch, day) for day in dates[index:index + workers]]):
                result = future.result()
                results.append(result)
                failures = failures + 1 if result["status"] == "failed" else 0
                (output / "factor-stage.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(json.dumps({"progress": len(results), "total": len(dates), **result}), flush=True)
            recent = results[-16:]
            degraded = len(recent) == 16 and sum(r["status"] == "failed" for r in recent) >= 4
            if degraded or failures >= 4 or any(r.get("reason") == "permission_denied" for r in results[-workers:]):
                break
    summary = {"planned_dates": len(dates), "checked_dates": len(results),
               "staged_dates": sum(r["status"] in {"staged", "partial"} for r in results),
               "rows": sum(r.get("rows", 0) for r in results), "production_writes": 0}
    (output / "factor-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--workers", default=2, type=int, choices=range(1, 5))
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Staging must be outside production data")
    output.mkdir(parents=True, exist_ok=True)
    from app.config import settings
    settings.data_dir = output / "provider_journal"
    logging.disable(logging.CRITICAL)
    provider = TushareProvider()
    api = provider._ensure_api()
    if provider._connection_mode() != "promax":
        raise RuntimeError("This repair uses the configured Promax daily factor contract")
    api._client.timeout = httpx.Timeout(20, connect=3)
    try:
        stage_factors(root, output, provider, args.limit, args.workers)
    finally:
        api.close()


if __name__ == "__main__":
    main()
