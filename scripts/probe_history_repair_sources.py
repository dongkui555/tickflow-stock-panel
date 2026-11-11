"""Save source responses and validate them without changing production history."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl

from app.plugins.tushare.minute_contract import normalize_history
from app.plugins.tushare.provider import TushareProvider, minute_error_details
from app.services.minute_repair import checked_additions, daily_reference, keyed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", choices=("all", "tdx", "promax"), default="all")
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Probe output must be outside production data")
    output.mkdir(parents=True, exist_ok=True)
    from app.config import settings

    settings.data_dir = output / "provider_journal"
    logging.disable(logging.CRITICAL)
    pro = TushareProvider()
    api = pro._ensure_api()
    api._client.timeout = httpx.Timeout(20, connect=5)
    results = []
    try:
        for endpoint in (() if args.source == "tdx" else ("a_share_mins", "stk_mins")):
            day, symbol = "2026-09-07", "000001.SZ"
            result = {"endpoint": endpoint, "symbol": symbol, "date": day}
            try:
                raw = getattr(api, endpoint)(
                    ts_code=symbol, freq="1MIN" if endpoint == "a_share_mins" else "1min",
                    start_date=day + " 09:25:00", end_date=day + " 15:10:00", limit=5000,
                )
                result.update(rows=len(raw), columns=list(raw.columns))
                if not raw.empty:
                    pl.from_pandas(raw).write_parquet(output / f"{endpoint}-raw.parquet")
                    reference = api.daily(ts_code=symbol, trade_date=day.replace("-", ""))
                    pl.from_pandas(reference).write_parquet(output / f"{endpoint}-daily.parquet")
                    outcomes = []
                    normalized = normalize_history(raw, reference, symbol, output / "evidence", outcomes=outcomes)
                    result["outcomes"] = [{k: v for k, v in r.items() if k != "evidence"} for r in outcomes]
                    if not normalized.is_empty():
                        refs = daily_reference(pl.read_parquet(root / "kline_daily" / f"date={day}" / "part.parquet"), data_dir=root)
                        candidate = keyed(normalized)
                        accepted, reason = checked_additions(candidate, candidate.clear(), refs.get(symbol[:6]), date.fromisoformat(day))
                        result["local_validation"] = reason
                        if reason == "accepted":
                            folder = output / "minutes"
                            folder.mkdir(exist_ok=True)
                            accepted.write_parquet(folder / f"{symbol}_{day}_{endpoint}.parquet")
            except Exception as exc:
                result.update(minute_error_details(exc))
            results.append(result)
            (output / "source-probe.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
            print(json.dumps(result, default=str), flush=True)
    finally:
        api.close()

    if args.source == "promax":
        return
    from app.plugins.tdx.provider import TdxProvider
    tdx = None
    try:
        tdx = TdxProvider()
        frame, issues = tdx.get_minute_diagnostics(
            ["000001"], datetime(2026, 9, 7, 9, 25), datetime(2026, 9, 7, 15, 10), freq="1m",
        )
        result = {"endpoint": "tdx", "rows": frame.height,
                  "issues": [{k: v for k, v in r.items() if k in {"reason", "http_status", "retryable"}} for r in issues]}
        if not frame.is_empty():
            frame.write_parquet(output / "tdx-raw.parquet")
            refs = daily_reference(pl.read_parquet(root / "kline_daily/date=2026-09-07/part.parquet"), data_dir=root)
            accepted, reason = checked_additions(frame, frame.clear(), refs.get("000001"), date(2026, 9, 7))
            result["local_validation"] = reason
            if reason == "accepted":
                folder = output / "minutes"
                folder.mkdir(exist_ok=True)
                accepted.write_parquet(folder / "000001.SZ_2026-09-07_tdx.parquet")
    except Exception as exc:
        result = {"endpoint": "tdx", **minute_error_details(exc)}
    finally:
        if tdx is not None:
            tdx.close()
    results.append(result)
    (output / "source-probe.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
