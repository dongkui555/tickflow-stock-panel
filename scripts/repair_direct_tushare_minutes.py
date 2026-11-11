"""Repair audited minute gaps through the official Tushare Pro HTTP endpoint.

This is a bounded maintenance path for cases where the Promax gateway is
pending or unavailable.  It never fabricates bars, writes only after the
existing minute validation passes, and lets ``repair_minute_gaps`` create the
per-date backup before an atomic partition update.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from threading import RLock

import httpx
import pandas as pd
import polars as pl

from app.plugins.tushare.provider import (
    _extract_rows,
    _normalize_minute_frame,
    _to_ts_code,
)
from app.services.minute_repair import repair_minute_gaps


FIELDS = "ts_code,trade_time,open,close,high,low,vol,amount"
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class DirectTushareError(RuntimeError):
    def __init__(self, reason: str, *, retryable: bool, status: int | None = None):
        self.reason = reason
        self.retryable = retryable
        self.status = status
        super().__init__(reason)


class DirectTushareProvider:
    """Small thread-safe provider for one-symbol historical minute windows."""

    name = "tushare"

    def __init__(self, token: str, *, timeout: float = 35.0, attempts: int = 4):
        self.token = token.strip()
        self.timeout = timeout
        self.attempts = max(1, attempts)

    @staticmethod
    def _window(value: datetime | None, default: datetime) -> str:
        return (value or default).strftime("%Y-%m-%d %H:%M:%S")

    def _request(self, symbol: str, start_time: datetime, end_time: datetime) -> list[dict]:
        params = {
            "ts_code": _to_ts_code(symbol),
            "freq": "1min",
            "start_date": self._window(start_time.replace(hour=9, minute=25), datetime.now() - timedelta(days=5)),
            "end_date": self._window(end_time.replace(hour=15, minute=10), datetime.now()),
        }
        payload = {
            "api_name": "stk_mins",
            "token": self.token,
            "params": params,
            "fields": FIELDS,
        }
        last: DirectTushareError | None = None
        for attempt in range(self.attempts):
            try:
                response = httpx.post(
                    "https://api.tushare.pro",
                    json=payload,
                    timeout=self.timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = DirectTushareError("request_timeout" if isinstance(exc, httpx.TimeoutException) else "transport_error", retryable=True)
            else:
                if response.status_code in RETRYABLE_STATUS:
                    last = DirectTushareError("rate_limited" if response.status_code == 429 else "upstream_unavailable", retryable=True, status=response.status_code)
                elif response.status_code >= 400:
                    raise DirectTushareError("permission_denied" if response.status_code in (401, 403) else "request_rejected", retryable=False, status=response.status_code)
                else:
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise DirectTushareError("source_schema_invalid", retryable=False) from exc
                    if body.get("code", 0) not in (0, None):
                        message = str(body.get("msg") or "").lower()
                        retryable = any(word in message for word in ("频繁", "限流", "频率", "超限", "timeout", "temporar", "busy"))
                        raise DirectTushareError("rate_limited" if retryable else "request_rejected", retryable=retryable)
                    return _extract_rows(body)
            if last is not None and attempt + 1 < self.attempts:
                time.sleep(min(12.0, 1.5 * (2 ** attempt)))
        assert last is not None
        raise last

    def _fetch_one(self, symbol: str, start_time: datetime, end_time: datetime) -> pl.DataFrame:
        rows = self._request(symbol, start_time, end_time)
        if not rows:
            return pl.DataFrame()
        return _normalize_minute_frame(pd.DataFrame(rows), symbol)

    def get_minute_diagnostics(self, symbols, start_time, end_time, asset_type="stock", freq="1m", **_kwargs):
        if asset_type != "stock" or str(freq).lower().replace(" ", "") not in {"1m", "1min", "1minute"}:
            return pl.DataFrame(), [{"reason": "unsupported_request", "retryable": False}]
        frames: list[pl.DataFrame] = []
        issues: list[dict] = []
        for symbol in symbols:
            try:
                frame = self._fetch_one(symbol, start_time, end_time)
                if frame.is_empty():
                    issues.append({"symbol": str(symbol), "reason": "source_empty", "retryable": True})
                else:
                    frames.append(frame)
            except DirectTushareError as exc:
                issue = {"symbol": str(symbol), "reason": exc.reason, "retryable": exc.retryable}
                if exc.status is not None:
                    issue["http_status"] = exc.status
                issues.append(issue)
        return (pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()), issues

    def get_minute(self, symbols, start_time, end_time, asset_type="stock", freq="1m", **kwargs):
        frame, issues = self.get_minute_diagnostics(symbols, start_time, end_time, asset_type, freq, **kwargs)
        if frame.is_empty() and issues and all(not item.get("retryable") for item in issues):
            raise DirectTushareError(str(issues[-1].get("reason", "source_error")), retryable=False)
        return frame


def _stock_days(manifest: pl.DataFrame, args) -> list[tuple[str, date]]:
    required = {"market", "code", "date", "status"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest lacks columns: {sorted(missing)}")
    selected = manifest.filter(pl.col("status").is_in(["missing_0", "invalid_partial"]))
    if args.market:
        selected = selected.filter(pl.col("market").is_in(args.market))
    if args.start:
        selected = selected.filter(pl.col("date") >= args.start)
    if args.end:
        selected = selected.filter(pl.col("date") <= args.end)
    selected = selected.with_columns(pl.col("code").cast(pl.String).str.pad_start(6, "0"))
    if args.symbol:
        selected = selected.filter(pl.col("code").is_in([s.zfill(6) for s in args.symbol]))
    selected = selected.unique(["market", "code", "date"], maintain_order=True)
    rows = []
    for row in selected.iter_rows(named=True):
        rows.append((f"{row['code']}.{row['market']}", date.fromisoformat(str(row['date']))))
    rows.sort(key=lambda item: (item[1], item[0]), reverse=args.newest_first)
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--market", action="append", choices=("BJ", "SH", "SZ"))
    parser.add_argument("--symbol", action="append", help="Restrict to a six-digit code for probing")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected stock-days")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--repair-conflicts", action="store_true")
    parser.add_argument("--newest-first", action="store_true")
    args = parser.parse_args()

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")

    manifest = pl.read_csv(args.manifest, try_parse_dates=True)
    stock_days = _stock_days(manifest, args)
    print({"selected_stock_days": len(stock_days), "first": str(stock_days[:3])}, flush=True)
    if not stock_days:
        return

    provider = DirectTushareProvider(token, timeout=args.timeout)
    repo = SimpleNamespace(
        store=SimpleNamespace(data_dir=args.data_dir),
        _write_lock=RLock(),
        rebuild_views=lambda: None,
        clear_cache=lambda: None,
    )
    began = time.monotonic()

    def progress(done: int, total: int, message: str) -> None:
        if done == total or done % 10 == 0:
            print({"requests": done, "total": total, "elapsed_s": round(time.monotonic() - began), "message": message}, flush=True)

    result = repair_minute_gaps(
        repo,
        start_date=min(day for _, day in stock_days),
        end_date=max(day for _, day in stock_days),
        provider_name="tushare",
        source_provider=provider,
        stock_days=stock_days,
        workers=max(1, min(8, args.workers)),
        repair_conflicts=args.repair_conflicts,
        bypass_failover=True,
        progress=progress,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
