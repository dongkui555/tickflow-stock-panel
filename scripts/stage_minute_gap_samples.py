"""Download exact missing stock-days without changing live market partitions."""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from app.market_time import cn_today
from app.plugins.tushare.provider import TushareProvider, minute_error_details
from app.services.minute_repair import COLS, checked_additions, daily_reference, keyed, regular


def stage_one(provider, root: Path, output: Path, symbol: str, day: date) -> dict:
    if not re.fullmatch(r"\d{6}\.(SZ|SH|BJ)", symbol) or day >= cn_today():
        raise ValueError("Only canonical stock symbols and closed historical dates are accepted")
    code = symbol[:6]
    result = {"symbol": symbol, "date": str(day), "staged_rows": 0, "addable_rows": 0}
    daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
    refs = daily_reference(pl.read_parquet(daily_path), data_dir=root) if daily_path.exists() else {}
    if code not in refs:
        return {**result, "reason": "daily_reference_missing"}
    source, issues = provider.get_minute_diagnostics(
        [symbol], datetime.combine(day, time(9, 30)), datetime.combine(day, time(15)), freq="1m",
    )
    if source.is_empty():
        exact = [i for i in issues if i.get("date") == str(day)]
        issue = (exact or issues or [{"reason": "source_empty"}])[-1]
        return {**result, **{k: issue[k] for k in ("reason", "retryable", "http_status") if k in issue}}
    if not keyed(source).select((pl.col("_code") == code).fill_null(False).all()).item():
        return {**result, "reason": "source_symbol_conflict"}
    verified, reason = checked_additions(source, source.clear(), refs[code], day)
    if reason != "accepted":
        return {**result, "reason": reason}
    minute_path = root / "kline_minute" / f"date={day}" / "part.parquet"
    if minute_path.exists():
        existing = (
            pl.scan_parquet(minute_path).filter(pl.col("symbol").str.extract(r"(\d{6})", 1) == code)
            .select(COLS).collect()
        )
    else:
        existing = verified.clear()
    additions, reason = checked_additions(verified, existing, refs[code], day)
    result.update(reason=reason, addable_rows=additions.height,
                  addable_regular_rows=regular(additions, day).height)
    if reason == "accepted":
        # Staged full sessions must be revalidated against the then-current
        # partition before import; this file is not a production write.
        target = output / f"{symbol}_{day}.parquet"
        verified.write_parquet(target)
        if not pl.read_parquet(target).equals(verified):
            raise RuntimeError("Staging readback failed")
        result.update(staged_rows=verified.height, file=target.name)
    return result


def stage_manifest(provider, root: Path, output: Path, manifest: pl.DataFrame) -> dict:
    rows = []
    failure_streak = 0
    stop_reason = None
    for row in manifest.iter_rows(named=True):
        try:
            result = stage_one(provider, root, output, row["symbol"], date.fromisoformat(row["date"]))
        except Exception as exc:
            result = {"symbol": row["symbol"], "date": row["date"], "staged_rows": 0,
                      "addable_rows": 0, **minute_error_details(exc)}
        rows.append(result)
        (output / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": len(rows), "total": manifest.height, **result}), flush=True)
        reason = result.get("reason")
        failure_streak = failure_streak + 1 if reason in {
            "request_timeout", "upstream_unavailable", "transport_error",
        } else 0
        if reason in {"rate_limited", "permission_denied"} or failure_streak >= 5:
            stop_reason = reason
            break
    summary = {
        "stock_days_planned": manifest.height,
        "stock_days_checked": len(rows),
        "deferred_stock_days": manifest.height - len(rows),
        "stop_reason": stop_reason,
        "addable_rows": sum(row.get("addable_rows", 0) for row in rows),
        "complete": len(rows) == manifest.height and all(
            row.get("reason") in {"accepted", "already_complete"} for row in rows
        ),
        "production_rows_written": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=13)
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output.resolve()
    if not output.is_relative_to(root / "import_reports") or output == root / "import_reports":
        raise ValueError("Staging must use a new subdirectory of import_reports")
    if not 1 <= args.limit <= 100 or not 1 <= args.timeout <= 30:
        raise ValueError("Use a bounded probe: 1-100 stock-days, 1-30 seconds per request")
    manifest = pl.read_csv(args.manifest).unique(subset=["symbol", "date"], maintain_order=True).head(args.limit)
    output.mkdir(parents=True, exist_ok=False)
    from app.config import settings

    # Provider journals also stay inside staging, separate from production data.
    settings.data_dir = output / "provider_journal"
    logging.disable(logging.CRITICAL)
    provider = TushareProvider()
    api = provider._ensure_api()
    if provider._connection_mode() != "promax":
        raise RuntimeError("This bounded staging runner requires the configured Promax client")
    import httpx

    api._client.timeout = httpx.Timeout(args.timeout)
    try:
        stage_manifest(provider, root, output, manifest)
    finally:
        api.close()


if __name__ == "__main__":
    main()
