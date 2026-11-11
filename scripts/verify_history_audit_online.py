"""Bounded read-only upstream probes of local audit flags, without app startup."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import polars as pl
from dotenv import dotenv_values

from app.plugins.tushare.provider import _extract_rows
from audit_history_semantics import audit_symbol_expr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--daily-only", action="store_true")
    parser.add_argument("--adj-symbol")
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside source data")
    output.mkdir(parents=True, exist_ok=False)
    config = dotenv_values(args.env_file)
    cases = json.loads(args.cases_file.read_text())
    evidence, failures = [], {}
    client = httpx.Client(timeout=httpx.Timeout(15, connect=5), trust_env=False, verify=True)

    def probe(source, endpoint, params, *, headers=None, payload=None, transform=None):
        record = {"source": source, "host": urlparse(endpoint).netloc,
                  "params": params, "queried_at": datetime.now(timezone.utc).isoformat()}
        if failures.get(source, 0) >= 2:
            record["status"] = "skipped_after_two_source_failures"
            evidence.append(record)
            return []
        began = time.monotonic()
        try:
            response = client.post(endpoint, json=payload) if payload else client.get(endpoint, params=params, headers=headers)
            record["http_status"] = response.status_code
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict) and (body.get("success") is False or body.get("code") not in (None, 0, "0", 200, "200")):
                raise ValueError("upstream_error_envelope")
            rows = transform(body) if transform else _extract_rows(body)
            record.update(status="returned_rows" if rows else "empty", rows=rows)
            if not rows:
                failures[source] = failures.get(source, 0) + 1
        except Exception as exc:
            record.update(status="unavailable", error_type=type(exc).__name__)
            failures[source] = failures.get(source, 0) + 1
        record["elapsed_s"] = round(time.monotonic() - began, 3)
        evidence.append(record)
        (output / "evidence.json").write_text(json.dumps(evidence, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"source": source, "status": record["status"], "rows": len(record.get("rows", [])), "elapsed_s": record["elapsed_s"]}), flush=True)
        return record.get("rows", [])

    promax_url = str(config.get("TUSHARE_PROMAX_URL") or "").rstrip("/")
    promax_key = str(config.get("TUSHARE_PROMAX_KEY") or "")
    proxy_url = str(config.get("TUSHARE_HTTP_URL") or "")
    proxy_token = str(config.get("TUSHARE_TOKEN") or "")
    local = []
    for case in cases:
        symbol, day = case["symbol"], case["date"]
        f = root / "kline_daily" / ("date=" + day) / "part.parquet"
        if f.exists():
            df = pl.read_parquet(f).with_columns(pl.col("symbol").alias("stored_symbol")).with_columns(audit_symbol_expr("symbol"))
            local.append({**case, "rows": df.filter(pl.col("symbol") == symbol).to_dicts()})
        params = {"ts_code": symbol, "start_date": day.replace("-", ""), "end_date": day.replace("-", "")}
        if promax_url and promax_key:
            probe("promax_daily", promax_url + "/daily", params, headers={"X-API-Key": promax_key})
        if proxy_url and proxy_token:
            probe("tushare_proxy_daily", proxy_url, params, payload={"api_name": "daily", "token": proxy_token, "params": params, "fields": ""})
        prefix = symbol[-2:].lower()
        code = prefix + symbol[:6]
        def tencent_rows(body):
            item = (body.get("data") or {}).get(code, {})
            return [dict(zip(["date", "open", "close", "high", "low", "volume"], row[:6]))
                    for row in item.get("day", []) if row[0] == day]
        probe("tencent_unadjusted_daily", "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
              {"param": f"{code},day,{day},{day},1,"}, transform=tencent_rows)
    if promax_url and promax_key and args.adj_symbol:
        probe("promax_adj_factor", promax_url + "/adj_factor",
              {"ts_code": args.adj_symbol, "start_date": "20240909", "end_date": "20260909"},
              headers={"X-API-Key": promax_key})
    if promax_url and promax_key and not args.daily_only:
        headers = {"X-API-Key": promax_key}
        probe("promax_trade_cal", promax_url + "/trade_cal",
              {"exchange": "SSE", "start_date": "20240909", "end_date": "20260909", "is_open": "1"}, headers=headers)
        for symbol, day in [("920504.BJ", "20260806"), ("920895.BJ", "20260901")]:
            probe("promax_block_trade", promax_url + "/block_trade", {"ts_code": symbol, "trade_date": day}, headers=headers)
        probe("promax_listing", promax_url + "/stock_basic", {"ts_code": "920100.BJ"}, headers=headers)
    # The bridge itself is read-only; only selected recent histories are requested.
    for symbol in ([] if args.daily_only else ["000001.SZ", "002945.SZ", "920504.BJ"]):
        probe("tdx_daily", "http://host.docker.internal:8080/api/kline",
              {"code": symbol[:6], "type": "day", "limit": 100},
              transform=lambda body: body.get("data", {}).get("List", []) if isinstance(body.get("data"), dict) else body.get("data", []))
    (output / "local-cases.json").write_text(json.dumps(local, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "evidence.json").write_text(json.dumps(evidence, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "scope.json").write_text(json.dumps({"cases": cases, "configured_sources": {
        "promax": bool(promax_url and promax_key), "proxy": bool(proxy_url and proxy_token)},
        "tls_verification": True, "note": "No app jobs imported, no repair or sync endpoints called. Gateways are not assumed independent."}, indent=2))
    client.close()


if __name__ == "__main__":
    main()
