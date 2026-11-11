"""Read-only three-way audit for unresolved Beijing Stock Exchange minute gaps.

The audit compares the local daily baseline, QMT minute aggregation, and the
Eastmoney historical daily endpoint.  It never writes market data.  The
result is a decision aid because a daily volume mismatch can be a unit/scope
problem rather than a bad minute price path.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime, time as clock_time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from app.services.minute_repair import daily_reference, regular
from repair_targeted_qmt_dat import QmtDatProvider


PRICE_FIELDS = ("open", "high", "low", "close")
EASTMONEY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_UT = "fa5fd1943c7b386f172d6893dbfba10b"


def _price_ok(left: dict, right: dict) -> bool:
    if not left or not right:
        return False
    tolerance = max(0.011, abs(float(right["close"])) * 1e-6)
    return all(
        abs(float(left[field]) - float(right[field])) <= tolerance
        for field in PRICE_FIELDS
    )


def _volume_ok(left: float, right: float) -> bool:
    return abs(left - right) <= max(1.0, abs(right) * 0.008)


def _amount_ok(left: float, right: float) -> bool:
    return abs(left - right) <= max(1.0, abs(right) * 0.008)


def _aggregate(frame: pl.DataFrame, day) -> dict | None:
    part = regular(frame, day, include_opening=True)
    if part.is_empty():
        return None
    row = part.sort("datetime").select(
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume_raw"),
        pl.col("amount").sum().alias("amount"),
        pl.col("datetime").n_unique().alias("rows"),
    ).row(0, named=True)
    return {key: (int(value) if key == "rows" else float(value)) for key, value in row.items()}


def _fetch_eastmoney(code: str, day: str) -> dict:
    params = {
        "secid": f"0.{code}",
        "klt": "101",
        "fqt": "1",
        "beg": day.replace("-", ""),
        "end": day.replace("-", ""),
        "lmt": "5",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": EASTMONEY_UT,
    }
    request = Request(
        f"{EASTMONEY_URL}?{urlencode(params)}",
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"},
    )
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed HTTPS host
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    else:
        raise last_error
    data = payload.get("data") or {}
    for raw in data.get("klines") or []:
        fields = raw.split(",")
        if len(fields) < 7 or fields[0] != day:
            continue
        return {
            "symbol": f"{code}.BJ",
            "open": float(fields[1]),
            "close": float(fields[2]),
            "high": float(fields[3]),
            "low": float(fields[4]),
            # Eastmoney historical daily volume is in lots; local/QMT minute
            # volumes are normalized to shares by multiplying by 100.
            "volume": float(fields[5]) * 100.0,
            "amount": float(fields[6]),
            "raw_volume": float(fields[5]),
            "source": "eastmoney_daily",
        }
    return {}


def _classify(local: dict | None, qmt: dict | None, remote: dict | None) -> str:
    if not remote:
        return "eastmoney_daily_unavailable"
    if not local:
        return "local_daily_missing"
    if not qmt:
        return "qmt_empty"
    local_remote = (
        _price_ok(local, remote)
        and _volume_ok(float(local["volume"]), float(remote["volume"]))
        and _amount_ok(float(local["amount"]), float(remote["amount"]))
    )
    qmt_remote = (
        _price_ok(qmt, remote)
        and _volume_ok(float(qmt["volume_raw"]) * 100.0, float(remote["volume"]))
        and _amount_ok(float(qmt["amount"]), float(remote["amount"]))
    )
    local_qmt = (
        _price_ok(qmt, local)
        and _volume_ok(float(qmt["volume_raw"]) * 100.0, float(local["volume"]))
        and _amount_ok(float(qmt["amount"]), float(local["amount"]))
    )
    if qmt_remote and not local_remote:
        return "daily_baseline_replacement_candidate"
    if local_remote and qmt_remote:
        return "minute_import_candidate"
    if local_remote and not qmt_remote:
        return "qmt_turnover_conflict"
    if local_qmt and not local_remote:
        return "eastmoney_daily_conflict"
    return "three_way_conflict"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qmt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-seconds", type=float, default=0.15)
    args = parser.parse_args()

    root = args.data_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = (
        pl.read_csv(args.manifest, try_parse_dates=True)
        .filter((pl.col("market") == "BJ") & pl.col("status").is_in(["missing_0", "invalid_partial"]))
        .with_columns(pl.col("code").cast(pl.String).str.zfill(6))
        .unique(["date", "code"])
        .sort(["date", "code"])
    )
    provider = QmtDatProvider(args.qmt_root)
    refs_by_day: dict = {}
    rows: list[dict] = []
    try:
        for item in targets.iter_rows(named=True):
            day = item["date"]
            code = item["code"]
            if day not in refs_by_day:
                path = root / "kline_daily" / f"date={day}" / "part.parquet"
                refs_by_day[day] = daily_reference(pl.read_parquet(path), data_dir=root) if path.exists() else {}
            local = refs_by_day[day].get(code)
            qmt = _aggregate(
                provider.get_minute(
                    [f"{code}.BJ"],
                    datetime.combine(day, clock_time(9, 25)),
                    datetime.combine(day, clock_time(15, 30)),
                ),
                day,
            )
            try:
                remote = _fetch_eastmoney(code, str(day))
                error = ""
            except Exception as exc:  # noqa: BLE001
                remote = {}
                error = f"{type(exc).__name__}: {exc}"
            status = "eastmoney_transport_error" if error else _classify(local, qmt, remote)
            row = {
                "date": str(day),
                "symbol": f"{code}.BJ",
                "original_status": item["status"],
                "classification": status,
                "qmt_rows": qmt.get("rows") if qmt else 0,
                "qmt_price_ok": _price_ok(qmt, remote) if qmt and remote else False,
                "local_eastmoney_ok": (
                    _price_ok(local, remote)
                    and _volume_ok(float(local["volume"]), float(remote["volume"]))
                    and _amount_ok(float(local["amount"]), float(remote["amount"]))
                ) if local and remote else False,
                "qmt_eastmoney_ok": (
                    _price_ok(qmt, remote)
                    and _volume_ok(float(qmt["volume_raw"]) * 100.0, float(remote["volume"]))
                    and _amount_ok(float(qmt["amount"]), float(remote["amount"]))
                ) if qmt and remote else False,
                "local_volume": local.get("volume") if local else None,
                "eastmoney_volume": remote.get("volume") if remote else None,
                "qmt_volume_normalized": qmt.get("volume_raw", 0.0) * 100.0 if qmt else None,
                "local_amount": local.get("amount") if local else None,
                "eastmoney_amount": remote.get("amount") if remote else None,
                "qmt_amount": qmt.get("amount") if qmt else None,
                "eastmoney_error": error,
            }
            rows.append(row)
            print(json.dumps({"checked": len(rows), "total": targets.height, "symbol": row["symbol"], "status": status}, ensure_ascii=False), flush=True)
            if args.delay_seconds > 0:
                time.sleep(args.delay_seconds)
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    with (output / "details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["date", "symbol", "classification"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {"stock_days": len(rows), "classifications": dict(Counter(row["classification"] for row in rows))}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
