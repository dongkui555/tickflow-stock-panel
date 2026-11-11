"""Repair only independently verified daily baselines before minute import.

This is intentionally narrow: a local daily row may be replaced only when the
configured Fuyao daily source and QMT minute aggregation agree on identity,
prices, volume, and turnover, while the local row disagrees.  The old parquet
partition is backed up and read back before the script reports success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from app.plugins.fuyao.client import FuyaoClient
from app.plugins.fuyao.provider import _date_of_ms, _ms_of_date, _to_thscode, get_api_key
from app.services.minute_repair import regular
from repair_targeted_qmt_dat import QmtDatProvider


PRICE_FIELDS = ("open", "high", "low", "close")


def _hash(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _price_ok(left: dict, right: dict) -> bool:
    tolerance = max(0.011, abs(float(right["close"])) * 1e-6)
    return all(abs(float(left[field]) - float(right[field])) <= tolerance for field in PRICE_FIELDS)


def _number_ok(left: float, right: float, tolerance: float = 0.008) -> bool:
    return abs(float(left) - float(right)) <= max(1.0, abs(float(right)) * tolerance)


def _aggregate_qmt(provider: QmtDatProvider, symbol: str, day: date) -> dict | None:
    frame = provider.get_minute(
        [symbol],
        datetime.combine(day, time(9, 25)),
        datetime.combine(day, time(15, 30)),
    )
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
    return {
        **{field: float(row[field]) for field in (*PRICE_FIELDS, "volume_raw", "amount")},
        "rows": int(row["rows"]),
        "volume": float(row["volume_raw"]) * 100.0,
    }


def _fuyao_daily(client: FuyaoClient, symbol: str, day: date) -> dict | None:
    thscode = _to_thscode(symbol)
    rows = client.historical_kline(thscode, _ms_of_date(day), _ms_of_date(day), adjust="none")
    candidates = [row for row in rows if _date_of_ms(row.get("date_ms")) == day]
    if len(candidates) != 1:
        return None
    row = candidates[0]
    return {
        "symbol": symbol,
        "date": day.isoformat(),
        "open": float(row["open_price"]),
        "high": float(row["high_price"]),
        "low": float(row["low_price"]),
        "close": float(row["close_price"]),
        "volume": float(row["volume"]),
        "amount": float(row["turnover"]),
        "source": "fuyao_historical_daily",
    }


def _local_row(frame: pl.DataFrame, code: str) -> dict | None:
    rows = frame.filter(pl.col("symbol").cast(pl.Utf8).str.extract(r"(\d{6})", 1) == code)
    if rows.height != 1:
        return None
    row = rows.row(0, named=True)
    return {field: float(row[field]) if field != "date" else str(row[field]) for field in (*PRICE_FIELDS, "volume", "amount", "date")}


def _replace_local_row(frame: pl.DataFrame, code: str, source: dict) -> pl.DataFrame:
    mask = pl.col("symbol").cast(pl.Utf8).str.extract(r"(\d{6})", 1) == code
    old = frame.filter(mask)
    if old.height != 1:
        raise RuntimeError(f"本地日K身份不唯一或不存在: {code}")
    replacement = old.with_columns(
        pl.lit(source["open"]).cast(frame.schema["open"]).alias("open"),
        pl.lit(source["high"]).cast(frame.schema["high"]).alias("high"),
        pl.lit(source["low"]).cast(frame.schema["low"]).alias("low"),
        pl.lit(source["close"]).cast(frame.schema["close"]).alias("close"),
        pl.lit(source["volume"]).cast(frame.schema["volume"]).alias("volume"),
        pl.lit(source["amount"]).cast(frame.schema["amount"]).alias("amount"),
    )
    return pl.concat([frame.filter(~mask), replacement], how="diagonal_relaxed").sort(["symbol", "date"])


def _backup_replace(root: Path, report: Path, relative: Path, frame: pl.DataFrame) -> dict:
    path = root / relative
    if not path.exists():
        raise RuntimeError(f"日K分区不存在: {path}")
    before = _hash(path)
    backup = report / "backups" / relative
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup)
    if _hash(backup) != before:
        raise RuntimeError("日K备份校验失败")
    tmp = path.with_name(path.name + ".repair-tmp")
    frame.write_parquet(tmp)
    readback = pl.read_parquet(tmp)
    if not readback.equals(frame):
        raise RuntimeError("日K临时文件读回不一致")
    if _hash(path) != before:
        raise RuntimeError("日K源文件在修复前发生变化")
    tmp.replace(path)
    return {
        "file": str(relative),
        "backup": str(backup.relative_to(report)),
        "before_sha256": before,
        "after_sha256": _hash(path),
        "rows": frame.height,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--qmt-root", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--confirm-replace", action="store_true")
    args = parser.parse_args()
    if not args.confirm_replace:
        raise SystemExit("需要显式传入 --confirm-replace 才会备份并替换")

    root = args.data_dir.resolve()
    day = args.date
    relative = Path("kline_daily") / f"date={day.isoformat()}" / "part.parquet"
    path = root / relative
    local = pl.read_parquet(path)
    report = root / "import_reports" / f"verified_daily_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report.mkdir(parents=True, exist_ok=False)
    qmt = QmtDatProvider(args.qmt_root)
    client = FuyaoClient(api_key=get_api_key())
    results: list[dict] = []
    updated = local
    try:
        for raw_symbol in args.symbols:
            code = "".join(ch for ch in str(raw_symbol) if ch.isdigit())[-6:].zfill(6)
            symbol = f"{code}.BJ"
            local_row = _local_row(local, code)
            fuyao = _fuyao_daily(client, symbol, day)
            qmt_row = _aggregate_qmt(qmt, symbol, day)
            result = {"symbol": symbol, "date": day.isoformat(), "qmt_rows": qmt_row.get("rows", 0) if qmt_row else 0}
            if not local_row or not fuyao or not qmt_row:
                result.update(status="not_repairable", reason="missing_identity_or_source")
                results.append(result)
                continue
            qmt_matches = (
                _price_ok(qmt_row, fuyao)
                and _number_ok(qmt_row["volume"], fuyao["volume"])
                and _number_ok(qmt_row["amount"], fuyao["amount"])
            )
            local_matches = (
                _price_ok(local_row, fuyao)
                and _number_ok(local_row["volume"], fuyao["volume"])
                and _number_ok(local_row["amount"], fuyao["amount"])
            )
            result.update(
                status="candidate" if qmt_matches and not local_matches else "not_repairable",
                reason="qmt_fuyao_agree_local_daily_stale" if qmt_matches and not local_matches else "no_independent_consensus",
                local=local_row,
                fuyao={key: fuyao[key] for key in (*PRICE_FIELDS, "volume", "amount")},
                qmt={key: qmt_row[key] for key in (*PRICE_FIELDS, "volume", "amount", "rows")},
            )
            if qmt_matches and not local_matches:
                updated = _replace_local_row(updated, code, fuyao)
            results.append(result)
    finally:
        client.close()

    candidates = [row for row in results if row["status"] == "candidate"]
    if candidates:
        result = _backup_replace(root, report, relative, updated)
        for row in candidates:
            row["replacement"] = result
    (report / "details.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "completed",
        "date": day.isoformat(),
        "checked": len(results),
        "candidates": len(candidates),
        "replaced": len(candidates),
        "report_dir": str(report),
    }
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
