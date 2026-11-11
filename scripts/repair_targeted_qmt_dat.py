"""Repair audited minute gaps by reading QMT's local 64-byte minute records."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
from qmt_dat import QMT_DTYPE, qmt_minute_volume_lots, read_qmt_minute_records
from repair_targeted_qmt_gaps import TARGETS

from app.services.minute_repair import checked_additions, daily_reference, keyed, repair_minute_gaps
from app.tickflow.repository import DataStore, KlineRepository


class QmtDatProvider:
    name = "qmt"

    def __init__(self, root: Path):
        self.root = root

    @staticmethod
    def _symbol(value: str) -> tuple[str, str]:
        text = str(value).upper()
        code = "".join(char for char in text if char.isdigit())[-6:].zfill(6)
        if code.startswith(("43", "83", "87", "88", "92")) or text.endswith(".BJ"):
            return code, "BJ"
        if code.startswith(("5", "6", "9")) or text.endswith(".SH"):
            return code, "SH"
        return code, "SZ"

    def get_minute(self, symbols, start_time, end_time, asset_type="stock", freq="1m", on_chunk_done=None):
        frames = []
        china_tz = timezone(timedelta(hours=8))
        start_epoch = int(start_time.replace(tzinfo=china_tz).timestamp()) if start_time else 0
        end_epoch = int(end_time.replace(tzinfo=china_tz).timestamp()) if end_time else 2**32 - 1
        for symbol in symbols:
            code, market = self._symbol(symbol)
            path = self.root / market / "60" / f"{code}.DAT"
            if not path.exists() or path.stat().st_size < 72 or (path.stat().st_size - 8) % QMT_DTYPE.itemsize:
                continue
            selected = read_qmt_minute_records(path, start_epoch, end_epoch)
            if selected.size == 0:
                continue
            timestamps = pl.from_epoch(selected["timestamp"].astype(np.int64), time_unit="s") + timedelta(hours=8)
            frames.append(pl.DataFrame({
                "symbol": [f"{code}.{market}"] * selected.size,
                "datetime": timestamps,
                "open": selected["open"].astype(np.float64) / 1000.0,
                "high": selected["high"].astype(np.float64) / 1000.0,
                "low": selected["low"].astype(np.float64) / 1000.0,
                "close": selected["close"].astype(np.float64) / 1000.0,
                "volume": qmt_minute_volume_lots(selected),
                "amount": selected["amount"].astype(np.float64),
            }))
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, nargs="+")
    parser.add_argument("--qmt-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--all-missing", action="store_true", help="Use every audited missing or invalid stock-day")
    parser.add_argument("--markets", nargs="+", choices=("BJ", "SH", "SZ"))
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--reachable-only", action="store_true", help="Skip QMT files whose last record is before the target date")
    parser.add_argument("--repair-conflicts", action="store_true", help="Back up and replace independently verified conflicting stock-days")
    parser.add_argument("--limit", type=int, help="Maximum number of audited stock-days to process")
    parser.add_argument("--probe-symbol", default="")
    parser.add_argument("--probe-date", type=date.fromisoformat)
    parser.add_argument("--probe-validate", action="store_true")
    args = parser.parse_args()
    if args.probe_symbol and args.probe_date:
        frame = QmtDatProvider(args.qmt_root).get_minute(
            [args.probe_symbol],
            datetime.combine(args.probe_date, datetime.min.time()).replace(hour=9, minute=30),
            datetime.combine(args.probe_date, datetime.min.time()).replace(hour=15),
        )
        print(frame.head(1))
        print(frame.tail(1))
        print(frame.select(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
            pl.col("close").last(), pl.col("volume").sum(), pl.col("amount").sum(),
        ))
        if args.probe_validate:
            store = DataStore()
            daily_path = store.data_dir / "kline_daily" / f"date={args.probe_date}" / "part.parquet"
            minute_path = store.data_dir / "kline_minute" / f"date={args.probe_date}" / "part.parquet"
            refs = daily_reference(pl.read_parquet(daily_path), data_dir=store.data_dir)
            code = "".join(char for char in args.probe_symbol if char.isdigit())[-6:]
            old = pl.read_parquet(minute_path) if minute_path.exists() else frame.clear()
            old = keyed(old).filter(pl.col("_code") == code).drop("_code")
            accepted, reason = checked_additions(frame, old, refs.get(code), args.probe_date, source="qmt")
            print(json.dumps({"reason": reason, "accepted_rows": accepted.height,
                              "existing_rows": old.height, "reference": refs.get(code)}, default=str))
        return
    if args.audit is None:
        parser.error("--audit is required unless probe options are used")
    issues = pl.concat(
        [pl.read_csv(path, try_parse_dates=True) for path in args.audit],
        how="diagonal_relaxed",
    ).with_columns(
        pl.col("code").cast(pl.String).str.pad_start(6, "0")
    )
    missing = pl.col("status").is_in(["missing_0", "invalid_partial"])
    if args.all_missing:
        selected = issues.filter(missing).unique(["code", "date"])
    else:
        filters = [
            (pl.col("market") == market)
            & pl.col("date").is_between(start, end)
            for market, start, end in TARGETS
        ]
        selected = issues.filter(missing & pl.any_horizontal(filters)).unique(["code", "date"])
    if args.markets:
        selected = selected.filter(pl.col("market").is_in(args.markets))
    if args.start:
        selected = selected.filter(pl.col("date") >= args.start)
    if args.end:
        selected = selected.filter(pl.col("date") <= args.end)
    if args.reachable_only:
        import struct
        from datetime import timezone
        cn_tz = timezone(timedelta(hours=8))
        reachable = []
        for row in selected.select("market", "code", "date").iter_rows(named=True):
            code = str(row["code"]).zfill(6)
            path = args.qmt_root / str(row["market"]) / "60" / f"{code}.DAT"
            if not path.exists() or path.stat().st_size < 72:
                continue
            with path.open("rb") as handle:
                handle.seek(-64, 2)
                last_epoch = struct.unpack("<I", handle.read(4))[0]
            if datetime.fromtimestamp(last_epoch, cn_tz).date() >= row["date"]:
                reachable.append(row)
        selected = pl.DataFrame(reachable, schema=selected.schema) if reachable else selected.head(0)
    identity_lookup = {}
    ambiguous_identities = set()
    for row in selected.select("code", "market").unique().iter_rows(named=True):
        code = str(row["code"]).zfill(6)
        market = str(row["market"]).upper()
        if market not in {"BJ", "SH", "SZ"}:
            continue
        symbol = f"{code}.{market}"
        previous = identity_lookup.get(code)
        if previous and previous != symbol:
            ambiguous_identities.add(code)
        else:
            identity_lookup[code] = symbol
    for code in ambiguous_identities:
        identity_lookup.pop(code, None)
    stock_days = sorted(selected.select("code", "date").iter_rows(), key=lambda row: (row[0], row[1]))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        stock_days = stock_days[:args.limit]
    print(json.dumps({"selected_stock_days": len(stock_days)}, ensure_ascii=False), flush=True)
    if not stock_days:
        return

    def progress(done: int, total: int, message: str) -> None:
        if done == total or done % 40 == 0:
            print(json.dumps({"done": done, "total": total, "message": message}, ensure_ascii=False), flush=True)

    result = repair_minute_gaps(
        KlineRepository(DataStore()),
        start_date=min(day for _, day in stock_days),
        end_date=max(day for _, day in stock_days),
        provider_name="qmt",
        source_provider=QmtDatProvider(args.qmt_root),
        stock_days=stock_days,
        workers=args.workers,
        repair_conflicts=args.repair_conflicts,
        bypass_failover=True,
        identity_lookup=identity_lookup or None,
        progress=progress,
    )
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
