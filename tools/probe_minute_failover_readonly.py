"""Read-only comparison of minute sources, local bars and daily references."""
import argparse
import json
from datetime import datetime, time, timedelta
from pathlib import Path

import polars as pl

from app.config import settings
from app.data_providers import custom
from app.services.minute_failover import _intraday_reason, _part, _read_minutes
from app.services.minute_repair import checked_additions, daily_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True, help="Comma-separated six-digit stock codes")
    parser.add_argument("--start", required=True, type=datetime.fromisoformat)
    parser.add_argument("--end", required=True, type=datetime.fromisoformat)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.start > args.end or (args.end - args.start).days > 5:
        parser.error("Use an ordered range of at most five days for bounded diagnostics")
    root = Path(settings.data_dir)
    symbols = args.symbols.split(",")
    if len(symbols) > 10 or any(len(s) != 6 or not s.isdigit() for s in symbols):
        parser.error("Use at most ten six-digit stock codes")
    start, end = args.start, args.end
    frames = {}
    result = {"start": str(start), "end": str(end), "sources": {}, "comparisons": []}
    for name in ("tushare", "tdx"):
        provider = custom.get_provider(name)
        frames[name], issues = _read_minutes(provider, symbols, start, end, "stock", "1m")
        result["sources"][name] = {"rows": frames[name].height, "issues": [
            {k: v for k, v in item.items() if k in ("reason", "symbol", "date", "retryable")}
            for item in issues]}
    for offset in range((end.date() - start.date()).days + 1):
        day = start.date() + timedelta(days=offset)
        dp = root / f"kline_daily/date={day}/part.parquet"
        refs = daily_reference(pl.read_parquet(dp), data_dir=root) if dp.exists() else {}
        mp = root / f"kline_minute/date={day}/part.parquet"
        old = (pl.scan_parquet(mp).filter(pl.col("symbol").str.extract(r"(\d{6})", 1).is_in(symbols))
               .collect() if mp.exists() else pl.DataFrame())
        for symbol in symbols:
            item = {"symbol": symbol, "day": str(day), "reference": refs.get(symbol), "bars": {}}
            for name, frame in {**frames, "local": old}.items():
                part = _part(frame, symbol, day)
                if part.is_empty():
                    item["bars"][name] = {"rows": 0}
                    continue
                cutoff = datetime.combine(day, time(15)) if day < end.date() else end
                expected = [datetime.combine(day, time(9, 31)) + timedelta(minutes=i) for i in range(120)]
                expected += [datetime.combine(day, time(13, 1)) + timedelta(minutes=i) for i in range(120)]
                missing = [str(t.time()) for t in expected if t <= cutoff and t not in set(part["datetime"])]
                _, reason = checked_additions(part, part.clear(), refs.get(symbol), day)
                _, old_reason = checked_additions(part, _part(old, symbol, day), refs.get(symbol), day)
                item["bars"][name] = {
                    "rows": part.height, "first": str(part["datetime"].min()), "last": str(part["datetime"].max()),
                    "missing_count": len(missing), "missing": missing[:8],
                    "daily_check": reason, "existing_check": old_reason,
                    "intraday_check": _intraday_reason(part, day, start, cutoff),
                    "open": part["open"][0], "high": part["high"].max(), "low": part["low"].min(),
                    "close": part["close"][-1], "volume": part["volume"].sum(), "amount": part["amount"].sum(),
                    "head": part.head(2).to_dicts(), "tail": part.tail(2).to_dicts(),
                }
            result["comparisons"].append(item)
    report = json.dumps(result, default=str, ensure_ascii=True, indent=2)
    if args.output:
        if args.output.exists():
            parser.error("Refusing to overwrite an existing diagnostic report")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(report)
        print(json.dumps({"report": str(args.output), "sources": result["sources"]}))
    else:
        print(report)


if __name__ == "__main__":
    main()
