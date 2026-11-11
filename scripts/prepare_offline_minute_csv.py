from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import zipfile

import polars as pl

from import_offline_minute_day import _drop_invalid_price_rows, _load_day_fast


def missing_pairs(data_dir: Path, start: str, end: str) -> pl.DataFrame:
    frames = []
    paths = sorted((data_dir / "kline_daily").glob("date=*/part.parquet"))
    for index, daily_path in enumerate(paths, 1):
        day = daily_path.parent.name.removeprefix("date=")
        if not start <= day <= end:
            continue
        expected = pl.read_parquet(daily_path, columns=["symbol"]).unique()
        minute_path = data_dir / "kline_minute" / f"date={day}" / "part.parquet"
        if minute_path.exists():
            covered = (
                pl.scan_parquet(minute_path).group_by("symbol")
                .agg(pl.col("datetime").n_unique().alias("bars"))
                .filter(pl.col("bars") >= 240).select("symbol").collect()
            )
            expected = expected.join(covered, on="symbol", how="anti")
        if not expected.is_empty():
            frames.append(expected.with_columns(pl.lit(day).alias("_trade_date")))
        if index % 25 == 0:
            print(json.dumps({"stage": "coverage", "date": day, "gap_days": len(frames)}), flush=True)
    if not frames:
        return pl.DataFrame(schema={"symbol": pl.String, "_trade_date": pl.String})
    return pl.concat(frames).sort(["_trade_date", "symbol"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage only missing stock-days from offline minute CSVs.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True, help="New zip for the existing verified-fill importer.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--volume-unit", required=True, choices=["lots", "shares"])
    args = parser.parse_args()
    source = Path(args.source).resolve()
    out = Path(args.output).resolve()
    report = Path(args.report).resolve()
    if out.exists() or report.exists():
        raise ValueError("Staging output or report already exists; refusing to overwrite")
    started = time.perf_counter()
    needed = missing_pairs(Path(args.data_dir), args.start_date, args.end_date)
    print(json.dumps({"stage": "gaps", "stock_days": needed.height,
                      "dates": needed["_trade_date"].n_unique()}), flush=True)
    result = {"source": str(source), "volume_unit": args.volume_unit,
              "start_date": args.start_date, "end_date": args.end_date,
              "needed_stock_days": needed.height, "staged_dates": []}
    if not needed.is_empty():
        frame, meta = _load_day_fast(source, None, len(list(source.glob("*.csv"))), needed_pairs=needed)
        frame, clean = _drop_invalid_price_rows(frame)
        if args.volume_unit == "shares":
            frame = frame.with_columns((pl.col("volume") / 100).alias("volume"))
        partitioned = frame.with_columns(pl.col("datetime").dt.date().alias("_date"))
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_STORED) as zf:
            for day_frame in partitioned.partition_by("_date", maintain_order=True):
                day = day_frame["_date"][0]
                payload = io.BytesIO()
                day_frame.drop("_date").write_parquet(payload, compression="zstd")
                zf.writestr(day.strftime("%Y%m%d") + ".parquet", payload.getvalue())
                result["staged_dates"].append({"date": str(day), "rows": day_frame.height,
                                               "symbols": day_frame["symbol"].n_unique()})
        result.update({"scan": meta, "cleaning": clean, "staged_rows": frame.height, "output": str(out)})
    result["elapsed_s"] = round(time.perf_counter() - started, 2)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, default=str, ensure_ascii=False, indent=2)
    print(json.dumps(result, default=str, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
