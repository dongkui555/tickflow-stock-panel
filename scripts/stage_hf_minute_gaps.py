"""Stage missing 1-minute bars from a public Hugging Face snapshot.

The snapshot is an external, non-official source.  This command is deliberately
staging-only: it downloads only instruments present in an exact gap manifest,
validates each stock-day against the local daily reference and current minute
partition, and writes accepted files plus a report under ``data/import_reports``.
Production writes are performed separately by ``import_staged_minute_gaps.py``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from threading import Lock

import httpx
import polars as pl

from app.services.minute_repair import COLS, checked_additions, daily_reference, regular


DATASET_REVISION = "ba589a11534825044fe5a6b84838f50ba8d8d188"
DATASET_ROOT = "https://huggingface.co/datasets/neigezhu/china-a-share-1min-ohlcv/resolve/"


def _load_manifest(path: Path) -> dict[tuple[str, str], set[date]]:
    result: dict[tuple[str, str], set[date]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            status = str(row.get("status") or "").strip().lower()
            if status not in {"missing_0", "invalid_partial"}:
                continue
            market = str(row.get("market") or "").upper()
            code = str(row.get("code") or "").zfill(6)
            if market not in {"BJ", "SH", "SZ"} or not re.fullmatch(r"\d{6}", code):
                continue
            result.setdefault((market, code), set()).add(date.fromisoformat(str(row["date"])))
    return result


def _download_instrument(client: httpx.Client, market: str, code: str) -> pl.DataFrame:
    url = f"{DATASET_ROOT}{DATASET_REVISION}/data/stock_1m/{market}/{code}.parquet"
    response = client.get(url)
    response.raise_for_status()
    return pl.read_parquet(io.BytesIO(response.content))


def _existing(root: Path, day: date, code: str) -> pl.DataFrame:
    path = root / "kline_minute" / f"date={day}" / "part.parquet"
    if not path.exists():
        return pl.DataFrame(schema={name: pl.String for name in COLS})
    frame = pl.scan_parquet(path).filter(
        pl.col("symbol").cast(pl.String).str.extract(r"(\d{6})", 1) == code,
    ).select(COLS).collect()
    return frame


def _aggregate(frame: pl.DataFrame, day: date) -> dict:
    frame = regular(frame, day, include_opening=True).sort("datetime")
    if frame.is_empty():
        return {}
    traded = (pl.col("volume") > 0) | (pl.col("amount") > 0)
    return frame.select(
        pl.len().alias("rows"),
        pl.col("open").filter(traded).first(),
        pl.col("high").filter(traded).max(),
        pl.col("low").filter(traded).min(),
        pl.col("close").filter(traded).last(),
        pl.col("volume").sum(), pl.col("amount").sum(),
    ).row(0, named=True)


def _stage_one(
    root: Path,
    output: Path,
    market: str,
    code: str,
    days: set[date],
    raw: pl.DataFrame,
) -> list[dict]:
    # Fail closed if the provider returns a file whose identity does not match
    # the requested instrument.  A wrong-file match can otherwise look like a
    # valid minute session after the local symbol suffix is reconstructed.
    raw_code_series = raw.get_column("symbol").cast(pl.String).str.zfill(6).str.extract(r"^(\d{6})$", 1)
    raw_codes = raw_code_series.filter(raw_code_series.is_not_null()).unique().to_list()
    if "exchange" in raw.columns:
        raw_exchange_series = (
            raw.get_column("exchange")
            .cast(pl.String)
            .str.to_uppercase()
            .replace({"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"})
        )
        raw_exchanges = raw_exchange_series.filter(raw_exchange_series.is_not_null()).unique().to_list()
    else:
        raw_exchanges = []
    if (raw_codes != [code] or raw_exchanges != [market]
            or raw_code_series.null_count()
            or ("exchange" in raw.columns and raw.get_column("exchange").null_count())):
        return [
            {
                "symbol": f"{code}.{market}",
                "date": str(day),
                "staged_rows": 0,
                "addable_rows": 0,
                "reason": "source_identity_conflict",
                "source_codes": raw_codes,
                "source_exchanges": raw_exchanges,
            }
            for day in sorted(days)
        ]
    raw = raw.with_columns(
        pl.col("timestamp").alias("datetime"),
        pl.col("symbol").cast(pl.String).str.zfill(6).alias("_code"),
        # The public snapshot stores volume in shares; TickFlow stores lots.
        pl.col("volume").cast(pl.Float64).truediv(100).alias("volume"),
        pl.col("turnover").cast(pl.Float64).alias("amount"),
    ).with_columns(
        (pl.col("_code") + pl.lit(".") + pl.lit(market)).alias("symbol"),
    ).select(COLS)
    results: list[dict] = []
    for day in sorted(days):
        result = {"symbol": f"{code}.{market}", "date": str(day), "staged_rows": 0, "addable_rows": 0}
        daily_path = root / "kline_daily" / f"date={day}" / "part.parquet"
        if not daily_path.exists():
            result["reason"] = "daily_reference_missing"
            results.append(result)
            continue
        refs = daily_reference(pl.read_parquet(daily_path), data_dir=root)
        reference = refs.get(code)
        if reference is None:
            result["reason"] = "daily_reference_missing"
            results.append(result)
            continue
        incoming = raw.filter(pl.col("datetime").dt.date() == day)
        existing = _existing(root, day, code)
        result.update(
            daily_reference={key: reference[key] for key in COLS[2:]},
            source_aggregate=_aggregate(incoming, day),
            local_aggregate=_aggregate(existing, day) if existing.height else {},
        )
        verified, reason = checked_additions(incoming, existing, reference, day, source="hf_public_snapshot")
        result.update(reason=reason, source_rows=incoming.height, addable_rows=verified.height)
        if reason == "accepted":
            # The importer must revalidate a complete session even when only a
            # few timestamps are missing in the current production partition.
            complete, complete_reason = checked_additions(
                incoming, incoming.clear(), reference, day, source="hf_public_snapshot",
            )
            if complete_reason != "accepted":
                raise RuntimeError(f"full session failed: {complete_reason}")
            target = output / "minutes" / f"{code}.{market}_{day}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            complete.write_parquet(target, compression="zstd")
            if pl.read_parquet(target).height != complete.height:
                raise RuntimeError(f"staged readback failed: {target}")
            result.update(staged_rows=complete.height, file=str(target.relative_to(output)))
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-instruments", type=int, default=0)
    parser.add_argument("--market", action="append", choices=("BJ", "SH", "SZ"))
    args = parser.parse_args()
    root = args.data_dir.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root / "import_reports") or output == root / "import_reports":
        raise ValueError("output must be a new directory inside data/import_reports")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    targets = _load_manifest(args.manifest)
    if args.market:
        allowed_markets = set(args.market)
        targets = {key: days for key, days in targets.items() if key[0] in allowed_markets}
    if args.max_instruments:
        targets = dict(list(sorted(targets.items()))[:args.max_instruments])
    if not targets:
        raise RuntimeError("manifest has no missing_0 or invalid_partial rows")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.json").write_text(json.dumps({
        "provider": "huggingface_public_snapshot",
        "dataset": "neigezhu/china-a-share-1min-ohlcv",
        "revision": DATASET_REVISION,
        "official": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = Lock()
    rows: list[dict] = []
    started = time.monotonic()

    def fetch(item: tuple[tuple[str, str], set[date]]) -> list[dict]:
        (market, code), days = item
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(90, connect=15)) as client:
            try:
                raw = _download_instrument(client, market, code)
                return _stage_one(root, output, market, code, days, raw)
            except Exception as exc:  # keep one bad file from aborting the batch
                return [{"symbol": f"{code}.{market}", "date": str(day),
                         "reason": "download_or_parse_error", "error": str(exc)[:500]}
                        for day in sorted(days)]

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="hf-minute") as pool:
        futures = [pool.submit(fetch, item) for item in sorted(targets.items())]
        for index, future in enumerate(as_completed(futures), 1):
            batch = future.result()
            with lock:
                rows.extend(batch)
                (output / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"instruments_done": index, "instruments_total": len(futures),
                              "stock_days_done": len(rows), "accepted": sum(r.get("reason") == "accepted" for r in rows),
                              "elapsed_s": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)
    summary = {
        "provider": "huggingface_public_snapshot",
        "instruments": len(targets),
        "stock_days_checked": len(rows),
        "accepted_stock_days": sum(row.get("reason") == "accepted" for row in rows),
        "staged_rows": sum(int(row.get("staged_rows", 0)) for row in rows),
        "reasons": dict(Counter(row["reason"] for row in rows)),
        "complete": len(rows) > 0 and all(row.get("reason") in {"accepted", "already_complete"} for row in rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
