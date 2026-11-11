"""Bounded, resumable extraction of local L2 archives for research only."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import polars as pl

from app.services.l2_daily import canonical_equity_rows, extract_day, write_json


def select_universe(
    data_dir: Path,
    before: date,
    count: int,
    *,
    min_history_days: int = 60,
) -> dict:
    """Use only pre-sample history and turnover, stratified by market-code group."""
    if count < 4 or count > 200:
        raise ValueError("Pilot universe size must be within 4..200")
    if min_history_days < 1:
        raise ValueError("min_history_days must be positive")
    lookback_days = max(60, min_history_days * 3)
    paths = sorted(
        p
        for p in (data_dir / "kline_daily").glob("date=*/part.parquet")
        if before - timedelta(days=lookback_days)
        <= date.fromisoformat(p.parent.name[5:])
        < before
    )
    if len(paths) < min_history_days:
        raise ValueError(
            f"Need at least {min_history_days} pre-sample market dates for universe selection"
        )
    frames = []
    for path in paths:
        frame = canonical_equity_rows(pl.read_parquet(path, columns=["symbol", "amount"]))
        if frame["symbol"].n_unique() != frame.height:
            raise ValueError("Ambiguous pre-sample daily references")
        frames.append(
            frame.with_columns(
                pl.lit(date.fromisoformat(path.parent.name[5:])).alias("date")
            )
        )
    panel = pl.concat(frames).sort(["symbol", "date"])
    panel = panel.filter(pl.col("amount").is_finite() & (pl.col("amount") > 0))
    summary = (
        panel.group_by("symbol")
        .agg(
            pl.len().alias("days"),
            pl.col("amount").tail(20).mean().alias("amount"),
        )
        .filter(pl.col("days") >= min_history_days)
    )
    selected = []
    for i, prefix in enumerate(["00", "30", "60", "68"]):
        group = summary.filter(pl.col("symbol").str.starts_with(prefix)).sort(
            ["amount", "symbol"]
        )
        n = count // 4 + int(i < count % 4)
        if group.height < n:
            raise ValueError("Pre-sample universe too small")
        # Spread across liquidity ranks, omitting the thinnest 20 percent.
        indices = [
            min(group.height - 1, int(group.height * (0.2 + 0.75 * (j + 0.5) / n)))
            for j in range(n)
        ]
        selected.extend(group[indices].to_dicts())
    return {
        "symbols": sorted(r["symbol"] for r in selected),
        "selected": selected,
        "selection_before": str(before),
        "history_start": paths[0].parent.name[5:],
        "history_end": paths[-1].parent.name[5:],
        "min_history_days": min_history_days,
        "selection_method": (
            "pre-sample history eligibility plus trailing-20-session liquidity "
            "strata by market-code group"
        ),
        "limitations": [
            "research pilot, not full market",
            "no future returns used",
            "not industry-balanced",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--research-dir", type=Path)
    parser.add_argument(
        "--symbols",
        help="Explicit comma-separated pilot codes, or omit for pre-sample selection",
    )
    parser.add_argument("--max-symbols", type=int, default=12)
    parser.add_argument("--selection-min-history-days", type=int, default=60)
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--batch-rows", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--daily-volume-unit", choices=["shares", "lots"], default="lots")
    parser.add_argument(
        "--archive-tool",
        "--tar",
        dest="tar",
        default="tar",
        help="7z/7zr (recommended) or bsdtar executable",
    )
    args = parser.parse_args()
    if args.end < args.start or (args.max_days is not None and args.max_days < 1):
        parser.error("Invalid date range or max-days")
    root = args.research_dir or args.data_dir / "l2_research"
    if args.symbols:
        symbols = sorted(set(args.symbols.split(",")))
        if not all(
            re.fullmatch(r"(?:(?:00|30)\d{4}\.SZ|(?:60|68)\d{4}\.SH)", s)
            for s in symbols
        ):
            parser.error("Only explicit SH/SZ A-equity codes are accepted")
        universe = {
            "symbols": symbols,
            "selection_method": "explicit user pilot; selection bias not ruled out",
        }
    else:
        universe = select_universe(
            args.data_dir,
            args.start,
            args.max_symbols,
            min_history_days=args.selection_min_history_days,
        )
    universe_path = root / "universe.json"
    if (
        universe_path.exists()
        and json.loads(universe_path.read_text(encoding="utf-8")) != universe
    ):
        raise ValueError(
            "Research universe is frozen; choose another research directory"
        )
    write_json(universe_path, universe)
    archives = {}
    for path in sorted(args.source.rglob("*.7z")):
        if re.fullmatch(r"\d{8}", path.stem):
            day = date.fromisoformat(path.stem)
            if args.start <= day <= args.end:
                if day in archives:
                    raise ValueError("Ambiguous archive date")
                archives[day] = path
    plans = sorted(archives.items())[: args.max_days]
    if not plans:
        raise ValueError("No dated L2 archives found")
    results = []
    for index, (day, archive) in enumerate(plans, 1):
        begin = time.monotonic()
        print(
            json.dumps(
                {
                    "event": "start",
                    "date": str(day),
                    "index": index,
                    "total": len(plans),
                }
            ),
            flush=True,
        )
        try:
            report = extract_day(
                args.data_dir,
                archive,
                day,
                universe["symbols"],
                batch_rows=args.batch_rows,
                timeout=args.timeout,
                tar=args.tar,
                research_dir=root,
                daily_volume_unit=args.daily_volume_unit,
            )
            result = {
                k: report.get(k)
                for k in [
                    "date",
                    "status",
                    "accepted",
                    "rejected",
                    "input_rows",
                    "output_bytes",
                    "resumed",
                ]
            }
        except (ValueError, OSError, pl.exceptions.PolarsError) as exc:
            result = {
                "date": str(day),
                "status": "failed",
                "error": type(exc).__name__ + ": " + str(exc)[:250],
            }
        result["seconds"] = round(time.monotonic() - begin, 2)
        results.append(result)
        summary = {
            "results": results,
            "planned_dates": len(plans),
            "finished_dates": len(results),
            "accepted_stock_days": sum(r.get("accepted") or 0 for r in results),
            "failed_dates": sum(r["status"] == "failed" for r in results),
            "production_writes": 0,
            "raw_files_extracted_to_disk": 0,
        }
        write_json(root / "extraction_summary.json", summary)
        print(json.dumps(result), flush=True)
        if len(results) >= 3 and all(r["status"] == "failed" for r in results[-3:]):
            raise RuntimeError(
                "Three consecutive archive failures; stopped for diagnosis"
            )


if __name__ == "__main__":
    main()
