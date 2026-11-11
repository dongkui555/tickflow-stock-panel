"""Read-only multi-source probe for unresolved Beijing minute stock-days."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time
from pathlib import Path

import polars as pl

from app.config import settings
from app.plugins.tdx.provider import TdxProvider
from app.plugins.tushare.provider import TushareProvider, minute_error_details
from app.reporting import csv_safe_rows
from app.services.minute_repair import checked_additions, daily_reference, regular


def load_targets(path: Path) -> list[tuple[str, str, object]]:
    frame = (
        pl.read_csv(path, try_parse_dates=True)
        .filter(
            (pl.col("market") == "BJ")
            & pl.col("status").is_in(["missing_0", "invalid_partial"])
        )
        .with_columns(pl.col("code").cast(pl.String).str.zfill(6))
        .unique(["code", "date"])
        .sort(["date", "code"])
    )
    return [
        (f"{row['code']}.BJ", row["code"], row["date"])
        for row in frame.select("code", "date").iter_rows(named=True)
    ]


def references(root: Path, targets: list[tuple[str, str, object]]) -> dict:
    result = {}
    for day in sorted({day for _, _, day in targets}):
        path = root / "kline_daily" / f"date={day}" / "part.parquet"
        if path.exists():
            result[day] = daily_reference(pl.read_parquet(path), data_dir=root)
    return result


def validate(frame: pl.DataFrame, code: str, day, refs: dict, source: str, *, issues=None) -> dict:
    if frame is None or frame.is_empty():
        # An empty normalized frame can follow a transport or validation failure.
        # Preserve that diagnostic instead of certifying that the source has no bars.
        reasons = list(dict.fromkeys(item["reason"] for item in (issues or []) if item.get("reason")))
        reason = reasons[0] if len(reasons) == 1 else "source_multiple_issues" if reasons else "source_empty"
        return {"source": source, "rows": 0, "regular_rows": 0, "validation": reason}
    block = frame.filter(pl.col("symbol").cast(pl.String).str.slice(0, 6) == code)
    regular_rows = regular(block, day).height
    accepted, reason = checked_additions(
        block, block.clear(), refs.get(day, {}).get(code), day, source=source
    )
    return {
        "source": source,
        "rows": block.height,
        "regular_rows": regular_rows,
        "validation": reason,
        "accepted_rows": accepted.height,
        "first": str(block["datetime"].min()) if not block.is_empty() else None,
        "last": str(block["datetime"].max()) if not block.is_empty() else None,
    }


def probe_local_qmt(root: Path, targets, refs) -> list[dict]:
    from repair_targeted_qmt_dat import QmtDatProvider

    provider = QmtDatProvider(root)
    results = []
    for symbol, code, day in targets:
        frame = provider.get_minute(
            [symbol],
            datetime.combine(day, time(9, 25)),
            datetime.combine(day, time(15, 30)),
        )
        result = validate(frame, code, day, refs, "qmt")
        result.update({"symbol": symbol, "date": str(day)})
        results.append(result)
    return results


def probe_tdx_one(item, refs):
    symbol, code, day = item
    provider = TdxProvider()
    try:
        frame, issues = provider.get_minute_diagnostics(
            [symbol],
            datetime.combine(day, time(9, 25)),
            datetime.combine(day, time(15, 30)),
            freq="1m",
        )
        result = validate(frame, code, day, refs, "tdx", issues=issues)
        result["issues"] = issues
    except Exception as exc:  # noqa: BLE001
        result = {"source": "tdx", **minute_error_details(exc)}
    finally:
        provider.close()
    result.update({"symbol": symbol, "date": str(day)})
    return result


def probe_tdx(targets, refs, workers: int) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(probe_tdx_one, item, refs) for item in targets]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            result["completed"] = index
            results.append(result)
            if index % 20 == 0 or index == len(futures):
                print(json.dumps({"source": "tdx", "completed": index, "total": len(futures)}, ensure_ascii=False), flush=True)
    return sorted(results, key=lambda row: (row.get("date", ""), row.get("symbol", "")))


def probe_promax_one(item, refs, output: Path):
    symbol, code, day = item
    provider = TushareProvider()
    try:
        frame, issues = provider.get_minute_diagnostics(
            [symbol],
            datetime.combine(day, time(9, 25)),
            datetime.combine(day, time(15, 30)),
            freq="1m",
        )
        result = validate(frame, code, day, refs, "promax", issues=issues)
        result["issues"] = issues
    except Exception as exc:  # noqa: BLE001
        result = {"source": "promax", **minute_error_details(exc)}
    finally:
        provider.close()
    result.update({"symbol": symbol, "date": str(day)})
    return result


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for source in sorted({row.get("source") for row in rows}):
        values = [row for row in rows if row.get("source") == source]
        counts = Counter(row.get("validation", row.get("reason", "unknown")) for row in values)
        summary[source] = {"stock_days": len(values), "classifications": dict(counts)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qmt-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=("qmt", "tdx", "promax"), default=["qmt", "tdx"])
    parser.add_argument("--tdx-workers", type=int, default=4)
    args = parser.parse_args()

    root = args.data_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.manifest)
    refs = references(root, targets)
    print(json.dumps({"selected_stock_days": len(targets), "sources": args.sources}, ensure_ascii=False), flush=True)
    rows = []
    if "qmt" in args.sources:
        if not args.qmt_root:
            raise ValueError("--qmt-root is required for qmt")
        qmt_rows = probe_local_qmt(args.qmt_root, targets, refs)
        rows.extend(qmt_rows)
        print(json.dumps({"source": "qmt", "completed": len(qmt_rows), "total": len(targets)}, ensure_ascii=False), flush=True)
    if "tdx" in args.sources:
        rows.extend(probe_tdx(targets, refs, args.tdx_workers))
    if "promax" in args.sources:
        settings.data_dir = output / "provider_journal"
        promax_rows = []
        for index, item in enumerate(targets, start=1):
            promax_rows.append(probe_promax_one(item, refs, output))
            if index % 10 == 0 or index == len(targets):
                print(json.dumps({"source": "promax", "completed": index, "total": len(targets)}, ensure_ascii=False), flush=True)
        rows.extend(promax_rows)

    (output / "summary.json").write_text(
        json.dumps({"selected_stock_days": len(targets), "summary": summarize(rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "details.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    pl.DataFrame(csv_safe_rows(rows)).write_csv(output / "details.csv")
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
