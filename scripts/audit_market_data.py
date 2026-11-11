"""Read-only market-data audit. Event tables are not assumed to publish daily."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import polars as pl


CONTINUOUS = {
    "kline_daily", "kline_daily_enriched", "kline_etf_daily",
    "kline_index_daily", "moneyflow", "auction_open",
    "ext_data/ext_tushare_stk_limit/timeseries",
    "ext_data/ext_tushare_limit_list_d/timeseries",
    "ext_data/ext_tushare_cyq_perf/timeseries",
    "ext_data/ext_tushare_dc_daily/timeseries",
    "ext_data/ext_tushare_sw_daily/timeseries",
    "ext_data/ext_tushare_ths_daily/timeseries",
    "ext_data/ext_tushare_moneyflow_dc/timeseries",
    "ext_data/ext_tushare_moneyflow_ths/timeseries",
    "ext_data/ext_tushare_margin/timeseries",
}


def as_dates(values):
    result = set()
    for value in values:
        text = str(value or "")[:10]
        if len(text) == 8 and text.isdigit():
            text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
        try:
            result.add(date.fromisoformat(text).isoformat())
        except ValueError:
            pass
    return result


def audit(root: Path, output: Path, start: str, end: str):
    output.mkdir(parents=True, exist_ok=False)
    bases = [p for p in root.iterdir() if p.is_dir() and (
        p.name.startswith(("kline_", "instruments", "adj_factor"))
        or p.name in {"moneyflow", "auction_open", "news", "dragon_tiger", "depth5"}
    ) and "minute" not in p.name]
    bases += list((root / "financials").glob("*"))
    bases += [p for p in (root / "ext_data").glob("*/*") if p.is_dir()]
    bases += [p for p in (root / "ext_data").glob("*") if p.is_dir() and list(p.glob("*.parquet"))]
    calendar = set()
    for name in ("kline_daily", "kline_index_daily", "kline_etf_daily", "kline_minute"):
        calendar |= {p.name[5:] for p in (root / name).glob("date=*")
                     if start <= p.name[5:] <= end}
    summary, details = [], []
    for base in sorted(set(bases)):
        if not base.is_dir():
            continue
        name = base.relative_to(root).as_posix()
        files = sorted(base.rglob("*.parquet"))
        dates, symbols = set(), set()
        rows = invalid = duplicates = errors = 0
        for path in files:
            record = {"dataset": name, "file": path.relative_to(root).as_posix()}
            try:
                df = pl.read_parquet(path)
                rows += df.height
                date_col = next((c for c in ("date", "trade_date", "ann_date", "period_end", "as_of") if c in df.columns), None)
                values = as_dates(df[date_col].unique().to_list()) if date_col else set()
                dates |= values
                symbol = next((c for c in ("symbol", "ts_code", "exchange_id") if c in df.columns), None)
                if symbol:
                    symbols.update(str(v) for v in df[symbol].drop_nulls().unique())
                # Only identical whole records count as duplicates in event/revision tables.
                dup = df.height - df.unique().height
                duplicates += dup
                bad = 0
                if name.startswith("kline_") and {"open", "high", "low", "close"} <= set(df.columns):
                    prices = [pl.col(c).cast(pl.Float64, strict=False) for c in ("open", "high", "low", "close")]
                    inactive = pl.lit(False)
                    if {"volume", "amount"} <= set(df.columns):
                        inactive = pl.all_horizontal([c.is_null() for c in prices]) & (pl.col("volume").fill_null(0) == 0) & (pl.col("amount").fill_null(0) == 0)
                    bad = df.filter(~inactive & (pl.any_horizontal([~c.is_finite() | (c <= 0) | c.is_null() for c in prices])
                                    | (prices[1] < pl.max_horizontal(prices[0], prices[2], prices[3]))
                                    | (prices[2] > pl.min_horizontal(prices[0], prices[1], prices[3])))).height
                invalid += bad
                parent_date = next((p.name[5:] for p in path.parents if p.name.startswith("date=")), None)
                mismatch = bool(parent_date and values and values != {parent_date})
                future = len([v for v in values if v > date.today().isoformat()])
                if dup or bad or mismatch or future:
                    details.append({**record, "duplicates": dup, "invalid_ohlc": bad,
                                    "partition_mismatch": mismatch, "future_dates": future})
            except Exception as exc:
                errors += 1
                details.append({**record, "error": f"{type(exc).__name__}: {exc}"})
        missing = sorted(calendar - dates) if name in CONTINUOUS else []
        row = {"dataset": name, "files": len(files), "rows": rows, "symbols": len(symbols),
               "first_date": min(dates) if dates else None, "last_date": max(dates) if dates else None,
               "exact_duplicates": duplicates, "invalid_ohlc": invalid, "file_errors": errors,
               "missing_dates": missing,
               "coverage_basis": "local cross-dataset trading dates" if name in CONTINUOUS else "event/snapshot; no daily completeness assumption"}
        summary.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "missing_dates"}), flush=True)
    result = {"start": start, "end": end, "calendar_days": len(calendar),
              "calendar_basis": "Local union, not independently verified exchange calendar",
              "datasets": summary, "issues": details}
    overseas = root / "aiagents_overseas.db"
    if overseas.exists():
        with sqlite3.connect(f"file:{overseas.as_posix()}?mode=ro", uri=True) as conn:
            tables = []
            for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                if table.startswith("sqlite_"):
                    continue
                quoted = '"' + table.replace('"', '""') + '"'
                columns = [r[1] for r in conn.execute(f"PRAGMA table_info({quoted})")]
                item = {"table": table, "rows": conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]}
                col = next((c for c in ("date", "trade_date", "published_at", "timestamp") if c in columns), None)
                if col:
                    identifier = '"' + col.replace('"', '""') + '"'
                    item["date_bounds"] = conn.execute(f"SELECT MIN({identifier}), MAX({identifier}) FROM {quoted}").fetchone()
                tables.append(item)
            result["overseas_sqlite"] = {"quick_check": conn.execute("PRAGMA quick_check").fetchall(), "tables": tables}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows = [{**r, "missing_dates": ",".join(r["missing_dates"])} for r in summary]
    pl.DataFrame(csv_rows).write_csv(output / "datasets.csv")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2024-09-04")
    parser.add_argument("--end", default="2026-09-03")
    args = parser.parse_args()
    audit(args.data_dir, args.output_dir, args.start, args.end)
