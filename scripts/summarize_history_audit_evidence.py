"""Save cross-source comparisons without touching application state."""
import argparse
import json
from collections import Counter
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    root, reports = args.data_dir.resolve(), args.report_dir.resolve()
    if reports == root or root in reports.parents:
        raise ValueError("Output must be outside source data")
    comparisons, source_status, factors = [], Counter(), []
    for folder in ["online", "online-minute"]:
        evidence = json.loads((reports / folder / "evidence.json").read_text())
        cases = json.loads((reports / folder / "local-cases.json").read_text())
        for record in evidence:
            source_status[record["source"] + ":" + record["status"]] += 1
            if record["source"] == "promax_adj_factor":
                factors = record.get("rows", [])
        for case in cases:
            symbol, day = case["symbol"], case["date"]
            promax = next((x.get("rows", []) for x in evidence if x["source"] == "promax_daily"
                           and x["params"].get("ts_code") == symbol and x["params"].get("start_date") == day.replace("-", "")), [])
            code = symbol[-2:].lower() + symbol[:6]
            tencent = next((x.get("rows", []) for x in evidence if x["source"] == "tencent_unadjusted_daily"
                            and x["params"].get("param", "").startswith(code + ",day," + day)), [])
            row = {"symbol": symbol, "date": day, "reason": case["reason"], "local_rows": len(case["rows"]),
                   "promax_rows": len(promax), "tencent_rows": len(tencent)}
            for name, online in [("promax", promax), ("tencent", tencent)]:
                if online:
                    row[name + "_local_ohlc_match"] = all(
                        all(abs(float(r[c]) - float(online[0][c])) <= 0.011 for c in ["open", "high", "low", "close"])
                        for r in case["rows"])
            comparisons.append(row)
    factor_map = {r["trade_date"]: float(r["adj_factor"]) for r in factors}
    adjusted = []
    for day in ["2024-09-09", "2025-01-02", "2025-07-01", "2026-07-01", "2026-07-28"]:
        frame = pl.read_parquet(root / "kline_daily_enriched" / ("date=" + day) / "part.parquet")
        for row in frame.filter(pl.col("symbol").is_in(["000001", "000001.SZ"])).select("symbol", "close", "raw_close").iter_rows(named=True):
            factor = factor_map.get(day.replace("-", ""))
            anchor = factor_map[max(factor_map)]
            expected = row["raw_close"] * factor / anchor
            adjusted.append({"date": day, **row, "gateway_factor": factor, "anchor_factor": anchor,
                             "anchor_date": max(factor_map), "expected_qfq_close": expected,
                             "stored_close_diff": row["close"] - expected})
    review = json.loads((reports / "review.json").read_text())
    older = [r for r in review["corrected_days"] if r["date"] < "2026-09-07"]
    missing = {key: sum(r.get(key, 0) for r in older) for key in ["missing_active_symbols", "partial_active_symbols"]}
    counts = {k: len(v["missing_partitions"]) for k, v in review["external_calendar_coverage"].items()}
    result = {"source_status_counts": dict(source_status), "online_case_comparisons": comparisons,
              "adjusted_price_samples": adjusted, "pre_september7_minute_missing": missing,
              "missing_partition_counts_2y": counts,
              "note": "Full local scan plus targeted online sample, not exhaustive external certification. No repair/recalculation performed."}
    (reports / "conclusions-evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
