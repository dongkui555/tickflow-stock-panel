"""Classify supplementation candidates by security lifecycle and identity.

This is read-only. It never downloads, merges or overwrites market data.
Use the result to decide whether a gap should be repaired, skipped, retried
from another source, or isolated for an explicit identity mapping.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import polars as pl

from app.services.market_status import historical_suspension_status, market_day_status
from app.services.security_lifecycle import load_lifecycle_index, validate_security_lifecycle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market", default="", help="optional SH/SZ/BJ filter")
    args = parser.parse_args()

    frame = pl.read_csv(args.audit, ignore_errors=True)
    required = {"date", "code"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"audit missing columns: {sorted(missing)}")
    if args.market:
        frame = frame.filter(pl.col("market").cast(pl.String).str.to_uppercase() == args.market.upper())

    index = load_lifecycle_index(args.data_dir)
    rows = []
    counts = Counter()
    for raw in frame.iter_rows(named=True):
        code = str(raw.get("code") or "").split(".")[0].zfill(6)
        market = str(raw.get("market") or "").upper()
        symbol = f"{code}.{market}" if market else code
        decision = validate_security_lifecycle(symbol, raw.get("date"), args.data_dir, index=index)
        reason = decision.reason
        counts[reason] += 1
        market = market_day_status(args.data_dir, raw.get("date"))
        suspension = historical_suspension_status(args.data_dir, symbol, raw.get("date"))
        action = {
            "active": "repair_or_retry_source",
            "not_listed_yet": "skip_no_request",
            "delisted_on_date": "skip_no_request",
            "code_changed": "use_historical_alias_mapping",
            "market_identity_conflict": "quarantine_identity",
            "identity_history_unknown": "quarantine_need_dimension",
            "lifecycle_date_missing": "quarantine_need_effective_date",
        }.get(reason, "review")
        quality_status = str(raw.get("status") or "")
        if reason == "active" and quality_status == "complete_conflict":
            action = "quarantine_verify_daily_and_second_minute_source"
        elif reason == "active" and quality_status in {"missing_0", "invalid_partial"}:
            action = "repair_from_verified_source"
        if market["status"] == "market_closed":
            action = "skip_market_closed"
        elif suspension["status"] == "suspended":
            action = "skip_confirmed_suspension"
        rows.append({
            **raw,
            "normalized_symbol": decision.symbol,
            "lifecycle_reason": reason,
            "lifecycle_action": action,
            "lifecycle_status": decision.status,
            "listing_date": decision.listing_date.isoformat() if decision.listing_date else None,
            "delist_date": decision.delist_date.isoformat() if decision.delist_date else None,
            "successor_symbol": decision.successor_symbol,
            "lifecycle_source": decision.source,
            "market_day_status": market["status"],
            "market_day_reason": market["reason"],
            "suspension_status": suspension["status"],
            "suspension_reason": suspension["reason"],
        })

    args.output.mkdir(parents=True, exist_ok=False)
    detail = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    if not detail.is_empty():
        detail.write_csv(args.output / "lifecycle_classification.csv")
    summary = {
        "audit": str(args.audit),
        "rows": len(rows),
        "reason_counts": dict(counts),
        "repairable": counts.get("active", 0),
        "terminal_or_quarantine": len(rows) - counts.get("active", 0),
        "notes": [
            "退市/未上市只跳过相应日期，不等同于整只股票永久无数据。",
            "重组或更名不凭名称推断，必须有有效日期的历史代码映射。",
            "身份未知或市场冲突不请求补数，避免把BJ合并到SZ。",
            "市场休市优先使用已落盘交易日历; 日历缺失时保持unknown, 不把工作日猜成休市。",
            "历史停牌必须有明确股票身份和停牌/复牌有效区间; 当前P状态不能反推过去停牌。",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
