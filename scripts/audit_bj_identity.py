"""Classify Beijing Stock Exchange minute gaps by security identity and dates."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from app.plugins.tushare.provider import build_datahub_client


def parse_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text or text in {"None", "nan", "NaT"}:
        return None
    try:
        return date.fromisoformat(text[:10].replace("/", "-"))
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    missing = pl.read_csv(args.audit, ignore_errors=True).filter(
        (pl.col("market") == "BJ")
        & pl.col("code").cast(pl.String).str.zfill(6).str.starts_with("92")
    )
    codes = sorted(set(missing.get_column("code").cast(pl.String).str.zfill(6).to_list()))

    api = build_datahub_client()
    records: dict[str, list[dict]] = {}
    for status in ("L", "D", "P"):
        try:
            frame = api.stock_basic(
                exchange="BSE",
                list_status=status,
                fields="ts_code,symbol,name,market,list_date,delist_date",
            )
        except Exception:
            continue
        if frame is None or len(frame) == 0:
            continue
        for row in frame.to_dict("records"):
            code = str(row.get("symbol") or row.get("ts_code") or "").split(".")[0].zfill(6)
            if code in codes:
                records.setdefault(code, []).append(row)

    counts: dict[str, int] = {}
    rows: list[dict] = []
    for code in codes:
        target_dates = [str(value)[:10] for value in missing.filter(
            pl.col("code").cast(pl.String).str.zfill(6) == code
        ).get_column("date").to_list()]
        hits = records.get(code, [])
        active_dates: list[str] = []
        for row in hits:
            listed = parse_date(row.get("list_date"))
            delisted = parse_date(row.get("delist_date"))
            for text in target_dates:
                current = parse_date(text)
                if listed and current and listed <= current and (delisted is None or current <= delisted):
                    active_dates.append(text)

        if not hits:
            status = "not_in_authoritative_dimension"
        elif not active_dates:
            status = "not_listed_on_audited_dates"
        elif any(parse_date(row.get("delist_date")) is None for row in hits):
            status = "listed"
        else:
            status = "delisted_or_relisted"
        counts[status] = counts.get(status, 0) + 1
        rows.append({
            "code": code,
            "status": status,
            "audited_dates": target_dates,
            "active_audited_dates": sorted(set(active_dates)),
            "authoritative_records": hits,
        })

    result = {
        "generated_at": date.today().isoformat(),
        "source": "Tushare Datahub stock_basic BSE list_status L/D/P",
        "audited_codes": len(codes),
        "status_counts": counts,
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, default=str, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "audited_codes": len(codes),
        "status_counts": counts,
        "report": str(args.output / "summary.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
