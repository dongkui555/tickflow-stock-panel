"""Stage unresolved BSE complete-conflict stock-days for bounded verification.

This command does not modify daily or minute partitions.  It creates a durable,
identity-qualified CSV manifest for the existing cross-source verifier.  Rows
already classified as a known BSE regular-session scope are excluded from the
manifest so a repeated repair run cannot reopen them.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


TERMINAL_SCOPES = {"regular_session", "regular_session_close_auction"}


def _load_coverage(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", payload) if isinstance(payload, dict) else {}
    result = {}
    if not isinstance(items, dict):
        return result
    for key, value in items.items():
        if not isinstance(value, dict):
            continue
        symbol = str(value.get("symbol") or key.split("|", 1)[0]).strip().zfill(6)
        day = str(value.get("date") or (key.split("|", 1)[1] if "|" in key else ""))[:10]
        if len(day) == 10:
            result[(symbol, day)] = value
    return result


def _reason(row: dict) -> str:
    if str(row.get("price_match")).lower() != "true":
        return "price_path_conflict"
    if str(row.get("volume_match")).lower() != "true":
        return "volume_conflict"
    return "daily_turnover_conflict"


def residual_rows(audit: Path, coverage: Path) -> tuple[list[dict], Counter]:
    statuses = _load_coverage(coverage)
    selected: list[dict] = []
    skipped = Counter()
    with audit.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status") or "") != "complete_conflict":
                continue
            code = str(row.get("code") or "").strip().zfill(6)
            day = str(row.get("date") or "")[:10]
            if not code or len(day) != 10:
                skipped["invalid_identity"] += 1
                continue
            current = statuses.get((code, day), {})
            scope = str(current.get("status") or "")
            if scope in TERMINAL_SCOPES:
                skipped[scope] += 1
                continue
            selected.append({
                "symbol": f"{code}.BJ",
                "date": day,
                "status": "existing_values_conflict",
                "reason": _reason(row),
                "market": "BJ",
                "code": code,
                "bars": row.get("bars") or "0",
                "price_match": row.get("price_match") or "",
                "volume_match": row.get("volume_match") or "",
                "amount_match": row.get("amount_match") or "",
                "daily_amount": row.get("amount") or "",
                "minute_amount": row.get("minute_amount") or "",
            })
    selected.sort(key=lambda item: (item["date"], item["symbol"]))
    return selected, skipped


def write_manifest(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "symbol", "date", "status", "reason", "market", "code", "bars",
        "price_match", "volume_match", "amount_match", "daily_amount",
        "minute_amount",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, skipped = residual_rows(args.audit, args.coverage)
    write_manifest(rows, args.output)
    print(json.dumps({
        "status": "staged",
        "rows": len(rows),
        "skipped": dict(skipped),
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
