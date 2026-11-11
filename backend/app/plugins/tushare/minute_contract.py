"""Validated session-boundary adapter for Promax historical one-minute bars."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import polars as pl


def _journal(root: Path, code: str, day, raw, daily, reason: str, mapped: bool,
             duplicate_rows: int = 0, snapshot_rows: int = 0) -> str:
    payload = json.dumps({
        "contract": "promax_stk_mins_boundaries_v2",
        "symbol": code, "date": str(day), "result": reason,
        "closing_1501_mapped_to_1500": mapped,
        "exact_duplicate_rows_removed": duplicate_rows,
        "after_close_snapshot_rows_removed": snapshot_rows,
        "raw_minute": raw.to_dict(orient="records"),
        "raw_daily": daily.to_dict(orient="records"),
    }, ensure_ascii=False, default=str, sort_keys=True).encode("utf-8")
    folder = root / str(day) / code
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (hashlib.sha256(payload).hexdigest()[:20] + ".json.gz")
    if target.exists():
        return str(target)
    temporary = target.with_suffix(".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_bytes(gzip.compress(payload, mtime=0))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return str(target)


def normalize_history(frame, daily, symbol: str, audit_dir: Path, *, outcomes: list[dict] | None = None) -> pl.DataFrame:
    # Reuse the same acceptance rules as disk repair; no tolerance is relaxed.
    from app.plugins.tushare.provider import _compact_symbol, _normalize_minute_frame
    from app.services.minute_repair import checked_additions

    required = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
    if not required.issubset(frame.columns):
        raise ValueError("Promax minute response lacks required fields")
    raw = frame.copy()
    stamps = pd.to_datetime(raw["trade_time"], format="mixed", errors="coerce")
    if stamps.isna().any():
        raise ValueError("Promax minute response has invalid timestamps")
    if not stamps.eq(stamps.dt.floor("min")).all():
        raw["trade_time"] = stamps
        for day, block in raw.groupby(raw.trade_time.dt.date):
            evidence = _journal(audit_dir, _compact_symbol(symbol), day, block, daily,
                                "source_subminute_timestamps", False)
            if outcomes is not None:
                outcomes.append({"date": str(day), "symbol": _compact_symbol(symbol),
                                 "reason": "source_subminute_timestamps", "retryable": False,
                                 "evidence": evidence})
        raise ValueError("Promax minute response has sub-minute timestamps")
    raw["trade_time"] = stamps
    code = _compact_symbol(symbol)
    if not raw.ts_code.map(_compact_symbol).eq(code).all():
        raise ValueError("Promax minute response returned another symbol")
    references = daily.copy()
    daily_cols = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
    if not daily_cols.issubset(references.columns):
        references = pd.DataFrame(columns=sorted(daily_cols))
    references = references[references.ts_code.map(_compact_symbol).eq(code)]
    references["_date"] = pd.to_datetime(references.trade_date.astype(str), format="%Y%m%d", errors="coerce").dt.date
    output = []
    for day, block in raw.groupby(raw.trade_time.dt.date):
        block = block.sort_values("trade_time")
        ref = references[references._date.eq(day)].drop(columns="_date")
        unique = ref[list(sorted(daily_cols))].drop_duplicates()
        candidate = block.drop_duplicates().copy()
        duplicate_rows = len(block) - len(candidate)
        snapshot_rows = 0
        clock = candidate.trade_time.dt.strftime("%H:%M:%S")
        mapped = False
        reason = "accepted"
        if candidate.trade_time.duplicated().any():
            reason = "source_duplicate_timestamps"
        late = candidate[clock.eq("15:01:00")]
        after = candidate[clock.gt("15:00:00")]
        if not after.empty:
            numeric = after[["vol", "amount"]].apply(pd.to_numeric, errors="coerce")
            if numeric.isna().any().any() or (numeric < 0).any().any():
                reason = "invalid_after_close_turnover"
            else:
                close = candidate[clock.eq("15:00:00")]
                values = ["open", "high", "low", "close", "vol", "amount"]
                closing_values = close[values].apply(pd.to_numeric, errors="coerce")
                after_values = after[values].apply(pd.to_numeric, errors="coerce")
                if (len(close) == 1 and closing_values.notna().all().all()
                        and closing_values[["vol", "amount"]].gt(0).all().all()
                        and after_values.eq(closing_values.iloc[0]).all().all()):
                    # Repeated copies of an existing closing bar are snapshots, not
                    # additional trades. Full-session daily validation is still mandatory.
                    snapshot_rows = len(after)
                    candidate = candidate[clock.le("15:00:00")].copy()
                    after = after.iloc[:0]
            if (not after.empty and reason == "accepted" and not late.empty
                    and ((pd.to_numeric(late.vol) > 0) | (pd.to_numeric(late.amount) > 0)).any()):
                close = candidate[clock.eq("15:00:00")]
                rest = after[~after.trade_time.dt.strftime("%H:%M:%S").eq("15:01:00")]
                if (len(close) != 1 or len(late) != 1
                        or not close[["vol", "amount"]].apply(pd.to_numeric).eq(0).all().all()
                        or not rest[["vol", "amount"]].apply(pd.to_numeric).eq(0).all().all()):
                    reason = "ambiguous_closing_bucket"
                elif reason == "accepted":
                    # Only replace a zero-turnover placeholder with the actual
                    # source closing bucket. Keep original timestamps in the journal.
                    candidate = candidate[~clock.eq("15:00:00")].copy()
                    candidate.loc[late.index, "trade_time"] = late.trade_time - pd.Timedelta(minutes=1)
                    mapped = True
            elif not after.empty and reason == "accepted" and not numeric.eq(0).all().all():
                reason = "unsupported_after_close_turnover"
        normalized = _normalize_minute_frame(candidate, symbol)
        reference = None
        if len(unique) == 1:
            row = unique.iloc[0]
            reference = {c: float(row[c]) for c in ("open", "high", "low", "close")}
            reference.update(volume=float(row.vol), amount=float(row.amount) * 1000.0)
            if not all(pd.notna(v) and float("-inf") < v < float("inf") and v > 0 for v in reference.values()):
                reference = None
        if reason == "accepted":
            accepted, reason = checked_additions(
                normalized,
                normalized.clear(),
                reference,
                day,
                source="tushare",
            )
            if reason == "accepted" and abs(accepted["volume"].sum() - reference["volume"]) > max(1.0, reference["volume"] * 0.002):
                accepted = normalized.clear()
                reason = "daily_volume_unit_conflict"
        else:
            accepted = normalized.clear()
        evidence = _journal(audit_dir, code, day, block, ref, reason, mapped, duplicate_rows, snapshot_rows)
        if outcomes is not None:
            outcomes.append({"symbol": code, "date": str(day), "reason": reason,
                             "retryable": False, "evidence": evidence})
        if reason == "accepted":
            output.append(accepted)
    return pl.concat(output) if output else pl.DataFrame()
