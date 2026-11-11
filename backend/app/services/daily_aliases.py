"""Conservative removal of equivalent stock daily aliases, not price repair."""
from __future__ import annotations

import math
import re

import polars as pl

from app.services.security_identity import exchange_for_code


def deduplicate_verified_aliases(frame: pl.DataFrame) -> tuple[pl.DataFrame, list[dict]]:
    required = {"symbol", "date", "open", "high", "low", "close", "volume", "amount"}
    if frame.is_empty() or not required <= set(frame.columns):
        return frame, []
    keyed = frame.with_row_index("_alias_row").with_columns(
        pl.col("symbol").cast(pl.String).str.extract(r"^(\d{6})(?:\.(?:SH|SZ|BJ))?$", 1).alias("_alias_code"))
    duplicates = keyed.filter(pl.col("_alias_code").is_not_null()).filter(
        pl.len().over(["_alias_code", "date"]) > 1)
    removed, report = [], []
    for group in duplicates.partition_by(["_alias_code", "date"]):
        rows = group.to_dicts()
        bare = [r for r in rows if re.fullmatch(r"\d{6}", r["symbol"])]
        evidence = {"code": rows[0]["_alias_code"], "date": str(rows[0]["date"])}
        if len(bare) != 1:
            report.append({**evidence, "status": "ambiguous_aliases"})
            continue
        canonical = bare[0]
        code = canonical["symbol"]
        exchange = exchange_for_code(code, "stock")
        if exchange is None:
            report.append({**evidence, "status": "unknown_exchange"})
            continue
        equivalent = True
        for row in rows:
            if row["symbol"] not in {code, f"{code}.{exchange}"}:
                equivalent = False
                break
            values = [row[k] for k in ("open", "high", "low", "close", "volume", "amount")]
            if any(v is None or not math.isfinite(v) or v < 0 for v in values):
                equivalent = False
                break
            if any(not math.isclose(row[k], canonical[k], rel_tol=1e-6, abs_tol=1e-5)
                   for k in ("open", "high", "low", "close")):
                equivalent = False
                break
            # A suffix alone does not prove a unit. Require agreement of both
            # price and turnover, allowing only shares/lots and rounding.
            volume_ok = any(math.isclose(row["volume"] * scale, canonical["volume"],
                                         rel_tol=1e-5, abs_tol=max(scale, 1.0))
                            for scale in (0.01, 1, 100))
            amount_ok = math.isclose(row["amount"], canonical["amount"], rel_tol=1e-6, abs_tol=1)
            if not volume_ok or not amount_ok:
                equivalent = False
                break
        if equivalent:
            removed.extend(r["_alias_row"] for r in rows if r["_alias_row"] != canonical["_alias_row"])
        report.append({**evidence, "status": "verified_duplicate" if equivalent else "values_conflict",
                       "removed": len(rows) - 1 if equivalent else 0})
    clean = keyed.filter(~pl.col("_alias_row").is_in(removed)).drop("_alias_row", "_alias_code")
    return clean, report
