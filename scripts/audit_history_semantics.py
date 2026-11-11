"""Read-only raw/enriched consistency and daily plausibility evidence."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from import_offline_minute_zip_batch import _symbol_expr


def audit_symbol_expr(name: str) -> pl.Expr:
    # Legacy stock-only partitions lost leading zeroes. Normalize in memory only.
    text = pl.col(name).cast(pl.String)
    padded = pl.when(text.str.contains(r"^\d{1,5}$")).then(text.str.pad_start(6, "0")).otherwise(text)
    code = padded.str.extract(r"^(\d{6})$", 1)
    return pl.when(code.str.starts_with("6")).then(code + ".SH").when(
        code.str.starts_with("0") | code.str.starts_with("3")
    ).then(code + ".SZ").when(
        code.str.starts_with("4") | code.str.starts_with("8") | code.str.starts_with("92")
    ).then(code + ".BJ").otherwise(_symbol_expr(name)).alias(name)


def daily_flags(frame: pl.DataFrame) -> pl.DataFrame:
    frame = frame.with_columns(pl.col("symbol").alias("stored_symbol")).with_columns(audit_symbol_expr("symbol"))
    valid_turnover = (pl.col("volume") > 0) & (pl.col("amount") > 0)
    tolerance = pl.max_horizontal(pl.lit(0.011), pl.col("high").abs() * 0.002)
    vwap = pl.col("amount") / pl.col("volume")
    valid_vwap = lambda x: x.is_between(pl.col("low") - tolerance, pl.col("high") + tolerance)
    return frame.with_columns(
        ((pl.col("volume") < 0) | (pl.col("amount") < 0)).alias("negative_turnover"),
        (valid_turnover & ~valid_vwap(vwap) & ~valid_vwap(vwap / 100)).alias("vwap_outside_range"),
        ((pl.col("close") > 0) & (pl.col("close") < 0.01)).alias("sub_cent_price"),
        (pl.col("symbol").is_not_null() & pl.len().over(["symbol", "date"]).gt(1)).alias("duplicate_key"),
        pl.col("stored_symbol").str.contains(r"^\d{1,5}$").alias("legacy_short_code"),
        pl.col("symbol").is_null().alias("unresolved_symbol"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside source data")
    output.mkdir(parents=True, exist_ok=False)
    details, metadata, history = [], [], []
    listings = pl.read_parquet(root / "instruments/instruments.parquet").select(
        audit_symbol_expr("symbol"), pl.col("listing_date").cast(pl.String).str.strptime(pl.Date, "%Y%m%d", strict=False),
    ).unique("symbol")
    for index, path in enumerate(sorted((root / "kline_daily").glob("date=*/*.parquet")), 1):
        before = path.stat()
        try:
            frame = daily_flags(pl.read_parquet(path)).join(listings, on="symbol", how="left").with_columns(
                (pl.col("date") < pl.col("listing_date")).fill_null(False).alias("before_listing_metadata"),
            )
            keys = ["negative_turnover", "vwap_outside_range", "sub_cent_price", "duplicate_key", "before_listing_metadata", "legacy_short_code", "unresolved_symbol"]
            counts = {k: frame.filter(pl.col(k)).height for k in keys}
            cols = ["stored_symbol", "symbol", "date", "open", "high", "low", "close", "volume", "amount", "listing_date"]
            for key in keys:
                details.extend({**row, "issue": key, "file": path.relative_to(root).as_posix()}
                               for row in frame.filter(pl.col(key)).select(cols).iter_rows(named=True))
            enriched_path = root / "kline_daily_enriched" / path.parent.name / path.name
            if enriched_path.exists():
                enriched_before = enriched_path.stat()
                enriched = pl.read_parquet(enriched_path).with_columns(audit_symbol_expr("symbol"))
                raw_columns = [c for c in ["close", "high", "low"] if "raw_" + c in enriched.columns]
                raw = frame.filter(~pl.col("duplicate_key")).select("symbol", "date", *raw_columns)
                enriched = enriched.filter(pl.len().over(["symbol", "date"]) == 1)
                joined = raw.join(enriched.select("symbol", "date", *["raw_" + c for c in raw_columns]), on=["symbol", "date"], how="inner")
                if raw_columns:
                    mismatches = joined.filter(pl.any_horizontal([
                        ((pl.col(c) - pl.col("raw_" + c)).abs() > 1e-6)
                        | (pl.col(c).is_null() != pl.col("raw_" + c).is_null()) for c in raw_columns
                    ]))
                    counts["enriched_raw_mismatch"] = mismatches.height
                    details.extend({**row, "issue": "enriched_raw_mismatch", "file": path.relative_to(root).as_posix()}
                                   for row in mismatches.iter_rows(named=True))
                enriched_after = enriched_path.stat()
                counts["enriched_changed_during_read"] = enriched_before.st_mtime_ns != enriched_after.st_mtime_ns or enriched_before.st_size != enriched_after.st_size
            after = path.stat()
            metadata.append({"file": path.relative_to(root).as_posix(), "rows": frame.height, **counts,
                             "changed_during_read": before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size})
            history.append(frame.filter(~pl.col("duplicate_key")).select("symbol", "date", "close", "volume", "amount"))
        except Exception as exc:
            metadata.append({"file": path.relative_to(root).as_posix(), "error": type(exc).__name__})
        if index % 100 == 0:
            print(json.dumps({"semantic_files": index, "flagged_records": len(details)}), flush=True)
    if history:
        prices = pl.concat(history).sort(["symbol", "date"]).with_columns(
            pl.col("close").shift(1).over("symbol").alias("previous_observed_close"),
            pl.col("date").shift(1).over("symbol").alias("previous_observed_date"),
        ).with_columns((pl.col("close") / pl.col("previous_observed_close") - 1).alias("observed_change"))
        jumps = prices.filter((pl.col("observed_change").abs() > 0.4) & (pl.col("close") > 0) & (pl.col("previous_observed_close") > 0))
        details.extend({**row, "issue": "large_observed_price_step_review"} for row in jumps.iter_rows(named=True))
    result = {"files": metadata, "issues": details,
              "notes": ["Flags are review candidates, not automatic error findings.",
                        "VWAP accepts either shares or lots; block trades may explain deviations.",
                        "Listing metadata can use successor codes; pre-listing flags require identity history.",
                        "Large observed steps may span missing days, suspensions or corporate actions."]}
    (output / "summary.json").write_text(json.dumps(result, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    if details:
        pl.DataFrame(details, infer_schema_length=None).write_csv(output / "issues.csv")
    print(json.dumps({"semantic_files": len(metadata), "flags": len(details)}), flush=True)


if __name__ == "__main__":
    main()
