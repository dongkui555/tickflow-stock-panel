"""Remove only the 17 duplicate aliases introduced during backup-route QA.

Run with the app stopped. Original six-digit rows are retained unchanged.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import os
import shutil

import polars as pl

ALIASES = {
    "301655.SZ", "301688.SZ", "301697.SZ", "301699.SZ", "601123.SH",
    "603448.SH", "688826.SH", "688835.SH", "688836.SH", "920059.BJ",
    "920071.BJ", "920093.BJ", "920107.BJ", "920268.BJ", "920269.BJ",
    "920288.BJ", "920289.BJ",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "data"
    target = root / "kline_daily/date=2026-09-09/part.parquet"
    digest = hashlib.sha256(target.read_bytes()).digest()
    frame = pl.read_parquet(target)
    symbols = set(frame["symbol"].to_list())
    aliases = symbols & ALIASES
    assert all(s[:6] in symbols for s in aliases), "Missing original six-digit row"
    clean = frame.filter(~pl.col("symbol").is_in(sorted(aliases)))
    print({"before": frame.height, "duplicates": len(aliases), "after": clean.height})
    if not args.apply or not aliases:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / "import_reports/quote_source_review_20260909" / f"alias_backup_{stamp}.parquet"
    backup.parent.mkdir(parents=True, exist_ok=True)
    assert not backup.exists(), "Backup already exists"
    shutil.copy2(target, backup)
    assert hashlib.sha256(backup.read_bytes()).digest() == digest
    tmp = target.with_name("part.alias-repair.parquet")
    clean.write_parquet(tmp)
    assert pl.read_parquet(tmp).equals(clean), "Round-trip verification failed"
    assert hashlib.sha256(target.read_bytes()).digest() == digest, "Concurrent writer detected"
    os.replace(tmp, target)
    print({"backup": str(backup), "removed": sorted(aliases), "original_rows_preserved": True})


if __name__ == "__main__":
    main()
