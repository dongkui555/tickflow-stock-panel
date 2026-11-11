from datetime import date, datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import polars as pl

from audit_minute_integrity import compare_daily, main, summarize_minutes


class AuditMinuteTests(unittest.TestCase):
    def test_manifest_audit_does_not_scan_unselected_dates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for day in ("2026-05-07", "2026-05-08"):
                path = root / "kline_minute" / f"date={day}" / "part.parquet"
                path.parent.mkdir(parents=True)
                self.frame().write_parquet(path)
            manifest = root / "dates.csv"
            pl.DataFrame({"date": ["2026-05-08"]}).write_csv(manifest)
            output = root / "audit"
            with patch("sys.argv", ["audit", "--data-dir", str(root), "--start-date", "2026-05-07",
                                    "--end-date", "2026-05-08", "--dates-file", str(manifest),
                                    "--output-dir", str(output)]):
                self.assertEqual(main(), 0)
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["partition_days"], 1)
            self.assertEqual(result["days"][0]["date"], "2026-05-08")

    def frame(self):
        stamps = [datetime(2026, 5, 8, 9, 31) + timedelta(minutes=i) for i in range(120)]
        stamps += [datetime(2026, 5, 8, 13, 1) + timedelta(minutes=i) for i in range(120)]
        return pl.DataFrame({"symbol": ["000001.SZ"] * 240, "datetime": stamps,
                             "open": [10.] * 240, "high": [11.] * 240,
                             "low": [9.] * 240, "close": [10.] * 240,
                             "volume": [1.] * 240, "amount": [1000.] * 240})

    def test_complete_session(self):
        result = summarize_minutes(self.frame(), date(2026, 5, 8)).row(0, named=True)
        self.assertEqual(result["session_minutes"], 240)
        self.assertEqual(result["bad_time_rows"], 0)
        self.assertEqual(result["bad_price_rows"], 0)

    def test_duplicate_does_not_hide_gap(self):
        frame = self.frame()
        result = summarize_minutes(pl.concat([frame.head(239), frame.head(1)]), date(2026, 5, 8)).row(0, named=True)
        self.assertEqual(result["rows"], 240)
        self.assertEqual(result["session_minutes"], 239)
        self.assertEqual(result["duplicate_rows"], 1)

    def test_wrong_date_and_bad_ohlc(self):
        frame = self.frame().head(1).with_columns(pl.lit(0.).alias("high"))
        result = summarize_minutes(frame, date(2026, 5, 7)).row(0, named=True)
        self.assertEqual(result["bad_time_rows"], 1)
        self.assertEqual(result["bad_price_rows"], 1)

    def test_daily_reference_missing_symbol(self):
        summary = summarize_minutes(self.frame(), date(2026, 5, 8))
        daily = summary.select("symbol", "open", "high", "low", "close", "volume", "amount")
        daily = pl.concat([daily, daily.with_columns(pl.lit("000002.SZ").alias("symbol"))])
        result, meta = compare_daily(summary, daily)
        self.assertEqual(meta["expected_active_symbols"], 2)
        self.assertEqual(result["missing"].sum(), 1)
        self.assertEqual(result["price_mismatch"].sum(), 0)

    def test_legacy_bare_code_is_not_excluded(self):
        summary = summarize_minutes(self.frame(), date(2026, 5, 8))
        daily = summary.select("symbol", "open", "high", "low", "close", "volume", "amount")
        daily = daily.with_columns(pl.lit("000001").alias("symbol"))
        result, meta = compare_daily(summary, daily)
        self.assertEqual(meta["expected_active_symbols"], 1)
        self.assertEqual(result["missing"].sum(), 0)

    def test_legacy_minute_code_is_not_reported_missing(self):
        summary = summarize_minutes(self.frame(), date(2026, 5, 8))
        daily = summary.select("symbol", "open", "high", "low", "close", "volume", "amount")
        legacy_summary = summary.with_columns(pl.lit("000001").alias("symbol"))
        result, _ = compare_daily(legacy_summary, daily)
        self.assertEqual(result["missing"].sum(), 0)
        self.assertEqual(result["incomplete"].sum(), 0)

    def test_canonical_alias_preferred_but_ambiguous_duplicates_excluded(self):
        summary = summarize_minutes(self.frame(), date(2026, 5, 8))
        daily = summary.select("symbol", "open", "high", "low", "close", "volume", "amount")
        alias = daily.with_columns(pl.lit("000001").alias("symbol"), pl.lit(99.).alias("close"))
        result, meta = compare_daily(summary, pl.concat([daily, alias]))
        self.assertEqual(meta["expected_active_symbols"], 1)
        self.assertEqual(result["price_mismatch"].sum(), 0)
        _, meta = compare_daily(summary, pl.concat([daily, daily]))
        self.assertEqual(meta["expected_active_symbols"], 0)


if __name__ == "__main__":
    unittest.main()
