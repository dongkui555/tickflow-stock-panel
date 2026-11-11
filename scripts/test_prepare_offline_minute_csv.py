from datetime import date
from pathlib import Path
import tempfile
import unittest

import polars as pl

from import_offline_minute_day import _load_day_fast
from prepare_offline_minute_csv import missing_pairs
from test_import_offline_minute_zip_batch import minutes


class CsvGapFilterTests(unittest.TestCase):
    def test_gap_planner_uses_unique_minutes_and_date_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "kline_daily/date=2026-04-17/part.parquet"
            reference.parent.mkdir(parents=True)
            pl.DataFrame({"symbol": ["000001.SZ", "000002.SZ", "000003.SZ"]}).write_parquet(reference)
            target = root / "kline_minute/date=2026-04-17/part.parquet"
            target.parent.mkdir(parents=True)
            short = minutes("000002.SZ").head(120)
            pl.concat([minutes(), short, short]).write_parquet(target)
            gaps = missing_pairs(root, "2026-04-17", "2026-04-17")
            self.assertEqual(gaps["symbol"].to_list(), ["000002.SZ", "000003.SZ"])
            self.assertTrue(missing_pairs(root, "2026-04-18", "2026-04-18").is_empty())

    def test_corrupt_existing_partition_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "kline_daily/date=2026-04-17/part.parquet"
            reference.parent.mkdir(parents=True)
            pl.DataFrame({"symbol": ["000001.SZ"]}).write_parquet(reference)
            target = root / "kline_minute/date=2026-04-17/part.parquet"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"broken")
            with self.assertRaises(pl.exceptions.ComputeError):
                missing_pairs(root, "2026-04-17", "2026-04-17")
            self.assertEqual(target.read_bytes(), b"broken")

    def test_scan_reads_only_requested_symbol_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pl.DataFrame({
                "date": ["2026-05-07", "2026-05-08", "2026-05-08"],
                "time": ["09:31", "09:31", "09:32"],
                "open": [10.0] * 3, "high": [10.0] * 3,
                "low": [10.0] * 3, "close": [10.0] * 3,
                "volume": [1.0] * 3, "amount": [1000.0] * 3,
            })
            for symbol in ["sh600000", "sz000001"]:
                frame.write_csv(root / f"{symbol}.csv")
            needed = pl.DataFrame({"symbol": ["600000.SH"], "_trade_date": ["2026-05-08"]})
            result, _ = _load_day_fast(root, None, 2, needed_pairs=needed)
            self.assertEqual(result.height, 2)
            self.assertEqual(result["symbol"].unique().to_list(), ["600000.SH"])
            self.assertEqual(result["datetime"].dt.date().unique().to_list(), [date(2026, 5, 8)])
            self.assertEqual(result["volume"].sum(), 2.0)


if __name__ == "__main__":
    unittest.main()
