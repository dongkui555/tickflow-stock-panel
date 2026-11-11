from datetime import date
from hashlib import md5
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from source import LC1, day_records, latest_financial_package, verify_package


class TdxSourceTest(TestCase):
    def test_lc1_date_and_market_are_explicit(self):
        day = date(2026, 11, 9)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            minute = root / "sz/minline"
            minute.mkdir(parents=True)
            first = LC1.pack(22 * 2048 + 1109, 571, 10, 10, 10, 10, 1000, 100, 0)
            prior = LC1.pack(22 * 2048 + 1106, 900, 9, 9, 9, 9, 900, 100, 0)
            path = minute / "sz000001.lc1"
            path.write_bytes(prior + first)
            self.assertEqual(day_records(root, "000001.SZ", day), (first, "captured"))
            self.assertEqual(day_records(root, "000001.SH", day), (b"", "source_empty"))
            self.assertEqual(day_records(root, "000001.SZ", date(2026, 11, 10)), (b"", "source_empty"))
            path.write_bytes(first + b"\x00")
            self.assertEqual(day_records(root, "000001.SZ", day), (b"", "source_incomplete"))

    def test_financial_catalog_excludes_future_and_placeholder_periods(self):
        digest = "a" * 32
        catalog = (
            f"gpcw20261231.zip,{digest},164\n"
            f"gpcw20260930.zip,{digest},5100000\n"
            f"gpcw20260630.zip,{digest},4900000\n"
        )
        self.assertEqual(latest_financial_package(catalog, date(2026, 11, 10))["period"], "2026-09-30")
        self.assertEqual(latest_financial_package(catalog, date(2026, 8, 10))["period"], "2026-06-30")

    def test_archive_bytes_must_match_the_catalog(self):
        blob = b"trusted contents"
        package = {"size": len(blob), "md5": md5(blob, usedforsecurity=False).hexdigest()}
        verify_package(blob, package)
        with self.assertRaises(ValueError):
            verify_package(b"altered contents", package)
