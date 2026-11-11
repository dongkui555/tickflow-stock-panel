from datetime import date
import unittest

import polars as pl

from audit_history_semantics import daily_flags


class DailyFlagsTests(unittest.TestCase):
    def frame(self, **overrides):
        row = {"symbol": "000001", "date": date(2026, 9, 8), "open": 10., "high": 11.,
               "low": 9., "close": 10., "volume": 100., "amount": 1000.}
        return pl.DataFrame([{**row, **overrides}])

    def test_accepts_consistent_share_and_lot_units(self):
        for amount in [1000., 100000.]:
            self.assertFalse(daily_flags(self.frame(amount=amount))["vwap_outside_range"][0])

    def test_flags_impossible_turnover(self):
        self.assertTrue(daily_flags(self.frame(amount=5000.))["vwap_outside_range"][0])

    def test_flags_canonical_alias_duplicate(self):
        rows = pl.concat([self.frame(), self.frame(symbol="000001.SZ")])
        self.assertEqual(daily_flags(rows)["duplicate_key"].sum(), 2)

    def test_short_codes_are_not_grouped_as_null_duplicates(self):
        rows = daily_flags(pl.concat([self.frame(symbol="1"), self.frame(symbol="2")]))
        self.assertEqual(rows["symbol"].to_list(), ["000001.SZ", "000002.SZ"])
        self.assertEqual(rows["duplicate_key"].sum(), 0)
        self.assertEqual(rows["legacy_short_code"].sum(), 2)

    def test_zero_turnover_not_assumed_error(self):
        self.assertFalse(daily_flags(self.frame(volume=0., amount=0.))["vwap_outside_range"][0])

    def test_flags_negative_turnover_and_sub_cent(self):
        row = daily_flags(self.frame(volume=-1., close=0.001))
        self.assertTrue(row["negative_turnover"][0])
        self.assertTrue(row["sub_cent_price"][0])


if __name__ == "__main__":
    unittest.main()
