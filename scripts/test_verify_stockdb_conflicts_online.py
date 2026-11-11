"""Classification gates for read-only three-way minute verification."""
import unittest

from verify_stockdb_conflicts_online import classify, matches


DAILY = {"open": 10.0, "high": 10.5, "low": 9.9, "close": 10.2,
         "volume": 1000000, "amount": 10200000}
OTHER = {**DAILY, "close": 10.3}


class CrossCheckTest(unittest.TestCase):
    def test_online_daily_and_minutes_confirm_local(self):
        self.assertEqual(classify(DAILY, DAILY, OTHER, "accepted", DAILY, "accepted"),
                         "replacement_candidate")

    def test_minute_sources_agree_but_daily_conflicts(self):
        self.assertEqual(classify(DAILY, OTHER, OTHER, "accepted", OTHER, "accepted"),
                         "minute_sources_agree_daily_conflict")

    def test_rate_limit_never_looks_like_empty_source(self):
        self.assertEqual(classify(DAILY, DAILY, OTHER, "accepted", None, "rate_limited"),
                         "online_minute_retry")

    def test_missing_daily_blocks_replacement(self):
        self.assertEqual(classify(DAILY, None, OTHER, "accepted", DAILY, "accepted"),
                         "online_daily_missing")

    def test_volume_unit_and_amount_tolerance(self):
        lots = {**DAILY, "volume": 10000, "amount": DAILY["amount"] + 100}
        self.assertTrue(matches(lots, DAILY))
        self.assertFalse(matches({**lots, "amount": DAILY["amount"] + 50000}, DAILY))


if __name__ == "__main__":
    unittest.main()
