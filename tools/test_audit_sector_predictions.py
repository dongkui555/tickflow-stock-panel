"""Tests for diagnostic sample eligibility, not trading strategy performance."""
import unittest

from audit_sector_predictions import metrics


def validation(horizon, day, status, ret=0.01):
    return {"horizon_days": horizon, "verify_date": day, "status": status,
            "actual_ret_1d": ret}


class EligibilityTests(unittest.TestCase):
    def test_first_day_hit_is_not_mature_two_day_sample(self):
        rows = [{"validations": [validation(1, "2026-09-08", "hit")]}]
        self.assertEqual(metrics(rows, "2026-09-09")["n"], 0)
        self.assertEqual(metrics(rows, "2026-09-09", 1)["hits"], 1)

    def test_current_day_is_excluded(self):
        rows = [{"validations": [validation(1, "2026-09-09", "hit")]}]
        self.assertEqual(metrics(rows, "2026-09-09", 1)["n"], 0)

    def test_both_results_required_and_count_once(self):
        rows = [{"validations": [validation(1, "2026-09-07", "miss"),
                                 validation(2, "2026-09-08", "hit")]}]
        self.assertEqual(metrics(rows, "2026-09-09")["n"], 1)
        self.assertEqual(metrics(rows, "2026-09-09")["hits"], 1)

    def test_missing_data_is_not_a_prediction_failure(self):
        rows = [{"validations": [validation(1, "2026-09-07", "missing"),
                                 validation(2, "2026-09-08", "miss")]}]
        self.assertEqual(metrics(rows, "2026-09-09")["n"], 0)
        self.assertEqual(metrics(rows, "2026-09-09", 1)["n"], 0)

    def test_return_keeps_negative_sign(self):
        rows = [{"validations": [validation(1, "2026-09-08", "miss", -.02)]}]
        result = metrics(rows, "2026-09-09", 1)
        self.assertEqual(result["hits"], 0)
        self.assertEqual(result["mean_actual_day_return"], -.02)


if __name__ == "__main__":
    unittest.main()
