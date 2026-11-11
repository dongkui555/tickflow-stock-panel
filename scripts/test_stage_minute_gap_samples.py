import json
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import polars as pl

from stage_minute_gap_samples import stage_one
import stage_minute_gap_samples as staging


class StageMinuteSamplesTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "staged"
        self.output.mkdir()
        self.day = date(2024, 9, 6)
        stamps = [datetime(2024, 9, 6, 9, 31) + timedelta(minutes=i) for i in range(120)]
        stamps += [datetime(2024, 9, 6, 13, 1) + timedelta(minutes=i) for i in range(120)]
        self.frame = pl.DataFrame({"symbol": ["302132.SZ"] * 240, "datetime": stamps,
                                   **{c: [10.0] * 240 for c in ["open", "high", "low", "close"]},
                                   "volume": [1.0] * 240, "amount": [1000.0] * 240})
        self.daily = self.root / "kline_daily" / f"date={self.day}" / "part.parquet"
        self.minute = self.root / "kline_minute" / f"date={self.day}" / "part.parquet"
        self.daily.parent.mkdir(parents=True)
        pl.DataFrame({"symbol": ["302132.SZ"], "date": [self.day],
                      **{c: [10.0] for c in ["open", "high", "low", "close"]},
                      "volume": [24000.0], "amount": [240000.0]}).write_parquet(self.daily)

    def run_stage(self, frame=None, issues=None):
        source = self.frame if frame is None else frame
        provider = SimpleNamespace(get_minute_diagnostics=lambda *a, **kw: (source, issues or []))
        daily_before = self.daily.read_bytes()
        minute_before = self.minute.read_bytes() if self.minute.exists() else None
        result = stage_one(provider, self.root, self.output, "302132.SZ", self.day)
        self.assertEqual(self.daily.read_bytes(), daily_before)
        self.assertEqual(self.minute.read_bytes() if self.minute.exists() else None, minute_before)
        return result

    def write_existing(self, frame):
        self.minute.parent.mkdir(parents=True)
        frame.write_parquet(self.minute)

    def test_valid_download_only_writes_staging(self):
        result = self.run_stage()
        self.assertEqual(result["addable_regular_rows"], 240)
        self.assertEqual(result["reason"], "accepted")
        self.assertTrue(pl.read_parquet(self.output / result["file"]).equals(self.frame))
        self.assertFalse(self.minute.exists())

    def test_existing_rows_not_counted_as_new(self):
        self.write_existing(self.frame.head(139))
        result = self.run_stage()
        self.assertEqual(result["addable_rows"], 101)
        self.assertEqual(result["staged_rows"], 240)

    def test_existing_conflict_is_not_staged(self):
        self.write_existing(self.frame.head(1).with_columns(pl.lit(9.0).alias("open")))
        result = self.run_stage()
        self.assertEqual(result["reason"], "existing_values_conflict")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_source_daily_conflict_is_not_staged(self):
        result = self.run_stage(self.frame.with_columns(pl.lit(11.0).alias("high")))
        self.assertEqual(result["reason"], "daily_price_conflict")
        self.assertEqual(list(self.output.iterdir()), [])

    def test_wrong_stock_is_rejected(self):
        result = self.run_stage(self.frame.with_columns(pl.lit("000001.SZ").alias("symbol")))
        self.assertEqual(result["reason"], "source_symbol_conflict")

    def test_incomplete_session_is_rejected(self):
        result = self.run_stage(self.frame.head(239))
        self.assertEqual(result["reason"], "source_incomplete")

    def test_upstream_error_preserves_reason(self):
        result = self.run_stage(self.frame.clear(), [{"date": str(self.day), "reason": "upstream_unavailable",
                                                    "http_status": 503, "retryable": True}])
        self.assertEqual(result["http_status"], 503)
        self.assertEqual(result["staged_rows"], 0)

    def test_already_complete_is_not_staged(self):
        self.write_existing(self.frame)
        result = self.run_stage()
        self.assertEqual(result["reason"], "already_complete")
        self.assertEqual(result["addable_rows"], 0)
        self.assertEqual(list(self.output.iterdir()), [])

    def run_manifest(self, outcomes):
        manifest = pl.DataFrame({"symbol": [f"{i:06d}.SZ" for i in range(20)],
                                 "date": [str(self.day)] * 20})
        with patch.object(staging, "stage_one", side_effect=outcomes) as call, patch("builtins.print"):
            summary = staging.stage_manifest(None, self.root, self.output, manifest)
        return summary, call.call_count

    def test_consecutive_outages_stop_and_preserve_success(self):
        ok = {"reason": "accepted", "addable_rows": 240}
        bad = {"reason": "upstream_unavailable", "http_status": 503, "retryable": True}
        summary, count = self.run_manifest([ok] + [bad] * 5)
        self.assertEqual(count, 6)
        self.assertEqual(summary["deferred_stock_days"], 14)
        self.assertFalse(summary["complete"])
        saved = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(saved), 6)
        self.assertEqual(saved[0]["reason"], "accepted")
        self.assertEqual(summary["addable_rows"], 240)

    def test_validation_response_resets_transport_failure_streak(self):
        bad = {"reason": "request_timeout", "retryable": True}
        conflict = {"reason": "daily_price_conflict", "retryable": False}
        summary, count = self.run_manifest([bad] * 4 + [conflict] + [bad] * 5)
        self.assertEqual(count, 10)
        self.assertEqual(summary["deferred_stock_days"], 10)

    def test_rate_limit_stops_without_requesting_another_stock(self):
        summary, count = self.run_manifest([{"reason": "rate_limited", "retryable": True}])
        self.assertEqual(count, 1)
        self.assertEqual(summary["stop_reason"], "rate_limited")

    def test_permission_denied_stops_without_retry(self):
        summary, count = self.run_manifest([{"reason": "permission_denied", "retryable": False}])
        self.assertEqual(count, 1)
        self.assertFalse(summary["complete"])

    def test_finished_checks_are_not_reported_as_complete_import(self):
        summary, count = self.run_manifest([{"reason": "daily_price_conflict"}] * 20)
        self.assertEqual(count, 20)
        self.assertEqual(summary["deferred_stock_days"], 0)
        self.assertFalse(summary["complete"])


if __name__ == "__main__":
    unittest.main()
