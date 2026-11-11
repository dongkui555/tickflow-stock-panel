import json
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import polars as pl

from app.services.minute_repair import repair_minute_gaps
from import_staged_minute_gaps import StagedMinuteProvider, require_stopped


class StagedImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.day = date(2024, 9, 6)
        stamps = [datetime(2024, 9, 6, 9, 31) + timedelta(minutes=i) for i in range(120)]
        stamps += [datetime(2024, 9, 6, 13, 1) + timedelta(minutes=i) for i in range(120)]
        self.frame = pl.DataFrame({"symbol": ["302132.SZ"] * 240, "datetime": stamps,
                                   **{c: [10.0] * 240 for c in ["open", "high", "low", "close"]},
                                   "volume": [1.0] * 240, "amount": [1000.0] * 240})
        self.stage = self.root / "import_reports/stage"
        self.stage.mkdir(parents=True)
        self.frame.write_parquet(self.stage / "source.parquet")
        self.rows = [{"symbol": "302132.SZ", "date": str(self.day), "reason": "accepted", "file": "source.parquet"}]
        self.write_report()
        self.daily = self.root / f"kline_daily/date={self.day}/part.parquet"
        self.daily.parent.mkdir(parents=True)
        pl.DataFrame({"symbol": ["302132.SZ"], **{c: [10.0] for c in ["open", "high", "low", "close"]},
                      "volume": [24000.0], "amount": [240000.0]}).write_parquet(self.daily)
        self.minute = self.root / f"kline_minute/date={self.day}/part.parquet"
        self.minute.parent.mkdir(parents=True)
        self.old = self.frame.head(10)
        self.old.write_parquet(self.minute)
        self.repo = SimpleNamespace(store=SimpleNamespace(data_dir=self.root), _write_lock=RLock(),
                                    rebuild_views=Mock(), clear_cache=Mock())

    def write_report(self):
        (self.stage / "report.json").write_text(json.dumps(self.rows), encoding="utf-8")

    def run_repair(self):
        source = StagedMinuteProvider([self.stage])
        with patch("app.services.minute_repair.kline_sync._resolve_minute_provider") as resolve:
            result = repair_minute_gaps(self.repo, start_date=self.day, end_date=self.day,
                                       provider_name="staged_minute", source_provider=source,
                                       stock_days=list(source.files), repair_conflicts=False)
            resolve.assert_not_called()
        return result

    def test_add_only_backup_readback_and_idempotence(self):
        result = self.run_repair()
        self.assertEqual((result["added_rows"], result["replaced_rows"], result["removed_rows"]), (230, 0, 0))
        self.assertTrue(pl.read_parquet(self.minute).equals(self.frame))
        backup = next((self.root / "import_backups").glob("*/date=*/part.parquet"))
        self.assertTrue(pl.read_parquet(backup).equals(self.old))
        self.repo.rebuild_views.assert_called_once()
        self.repo.clear_cache.assert_called_once()
        self.assertEqual(self.run_repair()["added_rows"], 0)

    def test_conflicting_existing_record_is_not_overwritten(self):
        bad = self.old.with_columns(pl.lit(9.0).alias("close"))
        bad.write_parquet(self.minute)
        result = self.run_repair()
        self.assertEqual(result["added_rows"], 0)
        self.assertEqual(result["reasons"]["existing_values_conflict"], 1)
        self.assertTrue(pl.read_parquet(self.minute).equals(bad))

    def test_changed_daily_reference_is_rechecked(self):
        pl.read_parquet(self.daily).with_columns(pl.lit(11.0).alias("high")).write_parquet(self.daily)
        result = self.run_repair()
        self.assertEqual(result["added_rows"], 0)
        self.assertEqual(result["reasons"]["daily_price_conflict"], 1)

    def test_wrong_stock_is_rejected(self):
        self.frame.with_columns(pl.lit("000001.SZ").alias("symbol")).write_parquet(self.stage / "source.parquet")
        result = self.run_repair()
        self.assertEqual(result["added_rows"], 0)
        self.assertEqual(result["reasons"]["source_symbol_conflict"], 1)

    def test_wrong_date_is_rejected(self):
        self.frame.with_columns(pl.col("datetime") + timedelta(days=1)).write_parquet(self.stage / "source.parquet")
        source = StagedMinuteProvider([self.stage])
        with self.assertRaises(ValueError):
            source.get_minute(["302132"], datetime(2024, 9, 6, 9, 30), datetime(2024, 9, 6, 15))

    def test_staging_path_cannot_escape(self):
        self.rows[0]["file"] = "../escape.parquet"
        self.write_report()
        with self.assertRaises(ValueError):
            StagedMinuteProvider([self.stage])

    def test_only_accepted_reports_and_unique_stock_days(self):
        self.rows[0]["reason"] = "daily_price_conflict"
        self.write_report()
        self.assertEqual(StagedMinuteProvider([self.stage]).files, {})
        self.rows[0]["reason"] = "accepted"
        self.write_report()
        with self.assertRaises(ValueError):
            StagedMinuteProvider([self.stage, self.stage])

    def test_running_unknown_or_restarting_container_blocks_import(self):
        for state in [{"Running": True, "Restarting": False}, {"Running": False, "Restarting": True}, {}]:
            with patch("import_staged_minute_gaps.subprocess.run", return_value=SimpleNamespace(stdout=json.dumps(state))):
                with self.assertRaises(RuntimeError):
                    require_stopped("app")
        with patch("import_staged_minute_gaps.subprocess.run", return_value=SimpleNamespace(stdout='{"Running":false,"Restarting":false}')):
            require_stopped("app")


if __name__ == "__main__":
    unittest.main()
