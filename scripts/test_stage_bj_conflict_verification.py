import csv
import json
import tempfile
import unittest
from pathlib import Path

from stage_bj_conflict_verification import residual_rows, write_manifest


class StageBJConflictVerificationTests(unittest.TestCase):
    def test_stages_only_unresolved_complete_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.csv"
            audit.write_text(
                "status,code,date,bars,price_match,volume_match,amount_match,amount,minute_amount\n"
                "complete_conflict,920001,2024-01-15,240,true,true,false,110,100\n"
                "complete_conflict,920002,2024-01-15,240,false,true,false,110,100\n"
                "complete_conflict,920003,2024-01-15,240,true,true,false,110,100\n",
                encoding="utf-8",
            )
            coverage = root / "coverage.json"
            coverage.write_text(json.dumps({
                "items": {
                    "920003|2024-01-15": {
                        "symbol": "920003", "date": "2024-01-15",
                        "status": "regular_session_close_auction",
                    }
                }
            }), encoding="utf-8")

            rows, skipped = residual_rows(audit, coverage)

            self.assertEqual([row["symbol"] for row in rows], ["920001.BJ", "920002.BJ"])
            self.assertEqual(
                [row["reason"] for row in rows],
                ["daily_turnover_conflict", "price_path_conflict"],
            )
            self.assertEqual(skipped["regular_session_close_auction"], 1)

    def test_manifest_has_identity_and_conflict_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "nested" / "manifest.csv"
            write_manifest([{
                "symbol": "920001.BJ", "date": "2024-01-15",
                "status": "existing_values_conflict", "reason": "daily_turnover_conflict",
            }], manifest)
            with manifest.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["symbol"], "920001.BJ")
            self.assertEqual(row["date"], "2024-01-15")
            self.assertEqual(row["status"], "existing_values_conflict")


if __name__ == "__main__":
    unittest.main()
