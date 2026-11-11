from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import polars as pl

from import_offline_minute_zip_batch import ZipParquetEntry, _read_entry, _verified_missing_rows, main


def minutes(symbol="000001.SZ", volume=1.0, price=10.0):
    times = pl.datetime_range(
        datetime(2026, 4, 17, 9, 31), datetime(2026, 4, 17, 11, 30),
        interval="1m", eager=True,
    ).append(pl.datetime_range(
        datetime(2026, 4, 17, 13, 1), datetime(2026, 4, 17, 15),
        interval="1m", eager=True,
    ))
    return pl.DataFrame({
        "symbol": [symbol] * 240, "datetime": times,
        **{c: [price] * 240 for c in ["open", "high", "low", "close"]},
        "volume": [volume] * 240, "amount": [price * volume * 100] * 240,
    })


def daily(df):
    return df.group_by("symbol").agg(
        pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
        pl.col("close").last(), pl.col("volume").sum(), pl.col("amount").sum(),
    )


class VerifiedFillTests(unittest.TestCase):
    def test_numeric_daily_code_matches_exchange_suffixed_minute(self):
        source = minutes("600000.SH")
        reference = daily(source).with_columns(pl.lit("600000").alias("symbol"))
        accepted, meta = _verified_missing_rows(source, source.clear(), reference)
        self.assertEqual(meta["added_rows"], 240)
        self.assertTrue(accepted.equals(source))

    def test_shares_unit_and_market_filter_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "2026.zip"
            source = pl.concat([minutes("600000.SH"), minutes("000001.SZ")]).with_columns(
                (pl.col("amount") / 100).alias("amount")
            )
            buffer = io.BytesIO()
            source.write_parquet(buffer)
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("20260417.parquet", buffer.getvalue())
            entry = ZipParquetEntry(archive, "20260417.parquet", source["datetime"][0].date(), len(buffer.getvalue()))
            parsed, meta = _read_entry(entry, volume_unit="shares", markets={"SH"})
            self.assertEqual(parsed.height, 240)
            self.assertEqual(parsed["symbol"].unique().to_list(), ["600000.SH"])
            self.assertEqual(parsed["volume"].sum(), 240)
            self.assertEqual(meta["volume_unit"], "shares")

    def test_missing_symbol_is_added_and_repeat_is_idempotent(self):
        source = minutes()
        accepted, meta = _verified_missing_rows(source, source.clear(), daily(source))
        self.assertTrue(accepted.equals(source))
        self.assertEqual(meta["added_rows"], 240)
        repeated, _ = _verified_missing_rows(source, accepted, daily(source))
        self.assertTrue(repeated.is_empty())

    def test_existing_values_are_not_overwritten(self):
        source = minutes(price=11.0)
        old = minutes()
        accepted, meta = _verified_missing_rows(source, old, daily(source))
        self.assertTrue(accepted.is_empty())
        self.assertEqual(meta["existing_keys_preserved"], 240)
        self.assertEqual(old["close"][0], 10.0)

    def test_zero_trade_day_is_not_a_completed_day(self):
        source = minutes(volume=0.0)
        accepted, meta = _verified_missing_rows(source, source.clear(), daily(source))
        self.assertTrue(accepted.is_empty())
        self.assertEqual(meta["excluded_rows"], 240)

    def test_daily_close_mismatch_is_rejected(self):
        source = minutes()
        check = daily(source).with_columns(pl.lit(10.5).alias("close"))
        accepted, _ = _verified_missing_rows(source, source.clear(), check)
        self.assertTrue(accepted.is_empty())

    def test_daily_turnover_mismatch_is_rejected(self):
        source = minutes()
        check = daily(source).with_columns((pl.col("volume") * 1.02).alias("volume"))
        accepted, _ = _verified_missing_rows(source, source.clear(), check)
        self.assertTrue(accepted.is_empty())

    def test_missing_daily_is_rejected(self):
        source = minutes()
        accepted, _ = _verified_missing_rows(source, source.clear(), daily(source).clear())
        self.assertTrue(accepted.is_empty())

    def test_partial_existing_symbol_is_not_mixed_with_another_source(self):
        source = minutes()
        accepted, meta = _verified_missing_rows(source, source.head(20), daily(source))
        self.assertTrue(accepted.is_empty())
        self.assertEqual(meta["excluded_rows"], 220)

    def test_nonfinite_and_partial_source_are_rejected(self):
        source = minutes()
        check = daily(source)
        for bad in [source.head(239), source.with_columns(pl.lit(float("inf")).alias("high"))]:
            accepted, _ = _verified_missing_rows(bad, source.clear(), check)
            self.assertTrue(accepted.is_empty())

    def test_wrong_timestamp_grid_is_rejected(self):
        source = minutes()
        shifted = source.with_columns((pl.col("datetime") + pl.duration(seconds=30)))
        accepted, _ = _verified_missing_rows(shifted, source.clear(), daily(source))
        self.assertTrue(accepted.is_empty())

    def test_cli_preserves_backup_and_reports_actual_additions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = minutes("000002.SZ")
            incoming = minutes()
            target = root / "kline_minute/date=2026-04-17/part.parquet"
            target.parent.mkdir(parents=True)
            original.write_parquet(target)
            reference = root / "kline_daily/date=2026-04-17/part.parquet"
            reference.parent.mkdir(parents=True)
            daily(incoming).write_parquet(reference)
            buffer = io.BytesIO()
            incoming.write_parquet(buffer)
            archive = root / "input.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("20260417.parquet", buffer.getvalue())
            for index, expected_additions in enumerate([240, 0]):
                report = root / f"report{index}.json"
                argv = ["import", "--source-root", str(archive), "--data-dir", str(root),
                        "--mode", "verified-fill", "--report", str(report), "--summary-only"]
                with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
                audit = json.loads(report.read_text(encoding="utf-8"))
                self.assertEqual(audit["added_rows"], expected_additions)
                self.assertEqual(audit["written_rows"], expected_additions)
            persisted = pl.read_parquet(target)
            self.assertEqual(persisted.height, 480)
            self.assertTrue(persisted.filter(pl.col("symbol") == "000002.SZ").equals(original))
            backups = list((root / "import_backups").rglob("*.parquet"))
            self.assertEqual(len(backups), 1)
            self.assertTrue(pl.read_parquet(backups[0]).equals(original))


if __name__ == "__main__":
    unittest.main()
