import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import Mock, patch

from qmt_http_api import QmtApi

DAY = datetime(2026, 9, 22, 9, 31)
TZ = timezone(timedelta(hours=8))


def record(stamp, lots=7, remainder=25):
    return struct.pack(
        "<8IQI2f3I", int(stamp.replace(tzinfo=TZ).timestamp()),
        10000, 10000, 10000, 10000, 0, lots, 399,
        (lots * 100 + remainder) * 10, 0, 1.0, 1.0, 0, remainder, 32761,
    )


def row(stamp=DAY, volume=7.0):
    return {"symbol": "920701.BJ", "datetime": str(stamp), "open": 10.0,
            "high": 10.0, "low": 10.0, "close": 10.0, "volume": volume, "amount": 7250.0}


class QmtPrecisionTest(TestCase):
    def request(self, root, rows):
        api = QmtApi(Mock(), Mock(), qmt_root=root)
        with patch.object(api, "_kline", return_value=rows):
            return api.kline("920701.BJ", "1m", "20260922000000", "20260922235959", -1)

    def file(self, root, records=None):
        path = root / "BJ/60/920701.DAT"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"\0" * 8 + (records if records is not None else record(DAY)))
        return path

    def test_corrects_only_verified_fractional_lot_volume(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.file(root)
            original = row()
            result = self.request(root, [original])
            self.assertEqual(result[0]["volume"], 7.25)
            self.assertEqual(result[0]["amount"], original["amount"])
            self.assertEqual(result[0]["volume_source"], "qmt_dat_exact")
            self.assertEqual(original["volume"], 7.0)

    def test_never_appends_missing_session_minutes(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.file(root, record(DAY) + record(DAY + timedelta(minutes=1)))
            result = self.request(root, [row()])
            self.assertEqual(len(result), 1)

    def test_preserves_already_precise_volume(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.file(root)
            self.assertEqual(self.request(root, [row(volume=7.25)])[0]["volume"], 7.25)

    def test_missing_dat_record_fails_without_old_volume_fallback(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.file(root)
            with self.assertRaisesRegex(ValueError, "precision unavailable"):
                self.request(root, [row(DAY + timedelta(minutes=1))])

    def test_invalid_remainder_is_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.file(root, record(DAY, remainder=100))
            with self.assertRaisesRegex(ValueError, "remainder"):
                self.request(root, [row()])

    def test_disagreeing_source_fields_are_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.file(root)
            for field, value in (("symbol", "920701.SH"), ("close", 10.01),
                                 ("amount", 7260.0), ("volume", 8.0)):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "conflict"):
                    self.request(root, [{**row(), field: value}])

    def test_unconfigured_bridge_retains_existing_response(self):
        api = QmtApi(Mock(), Mock())
        with patch.object(api, "_kline", return_value=[row()]) as fetch:
            self.assertEqual(api.kline("920701.BJ", "1m", "20260922", "20260922", -1), [row()])
        fetch.assert_called_once()


if __name__ == "__main__":
    main()
