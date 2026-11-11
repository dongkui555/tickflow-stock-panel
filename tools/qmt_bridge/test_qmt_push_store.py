from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from qmt_http_api import QmtPushStore


class QmtPushStoreTest(TestCase):
    def test_index_daily_push_validates_and_upserts_bars(self):
        with TemporaryDirectory() as folder:
            store = QmtPushStore(Path(folder) / "push.sqlite")
            result = store.ingest_daily_bars([
                {"symbol": "000001.SH", "date": "2026-09-11", "open": 10.0,
                 "high": 10.4, "low": 9.8, "close": 10.2, "volume": 1000, "amount": 10000},
                {"symbol": "000002.SH", "date": "2026-09-11", "open": 10.0,
                 "high": 9.0, "low": 9.8, "close": 10.2, "volume": 1000, "amount": 10000},
                {"symbol": "000003.SH", "date": "2099-01-01", "open": 10.0,
                 "high": 10.4, "low": 9.8, "close": 10.2, "volume": 1000, "amount": 10000},
            ])
            self.assertEqual(result["accepted"], 1)
            self.assertEqual(result["rejected"], 2)
            bars = store.klines("000001.SH", "1d", "20260911", "20260911", -1)
            self.assertEqual(len(bars), 1)
            self.assertEqual(bars[0]["close"], 10.2)
            self.assertEqual(store.capabilities()["push_daily_index_sync"]["accepted"], 1)

    def test_minute_push_validates_and_upserts_bars(self):
        with TemporaryDirectory() as folder:
            store = QmtPushStore(Path(folder) / "push.sqlite")
            result = store.ingest_minute_bars([
                {"symbol": "000001.SZ", "datetime": "2026-09-11 09:31:00",
                 "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1,
                 "volume": 1200, "amount": 12120},
                {"symbol": "000001.SZ", "datetime": "2026-09-11 09:32:00",
                 "open": 10.1, "high": 10.0, "low": 10.1, "close": 10.2,
                 "volume": 1, "amount": 1},
                {"symbol": "000001.SZ", "datetime": "2026-09-11 12:00:00",
                 "open": 10.1, "high": 10.2, "low": 10.0, "close": 10.1,
                 "volume": 1, "amount": 1},
            ])
            self.assertEqual(result["accepted"], 1)
            self.assertEqual(result["rejected"], 2)
            self.assertEqual(result["inserted"], 1)
            self.assertEqual(result["updated"], 0)
            self.assertEqual(result["duplicate"], 0)
            second = store.ingest_minute_bars([{
                "symbol": "000001.SZ", "datetime": "2026-09-11 09:31:00",
                "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1,
                "volume": 1200, "amount": 12120,
            }])
            self.assertEqual(second["duplicate"], 1)
            changed = store.ingest_minute_bars([{
                "symbol": "000001.SZ", "datetime": "2026-09-11 09:31:00",
                "open": 10.0, "high": 10.3, "low": 9.9, "close": 10.2,
                "volume": 1200, "amount": 12120,
            }])
            self.assertEqual(changed["updated"], 1)
            bars = store.klines("000001.SZ", "1m", "20260911", "20260911", -1)
            self.assertEqual(len(bars), 1)
            self.assertEqual(bars[0]["datetime"], "2026-09-11 09:31:00")
            self.assertEqual(bars[0]["amount"], 12120.0)
            capabilities = store.capabilities()
            self.assertFalse(capabilities["push_active"])
            self.assertEqual(capabilities["push_minute_rows"], 1)
            self.assertEqual(capabilities["push_minute_sync"]["accepted"], 1)

    def test_quotes_build_minute_daily_and_auction_cache(self):
        with TemporaryDirectory() as folder:
            store = QmtPushStore(Path(folder) / "push.sqlite")
            first = store.ingest_quotes([
                {"symbol": "000001.SH", "stime": "20260911092500", "lastPrice": 10.0,
                 "open": 10.0, "high": 10.0, "low": 10.0, "volume": 100, "pvolume": 10000, "amount": 100000},
                {"symbol": "399001.SZ", "stime": "20260911092500", "lastPrice": 20.0,
                 "open": 20.0, "high": 20.0, "low": 20.0, "volume": 200, "pvolume": 20000, "amount": 200000},
            ])
            self.assertEqual(first["accepted"], 2)
            self.assertEqual(len(store.auctions("2026-09-11", [])), 2)

            store.ingest_quotes([
                {"symbol": "000001.SH", "stime": "20260911093010", "lastPrice": 10.1,
                 "open": 10.0, "high": 10.1, "low": 9.9, "volume": 110, "amount": 111000},
                {"symbol": "000001.SH", "stime": "20260911093040", "lastPrice": 10.2,
                 "open": 10.0, "high": 10.2, "low": 9.9, "volume": 150, "amount": 151000},
            ])
            bars = store.klines("000001.SH", "1m", "20260911", "20260911", -1)
            self.assertEqual(len(bars), 1)
            self.assertEqual(bars[0]["datetime"], "2026-09-11 09:31:00")
            self.assertEqual(bars[0]["close"], 10.2)
            self.assertEqual(bars[0]["volume"], 50.0)
            self.assertEqual(bars[0]["amount"], 51000.0)
            daily = store.klines("000001.SH", "1d", "20260911", "20260911", -1)
            self.assertEqual(daily[0]["close"], 10.2)


if __name__ == "__main__":
    main()
