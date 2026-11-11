from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import polars as pl

from app.services import minute_gap_queue


DAY = date(2026, 5, 8)


def _bars(symbol="000001.SZ", count=240):
    stamps = [datetime(2026, 5, 8, 9, 31) + timedelta(minutes=i) for i in range(120)]
    stamps += [datetime(2026, 5, 8, 13, 1) + timedelta(minutes=i) for i in range(120)]
    stamps = stamps[:count]
    return pl.DataFrame({
        "symbol": [symbol] * count,
        "datetime": stamps,
        "open": [10.0] * count,
        "high": [10.0] * count,
        "low": [10.0] * count,
        "close": [10.0] * count,
        "volume": [1.0] * count,
        "amount": [1000.0] * count,
    })


class MinuteGapQueueTests(TestCase):
    def test_queue_stops_market_closed_and_confirmed_suspension_before_source_requests(self):
      with TemporaryDirectory() as folder:
        root = Path(folder)
        calendar = root / "market_calendar"
        calendar.mkdir(parents=True)
        pl.DataFrame({
            "trade_date": ["2026-05-07", "2026-05-08"],
            "is_open": [0, 1],
        }).write_parquet(calendar / "trade_calendar.parquet")
        history = root / "instruments_history"
        history.mkdir(parents=True)
        pl.DataFrame({
            "symbol": ["000002.SZ"],
            "suspend_date": ["2026-05-08"],
            "resume_date": ["2026-05-09"],
        }).write_parquet(history / "suspensions.parquet")
        daily = root / "kline_daily/date=2026-05-08/part.parquet"
        daily.parent.mkdir(parents=True)
        pl.DataFrame([
            {"symbol": "000001.SZ", "open": 10.0, "high": 10.0, "low": 10.0,
             "close": 10.0, "volume": 100.0, "amount": 1000.0},
            {"symbol": "000002.SZ", "open": 10.0, "high": 10.0, "low": 10.0,
             "close": 10.0, "volume": 0.0, "amount": 0.0},
        ]).write_parquet(daily)

        result = minute_gap_queue.ensure_date(
            root, ["000001.SZ", "000002.SZ"], date(2026, 5, 8)
        )

        assert result["queued"] == 1
        assert minute_gap_queue.summary(root)["suspended"]["count"] == 1

        closed_daily = root / "kline_daily/date=2026-05-07/part.parquet"
        closed_daily.parent.mkdir(parents=True)
        pl.DataFrame([{
            "symbol": "000003.SZ", "open": 10.0, "high": 10.0, "low": 10.0,
            "close": 10.0, "volume": 100.0, "amount": 1000.0,
        }]).write_parquet(closed_daily)
        closed = minute_gap_queue.ensure_date(root, ["000003.SZ"], date(2026, 5, 7))

        assert closed["terminal"] == 1
        assert minute_gap_queue.summary(root)["market_closed"]["count"] == 1

    def test_queue_is_stock_day_scoped_and_skips_no_trade(self):
      with TemporaryDirectory() as folder:
        root = Path(folder)
        daily = root / f"kline_daily/date={DAY}/part.parquet"
        daily.parent.mkdir(parents=True)
        pl.DataFrame([
            {"symbol": "000001.SZ", "open": 10.0, "high": 10.0, "low": 10.0,
             "close": 10.0, "volume": 239.0, "amount": 239000.0},
            {"symbol": "000002.SZ", "open": 10.0, "high": 10.0, "low": 10.0,
             "close": 10.0, "volume": 0.0, "amount": 0.0},
        ]).write_parquet(daily)
        minute = root / f"kline_minute/date={DAY}/part.parquet"
        minute.parent.mkdir(parents=True)
        _bars(count=239).write_parquet(minute)

        result = minute_gap_queue.ensure_date(root, ["000001.SZ", "000002.SZ"], DAY)
        self.assertEqual(result, {"date": DAY.isoformat(), "queued": 1, "terminal": 1, "complete": 0})
        self.assertEqual(minute_gap_queue.pending_batch(root, DAY, limit=10), ["000001.SZ"])
        self.assertEqual(minute_gap_queue.summary(root)["pending"]["count"], 1)

        minute_gap_queue.record_statuses(root, [{
            "symbol": "000001.SZ", "date": DAY.isoformat(), "status": "complete",
            "expected_minutes": 240, "actual_minutes": 240,
        }])
        self.assertEqual(minute_gap_queue.pending_batch(root, DAY, limit=10), [])
        self.assertEqual(minute_gap_queue.summary(root)["complete"]["count"], 1)

    def test_regular_session_verdict_is_preserved_for_a_complete_240_bar_day(self):
      with TemporaryDirectory() as folder:
        root = Path(folder)
        daily = root / f"kline_daily/date={DAY}/part.parquet"
        daily.parent.mkdir(parents=True)
        pl.DataFrame([{
            "symbol": "920001.BJ", "open": 10.0, "high": 10.0,
            "low": 10.0, "close": 10.0, "volume": 240.0,
            "amount": 240000.0,
        }]).write_parquet(daily)
        minute = root / f"kline_minute/date={DAY}/part.parquet"
        minute.parent.mkdir(parents=True)
        _bars(symbol="920001.BJ", count=240).write_parquet(minute)
        minute_gap_queue.record_statuses(root, [{
            "symbol": "920001.BJ", "date": DAY.isoformat(),
            "status": "regular_session", "expected_minutes": 240,
            "actual_minutes": 240, "missing_minutes": 0,
            "reason": "日K成交额包含盘后固定价格交易", "source": "local",
        }])

        result = minute_gap_queue.ensure_date(root, ["920001.BJ"], DAY)

        self.assertEqual(result["terminal"], 1)
        self.assertEqual(minute_gap_queue.pending_batch(root, DAY, limit=10), [])
        self.assertEqual(minute_gap_queue.summary(root)["regular_session"]["count"], 1)

    def test_summary_exposes_actionable_categories_instead_of_only_status_counts(self):
      with TemporaryDirectory() as folder:
        root = Path(folder)
        minute_gap_queue.record_statuses(root, [
            {"symbol": "000001.SZ", "date": DAY.isoformat(),
             "status": "unknown", "reason": "source_empty", "actual_minutes": 0},
            {"symbol": "000002.SZ", "date": DAY.isoformat(),
             "status": "unknown", "reason": "source_incomplete", "actual_minutes": 239},
            {"symbol": "000003.SZ", "date": DAY.isoformat(),
             "status": "unknown", "reason": "daily_turnover_conflict", "actual_minutes": 240},
        ])
        minute_gap_queue.record_cross_source_results(root, [{
            "symbol": "000004.SZ", "date": DAY.isoformat(),
            "status": "minute_import_candidate", "reason": "verified_by_three_sources",
            "actual_minutes": 240,
        }])

        summary = minute_gap_queue.summary(root)

        assert summary["categories"]["source_empty"]["count"] == 1
        assert summary["categories"]["source_incomplete"]["count"] == 1
        assert summary["categories"]["conflict"]["count"] == 1
        assert summary["categories"]["verified_candidate"]["count"] == 1
        assert summary["categories"]["pending_gap"]["count"] == 0
