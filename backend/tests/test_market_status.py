from datetime import date

import polars as pl

from app.services.market_status import (
    historical_suspension_status,
    market_day_status,
)


def test_market_day_status_uses_persisted_calendar_before_local_partitions(tmp_path):
    calendar = tmp_path / "market_calendar"
    calendar.mkdir(parents=True)
    pl.DataFrame({
        "trade_date": ["2026-10-01", "2026-10-02"],
        "is_open": [0, 1],
    }).write_parquet(calendar / "trade_calendar.parquet")

    assert market_day_status(tmp_path, date(2026, 10, 1))["status"] == "market_closed"
    assert market_day_status(tmp_path, date(2026, 10, 2))["status"] == "open"


def test_market_day_status_does_not_treat_missing_calendar_as_closed(tmp_path):
    result = market_day_status(tmp_path, date(2026, 10, 2))

    assert result["status"] == "unknown"
    assert result["reason"] == "market_calendar_missing"


def test_historical_suspension_requires_identity_and_effective_interval(tmp_path):
    history = tmp_path / "instruments_history"
    history.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["000001.SZ"],
        "suspend_date": ["2026-10-02"],
        "resume_date": ["2026-10-03"],
        "reason": ["重大事项停牌"],
    }).write_parquet(history / "suspensions.parquet")

    suspended = historical_suspension_status(tmp_path, "000001.SZ", date(2026, 10, 2))
    wrong_market = historical_suspension_status(tmp_path, "000001.SH", date(2026, 10, 2))
    outside = historical_suspension_status(tmp_path, "000001.SZ", date(2026, 10, 4))

    assert suspended["status"] == "suspended"
    assert suspended["reason"] == "重大事项停牌"
    assert wrong_market["status"] == "unknown"
    assert outside["status"] == "not_suspended"
