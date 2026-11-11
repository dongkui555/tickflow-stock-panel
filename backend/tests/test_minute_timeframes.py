from datetime import datetime, timedelta

import polars as pl
import pytest

from app.services.minute_timeframes import aggregate_minutes, five_minute_confirmation


def session(day=datetime(2026, 9, 4)):
    stamps = [day.replace(hour=h, minute=m) + timedelta(minutes=i)
              for h, m in ((9, 30), (13, 0)) for i in range(1, 121)]
    return pl.DataFrame({"symbol": ["000001"] * 240, "datetime": stamps,
                         "open": [10.] * 240, "high": [11.] * 240,
                         "low": [9.] * 240, "close": [10.5] * 240,
                         "volume": [100.] * 240, "amount": [1050.] * 240})


@pytest.mark.parametrize("period,count", [(5, 48), (15, 16), (30, 8), (60, 4)])
def test_session_alignment_and_turnover(period, count):
    bars = aggregate_minutes(session(), period, datetime(2026, 9, 4, 15, 1))
    assert bars.height == count
    assert bars["complete"].all()
    assert bars["volume"].sum() == 24000
    assert bars["amount"].sum() == 252000
    if period == 60:
        assert [d.strftime("%H:%M") for d in bars["datetime"]] == ["10:30", "11:30", "14:00", "15:00"]


def test_gaps_duplicates_and_unfinished_bars_are_not_signals():
    frame = session().head(17)
    bars = aggregate_minutes(frame, 15, datetime(2026, 9, 4, 9, 47))
    assert bars["complete"].to_list() == [True, False]
    assert bars["quality"].to_list() == ["complete", "forming"]
    missing = aggregate_minutes(frame.filter(pl.col("datetime").dt.minute() != 35), 15,
                                datetime(2026, 9, 4, 15, 0))
    assert not missing["complete"].any()
    duplicate = aggregate_minutes(pl.concat([session(), session().head(1)]), 60,
                                  datetime(2026, 9, 4, 15, 0))
    assert duplicate["quality"][0] == "invalid"


def test_opening_auction_and_timezone():
    opening = session().head(1).with_columns(pl.lit(datetime(2026, 9, 4, 9, 30)).alias("datetime"))
    frame = pl.concat([opening, session()]).with_columns(
        pl.col("datetime").dt.replace_time_zone("Asia/Shanghai").dt.convert_time_zone("UTC"))
    bars = aggregate_minutes(frame, 60, datetime(2026, 9, 4, 15, 1))
    assert bars["complete"].all()
    assert bars["volume"].sum() == 24100


def test_confirmation_fails_closed_for_missing_stale_and_after_close():
    assert five_minute_confirmation(pl.DataFrame(), datetime(2026, 9, 4, 10))["status"] == "missing"
    bars = aggregate_minutes(session().head(30), 5, datetime(2026, 9, 4, 10))
    assert five_minute_confirmation(bars, datetime(2026, 9, 4, 10, 12))["status"] == "stale"
    assert five_minute_confirmation(bars, datetime(2026, 9, 4, 19))["status"] == "closed"


def test_no_future_bar_used_and_breakout_confirmation():
    bars = aggregate_minutes(session(), 5, datetime(2026, 9, 4, 15))
    bars = bars.with_columns(
        pl.when(pl.col("datetime") == datetime(2026, 9, 4, 9, 45)).then(11.1).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.col("datetime") == datetime(2026, 9, 4, 9, 45)).then(600.).otherwise(pl.col("volume")).alias("volume"))
    result = five_minute_confirmation(bars, datetime(2026, 9, 4, 9, 46))
    assert result["status"] == "confirmed"
    assert result["bar_end"] == "2026-09-04T09:45:00"


def test_unsupported_period_and_invalid_price():
    with pytest.raises(ValueError):
        aggregate_minutes(session(), 7, datetime(2026, 9, 4, 15))
    bars = aggregate_minutes(session().with_columns(pl.lit(0.).alias("close")), 15,
                             datetime(2026, 9, 4, 15))
    assert not bars["complete"].any()


def test_aware_clock_and_closed_session_do_not_read_store():
    from app.market_time import CN_TZ
    from app.services.minute_timeframes import load_five_minute_confirmations
    bars = aggregate_minutes(session(), 5, datetime(2026, 9, 4, 15))
    assert five_minute_confirmation(bars, datetime(2026, 9, 4, 10, tzinfo=CN_TZ))["status"] == "waiting"
    assert load_five_minute_confirmations(object(), ["000001"], datetime(2026, 9, 4, 19, tzinfo=CN_TZ))["000001"]["status"] == "closed"


def test_chart_missing_whole_bucket_restarts_indicators():
    from types import SimpleNamespace
    from app.services.minute_timeframes import chart_minutes
    frame = session().filter(~pl.col("datetime").is_between(datetime(2026, 9, 4, 10, 1), datetime(2026, 9, 4, 10, 5)))
    repo = SimpleNamespace(get_minute_range=lambda *a: frame)
    result = chart_minutes(repo, "000001", "stock", 5, 1, datetime(2026, 9, 4, 15))
    next_bar = next(row for row in result["rows"] if row["date"] == datetime(2026, 9, 4, 10, 10))
    assert next_bar["ma5"] is None
