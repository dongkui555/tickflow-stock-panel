from datetime import date

import polars as pl
import pytest

from app.services.market_repair import (
    append_missing_stocks,
    daily_from_complete_minutes,
    validate_partition,
)


def test_daily_aggregation_requires_complete_session_and_converts_lots_to_shares():
    from datetime import datetime, timedelta

    stamps = [datetime(2025, 6, 3, 9, 31) + timedelta(minutes=i) for i in range(120)]
    stamps += [datetime(2025, 6, 3, 13, 1) + timedelta(minutes=i) for i in range(120)]
    df = pl.DataFrame(
        {
            "symbol": ["000001.SZ"] * 240,
            "datetime": stamps,
            "open": [10.0] * 240,
            "high": [10.0] * 240,
            "low": [10.0] * 240,
            "close": [10.0] * 240,
            "volume": [1.0] * 240,
            "amount": [1000.0] * 240,
        }
    )
    result = daily_from_complete_minutes(df, date(2025, 6, 3), ["000001"])
    assert result["volume"].item() == 24000.0
    assert result["amount"].item() == 240000.0
    assert result["data_source"].item() == "local_minute_aggregate"
    assert daily_from_complete_minutes(df.head(239), date(2025, 6, 3), ["000001"]).is_empty()
    assert daily_from_complete_minutes(
        df.with_columns(pl.lit(1.0).alias("amount")), date(2025, 6, 3), ["000001"]
    ).is_empty()


def test_sparse_day_repair_preserves_existing_values_and_deduplicates_aliases():
    old = pl.DataFrame({"symbol": ["000001.SZ"], "close": [10.0]})
    new = pl.DataFrame({"symbol": ["000001", "600000"], "close": [99.0, 12.0]})
    result = append_missing_stocks(old, new)
    assert result.to_dicts() == [
        {"symbol": "000001.SZ", "close": 10.0},
        {"symbol": "600000", "close": 12.0},
    ]


def frame():
    return pl.DataFrame(
        {
            "symbol": ["000001"],
            "date": [date(2026, 9, 3)],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000.0],
            "amount": [10500.0],
        }
    )


def test_accepts_exact_date_and_valid_daily_prices():
    assert validate_partition(frame(), date(2026, 9, 3), "date", ohlc=True).height == 1


def test_wrong_day_or_empty_response_is_not_written():
    with pytest.raises(ValueError, match="requested"):
        validate_partition(frame(), date(2026, 9, 2), "date")
    with pytest.raises(ValueError, match="empty"):
        validate_partition(pl.DataFrame(), date(2026, 9, 2), "date")


def test_invalid_prices_or_duplicate_symbols_rejected():
    with pytest.raises(ValueError, match="valid"):
        validate_partition(
            frame().with_columns(pl.lit(9.0).alias("high")), date(2026, 9, 3), "date", ohlc=True
        )
    with pytest.raises(ValueError, match="duplicate"):
        validate_partition(pl.concat([frame(), frame()]), date(2026, 9, 3), "date", ohlc=True)


def test_negative_net_flow_is_valid():
    df = pl.DataFrame({"trade_date": ["20260903"], "net_mf_amount": [-123.0]})
    assert validate_partition(df, date(2026, 9, 3), "trade_date").height == 1
