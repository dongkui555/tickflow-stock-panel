from datetime import date, datetime, timedelta

import polars as pl
import pytest

from app.tickflow.repository import KlineRepository

DAY = date(2024, 9, 6)


@pytest.fixture
def repo(tmp_path):
    instance = KlineRepository.__new__(KlineRepository)
    instance._minute_glob = str(tmp_path / "kline_minute/**/*.parquet")
    instance._etf_minute_glob = str(tmp_path / "kline_etf_minute/**/*.parquet")
    for table, symbols in [("kline_minute", ["302132", "920001.BJ"]), ("kline_etf_minute", ["510300.SH"])]:
        for day in [DAY, DAY + timedelta(days=3)]:
            folder = tmp_path / table / f"date={day}"
            folder.mkdir(parents=True)
            pl.DataFrame({"symbol": symbols, "datetime": [datetime.combine(day, datetime.min.time()) + timedelta(hours=9, minutes=31)] * len(symbols),
                          **{c: [10.0] * len(symbols) for c in ["open", "high", "low", "close"]},
                          "volume": [1.0] * len(symbols), "amount": [1000.0] * len(symbols)}).write_parquet(folder / "part.parquet")
    return instance


@pytest.mark.parametrize("symbol", ["302132", "302132.SZ", "920001", "920001.BJ"])
def test_single_returns_local_rows_in_requested_symbol_format(repo, symbol):
    result = repo.get_minute(symbol, DAY)
    assert result.height == 1
    assert result["symbol"].to_list() == [symbol]
    assert result["datetime"].dt.date().to_list() == [DAY]


def test_batch_accepts_mixed_aliases(repo):
    result = repo.get_minute_batch(["302132.SZ", "920001"], DAY)
    assert result["symbol"].to_list() == ["302132.SZ", "920001"]
    assert result.height == 2


def test_single_and_batch_do_not_scan_unrelated_corrupt_partitions(repo, tmp_path):
    corrupt = tmp_path / "kline_minute/date=2024-09-05/part.parquet"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"broken")
    assert repo.get_minute("302132.SZ", DAY).height == 1
    assert repo.get_minute_batch(["302132.SZ"], DAY).height == 1


def test_range_preserves_aliases_and_date_bounds(repo):
    result = repo.get_minute_range(["302132.SZ"], DAY, DAY + timedelta(days=2))
    assert result.height == 1
    assert result["symbol"].to_list() == ["302132.SZ"]
    assert repo.get_minute_range(["302132.SZ"], DAY, DAY + timedelta(days=3)).height == 2
    assert repo.get_minute_range(["302132.SZ"], DAY, DAY - timedelta(days=1)).is_empty()


def test_assets_and_missing_dates_remain_separate(repo):
    assert repo.get_minute("510300", DAY, asset_type="etf").height == 1
    assert repo.get_minute("510300", DAY, asset_type="stock").is_empty()
    assert repo.get_minute("302132.SZ", DAY + timedelta(days=1)).is_empty()
    assert repo.get_minute_batch([], DAY).is_empty()
