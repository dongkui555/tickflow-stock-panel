from datetime import date

import polars as pl

from app.services.daily_aliases import deduplicate_verified_aliases


def sample():
    return pl.DataFrame({"symbol": ["000001", "000001.SZ"], "date": [date(2026, 8, 17)] * 2,
                         "open": [11.2] * 2, "high": [11.22] * 2, "low": [11.07] * 2,
                         "close": [11.1] * 2, "volume": [92904344., 929043.],
                         "amount": [1031951473.98, 1031951473.]})


def test_verified_alias_keeps_precision_and_is_idempotent():
    clean, report = deduplicate_verified_aliases(sample())
    assert clean["symbol"].to_list() == ["000001"]
    assert clean["volume"][0] == 92904344.
    assert report[0]["removed"] == 1
    assert deduplicate_verified_aliases(clean)[0].equals(clean)


def test_price_or_turnover_conflict_is_not_discarded():
    for column, value in [("open", 10.), ("volume", 100.), ("amount", 100.)]:
        source = sample().with_columns(pl.when(pl.col("symbol") == "000001.SZ")
                                      .then(value).otherwise(pl.col(column)).alias(column))
        clean, report = deduplicate_verified_aliases(source)
        assert clean.equals(source)
        assert report[0]["status"] == "values_conflict"


def test_unknown_identifier_is_preserved():
    source = sample().with_columns(pl.lit("T00018").alias("symbol"))
    assert deduplicate_verified_aliases(source)[0].equals(source)


def test_foreign_exchange_is_not_treated_as_stock_alias():
    source = sample().with_columns(pl.col("symbol").str.replace("SZ", "SH"))
    clean, report = deduplicate_verified_aliases(source)
    assert clean.equals(source)
    assert report[0]["status"] == "values_conflict"


def test_legacy_daily_codes_match_minutes_without_guessing_unknown_ids():
    from app.services.minute_repair import keyed
    source = pl.DataFrame({"symbol": ["1", "000001.SZ", "920001.BJ", "T00018"]})
    assert keyed(source)["_code"].to_list() == ["000001", "000001", "920001", None]


def test_stock_write_deduplicates_but_index_is_separate(tmp_path):
    from app.tickflow.repository import DataStore, KlineRepository
    repo = KlineRepository(DataStore(tmp_path))
    pl.DataFrame({"symbol": ["000001.SZ"], "code": ["000001"], "name": ["平安银行"]}).write_parquet(
        tmp_path / "instruments/instruments.parquet"
    )
    repo._refresh_instruments()
    repo.save_index_instruments(pl.DataFrame({"symbol": ["000001.SH"], "code": ["000001"], "name": ["上证指数"]}))
    repo.append_daily(sample().head(1))
    repo.append_daily(sample().tail(1))
    stock = pl.read_parquet(tmp_path / "kline_daily/date=2026-08-17/part.parquet")
    assert stock.height == 1
    assert stock["volume"][0] == sample()["volume"][0]
    assert stock["amount"][0] == sample()["amount"][0]
    repo.append_daily(sample().tail(1))
    assert pl.read_parquet(tmp_path / "kline_daily/date=2026-08-17/part.parquet").equals(stock)
    repo.append_index_daily(sample().head(1).with_columns(pl.lit("000001.SH").alias("symbol")))
    assert (tmp_path / "kline_index_daily/date=2026-08-17/part.parquet").exists()


def test_explicit_history_factor_plan_backfills_head_and_interior(tmp_path):
    from app.services import extend_history
    path = tmp_path / "adj_factor/all.parquet"
    path.parent.mkdir()
    pl.DataFrame({"trade_date": [date(2026, 7, 28), date(2026, 9, 9)]}).write_parquet(path)
    ranges = extend_history._historical_adj_segments(tmp_path, date(2024, 9, 9), date(2026, 9, 9))
    assert any(start == date(2024, 9, 9) for start, end in ranges)
    path = tmp_path / "kline_daily/date=2026-07-30/part.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"date": [date(2026, 7, 30)]}).write_parquet(path)
    ranges = extend_history._historical_adj_segments(tmp_path, date(2026, 7, 28), date(2026, 9, 9), cumulative=True)
    assert any(start <= date(2026, 7, 30) <= end for start, end in ranges)
    event_ranges = extend_history._historical_adj_segments(tmp_path, date(2026, 7, 28), date(2026, 9, 9))
    assert not any(start <= date(2026, 7, 30) <= end for start, end in event_ranges)
