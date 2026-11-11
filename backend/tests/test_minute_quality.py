from datetime import timedelta

import polars as pl
import pytest
from test_minute_repair import DAY, bars, reference

from app.services.minute_quality import filter_minute_sessions


def test_gap_removes_whole_symbol_day_not_other_symbols_or_following_day():
    frame = pl.concat([bars().head(239), bars('000002.SZ'), bars().with_columns(pl.col('datetime') + timedelta(days=1))])
    valid, issues = filter_minute_sessions(frame)
    assert valid.height == 480
    assert issues == [{'date': str(DAY), 'symbol': '000001.SZ', 'reason': 'incomplete_session'}]


@pytest.mark.parametrize('bad', ['duplicate', 'timezone', 'price', 'negative'])
def test_bad_sessions_are_rejected(bad):
    frame = bars()
    if bad == 'duplicate':
        frame = pl.concat([frame, frame.head(1)])
    if bad == 'timezone':
        frame = frame.with_columns(pl.col('datetime') - timedelta(hours=8))
    if bad == 'price':
        frame = frame.with_columns(pl.lit(0.).alias('close'))
    if bad == 'negative':
        frame = frame.with_columns(pl.lit(-1.).alias('amount'))
    valid, issues = filter_minute_sessions(frame)
    assert valid.is_empty()
    assert len(issues) == 1


def test_full_count_does_not_hide_daily_price_conflict():
    daily = pl.DataFrame([reference() | {'date': DAY, 'close': 11.}])
    valid, issues = filter_minute_sessions(bars(), daily)
    assert valid.is_empty()
    assert issues[0]['reason'] == 'daily_price_or_turnover_conflict'
    valid, issues = filter_minute_sessions(bars(), pl.DataFrame([reference() | {'date': DAY}]))
    assert valid.height == 240
    assert issues == []


def test_optional_auction_is_not_a_missing_regular_minute():
    opening = bars().head(1).with_columns(pl.col('datetime') - timedelta(minutes=1))
    valid, issues = filter_minute_sessions(pl.concat([opening, bars()]))
    assert valid.height == 241
    assert not issues


def test_equivalent_symbol_aliases_are_not_merged_into_false_turnover_conflict():
    bare = bars("000001")
    native = bars("000001.SZ").head(239)
    daily = pl.DataFrame([reference() | {"date": DAY}])
    valid, issues = filter_minute_sessions(pl.concat([bare, native]), daily)
    assert valid.height == 240
    assert valid["symbol"].unique().to_list() == ["000001"]
    assert issues == []


def test_repository_matches_code_formats_and_revalidates_after_file_change(tmp_path):
    from types import SimpleNamespace

    from app.tickflow.repository import KlineRepository
    minute = tmp_path / f'kline_minute/date={DAY}/part.parquet'
    minute.parent.mkdir(parents=True)
    bars().write_parquet(minute)
    daily = tmp_path / f'kline_daily/date={DAY}/part.parquet'
    daily.parent.mkdir(parents=True)
    pl.DataFrame([reference() | {'date': DAY}]).write_parquet(daily)
    repo = KlineRepository.__new__(KlineRepository)
    repo.store = SimpleNamespace(data_dir=tmp_path)
    repo._minute_glob = str(tmp_path / 'kline_minute/**/*.parquet')
    valid, issues = repo.get_minute_quality(['000001'], [DAY])
    assert valid.height == 240 and not issues
    assert valid['symbol'].unique().to_list() == ['000001']
    bars().head(239).write_parquet(minute)
    valid, issues = repo.get_minute_quality(['000001'], [DAY])
    assert valid.is_empty() and issues[0]['reason'] == 'incomplete_session'
