from datetime import date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from app.services.minute_features import build_daily_features


def write_bars(root, symbol, day=date(2025, 1, 2)):
    part = root / 'kline_minute' / f'date={day.isoformat()}'
    part.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({
        'symbol': [symbol, symbol],
        'datetime': [datetime.combine(day, datetime.min.time()).replace(hour=9, minute=31),
                     datetime.combine(day, datetime.min.time()).replace(hour=15)],
        'open': [10., 10.1], 'high': [10.2, 10.3], 'low': [9.9, 10.],
        'close': [10.1, 10.2], 'volume': [100., 120.], 'amount': [1000., 1224.],
    })
    frame.write_parquet(part / f'{symbol}.parquet')
    return frame


@pytest.mark.parametrize('symbol,requested', [
    ('000001.SZ', '000001'), ('600000.SH', '600000'),
    ('830799.BJ', '830799'), ('920002.BJ', '920002'),
    ('SZ000001', '000001'), ('000001', '000001.SZ'),
])
def test_stock_identity_matches_training_key(tmp_path, symbol, requested):
    write_bars(tmp_path, symbol)
    result = build_daily_features(tmp_path, [requested], date(2025, 1, 2), date(2025, 1, 2))
    assert result.height == 1
    assert result['symbol'].item() == ('000001' if '000001' in symbol else symbol[:6])
    assert result['minute_bar_count'].item() == 2
    assert result['minute_intraday_return'].item() == pytest.approx(.02)


def test_different_market_cannot_join_same_numeric_code(tmp_path):
    write_bars(tmp_path, '000001.SH')
    result = build_daily_features(tmp_path, ['000001.SZ'], date(2025, 1, 2), date(2025, 1, 2))
    assert result.is_empty()


def test_unrequested_corrupt_partition_does_not_hide_requested_data(tmp_path):
    write_bars(tmp_path, '000001.SZ')
    old = tmp_path / 'kline_minute' / 'date=2020-01-02'
    old.mkdir()
    (old / 'part.parquet').write_bytes(b'broken')
    result = build_daily_features(tmp_path, ['000001'], date(2025, 1, 2), date(2025, 1, 2))
    assert result.height == 1


def test_alias_duplicates_do_not_inflate_bar_count(tmp_path):
    write_bars(tmp_path, '000001.SZ')
    write_bars(tmp_path, '000001')
    result = build_daily_features(tmp_path, ['000001'], date(2025, 1, 2), date(2025, 1, 2))
    assert result.height == 1
    assert result['minute_bar_count'].item() == 2


def test_conflicting_alias_values_do_not_become_extra_bars(tmp_path):
    write_bars(tmp_path, '000001.SZ')
    frame = write_bars(tmp_path, '000001')
    frame.with_columns((pl.col('amount') * 2).alias('amount')).write_parquet(
        tmp_path / 'kline_minute' / 'date=2025-01-02' / '000001.parquet')
    result = build_daily_features(tmp_path, ['000001'], date(2025, 1, 2), date(2025, 1, 2))
    assert result.height == 1
    assert result['minute_bar_count'].item() == 2


def test_multi_day_features_scan_only_one_session_at_a_time(tmp_path, monkeypatch):
    days = [date(2025, 1, 2), date(2025, 1, 3)]
    for day in days:
        write_bars(tmp_path, '000001.SZ', day)
        write_bars(tmp_path, '000001', day)
    expected = pl.concat([
        build_daily_features(tmp_path, ['000001'], day, day) for day in days
    ]).sort(['symbol', 'date'])
    original = pl.scan_parquet
    scanned_days = []

    def bounded_scan(files, *args, **kwargs):
        from pathlib import Path
        partitions = {Path(path).parent.name for path in files}
        scanned_days.append(partitions)
        return original(files, *args, **kwargs)

    monkeypatch.setattr(pl, 'scan_parquet', bounded_scan)
    result = build_daily_features(tmp_path, ['000001'], days[0], days[-1])
    assert scanned_days and all(len(batch) == 1 for batch in scanned_days)
    assert_frame_equal(result.sort(['symbol', 'date']), expected)


def test_minute_progress_can_cancel_between_sessions(tmp_path):
    for day in [date(2025, 1, 2), date(2025, 1, 3)]:
        write_bars(tmp_path, '000001.SZ', day)

    def cancel(completed, total, day):
        raise RuntimeError('cancelled by user')

    with pytest.raises(RuntimeError, match='cancelled by user'):
        build_daily_features(tmp_path, ['000001'], date(2025, 1, 2), date(2025, 1, 3),
                             on_progress=cancel)
