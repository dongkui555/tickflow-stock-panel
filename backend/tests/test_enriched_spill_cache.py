import math
from datetime import date, timedelta

import polars as pl
from polars.testing import assert_frame_equal

from app.indicators import pipeline


def history():
    return pl.DataFrame([
        {'symbol': symbol, 'date': date(2025, 1, 1) + timedelta(days=day),
         'open': 5.0 + day / 100, 'high': 6.0 + day / 100,
         'low': 4.0 + day / 100, 'close': 5.0 + day / 100,
         'volume': 1000., 'amount': 10000.}
        for symbol in ['000001.SZ', '600000.SH', '920001.BJ'] for day in range(90)
    ])


def test_spilled_history_is_identical_and_memory_mapped(tmp_path, monkeypatch):
    source = history()
    expected = pipeline.compute_enriched_history_window(source, tmp_path, sym_batch=1)
    original_read = pl.read_ipc
    reads = []

    def read(path, **kwargs):
        assert kwargs['memory_map'] is True and kwargs['rechunk'] is False
        reads.append(path)
        return original_read(path, **kwargs)

    monkeypatch.setattr(pl, 'read_ipc', read)
    spill = tmp_path / 'spill'
    spill.mkdir()
    actual = pipeline.compute_enriched_history_window(source, tmp_path, sym_batch=1,
                                                      spill_directory=spill)
    assert len(reads) == 3
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual.select('symbol', 'date').equals(source.select('symbol', 'date').sort('symbol', 'date'))


def test_ma20_equality_is_independent_of_history_prefix():
    values = [7.31 + math.sin(i) for i in range(130)] + [5.19] * 20
    source = pl.DataFrame({'symbol': ['000001.SZ'] * len(values),
                          'date': [date(2025, 1, 1) + timedelta(days=i) for i in range(len(values))],
                          'close': values})
    for frame in [source, source.tail(60), source.tail(20)]:
        result = pipeline.compute_indicators(frame, needed={'ma20'})
        assert result['ma20'][-1] == 5.19
        assert not result.select((pl.col('close') > pl.col('ma20')).last()).item()
    short = pipeline.compute_indicators(source.tail(19), needed={'ma20'})
    assert short['ma20'].null_count() == 19


def test_ma20_preserves_a_material_above_average_difference():
    frame = history().filter(pl.col('symbol') == '000001.SZ')
    result = pipeline.compute_indicators(frame, needed={'ma20'})
    assert result['close'][-1] > result['ma20'][-1]


def test_repository_owns_spill_until_cache_is_cleared(tmp_path, monkeypatch):
    from app.tickflow import repository

    frame = history()
    latest = frame['date'].max()
    path = tmp_path / 'kline_daily_enriched' / f'date={latest}' / 'part.parquet'
    path.parent.mkdir(parents=True)
    frame.filter(pl.col('date') == latest).write_parquet(path)
    repo = repository.KlineRepository(repository.DataStore(tmp_path))
    monkeypatch.setattr(repo, '_latest_enriched_date_duckdb', lambda: latest)
    monkeypatch.setattr(repo, '_build_live_agg', lambda *_: None)
    monkeypatch.setattr(repo, '_restore_missing_latest_rows', lambda _d, today, _h: today)
    monkeypatch.setattr(repository, 'guarded_collect', lambda *_a, **_k: frame)
    repo._refresh_enriched_impl()
    assert repo._enriched_history_spill is not None
    folder = repo._enriched_history_spill.name
    borrowed = repo.get_enriched_range(latest, latest)
    assert borrowed is not None
    repo.clear_cache()
    assert repo._enriched_history_spill is None
    assert borrowed['close'].to_list() == [5.89] * 3
    # On Windows the borrowed mmap may defer deletion, without invalidating it.
    if __import__('os').name == 'posix':
        from pathlib import Path
        assert not Path(folder).exists()
