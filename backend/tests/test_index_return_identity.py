"""Legacy bare index storage must feed qualified index returns to regime."""
from datetime import date

import polars as pl
import pytest

from app.services import regime_builder
from app.tickflow.repository import DataStore, KlineRepository


def write_rows(root, day, symbols, closes):
    folder = root / 'kline_index_enriched' / f'date={day}'
    folder.mkdir(parents=True)
    pl.DataFrame({'symbol': symbols, 'date': [day] * len(symbols),
        'open': closes, 'high': closes, 'low': closes, 'close': closes,
        'volume': [1000.] * len(symbols), 'amount': [100000.] * len(symbols)}).write_parquet(folder / 'part.parquet')


def test_regime_index_return_reads_bare_history_and_prior_session(tmp_path):
    repo = KlineRepository(DataStore(tmp_path))
    write_rows(tmp_path, date(2026, 10, 21), ['000001'], [3713.03])
    write_rows(tmp_path, date(2026, 10, 22), ['000001'], [3658.03])
    # A later partition must not influence the requested close-to-close return.
    write_rows(tmp_path, date(2026, 10, 23), ['000001'], [9000.])
    result = regime_builder._load_index_pct(repo, date(2026, 10, 22), date(2026, 10, 22))
    assert result == {date(2026, 10, 22): pytest.approx(3658.03 / 3713.03 - 1)}


def test_index_read_prefers_exact_identity_and_excludes_other_exchange(tmp_path):
    repo = KlineRepository(DataStore(tmp_path))
    day = date(2026, 10, 22)
    write_rows(tmp_path, day, ['000001', '000001.SH', '000001.SZ'], [1., 3658.03, 99.])
    result = repo.get_index_daily('000001.SH', day, day, columns=['symbol', 'date', 'close'])
    assert result.to_dicts() == [{'symbol': '000001.SH', 'date': day, 'close': 3658.03}]


def test_bare_index_read_returns_canonical_requested_identity(tmp_path):
    repo = KlineRepository(DataStore(tmp_path))
    day = date(2026, 10, 22)
    write_rows(tmp_path, day, ['399001', '000001.SZ'], [12000., 99.])
    result = repo.get_index_daily('399001.SZ', day, day, columns=['symbol', 'date', 'close'])
    assert result.to_dicts() == [{'symbol': '399001.SZ', 'date': day, 'close': 12000.}]
