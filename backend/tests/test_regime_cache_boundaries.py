"""Cold range reads must stay bounded and warm slices must keep predecessors."""
from datetime import date

import polars as pl
import pytest

from app.services import market_phase, regime_builder
from app.tickflow.repository import KlineRepository


def repo_with(cache):
    repo = KlineRepository.__new__(KlineRepository)
    repo._enriched_history_cache = cache
    repo._enriched_warming = False
    return repo


@pytest.mark.parametrize('warming', [False, True])
def test_cold_read_returns_miss_without_synchronous_full_refresh(warming):
    repo = repo_with(None)
    repo._enriched_warming = warming
    calls = []
    repo._refresh_enriched = lambda: calls.append('unbounded_refresh')
    assert repo.get_enriched_range(date(2026, 10, 21), date(2026, 10, 22)) is None
    assert calls == []


def history():
    return pl.DataFrame([{'symbol': str(i), 'date': when, 'close': 10., 'amount': 1e8,
        'change_pct': .01, 'ma20': 9., 'consecutive_limit_ups': n,
        'signal_limit_up': n > 0, 'signal_limit_down': False, 'signal_broken_limit_up': False,
        'unused_wide_column': 1.}
        for i in range(20) for when, n in [(date(2024, 2, 8), 1),
            (date(2024, 2, 19), 2 if i < 10 else 0), (date(2024, 2, 20), 9)]])


def test_single_session_promotion_uses_prior_session_across_holiday():
    cache = history()
    repo = repo_with(cache)
    result = repo.get_enriched_range(date(2024, 2, 19), date(2024, 2, 19))
    row = regime_builder._aggregate_daily(result).row(0, named=True)
    assert row['promo_pool'] == 20
    assert row['promo_rate'] == .5
    assert result['date'].unique().to_list() == [date(2024, 2, 19)]
    assert repo._enriched_history_cache.equals(cache)


def test_promotion_is_invariant_to_requested_batch_start():
    repo = repo_with(history())
    first = regime_builder._aggregate_daily(repo.get_enriched_range(date(2024, 2, 19), date(2024, 2, 19)))
    together = regime_builder._aggregate_daily(repo.get_enriched_range(date(2024, 2, 8), date(2024, 2, 19)))
    assert first.equals(together.filter(pl.col('date') == date(2024, 2, 19)))


def test_predecessor_projection_does_not_sort_or_copy_wide_feature_history(monkeypatch):
    repo = repo_with(history())
    original = market_phase.with_prev_consecutive
    calls = []

    def narrow(frame):
        calls.append(frame.columns)
        assert set(frame.columns) == {'symbol', 'date', 'consecutive_limit_ups'}
        assert frame['date'].max() <= date(2024, 2, 19)
        return original(frame)

    monkeypatch.setattr(market_phase, 'with_prev_consecutive', narrow)
    result = repo.get_enriched_range(date(2024, 2, 19), date(2024, 2, 19), symbols=['0'])
    assert calls
    assert result['_prev_consec'].to_list() == [1]
    assert result['symbol'].to_list() == ['0']


def test_requested_columns_keep_original_projection_contract():
    repo = repo_with(history())
    result = repo.get_enriched_range(date(2024, 2, 19), date(2024, 2, 19), columns=['close'])
    assert result.columns == ['symbol', 'date', 'close']


def test_existing_predecessor_is_not_replaced_by_slice_boundary():
    frame = history().filter(pl.col('date') == date(2024, 2, 19)).with_columns(pl.lit(3).alias('_prev_consec'))
    result = repo_with(frame).get_enriched_range(date(2024, 2, 19), date(2024, 2, 19))
    assert result['_prev_consec'].to_list() == [3] * 20
