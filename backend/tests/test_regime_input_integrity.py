"""Missing trend inputs must not masquerade as zero breadth or an open gate."""
from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl
import pytest
from app.services import market_gate, preferences, regime_builder

DAY = date(2026, 10, 15)


@pytest.mark.parametrize('ma20', ['absent', None, float('nan'), float('inf'), 0.0])
def test_unavailable_ma20_is_explicit_not_zero(ma20):
    frame = pl.DataFrame({'symbol': ['000001'], 'date': [DAY], 'close': [10.0],
                          'change_pct': [0.01], 'amount': [1e8]})
    if ma20 != 'absent':
        frame = frame.with_columns(pl.lit(ma20, dtype=pl.Float64).alias('ma20'))
    result = regime_builder._aggregate_daily(frame).row(0, named=True)
    assert result['above_ma20_pct'] is None
    assert result['ma20_valid_count'] == 0


def test_actual_zero_breadth_has_valid_observations_and_is_not_neutral():
    frame = pl.DataFrame({'symbol': ['000001'], 'date': [DAY], 'close': [9.0],
                          'change_pct': [-0.01], 'amount': [1e8], 'ma20': [10.0]})
    result = regime_builder._aggregate_daily(frame).row(0, named=True)
    assert result['above_ma20_pct'] == 0.0
    assert result['ma20_valid_count'] == 1
    zero = regime_builder._compute_subscores({'above_ma20_pct': 0.0})
    neutral = regime_builder._compute_subscores({'above_ma20_pct': None})
    assert zero['trend'] < neutral['trend']


@pytest.mark.parametrize('identity_first', [True, False])
def test_thin_cached_input_recomputes_ma20_with_mixed_history_schema(tmp_path, monkeypatch, identity_first):
    start = DAY - timedelta(days=65)
    days = [start + timedelta(days=i) for i in range(66) if (start + timedelta(days=i)).weekday() < 5]
    history = pl.DataFrame({'symbol': ['000001'] * len(days), 'date': days,
        **{c: [10.0 + i * 0.1 for i in range(len(days))] for c in
           ('open', 'high', 'low', 'close', 'raw_close', 'raw_high', 'raw_low')},
        'volume': [1000.0] * len(days), 'amount': [10000.0] * len(days),
        'consecutive_limit_ups': [0] * len(days)})
    for i, part in enumerate(history.partition_by('date', maintain_order=True)):
        if i == (0 if identity_first else len(days) - 1):
            part = part.with_columns(pl.lit('stock').alias('asset_type'), pl.lit('SZ').alias('exchange'))
        path = tmp_path / f'kline_daily_enriched/date={days[i]}/part.parquet'
        path.parent.mkdir(parents=True)
        part.write_parquet(path)
    thin = history.filter(pl.col('date') == DAY).with_columns(pl.lit(0.01).alias('change_pct'))
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path),
        get_enriched_range=lambda *a: thin, get_instruments=pl.DataFrame,
        get_historical_shares=pl.DataFrame)
    monkeypatch.setattr(preferences, 'get_sentiment_exclude_st', lambda: False)
    monkeypatch.setattr(regime_builder, '_load_index_pct', lambda *a: {})
    result = regime_builder.run_regime_batch(repo, DAY, DAY).row(0, named=True)
    assert result['above_ma20_pct'] == 1.0
    assert result['ma20_valid_count'] == 1


def test_historical_gate_blocks_missing_breadth_without_fabricated_half(tmp_path, monkeypatch):
    frame = pl.DataFrame({'date': [DAY], 'state': ['strong'], 'phase': ['rally'], 'score': [90.0],
                         'above_ma20_pct': [None], 'ma20_valid_count': [0]})
    monkeypatch.setattr(regime_builder, 'load_regime_history', lambda _: frame)
    row = market_gate.market_gate_frame(tmp_path).row(0, named=True)
    assert row['market_above_ma20_pct'] is None
    assert row['market_gate'] == 'closed'
    assert row['market_allows_open'] is False


def test_live_gate_explains_unavailable_breadth(tmp_path, monkeypatch):
    monkeypatch.setattr(market_gate, '_latest_regime_row', lambda *a: {
        'date': DAY, 'state': 'strong', 'phase': 'rally', 'score': 90,
        'above_ma20_pct': None, 'ma20_valid_count': 0})
    result = market_gate.build_market_gate(tmp_path, as_of=DAY, use_overview=False)
    assert result['gate'] == 'closed'
    assert result['metrics']['above_ma20_pct'] is None
    assert any('MA20' in reason and '不足' in reason for reason in result['reasons'])


def test_st_filter_matches_qualified_registry_with_bare_daily_codes():
    frame = pl.DataFrame({'symbol': ['000001', '600000.SH', 'A'], 'close': [10.0] * 3})
    result = regime_builder._filter_excluded_symbols(frame, ['000001.SZ', 'A'])
    assert result['symbol'].to_list() == ['600000.SH']


def test_warmup_counts_observations_for_sparse_stock_not_market_folders(tmp_path):
    # B traded before a long gap. Its target-day MA still uses its previous
    # 19 observations under the existing indicator contract.
    start = DAY - timedelta(days=130)
    days = [start + timedelta(days=i) for i in range(131) if (start + timedelta(days=i)).weekday() < 5]
    for i, day in enumerate(days):
        symbols = ['000001'] + (['600000'] if i < 25 or day == DAY else [])
        frame = pl.DataFrame({'symbol': symbols, 'date': [day] * len(symbols),
            'close': [10.0 + i * 0.01] * len(symbols), 'amount': [1e8] * len(symbols),
            'consecutive_limit_ups': [0] * len(symbols)})
        path = tmp_path / f'kline_daily_enriched/date={day}/part.parquet'
        path.parent.mkdir(parents=True)
        frame.write_parquet(path)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    result = regime_builder._compute_batch(repo, tmp_path / 'kline_daily_enriched',
        pl.DataFrame(), pl.DataFrame(), DAY, DAY, 40)
    assert result.filter(pl.col('symbol') == '600000')['ma20'].is_finite().all()
    assert result.filter(pl.col('symbol') == '600000')['ma20'].item() > 10.0
