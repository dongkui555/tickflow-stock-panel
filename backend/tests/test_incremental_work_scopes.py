from datetime import date
from types import SimpleNamespace

import polars as pl

from app.services import kline_sync, regime_builder


def factors(rows):
    return pl.DataFrame(rows, schema=[("symbol", pl.String), ("trade_date", pl.Date),
                                      ("ex_factor", pl.Float64)], orient="row")


def test_cumulative_repeat_is_not_a_historical_change():
    before = factors([("A", date(2026, 9, 4), 2.0), ("B", date(2026, 9, 4), 3.0)])
    after = factors([("A", date(2026, 9, 7), 2.0), ("B", date(2026, 9, 7), 3.1)])
    assert kline_sync._changed_factor_symbols(before, after, cumulative=True) == ["B"]
    assert kline_sync._changed_factor_symbols(before, before, cumulative=True) == []


def test_factor_corrections_backfill_and_events():
    before = factors([("A", date(2026, 9, 4), 2.0), ("A", date(2026, 9, 7), 3.0)])
    correction = factors([("A", date(2026, 9, 4), 2.1)])
    backfill = factors([("A", date(2026, 9, 1), 1.5)])
    unchanged_backfill = factors([("A", date(2026, 9, 1), 2.0)])
    assert kline_sync._changed_factor_symbols(before, correction, cumulative=True) == ["A"]
    assert kline_sync._changed_factor_symbols(before, backfill, cumulative=True) == ["A"]
    assert kline_sync._changed_factor_symbols(before, unchanged_backfill, cumulative=True) == []
    assert kline_sync._changed_factor_symbols(before, factors([("A", date(2026, 9, 8), 3.0)]),
                                               cumulative=False) == ["A"]
    assert kline_sync._changed_factor_symbols(before, before, cumulative=False) == []


def test_regime_explicit_dates_do_not_span_unaffected_days(tmp_path, monkeypatch):
    days = [date(2026, 9, d) for d in (1, 2, 3, 4, 7, 8)]
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(regime_builder, "enriched_date_set", lambda _: set(days))
    calls = []

    def batch(repo, start, end):
        calls.append((start, end))
        return pl.DataFrame({"date": [d for d in days if start <= d <= end], "score": [1.0]})

    monkeypatch.setattr(regime_builder, "run_regime_batch", batch)
    out = regime_builder.compute_regime_incremental(
        repo, tmp_path, today=days[-1], target_dates=[days[0], days[-1], date(2026, 9, 9)])
    assert out["date"].to_list() == [days[0], days[-1]]
    assert calls == [(days[0], days[0]), (days[-1], days[-1])]


def test_factor_persistence_does_not_requeue_identical_data(tmp_path):
    from app.services import maintenance_queue
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    old = factors([("A", date(2026, 9, 4), 2.0), ("B", date(2026, 9, 4), 3.0)])
    path = tmp_path / "adj_factor" / "all.parquet"
    path.parent.mkdir()
    old.write_parquet(path)
    stamp = path.stat().st_mtime_ns
    assert kline_sync._persist_adj_factors(repo, old, asset_type="stock", cumulative=True) == (0, [])
    assert path.stat().st_mtime_ns == stamp
    assert maintenance_queue.status(tmp_path) == []
    new = factors([("A", date(2026, 9, 7), 2.0), ("B", date(2026, 9, 7), 3.1)])
    assert kline_sync._persist_adj_factors(repo, new, asset_type="stock", cumulative=True) == (2, ["B"])
    assert maintenance_queue.status(tmp_path)[0]["total"] == 1
    assert kline_sync._persist_adj_factors(repo, new, asset_type="stock", cumulative=True) == (0, [])


def test_recent_regime_write_does_not_hide_older_stale_date(tmp_path, monkeypatch):
    import os
    old, recent = date(2026, 9, 1), date(2026, 9, 8)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    for d in [old, recent]:
        path = tmp_path / "kline_daily_enriched" / f"date={d}" / "part.parquet"
        path.parent.mkdir(parents=True)
        pl.DataFrame({"date": [d]}).write_parquet(path)
        os.utime(path, ns=(300, 300))
    regime_builder.upsert_regime_history(tmp_path, pl.DataFrame({"date": [old, recent], "score": [1, 1]}))
    os.utime(regime_builder.regime_path(tmp_path), ns=(100, 100))
    assert set(regime_builder.detect_stale_dates(tmp_path, repo)) == {old, recent}
    monkeypatch.setattr(regime_builder, "run_regime_batch",
                        lambda repo, start, end: pl.DataFrame({"date": [start], "score": [2]}))
    regime_builder.compute_regime_incremental(repo, tmp_path, today=recent, target_dates=[recent])
    assert regime_builder.detect_stale_dates(tmp_path, repo) == [old]


def test_provider_factor_contract_distinguishes_events():
    from app.plugins.fuyao.provider import FuyaoProvider
    from app.plugins.tushare.provider import TushareProvider
    assert TushareProvider.adj_factor_kind == "cumulative"
    assert getattr(FuyaoProvider, "adj_factor_kind", "event") == "event"
