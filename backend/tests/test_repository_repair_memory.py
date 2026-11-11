from datetime import date, timedelta

import polars as pl
from app.indicators import pipeline
from app.tickflow import repository
from polars.testing import assert_frame_equal


def test_latest_repair_keeps_sorted_history_without_copying_wide_cache(tmp_path, monkeypatch):
    latest = date(2026, 10, 14)
    history = pl.DataFrame([
        {"symbol": symbol, "date": latest - timedelta(days=day), "close": 10.0 + day,
         **{f"signal_{i}": float(i + day) for i in range(20)}}
        for symbol in ["000002.SZ", "000004.SZ", "000006.SZ"]
        for day in [2, 1, 0]
        if not (symbol == "000004.SZ" and day == 0)
    ]).sort("symbol", "date")
    today = history.filter(pl.col("date") == latest)
    recovered = pl.DataFrame({
        "symbol": ["000001.SZ", "000004.SZ", "000009.SZ"],
        "date": [latest] * 3, "close": [5.0, 6.0, 7.0],
        "new_metadata": ["first", "middle", "last"],
    })
    repaired = pl.concat([today, recovered], how="diagonal_relaxed").sort("symbol")
    expected = pl.concat([
        history.filter(pl.col("date") != latest), repaired,
    ], how="diagonal_relaxed").sort("symbol", "date")
    target = tmp_path / "kline_daily_enriched" / f"date={latest}" / "part.parquet"
    target.parent.mkdir(parents=True)
    today.select("symbol", "date", "close").write_parquet(target)
    repo = repository.KlineRepository(repository.DataStore(tmp_path))
    repo._enriched_history_cache = history.clone()
    repo._enriched_cache = today
    monkeypatch.setattr(repo, "_latest_enriched_date_duckdb", lambda: latest)
    monkeypatch.setattr(repo, "_build_live_agg", lambda *_: None)
    monkeypatch.setattr(repo, "_restore_missing_latest_rows", lambda *_: repaired)
    monkeypatch.setattr(repository, "guarded_collect", lambda *_a, **_k: history.select("symbol", "date", "close"))
    def compute_window(*_args, **_kwargs):
        assert repo._enriched_history_cache is None, "refresh retained two full history caches"
        assert repo._enriched_cache is today, "latest quotes remain available during refresh"
        return history

    monkeypatch.setattr(pipeline, "compute_enriched_history_window", compute_window)
    original_sort = pl.DataFrame.sort
    original_filter = pl.DataFrame.filter

    def bounded_sort(frame, *args, **kwargs):
        assert not (frame.width > 20 and frame.height > repaired.height), "wide history sort copies the cache"
        return original_sort(frame, *args, **kwargs)

    def bounded_filter(frame, *args, **kwargs):
        result = original_filter(frame, *args, **kwargs)
        if frame is history:
            assert result.height <= today.height, "repair copied unrelated historical rows"
        return result

    monkeypatch.setattr(pl.DataFrame, "sort", bounded_sort)
    monkeypatch.setattr(pl.DataFrame, "filter", bounded_filter)
    repo._refresh_enriched_impl()

    assert_frame_equal(repo._enriched_history_cache, expected, check_exact=True)
    assert_frame_equal(repo._enriched_cache, repaired, check_exact=True)


def test_latest_repair_does_not_duplicate_bare_and_qualified_stock(tmp_path, monkeypatch):
    latest = date(2026, 10, 15)
    repo = repository.KlineRepository(repository.DataStore(tmp_path))
    repo._live_agg_cache = pl.DataFrame({"symbol": ["000001"]})
    today = pl.DataFrame({"symbol": ["000001"], "date": [latest], "close": [10.0]})
    daily = today.with_columns(pl.lit("000001.SZ").alias("symbol"), pl.lit(1000.0).alias("volume"))
    target = tmp_path / "kline_daily" / f"date={latest}" / "part.parquet"
    target.parent.mkdir(parents=True)
    daily.write_parquet(target)
    monkeypatch.setattr(pipeline, "compute_enriched_today", lambda *_a, **_k: daily)
    monkeypatch.setattr(repo, "_with_instrument_metadata", lambda _asset, frame: frame)
    restored = repo._restore_missing_latest_rows(latest, today, today)
    assert_frame_equal(restored, today, check_exact=True)
