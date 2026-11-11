from __future__ import annotations

from datetime import date

import polars as pl

from app.tickflow.repository import DataStore, KlineRepository


def _symbols(count: int) -> list[str]:
    return [f"{i:06d}.SZ" for i in range(1, count + 1)]


def _write_instruments(repo: KlineRepository, count: int) -> None:
    symbols = _symbols(count)
    pl.DataFrame(
        {
            "symbol": symbols,
            "name": [f"stock{i}" for i in range(count)],
            "code": [s.split(".", 1)[0] for s in symbols],
            "exchange": ["SZ"] * count,
        }
    ).write_parquet(repo.store.data_dir / "instruments" / "instruments.parquet")
    repo._refresh_instruments()


def _write_partition(repo: KlineRepository, table: str, day: date, symbols: list[str]) -> None:
    root = repo.store.data_dir / table / f"date={day.isoformat()}"
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": symbols,
            "date": [day] * len(symbols),
            "open": [10.0] * len(symbols),
            "high": [10.5] * len(symbols),
            "low": [9.8] * len(symbols),
            "close": [10.2] * len(symbols),
            "volume": [1000.0] * len(symbols),
            "amount": [10200.0] * len(symbols),
        }
    ).write_parquet(root / "part.parquet")


def test_latest_stock_dates_skip_sparse_realtime_partitions(tmp_path) -> None:
    repo = KlineRepository(DataStore(tmp_path))
    _write_instruments(repo, 1200)

    full_day = date(2026, 8, 31)
    sparse_day = date(2026, 9, 1)
    full_symbols = _symbols(900)
    sparse_symbols = _symbols(9)

    for table in ("kline_daily", "kline_daily_enriched"):
        _write_partition(repo, table, full_day, full_symbols)
        _write_partition(repo, table, sparse_day, sparse_symbols)

    assert repo.latest_daily_date() == full_day
    assert repo.latest_enriched_date("stock") == full_day
    assert repo._latest_enriched_date_duckdb() == full_day


def test_append_daily_history_update_replaces_live_snapshot_marker(tmp_path) -> None:
    repo = KlineRepository(DataStore(tmp_path))
    day = date(2026, 8, 31)
    symbol = "600001.SH"
    pl.DataFrame({"symbol": [symbol], "code": ["600001"], "name": ["测试股票"]}).write_parquet(
        tmp_path / "instruments/instruments.parquet"
    )
    repo._refresh_instruments()
    root = tmp_path / "kline_daily" / f"date={day.isoformat()}"
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "symbol": [symbol],
            "date": [day],
            "open": [10.0],
            "high": [10.3],
            "low": [9.8],
            "close": [10.1],
            "volume": [1000.0],
            "amount": [10100.0],
            "quote_ts": [1788146302000],
        }
    ).write_parquet(root / "part.parquet")

    repo.append_daily(
        pl.DataFrame(
            {
                "symbol": [symbol],
                "date": [day],
                "open": [10.0],
                "high": [10.5],
                "low": [9.7],
                "close": [10.4],
                "volume": [1800.0],
                "amount": [18720.0],
            }
        )
    )

    saved = pl.read_parquet(root / "part.parquet")
    assert saved.height == 1
    assert saved["close"].to_list() == [10.4]
    assert saved["quote_ts"].to_list() == [None]
