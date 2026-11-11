from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.services.history_import import import_history_csv, import_history_csv_batch, import_history_minute_batch
from app.tickflow.repository import DataStore, KlineRepository


def _write_stock_instruments(repo: KlineRepository) -> None:
    pl.DataFrame({
        "symbol": ["000001.SZ"],
        "name": ["平安银行"],
        "code": ["000001"],
        "exchange": ["SZ"],
        "region": ["CN"],
        "type": ["stock"],
        "listing_date": [None],
        "total_shares": [None],
        "float_shares": [None],
        "tick_size": [None],
        "limit_up": [None],
        "limit_down": [None],
        "as_of": ["2026-08-25"],
    }).write_parquet(repo.store.data_dir / "instruments" / "instruments.parquet")
    repo._refresh_instruments()


def test_import_history_csv_writes_stock_history(tmp_path):
    repo = KlineRepository(DataStore(tmp_path))
    _write_stock_instruments(repo)

    start = date(2026, 7, 1)
    rows = ["code,trade_date,open,high,low,close,vol,amount"]
    for i in range(30):
        d = start + timedelta(days=i)
        open_ = 10 + i * 0.1
        close = open_ + 0.2
        high = close + 0.3
        low = open_ - 0.2
        vol = 100_000 + i * 1_000
        amount = vol * close
        rows.append(
            f"000001,{d:%Y%m%d},{open_:.2f},{high:.2f},{low:.2f},{close:.2f},{vol},{amount:.2f}"
        )

    csv_path = tmp_path / "000001_history.csv"
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    result = import_history_csv(repo, csv_path, asset_type_hint="auto")

    assert result["status"] == "ok"
    assert result["written_rows"] == 30
    assert result["recomputed_rows"] > 0
    assert result["groups"][0]["asset_type"] == "stock"

    raw_part = pl.read_parquet(tmp_path / "kline_daily" / f"date={start.isoformat()}" / "part.parquet")
    assert raw_part["symbol"].to_list() == ["000001.SZ"]
    assert raw_part["date"][0] == start

    latest = repo.latest_enriched_date("stock")
    assert latest == start + timedelta(days=29)

    daily = repo.get_daily_asset(
        "stock",
        "000001.SZ",
        start,
        start + timedelta(days=29),
        columns=["symbol", "date", "close"],
    )
    assert not daily.is_empty()


def test_import_history_csv_batch_merges_multiple_files(tmp_path):
    repo = KlineRepository(DataStore(tmp_path))
    _write_stock_instruments(repo)

    start = date(2026, 7, 1)
    csv_paths = []
    for file_idx in range(2):
        rows = ["code,trade_date,open,high,low,close,vol,amount"]
        for i in range(10):
            d = start + timedelta(days=file_idx * 10 + i)
            open_ = 10 + (file_idx * 10 + i) * 0.1
            close = open_ + 0.2
            high = close + 0.3
            low = open_ - 0.2
            vol = 100_000 + i * 1_000
            amount = vol * close
            rows.append(
                f"000001,{d:%Y%m%d},{open_:.2f},{high:.2f},{low:.2f},{close:.2f},{vol},{amount:.2f}"
            )
        csv_path = tmp_path / f"000001_history_{file_idx}.csv"
        csv_path.write_text("\n".join(rows), encoding="utf-8")
        csv_paths.append(csv_path)

    result = import_history_csv_batch(repo, csv_paths, asset_type_hint="auto")

    assert result["status"] == "ok"
    assert result["file_count"] == 2
    assert result["written_rows"] == 20
    assert result["recomputed_rows"] > 0

    latest = repo.latest_enriched_date("stock")
    assert latest == start + timedelta(days=19)


def test_import_history_minute_batch_writes_minute_partitions(tmp_path):
    repo = KlineRepository(DataStore(tmp_path))
    _write_stock_instruments(repo)

    csv_path = tmp_path / "000001_minute.csv"
    csv_path.write_text(
        "\n".join([
            "code,trade_time,open,high,low,close,vol,amount",
            "000001,2026-07-01 09:25:00,9.90,10.00,9.80,9.95,1000,9950",
            "000001,2026-07-01 09:31:00,10.00,10.10,9.90,10.05,1200,12060",
            "000001,2026-07-01 13:01:00,10.05,10.20,10.00,10.15,1500,15225",
        ]),
        encoding="utf-8",
    )

    result = import_history_minute_batch(repo, [csv_path])

    assert result["status"] == "ok"
    assert result["data_kind"] == "minute"
    assert result["import_rows"] == 2
    assert result["symbols"] == 1

    part = pl.read_parquet(tmp_path / "kline_minute" / "date=2026-07-01" / "part.parquet")
    assert part.height == 2
    assert part["symbol"].to_list() == ["000001.SZ", "000001.SZ"]
