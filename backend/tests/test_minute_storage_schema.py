from datetime import datetime

import polars as pl
import pytest

from app.services.kline_sync import _write_minute_partition


def test_identity_normalization_preserves_order_and_is_idempotent():
    from app.services.kline_sync import _ensure_minute_identity_columns

    frame = pl.DataFrame({
        "symbol": ["600000.SH"],
        "datetime": [datetime(2026, 9, 23, 9, 31)],
        "close": [10.0],
    })
    params = dict(asset_type="stock", identity_lookup={"600000": "600000.SH"})
    once = _ensure_minute_identity_columns(frame, **params)
    twice = _ensure_minute_identity_columns(once, **params)
    assert once.columns[:len(frame.columns)] == frame.columns
    assert once.select(frame.columns).equals(frame)
    assert twice.equals(once)
    assert once["security_key"].to_list() == ["stock|SH|600000"]


def test_minute_delta_merges_with_identity_enriched_partition(tmp_path):
    minute_dir = tmp_path / "kline_minute"
    day_dir = minute_dir / "date=2026-09-23"
    day_dir.mkdir(parents=True)
    existing = pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "datetime": [datetime(2026, 9, 23, 9, 31)],
            "open": [10.0], "high": [10.1], "low": [9.9], "close": [10.0],
            "volume": [100.0], "amount": [1000.0],
            "asset_type": ["stock"], "exchange": ["SH"], "code": ["600000"],
            "security_key": ["stock|SH|600000"],
        }
    )
    existing.write_parquet(day_dir / "part.parquet")

    incoming = pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "datetime": [datetime(2026, 9, 23, 9, 32)],
            "open": [10.0], "high": [10.2], "low": [9.95], "close": [10.1],
            "volume": [120.0], "amount": [1210.0],
        }
    )
    _write_minute_partition(
        incoming,
        minute_dir,
        asset_type="stock",
        identity_lookup={"600000": "600000.SH"},
    )

    merged = pl.read_parquet(day_dir / "part.parquet")
    assert merged.height == 2
    assert set(("asset_type", "exchange", "code", "security_key")) <= set(merged.columns)


def test_partial_lookup_does_not_revalidate_unrelated_stored_stock(tmp_path):
    minute_dir = tmp_path / "kline_minute"
    day_dir = minute_dir / "date=2026-09-23"
    day_dir.mkdir(parents=True)
    existing = pl.DataFrame({
        "symbol": ["600000.SH"], "datetime": [datetime(2026, 9, 23, 9, 31)],
        "open": [10.0], "high": [10.1], "low": [9.9], "close": [10.0],
        "volume": [100.0], "amount": [1000.0],
        "asset_type": ["stock"], "exchange": ["SH"], "code": ["600000"],
        "security_key": ["stock|SH|600000"],
    })
    path = day_dir / "part.parquet"
    existing.write_parquet(path)
    incoming = existing.select(existing.columns[:8]).with_columns(pl.lit("000001.SZ").alias("symbol"))
    _write_minute_partition(incoming, minute_dir, identity_lookup={"000001": "000001.SZ"})
    actual = pl.read_parquet(path)
    assert actual.height == 2
    assert actual.filter(pl.col("symbol") == "600000.SH").equals(existing)
    assert actual.filter(pl.col("symbol") == "000001.SZ")["security_key"].to_list() == ["stock|SZ|000001"]


def test_incoming_unknown_identity_still_rejected_without_writing(tmp_path):
    from app.services.security_identity import IdentityBatchError

    incoming = pl.DataFrame({"symbol": ["600000.SH"], "datetime": [datetime(2026, 9, 23, 9, 31)]})
    minute_dir = tmp_path / "kline_minute"
    with pytest.raises(IdentityBatchError):
        _write_minute_partition(incoming, minute_dir, identity_lookup={"000001": "000001.SZ"})
    assert not minute_dir.exists()


def test_repair_with_scoped_lookup_preserves_unrelated_partition_rows(tmp_path):
    from threading import RLock
    from types import SimpleNamespace

    from test_minute_repair import DAY, bars, reference

    from app.services.minute_repair import repair_minute_gaps

    minute_path = tmp_path / f"kline_minute/date={DAY}/part.parquet"
    daily_path = tmp_path / f"kline_daily/date={DAY}/part.parquet"
    minute_path.parent.mkdir(parents=True)
    daily_path.parent.mkdir(parents=True)
    unrelated = bars("600000.SH").with_columns(
        pl.lit("stock").alias("asset_type"), pl.lit("SH").alias("exchange"),
        pl.lit("600000").alias("code"), pl.lit("stock|SH|600000").alias("security_key"),
    )
    unrelated.write_parquet(minute_path)
    pl.DataFrame([reference()]).write_parquet(daily_path)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), _write_lock=RLock(),
                           rebuild_views=lambda: None, clear_cache=lambda: None)
    result = repair_minute_gaps(
        repo, start_date=DAY, end_date=DAY, provider_name="qmt",
        source_provider=SimpleNamespace(get_minute=lambda *a, **k: bars()),
        bypass_failover=True, stock_days=[("000001.SZ", DAY)],
        identity_lookup={"000001": "000001.SZ"},
    )
    assert result["added_rows"] == 240
    actual = pl.read_parquet(minute_path)
    assert actual.height == 480
    assert actual.filter(pl.col("symbol") == "600000.SH").equals(unrelated)
    backup = next((tmp_path / "import_backups").glob("*/date=*/part.parquet"))
    assert pl.read_parquet(backup).equals(unrelated)
