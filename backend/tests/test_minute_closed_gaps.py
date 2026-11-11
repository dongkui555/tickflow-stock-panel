from datetime import datetime, timedelta
from types import SimpleNamespace

import polars as pl

from app.services.kline_sync import _closed_minute_gap_starts


def test_last_bar_at_close_does_not_hide_internal_gaps(tmp_path):
    day = datetime(2026, 9, 3)
    daily_path = tmp_path / "kline_daily" / "date=2026-09-03" / "part.parquet"
    minute_path = tmp_path / "kline_minute" / "date=2026-09-03" / "part.parquet"
    daily_path.parent.mkdir(parents=True)
    minute_path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["000001", "600000"],
            "date": [day.date()] * 2,
            "open": [10.0] * 2,
            "high": [10.0] * 2,
            "low": [10.0] * 2,
            "close": [10.0] * 2,
            "volume": [24000.0] * 2,
            "amount": [240000.0] * 2,
        }
    ).write_parquet(daily_path)
    stamps = [day.replace(hour=9, minute=31) + timedelta(minutes=i) for i in range(120)]
    stamps += [day.replace(hour=13, minute=1) + timedelta(minutes=i) for i in range(120)]
    pl.DataFrame(
        {
            "symbol": ["000001.SZ"] * 240 + ["600000.SH"] * 2,
            "datetime": stamps + [stamps[0], stamps[-1]],
        }
    ).write_parquet(minute_path)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    result = _closed_minute_gap_starts(repo, ["000001", "600000"], day, day.replace(hour=15))
    assert result == {"600000": day}
    assert _closed_minute_gap_starts(repo, ["000001.SZ", "600000.SH"], day, day.replace(hour=15)) == {"600000.SH": day}
    assert _closed_minute_gap_starts(repo, ["600000.SZ"], day, day.replace(hour=15)) == {}
    assert _closed_minute_gap_starts(repo, ["000001", "600000"], day, day.replace(hour=14)) == {}


def test_automatic_closed_gap_is_backed_up_and_only_missing_rows_added(tmp_path, monkeypatch):
    from threading import RLock
    from app.services import kline_sync

    day = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    dp = tmp_path / "kline_daily" / f"date={day.date()}" / "part.parquet"
    mp = tmp_path / "kline_minute" / f"date={day.date()}" / "part.parquet"
    dp.parent.mkdir(parents=True)
    mp.parent.mkdir(parents=True)
    stamps = [day.replace(hour=9, minute=31) + timedelta(minutes=i) for i in range(120)]
    instruments = tmp_path / "instruments"
    instruments.mkdir()
    pl.DataFrame({"symbol": ["000001.SZ"], "code": ["000001"]}).write_parquet(instruments / "instruments.parquet")
    stamps += [day.replace(hour=13, minute=1) + timedelta(minutes=i) for i in range(120)]
    incoming = pl.DataFrame({
        "symbol": ["000001.SZ"] * 240, "datetime": stamps,
        **{c: [10.0] * 240 for c in ("open", "high", "low", "close")},
        "volume": [1.0] * 240, "amount": [1000.0] * 240,
    })
    old = incoming.filter(pl.col("datetime") != stamps[12])
    old.write_parquet(mp)
    pl.DataFrame({"symbol": ["000001"], "date": [day.date()],
                  **{c: [10.0] for c in ("open", "high", "low", "close")},
                  "volume": [24000.0], "amount": [240000.0]}).write_parquet(dp)
    monkeypatch.setattr(kline_sync, "_resolve_minute_provider", lambda name: (object(), False, None))
    monkeypatch.setattr(kline_sync, "resolve_limit", lambda *a, **kw: SimpleNamespace(batch=100, rpm=30))
    monkeypatch.setattr(kline_sync, "sync_minute_batch", lambda *a, **kw: kw["on_segment"](incoming))
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), _write_lock=RLock(),
                           db=SimpleNamespace(execute=lambda *a: None))
    kline_sync.sync_and_persist_minute(["000001"], repo, SimpleNamespace(), days=5)
    assert pl.read_parquet(mp).select(incoming.columns).equals(incoming)
    backups = list((tmp_path / "import_backups").glob("minute_incremental_*/date=*/part.parquet"))
    assert len(backups) == 1
    assert pl.read_parquet(backups[0]).equals(old)
