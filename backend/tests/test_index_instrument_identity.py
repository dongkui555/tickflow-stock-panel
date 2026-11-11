from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest
from app.services import index_sync
from app.services.security_identity import IdentityBatchError
from app.tickflow.repository import DataStore, KlineRepository


@pytest.mark.parametrize("asset_type,code", [("etf", "560650"), ("index", "000001")])
def test_exchange_scoped_instruments_preserve_bare_code_identity(tmp_path, monkeypatch, asset_type, code):
    def fetch(exchange, **kwargs):
        return [{"symbol": code, "name": "test"}] if exchange == "SH" else []

    monkeypatch.setattr(index_sync, "get_client", lambda: SimpleNamespace(
        exchanges=SimpleNamespace(get_instruments=fetch)))
    frame = index_sync._fetch_instruments_by_type(asset_type, asset_type)
    repo = KlineRepository(DataStore(tmp_path))
    save = repo.save_etf_instruments if asset_type == "etf" else repo.save_index_instruments
    save(frame)
    instruments = repo.get_etf_instruments() if asset_type == "etf" else repo.get_index_instruments()
    assert instruments["symbol"].to_list() == [f"{code}.SH"]
    assert instruments["exchange"].to_list() == ["SH"]


def test_exchange_scoped_instruments_reject_conflicting_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(index_sync, "get_client", lambda: SimpleNamespace(
        exchanges=SimpleNamespace(get_instruments=lambda exchange, **kw:
                                 [{"symbol": "560650.SZ"}] if exchange == "SH" else [])))
    with pytest.raises(IdentityBatchError, match="exchange"):
        frame = index_sync._fetch_instruments_by_type("etf", "etf")
        KlineRepository(DataStore(tmp_path)).save_etf_instruments(frame)


@pytest.mark.parametrize("table", ["kline_etf_daily", "kline_etf_enriched"])
def test_etf_upsert_preserves_unknown_legacy_rows_and_validates_new_rows(tmp_path, table):
    repo = KlineRepository(DataStore(tmp_path))
    repo.save_etf_instruments(pl.DataFrame({
        "symbol": ["510300.SH"], "code": ["510300"], "exchange": ["SH"], "name": ["known"],
    }))
    legacy = pl.DataFrame({
        "symbol": ["510300", "560650"], "date": [date(2026, 10, 14)] * 2,
        "open": [4.0, 2.0], "high": [4.0, 2.0], "low": [4.0, 2.0], "close": [4.0, 2.0],
        "volume": [1000.0, 2000.0], "amount": [4000.0, 4000.0],
    })
    target = tmp_path / table / "date=2026-10-14" / "part.parquet"
    target.parent.mkdir(parents=True)
    legacy.write_parquet(target)
    incoming = legacy.head(1).with_columns(pl.lit("510300.SH").alias("symbol"), pl.lit(4.1).alias("close"))
    append = repo.append_etf_daily if table == "kline_etf_daily" else repo.append_etf_enriched
    append(incoming)
    stored = pl.read_parquet(target)
    assert stored["symbol"].to_list() == ["510300.SH", "560650"]
    assert stored.filter(pl.col("symbol") == "560650").equals(legacy.tail(1))
    assert stored.filter(pl.col("symbol") == "510300.SH")["close"].item() == 4.1
    before = target.read_bytes()
    with pytest.raises(IdentityBatchError):
        append(legacy.tail(1).with_columns(pl.lit("560650.SH").alias("symbol")))
    assert target.read_bytes() == before
