from pathlib import Path

import polars as pl

from app.services.identity_migration import apply_migration, inventory, write_manifest


def test_inventory_reports_bare_and_canonical_symbols(tmp_path: Path):
    dataset = tmp_path / "kline_daily"
    dataset.mkdir()
    pl.DataFrame(
        {
            "symbol": ["000001", "000002.SZ", "920305"],
            "trade_date": ["2024-01-02"] * 3,
            "close": [1.0, 2.0, 3.0],
        }
    ).write_parquet(dataset / "part.parquet")

    manifest = inventory(tmp_path)

    assert manifest.file_count == 1
    assert manifest.dataset_stats["kline_daily"].bare_symbol_rows == 2
    assert manifest.dataset_stats["kline_daily"].qualified_symbol_rows == 1
    assert manifest.dataset_stats["kline_daily"].row_count == 3


def test_write_manifest_is_json_and_does_not_modify_source(tmp_path: Path):
    dataset = tmp_path / "instruments"
    dataset.mkdir()
    source = dataset / "part.parquet"
    pl.DataFrame({"symbol": ["000001.SZ"]}).write_parquet(source)
    before = source.read_bytes()

    report = tmp_path / "report" / "manifest.json"
    manifest = inventory(tmp_path)
    write_manifest(manifest, report)

    assert report.exists()
    assert source.read_bytes() == before
    assert report.read_text(encoding="utf-8").startswith("{")


def test_apply_migration_backups_and_canonicalizes_stock_files(tmp_path: Path):
    instruments = tmp_path / "instruments"
    daily = tmp_path / "kline_daily" / "date=2024-01-02"
    instruments.mkdir()
    daily.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["000001"],
        "code": ["000001"],
        "exchange": ["SZ"],
        "name": ["平安银行"],
    }).write_parquet(instruments / "instruments.parquet")
    source = daily / "part.parquet"
    pl.DataFrame({"symbol": ["000001"], "date": ["2024-01-02"], "close": [10.0]}).write_parquet(source)

    manifest = inventory(tmp_path, full_scan=True)
    result = apply_migration(tmp_path, manifest, migration_id="test-migration")

    assert result.replaced_files == 2
    assert pl.read_parquet(source).get_column("symbol").to_list() == ["000001.SZ"]
    assert (tmp_path / "migration_backups" / "test-migration" / "kline_daily" / "date=2024-01-02" / "part.parquet").exists()


def test_apply_migration_quarantines_unknown_and_conflicting_rows(tmp_path: Path):
    instruments = tmp_path / "instruments"
    daily = tmp_path / "kline_daily"
    instruments.mkdir()
    daily.mkdir()
    pl.DataFrame({
        "symbol": ["000001"],
        "code": ["000001"],
        "exchange": ["SZ"],
    }).write_parquet(instruments / "instruments.parquet")
    source = daily / "part.parquet"
    pl.DataFrame({
        "symbol": ["1", "000001.SH", "999999"],
        "date": ["2024-01-02"] * 3,
        "close": [1.0, 2.0, 3.0],
    }).write_parquet(source)

    manifest = inventory(tmp_path, full_scan=True)
    result = apply_migration(tmp_path, manifest, migration_id="quarantine-test")

    assert result.replaced_files == 2
    assert result.quarantined_rows == 2
    assert pl.read_parquet(source).get_column("symbol").to_list() == ["000001.SZ"]
    quarantine = tmp_path / "migration_backups" / "quarantine-test" / "quarantine" / "kline_daily" / "part.parquet.parquet"
    assert quarantine.exists()
