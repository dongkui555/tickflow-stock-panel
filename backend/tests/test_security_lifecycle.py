from datetime import date

import polars as pl

from app.services.security_lifecycle import (
    load_lifecycle_index,
    validate_security_lifecycle,
    validate_supplement_identity,
)


def _write_dimensions(root):
    root.joinpath("instruments").mkdir(parents=True)
    root.joinpath("instruments_history").mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["920305"], "code": ["920305"], "exchange": ["BJ"],
        "type": ["stock"], "listing_date": ["20210826"],
    }).write_parquet(root / "instruments/instruments.parquet")
    pl.DataFrame({
        "symbol": ["920680.BJ"], "code": ["920680"], "exchange": ["BJ"],
        "type": ["stock"], "listing_date": ["20211115"],
        "list_status": ["D"], "delist_date": ["20260105"],
        "effective_from": ["20211115"], "effective_to": ["20260105"],
    }).write_parquet(root / "instruments_history/instruments_history.parquet")


def test_lifecycle_blocks_before_listing_and_after_delisting(tmp_path):
    _write_dimensions(tmp_path)
    index = load_lifecycle_index(tmp_path)
    before = validate_security_lifecycle("920305.BJ", date(2020, 1, 1), tmp_path, index=index)
    assert before.reason == "not_listed_yet"
    after = validate_security_lifecycle("920680.BJ", date(2026, 1, 6), tmp_path, index=index)
    assert after.reason == "delisted_on_date"


def test_source_market_must_match_target(tmp_path):
    _write_dimensions(tmp_path)
    result = validate_supplement_identity(
        "920305.BJ", "920305.SZ", date(2024, 1, 2), tmp_path,
        index=load_lifecycle_index(tmp_path),
    )
    assert result["reason"] == "market_identity_conflict"


def test_explicit_code_change_stops_old_identity(tmp_path):
    _write_dimensions(tmp_path)
    changes = tmp_path / "instruments_history/code_changes.parquet"
    pl.DataFrame({
        "symbol": ["300114.SZ"], "code": ["300114"], "exchange": ["SZ"],
        "type": ["stock"], "effective_from": ["2010-08-27"],
        "effective_to": ["2025-02-16"], "successor_symbol": ["302132.SZ"],
        "event_type": ["official_code_change"],
    }).write_parquet(changes)
    decision = validate_security_lifecycle(
        "300114.SZ", date(2025, 2, 17), tmp_path,
        index=load_lifecycle_index(tmp_path),
    )
    assert decision.reason == "code_changed"
    assert decision.successor_symbol == "302132.SZ"
