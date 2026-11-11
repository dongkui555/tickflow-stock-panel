import pytest
import polars as pl

from app.services.security_identity import (
    IdentityBatchError,
    IdentityError,
    build_instrument_lookup,
    canonicalize_identity_frame,
    identity_key,
    parse_identity,
    validate_frame_identity,
)


def test_build_lookup_uses_exchange_column_for_bare_dimension_symbols():
    lookup = build_instrument_lookup((pl.DataFrame({
        "code": ["920305", "000001"],
        "symbol": ["920305", "000001"],
        "exchange": ["BJ", "SZ"],
    }),))
    assert lookup == {"920305": "920305.BJ", "000001": "000001.SZ"}


def test_wrong_exchange_suffix_is_rejected():
    with pytest.raises(IdentityBatchError) as exc:
        validate_frame_identity(
            pl.DataFrame({"symbol": ["920305.SZ"]}),
            asset_type="stock",
            lookup={"920305": "920305.BJ"},
            require_lookup=True,
        )
    assert "dimension_exchange_conflict" in str(exc.value)


def test_unknown_bare_code_is_isolated_instead_of_defaulting_to_sz():
    with pytest.raises(IdentityBatchError) as exc:
        validate_frame_identity(
            pl.DataFrame({"symbol": ["920305"]}),
            asset_type="stock",
            lookup={},
            require_lookup=True,
        )
    assert "security_not_in_dimension" in str(exc.value)


def test_same_numeric_code_keeps_asset_type_and_exchange_in_key():
    sh_key = identity_key("000001.SH", "2024-01-02 09:31", asset_type="index")
    sz_key = identity_key("000001.SZ", "2024-01-02 09:31", asset_type="stock")
    assert sh_key != sz_key
    assert sh_key.startswith("index|SH|000001|")
    assert sz_key.startswith("stock|SZ|000001|")


def test_no_market_guess_for_active_bare_symbol():
    with pytest.raises(IdentityError, match="exchange_unknown"):
        parse_identity("000001", asset_type="stock", require_explicit=True)


def test_asset_market_collision_is_not_the_same_identity():
    stock = parse_identity("000001.SZ", asset_type="stock", require_explicit=True)
    index = parse_identity("000001.SH", asset_type="index", require_explicit=True)
    assert stock.security_key != index.security_key
    assert stock.symbol == "000001.SZ"
    assert index.symbol == "000001.SH"


def test_unknown_asset_is_rejected_without_market_guess():
    with pytest.raises(IdentityError, match="asset_type_unknown"):
        parse_identity("000001.SZ", asset_type="", require_explicit=True)


def test_bj_explicit_market_is_accepted_for_92_prefix():
    identity = parse_identity("920305.BJ", asset_type="stock", require_explicit=True)
    assert identity.exchange == "BJ"
    assert identity.symbol == "920305.BJ"


def test_b_share_code_namespaces_are_market_specific():
    assert parse_identity("200001", asset_type="stock").symbol == "200001.SZ"
    assert parse_identity("900901", asset_type="stock").symbol == "900901.SH"


def test_source_target_exchange_conflict_is_rejected():
    with pytest.raises(IdentityError, match="expected_exchange_conflict"):
        parse_identity(
            "000001.SH",
            asset_type="stock",
            expected_exchange="SZ",
            require_explicit=True,
        )


def test_overseas_ticker_is_not_coerced_to_a_share():
    with pytest.raises(IdentityError, match="code_unparseable"):
        parse_identity("AAPL", asset_type="stock", require_explicit=True)


def test_canonicalize_frame_accepts_explicit_identity_and_quarantines_unknown():
    result = canonicalize_identity_frame(
        pl.DataFrame({"symbol": ["000001.SZ", "920305"], "close": [10.0, 12.0]}),
        asset_type="stock",
        lookup={"000001": "000001.SZ"},
    )
    assert result.accepted.get_column("symbol").to_list() == ["000001.SZ"]
    assert result.quarantined.get_column("original_symbol").to_list() == ["920305"]
    assert result.reasons == {"exchange_unknown": 1}


def test_instrument_sync_never_persists_bare_symbol():
    from app.services.instrument_sync import _flatten_instruments

    rows = _flatten_instruments([
        {"symbol": "000001", "code": "000001", "exchange": "SZ", "type": "stock"},
    ])
    assert rows[0]["symbol"] == "000001.SZ"
    assert rows[0]["asset_type"] == "stock"


def test_instrument_sync_isolates_unresolved_identity():
    from app.services.instrument_sync import _flatten_instruments

    rows = _flatten_instruments([
        {"symbol": "000001", "code": "000001", "exchange": None, "type": "stock"},
    ])
    assert rows[0]["symbol"] is None
    assert rows[0]["identity_reason"] == "exchange_unknown"
