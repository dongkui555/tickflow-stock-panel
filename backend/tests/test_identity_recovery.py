from datetime import date

from app.services.identity_recovery import (
    dedupe_key,
    is_valid_ohlcv_row,
    resolve_quarantined_identity,
)


def lifecycle(*symbols):
    result = {}
    for symbol in symbols:
        code, exchange = symbol.split(".")
        result[("stock", exchange, code)] = [{
            "symbol": symbol,
            "code": code,
            "exchange": exchange,
            "asset_type": "stock",
            "status": "listed",
            "listing_date": date(2000, 1, 1),
            "delist_date": None,
            "effective_from": date(2000, 1, 1),
            "effective_to": None,
            "successor_symbol": "",
            "event_type": "",
            "source": "test",
        }]
    return result


def test_recovery_uses_dimension_and_date_validity_for_bare_code():
    decision = resolve_quarantined_identity(
        "1",
        asset_type="stock",
        trade_date=date(2024, 1, 2),
        lookup={"000001": "000001.SZ"},
        lifecycle_index=lifecycle("000001.SZ"),
    )

    assert decision.ok is True
    assert decision.symbol == "000001.SZ"
    assert decision.basis == "dimension"


def test_recovery_rejects_ambiguous_historical_market_identity():
    ambiguous = lifecycle("000001.SZ", "000001.SH")

    decision = resolve_quarantined_identity(
        "1",
        asset_type="stock",
        trade_date=date(2024, 1, 2),
        lookup={},
        lifecycle_index=ambiguous,
        allow_structural_stock_market=True,
    )

    assert decision.ok is False
    assert decision.reason == "market_identity_ambiguous"


def test_b_share_structural_rule_is_opt_in_and_requires_lifecycle():
    index = lifecycle("200011.SZ")
    rejected = resolve_quarantined_identity(
        "200011",
        asset_type="stock",
        trade_date=date(2024, 1, 2),
        lookup={},
        lifecycle_index=index,
    )
    accepted = resolve_quarantined_identity(
        "200011",
        asset_type="stock",
        trade_date=date(2024, 1, 2),
        lookup={},
        lifecycle_index=index,
        allow_structural_stock_market=True,
    )

    assert rejected.reason == "exchange_unknown"
    assert accepted.ok is True
    assert accepted.symbol == "200011.SZ"
    assert accepted.basis == "code_rule"


def test_invalid_ohlcv_is_not_recoverable():
    assert is_valid_ohlcv_row({
        "open": 10,
        "high": 9,
        "low": 8,
        "close": 8.5,
        "volume": 100,
        "amount": 850,
    }) is False
    assert is_valid_ohlcv_row({
        "open": 10,
        "high": 11,
        "low": 9,
        "close": 10.5,
        "volume": 100,
        "amount": 1050,
    }) is True


def test_recovery_dedupe_keys_preserve_financial_and_adjustment_history():
    import polars as pl

    financials = pl.DataFrame({
        "symbol": ["000001.SZ"],
        "period_end": ["2024-12-31"],
        "announce_date": ["2025-04-30"],
    })
    adj_factor = pl.DataFrame({
        "symbol": ["000001.SZ"],
        "trade_date": ["2024-01-02"],
    })

    assert dedupe_key("financials", financials) == ["symbol", "period_end", "announce_date"]
    assert dedupe_key("adj_factor", adj_factor) == ["symbol", "trade_date"]
