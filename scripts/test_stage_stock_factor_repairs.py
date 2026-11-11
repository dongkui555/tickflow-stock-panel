from datetime import date

import polars as pl
import pytest

from stage_stock_factor_repairs import validate_factors


def fixtures():
    raw = pl.DataFrame({"ts_code": ["000001.SZ"] * 2, "trade_date": ["20260102", "20260105"], "adj_factor": [1.1, 1.2]})
    anchor = pl.DataFrame({"symbol": ["000001"], "trade_date": [date(2026, 1, 5)], "ex_factor": [1.2]})
    return raw, anchor


def test_requires_exact_coverage():
    raw, anchor = fixtures()
    _, missing = validate_factors(raw, "000001.SZ", [date(2026, 1, 2), date(2026, 1, 5)], anchor)
    assert missing == []
    _, missing = validate_factors(raw.tail(1), "000001.SZ", [date(2026, 1, 2)], anchor)
    assert missing == [date(2026, 1, 2)]


@pytest.mark.parametrize("change,reason", [
    (lambda r: r.with_columns(pl.lit("000002.SZ").alias("ts_code")), "unexpected_factor_symbol"),
    (lambda r: pl.concat([r, r]), "duplicate_factor_dates"),
    (lambda r: r.with_columns(pl.lit(0.0).alias("adj_factor")), "invalid_factor_values"),
    (lambda r: r.with_columns(pl.lit(float("nan")).alias("adj_factor")), "invalid_factor_values"),
    (lambda r: r.with_columns(pl.lit(2.0).alias("adj_factor")), "factor_anchor_conflict"),
    (lambda r: r.head(1), "factor_anchor_missing"),
])
def test_rejects_unverifiable_factors(change, reason):
    raw, anchor = fixtures()
    with pytest.raises(ValueError, match=reason):
        validate_factors(change(raw), "000001.SZ", [date(2026, 1, 2)], anchor)
