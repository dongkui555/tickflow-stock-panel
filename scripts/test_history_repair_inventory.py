from datetime import date

import polars as pl
import pytest

from finalize_history_repair_inventory import factor_coverage


def test_partial_date_remains_missing_and_zero_volume_is_not_required():
    day = date(2026, 9, 8)
    factor = pl.DataFrame({"symbol": ["000001"], "trade_date": [day], "ex_factor": [1.1]})
    daily = pl.DataFrame({"symbol": ["1", "000001.SZ", "600000", "600001"],
                          "volume": [100, 1, 100, 0], "close": [10., 10., 10., 0.]})
    assert factor_coverage(factor, daily, day) == ["600000"]
    assert factor_coverage(factor, daily.head(2), day) == []


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), None])
def test_corrupt_staged_factor_is_not_complete(value):
    day = date(2026, 9, 8)
    factor = pl.DataFrame({"symbol": ["000001"], "trade_date": [day],
                           "ex_factor": pl.Series([value], dtype=pl.Float64)})
    daily = pl.DataFrame({"symbol": ["000001"], "volume": [100], "close": [10.]})
    with pytest.raises(ValueError, match="Invalid staged factors"):
        factor_coverage(factor, daily, day)
