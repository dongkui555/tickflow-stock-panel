"""Explicit stable price units for this new research dataset only."""
import polars as pl

from app.services import stockformer_pit_inputs as pit

VERSION = 'raw_ohlc_times_same_day_cumulative_factor_v1'


def feature_price_view(panel, raw, factors):
    """Use same-day absolute cumulative factors; no future normalization base.

    Execution keeps its separate QFQ/raw panels. These feature prices are
    internal calculation units and are never published as enriched QFQ bars.
    """
    keys, prices = ['symbol', 'date'], ['open', 'high', 'low', 'close']
    factors = pit._keys(factors, 'complete research factors').select(*keys, pl.col('adj_factor').cast(pl.Float64))
    assert factors.filter(~pl.col('adj_factor').is_finite() | (pl.col('adj_factor') <= 0)).is_empty()
    native = pit._keys(raw, 'research raw prices').select(*keys,
        *[pl.col(c).alias('_basis_raw_' + c) for c in prices])
    frame = panel.join(native, on=keys, how='left').join(factors, on=keys, how='left')
    assert frame.height == panel.height
    assert frame.filter(~pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0)
        for c in ['adj_factor', *['_basis_raw_' + c for c in prices]]]).fill_null(False)).is_empty()
    frame = frame.with_columns([(pl.col('_basis_raw_' + c) * pl.col('adj_factor')).alias(c)
                                for c in prices])
    if '_auction_open_price' in frame.columns:
        assert '_raw_auction_open_price' in frame.columns
        frame = frame.with_columns((pl.col('_raw_auction_open_price') * pl.col('adj_factor'))
                                    .alias('_auction_open_price'))
    return frame.drop('adj_factor', *['_basis_raw_' + c for c in prices]).sort(keys)
