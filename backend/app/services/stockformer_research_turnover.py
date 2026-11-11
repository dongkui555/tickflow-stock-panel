"""Research-only historical turnover attachment; no current-share inference."""
import polars as pl

VERSION = 'dated_daily_basic_turnover_percent_quarantine_v2'
KEYS = ['symbol', 'date']


def attach_turnover(panel, observations):
    if panel.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise ValueError('Duplicate research panel keys')
    if observations.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise ValueError('Duplicate daily turnover keys')
    source = observations.select(*KEYS, pl.col('turnover_rate').cast(pl.Float64).alias('_dated_turnover'))
    if source.filter(~(pl.col('_dated_turnover').is_finite()
                       & (pl.col('_dated_turnover') >= 0)).fill_null(False)).height:
        raise ValueError('Invalid dated turnover')
    result = panel.join(source, on=KEYS, how='left')
    required = ~pl.col('_pit_lifecycle_excluded').fill_null(False)
    if result.filter(required & pl.col('_dated_turnover').is_null()).height:
        raise ValueError('Incomplete historical turnover coverage')
    unchanged = [c for c in panel.columns if c != 'turnover_rate']
    result = result.with_columns(pl.coalesce('_dated_turnover', 'turnover_rate').alias('turnover_rate'))
    result = result.drop('_dated_turnover').select(panel.columns)
    if not result.select(unchanged).equals(panel.select(unchanged)):
        raise ValueError('Unrelated research inputs changed')
    return result


def turnover_quality_context(panel, disputed_keys):
    """Exclude a disputed observation and the next four stock observations only."""
    keys = panel.select(KEYS).sort(KEYS)
    if keys.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise ValueError('Duplicate turnover chronology keys')
    bad = disputed_keys.select(KEYS).unique().with_columns(pl.lit(True).alias('_disputed'))
    frame = keys.join(bad, on=KEYS, how='left').sort(KEYS).with_columns(pl.col('_disputed').fill_null(False))
    affected = pl.any_horizontal([pl.col('_disputed').shift(i).over('symbol').fill_null(False)
                                  for i in range(5)])
    return frame.select(*KEYS, (~affected).alias('turnover_context_ready'))


def mask_turnover_context(features, context):
    result = features.join(context, on=KEYS, how='left')
    if result.height != features.height or result['turnover_context_ready'].null_count():
        raise ValueError('Incomplete turnover quality context')
    # Every paired fit/inference already rejects non-finite required features.
    # Keep all labels/industry benchmarks, rather than silently imputing a ratio.
    return result.with_columns(pl.when(pl.col('turnover_context_ready')).then(pl.col('turnover_rate'))
                               .otherwise(None).alias('turnover_rate'))
