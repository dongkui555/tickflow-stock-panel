"""Fixed, date-local minute representation; labels never enter this transform."""
from bisect import bisect_left

import polars as pl

KEYS = ['symbol', 'date']
METADATA = {'minute_session_coverage', 'minute_bar_count', 'minute_data_available'}
ARMS = ['daily68', 'minute_raw77', 'minute_rank77']


def feature_sets(columns):
    daily = [c for c in columns if not c.startswith('minute_')]
    behavior = [c for c in columns if c.startswith('minute_') and c not in METADATA]
    if len(columns) != 80 or len(daily) != 68 or len(behavior) != 9:
        raise ValueError('Frozen 68 daily / 9 minute behavior / 3 readiness fields required')
    return dict(daily68=daily, minute_raw77=[*daily, *behavior],
                minute_rank77=[*daily, *['xr_' + c for c in behavior]])


def rank_minutes(reference, columns):
    """Use the full known-day reference, before necessary-pool/label selection."""
    if reference.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise ValueError('Duplicate fixed reference keys')
    behavior = [c for c in columns if c.startswith('minute_') and c not in METADATA]
    if any(c not in reference.columns for c in behavior):
        raise ValueError('Missing minute behavior input')
    if reference.filter(~pl.all_horizontal([pl.col(c).is_finite().fill_null(False) for c in behavior])).height:
        raise ValueError('Nonfinite minute input; never silently drop or impute reference rows')
    return reference.with_columns([
        ((pl.col(c).rank(method='average').over('date')-.5)/pl.len().over('date')-.5).alias('xr_'+c)
        for c in behavior])


def rolling_windows(calendar, cutoff):
    """Dates only: three 21-session diagnostics, with six settlement sessions."""
    dates = sorted(set(calendar))
    stop = bisect_left(dates, cutoff)-6
    if stop-63-6-20-6-252 < 0:
        raise ValueError('Insufficient fixed rolling history')
    windows = []
    for i in range(3):
        first = stop-63+21*i
        cal_first = first-6-20
        windows.append(dict(name=f'rolling-{i+1}', train_start=dates[cal_first-6-252],
            calibration_start=dates[cal_first], validation_start=dates[first],
            validation_dates=dates[first:first+21], settlement_dates=dates[first+21:first+27],
            label_cut_exclusive=dates[first+27]))
    return windows


def latest_window(calendar, cutoff):
    dates = sorted(set(calendar))
    first = bisect_left(dates, cutoff)
    cal_first = first-6-20
    if cal_first-6-252 < 0:
        raise ValueError('Insufficient latest rolling history')
    return dict(name='latest-primary', train_start=dates[cal_first-6-252],
        calibration_start=dates[cal_first], validation_start=cutoff,
        validation_dates=[], settlement_dates=[], label_cut_exclusive=cutoff)
