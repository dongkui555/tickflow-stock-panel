"""Offline calibration experiments; never wired into frozen live inference."""
from __future__ import annotations

import bisect

import numpy as np
import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_training as training

METHODS = ('identity', 'legacy_tail', 'affine', 'isotonic_raw', 'isotonic_rank')


def internal_windows(calendar, development_end, *, check_sessions=21, calibration_sessions=42):
    dates = sorted(set(calendar))
    stop = bisect.bisect_left(dates, development_end)-6
    first = stop-3*check_sessions
    if first-calibration_sessions < 100:
        raise ValueError('Insufficient training-only history')
    return [{'fold': i+1, 'calibration_start': dates[first+i*check_sessions-calibration_sessions],
             'check_start': dates[first+i*check_sessions],
             'calibration_dates': dates[first+i*check_sessions-calibration_sessions:first+i*check_sessions],
             'check_dates': dates[first+i*check_sessions:first+(i+1)*check_sessions]}
            for i in range(3)]


def coverage_windows(frame, columns, development_end, *, calendar=None):
    """Choose geometry by availability only, before reading prediction results."""
    valid = pl.all_horizontal([pl.col(c).is_finite() for c in [*columns, 'target_5d_rel_net']])
    strict, _ = sf.prepare_training_features(frame, 'minute_enhanced')
    train_keys = strict.filter(valid).select('date', 'label_end_5d').unique()
    calibration_keys = strict.filter(valid).select('date', 'label_end_5d').unique()
    attempts = []
    for check_sessions in range(21, 13, -1):
        windows = internal_windows(calendar if calendar is not None else frame['date'].unique().to_list(),
                                   development_end, check_sessions=check_sessions)
        coverage = []
        for window in windows:
            cal_start, check_start = window['calibration_start'], window['check_start']
            coverage.append({'fold': window['fold'],
                'base_training_dates': train_keys.filter(pl.col('label_end_5d') < cal_start)['date'].n_unique(),
                'calibration_dates': calibration_keys.filter(pl.col('date').is_in(window['calibration_dates'])
                    & (pl.col('label_end_5d') < check_start))['date'].n_unique(),
                'check_label_dates': calibration_keys.filter(pl.col('date').is_in(window['check_dates']))['date'].n_unique()})
        attempts.append({'check_sessions': check_sessions, 'coverage': coverage})
        if all(r['base_training_dates'] >= 100 and r['calibration_dates'] >= 20
               and r['check_label_dates'] >= 10 for r in coverage):
            return windows, attempts
    raise ValueError(f'Insufficient training-only coverage without relaxing minimums: {attempts}')


def with_percentile(frame):
    """Rank the available prediction universe before inspecting future labels."""
    value = pl.when(pl.col('raw').is_finite()).then(pl.col('raw')).otherwise(None)
    return frame.with_columns(((value.rank(method='average').over('date')-.5)
                               / value.count().over('date')).alias('percentile'))


def equal_date_weights(frame):
    return frame.select((1./pl.len().over('date')).alias('weight'))['weight'].to_numpy()


def fit_method(frame, method):
    if method not in METHODS:
        raise ValueError('Unknown calibration method')
    valid = frame.filter(pl.col('raw').is_finite() & pl.col('target').is_finite()
                         & pl.col('percentile').is_finite())
    if valid.height < 1000 or valid['date'].n_unique() < 20:
        raise ValueError('Insufficient independent calibration rows/dates')
    x, y = valid['raw'].to_numpy(), valid['target'].to_numpy()
    weights = equal_date_weights(valid)
    result = {'method': method, 'version': 'training_only_calibration_v1',
              'rows': valid.height, 'dates': valid['date'].n_unique()}
    if method == 'legacy_tail':
        result['mapping'] = sf._fit_extreme_prediction_calibration(x, y)
    elif method == 'affine':
        xm, ym = np.average(x, weights=weights), np.average(y, weights=weights)
        variance = np.average((x-xm)**2, weights=weights)
        slope = max(0., np.average((x-xm)*(y-ym), weights=weights)/variance) if variance > 1e-16 else 0.
        result.update(slope=float(slope), intercept=float(ym-slope*xm))
    elif method.startswith('isotonic_'):
        from sklearn.isotonic import IsotonicRegression

        if method == 'isotonic_rank':
            x = valid['percentile'].to_numpy()
        edges = np.unique(np.quantile(x, np.linspace(0., 1., 21)))
        groups = np.searchsorted(edges[1:-1], x, side='right')
        centres, means, mass = [], [], []
        for group in np.unique(groups):
            mask = groups == group
            centres.append(float(np.average(x[mask], weights=weights[mask])))
            means.append(float(np.average(y[mask], weights=weights[mask])))
            mass.append(float(weights[mask].sum()))
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        iso.fit(centres, means, sample_weight=mass)
        result.update(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist(), bins=20,
                      weighting='equal_dates_then_bin_mass', out_of_bounds='clip')
    return result


def apply_method(frame, spec):
    method = spec['method']
    raw = frame['raw'].to_numpy()
    if method == 'identity':
        return raw.copy()
    if method == 'legacy_tail':
        return sf._apply_extreme_prediction_calibration(raw, spec['mapping'])
    if method == 'affine':
        return spec['slope']*raw + spec['intercept']
    if method in ('isotonic_raw', 'isotonic_rank'):
        x = frame['percentile'].to_numpy() if method == 'isotonic_rank' else raw
        return np.interp(x, spec['x'], spec['y'])
    raise ValueError('Unknown calibration method')


def ks_distance(left, right):
    left, right = np.sort(left[np.isfinite(left)]), np.sort(right[np.isfinite(right)])
    if not len(left) or not len(right):
        return None
    points = np.union1d(left, right)
    return float(np.max(np.abs(np.searchsorted(left, points, side='right')/len(left)
                               - np.searchsorted(right, points, side='right')/len(right))))


def distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'rows': 0}
    return {'rows': len(values), 'mean': float(np.mean(values)), 'std': float(np.std(values)),
            'positive_count': int((values > 0).sum()),
            **{f'q{int(q*100):02}': float(np.quantile(values, q)) for q in (0., .1, .5, .9, .99, 1.)}}


def compare_scores(base, refit):
    if not base.select('symbol', 'date').equals(refit.select('symbol', 'date')):
        raise ValueError('Refit comparison requires identical prediction rows')
    daily = base.select('date', pl.col('raw').alias('base')).with_columns(refit['raw'].alias('refit')).group_by('date').agg(
        pl.corr(pl.col('base').rank(), pl.col('refit').rank()).alias('rank_corr'))
    a, b = base['raw'].to_numpy(), refit['raw'].to_numpy()
    return {'ks_distance': ks_distance(a, b), 'raw_mean_abs_change': float(np.mean(np.abs(a-b))),
            'positive_mask_flips': int(((a > 0) != (b > 0)).sum()),
            'daily_rank_correlation': sf._safe_float(daily['rank_corr'].filter(daily['rank_corr'].is_finite()).mean()),
            'percentile_mean_abs_change': float(np.mean(np.abs(base['percentile'].to_numpy()-refit['percentile'].to_numpy()))),
            'base': distribution(a), 'refit': distribution(b)}


def evaluate(frame, prediction):
    data = frame.with_columns(pl.Series('prediction', prediction)).filter(
        pl.col('prediction').is_finite() & pl.col('target').is_finite())
    if data.is_empty():
        return {'rows': 0}
    p, y, w = data['prediction'].to_numpy(), data['target'].to_numpy(), equal_date_weights(data)
    positive = data.filter(pl.col('prediction') > 0)
    edge = data.filter(pl.col('prediction') >= .0005)
    def conditional(part):
        return {'rows': part.height, 'dates': part['date'].n_unique(),
                'mean_target': sf._safe_float(part.group_by('date').agg(pl.col('target').mean())['target'].mean())}
    return {'rows': data.height, 'dates': data['date'].n_unique(),
            'mse': float(np.average((p-y)**2, weights=w)), 'bias': float(np.average(p-y, weights=w)),
            'scores': distribution(prediction), 'positive': conditional(positive), 'edge_5bp': conditional(edge),
            'label_metrics': training.metrics(data, p, y, target_mode='industry_relative_net')}
