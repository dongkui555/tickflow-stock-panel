from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from app.services import stockformer_calibration_research as research


def calibration_frame():
    return pl.DataFrame([
        {'date': date(2024, 1, 1) + timedelta(days=d), 'symbol': str(s),
         'raw': (s-50)/1000., 'target': (s-50)/10000.-.004}
        for d in range(25) for s in range(100)
    ])


def test_day_rank_uses_unlabelled_rows_and_preserves_ties():
    frame = pl.DataFrame({'date': [date(2024, 1, 1)]*4, 'raw': [1., 2., 2., 4.],
                          'target': [None, .1, -.1, None]})
    ranked = research.with_percentile(frame)
    assert ranked['percentile'].to_list() == pytest.approx([.125, .5, .5, .875])
    scaled = research.with_percentile(frame.with_columns((pl.col('raw')*100+7).alias('raw')))
    assert scaled['percentile'].to_list() == ranked['percentile'].to_list()


@pytest.mark.parametrize('method', ['affine', 'isotonic_raw', 'isotonic_rank'])
def test_new_calibrations_are_serializable_monotone_and_do_not_force_positive(method):
    import json

    frame = research.with_percentile(calibration_frame().with_columns(pl.lit(-.01).alias('target')))
    spec = json.loads(json.dumps(research.fit_method(frame, method)))
    prediction = research.apply_method(frame, spec)
    assert np.max(prediction) < 0
    assert np.min(np.diff(prediction[:100])) >= -1e-12


def test_constant_or_negative_relation_does_not_reverse_ranking():
    frame = research.with_percentile(calibration_frame().with_columns((-pl.col('raw')).alias('target')))
    spec = research.fit_method(frame, 'affine')
    assert spec['slope'] == 0
    assert np.ptp(research.apply_method(frame, spec)) == 0


def test_equal_date_weights_and_invalid_inputs_fail_closed():
    frame = research.with_percentile(calibration_frame())
    weights = research.equal_date_weights(frame)
    assert weights[:100].sum() == pytest.approx(weights[-100:].sum())
    with pytest.raises(ValueError, match='calibration'):
        research.fit_method(frame.head(10), 'isotonic_raw')
    with pytest.raises(ValueError, match='method'):
        research.fit_method(frame, 'unknown')


def test_internal_windows_exclude_all_original_validation_labels():
    days = [date(2024, 1, 1)+timedelta(days=i) for i in range(250)]
    windows = research.internal_windows(days, days[-1])
    assert len(windows) == 3
    for window in windows:
        assert len(window['check_dates']) == 21
        assert len(window['calibration_dates']) == 42
        assert window['check_dates'][-1] < days[-7]
        assert window['calibration_start'] < window['check_start']
    with pytest.raises(ValueError, match='history'):
        research.internal_windows(days[:100], days[99])


def test_shift_comparison_is_zero_for_same_scores_and_separates_rank_from_scale():
    frame = research.with_percentile(calibration_frame())
    assert research.compare_scores(frame, frame)['ks_distance'] == 0
    changed = research.with_percentile(frame.with_columns((pl.col('raw')*10+1).alias('raw')))
    result = research.compare_scores(frame, changed)
    assert result['ks_distance'] > .9
    assert result['percentile_mean_abs_change'] == 0
    assert result['daily_rank_correlation'] == pytest.approx(1)


def test_geometry_uses_coverage_not_label_values_and_keeps_minimums(monkeypatch):
    days = [date(2024, 1, 1)+timedelta(days=i) for i in range(250)]
    frame = pl.DataFrame({'date': days, 'label_end_5d': [d+timedelta(days=6) for d in days],
                          'feature': [None]*45+[1.]*205, 'target_5d_rel_net': [.01]*250})
    monkeypatch.setattr(research.sf, 'prepare_training_features', lambda f, *a, **k: (f, {}))
    windows, attempts = research.coverage_windows(frame, ['feature'], days[-1])
    assert attempts[0]['coverage'][0]['base_training_dates'] < 100
    assert len(windows[0]['check_dates']) < 21
    assert all(c['base_training_dates'] >= 100 and c['calibration_dates'] >= 20
               for c in attempts[-1]['coverage'])
    changed = frame.with_columns(pl.lit(-10000.).alias('target_5d_rel_net'))
    assert research.coverage_windows(changed, ['feature'], days[-1]) == (windows, attempts)
