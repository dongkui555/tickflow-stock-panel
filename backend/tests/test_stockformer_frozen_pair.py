import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from app.services import stockformer as sf
from app.services import stockformer_frozen_pair as pair


def test_default_minute_policy_matches_all_stages_and_preserves_explicit_legacy():
    frame = pl.DataFrame({'symbol': ['full', 'sparse', 'gap', 'bad'], 'date': [date(2024, 1, 1)]*4,
        'minute_data_available': [1.]*4, 'minute_bar_count': [240., 239., 240., float('inf')],
        'minute_session_coverage': [1., 239/240, 239/240, 1.]})
    ready, report = sf.prepare_training_features(frame, 'minute_enhanced')
    assert ready['symbol'].to_list() == ['full']
    assert report['minimum_minute_bars'] == 240
    assert report['minimum_minute_coverage'] == 1
    assert report['minute_policy_version'] == sf.MINUTE_AVAILABILITY_POLICY['version']
    legacy, _ = sf.prepare_training_features(frame.head(2), 'minute_enhanced',
                                            minimum_minute_bars=228, minimum_minute_coverage=.95)
    assert legacy['symbol'].to_list() == ['full', 'sparse']


def bundle(tmp_path):
    (tmp_path/'xgboost.json').write_text('tree-x')
    (tmp_path/'lightgbm.txt').write_text('tree-l')
    calibration = {'method': 'affine', 'slope': .5, 'intercept': -.001}
    (tmp_path/'calibration.json').write_text(json.dumps(calibration))
    manifest = pair.freeze_manifest(tmp_path, feature_columns=['ret_1d'], provenance={
        'tree_max_label_end': '2024-07-31', 'calibration_start': '2024-08-01',
        'calibration_max_label_end': '2024-09-01', 'frozen_as_of': '2024-09-02'})
    return manifest


def test_pair_manifest_refuses_tree_calibration_or_policy_substitution(tmp_path):
    manifest = bundle(tmp_path)
    assert pair.verify_manifest(tmp_path)['pair_id'] == manifest['pair_id']
    tree = tmp_path/'xgboost.json'
    original = tree.read_text()
    tree.write_text('refitted-tree')
    with pytest.raises(ValueError, match='fingerprint'):
        pair.verify_manifest(tmp_path)
    tree.write_text(original)
    calibration = tmp_path/'calibration.json'
    calibration.write_text(json.dumps({'method': 'affine', 'slope': 1., 'intercept': 0.}))
    with pytest.raises(ValueError, match='fingerprint'):
        pair.verify_manifest(tmp_path)


def test_pair_refuses_label_overlap_and_overwrite(tmp_path):
    bundle(tmp_path)
    with pytest.raises(ValueError, match='exists'):
        pair.freeze_manifest(tmp_path, feature_columns=['ret_1d'], provenance={})
    manifest = json.loads((tmp_path/'pair.json').read_text())
    manifest['minute_policy']['minimum_minute_bars'] = 228
    manifest['pair_id'] = pair._identity(manifest)
    (tmp_path/'pair.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r'policy|fingerprint'):
        pair.verify_manifest(tmp_path)


def test_calibration_labels_cannot_overlap_the_checkpoint(tmp_path):
    (tmp_path/'xgboost.json').write_text('x')
    (tmp_path/'lightgbm.txt').write_text('l')
    (tmp_path/'calibration.json').write_text(json.dumps({'method': 'affine', 'slope': 1., 'intercept': 0.}))
    with pytest.raises(ValueError, match='overlap'):
        pair.freeze_manifest(tmp_path, feature_columns=['ret_1d'], provenance={
            'tree_max_label_end': '2024-08-01', 'calibration_start': '2024-08-01',
            'calibration_max_label_end': '2024-09-01', 'frozen_as_of': '2024-09-02'})


def test_no_future_labels_needed_and_sparse_rows_never_reach_trees(tmp_path, monkeypatch):
    bundle(tmp_path)
    frame = pl.DataFrame({'symbol': ['full', 'sparse'], 'date': [date(2024, 9, 3)]*2,
        'ret_1d': [.02, .03], 'minute_data_available': [1., 1.],
        'minute_bar_count': [240., 239.], 'minute_session_coverage': [1., 239/240]})
    monkeypatch.setattr(pair, '_load_trees', lambda root: (object(), object()))
    seen = []
    def predict(_model, x):
        seen.append(len(x))
        return x[:, 0]
    monkeypatch.setattr(sf, '_predict_xgboost', predict)
    monkeypatch.setattr(sf, '_predict_lightgbm', predict)
    prediction, coverage = pair.predict_pair(tmp_path, frame)
    assert seen == [1, 1]
    assert prediction['symbol'].to_list() == ['full']
    assert prediction['pred_ret_5d'][0] == pytest.approx(.009)
    assert coverage['ready_rows'] == 1


def test_empty_prediction_returns_a_typed_frame_without_loading_trees(tmp_path, monkeypatch):
    bundle(tmp_path)
    monkeypatch.setattr(pair, '_load_trees', lambda root: pytest.fail('No rows should reach trees'))
    prediction, _ = pair.predict_pair(tmp_path, pl.DataFrame({'symbol': ['sparse'], 'date': [date(2024, 9, 3)],
        'ret_1d': [.1], 'minute_data_available': [1.], 'minute_bar_count': [238.],
        'minute_session_coverage': [238/240]}))
    assert prediction.is_empty()
    assert prediction.schema['date'] == pl.Date


def test_pair_training_and_calibration_are_independent_of_parquet_row_order(tmp_path, monkeypatch):
    rows = []
    for day_index in range(175):
        day = date(2024, 1, 1)+timedelta(days=day_index)
        for symbol in range(25):
            value = (day_index % 17+symbol)/1000
            rows.append({'symbol': str(symbol), 'date': day, 'ret_1d': value,
                         pair.TARGET: value*.5-.002, 'label_end_5d': day+timedelta(days=6),
                         'minute_data_available': 1., 'minute_bar_count': 240., 'minute_session_coverage': 1.})
    frame = pl.DataFrame(rows)
    seen = []

    class DummyTree:
        booster_ = property(lambda self: self)

        def save_model(self, path):
            Path(path).write_text('fixed-test-tree')

    def fit(x, y, **kwargs):
        seen.append((x.copy(), y.copy()))
        return DummyTree()

    monkeypatch.setattr(sf, '_fit_xgboost', fit)
    monkeypatch.setattr(sf, '_fit_lightgbm', fit)
    monkeypatch.setattr(sf, '_predict_xgboost', lambda model, x: x[:, 0])
    monkeypatch.setattr(sf, '_predict_lightgbm', lambda model, x: x[:, 0])
    for index, sample in enumerate((frame, frame.reverse())):
        pair.fit_pair(tmp_path/str(index), sample, ['ret_1d'], calibration_start=date(2024, 5, 1),
                      label_cut=date(2024, 6, 25))
    np.testing.assert_array_equal(seen[0][0], seen[2][0])
    np.testing.assert_array_equal(seen[0][1], seen[2][1])
    assert (tmp_path/'0/calibration.json').read_bytes() == (tmp_path/'1/calibration.json').read_bytes()
