from datetime import date

import polars as pl
import pytest

from app.services import stockformer as sf
from app.services import minute_features


@pytest.mark.parametrize('profile,expected', [('daily_baseline', False), ('minute_enhanced', True)])
def test_training_only_loads_minutes_for_enhanced_profile(tmp_path, monkeypatch, profile, expected):
    monkeypatch.setattr(sf, '_date_range_from_data', lambda _: (date(2024, 1, 1), date(2025, 1, 1)))
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return pl.DataFrame()

    monkeypatch.setattr(sf, '_load_panel', load)
    sf.train_model(tmp_path, sf.StockFormerTrainConfig(training_profile=profile))
    assert calls[0]['include_minute'] is expected


def test_daily_auxiliary_features_do_not_touch_minute_tree(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('daily baseline must not scan raw minute history')

    monkeypatch.setattr(minute_features, 'build_daily_features', unexpected)
    day = date(2025, 1, 2)
    frame = pl.DataFrame({'symbol': ['000001'], 'date': [day]})
    result = sf._attach_auxiliary_features(tmp_path, frame, day, day, include_minute=False)
    assert result.height == 1
    assert result['minute_data_available'].item() == 0


@pytest.mark.parametrize('columns,expected', [(['ret_1d'], False), (['minute_vwap_gap'], True)])
def test_prediction_only_reads_minutes_if_model_uses_them(tmp_path, monkeypatch, columns, expected):
    day = date(2025, 1, 2)
    monkeypatch.setattr(sf, '_date_range_from_data', lambda _: (day, day))
    monkeypatch.setattr(sf, '_resolve_prediction_target_date', lambda *args: (day, {}))
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return pl.DataFrame()

    monkeypatch.setattr(sf, '_load_panel', load)
    sf._load_prediction_base(tmp_path, day, required_feature_columns=columns)
    assert calls[0]['include_minute'] is expected
