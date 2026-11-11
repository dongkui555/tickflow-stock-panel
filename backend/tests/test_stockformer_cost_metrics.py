from datetime import date

import numpy as np
import polars as pl
import pytest

from app.services import stockformer_training as training


@pytest.mark.parametrize('mode', ['net', 'cost_after', 'industry_relative_net'])
def test_net_target_metric_does_not_deduct_the_label_cost_twice(mode):
    frame = pl.DataFrame({'symbol': ['000001', '000002', '000003'], 'date': [date(2026, 9, 1)] * 3})
    result = training.metrics(frame, np.array([3., 2., 1.]), np.array([.001, .001, -.001]), target_mode=mode)
    assert result['top2_forward_return'] == pytest.approx(.001)
    assert result['top2_after_cost_proxy'] == pytest.approx(.001)


@pytest.mark.parametrize('mode', ['absolute', 'industry_relative'])
def test_gross_target_metric_still_deducts_cost_once(mode):
    frame = pl.DataFrame({'symbol': ['000001', '000002', '000003'], 'date': [date(2026, 9, 1)] * 3})
    result = training.metrics(frame, np.array([3., 2., 1.]), np.array([.003, .003, -.001]), target_mode=mode)
    assert result['top2_after_cost_proxy'] == pytest.approx(.001)
