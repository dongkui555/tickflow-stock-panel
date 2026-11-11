from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from app.services import stockformer_pit_sources as sources

DAY = date(2024, 11, 21)


class Pages:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def bak_basic(self, **params):
        self.calls.append(params)
        return pd.DataFrame(self.pages[len(self.calls)-1])


def row(code, day='20241121'):
    return {'ts_code': code+'.SZ', 'trade_date': day, 'name': '名称', 'industry': '行业'}


def test_pagination_checks_expected_coverage_and_actual_offsets():
    api = Pages([[row('000001'), row('000002')], [row('000003')]])
    records, report = sources.fetch_reference_day(api, 'bak_basic', DAY,
        ['000001', '000002', '000003'], page_size=2)
    assert len(records) == 3 and report['coverage_complete']
    assert [call['offset'] for call in api.calls] == [0, 2]


def test_short_gateway_response_is_not_complete_universe():
    api = Pages([[row('000001')]])
    with pytest.raises(sources.ReferenceCoverageError) as caught:
        sources.fetch_reference_day(api, 'bak_basic', DAY, ['000001', '000002'], page_size=2)
    assert caught.value.missing_symbols == ['000002']


def test_repeated_page_and_wrong_date_are_rejected():
    api = Pages([[row('000001')], [row('000001')]])
    with pytest.raises(ValueError, match='Repeated'):
        sources.fetch_reference_day(api, 'bak_basic', DAY, ['000001', '000002'], page_size=1)
    with pytest.raises(ValueError, match='date'):
        sources.fetch_reference_day(Pages([[row('000001', '20260101')]]), 'bak_basic', DAY, ['000001'])


def test_wrapped_429_exposes_cooldown_without_repeating_request():
    calls = []

    class Limited:
        def bak_basic(self, **params):
            calls.append(params)
            original = RuntimeError('sensitive upstream text')
            original.response = SimpleNamespace(status_code=429, headers={'Retry-After': '120'})
            raise RuntimeError('wrapped') from original

    with pytest.raises(sources.ReferenceRateLimitError) as caught:
        sources.fetch_reference_day(Limited(), 'bak_basic', DAY, ['000001'])
    assert caught.value.retry_after_seconds == 120 and len(calls) == 1
    assert 'sensitive' not in str(caught.value)


def test_two_sources_cannot_silently_mix_pages():
    class Changing(Pages):
        def bak_basic(self, **params):
            self.last_source = 'primary' if not self.calls else 'backup'
            return super().bak_basic(**params)

    with pytest.raises(ValueError, match='source changed'):
        sources.fetch_reference_day(Changing([[row('000001')], [row('000002')]]),
                                    'bak_basic', DAY, ['000001', '000002'], page_size=1)
