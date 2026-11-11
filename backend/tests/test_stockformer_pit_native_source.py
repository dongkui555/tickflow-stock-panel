from datetime import date

import pandas as pd
import pytest

from app.services.stockformer_pit_sources import (
    ReferenceCoverageError,
    ReferenceSourceError,
    fetch_reference_day,
    should_fetch_reference,
)


class Native:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def bak_basic(self, **params):
        self.calls.append(params)
        return pd.DataFrame(self.pages[len(self.calls)-1])


def row(code):
    return {'ts_code': code+'.SZ', 'trade_date': '20240102', 'name': '名称', 'industry': '行业'}


def test_complete_native_day_requires_only_one_request_without_paging_parameters():
    api = Native([[row('000001'), row('000002'), row('000003')]])
    records, report = fetch_reference_day(api, 'bak_basic', date(2024, 1, 2),
        ['000001', '000002', '000003'], page_size=2, native_first=True)
    assert len(records) == 3 and report['coverage_complete']
    assert len(api.calls) == 1 and 'offset' not in api.calls[0] and 'limit' not in api.calls[0]


def test_truncated_native_response_resumes_after_actual_returned_rows():
    api = Native([[row('000001'), row('000002')], [row('000003')]])
    records, report = fetch_reference_day(api, 'bak_basic', date(2024, 1, 2),
        ['000001', '000002', '000003'], page_size=2, native_first=True)
    assert len(records) == 3 and report['coverage_complete']
    assert 'offset' not in api.calls[0] and api.calls[1]['offset'] == 2


def test_source_failure_exposes_safe_category_without_upstream_text():
    class Failing:
        def bak_basic(self, **params):
            raise RuntimeError('private credential or upstream text')

    with pytest.raises(ReferenceSourceError) as caught:
        fetch_reference_day(Failing(), 'bak_basic', date(2024, 1, 2), ['000001'], native_first=True)
    assert caught.value.error_type == 'RuntimeError' and caught.value.retryable
    assert 'credential' not in str(caught.value)


def test_partial_reference_evidence_survives_coverage_error_without_success():
    api = Native([[row('000001')], []])
    with pytest.raises(ReferenceCoverageError) as caught:
        fetch_reference_day(api, 'bak_basic', date(2024, 1, 2), ['000001', '000002'],
            page_size=2, native_first=True)
    assert caught.value.missing_symbols == ['000002']
    assert caught.value.records == [row('000001')]
    assert not caught.value.verification['coverage_complete']
    assert caught.value.verification['covered_symbols'] == 1


def test_retry_plan_does_not_repeat_structural_gaps_or_permission_failures():
    for status in ['complete', 'coverage_gap', 'permission_denied', 'invalid_response']:
        previous = {'plan_sha256': 'v1', 'status': status}
        assert not should_fetch_reference(previous, 'v1', now=1000)
    assert should_fetch_reference({'plan_sha256': 'v1', 'status': 'coverage_gap'}, 'v2', now=1000)


def test_transient_attempts_and_rate_cooldown_have_different_budgets():
    deferred = {'plan_sha256': 'v1', 'status': 'retryable_error', 'retry_at': 200, 'failures': 1}
    assert not should_fetch_reference(deferred, 'v1', now=199)
    assert should_fetch_reference(deferred, 'v1', now=200)
    assert not should_fetch_reference({**deferred, 'failures': 3}, 'v1', now=300)
    assert should_fetch_reference({**deferred, 'status': 'rate_limited', 'failures': 0}, 'v1', now=300)
