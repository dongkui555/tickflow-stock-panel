from datetime import datetime

import httpx
import pandas as pd
import pytest

from app.plugins.tushare.provider import TushareProvider, _PromaxTushareAPI


def test_disclosed_code_change_splits_requests_and_keeps_current_identity(monkeypatch):
    calls = []
    class API:
        def stk_mins(self, **kwargs):
            calls.append(kwargs)
            stamp = '2025-02-14 10:00:00' if kwargs['ts_code'] == '300114.SZ' else '2025-02-17 10:00:00'
            return pd.DataFrame([{'ts_code': kwargs['ts_code'], 'trade_time': stamp,
                                  'open': 10., 'high': 10., 'low': 10., 'close': 10., 'vol': 100., 'amount': 1000.}])
    provider = TushareProvider()
    monkeypatch.setattr(provider, '_ensure_api', lambda: API())
    frame = provider.get_minute(['302132.SZ'], datetime(2025, 2, 14, 9, 30), datetime(2025, 2, 17, 15))
    assert [c['ts_code'] for c in calls] == ['300114.SZ', '302132.SZ']
    assert frame['symbol'].unique().to_list() == ['302132']
    assert frame.height == 2
    calls.clear()
    provider.get_minute(['302132.SZ'], datetime(2025, 2, 17, 9, 30), datetime(2025, 2, 17, 15))
    assert [c['ts_code'] for c in calls] == ['302132.SZ']


@pytest.mark.parametrize('status,reason,retryable', [
    (429, 'rate_limited', True), (503, 'upstream_unavailable', True),
    (401, 'permission_denied', False), (403, 'permission_denied', False),
    (404, 'endpoint_not_found', False),
])
def test_diagnostics_keep_http_category_without_secrets(monkeypatch, status, reason, retryable):
    api = _PromaxTushareAPI('https://example.test', 'secret-that-must-not-leak')
    api.close()
    api._client = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(status, headers={'Retry-After': '60'}, json={'message': 'secret-that-must-not-leak'})
    ))
    provider = TushareProvider()
    monkeypatch.setattr(provider, '_ensure_api', lambda: api)
    try:
        frame, issues = provider.get_minute_diagnostics(['000001'], datetime(2024, 9, 4, 9, 30), datetime(2024, 9, 4, 15))
        assert frame.is_empty()
        assert issues[-1]['reason'] == reason
        assert issues[-1]['retryable'] is retryable
        assert 'secret-that-must-not-leak' not in str(issues)
    finally:
        api.close()


def test_empty_is_not_a_transient_error(monkeypatch):
    class API:
        def stk_mins(self, **kwargs):
            return pd.DataFrame()
    provider = TushareProvider()
    monkeypatch.setattr(provider, '_ensure_api', lambda: API())
    frame, issues = provider.get_minute_diagnostics(['000001'], datetime(2024, 9, 4, 9, 30), datetime(2024, 9, 4, 15))
    assert frame.is_empty()
    assert issues[0]['reason'] == 'source_empty'
    assert issues[0]['retryable'] is False


def test_partial_window_preserves_exact_validation_reason(tmp_path, monkeypatch):
    from test_promax_minute_contract import daily, source

    from app.config import settings
    raw = source()
    raw2 = raw.assign(trade_time=raw.trade_time + pd.Timedelta(days=1))
    monkeypatch.setattr(settings, 'data_dir', tmp_path)
    api = _PromaxTushareAPI('https://example.test', 'key')
    api.close()
    def response(request):
        df = pd.concat([raw, raw2]).assign(trade_time=lambda x: x.trade_time.astype(str)) if request.url.path.endswith('stk_mins') else daily()
        return httpx.Response(200, json={'code': 0, 'data': df.to_dict(orient='records')})
    api._client = httpx.Client(transport=httpx.MockTransport(response))
    provider = TushareProvider()
    monkeypatch.setattr(provider, '_ensure_api', lambda: api)
    try:
        frame, issues = provider.get_minute_diagnostics(['920000'], datetime(2026, 5, 8, 9, 30), datetime(2026, 5, 9, 15))
        assert frame.height == 241
        rejected = [r for r in issues if r.get('date') == '2026-05-09']
        assert rejected[0]['reason'] == 'daily_reference_missing'
        assert rejected[0]['retryable'] is False
        assert rejected[0]['evidence']
    finally:
        api.close()
