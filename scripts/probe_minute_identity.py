"""Read-only source probes for a suspected historical security-code change."""
from __future__ import annotations

import json
from pathlib import Path

from app.services.kline_sync import _resolve_minute_provider
from app.services.tushare_ext_sync import _temporary_promax_timeout


def main():
    root = Path('/app/data/import_reports/minute_gap_triage_20260904')
    root.mkdir(parents=True, exist_ok=True)
    provider, fallback, error = _resolve_minute_provider('tushare')
    if not provider or fallback:
        raise RuntimeError(error or 'Configured source unavailable')
    api = provider._ensure_api()
    results = []
    queries = [(kind, code) for kind in ('stock_basic', 'namechange', 'daily', 'stk_mins')
               for code in ('302132.SZ', '300114.SZ')]
    with _temporary_promax_timeout(api, 10):
        for endpoint, code in queries:
            params = {'ts_code': code}
            if endpoint == 'daily':
                params.update(start_date='20240904', end_date='20240904')
            if endpoint == 'stk_mins':
                params.update(freq='1min', start_date='2024-09-04 09:25:00', end_date='2024-09-04 15:10:00')
            item = {'endpoint': endpoint, 'query_symbol': code}
            try:
                frame = getattr(api, endpoint)(**params)
                item.update(rows=len(frame), columns=list(frame.columns), records=frame.to_dict(orient='records'))
            except Exception as exc:
                item.update(error=type(exc).__name__)
            results.append(item)
            (root / 'identity_source_probes.json').write_text(json.dumps(results, default=str, ensure_ascii=False, indent=2), encoding='utf-8')
            visible = {k: v for k, v in item.items() if k != 'records'}
            if endpoint != 'stk_mins':
                visible['records'] = item.get('records', [])[:8]
            print(json.dumps(visible, default=str, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
