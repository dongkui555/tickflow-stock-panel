"""Classify recorded evidence before making a small bounded source probe."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from app.plugins.tushare.provider import minute_error_details
from app.services.kline_sync import _resolve_minute_provider
from app.services.minute_repair import checked_additions, daily_reference
from app.services.tushare_ext_sync import _temporary_promax_timeout


def classify(manifest: pl.DataFrame, data_dir: Path) -> list[dict]:
    evidence = {}
    for path in (data_dir / 'import_reports/gateway_minute_contract').glob('*/*/*.json.gz'):
        item = json.loads(gzip.decompress(path.read_bytes()))
        key = (item['symbol'], item['date'])
        stamp = path.stat().st_mtime_ns
        if key not in evidence or stamp > evidence[key][0]:
            evidence[key] = (stamp, item, str(path.relative_to(data_dir)))
    rows = []
    for row in manifest.iter_rows(named=True):
        code = row['symbol'].split('.')[0]
        result = {**row, 'reason': 'unclassified_source_error', 'retryable': False,
                  'evidence': '', 'source_symbol': code, 'probe_attempted': False}
        found = evidence.get((code, row['date']))
        if found:
            event = found[1]
            result.update(reason=event['result'], evidence=found[2])
            if result['reason'] == 'accepted':
                result['reason'] = 'source_verified_local_recheck_required'
        elif row['sync_result'] == 'source_empty':
            result['reason'] = 'source_empty'
        if row['sync_result'] == 'daily_turnover_conflict':
            result['reason'] = 'local_daily_turnover_conflict'
        if code == '302132' and row['date'] < '2025-02-17':
            result.update(reason='historical_code_mapping', source_symbol='300114', retryable=True,
                          evidence='https://static.cninfo.com.cn/finalpage/2025-02-15/1222544408.PDF')
        rows.append(result)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-limit', type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = classify(pl.read_csv(args.manifest), args.data_dir)
    pl.DataFrame(rows).write_csv(args.output / 'classified_before_probe.csv', include_bom=True)
    # Four mainland stock-days and one BJ example per remaining date. Known
    # immutable price conflicts do not trigger another identical query.
    candidates = [r for r in rows if r['reason'] in {'unclassified_source_error', 'source_empty'}]
    candidates.sort(key=lambda r: (r['symbol'].endswith('.BJ'), r['date'], r['symbol']))
    selected, buckets = [], set()
    for row in candidates:
        bucket = (row['date'], 'BJ' if row['symbol'].endswith('.BJ') else row['symbol'])
        if bucket not in buckets:
            selected.append(row)
            buckets.add(bucket)
        if len(selected) >= max(0, args.probe_limit):
            break
    if args.probe_limit <= 0:
        selected = []
    provider, fallback, error = _resolve_minute_provider('tushare')
    if not provider or fallback:
        raise RuntimeError(error or 'Source unavailable')
    for i, row in enumerate(selected, 1):
        day = date.fromisoformat(row['date'])
        row['probe_attempted'] = True
        try:
            with _temporary_promax_timeout(provider._ensure_api(), 10):
                frame, issues = provider.get_minute_diagnostics([row['symbol']], datetime.combine(day, time(9, 30)), datetime.combine(day, time(15)))
            if not frame.is_empty():
                refs = daily_reference(pl.read_parquet(args.data_dir / f'kline_daily/date={day}/part.parquet'), data_dir=args.data_dir)
                valid, reason = checked_additions(frame, frame.clear(), refs.get(row['symbol'].split('.')[0]), day)
                row.update(reason='probe_verified' if reason == 'accepted' else reason, retryable=reason == 'accepted')
                if not valid.is_empty():
                    folder = args.output / 'probe_samples'
                    folder.mkdir(exist_ok=True)
                    valid.write_parquet(folder / f"{row['symbol']}_{day}.parquet")
            else:
                exact = [e for e in issues if e.get('date') == str(day)]
                issue = (exact or issues or [{'reason': 'source_empty', 'retryable': False}])[-1]
                row.update({key: issue[key] for key in ('reason', 'retryable', 'evidence', 'http_status') if key in issue})
        except Exception as exc:
            row.update(minute_error_details(exc))
        pl.DataFrame(rows, infer_schema_length=None).write_csv(args.output / 'classified_gaps.csv', include_bom=True)
        print(json.dumps({'probe': i, 'total': len(selected), 'symbol': row['symbol'], 'date': row['date'],
                          'reason': row['reason'], 'retryable': row['retryable']}, ensure_ascii=False), flush=True)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    frame.write_csv(args.output / 'classified_gaps.csv', include_bom=True)
    frame.filter(pl.col('retryable')).write_csv(args.output / 'targeted_repair.csv', include_bom=True)
    summary = {'stock_days': len(rows), 'probe_requests': len(selected),
               'reasons': dict(Counter(r['reason'] for r in rows)),
               'targeted_stock_days': sum(r['retryable'] for r in rows),
               'basis': 'Local evidence plus bounded representative probes; not every remaining pair was queried again'}
    (args.output / 'triage_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
