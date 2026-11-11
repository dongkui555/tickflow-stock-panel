"""Build a remaining-gap handoff without querying or rewriting market data."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import polars as pl

from triage_minute_gaps import classify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--last-report', type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder
    remaining = pl.read_csv(folder / 'final_state/remaining_minute_stock_days.csv')
    previous = pl.read_csv(folder / 'classified_gaps.csv')
    original_results = {(r['symbol'], r['date']): r['sync_result'] for r in previous.to_dicts()}
    seed = [{**r, 'sync_result': original_results.get((r['symbol'], r['date']), r['sync_result'])}
            for r in remaining.to_dicts()]
    rows = classify(pl.DataFrame(seed), args.data_dir)
    by_key = {(r['symbol'].split('.')[0], r['date']): r for r in rows}

    # Events are supplied in actual execution order. Only directly probed rows
    # override an earlier diagnosis; unqueried rows remain explicitly unknown.
    events = previous.filter(pl.col('probe_attempted')).to_dicts()
    state = json.loads((folder / 'final_state/summary.json').read_text(encoding='utf-8'))
    for path in state['repair_reports']:
        if Path(path) != args.last_report:
            events.extend(json.loads(Path(path).read_text(encoding='utf-8'))['details'])
    events.extend(pl.read_csv(folder / 'edge_probe/classified_gaps.csv').filter(pl.col('probe_attempted')).to_dicts())
    report = json.loads(args.last_report.read_text(encoding='utf-8'))
    events.extend(report['details'])
    for event in events:
        row = by_key.get((event['symbol'].split('.')[0], event['date']))
        if row is None:
            continue
        for key in ('reason', 'retryable', 'evidence', 'http_status'):
            if event.get(key) is not None:
                row[key] = event[key]
        row['retryable'] = row['reason'] in {
            'request_timeout', 'upstream_unavailable', 'transport_error', 'rate_limited',
        }
    frame = pl.DataFrame(rows, infer_schema_length=None).sort('date', 'symbol')
    frame.write_csv(folder / 'classified_remaining.csv', include_bom=True)
    source_errors = {'daily_price_conflict', 'daily_turnover_conflict', 'local_daily_turnover_conflict',
                     'source_subminute_timestamps', 'ambiguous_closing_bucket', 'source_schema_invalid'}
    frame.filter(pl.col('reason').is_in(source_errors)).write_csv(folder / 'supplier_conflicts.csv', include_bom=True)
    frame.filter(pl.col('retryable')).write_csv(folder / 'retryable_remaining.csv', include_bom=True)
    by_date = frame.with_columns(pl.col('symbol').str.split('.').list.last().alias('exchange')).group_by(
        'exchange', 'date',
    ).agg(pl.len().alias('stock_days'), pl.col('missing_minutes').sum()).sort('exchange', 'date')
    by_date.write_csv(folder / 'remaining_by_exchange_date.csv', include_bom=True)
    summary = {
        'scope': 'Rechecked the original 646896-minute manifest, not a fresh full-database audit.',
        'remaining_stock_days': frame.height,
        'remaining_missing_minutes': int(frame['missing_minutes'].sum()),
        'reasons': dict(Counter(r['reason'] for r in rows)),
        'exchanges': by_date.group_by('exchange').agg(
            pl.col('stock_days').sum(), pl.col('missing_minutes').sum(),
        ).sort('exchange').to_dicts(),
        'known_retryable_stock_days': sum(r['retryable'] for r in rows),
        'note': 'Unknowns are not all network errors. Conflicts and empty responses are not automatically overwritten or zero-filled.',
    }
    (folder / 'handoff_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
