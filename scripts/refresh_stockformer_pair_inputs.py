"""Version recent research features/labels from read-only canonical sources."""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from app.services import stockformer as sf
from app.services import stockformer_frozen_pair as pair
from app.services import stockformer_training as training
from app.services.minute_features import FEATURE_DEFAULTS, build_daily_features


def write(path, value):
    path.write_text(json.dumps(sf._json_safe(value), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def source_versions(root, start, end):
    result = {}
    for table in ('kline_daily', 'kline_daily_enriched', 'kline_minute', 'moneyflow', 'auction_open',
                  'news', 'adj_factor', 'ext_data', 'instruments'):
        for path in sorted((root/table).rglob('*.parquet')):
            partition = next((p[5:] for p in path.parts if p.startswith('date=')), None)
            if partition and not str(start) <= partition <= str(end):
                continue
            stat = path.stat()
            result[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--old-features', type=Path, required=True)
    parser.add_argument('--old-panel', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start', type=date.fromisoformat, required=True)
    parser.add_argument('--end', type=date.fromisoformat, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out/'input-manifest.json').exists():
        raise ValueError('Refreshed inputs exist; preserve version')
    old = pl.read_parquet(args.old_features)
    symbols = sorted(old['symbol'].unique().to_list())
    source_start = args.end-timedelta(days=160)
    before = source_versions(args.data_dir, source_start, args.end)
    panel = sf._load_panel(args.data_dir, source_start, args.end, symbols=symbols, include_minute=False)
    if panel.is_empty() or panel['date'].max() != args.end:
        raise ValueError('Missing requested current daily panel')
    minute = build_daily_features(args.data_dir, symbols, args.start, args.end)
    minute = minute.with_columns(pl.col('date').alias('_minute_latest_date'))
    panel = panel.drop([c for c in [*FEATURE_DEFAULTS, '_minute_latest_date'] if c in panel.columns]).join(
        minute, on=['symbol', 'date'], how='left')
    # Match the current feature producer's availability descriptor; the common
    # complete240 readiness gate is applied after targets/industry benchmarks.
    panel = panel.with_columns((pl.col('minute_session_coverage').fill_null(0.) >= .95)
        .cast(pl.Float64).alias('minute_data_available')).with_columns(
        (pl.col('minute_data_available')*.30+pl.col('moneyflow_data_available')*.25
         +pl.col('auction_data_available')*.15+pl.col('news_data_available')*.15+.15).alias('data_coverage_score'))
    base = sf._feature_frame(panel, include_targets=False).filter(pl.col('date') >= args.start)
    execution = json.loads((args.data_dir/'stockformer/daily_shadow/config.json').read_text())['protocol']['execution']
    raw, conflicts = training.load_paper_raw_prices(args.data_dir, symbols, start=source_start, end=args.end)
    current = training.paper_execution_targets(base, panel, raw, execution=execution)
    assert current['date'].max() == args.end
    combined = pl.concat([old.filter(pl.col('date') < args.start), current], how='diagonal_relaxed').sort('symbol', 'date')
    if combined.select(pl.struct('symbol', 'date').is_duplicated().any()).item():
        raise ValueError('Duplicate refreshed feature keys')
    historical_panel = pl.read_parquet(args.old_panel).filter(pl.col('date') < args.start)
    new_panel = pl.concat([historical_panel, panel.filter(pl.col('date') >= args.start)],
                          how='diagonal_relaxed').sort('symbol', 'date')
    if new_panel.select(pl.struct('symbol', 'date').is_duplicated().any()).item():
        raise ValueError('Duplicate refreshed panel keys')
    if before != source_versions(args.data_dir, source_start, args.end):
        raise ValueError('Source generation changed during read; inputs not published')
    # Prove that the original development region was not revised in this refresh.
    old_part = old.filter(pl.col('date') < date(2024, 12, 19)).sort('symbol', 'date')
    new_part = combined.filter(pl.col('date') < date(2024, 12, 19)).select(old.columns).sort('symbol', 'date')
    if not old_part.equals(new_part):
        raise ValueError('Original development observations changed')
    ready, coverage = sf.prepare_training_features(combined, 'minute_enhanced')
    combined.write_parquet(out/'features-paper-current.parquet')
    new_panel.write_parquet(out/'panel-current.parquet')
    minute.write_parquet(out/'recent-minute-recomputed.parquet')
    write(out/'raw-price-quarantine.json', conflicts.to_dicts())
    write(out/'input-manifest.json', {'refresh_start': args.start, 'as_of': args.end,
        'old_feature_end': old['date'].max(), 'new_feature_end': combined['date'].max(),
        'latest_complete_minute_date': ready['date'].max(), 'coverage': coverage,
        'source_versions': before, 'source_generation_unchanged_during_capture': True,
        'source_files_unchanged_by_worker': True, 'original_development_rows_unchanged': True,
        'minute_policy': sf.MINUTE_AVAILABILITY_POLICY, 'raw_price_quarantines': conflicts.height,
        'old_feature_sha256': pair.digest(args.old_features), 'old_panel_sha256': pair.digest(args.old_panel),
        'feature_sha256': pair.digest(out/'features-paper-current.parquet'),
        'panel_sha256': pair.digest(out/'panel-current.parquet'), 'runner_sha256': pair.digest(__file__),
        'target_version': training.PAPER_COST_TARGET_VERSION, 'labels_recomputed_before_minute_filter': True,
        'rows': combined.height, 'recent_rows': current.height})
    print(json.dumps({'stage': 'inputs_refreshed', 'as_of': str(args.end), 'ready_end': str(ready['date'].max()),
                      'recent_rows': current.height, 'coverage': coverage}, default=str))


if __name__ == '__main__':
    main()
