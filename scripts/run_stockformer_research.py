"""Train an immutable research candidate without replacing the active model."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--start', type=date.fromisoformat, required=True)
    parser.add_argument('--end', type=date.fromisoformat, required=True)
    parser.add_argument('--max-symbols', type=int, default=2000)
    args = parser.parse_args()
    if not (args.data_dir / 'kline_daily_enriched').is_dir():
        parser.error('Missing enriched dataset')
    # Never initialize a second production JobStore or application lifecycle.
    os.environ['DATA_DIR'] = tempfile.mkdtemp(prefix='stockformer-research-')
    os.environ.setdefault('POLARS_MAX_THREADS', '4')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
    from app.services.stockformer import StockFormerTrainConfig, train_model
    print('Training purged validation candidate; active model will not change.', flush=True)
    result = train_model(args.data_dir.resolve(), StockFormerTrainConfig(
        start=args.start, end=args.end, max_symbols=args.max_symbols), activate=False)
    if result.get('error'):
        print(result['error'], flush=True)
        return 1
    model = result['model']
    summary = {key: model.get(key) for key in (
        'model_id', 'training_protocol', 'train_start', 'train_end', 'oos_start', 'oos_end',
        'n_symbols', 'elapsed_ms', 'signal_weights', 'holdout_comparison', 'promotion_status',
    )}
    summary['validation_selection'] = {h: {k: v[k] for k in ('family', 'alpha', 'feature_count')}
                                       for h, v in model['validation']['horizons'].items()}
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
