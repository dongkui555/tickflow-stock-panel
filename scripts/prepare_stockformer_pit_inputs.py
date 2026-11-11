"""Stage corrected research inputs without modifying original data or models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

from app.services import stockformer_pit_inputs as pit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['panel', 'raw_prices', 'names', 'industries', 'limits', 'output']:
        parser.add_argument('--'+key.replace('_', '-'), required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing an existing output directory')
    inputs = {key: getattr(args, key) for key in ['panel', 'raw_prices', 'names', 'industries', 'limits']}
    corrected, report = pit.prepare_panel(*[pl.read_parquet(path) for path in inputs.values()])
    args.output.mkdir(parents=True)
    corrected.write_parquet(args.output/'panel.parquet')
    report.update(training_ready=False, activation_allowed=False, source_files={})
    for key, path in inputs.items():
        with path.open('rb') as handle:
            report['source_files'][key] = {'sha256': hashlib.file_digest(handle, 'sha256').hexdigest()}
    try:
        features = pit.build_features(corrected)
    except ValueError as exc:
        report['training_blocker'] = str(exc)
    else:
        features.write_parquet(args.output/'features.parquet')
        report.update(training_ready=True, feature_rows=features.height)
    report['source_version'] = pit.VERSION
    report['code_sha256'] = hashlib.sha256(Path(pit.__file__).read_bytes()).hexdigest()
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
