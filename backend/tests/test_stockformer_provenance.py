import io
from datetime import date

import polars as pl

from app.services import stockformer_training as training


def test_training_manifest_identifies_the_actual_prices_features_and_labels():
    panel = pl.DataFrame({'symbol': ['000001'], 'date': [date(2026, 9, 1)], 'close': [10.]})
    features = panel.with_columns(pl.lit(.01).alias('ret_1d'), pl.lit(.02).alias('target_5d'))
    manifest = training.input_manifest(panel, features)
    assert manifest['version'] == 2
    assert manifest == training.input_manifest(panel, features)
    assert manifest['price_panel']['rows'] == 1
    assert manifest['prepared_features_and_labels']['rows'] == 1
    assert all(len(value) == 64 for value in manifest['code_sha256'].values())
    changed = training.input_manifest(panel.with_columns(pl.lit(11.).alias('close')), features)
    assert changed['price_panel']['sha256'] != manifest['price_panel']['sha256']
    assert changed['prepared_features_and_labels'] == manifest['prepared_features_and_labels']
    changed_label = training.input_manifest(panel, features.with_columns(pl.lit(.03).alias('target_5d')))
    assert changed_label['prepared_features_and_labels']['sha256'] != manifest['prepared_features_and_labels']['sha256']


def test_training_fingerprint_includes_nulls_schema_and_input_order():
    panel = pl.DataFrame({'symbol': ['000001', '600000'], 'date': [date(2026, 9, 1)] * 2,
                          'close': [10., None]})
    baseline = training.input_manifest(panel, panel)['price_panel']['sha256']
    for changed in (panel.reverse(), panel.fill_null(0), panel.with_columns(pl.col('close').cast(pl.Float32))):
        assert training.input_manifest(changed, changed)['price_panel']['sha256'] != baseline


def test_input_fingerprint_is_independent_of_internal_chunk_layout():
    values = pl.DataFrame({'symbol': ['000001'] * 8200, 'close': [None if i % 3 else float(i) for i in range(8200)]})
    chunked = pl.concat([values.head(13), values.slice(13)], rechunk=False)
    assert training.input_manifest(values, values) == training.input_manifest(chunked, chunked)


def test_input_fingerprint_survives_parquet_roundtrip_with_nulls_and_timestamps():
    values = pl.DataFrame({'symbol': ['000001', None, '600000'] * 3000,
        'score': [None, float('nan'), .12345678901234567] * 3000,
        'date': [date(2026, 9, 1)] * 9000,
        'timestamp': pl.Series(range(9000)).cast(pl.Datetime('ns'))})
    buffer = io.BytesIO()
    values.write_parquet(buffer)
    restored = pl.read_parquet(buffer)
    assert values.equals(restored)
    assert training.input_manifest(values, values) == training.input_manifest(restored, restored)
