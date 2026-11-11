from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import polars as pl

from stage_hf_minute_gaps import _stage_one


class StageHfTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'import_reports' / 'test'
        self.day = date(2024, 1, 3)
        timestamps = [datetime(2024, 1, 3, 9, 31) + timedelta(minutes=i) for i in range(120)]
        timestamps += [datetime(2024, 1, 3, 13, 1) + timedelta(minutes=i) for i in range(120)]
        self.raw = pl.DataFrame({
            'symbol': ['000001'] * 240, 'exchange': ['SZ'] * 240,
            'timestamp': timestamps, 'volume': [100.0] * 240,
            'turnover': [1000.0] * 240,
            **{c: [10.0] * 240 for c in ('open', 'high', 'low', 'close')},
        })
        self.reference = pl.DataFrame({
            'symbol': ['000001.SZ'], 'volume': [24000.0], 'amount': [240000.0],
            **{c: [10.0] for c in ('open', 'high', 'low', 'close')},
        })
        path = self.root / 'kline_daily' / f'date={self.day}' / 'part.parquet'
        path.parent.mkdir(parents=True)
        self.reference.write_parquet(path)

    def stage(self, raw=None):
        return _stage_one(self.root, self.output, 'SZ', '000001', {self.day},
                          self.raw if raw is None else raw)[0]

    def test_shares_to_lots_and_complete_session(self):
        result = self.stage()
        self.assertEqual(result['reason'], 'accepted')
        self.assertEqual(result['addable_rows'], 240)
        frame = pl.read_parquet(self.output / result['file'])
        self.assertEqual(frame['volume'].sum(), 240.0)

    def test_opening_snapshot_is_excluded_from_daily_ohlc(self):
        auction = pl.DataFrame({
            'symbol': ['000001'], 'exchange': ['SZ'],
            'timestamp': [datetime(2024, 1, 3, 9, 30)],
            'volume': [100.0], 'turnover': [1000.0],
            'open': [9.0], 'high': [99.0], 'low': [9.0], 'close': [9.0],
        })
        raw = pl.concat([auction, self.raw])
        reference = self.reference.with_columns([
            pl.lit(24100.0).alias('volume'),
            pl.lit(241000.0).alias('amount'),
        ])
        path = self.root / 'kline_daily' / f'date={self.day}' / 'part.parquet'
        reference.write_parquet(path)
        result = _stage_one(self.root, self.output, 'SZ', '000001', {self.day}, raw)[0]
        self.assertEqual(result['reason'], 'accepted')
        self.assertEqual(result['staged_rows'], 241)

    def test_partial_existing_stages_full_session(self):
        result = self.stage()
        frame = pl.read_parquet(self.output / result['file'])
        path = self.root / 'kline_minute' / f'date={self.day}' / 'part.parquet'
        path.parent.mkdir(parents=True)
        frame.head(239).write_parquet(path)
        result = self.stage()
        self.assertEqual(result['addable_rows'], 1)
        self.assertEqual(result['staged_rows'], 240)

    def test_identity_must_match_every_row(self):
        for column, value in [('symbol', '000002'), ('exchange', 'SH'), ('symbol', None)]:
            with self.subTest(column=column, value=value):
                raw = self.raw.with_columns(pl.lit(value, dtype=pl.String).alias(column))
                self.assertEqual(self.stage(raw)['reason'], 'source_identity_conflict')

    def test_duplicate_and_incomplete_rejected(self):
        self.assertEqual(self.stage(self.raw.head(239))['reason'], 'source_incomplete')
        self.assertEqual(self.stage(pl.concat([self.raw, self.raw.tail(1)]))['reason'], 'source_incomplete')

    def test_price_conflict_not_staged(self):
        raw = self.raw.with_columns([pl.lit(11.0).alias(c) for c in ('open','high','low','close')])
        raw = raw.with_columns(pl.lit(1100.0).alias('turnover'))
        self.assertEqual(self.stage(raw)['reason'], 'daily_price_conflict')
        self.assertFalse((self.output / 'minutes').exists())

    def test_existing_conflict_is_preserved(self):
        result = self.stage()
        frame = pl.read_parquet(self.output / result['file']).with_columns(pl.lit(9.0).alias('close'))
        path = self.root / 'kline_minute' / f'date={self.day}' / 'part.parquet'
        path.parent.mkdir(parents=True)
        frame.head(239).write_parquet(path)
        before = path.read_bytes()
        self.assertEqual(self.stage()['reason'], 'existing_values_conflict')
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
