"""Coverage auditing must use the same daily authority as minute import."""
import hashlib
import json
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import polars as pl

import audit_market_minute_coverage as audit


class CoverageDailyAuthorityTests(unittest.TestCase):
    def audit_rows(self, symbol, records, *, expected=1, valid=1, excluded=0):
        day = date(2026, 10, 23)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            daily_path = root / 'kline_daily/date=2026-10-23/part.parquet'
            minute_path = root / 'kline_minute/date=2026-10-23/part.parquet'
            daily_path.parent.mkdir(parents=True)
            minute_path.parent.mkdir(parents=True)
            pl.DataFrame(records).write_parquet(daily_path)
            stamps = [datetime(2026, 10, 23, 9, 31) + timedelta(minutes=i) for i in range(120)]
            stamps += [datetime(2026, 10, 23, 13, 1) + timedelta(minutes=i) for i in range(120)]
            pl.DataFrame({
                'symbol': [symbol] * 240, 'datetime': stamps,
                **{field: [10.] * 240 for field in ('open', 'high', 'low', 'close')},
                'volume': [1.] * 240, 'amount': [1000.] * 240,
            }).write_parquet(minute_path)
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (daily_path, minute_path)}
            output = root / 'audit'
            with patch.object(audit, 'parse_args', return_value=type('Args', (), {
                'data_dir': root, 'start': day, 'end': day, 'output_dir': output,
                'markets': ['BJ', 'SH', 'SZ'],
            })()), patch.object(audit, '_online_trade_dates', return_value=[]):
                audit.main()
            report = json.loads((output / 'report.json').read_text())
            summary = report['summary']
            self.assertEqual(sum(row['expected_active_stock_days'] for row in summary), expected)
            self.assertEqual(sum(row['complete_valid'] for row in summary), valid)
            self.assertEqual(sum(row['complete_conflict'] for row in summary), expected - valid)
            self.assertEqual(sum(row['zero_trade_excluded'] for row in summary), excluded)
            self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in before})

    @staticmethod
    def daily(symbol, **extra):
        return {'symbol': symbol, 'date': date(2026, 10, 23),
                **{field: 10. for field in ('open', 'high', 'low', 'close')},
                'volume': 24000., 'amount': 240000., **extra}

    def test_final_bare_history_beats_stale_qualified_snapshot_in_all_markets(self):
        for symbol in ['000001.SZ', '600000.SH', '920701.BJ']:
            final = self.daily(symbol[:6], quote_ts=None)
            stale = self.daily(symbol, quote_ts=1792742400000, close=9.5, volume=20000., amount=190000.)
            for records in ([final, stale], [stale, final]):
                with self.subTest(symbol=symbol, reversed=records[0] is stale):
                    self.audit_rows(symbol, records)

    def test_without_snapshot_marker_retains_qualified_alias_priority(self):
        self.audit_rows('000001.SZ', [self.daily('000001', close=99.), self.daily('000001.SZ')])

    def test_snapshot_is_used_when_no_batch_history_exists(self):
        self.audit_rows('000001.SZ', [self.daily('000001.SZ', quote_ts=1792742400000)])

    def test_real_price_conflict_remains_a_conflict(self):
        self.audit_rows('000001.SZ', [self.daily('000001', quote_ts=None, close=11., high=11.),
                                    self.daily('000001.SZ', quote_ts=1792742400000)], valid=0)

    def test_inactive_final_history_does_not_inherit_stale_positive_turnover(self):
        self.audit_rows('000001.SZ', [self.daily('000001', quote_ts=None, volume=0., amount=0.),
                                    self.daily('000001.SZ', quote_ts=1792742400000)],
                        expected=0, valid=0)  # A market without active stocks has no summary row.


if __name__ == '__main__':
    unittest.main()
