"""Revalue verified suspension marks without changing prices or trade ledgers."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import polars as pl
from app.services import stockformer as sf
from app.services import stockformer_training as training
from app.strategy import paper


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-study', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Suspension audit exists; preserve it')
    dates = ['2025-07-21', '2025-07-22']
    evidence = {'symbol': '002090.SZ', 'suspension_dates': dates, 'resume_date': '2025-07-23',
                'event': 'control_transfer_suspension',
                'announcement_date': '2025-07-21', 'resume_announcement_date': '2025-07-22',
                'verified_sources': [
                    'https://disc.static.szse.cn/disc/disk03/finalpage/2025-07-21/0a02ad72-8e3a-4b52-99c8-0f3993fda81f.PDF',
                    'https://disc.static.szse.cn/disc/disk03/finalpage/2025-07-22/cfc19c47-b94e-4b96-93ee-67d2dae55664.PDF'],
                'classification': 'verified_non_trading_sessions_not_missing_trades'}
    raw, conflicts = training.load_paper_raw_prices(args.reference/'data', ['002090'])
    assert conflicts.is_empty()
    assert raw.filter(pl.col('date').cast(pl.String).is_in(dates)).is_empty()
    last = raw.filter(pl.col('date') < date(2025, 7, 21)).sort('date').tail(1).row(0, named=True)
    assert str(last['date']) == '2025-07-18' and last['close'] > 0
    resumed = raw.filter(pl.col('date') == date(2025, 7, 23))
    assert resumed.height == 1 and resumed['volume'][0] > 0
    factor_path = args.reference/'data/adj_factor/all.parquet'
    factors = sf._normalize_symbol_column(pl.read_parquet(factor_path)).filter(
        (pl.col('symbol') == '002090') & (pl.col('trade_date') > last['date'])
        & (pl.col('trade_date') <= date(2025, 7, 22)))
    assert factors.filter(pl.col('ex_factor') != 1.).is_empty(), 'Corporate action requires separate valuation'
    report_path = args.prior_study/'report.json'
    protected = {str(report_path): digest(report_path)}
    results = {}
    for name in ('window-2/paper-raw_control', 'aggregate-paper-raw_control'):
        root = args.prior_study/name
        path = root/'replay-result.json'
        protected[str(path)] = digest(path)
        result = json.loads(path.read_text())
        nav, adjustments = [], []
        for row in result['daily']:
            value = row['settlement']['nav']['nav']
            if row['missing_position_marks']:
                assert row['missing_position_marks'] == ['002090'] and row['date'] in dates
                fills = [f for f in result['fills'] if f['date'] <= row['date']]
                positions, cash = paper.replay_positions(root, fills, account_id='tree_replay')
                assert abs(cash-row['settlement']['nav']['cash']) < .011
                pos = positions['002090']
                change = pos['qty']*(last['close']-pos['avg_cost'])
                adjusted = round(value+change, 2)
                adjustments.append({'date': row['date'], 'quantity': pos['qty'], 'cost_fallback': pos['avg_cost'],
                                    'last_traded_close': last['close'], 'last_traded_date': last['date'],
                                    'old_nav': value, 'marked_nav': adjusted, 'nav_change': round(adjusted-value, 2),
                                    'reason': 'verified_suspension_last_known_close_for_valuation_only'})
                value = adjusted
            nav.append({'date': row['date'], 'nav': value})
        assert len(adjustments) == 2
        values = [10000., *[r['nav'] for r in nav]]
        assert abs(values[-1]-result['final_nav']) < .011
        results[name] = {'adjustments': adjustments, 'old_drawdown_pct': result['max_drawdown_pct'],
                         'marked_drawdown_pct': paper.max_drawdown(values)*100, 'marked_nav': nav,
                         'final_nav_unchanged': True, 'trades_unchanged': True,
                         'complete_trades': result['completed_trades'], 'total_return_pct': result['total_return_pct'],
                         'unknown_missing_marks_after_classification': 0,
                         'original_passed_execution_gate': result['passed_execution_gate'],
                         'audit_changes_gate_or_activation': False}
    assert all(digest(Path(p)) == sha for p, sha in protected.items())
    report = {'evidence': evidence, 'prior_close': last, 'resume_observation': resumed.to_dicts(),
              'valuation_policy': 'last_known_traded_close_only_for_documented_suspension; no synthetic tradable OHLCV',
              'timing': 'resume announcement confirms interval; valuation price was already known July18',
              'results': results, 'protected_prior_results': protected, 'prior_results_unchanged': True,
              'activation_allowed': False}
    args.output.write_text(json.dumps(sf._json_safe(report), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    print(json.dumps({name: {k: v for k, v in value.items() if k != 'marked_nav'} for name, value in results.items()}, default=str))


if __name__ == '__main__':
    main()
