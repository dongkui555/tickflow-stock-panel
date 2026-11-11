import copy
import json
from datetime import date

import polars as pl
import pytest

from app.services.minute_repair import checked_additions, daily_reference
from app.services.minute_quality import filter_minute_sessions
from test_minute_repair import DAY, bars


def sample():
    ref = {'symbol': '920344.BJ', 'date': DAY, 'open': 10., 'high': 10.,
           'low': 10., 'close': 10., 'volume': 340., 'amount': 320000.}
    evidence = {
        'schema_version': 1, 'basis': 'bse_post_close_block_trade',
        'symbol': ref['symbol'], 'date': str(DAY),
        'daily_snapshot': {k: ref[k] for k in ('open', 'high', 'low', 'close', 'volume', 'amount')},
        'daily_volume_unit': 'lots',
        'daily_total': {'open': 10., 'high': 10., 'low': 10., 'close': 10.,
                        'volume_shares': 34000., 'amount_cny': 320000.},
        'blocks': [{'symbol': ref['symbol'], 'date': str(DAY), 'price': 8.,
                    'volume_shares': 10000., 'amount_cny': 80000.}],
        'sources': [{'provider': 'test', 'sha256': 'a' * 64}],
    }
    return ref, evidence


def write_evidence(root, evidence):
    path = root / 'minute_reference_evidence' / f'date={DAY}' / '920344.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence), encoding='utf-8')
    return path


def test_reference_without_evidence_unchanged(tmp_path):
    ref, _ = sample()
    frame = pl.DataFrame([ref])
    assert daily_reference(frame, data_dir=tmp_path) == daily_reference(frame)


def test_regular_reference_preserves_original_daily_and_research_contract(tmp_path):
    ref, evidence = sample()
    write_evidence(tmp_path, evidence)
    daily = pl.DataFrame([ref])
    before = daily.clone()
    frame = bars('920344.BJ')
    assert checked_additions(frame, frame.clear(), ref, DAY)[1] == 'daily_turnover_conflict'
    adjusted = daily_reference(daily, data_dir=tmp_path)['920344']
    assert adjusted['volume'] == 24000. and adjusted['amount'] == 240000.
    assert adjusted['_volume_scope'] == 'regular_session'
    assert checked_additions(frame, frame.clear(), adjusted, DAY)[1] == 'accepted'
    valid, issues = filter_minute_sessions(frame, daily, data_dir=tmp_path)
    assert valid.equals(frame) and not issues
    assert daily.equals(before)


@pytest.mark.parametrize('mutation', ['stale', 'wrong_symbol', 'wrong_date', 'bad_unit',
                                     'wrong_block_amount', 'too_many_shares', 'nan', 'bad_source'])
def test_invalid_evidence_fails_closed(tmp_path, mutation):
    ref, evidence = sample()
    evidence = copy.deepcopy(evidence)
    if mutation == 'stale':
        ref['amount'] += 1
    elif mutation == 'wrong_symbol':
        evidence['symbol'] = '000001.SZ'
    elif mutation == 'wrong_date':
        evidence['date'] = str(date(2025, 1, 2))
    elif mutation == 'bad_unit':
        evidence['daily_volume_unit'] = 'unknown'
    elif mutation == 'wrong_block_amount':
        evidence['blocks'][0]['amount_cny'] += 1
    elif mutation == 'too_many_shares':
        evidence['blocks'][0].update(volume_shares=40000., amount_cny=320000.)
    elif mutation == 'nan':
        evidence['daily_total']['volume_shares'] = float('nan')
    else:
        evidence['sources'] = []
    write_evidence(tmp_path, evidence)
    assert '920344' not in daily_reference(pl.DataFrame([ref]), data_dir=tmp_path)


def test_unrelated_stock_untouched_and_corrupt_json_rejected(tmp_path):
    ref, evidence = sample()
    path = write_evidence(tmp_path, evidence)
    other = ref | {'symbol': '000001.SZ'}
    path.write_text('{', encoding='utf-8')
    refs = daily_reference(pl.DataFrame([ref, other]), data_dir=tmp_path)
    assert set(refs) == {'000001'}
    assert refs['000001']['amount'] == other['amount']


def test_adjustment_does_not_hide_incorrect_minute_price(tmp_path):
    ref, evidence = sample()
    write_evidence(tmp_path, evidence)
    refs = daily_reference(pl.DataFrame([ref]), data_dir=tmp_path)
    bad = bars('920344.BJ').with_columns(pl.lit(9.).alias('low'))
    assert checked_additions(bad, bad.clear(), refs['920344'], DAY)[1] == 'daily_price_conflict'


def test_full_audit_uses_same_explicit_reference(tmp_path):
    from audit_minute_integrity import compare_daily, summarize_minutes
    ref, evidence = sample()
    write_evidence(tmp_path, evidence)
    summary = summarize_minutes(bars('920344.BJ'), DAY)
    compared, meta = compare_daily(summary, pl.DataFrame([ref]), data_dir=tmp_path)
    assert meta['regular_reference_adjusted'] == 1
    assert not compared['turnover_mismatch'].any()


def test_repair_keeps_daily_total_and_repeat_is_idempotent(tmp_path):
    from threading import RLock
    from types import SimpleNamespace
    from app.services.minute_repair import repair_minute_gaps
    ref, evidence = sample()
    write_evidence(tmp_path, evidence)
    path = tmp_path / 'kline_daily' / f'date={DAY}' / 'part.parquet'
    path.parent.mkdir(parents=True)
    pl.DataFrame([ref]).write_parquet(path)
    original = path.read_bytes()
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path), _write_lock=RLock(),
                           rebuild_views=lambda: None, clear_cache=lambda: None)
    source = SimpleNamespace(get_minute=lambda *a, **kw: bars('920344.BJ'))
    params = dict(start_date=DAY, end_date=DAY, provider_name='test', source_provider=source,
                  stock_days=[('920344', DAY)])
    result = repair_minute_gaps(repo, **params)
    repeated = repair_minute_gaps(repo, **params)
    assert result['added_rows'] == 240 and repeated['requests_planned'] == 0
    assert original == path.read_bytes()
