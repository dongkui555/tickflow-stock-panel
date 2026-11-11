from datetime import date

import polars as pl
import pytest

from app.services import stockformer_pit_inputs as pit


def events(known_on=date(2024, 3, 19)):
    return pl.DataFrame({'symbol': ['002089.SZ'], 'state': ['delisting_period'],
        'valid_from': [date(2024, 3, 26)], 'valid_to': [date(2024, 4, 17)],
        'known_on': [known_on], 'source': ['issuer_announcement_2024_026']})


def test_liquidation_exclusion_uses_announcement_and_exact_interval():
    panel = pl.DataFrame({'symbol': ['002089']*4,
        'date': [date(2024, 3, 25), date(2024, 3, 26), date(2024, 4, 17), date(2024, 4, 18)]})
    result = pit.attach_lifecycle_exclusions(panel, events())
    assert result['_pit_lifecycle_excluded'].to_list() == [False, True, True, False]
    assert result.height == panel.height
    late = pit.attach_lifecycle_exclusions(panel, events(date(2024, 4, 1)))
    assert late['_pit_lifecycle_excluded'].to_list() == [False, False, True, False]


def test_name_suffix_is_not_proof_of_liquidation_and_unknown_event_rejected():
    panel = pl.DataFrame({'symbol': ['002089'], 'date': [date(2024, 3, 26)], 'name': ['新海退']})
    assert not pit.attach_lifecycle_exclusions(panel)['_pit_lifecycle_excluded'].item()
    with pytest.raises(ValueError, match='lifecycle state'):
        pit.attach_lifecycle_exclusions(panel, events().with_columns(pl.lit('unknown').alias('state')))


def test_excluded_real_bars_remain_in_panel_but_never_enter_feature_context(monkeypatch):
    from test_stockformer_pit_inputs import industry, limits, names, panel, raw

    original = panel()
    liquidation = events().with_columns(pl.lit('002310').alias('symbol'),
        pl.lit(date(2024, 11, 21)).alias('valid_from'), pl.lit(date(2024, 11, 27)).alias('valid_to'))
    # Missing references on a confirmed liquidation period are explicit exclusions.
    fixed, report = pit.prepare_panel(original, raw(), names(), industry().filter(pl.col('symbol') == '002237'),
        limits().filter(pl.col('symbol') == '002237'), lifecycle_events=liquidation)
    assert fixed.height == original.height
    assert report['lifecycle_excluded_rows'] == 2
    assert not fixed.filter('_pit_lifecycle_excluded')['_pit_input_ready'].any()

    def generator(context, include_targets):
        assert set(context['symbol']) == {'002237'} and not include_targets
        return context.select('symbol', 'date', 'close')

    monkeypatch.setattr(pit.sf, '_feature_frame', generator)
    features = pit.build_features(fixed)
    assert features['symbol'].to_list() == ['002237']
    with pytest.raises(ValueError, match='rolling/industry context'):
        pit.build_features(fixed.with_columns(pl.lit(False).alias('_pit_lifecycle_excluded')))


def test_label_rebuild_keeps_next_open_costs_and_null_unmatured_rows():
    days = [date(2025, 1, d) for d in [2, 3, 6, 7, 8, 9, 10, 13]]
    panel = pl.DataFrame([{'symbol': symbol, 'date': day, 'open': 10., 'high': 10.,
        'low': 10., 'close': 10., 'volume': 10000., 'amount': 100000.,
        'minute_data_available': 1., 'minute_bar_count': 240., 'minute_session_coverage': 1.}
        for symbol in ['002237', '002238'] for day in days])
    names = pl.DataFrame({'symbol': ['002237', '002238'], 'name': ['甲', '乙'],
        'valid_from': [days[0]]*2, 'valid_to': [None, None], 'known_on': [days[0]]*2,
        'source': ['archive']*2})
    industries = names.drop('name').with_columns(pl.lit('gold').alias('industry'))
    limits = panel.select('symbol', 'date').with_columns(pl.lit(11.).alias('up_limit'),
        pl.lit(9.).alias('down_limit'))
    fixed, _ = pit.prepare_panel(panel, panel, names, industries, limits)
    execution = {'initial_cash': 10000., 'max_positions': 3, 'commission_pct': .0003,
        'stamp_tax_pct': .0005, 'slippage_bps': 5., 'minimum_commission': 5.}
    labels, _, report = pit.build_labels(fixed, panel, execution=execution)
    first = labels.filter(pl.col('date') == days[0]).row(0, named=True)
    assert first['label_end_5d'] == days[6]
    assert first['label_paper_qty'] == 300 and first['target_5d_rel_net'] < -.004
    assert labels.filter(pl.col('date') == days[-1])['target_5d_rel_net'].null_count() == 2
    assert report['minute_coverage']['minimum_minute_bars'] == 240
    assert report['conditional_labels_are_not_portfolio_returns']


def test_explicit_name_history_reason_requires_an_announcement():
    records = pl.DataFrame({'ts_code': ['002433.SZ', '002089.SZ', '000996.SZ'],
        'name': ['太安退', '新海退', '中期退'], 'start_date': ['20240614', '20240326', '20240606'],
        'end_date': [None, None, None], 'ann_date': ['20240606', None, '20240531'],
        'change_reason': ['退市整理期', '退市整理期', '其他']})
    accepted = pit.announced_liquidation_events(records)
    assert accepted['symbol'].to_list() == ['002433']
    assert accepted['known_on'].item() == date(2024, 6, 6)
