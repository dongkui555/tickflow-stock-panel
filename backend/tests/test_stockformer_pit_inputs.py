from datetime import date

import polars as pl
import pytest

from app.services import stockformer_pit_inputs as pit


def panel():
    return pl.DataFrame({'symbol': ['002310', '002310', '002237'],
        'date': [date(2024, 11, 21), date(2024, 11, 27), date(2024, 11, 21)],
        'open': [1., 1., 1.], 'high': [1., 1., 1.], 'low': [1., 1., 1.],
        'close': [1., 1., 1.], 'industry': ['未来行业']*3, 'name': ['未来名称']*3})


def names():
    return pl.DataFrame({'ts_code': ['002310.SZ', '002310.SZ', '002237.SZ'],
        'name': ['*ST东园', '东方新能', '恒邦股份'],
        'start_date': ['20240430', '20260114', '20080520'],
        'end_date': ['20250528', None, None],
        'ann_date': ['20240427', '20251216', '20080520']})


def industry():
    return pl.DataFrame({'symbol': ['002310', '002237'], 'industry': ['园林工程', '黄金'],
        'valid_from': [date(2020, 1, 1)]*2, 'valid_to': [None, None],
        'known_on': [date(2020, 1, 1)]*2, 'source': ['dated_archive']*2})


def raw():
    return panel().select('symbol', 'date').with_columns(
        pl.Series('open', [2.63, 2.09, 10.]), pl.Series('high', [2.63, 2.09, 11.]),
        pl.Series('low', [2.51, 2.09, 9.]), pl.Series('close', [2.56, 2.09, 10.]),
        pl.lit(10000.).alias('volume'))


def limits():
    return panel().select('symbol', 'date').with_columns(
        pl.Series('up_limit', [2.63, 2.31, 11.]), pl.Series('down_limit', [2.38, 2.09, 9.]))


def test_historical_name_and_limit_flags_use_known_day_raw_prices():
    fixed, report = pit.prepare_panel(panel(), raw(), names(), industry(), limits())
    st = fixed.filter(pl.col('symbol') == '002310').sort('date')
    assert st['name'].to_list() == ['*ST东园']*2
    assert st['industry'].to_list() == ['园林工程']*2
    assert st['signal_broken_limit_up'].to_list() == [True, False]
    assert st['signal_limit_down'].to_list() == [False, True]
    assert fixed.filter(pl.col('symbol') == '002237')['signal_broken_limit_up'].item()
    assert report['eligible_rows'] == 1 and report['st_excluded_rows'] == 2


def test_missing_industry_or_limits_are_unknown_and_block_training():
    historical = industry().with_columns(pl.lit(None, dtype=pl.Date).alias('known_on'))
    fixed, report = pit.prepare_panel(panel(), raw(), names(), historical, limits().head(1))
    assert report['eligible_rows'] == 0 and report['industry_unknown_rows'] == 3
    assert fixed['industry'].null_count() == 3
    assert fixed['signal_limit_down'].null_count() == 2
    with pytest.raises(ValueError, match='No eligible point-in-time'):
        pit.build_features(fixed)


def test_future_and_unannounced_records_never_backfill():
    late = names().with_columns(pl.lit('20250101').alias('ann_date'))
    fixed, _ = pit.prepare_panel(panel(), raw(), late, industry(), limits())
    assert fixed['name'].null_count() == 3
    late_industry = industry().with_columns(pl.lit(date(2026, 10, 21)).alias('known_on'))
    fixed, _ = pit.prepare_panel(panel(), raw(), names(), late_industry, limits())
    assert fixed['industry'].null_count() == 3


def test_conflicting_identity_is_not_arbitrarily_selected():
    conflict = pl.concat([names(), names().head(1).with_columns(pl.lit('普通股票').alias('name'))])
    fixed, report = pit.prepare_panel(panel(), raw(), conflict, industry(), limits())
    assert fixed.filter(pl.col('symbol') == '002310')['name'].null_count() == 2
    assert report['name_conflict_rows'] == 2


def test_duplicate_price_or_limit_keys_fail_closed():
    with pytest.raises(ValueError, match='Duplicate'):
        pit.prepare_panel(panel(), pl.concat([raw(), raw().head(1)]), names(), industry(), limits())
    with pytest.raises(ValueError, match='Duplicate'):
        pit.prepare_panel(panel(), raw(), names(), industry(), pl.concat([limits(), limits().head(1)]))


def test_zero_limit_is_not_no_limit_and_float32_prices_match_cents():
    authoritative = limits().with_columns(pl.when(pl.col('symbol') == '002237')
        .then(0.).otherwise(pl.col('down_limit')).alias('down_limit'))
    prices = raw().with_columns(pl.col('open', 'high', 'low', 'close').cast(pl.Float32))
    fixed, _ = pit.prepare_panel(panel(), prices, names(), industry(), authoritative)
    assert fixed.filter(pl.col('symbol') == '002237')['signal_limit_up'].item() is None
    assert fixed.filter(pl.col('date') == date(2024, 11, 27))['signal_limit_down'].item()


def test_dated_industry_snapshot_cannot_apply_before_capture():
    snapshot = pl.DataFrame({'symbol': ['002237.SZ'], '所属同花顺行业': ['有色金属-贵金属']})
    result = pit.industry_snapshot(snapshot, date(2026, 10, 21), source='ths_snapshot')
    assert result['known_on'].item() == date(2026, 10, 21)
    fixed, _ = pit.prepare_panel(panel(), raw(), names(), result, limits())
    assert fixed['industry'].null_count() == 3


def test_historical_gate_does_not_overwrite_st_with_current_name(tmp_path, monkeypatch):
    from test_stockformer_raw_score_research import build, gate_inputs

    from app.services import stockformer as sf

    features, prediction = gate_inputs(monkeypatch)
    features = features.with_columns(pl.Series('name', ['*ST历史', '正常甲', '正常乙']),
        pl.lit(True).alias('_pit_name_known'), pl.lit(True).alias('_pit_input_ready'))
    monkeypatch.setattr(sf, '_attach_instrument_names', lambda *a: (_ for _ in ()).throw(AssertionError('current names used')))
    monkeypatch.setattr(sf, '_filter_special_treatment', lambda f: f.filter(~sf._special_treatment_expr()))
    result = build(tmp_path, features, prediction, historical_identity=True, score_source='raw')
    assert set(result['symbol']) == {'000002', '000003'}
    with pytest.raises(ValueError, match='historical identity'):
        build(tmp_path, features.drop('_pit_name_known'), prediction, historical_identity=True)


def test_authoritative_st_limit_blocks_buy_and_restores_paper_matcher(tmp_path):
    from test_stockformer_tree_validation import DAYS, EXECUTION, raw_prices, signal

    from app.services import stockformer_tree_validation as validation
    from app.strategy import paper

    prices = raw_prices().with_columns([pl.when(pl.col('date') == DAYS[1]).then(10.5)
        .otherwise(pl.col(c)).alias(c) for c in ['open', 'high', 'low', 'close']])
    refs = prices.select('symbol', 'date').with_columns(pl.lit(10.5).alias('up_limit'),
                                                     pl.lit(9.5).alias('down_limit'))
    original = paper.limit_prices
    result = validation.replay_paper(tmp_path/'strict', prices, signal(), DAYS[:7], EXECUTION,
                                    limit_references=refs)
    assert result['fills'] == [] and result['price_limit_mode'] == 'authoritative_day'
    assert any('涨停' in str(order.get('reason')) for order in result['orders'])
    assert paper.limit_prices is original


def test_missing_execution_limit_rejected_before_account_created(tmp_path):
    from test_stockformer_tree_validation import DAYS, EXECUTION, raw_prices, signal

    from app.services import stockformer_tree_validation as validation

    refs = raw_prices().select('symbol', 'date').head(1).with_columns(
        pl.lit(11.).alias('up_limit'), pl.lit(9.).alias('down_limit'))
    with pytest.raises(ValueError, match='Missing authoritative'):
        validation.replay_paper(tmp_path/'missing', raw_prices(), signal(), DAYS[:7], EXECUTION,
                                limit_references=refs)
    assert not (tmp_path/'missing').exists()


def test_authoritative_limit_keeps_t1_five_sessions_and_costs(tmp_path):
    from test_stockformer_tree_validation import DAYS, EXECUTION, raw_prices, signal

    from app.services import stockformer_tree_validation as validation

    prices = raw_prices()
    refs = prices.select('symbol', 'date').with_columns(pl.lit(11.).alias('up_limit'),
                                                     pl.lit(9.).alias('down_limit'))
    result = validation.replay_paper(tmp_path/'strict', prices, signal(), DAYS[:7], EXECUTION,
                                    limit_references=refs)
    assert result['completed_trades'] == 1
    assert result['fills'][0]['date'] == str(DAYS[1])
    assert result['fills'][1]['date'] == str(DAYS[6])
    assert result['fills'][0]['fee'] == 5.
    assert result['total_return_pct'] == pytest.approx(-.145)


def test_incomplete_industry_context_is_rejected_even_when_current_row_known():
    complete, _ = pit.prepare_panel(panel(), raw(), names(), industry(), limits())
    with pytest.raises(ValueError, match='rolling/industry context'):
        pit.build_features(complete.with_columns(pl.when(pl.col('symbol') == '002310')
            .then(False).otherwise(pl.col('_pit_industry_known')).alias('_pit_industry_known')))


def test_authoritative_down_limit_prevents_fabricated_exit(tmp_path):
    from test_stockformer_tree_validation import DAYS, EXECUTION, raw_prices, signal

    from app.services import stockformer_tree_validation as validation

    prices = raw_prices().with_columns([pl.when(pl.col('date') == DAYS[6]).then(9.5)
        .otherwise(pl.col(c)).alias(c) for c in ['open', 'high', 'low', 'close']])
    refs = prices.select('symbol', 'date').with_columns(pl.lit(10.5).alias('up_limit'),
                                                     pl.lit(9.5).alias('down_limit'))
    result = validation.replay_paper(tmp_path/'strict', prices, signal(), DAYS[:7], EXECUTION,
                                    limit_references=refs)
    assert len(result['fills']) == 1 and result['fills'][0]['side'] == 'buy'
    assert result['completed_trades'] == 0 and not result['settled']
    assert any('跌停' in str(order.get('reason')) for order in result['orders'])


def test_daily_reference_has_exact_date_validity_and_distinct_provenance():
    observed = pl.DataFrame({'ts_code': ['002237.SZ'], 'trade_date': ['20241121'],
                             'name': ['恒邦股份'], 'industry': ['黄金']})
    name = pit.daily_reference_history(observed, 'name')
    sector = pit.daily_reference_history(observed, 'industry')
    fixed, report = pit.prepare_panel(panel(), raw(), name, sector, limits())
    assert report['eligible_rows'] == 1
    assert fixed.filter(pl.col('symbol') == '002310')['name'].null_count() == 2
    assert fixed.filter(pl.col('symbol') == '002237')['industry'].item() == '黄金'
    assert sector['source'].item() == 'tushare.bak_basic:historical_daily_snapshot'


def test_streak_is_rebuilt_and_unknown_day_does_not_bridge_a_run():
    days = [date(2024, 11, day) for day in [21, 22, 25]]
    base = pl.DataFrame({'symbol': ['002237']*3, 'date': days,
        'open': [11., 12.1, 13.31], 'high': [11., 12.1, 13.31], 'low': [11., 12.1, 13.31],
        'close': [11., 12.1, 13.31], 'volume': [10000.]*3, 'consecutive_limit_ups': [999]*3})
    refs = base.select('symbol', 'date', pl.col('close').alias('up_limit')).with_columns(
        pl.lit(9.).alias('down_limit'))
    complete, _ = pit.prepare_panel(base, base, names(), industry(), refs)
    assert complete['consecutive_limit_ups'].to_list() == [1, 2, 3]
    broken, _ = pit.prepare_panel(base, base, names(), industry(), refs.filter(pl.col('date') != days[1]))
    assert broken['consecutive_limit_ups'].to_list() == [1, None, 1]
