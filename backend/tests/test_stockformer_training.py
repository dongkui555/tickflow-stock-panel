import json
import threading
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from app.services import stockformer as sf
from app.services import stockformer_training as training
from app.services.minute_features import build_daily_features


def test_filtered_parquet_reader_limits_dates_and_symbols(tmp_path):
    root = tmp_path / "moneyflow"
    root.mkdir()
    pl.DataFrame({
        "symbol": ["000001", "000002", "000001"],
        "date": [date(2025, 1, 2), date(2025, 1, 2), date(2025, 1, 3)],
        "net_mf_amount": [10.0, 20.0, 30.0],
        "unused": ["a", "b", "c"],
    }).write_parquet(root / "part.parquet")

    result = sf._read_filtered_parquet_root(
        root,
        start=date(2025, 1, 3),
        end=date(2025, 1, 3),
        symbols=["000001.SZ"],
        columns=["symbol", "date", "net_mf_amount"],
    )

    assert result.select(["symbol", "date", "net_mf_amount"]).to_dicts() == [
        {"symbol": "000001", "date": date(2025, 1, 3), "net_mf_amount": 30.0},
    ]


def test_filtered_parquet_reader_skips_missing_date_partition(tmp_path):
    root = tmp_path / "auction"
    root.mkdir()
    pl.DataFrame({
        "symbol": ["000001.SZ"],
        "trade_date": [date(2025, 1, 3)],
        "amount": [1.0],
    }).write_parquet(root / "part-good.parquet")
    pl.DataFrame({"symbol": ["000002.SZ"], "amount": [2.0]}).write_parquet(
        root / "part-bad.parquet"
    )

    result = sf._read_filtered_parquet_root(
        root,
        start=date(2025, 1, 3),
        end=date(2025, 1, 3),
        symbols=["000001.SZ"],
        columns=["symbol", "trade_date", "amount"],
    )

    assert result.select(["symbol", "amount"]).to_dicts() == [
        {"symbol": "000001", "amount": 1.0},
    ]


def test_filtered_parquet_reader_tolerates_mixed_additive_schema(tmp_path):
    root = tmp_path / "auction"
    root.mkdir()
    pl.DataFrame({
        "symbol": ["000001.SZ"],
        "date": [date(2025, 1, 3)],
        "trade_date": [date(2025, 1, 3)],
        "market": ["open"],
        "amount": [1.0],
    }).write_parquet(root / "part-good.parquet")
    pl.DataFrame({
        "symbol": ["000002.SZ"],
        "date": [date(2025, 1, 3)],
        "market": ["open"],
        "amount": [2.0],
        "source": ["qmt"],
    }).write_parquet(root / "part-additive.parquet")

    result = sf._read_filtered_parquet_root(
        root,
        start=date(2025, 1, 3),
        end=date(2025, 1, 3),
        columns=["symbol", "date", "trade_date", "market", "amount"],
    )

    assert result.select(["symbol", "date", "trade_date", "market", "amount"]).to_dicts() == [
        {
            "symbol": "000002",
            "date": date(2025, 1, 3),
            "trade_date": None,
            "market": "open",
            "amount": 2.0,
        },
        {
            "symbol": "000001",
            "date": date(2025, 1, 3),
            "trade_date": date(2025, 1, 3),
            "market": "open",
            "amount": 1.0,
        },
    ]


def test_auction_features_do_not_double_count_identity_duplicate(tmp_path):
    root = tmp_path / "auction_open"
    root.mkdir()
    day = date(2025, 1, 3)
    pl.DataFrame({
        "symbol": ["920701", "920701.BJ"],
        "date": [day, day],
        "market": ["open", "open"],
        "open": [11.35, 11.35],
        "amount": [154519.0, 154519.0],
        "volume": [13614.0, 13614.0],
        "exchange": [None, "BJ"],
        "security_key": [None, "stock|BJ|920701"],
    }).write_parquet(root / "part.parquet")

    base = pl.DataFrame({
        "symbol": ["920701"],
        "date": [day],
        "amount": [100000.0],
        "volume": [1000.0],
    })
    result = sf._attach_auction_features(tmp_path, base, day, day)

    row = result.row(0, named=True)
    assert row["_auction_open_price"] == pytest.approx(11.35)
    assert row["_auction_amount"] == pytest.approx(154519.0)
    assert row["_auction_volume"] == pytest.approx(13614.0)


def test_repository_auction_writer_normalizes_symbol_aliases(tmp_path):
    from app.tickflow.repository import KlineRepository

    class Store:
        data_dir = tmp_path

    repo = object.__new__(KlineRepository)
    repo.store = Store()
    repo._write_lock = threading.RLock()

    day = date(2025, 1, 3)
    existing_dir = tmp_path / "auction_open" / "date=2025-01-03"
    existing_dir.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["920701.BJ"],
        "date": [day],
        "market": ["open"],
        "open": [11.35],
        "amount": [154519.0],
        "volume": [13614.0],
    }).write_parquet(existing_dir / "part.parquet")
    frame = pl.DataFrame({
        "symbol": ["920701"],
        "date": [day],
        "market": ["open"],
        "open": [11.35],
        "amount": [154519.0],
        "volume": [13614.0],
    })

    repo.append_auction_open(frame)

    saved = pl.read_parquet(tmp_path / "auction_open" / "date=2025-01-03" / "part.parquet")
    assert saved.height == 1
    assert saved["symbol"].to_list() == ["920701"]


def sample():
    rows = []
    rng = np.random.default_rng(18)
    for i in range(110):
        for s in range(8):
            x = rng.normal()
            row = dict(symbol=f'{s:06}', date=date(2025, 1, 1) + timedelta(days=i))
            row.update({c: x + rng.normal(scale=.1) for c in training.COMPACT_FEATURES})
            row['noisy'] = rng.normal()
            for h in sf.HORIZONS:
                row[f'target_{h}d'] = x * .01 + rng.normal(scale=.01)
                row[f'label_end_{h}d'] = row['date'] + timedelta(days=h + 1)
            rows.append(row)
    return pl.DataFrame(rows)


def test_scaler_is_fitted_once_and_inference_is_batch_independent():
    x = np.arange(100., dtype=float).reshape(-1, 1)
    w, scaler = sf._fit_ridge(x, x[:, 0] * .01, 1, clip_quantiles=(1, 99))
    assert scaler['clip_high'][0] == pytest.approx(98.01)
    assert scaler['clip_low'][0] == pytest.approx(.99)
    assert sf._predict(np.array([[1e9]]), w, scaler)[0] == pytest.approx(
        sf._predict(np.array([[98.01]]), w, scaler)[0])
    assert sf._predict(np.array([[10.]]), w, scaler)[0] == pytest.approx(
        sf._predict(np.array([[10.], [1e12]]), w, scaler)[0])
    frame = pl.DataFrame({'x': [0., 10., 1e9], 'target': [0., 1., 2.]})
    assert sf._prepare_xy(frame, 'target', ['x'])[0][-1, 0] == 1e9


def test_legacy_scaler_remains_readable():
    got = sf._predict(np.array([[3.]]), np.array([1., 2.]), {'mean': [1.], 'std': [2.]})
    assert got.tolist() == [3.]


def test_weights_never_depend_on_test_metrics():
    assert sf._active_horizon_weights({'oos_metrics': {'1d': {'rank_ic': 1}}}) == sf._active_horizon_weights(
        {'oos_metrics': {'10d': {'rank_ic': 1}}})
    assert sf._active_horizon_weights({'signal_weights': {'1d': .25, '3d': .75}}, [1, 3]) == {1: .25, 3: .75}


def test_explicit_empty_weights_keep_failed_model_inactive():
    assert sf._active_horizon_weights({'signal_weights': {}}, [1, 3]) == {}
    frame, weights = sf._add_adaptive_stockformer_score(
        pl.DataFrame({'pred_ret_1d': [.1, -.1]}), {'signal_weights': {}})
    assert weights == {}
    assert frame['stockformer_blend'].to_list() == [0., 0.]
    assert frame['stockformer_score'].to_list() == [50., 50.]


def test_frozen_weights_ignore_negative_holdout_and_role_adjustment():
    meta = {'signal_weights': {'1d': .25, '3d': .75},
            'horizon_role_weights': {'1d': .9, '3d': .1},
            'oos_metrics': {'1d': {'rank_ic': -1}, '3d': {'rank_ic': -1}}}
    assert sf._active_horizon_weights(meta, [1, 3]) == {1: .25, 3: .75}
    assert sf._active_horizon_weights({'holdout_comparison': {'1d': {'rank_ic': -1}}}) == sf._active_horizon_weights({})


def test_labels_use_next_global_open_and_purge_boundary():
    days = [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6), date(2025, 1, 7)]
    panel = pl.DataFrame([
        dict(symbol=s, date=d, open=p, high=p, low=p, close=p, volume=100., amount=p*100)
        for s in ['000001', '000002'] for d, p in zip(days, [10., 20., 22., 25.], strict=True)
        if not (s == '000002' and d == days[1])
    ])
    result = training.execution_targets(panel.select('symbol', 'date'), panel, (1,))
    one = result.filter((pl.col('symbol') == '000001') & (pl.col('date') == days[0])).row(0, named=True)
    assert one['target_1d'] == pytest.approx(.1)
    assert one['label_end_1d'] == days[2]
    assert result.filter((pl.col('symbol') == '000002') & (pl.col('date') == days[0]))['target_1d'].item() is None
    assert training.purged_slice(result, 1, days[0], days[2]).is_empty()


def test_future_changes_cannot_change_fitted_models_or_selection():
    frame = sample()
    validation = date(2025, 3, 1)
    test = date(2025, 3, 27)
    cols = [*training.COMPACT_FEATURES, 'noisy']
    first = training.fit_candidates(frame, cols, validation, test, 1.)
    future = frame.with_columns([
        pl.when(pl.col('date') >= test).then(pl.col(c) * -50).otherwise(pl.col(c)).alias(c)
        for c in ['noisy', *[f'target_{h}d' for h in sf.HORIZONS]]
    ])
    second = training.fit_candidates(future, cols, validation, test, 1.)
    assert first['models'] == second['models']
    assert first['validation'] == second['validation']
    assert first['signal_weights'] == second['signal_weights']
    assert first['oos_metrics'] != second['oos_metrics']
    for detail in first['validation']['horizons'].values():
        assert detail['max_train_label_end'] < validation.isoformat()
        assert detail['max_validation_label_end'] < test.isoformat()


def test_compact_family_is_not_filtered_by_sparse_full_feature():
    frame = sample().with_columns(
        pl.when(pl.col('date') >= date(2025, 3, 20))
        .then(pl.col('noisy'))
        .otherwise(None)
        .alias('fragile')
    )
    cols = [*training.COMPACT_FEATURES, 'fragile']
    result = training.fit_candidates(frame, cols, date(2025, 3, 1), date(2025, 3, 27), 1.)
    for detail in result['validation']['horizons'].values():
        assert detail['family'] != 'full'
        selected = next(candidate for candidate in detail['candidates']
                        if candidate['family'] == detail['family'] and candidate['alpha'] == detail['alpha'])
        assert 'fragile' not in ({
            'compact': training.COMPACT_FEATURES,
            'short_reversal': training.SHORT_REVERSAL_FEATURES,
            'trend': training.TREND_FEATURES,
            'price_volume': training.PRICE_VOLUME_FEATURES,
        }.get(detail['family'], ()))
        assert selected['coverage']['train_rows'] >= 40 * 8
        assert detail['train_rows'] >= 40 * 8


def test_daily_metrics_do_not_pool_dates():
    frame = pl.DataFrame({'date': [date(2025,1,2)]*3 + [date(2025,1,3)]*3,
                          'symbol': ['a','b','c']*2})
    m = training.metrics(frame, np.array([1.,2.,3.,100.,101.,102.]), np.array([3.,2.,1.,102.,101.,100.]))
    assert m['rank_ic'] == pytest.approx(-1)
    assert m['dates'] == 2


def test_nonfinite_labels_are_excluded():
    df = pl.DataFrame({'x': [1., 2., 3.], 'target': [float('inf'), float('nan'), 0.]})
    _, y, used = sf._prepare_xy(df, 'target', ['x'])
    assert y.tolist() == [0.]
    assert used.height == 1


def test_net_target_is_absolute_after_cost_and_keeps_its_level():
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(3)]
    panel = pl.DataFrame({
        'symbol': ['a', 'b'] * 3,
        'date': [day for day in dates for _ in range(2)],
        'industry': ['same', 'same'] * 3,
        'open': [10.0, 20.0, 11.0, 20.0, 12.0, 19.0],
        'high': [10.0, 20.0, 11.0, 20.0, 12.0, 19.0],
        'low': [10.0, 20.0, 11.0, 20.0, 12.0, 19.0],
        'close': [10.0, 20.0, 11.0, 20.0, 12.0, 19.0],
        'volume': [100.0] * 6,
    })
    targets = training.execution_targets(panel, panel, horizons=(1,))
    first = targets.filter(pl.col('date') == dates[0])
    assert training._target_column(targets, 1, 'net') == 'target_1d_net'
    assert training._target_column(targets, 1, 'industry_relative_net') == 'target_1d_rel_net'
    assert first['target_1d_net'].to_list() == pytest.approx([
        12.0 / 11.0 - 1.0 - 0.002,
        19.0 / 20.0 - 1.0 - 0.002,
    ])
    fitted = first.with_columns(training._fit_target_expr('target_1d_net', 'net').alias('fit'))
    assert fitted['fit'].to_list() == pytest.approx(first['target_1d_net'].to_list())


def test_industry_relative_target_is_cross_sectional_excess_without_cost_shift():
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(3)]
    panel = pl.DataFrame({
        'symbol': ['a', 'b'] * 3,
        'date': [day for day in dates for _ in range(2)],
        'industry': ['same', 'same'] * 3,
        'open': [10.0, 20.0, 11.0, 20.0, 12.0, 19.0],
        'high': [10.0, 20.0, 11.0, 20.0, 12.0, 19.0],
        'low': [10.0, 20.0, 11.0, 20.0, 12.0, 19.0],
        'close': [10.0, 20.0, 11.0, 20.0, 12.0, 19.0],
        'volume': [100.0] * 6,
    })
    targets = training.execution_targets(panel, panel, horizons=(1,))
    first = targets.filter(pl.col('date') == dates[0])
    raw = np.array([12.0 / 11.0 - 1.0, 19.0 / 20.0 - 1.0])
    expected = raw - raw.mean()
    assert first['target_1d_rel'].to_list() == pytest.approx(expected.tolist())
    assert training._target_column(targets, 1, 'industry_relative') == 'target_1d_rel'


def test_extreme_prediction_calibration_is_train_only_and_tail_aware():
    pred = np.linspace(-0.20, 0.20, 100)
    actual = np.where(pred >= 0.18, -0.03, np.where(pred <= -0.18, 0.02, pred * 0.1))
    calibration = sf._fit_extreme_prediction_calibration(pred, actual, min_tail_rows=5)
    assert calibration['enabled'] is True
    corrected = sf._apply_extreme_prediction_calibration(pred, calibration)
    assert corrected[-1] < 0
    assert corrected[0] >= 0
    assert corrected[50] == pytest.approx(pred[50])


def test_universe_does_not_use_future_liquidity():
    start = date(2025, 2, 1)
    frame = pl.DataFrame({'symbol': ['a','a','b','b'],
                          'date': [date(2025,1,1),date(2025,3,1)]*2,
                          'amount': [100., 0., 1., 1e20]})
    assert training.select_universe(frame, start, 1) == ['a']


def test_universe_falls_back_to_first_available_session_without_future_liquidity():
    start = date(2025, 1, 1)
    frame = pl.DataFrame({
        'symbol': ['b', 'a', 'c', 'b', 'a', 'c'],
        'date': [date(2025, 1, 2)] * 2 + [date(2025, 1, 3)] + [date(2025, 1, 4)] * 3,
        'amount': [2.0, 1.0, 1e20, 1.0, 1.0, 1.0],
    })
    assert training.select_universe(frame, start, 2) == ['a', 'b']


def test_proxy_cannot_pass_live_trading_gate():
    assert training.EVALUATION_LIMITATION
    assert sf._active_horizon_weights({}) == {1: .2, 3: .3, 5: .3, 10: .2}


def test_training_profiles_keep_minute_features_out_of_daily_baseline():
    assert "minute_intraday_return" not in sf.training_feature_columns("daily_baseline")
    assert "minute_intraday_return" in sf.training_feature_columns("minute_enhanced")
    assert "ret_5d" in sf.training_feature_columns("daily_baseline")


def test_prediction_base_restricts_panel_to_requested_universe(tmp_path, monkeypatch):
    calls = {}
    target = date(2025, 1, 2)
    frame = pl.DataFrame({
        "symbol": ["000001.SZ"],
        "date": [target],
        "open": [10.0], "high": [10.1], "low": [9.9], "close": [10.0],
        "volume": [1000.0], "amount": [10000.0], "turnover_rate": [1.0],
    })
    monkeypatch.setattr(sf, "_date_range_from_data", lambda _: (target, target))
    monkeypatch.setattr(sf, "_resolve_prediction_target_date", lambda *_: (target, {}))

    def load_panel(*args, **kwargs):
        calls.update(kwargs)
        return frame

    monkeypatch.setattr(sf, "_load_panel", load_panel)
    monkeypatch.setattr(sf, "_feature_frame", lambda panel, include_targets=False: panel)
    monkeypatch.setattr(sf, "_fill_prediction_feature_defaults", lambda panel, columns: panel)

    result, _, _ = sf._load_prediction_base(
        tmp_path,
        required_feature_columns=["close"],
        symbols=["000001.SZ"],
    )

    assert result.height == 1
    assert calls["symbols"] == ["000001.SZ"]


def test_minute_enhanced_profile_is_complete_by_default_with_explicit_legacy():
    frame = pl.DataFrame({
        "symbol": ["ready", "qmt_sparse", "missing", "partial"],
        "date": [date(2025, 1, 2)] * 4,
        "minute_data_available": [1.0, 1.0, 0.0, 1.0],
        "minute_bar_count": [240.0, 239.0, 0.0, 227.0],
        "minute_session_coverage": [1.0, 1.0, 0.0, 0.90],
        "minute_intraday_return": [0.01, 0.02, None, None],
    })
    filtered, report = sf.prepare_training_features(frame, "minute_enhanced")
    assert filtered["symbol"].to_list() == ["ready"]
    assert report["ready_rows"] == 1
    assert report["incomplete_rows"] == 2
    assert report["missing_rows"] == 1
    assert filtered["minute_intraday_return"].to_list() == [0.01]

    legacy, _ = sf.prepare_training_features(
        frame, "minute_enhanced", minimum_minute_bars=228, minimum_minute_coverage=.95,
    )
    assert legacy["symbol"].to_list() == ["ready", "qmt_sparse"]

    live_filtered, live_report = sf.prepare_training_features(
        frame,
        "minute_enhanced",
        minimum_minute_bars=sf.MINUTE_LIVE_MIN_VALID_BARS,
        minimum_minute_coverage=1.0,
    )
    assert live_filtered["symbol"].to_list() == ["ready"]
    assert live_report["minimum_minute_bars"] == 240
    assert live_report["minimum_minute_coverage"] == 1.0


def test_minute_features_match_exchange_qualified_symbols(tmp_path):
    day = date(2025, 1, 2)
    part = tmp_path / "kline_minute" / f"date={day.isoformat()}"
    part.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH"],
        "datetime": [
            datetime(2025, 1, 2, 9, 30),
            datetime(2025, 1, 2, 9, 31),
            datetime(2025, 1, 2, 9, 30),
            datetime(2025, 1, 2, 9, 31),
        ],
        "open": [10.0, 10.1, 8.0, 8.1],
        "high": [10.1, 10.2, 8.1, 8.2],
        "low": [9.9, 10.0, 7.9, 8.0],
        "close": [10.1, 10.2, 8.1, 8.2],
        "volume": [100.0, 120.0, 80.0, 90.0],
        "amount": [1000.0, 1200.0, 640.0, 738.0],
    }).write_parquet(part / "part.parquet")

    result = build_daily_features(tmp_path, ["000001", "600000"], day, day)

    assert sorted(result["symbol"].to_list()) == ["000001", "600000"]
    assert result.height == 2


def test_proxy_reports_execution_limitations(tmp_path, monkeypatch):
    rows = []
    for i in range(25):
        for s in range(3):
            rows.append(dict(symbol=f'{s:06}', date=date(2025, 1, 1)+timedelta(days=i),
                             pred_ret_3d=.01*s, target_3d=.01*s,
                             entry_score=90., risk_score=90., technical_score=90.,
                             fund_score=90., fundamental_score=90., sentiment_score=90.,
                             market_buy_threshold_add=0., market_position_multiplier=1., market_gate='open'))
    path = tmp_path / 'factor.parquet'
    pl.DataFrame(rows).write_parquet(path)
    monkeypatch.setattr(sf, '_model_meta', lambda *a: dict(factor_file=str(path), train_start='2024-01-01',
                                                        train_end='2024-12-01', oos_start='2025-01-01', oos_end='2025-01-25'))
    monkeypatch.setattr(sf, '_load_factor_feature_frame', lambda *a: pl.DataFrame())
    monkeypatch.setattr(sf, '_attach_instrument_names', lambda _, df: df)
    monkeypatch.setattr(sf, '_add_multifactor_scores', lambda df: df)
    monkeypatch.setattr(sf, '_attach_market_gate_for_oos', lambda _, df: df)
    monkeypatch.setattr(sf, '_external_feature_coverage', lambda _: {})
    for name in ['_entry_eligible_expr', '_entry_eligible_without_sector_expr', '_market_gate_oos_eligible_expr']:
        monkeypatch.setattr(sf, name, lambda df: pl.lit(True))
    result = sf.strict_oos_backtest(tmp_path, holding_days=3, max_positions=1)
    assert result['evaluation_kind'] == 'forward_label_proxy'
    assert result['evaluation_warning'] == training.EVALUATION_LIMITATION
    assert result['diagnosis']['passed'] is False
    assert result['diagnosis']['checks'][0]['passed'] is False


@pytest.mark.parametrize('paper_costs', [False, True])
def test_training_creates_research_model_without_changing_active(tmp_path, monkeypatch, paper_costs):
    rng = np.random.default_rng(77)
    dates = [date(2024, 10, 1)+timedelta(days=i) for i in range(260)]
    rows = []
    for s in range(4):
        prices = 10 * np.cumprod(1 + rng.normal(0., .008, len(dates)))
        for d, p in zip(dates, prices, strict=True):
            rows.append(dict(symbol=f'{s:06}', date=d, open=p, high=p*1.01, low=p*.99,
                             close=p, volume=1000., amount=p*1000., turnover_rate=1.))
    panel = pl.DataFrame(rows)
    monkeypatch.setattr(sf, '_date_range_from_data', lambda _: (dates[0], dates[-1]))
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return panel
    monkeypatch.setattr(sf, '_load_panel', load)
    execution = {'initial_cash': 10000., 'max_positions': 3, 'commission_pct': .0003,
                 'stamp_tax_pct': .0005, 'slippage_bps': 5., 'minimum_commission': 5.} if paper_costs else None
    if paper_costs:
        monkeypatch.setattr(training, 'load_paper_raw_prices', lambda *a, **k: (panel, pl.DataFrame()))
    monkeypatch.setattr(sf, 'predict', lambda *a, **k: {'rows': []})
    metric_modes = []
    original_metrics = training.metrics
    def recorded_metrics(*args, **kwargs):
        metric_modes.append(kwargs.get('target_mode'))
        return original_metrics(*args, **kwargs)
    monkeypatch.setattr(training, 'metrics', recorded_metrics)
    sf._latest_path(tmp_path).write_text('previous', encoding='utf8')
    result = sf.train_model(tmp_path, sf.StockFormerTrainConfig(start=dates[80], end=dates[-1],
                            target_mode='net' if paper_costs else 'industry_relative', execution_costs=execution), activate=False)
    assert 'error' not in result, result
    model = result['model']
    assert model['training_protocol'] == training.PROTOCOL
    assert model['validation']['start'] < model['train_end'] < model['oos_start']
    assert all(detail['refit']['max_label_end'] < model['oos_start']
               for detail in model['validation']['horizons'].values())
    assert model['promotion_status'] == 'research_only'
    assert metric_modes and set(metric_modes) == {model['target_mode']}
    manifest_path = tmp_path / model['training_input_manifest_file']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['price_panel']['rows'] == panel.height
    assert manifest['prepared_features_and_labels']['rows'] > 0
    assert manifest['split_date'] == model['oos_start']
    import hashlib
    assert model['training_input_manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert sf._latest_path(tmp_path).read_text() == 'previous'
    assert calls[0]['universe_as_of'] == dates[80]
    assert calls[0]['repair_adjusted'] is True
    assert (tmp_path / model['factor_file']).exists()
    payload = json.loads((tmp_path / model['model_file']).read_text(encoding='utf-8'))
    assert payload['execution_costs'] == execution
    assert payload['label_cost_version'] == (training.PAPER_COST_TARGET_VERSION if paper_costs else 'fixed_20bp_v1')
    assert set(payload['models']) == {str(h) for h in sf.HORIZONS}
    assert payload['training_input_manifest_sha256'] == model['training_input_manifest_sha256']
    for horizon_model in payload['models'].values():
        calibration = horizon_model['prediction_calibration']
        if calibration.get('calibration_start'):
            assert calibration['inner_train_max_label_end'] < calibration['calibration_start']
        assert (tmp_path / horizon_model['xgboost_file']).exists()
        assert (tmp_path / horizon_model['lightgbm_file']).exists()
    factor = pl.read_parquet(tmp_path / model['factor_file'])
    assert factor.height > 0
    assert factor['date'].max() < dates[-1]
    last = factor.filter(pl.col('date') == factor['date'].max())
    assert last.height == 4
    for h in sf.HORIZONS:
        evaluated = factor.filter(pl.col(f'pred_ret_{h}d').is_not_null())
        assert evaluated.height > 0
        assert evaluated[f'target_{h}d'].null_count() == 0
    assert sf._read_models(tmp_path)[0]['model_id'] == model['model_id']
