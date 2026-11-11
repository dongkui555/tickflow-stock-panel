"""Offline five-day tree/calibration validation using the real paper ledger.

Research workers only. Historical clocks and price readers are scoped to the
single-process replay and restored in finally; never call from a running app.
"""
from __future__ import annotations

import bisect
import gc
import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from app.market_time import CN_TZ
from app.services import stockformer as sf
from app.services import stockformer_daily_shadow as shadow
from app.services import stockformer_training as training
from app.services.stockformer_paper import _daily_observations
from app.strategy import paper

_REPLAY_LOCK = threading.Lock()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sf._json_safe(value), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def make_windows(calendar, holdout_start, *, count=3, window_sessions=63, train_sessions=252):
    dates = sorted(set(calendar))
    cutoff = bisect.bisect_left(dates, holdout_start)
    stop = cutoff - 6  # All five-day labels settle strictly before old holdout.
    first = stop - count * window_sessions
    if first < train_sessions:
        raise ValueError('Insufficient history for complete rolling training and validation windows')
    result = []
    for index in range(count):
        start = first + index * window_sessions
        end = start + window_sessions
        result.append({'window': index+1, 'train_start': dates[start-train_sessions],
                       'start': dates[start], 'end': dates[end-1],
                       'signal_dates': dates[start:end], 'replay_dates': dates[start:end+6]})
    return result


def training_parts(features, start, test_start):
    train = training.purged_slice(features, 5, start, test_start)
    pre_calibration, calibration, calibration_cut = training.purged_inner_split(train, 5)
    core, tuning, tuning_cut = training.purged_inner_split(pre_calibration, 5)
    if tuning_cut is None or calibration_cut is None or core.is_empty() or tuning.is_empty() or calibration.is_empty():
        raise ValueError('Insufficient purged training/tuning/calibration history')
    assert core['label_end_5d'].max() < tuning_cut
    assert tuning['label_end_5d'].max() < calibration_cut
    assert calibration['label_end_5d'].max() < test_start
    return train, core, tuning, calibration, tuning_cut, calibration_cut


def fit_window(features, columns, window, output, *, emit=None):
    """Tune only before the window; calibration is disjoint from early stopping."""
    emit = emit or (lambda *a: None)
    features, _ = sf.prepare_training_features(features, 'minute_enhanced')
    train, core, tune, cal, tune_cut, cal_cut = training_parts(features, window['train_start'], window['start'])
    target = 'target_5d_rel_net'
    x, y, core_used = sf._prepare_xy(core, target, columns)
    vx, vy, tune_used = sf._prepare_xy(tune, target, columns)
    if not len(y) or not len(vy):
        raise ValueError('Empty paper-cost training or tuning labels')
    emit('early_stopping', window['window'])
    txgb = sf._fit_xgboost(x, y, eval_x=vx, eval_y=vy)
    tlgb = sf._fit_lightgbm(x, y, eval_x=vx, eval_y=vy)
    px, py = sf._predict_xgboost(txgb, vx), sf._predict_lightgbm(tlgb, vx)
    candidates = []
    for weight in np.arange(0., 1.01, .1):
        score = training.metrics(tune_used, weight*px + (1-weight)*py, vy, target_mode='industry_relative_net')
        candidates.append((score.get('rank_ic') or -999., score.get('long_short_return') or -999., float(weight)))
    blend = max(candidates)[2]
    xbest, lbest = getattr(txgb, 'best_iteration', None), getattr(tlgb, 'best_iteration_', None)
    xr = int(xbest)+1 if xbest is not None else 600
    lr = int(lbest) if lbest else 600
    rounds = max(40, min(600, round((xr+lr)/2)))
    del txgb, tlgb, x, y, vx, vy
    gc.collect()
    emit('calibration_model', window['window'])
    pre_cal = training.purged_slice(train, 5, train['date'].min(), cal_cut)
    cx, cy, pre_used = sf._prepare_xy(pre_cal, target, columns)
    ex, ey, cal_used = sf._prepare_xy(cal, target, columns)
    cxgb = sf._fit_xgboost(cx, cy, n_estimators=rounds)
    clgb = sf._fit_lightgbm(cx, cy, n_estimators=rounds)
    raw_cal = blend*sf._predict_xgboost(cxgb, ex) + (1-blend)*sf._predict_lightgbm(clgb, ex)
    calibration = sf._fit_extreme_prediction_calibration(raw_cal, ey)
    calibration.update(split_policy='separate_tuning_then_purged_calibration_v1',
                       calibration_start=str(cal_cut), inner_train_max_label_end=str(pre_used['label_end_5d'].max()))
    assert pre_used['label_end_5d'].max() < cal_cut
    del cxgb, clgb, cx, cy, ex, ey
    gc.collect()
    emit('window_refit', window['window'])
    fx, fy, final_used = sf._prepare_xy(train, target, columns)
    final_xgb = sf._fit_xgboost(fx, fy, n_estimators=rounds)
    final_lgb = sf._fit_lightgbm(fx, fy, n_estimators=rounds)
    # Missing future labels never erase otherwise valid prediction rows.
    future = features.filter(pl.col('date').is_in(window['signal_dates']))
    strict, coverage = sf.prepare_training_features(future, 'minute_enhanced')
    predict_frame = strict.with_columns(pl.lit(0.).alias('_predict_mask'))
    qx, _, used = sf._prepare_xy(predict_frame, '_predict_mask', columns)
    raw = (blend*sf._predict_xgboost(final_xgb, qx) + (1-blend)*sf._predict_lightgbm(final_lgb, qx)
           if used.height else np.empty(0))
    corrected = sf._apply_extreme_prediction_calibration(raw, calibration)
    factor = used.select('symbol', 'date', pl.col(target).alias('target_5d')).with_columns(
        pl.Series('pred_ret_raw_5d', raw), pl.Series('pred_ret_5d', corrected))
    output.mkdir(parents=True, exist_ok=True)
    final_xgb.save_model(str(output/'xgboost_5d.json'))
    final_lgb.booster_.save_model(str(output/'lightgbm_5d.txt'))
    factor.write_parquet(output/'predictions.parquet')
    specification = {'window': window, 'model_kind': sf.GBDT_ENSEMBLE_MODEL_KIND,
                     'horizon': 5, 'feature_columns': columns, 'n_estimators': rounds,
                     'weights': {'xgboost': blend, 'lightgbm': 1-blend},
                     'prediction_calibration': calibration,
                     'cost_target_version': training.PAPER_COST_TARGET_VERSION,
                     'minute_availability_policy': sf.MINUTE_AVAILABILITY_POLICY,
                     'core_max_label_end': core_used['label_end_5d'].max(), 'tuning_start': tune_cut,
                     'tuning_max_label_end': tune_used['label_end_5d'].max(), 'calibration_start': cal_cut,
                     'calibration_max_label_end': cal_used['label_end_5d'].max(),
                     'refit_max_label_end': final_used['label_end_5d'].max(),
                     'train_rows': final_used.height, 'prediction_rows': factor.height, 'live_coverage': coverage,
                     'activation_allowed': False, 'old_holdout_used': False}
    _write(output/'model.json', specification)
    emit('window_predict_done', window['window'])
    del final_xgb, final_lgb, fx, fy, qx
    gc.collect()
    return factor, specification


@contextmanager
def _paper_environment(raw_prices, factors, limit_map=None, cumulative_factors=None):
    """Worker-local adapters; retain the ledger's order, fee and T+1 logic."""
    required = ('open', 'high', 'low', 'close')
    valid = raw_prices.filter(pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in required])
                              & (pl.col('volume') > 0)
                              & (pl.col('high') >= pl.max_horizontal('open', 'close', 'low'))
                              & (pl.col('low') <= pl.min_horizontal('open', 'close', 'high')))
    bars = {(r['symbol'], str(r['date'])): {'open': float(r['open']), 'close': float(r['close'])}
            for r in valid.select('symbol', 'date', 'open', 'close').to_dicts()}
    history = {}
    for group in valid.sort('symbol', 'date').partition_by('symbol', maintain_order=True):
        history[str(group['symbol'][0])] = ([str(d) for d in group['date']], group['close'].to_list())
    factor_map = {}
    if factors is not None and cumulative_factors is not None:
        raise ValueError('Supply event factors or explicit cumulative factors, never both')
    if factors is not None and not factors.is_empty():
        for row in factors.to_dicts():
            factor_map[(row['symbol'], str(row['trade_date']))] = float(row['ex_factor'])
    cumulative_map = {}
    if cumulative_factors is not None:
        required_factors = {'symbol', 'trade_date', 'adj_factor'}
        if not required_factors <= set(cumulative_factors.columns):
            raise ValueError('Cumulative replay factors require explicit adj_factor column')
        if cumulative_factors.select(pl.struct('symbol', 'trade_date').is_duplicated().any()).item():
            raise ValueError('Duplicate cumulative replay factors')
        if cumulative_factors.filter(~(pl.col('adj_factor').is_finite()
                                      & (pl.col('adj_factor') > 0)).fill_null(False)).height:
            raise ValueError('Invalid cumulative replay factor')
        points = cumulative_factors.sort('symbol', 'trade_date').with_columns(
            pl.col('adj_factor').shift(1).over('symbol').alias('_previous_factor'))
        cumulative_map = {(row['symbol'], str(row['trade_date'])):
                          (row['adj_factor'], row['_previous_factor']) for row in points.to_dicts()}

    def factor_on(_root, symbol, _asset, day):
        if cumulative_factors is None:
            return factor_map.get((symbol, day))
        point = cumulative_map.get((symbol, day))
        if point is None or point[1] is None:
            raise ValueError(f'Missing cumulative factor history for held stock: {symbol} {day}')
        if point[0] != point[1]:
            # A price adjustment can include cash dividends. Its ratio is not
            # evidence for additional shares; preserve failure for adjudication.
            raise ValueError(f'Corporate action requires verified share/cash accounting: {symbol} {day}')
        return None
    clock = {'day': None, 'counter': 0}

    def now():
        clock['counter'] += 1
        return datetime.combine(clock['day'], time(19), tzinfo=CN_TZ) + timedelta(microseconds=clock['counter'])

    def previous(_root, symbol, _asset, day):
        dates, prices = history.get(symbol, ([], []))
        index = bisect.bisect_left(dates, day)-1
        return prices[index] if index >= 0 else None

    replacements = {('paper', 'cn_now'): now, ('paper', 'cn_today'): lambda: clock['day'],
                    ('paper', 'read_daily_bar'): lambda _r, symbol, _a, day: bars.get((symbol, day)),
                    ('paper', '_prev_close'): previous,
                    ('paper', '_factor_on'): factor_on,
                    ('paper', '_index_close'): lambda *a: None,
                    ('shadow', 'cn_now'): now}
    if limit_map is not None:
        def authoritative_limits(_previous, symbol, asset_type):
            if asset_type != 'stock' or (symbol, clock['day']) not in limit_map:
                raise ValueError('Missing authoritative execution limit for replay order')
            return limit_map[(symbol, clock['day'])]

        replacements[('paper', 'limit_prices')] = authoritative_limits
    modules = {'paper': paper, 'shadow': shadow}
    originals = {(name, key): getattr(modules[name], key) for name, key in replacements}
    with _REPLAY_LOCK:
        try:
            for (name, key), fn in replacements.items():
                setattr(modules[name], key, fn)
            yield clock, bars
        finally:
            for (name, key), fn in originals.items():
                setattr(modules[name], key, fn)


def replay_paper(data_dir, raw_prices, signal, calendar, execution, *, factors=None,
                 ranking_column='net_margin_score', limit_references=None, cumulative_factors=None):
    """Sequential isolated replay; does not create backdated live shadow data."""
    data_dir = Path(data_dir)
    if ranking_column not in {'net_margin_score', 'pred_ret_raw_5d'}:
        raise ValueError('Unsupported ranking_column')
    if not signal.is_empty() and (ranking_column not in signal.columns or not signal.select(
            pl.col(ranking_column).is_finite().fill_null(False).all()).item()):
        raise ValueError('Missing or invalid ranking_column')
    if data_dir.exists() and any(data_dir.iterdir()):
        raise ValueError('Refusing non-empty paper replay directory')
    if not calendar:
        raise ValueError('Empty replay calendar')
    limit_map = None
    if limit_references is not None:
        from app.services.stockformer_pit_inputs import execution_limit_map

        limit_map = execution_limit_map(raw_prices, limit_references, calendar)
    data_dir.mkdir(parents=True, exist_ok=True)
    all_dates = sorted(set(raw_prices['date'].to_list()) | set(calendar))
    for day in all_dates:
        (data_dir/'kline_daily'/f'date={day}').mkdir(parents=True, exist_ok=True)
    protocol_id = hashlib.sha256(json.dumps(execution, sort_keys=True).encode()).hexdigest()
    cfg = {'account_id': 'tree_replay', 'protocol_id': protocol_id, 'protocol': {'execution': execution}}
    daily, missing_marks = [], 0
    with _paper_environment(raw_prices, factors, limit_map, cumulative_factors) as (clock, bars):
        clock['day'] = calendar[0]
        shadow._account(data_dir, cfg)
        for day in calendar:
            clock.update(day=day, counter=0)
            settled = paper.settle_day(data_dir, str(day), cfg['account_id'])
            today = signal.filter(pl.col('date') == day) if not signal.is_empty() else signal
            observation, _ = _daily_observations(today, available_dates=[day], pred_col='pred_ret_5d',
                                                max_positions=execution['max_positions'], forecast_ceiling=None)
            if today.is_empty():
                selected = []
            else:
                selected = today.filter(pl.col('_forecast_pass') & pl.col('_entry_rules_pass') & pl.col('_market_gate_pass')).sort(
                    [ranking_column, 'symbol'], descending=[True, False]).head(execution['max_positions']).to_dicts()
            snapshot = {'id': f'replay-{day}', 'as_of': str(day), 'eligible_for_forward': True, 'selected': selected,
                        'record_kind': 'retrospective_not_forward'}
            execution_result = shadow._orders(data_dir, cfg, snapshot, paper.cn_now())
            positions = paper.load_positions(data_dir, cfg['account_id'])
            missing = [symbol for symbol, position in positions.items() if position['qty'] > 0 and (symbol, str(day)) not in bars]
            missing_marks += len(missing)
            daily.append({'date': str(day), 'observation': observation[0], 'settlement': settled,
                          'execution': execution_result, 'missing_position_marks': missing})
        fills = paper.load_fills(data_dir, cfg['account_id'])
        orders = paper.load_orders(data_dir, cfg['account_id'])
        positions = paper.load_positions(data_dir, cfg['account_id'])
        trades = paper.round_trips(data_dir, cfg['account_id'])
        nav = paper.load_nav(data_dir, cfg['account_id'])
    capital = execution['initial_cash']
    values = [capital, *[float(row['nav']) for row in nav]]
    peak, drawdown = capital, 0.
    for value in values:
        peak = max(peak, value)
        drawdown = max(drawdown, (peak-value)/peak)
    complete = not any(p['qty'] > 0 for p in positions.values()) and not any(o['status'] == 'pending' for o in orders)
    result = {'replay_kind': 'retrospective_not_forward', 'ranking_column': ranking_column,
              'price_limit_mode': 'authoritative_day' if limit_map is not None else 'legacy_board_rule',
              'corporate_action_mode': ('explicit_cumulative_no_event_held_exposure_only'
                                        if cumulative_factors is not None else 'legacy_event_factors'),
              'final_nav': values[-1],
              'total_return_pct': (values[-1]/capital-1)*100, 'max_drawdown_pct': drawdown*100,
              'completed_trades': len(trades), 'fills': fills, 'orders': orders, 'trades': trades,
              'daily': daily, 'settled': complete, 'missing_position_marks': missing_marks,
              'candidate_count': sum(row['observation']['selected_count'] for row in daily),
              'passed_execution_gate': complete and missing_marks == 0 and len(trades) >= 3
                                       and values[-1] > capital and drawdown <= .10,
              'activation_allowed': False}
    for fill in fills:
        if fill.get('kind', 'fill') == 'fill':
            assert fill['qty'] > 0 and fill['qty'] % 100 == 0
    for trade in trades:
        assert str(trade['close_date']) > str(trade['open_date'])
    _write(data_dir/'replay-result.json', result)
    return result
