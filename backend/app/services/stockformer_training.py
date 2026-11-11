"""Chronological model selection, with one frozen historical holdout.

This is forward-return factor research, not a matching engine. Model choices
and preprocessing never use holdout observations. No production I/O here.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import timedelta
from pathlib import Path
import numpy as np
import polars as pl

from app.services import stockformer as sf

PROTOCOL = 'purged_expanding_walk_forward_factor_families_v7_gbdt_early_stop'
COMPACT_FEATURES = (
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d', 'ma20_gap',
    'vol_ratio_5d', 'intraday_strength', 'amplitude', 'rsi_proxy',
    'risk_20d', 'gap_open', 'amount_ratio_5d', 'price_position_20d', 'price_range_10d',
    # Daily aggregates generated from local minute bars. These stay compact
    # enough for the tabular model while exposing early/late-session behavior.
    'minute_session_coverage', 'minute_intraday_return', 'minute_early_return',
    'minute_open_strength', 'minute_intraday_volatility',
    'minute_early_amount_ratio', 'minute_late_amount_ratio',
    'minute_volume_concentration', 'minute_vwap_gap', 'minute_close_position',
)
SHORT_REVERSAL_FEATURES = (
    'ret_1d', 'ret_3d', 'gap_open', 'intraday_strength', 'amplitude',
    'vol_ratio_5d', 'amount_ratio_5d', 'risk_20d', 'price_range_10d',
)
TREND_FEATURES = (
    'ret_5d', 'ret_10d', 'ret_20d', 'ma20_gap', 'amount_ratio_5d',
    'risk_20d', 'price_position_20d', 'price_range_10d',
)
PRICE_VOLUME_FEATURES = (
    'intraday_strength', 'gap_open', 'amplitude', 'vol_ratio_5d',
    'amount_ratio_5d', 'risk_20d', 'price_range_10d',
)
EVALUATION_LIMITATION = (
    '收益标签估算, 不是成交回测: 未模拟实际持仓、T+1、涨跌停、整手、费用及滑点; '
    '重叠收益标签即使除以持有天数, 也不等于真实组合日收益。不得据此通过实盘门槛。'
)
DEFAULT_COST_AFTER_RETURN = 0.0020
COST_METRICS_VERSION = 'target_cost_once_v1'
PAPER_COST_TARGET_VERSION = 'paper_cash_return_cost_once_v1'


def _input_frame_fingerprint(frame):
    """Hash actual ordered inputs with bounded buffers, including types/nulls."""
    schema = [(name, str(dtype)) for name, dtype in frame.schema.items()]
    digest = hashlib.sha256(json.dumps(schema, ensure_ascii=False).encode('utf-8'))
    for batch in frame.iter_slices(8192):
        buffer = io.BytesIO()
        # IPC may serialize unused buffer padding/offsets differently after a
        # Parquet round trip. Fixed Parquet row groups encode logical values.
        batch.rechunk().write_parquet(buffer, compression='uncompressed',
                                      statistics=False, row_group_size=8192)
        digest.update(buffer.getbuffer())
    return {'sha256': digest.hexdigest(), 'rows': frame.height, 'schema': schema,
            'start': str(frame['date'].min()) if 'date' in frame.columns else None,
            'end': str(frame['date'].max()) if 'date' in frame.columns else None}


def input_manifest(panel, features):
    """Identify consumed prices/features/labels, not a historical PIT guarantee."""
    return {
        'version': 2,
        'algorithm': 'sha256_polars_parquet_ordered_batches_8192_v2',
        'polars_version': pl.__version__,
        'price_panel': _input_frame_fingerprint(panel),
        'prepared_features_and_labels': _input_frame_fingerprint(features),
        'code_sha256': {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(sf.__file__))
        },
        'limitation': '输入指纹用于重算比对，不替代原始来源快照或历史时点可得性证明',
    }


def select_universe(panel, before, limit):
    eligible = panel.filter(pl.col('date') < before)
    if eligible.is_empty():
        # A dataset may legitimately begin on the requested training date.
        # In that case there is no pre-start liquidity history to rank on. Use
        # only the first available training session and a deterministic symbol
        # order; never inspect later/OOS turnover to manufacture a universe.
        first_available = (
            panel.filter(pl.col('date') >= before)
            .select(pl.col('date').min())
            .item()
        )
        if first_available is None:
            raise ValueError('训练区间缺少可用于选池的历史行情')
        first_day = (
            panel.filter(pl.col('date') == first_available)
            .select('symbol')
            .unique()
            .sort('symbol')
            .head(limit)
        )
        if first_day.is_empty():
            raise ValueError('训练开始日缺少可用于选池的标的')
        return first_day['symbol'].to_list()
    return (eligible.group_by('symbol').agg(pl.len().alias('n'), pl.col('amount').mean().alias('liquidity'))
            .sort(['n', 'liquidity', 'symbol'], descending=[True, True, False])
            .head(limit)['symbol'].to_list())


def execution_targets(features, panel, horizons=sf.HORIZONS):
    # Global sessions prevent a suspended stock's next row from becoming tomorrow.
    calendar = panel.select('date').unique().sort('date')
    prices = panel.filter(
        pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in ('open', 'high', 'low', 'close')])
        & (pl.col('high') >= pl.max_horizontal('open', 'close', 'low'))
        & (pl.col('low') <= pl.min_horizontal('open', 'close', 'high'))
        & (pl.col('volume') > 0)
    ).select('symbol', 'date', 'open')
    if prices.select(pl.struct('symbol', 'date').is_duplicated().any()).item():
        raise ValueError('Duplicate price keys in training panel')
    out = features.drop([c for c in features.columns if c.startswith(('target_', 'label_end_'))])
    out = out.join(calendar.with_columns(pl.col('date').shift(-1).alias('_entry_date')), on='date')
    out = out.join(prices.rename({'date': '_entry_date', 'open': '_entry_open'}), on=['symbol', '_entry_date'], how='left')
    for h in horizons:
        end_col = f'label_end_{h}d'
        out = out.join(calendar.with_columns(pl.col('date').shift(-(h + 1)).alias(end_col)), on='date')
        out = out.join(prices.rename({'date': end_col, 'open': '_exit_open'}), on=['symbol', end_col], how='left')
        out = out.with_columns((pl.col('_exit_open') / pl.col('_entry_open') - 1).alias(f'target_{h}d')).drop('_exit_open')
    out = out.drop('_entry_date', '_entry_open').sort(['date', 'symbol'])
    # Train on relative strength versus the same industry's forward return;
    # retain raw targets for diagnostics and realistic trade replay.
    if 'industry' in out.columns:
        for h in horizons:
            raw = f'target_{h}d'
            if raw not in out.columns:
                continue
            industry_mean = (
                out.group_by(['date', 'industry'])
                .agg(pl.col(raw).mean().alias(f'_industry_{h}d'))
            )
            out = out.join(industry_mean, on=['date', 'industry'], how='left').with_columns(
                (pl.col(raw) - pl.col(f'_industry_{h}d')).alias(f'target_{h}d_rel')
            ).drop(f'_industry_{h}d')
    for h in horizons:
        raw = f'target_{h}d'
        if raw in out.columns:
            out = out.with_columns(
                (pl.col(raw) - DEFAULT_COST_AFTER_RETURN).alias(f'target_{h}d_net')
            )
            relative = f'target_{h}d_rel'
            if relative in out.columns:
                out = out.with_columns(
                    (pl.col(relative) - DEFAULT_COST_AFTER_RETURN).alias(f'{relative}_net')
                )
    return out


def paper_execution_targets(features, panel, raw_prices, *, execution, horizons=sf.HORIZONS):
    """Opt-in research labels with the paper ledger's exact fee arithmetic.

    Quantity uses the signal-day raw close and the frozen account's initial
    per-position budget, as in daily shadow sizing. Entry/exit opens are future
    label outcomes, never features. Relative targets subtract the existing gross
    industry benchmark AFTER costs, so industry demeaning cannot erase fees.
    Conditional fill labels are not a portfolio simulation: limits, competing
    orders, cash paths and delayed exits still require the paper matcher.
    """
    from app.strategy import paper

    required = ('initial_cash', 'max_positions', 'commission_pct', 'stamp_tax_pct', 'slippage_bps')
    if any(key not in execution or not np.isfinite(execution[key]) for key in required):
        raise ValueError('Incomplete or non-finite paper execution costs')
    if (execution['initial_cash'] <= 0 or execution['max_positions'] < 1
            or int(execution['max_positions']) != execution['max_positions']
            or any(execution[key] < 0 for key in required[2:])
            or execution['slippage_bps'] >= 10000):
        raise ValueError('Invalid paper execution costs')
    if execution.get('minimum_commission', paper.MIN_COMMISSION) != paper.MIN_COMMISSION:
        raise ValueError('Unsupported minimum commission; must match paper ledger')
    if not horizons or any(h < 1 or int(h) != h for h in horizons):
        raise ValueError('Paper target horizons must respect T+1')
    if raw_prices.select(pl.struct('symbol', 'date').is_duplicated().any()).item():
        raise ValueError('Duplicate raw price keys in paper training targets')
    price_columns = ('open', 'high', 'low', 'close')
    raw = raw_prices.filter(
        pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in price_columns])
        & (pl.col('high') >= pl.max_horizontal('open', 'close', 'low'))
        & (pl.col('low') <= pl.min_horizontal('open', 'close', 'high'))
        & (pl.col('volume') > 0)
    ).select('symbol', 'date', 'open', 'close')
    out = execution_targets(features, panel, horizons)
    calendar = panel.select('date').unique().sort('date')
    budget = execution['initial_cash'] / execution['max_positions']
    out = out.join(raw.select('symbol', 'date', pl.col('close').alias('_paper_ref')), on=['symbol', 'date'], how='left')
    out = out.with_columns(pl.col('_paper_ref').map_elements(
        lambda price: paper.qty_from_amount(max(0., budget - paper.MIN_COMMISSION),
                                           price * (1 + execution['slippage_bps'] / 10000)),
        return_dtype=pl.Int64,
    ).fill_null(0).alias('label_paper_qty'))
    out = out.join(calendar.with_columns(pl.col('date').shift(-1).alias('_paper_entry_date')), on='date')
    out = out.join(raw.select('symbol', pl.col('date').alias('_paper_entry_date'),
                             pl.col('open').alias('_paper_entry_open')), on=['symbol', '_paper_entry_date'], how='left')

    def cash_outcome(row):
        qty, entry, exit_price, gross = (row[k] for k in ('label_paper_qty', '_paper_entry_open', '_paper_exit_open', '_paper_gross'))
        if qty <= 0:
            return {'net': None, 'cost': None, 'status': 'one_lot_budget'}
        if any(value is None or not np.isfinite(value) or value <= 0 for value in (entry, exit_price)) or gross is None:
            return {'net': None, 'cost': None, 'status': 'missing_execution_prices'}
        raw_return = exit_price / entry - 1
        # Adjusted/raw divergence over the label interval requires share/cash
        # accounting; reject rather than label a split as a trading loss.
        if abs(raw_return - gross) > 0.0001:
            return {'net': None, 'cost': None, 'status': 'corporate_action_or_price_mismatch'}
        buy = paper.apply_slippage(entry, 'buy', execution['slippage_bps'])
        sell = paper.apply_slippage(exit_price, 'sell', execution['slippage_bps'])
        cost = qty * buy + paper.buy_fee(qty, buy, execution['commission_pct'])
        if cost > execution['initial_cash'] + 1e-6:
            return {'net': None, 'cost': None, 'status': 'account_cash'}
        proceeds = qty * sell - paper.sell_fee(qty, sell, execution['commission_pct'], execution['stamp_tax_pct'])
        net = (proceeds - cost) / cost
        return {'net': net, 'cost': raw_return - net, 'status': 'priced'}

    for horizon in horizons:
        end = f'label_end_{horizon}d'
        target = f'target_{horizon}d'
        out = out.join(raw.select('symbol', pl.col('date').alias(end), pl.col('open').alias('_paper_exit_open')),
                       on=['symbol', end], how='left')
        out = out.with_columns(pl.col(target).alias('_paper_gross'))
        outcome = pl.struct('label_paper_qty', '_paper_entry_open', '_paper_exit_open', '_paper_gross').map_elements(
            cash_outcome, return_dtype=pl.Struct({'net': pl.Float64, 'cost': pl.Float64, 'status': pl.String}),
        )
        out = out.with_columns(outcome.alias('_paper_result'))
        exprs = [pl.col('_paper_result').struct.field('net').alias(f'{target}_net'),
                 pl.col('_paper_result').struct.field('cost').alias(f'label_cost_{horizon}d'),
                 pl.col('_paper_result').struct.field('status').alias(f'label_paper_status_{horizon}d')]
        if f'{target}_rel' in out.columns:
            benchmark = pl.col(target) - pl.col(f'{target}_rel')
            exprs.append((pl.col('_paper_result').struct.field('net') - benchmark).alias(f'{target}_rel_net'))
        out = out.with_columns(exprs).drop('_paper_result', '_paper_gross', '_paper_exit_open')
    return out.drop('_paper_ref', '_paper_entry_date', '_paper_entry_open').with_columns(
        pl.lit(PAPER_COST_TARGET_VERSION).alias('label_cost_version'))


def load_paper_raw_prices(data_dir, symbols, *, start=None, end=None):
    """Canonical finalized raw prices; quarantine conflicting historical aliases."""
    from app.parquet import scan_daily_parquet

    scan = scan_daily_parquet(str(Path(data_dir) / 'kline_daily/**/*.parquet')).filter(sf._symbol_in_filter(symbols))
    if start is not None:
        scan = scan.filter(pl.col('date') >= start)
    if end is not None:
        scan = scan.filter(pl.col('date') <= end)
    raw = sf._normalize_symbol_column(scan.collect())
    if 'quote_ts' in raw.columns:
        raw = raw.filter(pl.col('quote_ts').is_null())
    fields = ['open', 'high', 'low', 'close', 'volume', 'amount']
    conflicts = raw.group_by('symbol', 'date').agg(pl.struct(fields).n_unique().alias('versions')).filter(pl.col('versions') > 1)
    safe = raw.join(conflicts.select('symbol', 'date'), on=['symbol', 'date'], how='anti')
    return safe.unique(['symbol', 'date']).sort('symbol', 'date'), conflicts


def purged_inner_split(frame, horizon):
    """Split chronological calibration without training labels crossing its start."""
    dates = sorted(frame['date'].unique().to_list())
    if len(dates) < 12:
        return frame, frame.clear(), None
    cutoff = dates[max(1, int(len(dates) * .80))]
    return (purged_slice(frame, horizon, frame['date'].min(), cutoff),
            frame.filter(pl.col('date') >= cutoff), cutoff)


def purged_slice(frame, horizon, start, upper=None):
    return frame.filter(
        (pl.col('date') >= start)
        & (pl.col('date') < upper if upper else pl.lit(True))
        & (pl.col(f'label_end_{horizon}d') < upper if upper else pl.col(f'label_end_{horizon}d').is_not_null())
    )


def metrics(frame, prediction, actual, *, target_mode='absolute'):
    valid = np.isfinite(prediction) & np.isfinite(actual)
    df = frame.select('date', 'symbol').with_columns(pl.Series('_p', prediction), pl.Series('_y', actual)).filter(pl.Series(valid))
    if df.is_empty():
        return {'rows': 0, 'dates': 0}
    ranked = df.sort(['date', '_p', 'symbol'], descending=[False, True, False]).with_columns(
        pl.int_range(0, pl.len()).over('date').alias('_rank'))
    daily = ranked.group_by('date').agg(
        pl.when(pl.len() >= 3).then(pl.corr('_p', '_y')).alias('ic'),
        pl.when(pl.len() >= 3).then(pl.corr(pl.col('_p').rank(), pl.col('_y').rank())).alias('rank_ic'),
        pl.col('_y').filter(pl.col('_rank') < (pl.len() * .2).ceil()).mean().alias('top_quantile_return'),
        pl.col('_y').filter(pl.col('_rank') >= pl.len() - (pl.len() * .2).ceil()).mean().alias('bottom_quantile_return'),
        pl.col('_y').filter(pl.col('_rank') < 2).mean().alias('top2_forward_return'),
        pl.col('_y').mean().alias('universe_forward_return'),
    ).sort('date')
    out = {c: sf._safe_float(daily[c].filter(daily[c].is_finite()).mean()) for c in daily.columns if c != 'date'}
    out.update(rows=df.height, dates=daily.height,
               direction_acc=float(np.mean((prediction[valid] > 0) == (actual[valid] > 0))),
               mse=float(np.mean((prediction[valid] - actual[valid])**2)))
    out['long_short_return'] = out['top_quantile_return'] - out['bottom_quantile_return']
    out['top2_excess'] = out['top2_forward_return'] - out['universe_forward_return']
    # Net labels already include the round-trip cost. This value gates every
    # validation fold, so deducting it twice can wrongly disable a horizon.
    additional_cost = 0.0 if target_mode in {'net', 'cost_after', 'industry_relative_net'} else DEFAULT_COST_AFTER_RETURN
    out['top2_after_cost_proxy'] = out['top2_forward_return'] - additional_cost
    return {key: round(value, 12) if isinstance(value, float) else value for key, value in out.items()}


def _fit_target_expr(target, target_mode):
    if target_mode in {'net', 'industry_relative_net', 'cost_after'}:
        return pl.col(target)
    return pl.col(target) - pl.col(target).mean().over('date')


def _family_arrays(parts, target, columns, target_mode='absolute'):
    """Prepare each feature family from its own available rows.

    The previous implementation filtered every family by the full feature set.
    A recently introduced field could therefore reduce even the compact OHLCV
    model to a handful of rows. Ridge is fitted to daily cross-sectional excess
    returns because the production use case is ranking stocks, not forecasting
    the market's absolute return.
    """
    arrays = {}
    for name, part in parts.items():
        fit_target = target
        prepared = part
        if name == 'train':
            fit_target = '_fit_target'
            prepared = part.with_columns(
                _fit_target_expr(target, target_mode).alias(fit_target)
            )
        arrays[name] = sf._prepare_xy(prepared, fit_target, columns)
    return arrays


def _candidate_is_reliable(arrays, parts):
    train_used = arrays['train'][2]
    validation_used = arrays['validation'][2]
    test_used = arrays['test'][2]
    expected_train_dates = max(1, parts['train']['date'].n_unique())
    expected_validation_dates = max(1, parts['validation']['date'].n_unique())
    expected_test_dates = max(1, parts['test']['date'].n_unique())
    return (
        train_used['date'].n_unique() >= 20
        and train_used['date'].n_unique() / expected_train_dates >= 0.60
        and validation_used['date'].n_unique() >= 8
        and validation_used['date'].n_unique() / expected_validation_dates >= 0.60
        and test_used['date'].n_unique() >= 8
        and test_used['date'].n_unique() / expected_test_dates >= 0.60
        and train_used['symbol'].n_unique() >= 3
        and validation_used['symbol'].n_unique() >= 3
    )


def _validation_folds(frame, validation_start, oos_start, horizon, max_folds=3):
    """Build chronological validation folds without borrowing future labels."""
    dates = (frame.filter((pl.col('date') >= validation_start) & (pl.col('date') < oos_start))
             .select('date').unique().sort('date')['date'].to_list())
    # A fold loses roughly ``horizon + 1`` observations to label purging.
    # Keep at least five fully observed validation dates in every fold.
    fold_count = min(max_folds, len(dates) // (horizon + 6))
    if fold_count < 2:
        return [(validation_start, oos_start)]
    chunks = np.array_split(np.asarray(dates, dtype=object), fold_count)
    return [
        (chunk[0], chunks[index + 1][0] if index + 1 < len(chunks) else oos_start)
        for index, chunk in enumerate(chunks) if len(chunk)
    ]


def _aggregate_fold_metrics(scores):
    keys = (
        'ic', 'rank_ic', 'top_quantile_return', 'bottom_quantile_return',
        'top2_forward_return', 'universe_forward_return', 'direction_acc',
        'mse', 'long_short_return', 'top2_excess', 'top2_after_cost_proxy',
    )
    out = {}
    for key in keys:
        values = [score.get(key) for score in scores]
        values = [float(value) for value in values if value is not None and np.isfinite(value)]
        out[key] = round(float(np.mean(values)), 12) if values else None
    rank_values = [float(score.get('rank_ic')) for score in scores
                   if score.get('rank_ic') is not None and np.isfinite(score.get('rank_ic'))]
    out.update(
        rows=sum(int(score.get('rows', 0)) for score in scores),
        dates=sum(int(score.get('dates', 0)) for score in scores),
        fold_count=len(scores),
        positive_rank_ic_folds=sum((score.get('rank_ic') or 0.) > 0 for score in scores),
        positive_long_short_folds=sum((score.get('long_short_return') or 0.) > 0 for score in scores),
        positive_top2_after_cost_folds=sum((score.get('top2_after_cost_proxy') or 0.) > 0 for score in scores),
        rank_ic_median=round(float(np.median(rank_values)), 12) if rank_values else None,
        rank_ic_worst=round(float(np.min(rank_values)), 12) if rank_values else None,
        folds=scores,
    )
    return out


def _target_column(frame, horizon, target_mode='absolute'):
    net = f'target_{horizon}d_net'
    relative = f'target_{horizon}d_rel'
    relative_net = f'{relative}_net'
    if target_mode == 'industry_relative_net' and relative_net in frame.columns:
        return relative_net
    if target_mode in {'net', 'cost_after', 'industry_relative_net'} and net in frame.columns:
        return net
    return relative if target_mode in {'industry_relative', 'industry_relative_net'} and relative in frame.columns else f'target_{horizon}d'


def rolling_ablation(frame, feature_columns, evaluation_end, *, target_mode='absolute'):
    """Compare feature families on several pre-holdout rolling windows.

    This is deliberately a diagnostic only: it never changes the selected
    production candidate and it never reads rows on/after the real holdout.
    """
    family_specs = {
        'daily_ohlcv': ('ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
                        'ma20_gap', 'vol_ratio_5d', 'intraday_strength', 'amplitude', 'rsi_proxy',
                        'risk_20d', 'gap_open', 'amount_ratio_5d', 'price_position_20d', 'price_range_10d'),
        'daily_plus_minute': COMPACT_FEATURES,
        'daily_plus_moneyflow': ('ret_1d', 'ret_3d', 'ret_5d', 'ret_20d', 'ma20_gap', 'vol_ratio_5d',
                                 'intraday_strength', 'risk_20d', 'amount_ratio_5d', 'price_position_20d',
                                 'moneyflow_strength_3d', 'moneyflow_strength_5d', 'moneyflow_acceleration_3d',
                                 'moneyflow_accumulation_score', 'moneyflow_absorption_efficiency'),
        'daily_plus_auction': ('ret_1d', 'ret_3d', 'ret_5d', 'ma20_gap', 'vol_ratio_5d', 'intraday_strength',
                               'risk_20d', 'amount_ratio_5d', 'auction_amount_ratio', 'auction_volume_ratio'),
        'daily_plus_news': ('ret_1d', 'ret_3d', 'ret_5d', 'ret_20d', 'ma20_gap', 'risk_20d',
                            'price_position_20d', 'news_sentiment_score', 'news_count_3d',
                            'overseas_news_sentiment_score', 'overseas_news_count_3d'),
        'all_features': feature_columns,
    }
    available_dates = sorted(frame.filter(pl.col('date') < evaluation_end)['date'].unique().to_list())
    result = {'protocol': 'rolling_ablation_pre_holdout_v1', 'evaluation_end': str(evaluation_end), 'windows': []}
    for train_months in (12, 15, 18):
        test_end = evaluation_end
        test_start = test_end - timedelta(days=92)
        train_start = test_start - timedelta(days=30 * train_months)
        window = {'train_months': train_months, 'test_months': 3, 'test_start': str(test_start),
                  'test_end': str(test_end), 'families': {}}
        for family, raw_columns in family_specs.items():
            columns = [c for c in raw_columns if c in feature_columns and c in frame.columns]
            horizon_scores = {}
            if not columns:
                continue
            for h in sf.HORIZONS:
                target = _target_column(frame, h, target_mode)
                train = purged_slice(frame, h, train_start, test_start)
                test = purged_slice(frame, h, test_start, test_end)
                if train.is_empty() or test.is_empty():
                    continue
                fit = train.with_columns(_fit_target_expr(target, target_mode).alias('_fit_target'))
                x, y, _ = sf._prepare_xy(fit, '_fit_target', columns)
                tx, ty, used = sf._prepare_xy(test, target, columns)
                if len(y) == 0 or len(ty) == 0:
                    continue
                weights, scaler = sf._fit_ridge(x, y, 1.0, len(columns), clip_quantiles=(1, 99))
                horizon_scores[f'{h}d'] = metrics(used, sf._predict(tx, weights, scaler), ty, target_mode=target_mode)
            if horizon_scores:
                window['families'][family] = horizon_scores
        result['windows'].append(window)
    return result


def fit_candidates(frame, feature_columns, validation_start, oos_start, alpha, *, target_mode='absolute'):
    models, oos_metrics, validation_metrics, comparisons, predictions = {}, {}, {}, {}, []
    fold_specs = {}
    family_specs = {
        'full': feature_columns,
        'compact': COMPACT_FEATURES,
        'short_reversal': SHORT_REVERSAL_FEATURES,
        'trend': TREND_FEATURES,
        'price_volume': PRICE_VOLUME_FEATURES,
    }
    families = {
        name: [column for column in columns if column in feature_columns]
        for name, columns in family_specs.items()
    }
    families = {name: columns for name, columns in families.items() if columns}
    if not families['compact'] or not np.isfinite(alpha) or alpha < 0:
        raise ValueError('Invalid model feature set or regularization')
    strengths = sorted(set([float(alpha), max(100., float(alpha))]))
    for h in sf.HORIZONS:
        validation_folds = _validation_folds(frame, validation_start, oos_start, h)
        fold_specs[f'{h}d'] = [
            {'start': str(start), 'end_exclusive': str(end)}
            for start, end in validation_folds
        ]
        target = _target_column(frame, h, target_mode)
        parts = {
            'train': purged_slice(frame, h, frame['date'].min(), validation_start),
            'validation': purged_slice(frame, h, validation_start, oos_start),
            'test': purged_slice(frame, h, oos_start),
        }
        candidates = []
        for family, columns in families.items():
            arrays = _family_arrays(parts, target, columns, target_mode)
            if not _candidate_is_reliable(arrays, parts):
                continue
            for strength in strengths:
                fold_scores = []
                for fold_start, fold_end in validation_folds:
                    fold_train = purged_slice(frame, h, frame['date'].min(), fold_start)
                    fold_train = fold_train.with_columns(
                        _fit_target_expr(target, target_mode).alias('_fit_target'))
                    x, y, _ = sf._prepare_xy(fold_train, '_fit_target', columns)
                    fold_validation = purged_slice(frame, h, fold_start, fold_end)
                    vx, vy, used = sf._prepare_xy(fold_validation, target, columns)
                    if used['date'].n_unique() < 5 or not len(y):
                        continue
                    weights, scaler = sf._fit_ridge(
                        x, y, strength, len(columns), clip_quantiles=(1, 99))
                    fold_scores.append(metrics(used, sf._predict(vx, weights, scaler), vy, target_mode=target_mode))
                if len(fold_scores) != len(validation_folds):
                    continue
                score = _aggregate_fold_metrics(fold_scores)
                candidates.append({'family': family, 'alpha': strength, 'feature_columns': columns,
                                   'validation': score,
                                   'coverage': {
                                       'train_rows': arrays['train'][2].height,
                                       'train_dates': arrays['train'][2]['date'].n_unique(),
                                       'train_symbols': arrays['train'][2]['symbol'].n_unique(),
                                       'validation_rows': arrays['validation'][2].height,
                                       'validation_dates': arrays['validation'][2]['date'].n_unique(),
                                   }})
        if not candidates:
            raise ValueError(f'{h}日没有满足训练覆盖门槛的特征族')
        selected = max(candidates, key=lambda c: (
            c['validation']['positive_rank_ic_folds'],
            c['validation']['positive_long_short_folds'],
            c['validation']['rank_ic_median'] if c['validation']['rank_ic_median'] is not None else -float('inf'),
            c['validation']['rank_ic_worst'] if c['validation']['rank_ic_worst'] is not None else -float('inf'),
            c['validation']['top2_excess'] if c['validation']['top2_excess'] is not None else -float('inf'),
            -c['validation']['mse'], -len(c['feature_columns']),
        ))
        baseline = candidates[0]
        validation_metrics[f'{h}d'] = selected['validation']
        test_scores = {}
        for name, choice in [('baseline', baseline), ('selected', selected)]:
            # Hyperparameters are selected only on the isolated validation
            # window. Afterwards refit on every label that ends before the
            # frozen holdout, so production weights are not stale by months.
            refit_part = purged_slice(frame, h, frame['date'].min(), oos_start)
            refit_frame = refit_part.with_columns(
                _fit_target_expr(target, target_mode).alias('_fit_target')
            )
            rx, ry, refit_used = sf._prepare_xy(
                refit_frame, '_fit_target', choice['feature_columns'])
            weights, scaler = sf._fit_ridge(
                rx, ry, choice['alpha'], len(choice['feature_columns']),
                clip_quantiles=(1, 99),
            )
            tx, ty, used = sf._prepare_xy(parts['test'], target, choice['feature_columns'])
            pred = sf._predict(tx, weights, scaler)
            test_scores[name] = metrics(used, pred, ty, target_mode=target_mode)
            if name == 'selected':
                models[str(h)] = {
                    'weights': weights.tolist(),
                    'scaler': scaler,
                    'feature_columns': choice['feature_columns'],
                }
                oos_metrics[f'{h}d'] = test_scores[name]
                # Missing future labels must not erase otherwise available forecasts.
                future = frame.filter(pl.col('date') >= oos_start).with_columns(pl.lit(0.).alias('_feature_mask'))
                fx, _, forecast_rows = sf._prepare_xy(future, '_feature_mask', choice['feature_columns'])
                forecast = sf._predict(fx, weights, scaler)
                predictions.append(
                    forecast_rows.select('symbol', 'date', target)
                    .rename({target: f'target_{h}d'})
                    .with_columns(pl.Series(f'pred_ret_{h}d', forecast))
                )
                selected_refit = {
                    'rows': refit_used.height,
                    'dates': refit_used['date'].n_unique(),
                    'symbols': refit_used['symbol'].n_unique(),
                    'max_label_end': str(refit_used[f'label_end_{h}d'].max()),
                }
        comparisons[f'{h}d'] = {
            'family': selected['family'], 'alpha': selected['alpha'], 'feature_count': len(selected['feature_columns']),
            'candidates': [{k: c[k] for k in ('family', 'alpha', 'validation', 'coverage')} for c in candidates],
            'train_rows': selected['coverage']['train_rows'],
            'train_dates': selected['coverage']['train_dates'],
            'train_symbols': selected['coverage']['train_symbols'],
            'validation_rows': selected['coverage']['validation_rows'],
            'purged_train_rows': frame.filter(pl.col('date') < validation_start).height - parts['train'].height,
            'max_train_label_end': str(_family_arrays(parts, target, selected['feature_columns'], target_mode)['train'][2][f'label_end_{h}d'].max()),
            'max_validation_label_end': str(_family_arrays(parts, target, selected['feature_columns'], target_mode)['validation'][2][f'label_end_{h}d'].max()),
            'refit': selected_refit,
            'test': test_scores,
        }
    # Every horizon must clear the net-return gate in every rolling validation
    # window before it can receive a live signal weight.  A majority rule is
    # too permissive here: one losing regime is enough to invalidate a frozen
    # horizon even when its aggregate IC remains positive.
    qualities = {
        h: max(0., validation_metrics[f'{h}d'].get('rank_ic') or 0.)
        if validation_metrics[f'{h}d']['positive_rank_ic_folds'] >= validation_metrics[f'{h}d']['fold_count']
        and validation_metrics[f'{h}d']['positive_long_short_folds'] >= validation_metrics[f'{h}d']['fold_count']
        and validation_metrics[f'{h}d']['positive_top2_after_cost_folds'] >= validation_metrics[f'{h}d']['fold_count']
        and (validation_metrics[f'{h}d'].get('long_short_return') or 0.) > 0
        and (validation_metrics[f'{h}d'].get('top2_after_cost_proxy') or 0.) > 0
        else 0.
        for h in sf.HORIZONS
    }
    total = sum(qualities.values())
    weights = {f'{h}d': v / total for h, v in qualities.items() if v > 0} if total else {}
    # Test diagnostics are stored separately and never influence frozen selection.
    validation = {
        'start': str(validation_start), 'end_exclusive': str(oos_start),
        'selection_metric': 'purged_walk_forward_majority_then_median_rank_ic',
        'folds': fold_specs,
        'metrics': validation_metrics,
        'has_positive_rank_ic': total > 0,
        'horizons': {h: {k: v for k, v in d.items() if k != 'test'} for h, d in comparisons.items()},
    }
    return {'models': models, 'oos_metrics': oos_metrics, 'prediction_frames': predictions,
            'validation': validation, 'signal_weights': weights,
            'holdout_comparison': {h: d['test'] for h, d in comparisons.items()}}
