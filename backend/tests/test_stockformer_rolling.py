from datetime import date, timedelta

import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_portfolio as portfolio
from app.services.stockformer_rolling import (
    REGIME_POLICIES,
    _regime_policy_expr,
    _rolling_feature_columns,
    _select_regime_policy,
    activation_decision,
    build_rolling_windows,
)


def test_gated_signal_keeps_industry_metadata(monkeypatch, tmp_path):
    factor = pl.DataFrame({
        "symbol": ["000001.SZ"],
        "date": [date(2026, 1, 2)],
        "pred_ret_5d": [0.01],
    })
    feature_frame = pl.DataFrame({
        "symbol": ["000001.SZ"],
        "date": [date(2026, 1, 2)],
        "amount": [1000000.0],
        "industry": ["银行"],
    })

    monkeypatch.setattr(sf, "_add_adaptive_stockformer_score", lambda frame, _meta: (frame, {}))
    monkeypatch.setattr(sf, "_filter_special_treatment", lambda frame: frame)
    monkeypatch.setattr(sf, "_attach_instrument_names", lambda _data_dir, frame: frame)

    def add_scores(frame):
        return frame.with_columns(
            pl.lit(80.0).alias("entry_score"),
            pl.lit(80.0).alias("stockformer_score"),
            pl.lit(80.0).alias("fund_score"),
            pl.lit(80.0).alias("risk_score"),
            pl.lit(80.0).alias("technical_score"),
        )

    monkeypatch.setattr(sf, "_add_multifactor_scores", add_scores)
    monkeypatch.setattr(
        sf,
        "_attach_market_gate_for_oos",
        lambda _data_dir, frame: frame.with_columns(
            pl.lit(0.0).alias("market_buy_threshold_add"),
            pl.lit("open").alias("market_gate"),
            pl.lit(70.0).alias("market_regime_score"),
            pl.lit(0.0).alias("market_strong_down_pct"),
            pl.lit(50.0).alias("market_above_ma20_pct"),
        ),
    )
    monkeypatch.setattr(sf, "_entry_eligible_expr", lambda _frame: pl.lit(True))
    monkeypatch.setattr(sf, "_market_gate_oos_eligible_expr", lambda _frame: pl.lit(True))

    result = portfolio._build_gated_signal_frame(
        tmp_path,
        factor,
        feature_frame,
        {},
        horizon=5,
    )

    assert result["industry"].to_list() == ["银行"]

    higher_floor = portfolio._build_gated_signal_frame(
        tmp_path, factor, feature_frame, {}, horizon=5, forecast_floor=0.02,
    )
    assert higher_floor["_forecast_pass"].to_list() == [False]


def test_legacy_portfolio_research_routes_frozen_policy_to_rolling(monkeypatch, tmp_path):
    """A frozen policy must bypass the legacy threshold-tuning researcher."""
    from app.services import stockformer_rolling as rolling

    meta = {"model_id": "sf_frozen"}
    policy = {
        "version": "stockformer_signal_policy_v1",
        "effective_from": "2026-10-01",
    }
    captured = {}

    monkeypatch.setattr(sf, "_model_meta", lambda _data_dir, _model_id: meta)
    monkeypatch.setattr(portfolio, "load_frozen_signal_policy", lambda _data_dir, _meta: policy)

    def run_frozen(data_dir, **kwargs):
        captured["data_dir"] = data_dir
        captured.update(kwargs)
        return {"status": "waiting_for_new_oos_window"}

    monkeypatch.setattr(rolling, "run_rolling_portfolio_research", run_frozen)

    result = portfolio.run_portfolio_research(
        tmp_path,
        model_id="sf_frozen",
        max_positions=2,
        initial_capital=12345.0,
    )

    assert result["status"] == "waiting_for_new_oos_window"
    assert captured == {
        "data_dir": tmp_path,
        "model_id": "sf_frozen",
        "max_positions": 2,
        "initial_capital": 12345.0,
        "on_progress": None,
    }


def test_build_rolling_windows_excludes_short_tail():
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(130)]
    windows = build_rolling_windows(dates, window_days=60, min_window_days=30)
    assert [item["trading_days"] for item in windows] == [60, 60]
    assert windows[0]["start"] == dates[0]
    assert windows[1]["start"] == dates[60]


def test_strict_oos_routes_frozen_policy_without_reading_old_holdout(monkeypatch, tmp_path):
    from app.services import stockformer_rolling as rolling

    meta = {"model_id": "sf_frozen", "frozen_signal_policy": {
        "version": "stockformer_signal_policy_v1", "effective_from": "2026-10-01"}}
    monkeypatch.setattr(sf, "_model_meta", lambda *_: meta)
    called = {}

    def progress(*_):
        pass

    def run_frozen(data_dir, **kwargs):
        called.update(data_dir=data_dir, **kwargs)
        return {"status": "waiting_for_new_oos_window", "activation_decision": {"eligible": False}}

    monkeypatch.setattr(rolling, "run_rolling_portfolio_research", run_frozen)
    result = sf.strict_oos_backtest(tmp_path, model_id="sf_frozen", max_positions=2,
                                   initial_capital=12000, on_progress=progress)
    assert result["status"] == "waiting_for_new_oos_window"
    assert called == {"data_dir": tmp_path, "model_id": "sf_frozen", "max_positions": 2,
                      "initial_capital": 12000, "on_progress": progress}


def test_rolling_family_uses_training_columns_instead_of_full_fallback():
    from app.services import stockformer_training as training

    columns = list(sf.FEATURE_COLUMNS)
    assert _rolling_feature_columns("trend", columns) == list(training.TREND_FEATURES)
    minute = _rolling_feature_columns("daily_plus_minute", columns)
    assert "minute_intraday_return" in minute
    assert "minute_vwap_gap" in minute
    assert _rolling_feature_columns("unknown", columns) == []


def test_minute_research_features_filter_incomplete_rows_without_imputation(monkeypatch):
    from app.services import stockformer_training as training

    day = date(2026, 1, 2)
    rows = pl.DataFrame({
        "symbol": ["ready", "short"],
        "date": [day, day],
        "minute_data_available": [1.0, 1.0],
        "minute_bar_count": [240.0, 227.0],
        "minute_session_coverage": [1.0, 1.0],
        "minute_intraday_return": [0.02, None],
    })
    for column in sf.MINUTE_FEATURE_COLUMNS:
        if column not in rows.columns:
            rows = rows.with_columns(pl.lit(0.1).alias(column))
    monkeypatch.setattr(sf, "_feature_frame", lambda _panel, include_targets=False: rows)
    monkeypatch.setattr(training, "execution_targets", lambda features, _panel: features)

    features, coverage = portfolio._prepare_research_features(rows, "minute_enhanced")

    assert features["symbol"].to_list() == ["ready"]
    assert features["minute_intraday_return"].to_list() == [0.02]
    assert coverage["incomplete_rows"] == 1


def test_activation_requires_trailing_positive_windows_and_final_confirmation():
    rolling = {
        "stability": {
            "window_count": 3,
            "trailing_positive_run": 2,
        }
    }
    final = {"full_oos_portfolio_stats": {
        "total_return": 2.0, "trade_count": 25, "max_drawdown": -4.0,
    }}
    result = activation_decision(rolling, final, required_positive_windows=3)
    assert result["eligible"] is False
    assert result["status"] == "research_only"
    assert "连续成本后盈利窗口" in result["reasons"][0]


def test_activation_uses_selected_policy_portfolio_not_reference_model():
    rolling = {"stability": {"window_count": 3, "trailing_positive_run": 3}}
    final = {
        "full_oos_portfolio_stats": {
            "total_return": 3.0, "trade_count": 24, "max_drawdown": -12.0,
        },
        "reference_backtest": {"horizons": {"5": {"selected_model": {"stats": {
            "total_return": 8.0, "trade_count": 30, "max_drawdown": -3.0,
        }}}}},
    }
    result = activation_decision(rolling, final, required_positive_windows=3)
    assert result["eligible"] is False
    assert result["final_portfolio"]["max_drawdown"] == -12.0
    assert any("最大回撤" in reason for reason in result["reasons"])


def test_activation_fails_closed_without_selected_portfolio_stats():
    rolling = {"stability": {"window_count": 3, "trailing_positive_run": 3}}
    final = {"reference_backtest": {"horizons": {"5": {"selected_model": {"stats": {
        "total_return": 8.0, "trade_count": 30, "max_drawdown": -3.0,
    }}}}}}
    result = activation_decision(rolling, final, required_positive_windows=3)
    assert result["eligible"] is False
    assert any("缺少" in reason for reason in result["reasons"])


def test_waiting_report_records_zero_candidates_only_when_no_new_data(tmp_path):
    from app.services.stockformer_rolling import _write_waiting_report

    common = {
        "model_id": "sf_test",
        "meta": {"model_name": "test", "training_profile": "minute_enhanced"},
        "policy": {"effective_from": "2026-10-01"},
        "report_filename": "waiting.json",
        "reason": "waiting for a complete window",
    }
    empty = _write_waiting_report(tmp_path, available_dates=[], **common)
    assert empty["signal_observation"]["candidate_count"] == 0
    assert empty["signal_observation"]["evaluated"] is False
    assert empty["signal_observation"]["no_trade_reason"] == "no_new_oos_data"
    assert empty["activation_decision"]["eligible"] is False

    partial = _write_waiting_report(
        tmp_path, available_dates=[date(2026, 10, 9)], **common,
    )
    assert partial["signal_observation"]["candidate_count"] is None
    assert partial["signal_observation"]["no_trade_reason"] == "incomplete_oos_window"


def test_frozen_short_window_waits_before_loading_market_history(monkeypatch, tmp_path):
    from app.services import stockformer_rolling as rolling

    factor_path = tmp_path / "factors.parquet"
    pl.DataFrame({"symbol": ["600000.SH", "600000.SH"],
                  "date": [date(2026, 9, 30), date(2026, 10, 8)],
                  "pred_ret_5d": [0.1, 0.01]}).write_parquet(factor_path)
    meta = {"model_id": "sf_frozen", "factor_file": "factors.parquet"}
    monkeypatch.setattr(sf, "_model_meta", lambda *_: meta)
    monkeypatch.setattr(portfolio, "load_frozen_signal_policy", lambda *_: {"effective_from": "2026-10-01"})

    def forbidden_panel(*_, **__):
        raise AssertionError("An incomplete frozen window must not scan market history")

    monkeypatch.setattr(sf, "_load_panel", forbidden_panel)
    result = rolling.run_rolling_portfolio_research(tmp_path, model_id="sf_frozen")
    assert result["status"] == "waiting_for_new_oos_window"
    assert result["sample_out"]["trading_days"] == 1
    assert result["sample_out"]["available_start"] == "2026-10-08"
    assert result["signal_observation"]["evaluated"] is False
    assert result["signal_observation"]["candidate_count"] is None
    assert result["activation_decision"]["eligible"] is False


def test_regime_policy_keeps_defensive_states_blocked():
    frame = pl.DataFrame({
        "market_gate": ["closed", "observe", "strict", "open"],
        "market_regime_score": [20.0, 70.0, 65.0, 65.0],
        "market_strong_down_pct": [2.0, 2.0, 4.0, 4.0],
        "market_above_ma20_pct": [50.0, 50.0, 35.0, 35.0],
        "_market_gate_pass": [False, False, False, True],
    })
    out = frame.with_columns(
        _regime_policy_expr(frame, "relax_strict").alias("relaxed"),
        _regime_policy_expr(frame, "adaptive_recovery").alias("adaptive"),
    )
    assert out["relaxed"].to_list() == [False, False, True, True]
    assert out["adaptive"].to_list() == [False, False, True, True]
    assert set(REGIME_POLICIES) == {"strict_all", "relax_strict", "adaptive_recovery"}


def test_regime_policy_selection_only_uses_prior_windows():
    prior = [
        {
            "end": "2026-03-11",
            "regime_policy": "strict_all",
            "regime_policy_results": {
                "strict_all": {"selection_stats": {"total_return": 2.0, "max_drawdown": -4.0, "trade_count": 5}},
                "relax_strict": {"selection_stats": {"total_return": 3.0, "max_drawdown": -5.0, "trade_count": 6}},
                "adaptive_recovery": {"selection_stats": {"total_return": 1.0, "max_drawdown": -2.0, "trade_count": 4}},
            },
        }
    ]
    selected = _select_regime_policy(prior, minimum_trades=3)
    assert selected["policy"] == "relax_strict"
    assert selected["cutoff"] == "2026-03-11"
    assert selected["history_windows"] == 1


def test_regime_policy_de_risks_after_negative_window():
    prior = [{
        "end": "2026-06-12",
        "stats": {"total_return": -2.0, "trade_count": 14},
        "regime_policy_results": {
            "strict_all": {"selection_stats": {"total_return": -2.0, "trade_count": 14}},
            "relax_strict": {"selection_stats": {"total_return": 3.0, "trade_count": 20}},
            "adaptive_recovery": {"selection_stats": {"total_return": 1.0, "trade_count": 12}},
        },
    }]
    selected = _select_regime_policy(prior, minimum_trades=3)
    assert selected["policy"] == "strict_all"
    assert selected["reason"] == "risk_off_after_negative_window"
