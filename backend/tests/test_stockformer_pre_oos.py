import json
from datetime import date, timedelta

import polars as pl

from app.backtest.engine import BacktestEngine
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_pre_oos as pre_oos


def test_training_windows_leave_six_settlement_sessions_before_holdout():
    days = [date(2025, 1, 1) + timedelta(days=offset) for offset in range(200)]
    windows = pre_oos._three_training_windows(days, days[0], days[-1] + timedelta(days=1))

    assert len(windows) == 3
    assert windows[0]["start"] == days[5]
    assert windows[-1]["end"] == days[193]
    assert windows[-1]["settlement_end"] == days[199]


def test_prediction_reverse_sells_at_next_open_before_five_day_expiry():
    first = date(2026, 1, 5)
    panel = pl.DataFrame({
        "symbol": ["A"] * 8,
        "date": [first + timedelta(days=offset) for offset in range(8)],
        "open": [10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 8.0, 8.0],
        "high": [10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 8.0, 8.0],
        "low": [10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 8.0, 8.0],
        "close": [10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 8.0, 8.0],
        "volume": [100000.0] * 8,
    })
    signal = panel.select("symbol", "date").with_columns(
        pl.Series("pred_ret_5d", [0.05, -0.01] + [0.01] * 6),
        pl.lit(True).alias("_entry_rules_pass"),
        pl.lit(True).alias("_market_gate_pass"),
    )
    engine = BacktestEngine(repo=None)
    common = dict(
        horizon=5, forecast_floor=0.004, max_positions=1,
        initial_capital=100000.0, signal_frame=signal, entry_signal_end=first,
    )

    fixed = portfolio._run_one(panel, signal, engine, exit_mode="max_hold", **common)
    reverse = portfolio._run_one(panel, signal, engine, exit_mode="prediction_reverse", **common)

    assert fixed["trade_count"] == reverse["trade_count"] == 1
    assert reverse["total_return"] > fixed["total_return"]


def test_future_validation_waits_for_complete_window(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"unseen_after": "2026-09-29"}), encoding="utf-8")
    monkeypatch.setattr(pre_oos, "_available_future_dates", lambda _data_dir, _after: [])

    result = pre_oos.validate_new_window(tmp_path, selection, tmp_path / "future.json")

    assert result["status"] == "pending_complete_window"
    assert result["required_sessions"] == 69
    assert result["activation_eligible"] is False
    assert not (tmp_path / "future.json").exists()


def test_forecast_ceiling_is_applied_before_daily_ranking():
    day = date(2026, 1, 5)
    panel = pl.DataFrame({
        "symbol": ["A", "B"] * 3,
        "date": [day, day, day + timedelta(days=1), day + timedelta(days=1), day + timedelta(days=2), day + timedelta(days=2)],
        "open": [10.0] * 6, "high": [10.0] * 6,
        "low": [10.0] * 6, "close": [10.0] * 6,
        "volume": [100000.0] * 6,
    })
    signal = panel.filter(pl.col("date") == day).with_columns(
        pl.Series("pred_ret_5d", [0.08, 0.02]),
        pl.lit(True).alias("_entry_rules_pass"),
        pl.lit(True).alias("_market_gate_pass"),
    )
    stats = portfolio._run_one(
        panel, signal, BacktestEngine(repo=None),
        horizon=5, forecast_floor=0.004, forecast_ceiling=0.03,
        max_positions=1, initial_capital=100000.0, signal_frame=signal,
    )
    assert stats["trade_count"] == 1
