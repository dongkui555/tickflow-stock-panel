from datetime import date, timedelta

import polars as pl

from app.backtest.engine import BacktestEngine, MatcherConfig
from app.services.stockformer_portfolio import _build_ranked_entry_orders
from app.services.stockformer_portfolio import _attach_net_margin
from app.services import stockformer as sf


def test_prediction_score_controls_fill_priority_when_positions_occupy_slots():
    first = date(2026, 1, 5)
    panel = pl.DataFrame(
        [
            {
                "symbol": symbol,
                "date": first + timedelta(days=offset),
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100000.0,
            }
            for symbol in ("A", "B", "C", "D")
            for offset in range(4)
        ]
    ).sort("symbol", "date")
    candidates = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "date": [first, first + timedelta(days=1), first + timedelta(days=1), first + timedelta(days=1)],
            "pred_ret_5d": [0.04, 0.01, 0.02, 0.03],
        }
    )
    scored_panel, entries, ranked = _build_ranked_entry_orders(
        panel, candidates, pred_col="pred_ret_5d", max_positions=3,
    )
    result = BacktestEngine(repo=None).simulate_portfolio(
        scored_panel,
        entries,
        pl.Series("_exit", [False] * scored_panel.height),
        MatcherConfig(
            matching="open_t+1", fees_pct=0, slippage_bps=0,
            max_positions=3, max_hold_days=5, initial_capital=100000,
        ),
    )

    assert ranked.filter(pl.col("date") == first + timedelta(days=1))["symbol"].to_list() == ["D", "C", "B"]
    assert {trade.symbol for trade in result.trades} == {"A", "C", "D"}
    assert {trade.symbol: trade.entry_score for trade in result.trades} == {
        "A": 0.04, "C": 0.02, "D": 0.03,
    }


def test_net_margin_score_replaces_raw_prediction_for_ranking():
    day = date(2026, 1, 5)
    panel = pl.DataFrame([
        {"symbol": symbol, "date": day + timedelta(days=offset), "open": 10.0,
         "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100000.0}
        for symbol in ("A", "B") for offset in range(3)
    ])
    candidates = pl.DataFrame({
        "symbol": ["A", "B"],
        "date": [day, day],
        "pred_ret_5d": [0.10, 0.01],
        "net_margin_score": [0.0006, 0.0012],
    })
    _, _, ranked = _build_ranked_entry_orders(
        panel, candidates, pred_col="pred_ret_5d", score_col="net_margin_score", max_positions=1,
    )
    assert ranked["symbol"].to_list() == ["B"]


def test_recent_candidate_gets_turnover_penalty_protection():
    day = date(2026, 1, 5)
    panel = pl.DataFrame([
        {"symbol": symbol, "date": day + timedelta(days=offset), "open": 10.0,
         "high": 10.0, "low": 10.0, "close": 10.0, "volume": 100000.0}
        for symbol in ("A", "B") for offset in range(4)
    ])
    candidates = pl.DataFrame({
        "symbol": ["A", "B", "A", "B"],
        "date": [day, day, day + timedelta(days=1), day + timedelta(days=1)],
        "pred_ret_5d": [0.01, 0.009, 0.0089, 0.0091],
        "net_margin_score": [0.0010, 0.0009, 0.0008, 0.0009],
    })
    _, _, ranked = _build_ranked_entry_orders(
        panel, candidates, pred_col="pred_ret_5d", score_col="net_margin_score", max_positions=1,
        turnover_penalty=0.0005, holding_lag_days=1,
    )
    assert ranked.filter(pl.col("date") == day)["symbol"].to_list() == ["A"]
    assert ranked.filter(pl.col("date") == day + timedelta(days=1))["symbol"].to_list() == ["A"]


def test_net_target_does_not_pay_fixed_cost_twice():
    frame = pl.DataFrame({
        "symbol": ["A"], "date": [date(2026, 1, 5)],
        "pred_ret_5d": [0.0006], "amount": [1_000_000.0],
    })
    result = _attach_net_margin(
        frame, pred_col="pred_ret_5d", min_net_edge=0.0005,
        target_mode="industry_relative_net",
    )
    assert result["estimated_round_trip_cost"].item() == 0.0
    assert result["_net_margin_pass"].item() is True


def test_adaptive_score_is_scoped_to_signal_date():
    frame = pl.DataFrame({
        "symbol": ["A", "B", "A", "B"],
        "date": [date(2026, 1, 5), date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 6)],
        "pred_ret_1d": [0.01, 0.02, 100.0, 200.0],
    })
    scored, _ = sf._add_adaptive_stockformer_score(
        frame, {"horizon_role_weights": {"1d": 1.0}},
    )
    first = scored.filter(pl.col("date") == date(2026, 1, 5))["stockformer_score"].to_list()
    assert first == [50.0, 100.0]
