from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest

from app.services import sector_research as research


def path(returns):
    return [
        dict(
            ret_1d=r,
            relative_return=r - 0.001,
            price_coverage=1.0,
            as_of=str(date(2026, 9, 1) + timedelta(days=i)),
        )
        for i, r in enumerate(returns)
    ]


def test_late_launch_never_rewrites_one_or_two_day_misses():
    prediction = {"forecast_kind": "start", "already_started": False}
    actuals = path([0.001, -0.003, 0.02, 0.0, 0.001])
    assert not research.assess(prediction, actuals[0], 1)["hit"]
    assert not research.assess(prediction, actuals[1], 2)["hit"]
    outcome = research.assess(prediction, {"window": actuals}, 5)
    assert outcome["hit"] and outcome["first_start_day"] == 3
    assert outcome["late_start"]
    assert outcome["period_return"] == pytest.approx(1.001 * 0.997 * 1.02 * 1.001 - 1)
    assert outcome["benchmark_return"] == pytest.approx(1.001**5 - 1)


def test_window_hit_is_not_a_profitable_trade_or_a_positive_terminal_return():
    outcome = research.assess({"forecast_kind": "start"}, {"window": path([0.02, -0.08, 0.001])}, 3)
    assert outcome["hit"]
    assert outcome["period_return"] < 0
    assert outcome["max_drawdown"] == pytest.approx(-0.08)
    assert outcome["first_start_day"] == 1 and not outcome["late_start"]


def test_already_started_is_not_early_layout():
    outcome = research.assess(
        {"forecast_kind": "continuation", "already_started": True},
        {"window": path([0.02, 0.03, 0.04])},
        3,
    )
    assert not outcome["hit"] and not outcome["new_start"]


@pytest.mark.parametrize("bad", ["missing", "coverage", "nan", "short"])
def test_partial_window_is_not_counted_as_hit_or_miss(bad):
    actuals = path([0.02, 0, 0.01])
    if bad == "missing":
        actuals[1] = None
    elif bad == "coverage":
        actuals[1]["price_coverage"] = 0.8
    elif bad == "nan":
        actuals[1]["ret_1d"] = float("nan")
    else:
        actuals.pop()
    assert research.assess({"forecast_kind": "start"}, {"window": actuals}, 3) is None


def test_outcomes_use_trading_sessions_and_do_not_compress_missing_dates():
    dates = [date(2026, 9, 4), date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9)]
    source = {d: ([dict(path([0.02])[0], sector="test", as_of=str(d))], {}) for d in dates}
    result = research.build_outcomes(dates, source, (1, 2, 3, 5))
    assert result[(dates[0], 3, "test")]["window"][-1]["as_of"] == "2026-09-09"
    assert (dates[0], 5, "test") not in result
    source[dates[2]] = ([], {})
    assert (dates[0], 3, "test") not in research.build_outcomes(dates, source, (3,))


def test_five_day_labels_end_before_the_next_fold():
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(220)]
    for fold in research.walk_forward_folds(dates, gap=5):
        assert dates.index(fold["train"][-1]) + 5 < dates.index(fold["test"][0])


def test_all_experimental_horizons_use_identical_risk_policy(monkeypatch):
    from app.backtest.engine import BacktestEngine
    from app.services.short_term_ai import SHORT_TERM_STOP_LOSS_PCT

    dates = [date(2026, 9, 1) + timedelta(days=i) for i in range(8)]
    bars = pl.DataFrame(
        [
            dict(
                date=d,
                symbol="000001",
                open=10.0,
                high=10.0,
                low=10.0,
                close=10.0,
                raw_close=10.0,
                volume=10000.0,
                amount=10000000.0,
                consecutive_limit_ups=0,
            )
            for d in dates
        ]
    )
    configs = []

    def simulate(self, matrix, config, **kwargs):
        configs.append(config)
        return SimpleNamespace(stats={}, equity_curve=[], trades=[])

    monkeypatch.setattr(BacktestEngine, "simulate_market_matrix", simulate)
    signals = {dates[0]: [dict(leader_stocks=[{"symbol": "000001"}], prediction_score=80)]}
    for horizon in (1, 2, 3, 5):
        research.trade_comparison(
            bars,
            signals,
            dates,
            horizon,
            100000,
            stop_loss_pct=SHORT_TERM_STOP_LOSS_PCT["balanced"],
        )
    assert {c.stop_loss_pct for c in configs} == {0.045}
    assert {c.max_exposure_pct for c in configs} == {0.6}
    assert {c.entry_fill for c in configs} == {"open_t+1"}
    assert [c.max_hold_days for c in configs] == [1, 2, 3, 5]


def test_experiment_results_are_separate_and_no_live_rule_is_changed(tmp_path, monkeypatch):
    from test_sector_forecast_v2 import evidence

    from app.services.sector_forecast_rules import MODEL_VERSION, rank_forecasts

    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(100)]
    monkeypatch.setattr(research, "trading_dates", lambda *a: dates)
    monkeypatch.setattr(research.rotation, "input_signature", lambda *a: [("same", 1)])
    monkeypatch.setattr(
        research.rotation, "_sector_daily_frame", lambda *a: (None, {}, pl.DataFrame())
    )
    monkeypatch.setattr(
        research.rotation,
        "rows_for_date",
        lambda *a: [
            evidence(sector="test", price_coverage=1.0, leader_stocks=[{"symbol": "000001"}])
        ],
    )
    monkeypatch.setattr(research, "enrich_context", lambda rows: (rows, {}))
    trade = Mock(return_value={"total_return": 0, "n_trades": 0})
    monkeypatch.setattr(research, "trade_comparison", trade)
    baseline_dir = tmp_path / "research/sector_forecasts"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "old.json").write_text('{"status":"research_only","comparisons":[]}')
    old_pointer = '{"id":"old"}'
    (baseline_dir / "latest.json").write_text(old_pointer)
    before = rank_forecasts([evidence()], {})
    report = research.run(
        tmp_path,
        lookback_days=100,
        train_days=20,
        test_days=10,
        holdout_days=20,
        experiment="early_layout",
    )
    assert report["model_version"] == MODEL_VERSION
    assert report["parameters"]["stop_loss_pct"] == 0.045
    assert {r["horizon"] for r in report["comparisons"]} == {1, 2, 3, 5}
    assert all(f["gap_sessions"] == 5 for f in report["folds"])
    assert report["forecast_records"]
    assert all(set(r["validations"]) == {"1", "2", "3", "5"} for r in report["forecast_records"])
    assert report["origin"] == "historical_replay"
    assert not (tmp_path / "user_data/sector_forecasts_v2.sqlite3").exists()
    assert (baseline_dir / "latest.json").read_text() == old_pointer
    assert research.latest(tmp_path, experiment="early_layout")["id"] == report["id"]
    assert research.latest(tmp_path)["comparisons"] == []
    assert rank_forecasts([evidence()], {}) == before


def test_long_holding_period_does_not_delay_a_stop_loss():
    dates = [date(2026, 9, 1) + timedelta(days=i) for i in range(8)]
    rows = []
    for index, day in enumerate(dates):
        price = 10.0 if index < 2 else 9.0
        rows.append(
            dict(
                date=day,
                symbol="000001",
                open=price,
                high=price,
                low=price,
                close=price,
                raw_close=price,
                volume=1000000.0,
                amount=1000000000.0,
                consecutive_limit_ups=0,
            )
        )
    bars = pl.DataFrame(rows)
    signals = {dates[0]: [dict(leader_stocks=[{"symbol": "000001"}], prediction_score=80)]}
    results = [
        research.trade_comparison(bars, signals, dates, h, 100000, stop_loss_pct=0.045)
        for h in (3, 5)
    ]
    for result in results:
        assert result["trades"]
        trade = result["trades"][0]
        assert trade["exit_reason"] == "stop_loss"
        assert date.fromisoformat(str(trade["exit_date"])) < dates[5]
        assert trade["pnl_pct"] < -0.045
    assert results[0]["trades"][0]["exit_date"] == results[1]["trades"][0]["exit_date"]


def test_calibration_for_long_window_does_not_use_test_labels():
    from test_sector_forecast_v2 import evidence

    from app.services.sector_forecast_rules import classify

    row = classify(evidence(), {})
    day = date(2026, 9, 1)
    training = {day: [row]}
    outcomes = {(day, 5, "test"): {"window": path([0, 0, 0, 0.02, 0])}}
    before = research.calibration(training, outcomes, 5)
    outcomes[(date(2026, 9, 8), 5, "test")] = {"window": path([-0.1] * 5)}
    assert research.calibration(training, outcomes, 5) == before


def test_api_experiment_mode_is_validated_and_isolated(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.short_term_ai import router

    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.include_router(router)
    with TestClient(app) as client:
        assert (
            client.get("/api/short-term-ai/sector-research?experiment=early_layout").json()[
                "status"
            ]
            == "empty"
        )
        assert client.get("/api/short-term-ai/sector-research?experiment=bad").status_code == 422
        assert (
            client.post(
                "/api/short-term-ai/sector-research", json={"experiment": "bad"}
            ).status_code
            == 422
        )
    assert not (tmp_path / "research").exists()
