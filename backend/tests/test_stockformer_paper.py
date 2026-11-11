import json
from datetime import date, timedelta

import polars as pl

from app.services import stockformer as sf
from app.services import stockformer_paper as paper
from app.services import stockformer_portfolio as portfolio
from app.services import stockformer_rolling as rolling
from app.services.stockformer_paper import _daily_observations, _waiting_result


def test_daily_observations_records_zero_candidate_reason_by_layer():
    start = date(2026, 10, 5)
    rows = [
        # Prediction gate removes the first day.
        {
            "symbol": "A", "date": start, "pred_ret_5d": 0.01,
            "_prediction_pass": False, "_forecast_pass": False,
            "_net_margin_pass": True, "_entry_rules_pass": True, "_market_gate_pass": True,
            "net_margin_score": 0.01,
        },
        # Prediction passes, but the net edge does not.
        {
            "symbol": "A", "date": start + timedelta(days=1), "pred_ret_5d": 0.01,
            "_prediction_pass": True, "_forecast_pass": False,
            "_net_margin_pass": False, "_entry_rules_pass": True, "_market_gate_pass": True,
            "net_margin_score": -0.01,
        },
        # Net edge passes, but entry rules remove the row.
        {
            "symbol": "A", "date": start + timedelta(days=2), "pred_ret_5d": 0.01,
            "_prediction_pass": True, "_forecast_pass": True,
            "_net_margin_pass": True, "_entry_rules_pass": False, "_market_gate_pass": True,
            "net_margin_score": 0.01,
        },
        # Entry passes, but the market gate removes the row.
        {
            "symbol": "A", "date": start + timedelta(days=3), "pred_ret_5d": 0.01,
            "_prediction_pass": True, "_forecast_pass": True,
            "_net_margin_pass": True, "_entry_rules_pass": True, "_market_gate_pass": False,
            "net_margin_score": 0.01,
        },
        # This is the only selected candidate.
        {
            "symbol": "A", "date": start + timedelta(days=4), "pred_ret_5d": 0.01,
            "_prediction_pass": True, "_forecast_pass": True,
            "_net_margin_pass": True, "_entry_rules_pass": True, "_market_gate_pass": True,
            "net_margin_score": 0.01,
        },
    ]
    signal = pl.DataFrame(rows)
    observations, reasons = _daily_observations(
        signal,
        available_dates=[start + timedelta(days=i) for i in range(6)],
        pred_col="pred_ret_5d",
        max_positions=1,
        forecast_ceiling=None,
    )

    assert [row["no_trade_reason"] for row in observations] == [
        "prediction_gate_empty",
        "net_margin_empty",
        "entry_rules_empty",
        "market_gate_empty",
        "candidate_selected",
        "feature_rows_empty",
    ]
    assert [row["selected_count"] for row in observations] == [0, 0, 0, 0, 1, 0]
    assert reasons["prediction_gate_empty"] == 1
    assert reasons["feature_rows_empty"] == 1


def test_waiting_result_records_no_new_data_and_research_only_activation():
    result = _waiting_result(
        selection={"model_id": "sf_test", "selection_id": "sel_test"},
        cutoff=date(2026, 9, 30),
        effective_end=date(2026, 9, 30),
        required=69,
    )

    assert result["signal_count"] == 0
    assert result["evaluated"] is False
    assert result["no_trade_reason"] == "no_new_oos_data"
    assert result["no_trade_reason_counts"] == {"no_new_oos_data": 1}
    assert result["activation"]["status"] == "research_only"
    assert result["activation_eligible"] is False


def test_shadow_tracking_persists_zero_candidate_day_when_signal_is_empty(monkeypatch, tmp_path):
    new_day = date(2026, 10, 9)
    selection_path = tmp_path / "selection.json"
    output_path = tmp_path / "shadow.json"
    selection_path.write_text(json.dumps({
        "model_id": "sf_test",
        "selection_id": "sel_test",
        "unseen_after": "2026-09-30",
        "training_profile": "minute_enhanced",
        "target_mode": "industry_relative_net",
        "alpha": 1.0,
        "forecast_floor": 0.0,
        "max_positions": 2,
        "selected": {"feature_family": "full", "forecast_ceiling": None},
    }), encoding="utf-8")
    panel = pl.DataFrame({"symbol": ["A"], "date": [new_day], "x": [1.0]})
    monkeypatch.setattr(sf, "_model_meta", lambda *_args: {
        "train_start": "2024-01-01", "training_universe": ["A"],
        "training_profile": "minute_enhanced",
    })
    monkeypatch.setattr(sf, "_date_range_from_data", lambda *_args: (date(2024, 1, 1), new_day))
    monkeypatch.setattr(sf, "_load_panel", lambda *_args, **_kwargs: panel)
    monkeypatch.setattr(sf, "training_feature_columns", lambda *_args: ["x"])
    monkeypatch.setattr(portfolio, "_prepare_research_features", lambda *_args: (panel, {}))
    monkeypatch.setattr(portfolio, "load_frozen_signal_policy", lambda *_args: {
        "prediction_rule": {"min_net_edge": 0.0005},
    })
    monkeypatch.setattr(portfolio, "_fit_candidate_predictions", lambda *_args, **_kwargs: pl.DataFrame())
    monkeypatch.setattr(portfolio, "_build_gated_signal_frame", lambda *_args, **_kwargs: pl.DataFrame())
    monkeypatch.setattr(rolling, "_rolling_feature_columns", lambda *_args: ["x"])

    result = paper.run_shadow_tracking(tmp_path, selection_path, output_path)

    assert result["available_sessions"] == 1
    assert result["signal_count"] == 0
    assert result["evaluated"] is True
    assert result["daily_observations"][0]["no_trade_reason"] == "feature_rows_empty"
    assert result["layer_diagnostics"]["total_rows"] == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
