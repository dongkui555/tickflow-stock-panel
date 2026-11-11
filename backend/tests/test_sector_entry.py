from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.services import sector_entry as entry


def stock(**updates):
    row = dict(
        symbol="000001",
        name="sample",
        sector="test",
        close=10.0,
        open=9.9,
        high=10.1,
        low=9.8,
        ret_3d=0.02,
        ret_5d=0.03,
        amount=100_000_000.0,
        amount_ratio_5d=1.2,
        history_ok=True,
        price_valid=True,
        limit_up_flag=0,
        limit_known=1,
        ma5=9.9,
        ma20=9.8,
        ma20_prior=9.75,
        low10=9.7,
        position20=0.6,
        atr_pct=0.02,
        close_location=0.8,
        relative5=0.01,
        prior_close=9.9,
        prior_high=9.98,
        turnover_rate=3.0,
    )
    return {**row, **updates}


def test_hot_winner_is_not_promoted_above_a_tradable_stock():
    hot = entry.evaluate_stock(stock(symbol="000002", ret_5d=0.20, position20=0.95))
    normal = entry.evaluate_stock(stock())
    assert not hot["eligible"] and "追涨位置" in hot["reasons"]
    assert normal["eligible"] and normal["score"] > 0


@pytest.mark.parametrize(
    "change",
    [dict(history_ok=False), dict(atr_pct=None), dict(amount=float("nan")), dict(limit_known=0)],
)
def test_missing_stock_inputs_fail_closed(change):
    assert not entry.evaluate_stock(stock(**change))["eligible"]


def test_confirmation_waits_and_invalidates_broken_support():
    frozen = stock()
    sector = dict(price_coverage=1, flow_coverage=1)
    assert entry.confirm(frozen, stock(), sector, {})["status"] == "confirmed"
    assert entry.confirm(frozen, stock(amount_ratio_5d=0.7), sector, {})["status"] == "waiting"
    assert entry.confirm(frozen, stock(close=9.3), sector, {})["status"] == "invalidated"
    assert entry.confirm(frozen, stock(), sector, {"weak": True})["status"] == "waiting"
    assert entry.confirm(frozen, None, sector, {})["status"] == "missing_data"


def panel():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(35)]
    return days, pl.DataFrame(
        [
            dict(
                date=d,
                symbol="000001",
                name="test",
                sector="test",
                open=10 + i * 0.01,
                high=10.15 + i * 0.01,
                low=9.85 + i * 0.01,
                close=10.05 + i * 0.01,
                amount=100_000_000.0,
                volume=10_000_000.0,
                price_valid=True,
                ret_3d=0.01,
                ret_5d=0.02,
                amount_ratio_5d=1.2,
                turnover_rate=3.0,
                limit_up_flag=0.0,
                limit_known=1.0,
            )
            for i, d in enumerate(days)
        ]
    )


def test_features_are_prefix_invariant_and_calendar_gaps_do_not_compress():
    days, bars = panel()
    full = entry.prepare_features(bars, days)
    prefix = entry.prepare_features(bars.filter(pl.col("date") <= days[25]), days[:26])
    assert full.filter(pl.col("date") == days[25]).to_dicts() == prefix.tail(1).to_dicts()
    broken = entry.prepare_features(bars.filter(pl.col("date") != days[22]), days)
    assert not broken.filter(pl.col("date") == days[25])["history_ok"][0]


def test_opening_guard_uses_only_open_and_never_next_close():
    from types import SimpleNamespace

    matrix = SimpleNamespace(
        entry=np.array([[False], [True]]),
        entry_signal_time=np.array([[-1], [0]]),
        symbols=["000001"],
        timestamp_labels=["2025-01-01", "2025-01-02"],
        open=np.array([[10.0], [10.5]]),
    )
    bounds = {
        ("2025-01-01", "000001"): {"floor": 9.7, "ceiling": 10.2, "execution_date": "2025-01-02"}
    }
    allowed, rejected = entry.apply_opening_guards(matrix, bounds)
    assert not allowed[1, 0] and rejected[0]["reason"] == "开盘超出入场区间"
    assert matrix.entry[1, 0]


def test_no_delayed_fill_after_missing_execution_session():
    from types import SimpleNamespace

    matrix = SimpleNamespace(
        entry=np.array([[False], [True]]),
        entry_signal_time=np.array([[-1], [0]]),
        symbols=["000001"],
        timestamp_labels=["2025-01-01", "2025-01-03"],
        open=np.array([[10.0], [10.0]]),
    )
    allowed, rejected = entry.apply_opening_guards(
        matrix,
        {("2025-01-01", "000001"): {"floor": 9.7, "ceiling": 10.2, "execution_date": "2025-01-02"}},
    )
    assert not allowed[1, 0] and rejected[0]["reason"] == "次日缺行情或停牌"


def test_confirmation_freezes_stock_and_uses_next_session_not_same_close():
    days = [date(2026, 9, 4), date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9)]
    features = pl.DataFrame([stock(date=d) for d in days])
    signals = {
        days[0]: [
            dict(
                sector="test",
                prediction_score=80,
                leader_stocks=[dict(symbol="000001", name="test")],
            )
        ]
    }
    source = {d: ([dict(sector="test", price_coverage=1, flow_coverage=1)], {}) for d in days}
    planned, audit, bounds = entry.plan_entries(features, signals, source, days, "ranked_confirm")
    assert list(planned) == [days[1]]
    assert audit[0]["confirmation_date"] == str(days[1])
    assert audit[0]["scheduled_entry_date"] == str(days[2])
    assert bounds[(str(days[1]), "000001")]["execution_date"] == str(days[2])
    assert planned[days[1]][0]["leader_stocks"][0]["symbol"] == "000001"


def test_broken_support_cannot_be_retried_after_a_later_rebound():
    days = [date(2026, 9, 7) + timedelta(days=i) for i in range(4)]
    features = pl.DataFrame(
        [stock(date=d, close=9.3 if i == 1 else 10.0) for i, d in enumerate(days)]
    )
    signals = {
        days[0]: [dict(sector="test", prediction_score=80, leader_stocks=[dict(symbol="000001")])]
    }
    source = {d: ([dict(sector="test", price_coverage=1, flow_coverage=1)], {}) for d in days}
    planned, audit, _ = entry.plan_entries(features, signals, source, days, "leader_confirm")
    assert not planned and audit[0]["status"] == "invalidated"
    assert len(audit[0]["checks"]) == 1


def test_four_arms_keep_same_risk_and_record_real_execution(monkeypatch):
    days = [date(2026, 9, 1) + timedelta(days=i) for i in range(12)]
    features = pl.DataFrame([stock(date=d) for d in days])
    monkeypatch.setattr(entry, "prepare_features", lambda *a, **kw: features)
    signals = {
        days[0]: [dict(sector="test", prediction_score=80, leader_stocks=[dict(symbol="000001")])]
    }
    source = {d: ([dict(sector="test", price_coverage=1, flow_coverage=1)], {}) for d in days}
    calls = []

    def matcher(bars, planned, dates, horizon, capital, **kwargs):
        calls.append((horizon, capital, kwargs["stop_loss_pct"]))
        decision_day = next(iter(planned))
        return dict(
            total_return=-0.01,
            n_trades=1,
            trades=[
                dict(
                    symbol="000001",
                    entry_signal_date=str(decision_day),
                    pnl_pct=-0.05,
                    exit_reason="stop_loss",
                )
            ],
        )

    comparisons, audit = entry.run_comparison(
        features,
        signals,
        source,
        days,
        [dict(kind="holdout", test=[days[0]])],
        100000,
        0.045,
        matcher,
        lambda *a: None,
    )
    assert len(comparisons) == 8 and len(audit) == 4
    assert set(calls) == {(3, 100000, 0.045), (5, 100000, 0.045)}
    assert all(r["stop_count"] == 1 for r in comparisons)
    assert all(r["executions"]["5"]["status"] == "closed" for r in audit)


def test_new_experiment_api_does_not_replace_prior_results(tmp_path):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.short_term_ai import SectorResearchRequest, router
    from app.services.sector_research import result_pointer

    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.include_router(router)
    assert SectorResearchRequest(experiment="stock_entry").experiment == "stock_entry"
    assert len({result_pointer(e) for e in ("baseline", "early_layout", "stock_entry")}) == 3
    with TestClient(app) as client:
        assert (
            client.get("/api/short-term-ai/sector-research?experiment=stock_entry").json()["status"]
            == "empty"
        )


def test_opening_guard_runs_through_real_portfolio_matcher():
    from app.services.sector_research import trade_comparison

    days, bars = panel()
    bars = bars.with_columns(
        pl.col("close").alias("raw_close"), pl.lit(0).alias("consecutive_limit_ups")
    )
    signals = {days[25]: [dict(prediction_score=80, leader_stocks=[dict(symbol="000001")])]}
    bounds = {(str(days[25]), "000001"): dict(execution_date=str(days[26]), floor=9, ceiling=10)}
    rejected = trade_comparison(
        bars, signals, days, 5, 100000, stop_loss_pct=0.045, opening_bounds=bounds
    )
    assert rejected["n_trades"] == 0 and len(rejected["opening_rejections"]) == 1
    bounds[(str(days[25]), "000001")]["ceiling"] = 11
    accepted = trade_comparison(
        bars, signals, days, 5, 100000, stop_loss_pct=0.045, opening_bounds=bounds
    )
    assert accepted["n_trades"] == 1
    assert accepted["trades"][0]["entry_signal_date"] == str(days[25])
    assert accepted["trades"][0]["entry_date"] == str(days[26])


def test_name_mapping_is_by_symbol_and_risk_names_are_excluded():
    days, bars = panel()
    features = entry.prepare_features(bars.drop("name"), days, name_map={"000001": "ST sample"})
    assert features["name"].unique().to_list() == ["ST sample"]
    assert not entry.evaluate_stock(stock(name="ST sample"))["eligible"]
    assert not entry.evaluate_stock(stock(name=None))["eligible"]


def test_ranked_stock_does_not_inherit_original_leader_name():
    days = [date(2026, 9, 7), date(2026, 9, 8)]
    features = pl.DataFrame([stock(date=days[0], name="chosen")])
    signals = {
        days[0]: [
            dict(
                sector="test",
                prediction_score=80,
                leader_stocks=[dict(symbol="000002", name="original")],
            )
        ]
    }
    planned, audit, _ = entry.plan_entries(features, signals, {}, days, "ranked_open")
    assert audit[0]["symbol"] == "000001" and audit[0]["name"] == "chosen"
    assert planned[days[0]][0]["leader_stocks"] == [dict(symbol="000001", name="chosen")]
    _, missing, _ = entry.plan_entries(
        features.with_columns(pl.lit(None).alias("name")), signals, {}, days, "ranked_open"
    )
    assert missing[0]["status"] == "no_candidate" and missing[0]["name"] is None


def test_partly_missing_confirmation_is_not_counted_as_normal_expiry():
    days = [date(2026, 9, 7) + timedelta(days=i) for i in range(4)]
    features = pl.DataFrame([stock(date=d, amount_ratio_5d=0.7) for d in days])
    signals = {
        days[0]: [dict(sector="test", prediction_score=80, leader_stocks=[dict(symbol="000001")])]
    }
    source = {days[1]: ([dict(sector="test", price_coverage=1, flow_coverage=1)], {})}
    planned, audit, _ = entry.plan_entries(features, signals, source, days, "ranked_confirm")
    assert not planned and audit[0]["status"] == "missing_data"


def test_strength_profile_rejects_low_weak_follower_not_merely_cheap():
    weak = stock(position20=0.4, relative5=-0.04, ret_3d=-0.03, ma5=10.05, close_location=0.35)
    assert entry.evaluate_stock(weak)["eligible"]
    checked = entry.evaluate_stock(weak, profile="strength")
    assert not checked["eligible"]
    assert any(
        c["id"] == "relative_strength" and c["status"] == "failed" for c in checked["conditions"]
    )


def test_tight_range_high_position_is_not_automatically_a_chase():
    tight = stock(position20=0.88, ret_3d=0.01, ret_5d=0.02)
    assert not entry.evaluate_stock(tight)["eligible"]
    assert entry.evaluate_stock(tight, profile="strength")["eligible"]
    assert not entry.evaluate_stock(stock(ret_3d=0.08), profile="strength")["eligible"]


def test_soft_ablation_is_one_condition_only_and_never_bypasses_risk():
    sector = dict(price_coverage=1, flow_coverage=1)
    row = stock(amount_ratio_5d=0.9)
    assert entry.confirm(stock(), row, sector, {})["status"] == "waiting"
    assert entry.confirm(stock(), row, sector, {}, mode="no_volume")["status"] == "confirmed"
    assert entry.confirm(stock(), row, sector, {}, mode="no_candle")["status"] == "waiting"
    assert (
        entry.confirm(stock(), row, sector, {"weak": True}, mode="no_volume")["status"] == "waiting"
    )
    assert (
        entry.confirm(stock(), stock(close=9.3), sector, {}, mode="no_volume")["status"]
        == "invalidated"
    )
    with pytest.raises(ValueError):
        entry.confirm(stock(), row, sector, {}, mode="no_market")


def test_adaptive_confirmation_distinguishes_reclaim_from_breakout():
    sector = dict(price_coverage=1, flow_coverage=1)
    reclaim = stock(prior_high=10.2, amount_ratio_5d=0.9)
    assert entry.confirm(stock(), reclaim, sector, {}, mode="adaptive")["status"] == "confirmed"
    breakout_only = stock(low=10.0, ma5=9.8, amount_ratio_5d=0.9)
    assert entry.confirm(stock(), breakout_only, sector, {}, mode="adaptive")["status"] == "waiting"
    assert (
        entry.confirm(
            stock(), stock(prior_high=10.2, amount_ratio_5d=0.7), sector, {}, mode="adaptive"
        )["status"]
        == "waiting"
    )


def test_conditions_record_observed_values_and_missing_evidence():
    result = entry.confirm(
        stock(), stock(amount_ratio_5d=0.9), dict(price_coverage=1, flow_coverage=1), {}
    )
    volume = next(c for c in result["conditions"] if c["id"] == "volume")
    assert volume["observed"]["amount_ratio_5d"] == 0.9
    assert volume["status"] == "failed" and not volume["hard"]
    missing = entry.evaluate_stock(stock(turnover_rate=None))
    assert missing["conditions"][0]["status"] == "missing"


def test_condition_funnel_distinguishes_overlapping_and_single_blockers():
    def make(key, passed, active=True):
        return entry.condition(key, key, passed, {}, key, active=active)

    groups = [
        [make("a", False), make("b", False)],
        [make("a", True), make("b", False)],
        [make("a", True), make("b", False, active=False)],
    ]
    rows = {r["id"]: r for r in entry.summarize_conditions(groups)}
    assert (
        rows["a"]["tested"] == 3 and rows["a"]["sole_blocker"] == 0 and rows["a"]["survivors"] == 2
    )
    assert (
        rows["b"]["sole_blocker"] == 1
        and rows["b"]["disabled"] == 1
        and rows["b"]["survivors"] == 1
    )
    merged = entry.combine_summaries([list(rows.values()), list(rows.values())])
    assert merged[1]["tested"] == 6 and merged[1]["sole_blocker"] == 2


@pytest.mark.parametrize("variant", entry.DIAGNOSTIC_VARIANTS)
def test_all_experiment_arms_preserve_next_session_bounds(variant):
    days = [date(2026, 9, 7) + timedelta(days=i) for i in range(4)]
    features = pl.DataFrame([stock(date=d) for d in days])
    signals = {days[0]: [dict(sector="test", leader_stocks=[dict(symbol="000001")])]}
    source = {d: ([dict(sector="test", price_coverage=1, flow_coverage=1)], {}) for d in days}
    planned, audit, bounds = entry.plan_entries(features, signals, source, days, variant)
    decision = days[0] if variant.endswith("open") else days[1]
    assert list(planned) == [decision]
    bound = bounds[(str(decision), "000001")]
    assert bound["execution_date"] == str(days[days.index(decision) + 1])
    if not variant.endswith("open"):
        assert bound["floor"] >= entry.support(stock()) and bound["ceiling"] == 10.2
        assert all(c["active"] for c in audit[0]["checks"][0]["conditions"] if c["hard"])
