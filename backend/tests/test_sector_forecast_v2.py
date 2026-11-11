from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import polars as pl

from app.services import sector_forecast
from app.services import sector_layout as rotation
from app.services.sector_forecast_rules import classify, evaluate_outcome
from app.services.sector_forecast_store import ForecastStore
from app.services.sector_research import calibration, clean_json, walk_forward_folds


def evidence(**changes):
    row = dict(
        sector="test",
        ret_1d=0.002,
        ret_5d=0.015,
        ret_10d=0.02,
        relative_return=0.008,
        flow_coverage=1.0,
        flow_history_days=20,
        flow_intensity=0.03,
        flow_acceleration=0.02,
        flow_zscore=1.5,
        prior_flow_intensity=-0.005,
        amount_ratio_5d=1.2,
        up_ratio=0.45,
        price_resilient=True,
        leader_strength=0.025,
        parent_flow_acceleration=0.01,
        risk_score=80.0,
        member_count=20,
        limit_up_count=0,
        sector_data_ready=True,
    )
    row.update(changes)
    return row


def test_weakness_alone_is_not_accumulation():
    row = classify(
        evidence(
            ret_1d=-0.02,
            up_ratio=0.10,
            relative_return=-0.03,
            price_resilient=False,
            flow_intensity=-0.01,
            flow_acceleration=0,
            flow_zscore=0,
        ),
        {},
    )
    assert row["prediction_level"] != "strong"
    assert not row["evidence"]["price_resilient"]


def test_fresh_money_and_confluence_can_confirm_start():
    row = classify(evidence(), {})
    assert row["forecast_kind"] == "start"
    assert row["prediction_level"] == "strong"
    assert row["probability"] is None


def test_risk_and_missing_data_can_abstain():
    assert classify(evidence(), {"very_weak": True})["prediction_level"] != "strong"
    assert classify(evidence(flow_coverage=0.50), {})["prediction_level"] == "unavailable"


def test_continuation_is_a_different_target():
    assert (
        classify(evidence(ret_1d=0.02, relative_return=0.018, up_ratio=0.7), {})["forecast_kind"]
        == "continuation"
    )
    actual = {"ret_1d": 0.02, "relative_return": 0.01, "actual_rank": 1}
    assert not evaluate_outcome({"forecast_kind": "start", "already_started": True}, actual)["hit"]
    assert evaluate_outcome({"forecast_kind": "continuation"}, actual)["hit"]
    assert not evaluate_outcome({"forecast_kind": "start"}, dict(actual, ret_1d=-0.01))["hit"]


def test_forecasts_are_insert_only_under_concurrency(tmp_path):
    store = ForecastStore(tmp_path)

    def save(i):
        return store.save(
            {"id": "same", "as_of": "2026-09-08", "origin": "published", "rows": [{"score": i}]}
        )

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(save, range(10)))
    assert len({str(r["rows"]) for r in results}) == 1
    assert len(store.list()) == 1
    original = store.list()[0]
    store.append_validation("same", 1, {"input_version": "a", "rows": []})
    store.append_validation("same", 1, {"input_version": "b", "rows": []})
    store.append_validation("same", 1, {"input_version": "b", "rows": []})
    assert store.list()[0] == original
    assert len(store.validations("same")) == 2


def test_ten_percent_gain_does_not_override_growth_board_limit_flag(tmp_path):
    rows = []
    for day, close in [(date(2026, 9, 7), 10.0), (date(2026, 9, 8), 11.3)]:
        folder = tmp_path / "kline_daily_enriched" / f"date={day}"
        folder.mkdir(parents=True)
        pl.DataFrame(
            [
                dict(
                    symbol="300290",
                    date=day,
                    close=close,
                    open=close,
                    high=close,
                    low=close,
                    raw_close=close,
                    volume=100,
                    amount=1000.0,
                    consecutive_limit_ups=0,
                )
            ]
        ).write_parquet(folder / "part.parquet")
        rows.append(day)
    industry = pl.DataFrame({"symbol": ["300290"], "sector": ["IT"], "industry_raw": ["IT"]})
    result, _ = rotation._load_daily_panel(tmp_path, rows, industry)
    assert result["limit_up_flag"].to_list() == [0.0, 0.0]


def test_published_generation_waits_for_close_and_ready_data(tmp_path, monkeypatch):
    from app.market_time import CN_TZ
    from app.services import prediction_workflow

    monkeypatch.setattr(sector_forecast, "cn_now", lambda: datetime(2026, 9, 9, 12, tzinfo=CN_TZ))
    result = sector_forecast.build(tmp_path)
    assert result["status"] == "waiting_data"
    assert not ForecastStore(tmp_path).list()
    monkeypatch.setattr(sector_forecast, "cn_now", lambda: datetime(2026, 9, 9, 20, tzinfo=CN_TZ))
    monkeypatch.setattr(
        prediction_workflow, "readiness", lambda *args: {"ready": False, "reason": "funds missing"}
    )
    assert sector_forecast.build(tmp_path)["summary"]["warning"] == "funds missing"
    assert not ForecastStore(tmp_path).list()


def test_legacy_does_not_enter_published_aggregate(tmp_path):
    import json

    folder = tmp_path / "user_data"
    folder.mkdir()
    raw = json.dumps(
        [
            {
                "id": "old",
                "as_of": "2026-09-08",
                "rows": [],
                "validation_summary": {"strong_hits": 10, "strong_verified": 10},
            }
        ]
    )
    path = folder / "sector_predictions.json"
    path.write_text(raw)
    assert not sector_forecast.records(tmp_path)["snapshots"]
    assert len(sector_forecast.records(tmp_path, origin="legacy")["snapshots"]) == 1
    assert path.read_text() == raw


def test_top5_missing_does_not_pull_in_rank6(tmp_path):
    rows = [dict(classify(evidence(sector=str(i)), {}), prediction_rank=i) for i in range(1, 7)]
    snapshot = {"id": "one", "as_of": "2026-09-07", "origin": "published", "rows": rows}
    store = ForecastStore(tmp_path)
    store.save(snapshot)
    results = [
        {
            "sector": str(i),
            "status": "missing" if i == 1 else "hit",
            "status_label": "x",
            "actual_ret_1d": 0.02,
        }
        for i in range(1, 7)
    ]
    store.append_validation("one", 1, {"verify_date": "2026-09-08", "rows": results})
    summary = sector_forecast.merge_validations(tmp_path, snapshot)["validation_summary"]
    assert summary["top5_verified"] == 4
    assert summary["horizons"]["2"]["top5_verified"] == 0


def test_walk_forward_has_purged_dates_and_untouched_holdout():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(220)]
    folds = walk_forward_folds(days)
    holdout = folds[-1]
    assert holdout["kind"] == "holdout"
    for fold in folds:
        assert days.index(fold["train"][-1]) + 2 < days.index(fold["test"][0])
        assert not set(fold["train"]) & set(fold["test"])
        assert not set(fold["train"]) & set(holdout["test"])


def test_calibration_does_not_use_test_outcomes():
    row = classify(evidence(), {})
    trained = {date(2026, 9, 1): [row]}
    outcomes = {(date(2026, 9, 1), 1, "test"): {"ret_1d": 0.02, "relative_return": 0.01}}
    before = calibration(trained, outcomes, 1)
    outcomes[(date(2026, 9, 8), 1, "test")] = {"ret_1d": -0.9, "relative_return": -0.9}
    assert before == calibration(trained, outcomes, 1)
    assert clean_json({"date": date(2026, 9, 8)}) == {"date": "2026-09-08"}


def test_low_breadth_is_not_rewarded_twice():
    a = classify(evidence(up_ratio=0.2), {})
    b = classify(evidence(up_ratio=0.5), {})
    assert a["component_scores"] == b["component_scores"]
    assert classify(evidence(), {"weak": True})["prediction_level"] == "watch"


def test_previous_launch_is_not_a_new_start_and_parent_excludes_self():
    row = classify(evidence(recently_started=True), {})
    assert row["forecast_kind"] == "continuation"
    rows, _ = sector_forecast.enrich_context(
        [evidence(sector="transport-shipping", flow_acceleration=0.1)]
    )
    assert rows[0]["parent_flow_acceleration"] == 0.0
    rows, _ = sector_forecast.enrich_context(
        [
            evidence(sector="transport-shipping", flow_acceleration=0.1),
            evidence(sector="transport-ports", flow_acceleration=-0.02),
        ]
    )
    assert rows[0]["parent_flow_acceleration"] == -0.02


def test_each_factor_can_be_ablated_independently():
    from app.services.sector_research import VARIANTS, select_variant

    assert len(VARIANTS) == 11
    baseline = classify(evidence(), {})
    for variant, component in [
        ("no_resilience", "relative_strength"),
        ("no_volume", "volume_wake"),
        ("no_leader", "leader_seed"),
        ("no_position", "price_position"),
    ]:
        modified = classify(evidence(), {}, ablation=variant)
        assert modified["component_scores"][component] == 50.0
        assert (
            modified["component_scores"]["fund_turn"] == baseline["component_scores"]["fund_turn"]
        )
        assert isinstance(select_variant([evidence()], {}, variant, 1), list)


def test_funds_use_same_members_and_yuan_contract(tmp_path):
    day = date(2026, 9, 8)
    folder = tmp_path / "moneyflow" / f"date={day}"
    folder.mkdir(parents=True)
    pl.DataFrame(
        [
            dict(
                symbol=s,
                date=day,
                buy_lg_amount=20000.0,
                sell_lg_amount=5000.0,
                buy_elg_amount=8000.0,
                sell_elg_amount=3000.0,
                net_mf_amount=-900000.0,
            )
            for s in ("000001", "000002")
        ]
    ).write_parquet(folder / "part.parquet")
    members = pl.DataFrame({"symbol": ["000001", "000002"], "sector": ["IT", "IT"]})
    bars = pl.DataFrame({"symbol": ["000001"], "date": [day]})
    flow, _ = rotation._load_flow_panel(tmp_path, [day], members, bars)
    assert flow["flow_member_count"].item() == 1
    assert flow["net_1d"].item() == 20000.0
    assert flow["active_net_1d"].item() == -900000.0


def test_generation_is_immutable_and_source_change_is_rejected(tmp_path, monkeypatch):
    from app.market_time import CN_TZ
    from app.services import prediction_workflow

    day = date(2026, 9, 9)
    monkeypatch.setattr(sector_forecast, "cn_now", lambda: datetime(2026, 9, 9, 20, tzinfo=CN_TZ))
    monkeypatch.setattr(prediction_workflow, "readiness", lambda *args: {"ready": True})
    monkeypatch.setattr(sector_forecast, "trading_dates", lambda *args: [day])
    monkeypatch.setattr(sector_forecast, "enrich_context", lambda rows: (rows, {}))
    monkeypatch.setattr(
        rotation,
        "build_sector_rotation",
        lambda *args, **kw: {"as_of": str(day), "rows": [evidence()]},
    )
    monkeypatch.setattr(rotation, "input_signature", lambda *args: [("same", 1)])
    original = sector_forecast.build(tmp_path)
    monkeypatch.setattr(
        rotation, "build_sector_rotation", lambda *args, **kw: {"as_of": str(day), "rows": []}
    )
    assert (
        sector_forecast.build(tmp_path)["rows"][0]["prediction_score"]
        == original["rows"][0]["prediction_score"]
    )
    versions = iter([[("one", 1)], [("two", 2)]])
    monkeypatch.setattr(rotation, "input_signature", lambda *args: next(versions))
    result = sector_forecast.build(tmp_path, origin="replay")
    assert result["status"] == "waiting_data"
    assert not ForecastStore(tmp_path).list("replay")


def test_validation_waits_for_complete_day_and_appends_only(tmp_path, monkeypatch):
    from app.market_time import CN_TZ
    from app.services import prediction_workflow

    day, next_day = date(2026, 9, 8), date(2026, 9, 9)
    store = ForecastStore(tmp_path)
    snapshot = {
        "id": "frozen",
        "as_of": str(day),
        "origin": "published",
        "rows": [classify(evidence(), {})],
    }
    store.save(snapshot)
    monkeypatch.setattr(sector_forecast, "trading_dates", lambda *args: [day, next_day])
    monkeypatch.setattr(sector_forecast, "cn_now", lambda: datetime(2026, 9, 9, 12, tzinfo=CN_TZ))
    sector_forecast.validate(tmp_path)
    assert not store.validations("frozen")
    monkeypatch.setattr(sector_forecast, "cn_now", lambda: datetime(2026, 9, 9, 20, tzinfo=CN_TZ))
    checks = dict.fromkeys(["日K", "技术特征", "核心指数", "复权历史一致性"], True)
    monkeypatch.setattr(prediction_workflow, "readiness", lambda *args: {"checks": checks})
    monkeypatch.setattr(rotation, "input_signature", lambda *args: [("same", 1)])
    actual = evidence(ret_1d=0.02, relative_return=0.015, price_coverage=1.0)
    monkeypatch.setattr(
        rotation,
        "build_sector_rotation",
        lambda *args, **kw: {"as_of": str(next_day), "rows": [actual]},
    )
    sector_forecast.validate(tmp_path)
    sector_forecast.validate(tmp_path)
    assert len(store.validations("frozen")) == 1
    assert store.list()[0] == snapshot
    assert (
        sector_forecast.merge_validations(tmp_path, snapshot)["validation_summary"]["horizons"][
            "1"
        ]["strong_hits"]
        == 1
    )


def test_research_api_is_read_only_and_serializes_dates(tmp_path):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.short_term_ai import router

    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.include_router(router)
    with TestClient(app) as client:
        assert client.get("/api/short-term-ai/sector-research").json()["status"] == "empty"
        assert (
            client.get("/api/short-term-ai/sector-predictions?origin=published").json()["snapshots"]
            == []
        )
        assert client.get("/api/short-term-ai/sector-predictions?origin=unknown").status_code == 422
    assert not ForecastStore(tmp_path).path.exists()
