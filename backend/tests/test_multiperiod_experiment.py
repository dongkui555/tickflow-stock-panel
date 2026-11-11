from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from app.services.multiperiod_experiment import MINUTE_FEATURES, MultiPeriodConfig, daily_minute_features, evaluate_multiperiod
from test_minute_timeframes import session


def fixture():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(400)
            if (date(2025, 1, 1) + timedelta(days=i)).weekday() < 5][:260]
    rows, features = [], []
    for j in range(20):
        for i, day in enumerate(days):
            p = 10 + j + i * .015 + np.sin(i * .3 + j) * .2
            rows.append({"symbol": f"{j + 1:06}", "date": day, "open": p,
                         "close": p * 1.002, "high": p * 1.01, "low": p * .99,
                         "volume": 10000. + i, "amount": p * 10000.})
            features.append({"symbol": f"{j + 1:06}", "date": day,
                             **{c: float(np.sin(i + j + k) * .01) for k, c in enumerate(MINUTE_FEATURES)}})
    config = MultiPeriodConfig(days[0], days[65], days[160], days[205], days[-1], max_symbols=20)
    return pl.DataFrame(rows), pl.DataFrame(features), config


def test_minute_feature_complete_session_required():
    day = date(2026, 9, 4)
    features = daily_minute_features(session(), day)
    assert features.height == 1
    assert set(MINUTE_FEATURES).issubset(features.columns)
    assert features["m60_last_return"][0] == features["m15_last_return"][0]
    assert daily_minute_features(session().head(239), day).is_empty()


def test_paired_masks_purge_and_no_future_fit():
    panel, features, config = fixture()
    report, predictions = evaluate_multiperiod(panel, features, config)
    assert report["production_model_changed"] is False
    for horizon in report["horizons"].values():
        assert len(horizon["metrics"]["test"]) == 4
        assert len({v["rows"] for v in horizon["metrics"]["test"].values()}) == 1
    assert predictions.filter(pl.col("split") == "train")["label_end"].max() < config.validation_start
    assert predictions.filter(pl.col("split") == "validation")["label_end"].max() < config.test_start
    altered = features.with_columns([
        pl.when(pl.col("date") >= config.validation_start).then(pl.col(c) + 100).otherwise(pl.col(c)).alias(c)
        for c in MINUTE_FEATURES])
    other, _ = evaluate_multiperiod(panel, altered, config)
    for h in report["horizons"]:
        assert report["horizons"][h]["models"] == other["horizons"][h]["models"]


def test_missing_minute_features_exclude_all_arms_equally():
    panel, features, config = fixture()
    features = features.filter(pl.col("symbol") != "000001")
    report, _ = evaluate_multiperiod(panel, features, config)
    assert report["coverage"]["common_ratio"] < 1
    for v in report["horizons"]["5"]["metrics"]["test"].values():
        assert v["symbols"] == 19


def test_chart_api_uses_local_data_and_validates_period(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api import kline
    class Repo:
        def resolve_asset_type(self, symbol):
            return "stock"
        def get_minute_range(self, symbols, start, end, asset_type):
            assert asset_type == "stock"
            return session()
    app = FastAPI()
    app.include_router(kline.router)
    app.state.repo = Repo()
    monkeypatch.setattr(kline, "cn_now", lambda: datetime(2026, 9, 4, 15, 1))
    client = TestClient(app)
    assert client.get("/api/kline/minute-bars?symbol=000001&period=7").status_code == 422
    response = client.get("/api/kline/minute-bars?symbol=000001&period=60").json()
    assert len(response["rows"]) == 4
    assert response["source"] == "local"
    assert response["rows"][0]["date"].endswith("10:30:00")
