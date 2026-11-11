import asyncio
import time
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import stockformer
from app.services.pipeline_jobs import JobStore


def test_stockformer_train_runs_as_background_job(monkeypatch, tmp_path):
    store = JobStore(store_dir=tmp_path / "jobs")
    monkeypatch.setattr(stockformer, "job_store", store)
    monkeypatch.setattr(stockformer, "try_acquire_run_slot", lambda _job_id: True)
    monkeypatch.setattr(stockformer, "release_run_slot", lambda _job_id: None)

    def fake_train(data_dir, config, *, activate=False, on_progress=None):
        assert data_dir == tmp_path
        assert config.max_symbols == 20
        assert activate is False
        on_progress("fit", 50, "测试训练中")
        return {"model": {"model_id": "sf_test"}, "predictions": {"rows": []}}

    monkeypatch.setattr(stockformer, "train_model", fake_train)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repo=repo)))

    response = asyncio.run(stockformer.train(
        stockformer.TrainRequest(
            start=date(2025, 1, 1),
            end=date(2025, 6, 30),
            max_symbols=20,
        ),
        request,
    ))

    assert set(response) == {"job_id", "reused"}
    assert response["reused"] is False
    for _ in range(100):
        job = store.get(response["job_id"])
        if job and job["status"] == "succeeded":
            break
        time.sleep(0.01)
    assert job["status"] == "succeeded"
    assert job["result"]["model"]["model_id"] == "sf_test"
    assert any(item["stage"] == "fit" for item in job["log"])


def test_inference_job_returns_immediately_and_reuses_predictions(monkeypatch, tmp_path):
    import threading

    store = JobStore(store_dir=tmp_path / "jobs")
    monkeypatch.setattr(stockformer, "job_store", store)
    monkeypatch.setattr(stockformer, "try_acquire_run_slot", lambda _: True)
    released = []
    monkeypatch.setattr(stockformer, "release_run_slot", released.append)
    allow = threading.Event()
    predictions = {"model": {"model_id": "sf_test"}, "rows": [{"symbol": "600000"}, {"symbol": "000001"}]}

    def predict(data_dir, **kwargs):
        assert data_dir == tmp_path
        assert kwargs["write_artifact"] is False
        assert kwargs["limit"] == 0
        assert allow.wait(5)
        return predictions

    def multifactor(data_dir, **kwargs):
        assert kwargs["prediction_result"] is predictions
        return {"rows": [], "signal_status": "no_signal"}

    monkeypatch.setattr(stockformer, "predict", predict)
    monkeypatch.setattr(stockformer, "multifactor_select", multifactor)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)))))
    response = asyncio.run(stockformer.model_inference_job(
        stockformer.PredictRequest(model_id="sf_test", limit=1), request))
    assert store.get(response["job_id"])["status"] in {"pending", "running"}
    allow.set()
    for _ in range(100):
        job = store.get(response["job_id"])
        if job["status"] == "succeeded" and released:
            break
        time.sleep(0.01)
    assert job["status"] == "succeeded"
    assert job["result"]["predictions"]["rows"] == [{"symbol": "600000"}]
    assert predictions["rows"] == [{"symbol": "600000"}, {"symbol": "000001"}]
    assert job["result"]["model_id"] == "sf_test"
    assert job["result"]["multifactor"]["signal_status"] == "no_signal"
    assert released == [response["job_id"]]


def test_inference_job_does_not_reuse_an_unrelated_job(monkeypatch, tmp_path):
    store = JobStore(store_dir=tmp_path / "jobs")
    store.create(long_running=True)
    monkeypatch.setattr(stockformer, "job_store", store)
    with pytest.raises(HTTPException) as error:
        asyncio.run(stockformer.model_inference_job(stockformer.PredictRequest(), None))
    assert error.value.status_code == 409


def test_inference_failure_releases_capacity(monkeypatch, tmp_path):
    store = JobStore(store_dir=tmp_path / "jobs")
    monkeypatch.setattr(stockformer, "job_store", store)
    monkeypatch.setattr(stockformer, "try_acquire_run_slot", lambda _: True)
    released = []
    monkeypatch.setattr(stockformer, "release_run_slot", released.append)
    monkeypatch.setattr(stockformer, "predict", lambda *_, **__: {"error": "missing features"})
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)))))
    response = asyncio.run(stockformer.model_inference_job(stockformer.PredictRequest(), request))
    for _ in range(100):
        job = store.get(response["job_id"])
        if job["status"] == "failed" and released:
            break
        time.sleep(0.01)
    assert job["status"] == "failed"
    assert "missing features" in job["error"]
    assert released == [response["job_id"]]


@pytest.mark.parametrize("handler,request_type", [
    (stockformer.model_strict_oos, stockformer.StrictOOSRequest),
    (stockformer.model_portfolio_research, stockformer.PortfolioResearchRequest),
])
def test_validation_does_not_adopt_an_inference_job(monkeypatch, tmp_path, handler, request_type):
    store = JobStore(store_dir=tmp_path / "jobs")
    store.create(long_running=True)
    monkeypatch.setattr(stockformer, "job_store", store)
    with pytest.raises(HTTPException) as error:
        asyncio.run(handler(request_type(), None))
    assert error.value.status_code == 409
