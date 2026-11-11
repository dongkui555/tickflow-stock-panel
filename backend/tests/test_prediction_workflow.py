from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from app.services import prediction_workflow as w

NOW = datetime(2026, 9, 8, 19, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
DAY = NOW.date()


def write(root, table, day, symbols, **columns):
    folder = root / table / f"date={day}"
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"symbol": symbols, "date": [day] * len(symbols),
                  **{k: [v] * len(symbols) for k, v in columns.items()}}).write_parquet(folder / "part.parquet")


def seed(root):
    from app.services.index_const import CORE_INDEX_SYMBOLS
    prices = dict(open=10., high=11., low=9., close=10., volume=100., amount=1000.)
    for table, day, symbols in [
        ("kline_daily", DAY - timedelta(days=1), ["000001.SZ"]),
        ("kline_daily", DAY, ["000001.SZ"]),
        ("kline_daily_enriched", DAY, ["000001.SZ"]),
        ("kline_index_daily", DAY, list(CORE_INDEX_SYMBOLS)),
    ]:
        write(root, table, day, symbols, **prices)
    write(root, "moneyflow", DAY, ["000001.SZ"], buy_lg_amount=0., sell_lg_amount=1.,
          buy_elg_amount=0., sell_elg_amount=0.)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(w, "config", lambda: {"enabled": True, "notify": True, "bot_configured": True,
                                            "bot_enabled": True, "chat_id": "user", "chat_bot_id": "bot"})
    monkeypatch.setattr("app.services.preferences.get_wecom_bot_id", lambda: "bot")
    return SimpleNamespace(status=lambda: {"connected": True}, send_markdown=lambda *args: {"status": "sent"})


def test_readiness_checks_actual_rows_and_accepts_zero_flow(tmp_path):
    seed(tmp_path)
    assert w.readiness(tmp_path, DAY)["ready"]
    write(tmp_path, "moneyflow", DAY, ["000001.SZ"], buy_lg_amount=float("nan"),
          sell_lg_amount=0., buy_elg_amount=0., sell_elg_amount=0.)
    assert not w.readiness(tmp_path, DAY)["ready"]
    assert not w.readiness(tmp_path, DAY + timedelta(days=1))["ready"]


@pytest.mark.parametrize('symbol', ['000001.SZ', '600000.SH', '920701.BJ'])
@pytest.mark.parametrize('raw_volume', [100., 10000.], ids=['legacy_lots', 'shares'])
def test_readiness_compares_enriched_in_share_units_without_writing(tmp_path, symbol, raw_volume):
    seed(tmp_path)
    prices = dict(open=10., high=11., low=9., close=10., volume=raw_volume, amount=100000.)
    write(tmp_path, 'kline_daily', DAY - timedelta(days=1), [symbol], **prices)
    write(tmp_path, 'kline_daily', DAY, [symbol], **prices)
    # Enriched OHLC can be adjusted; reference volume is always shares.
    write(tmp_path, 'kline_daily_enriched', DAY, [symbol], open=5., high=5.5, low=4.5,
          close=5., raw_close=10., volume=10000., amount=100000.)
    write(tmp_path, 'moneyflow', DAY, [symbol], buy_lg_amount=0., sell_lg_amount=0.,
          buy_elg_amount=0., sell_elg_amount=0.)
    paths = list(tmp_path.rglob('*.parquet'))
    before = {p: p.read_bytes() for p in paths}
    result = w.readiness(tmp_path, DAY)
    assert result['ready']
    assert result['counts']['enriched'] == 1
    assert result['missing']['enriched'] == []
    assert before == {p: p.read_bytes() for p in paths}


@pytest.mark.parametrize('field,value', [('close', 10.05), ('amount', 100001.),
                                         ('volume', 101.), ('volume', 20.)])
def test_readiness_volume_normalization_does_not_hide_reference_changes(tmp_path, field, value):
    seed(tmp_path)
    raw = dict(open=10., high=11., low=9., close=10., volume=100., amount=100000.)
    write(tmp_path, 'kline_daily', DAY, ['000001.SZ'], **raw)
    write(tmp_path, 'kline_daily_enriched', DAY, ['000001.SZ'], **{**raw, 'volume': 10000.}, raw_close=10.)
    assert w.readiness(tmp_path, DAY)['ready']
    write(tmp_path, 'kline_daily', DAY, ['000001.SZ'], **{**raw, field: value})
    result = w.readiness(tmp_path, DAY)
    assert not result['ready']
    assert result['missing']['enriched'] == ['000001']


def test_pending_adjustment_blocks_prediction_and_delivery(configured, monkeypatch, tmp_path):
    from app.services import maintenance_queue

    seed(tmp_path)
    snapshot = {"id": "before-adjustment", "as_of": str(DAY), "rows": [],
                "readiness": w.readiness(tmp_path, DAY)}
    calls = []
    monkeypatch.setattr("app.services.stockformer.multifactor_select", lambda *a, **k: calls.append("predict"))
    monkeypatch.setattr(configured, "send_markdown", lambda *a: calls.append("notify"))
    maintenance_queue.enqueue(tmp_path, "adjustment", ["000001.SZ"])

    assert not w.readiness(tmp_path, DAY)["checks"]["复权历史一致性"]
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    assert w.tick(repo, now=NOW, trading_day=True, bot_service=configured)["status"] == "waiting_data"
    assert w._notify(tmp_path, snapshot, NOW, configured)["status"] == "blocked"
    assert calls == []


def test_invalid_prices_block_readiness(tmp_path):
    seed(tmp_path)
    write(tmp_path, "kline_daily", DAY, ["000001.SZ"], open=10., high=9., low=8., close=10., volume=100., amount=1000.)
    assert not w.readiness(tmp_path, DAY)["ready"]


def test_waits_for_close_and_data_without_predicting(configured, tmp_path):
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    assert w.tick(repo, now=NOW.replace(hour=15), trading_day=True)["status"] == "waiting_close"
    assert w.tick(repo, now=NOW, trading_day=False)["status"] == "skipped"
    assert w.tick(repo, now=NOW, trading_day=True)["status"] == "waiting_data"
    assert not (w._root(tmp_path) / "latest.json").exists()


def test_snapshot_and_delivery_are_durable_and_deduplicated(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    calls = []
    monkeypatch.setattr("app.services.stockformer.multifactor_select", lambda *a, **k: {
        "as_of": str(DAY), "rows": [{"symbol": "000001.SZ", "name": "Test", "pred_ret_1d": .02}], "model": {"model_id": "m1"}})
    monkeypatch.setattr(configured, "send_markdown", lambda *a: calls.append(a) or {"status": "sent"})
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    first = w.tick(repo, now=NOW, trading_day=True, bot_service=configured)
    second = w.tick(repo, now=NOW + timedelta(minutes=31), trading_day=True, bot_service=configured)
    assert first["snapshot_id"] == second["snapshot_id"]
    assert second["notification"]["status"] == "sent"
    assert len(calls) == 1
    assert len(list((w._root(tmp_path) / "snapshots").glob("*.json"))) == 1


def test_prediction_date_mismatch_is_not_published(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    monkeypatch.setattr("app.services.stockformer.multifactor_select", lambda *a, **k: {"as_of": "2026-09-07", "rows": []})
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    assert w.tick(repo, now=NOW, trading_day=True)["status"] == "prediction_failed"
    assert not (w._root(tmp_path) / "latest.json").exists()


def test_valid_no_signal_is_archived_and_never_notified(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    calls = []
    monkeypatch.setattr("app.services.stockformer.multifactor_select", lambda *a, **k: {
        "as_of": str(DAY), "rows": [], "total": 0, "universe_total": 954,
        "signal_status": "no_signal", "signal_weights": {}, "model": {"model_id": "frozen"}})
    monkeypatch.setattr(configured, "send_markdown", lambda *a: calls.append(a))
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    result = w.tick(repo, now=NOW, trading_day=True, bot_service=configured)
    assert result["status"] == "predicted"
    snapshot = w._read(w._root(tmp_path) / "latest.json")
    assert snapshot["as_of"] == str(DAY) and snapshot["rows"] == []
    assert snapshot["no_candidates"] is True
    assert snapshot["empty_reason"] == "no_validated_horizon_weights"
    assert snapshot["universe_total"] == 954
    assert result["notification"]["status"] == "no_candidates" and calls == []
    assert len(list((w._root(tmp_path) / "snapshots").glob("*.json"))) == 1
    workflow = {"snapshot": snapshot, "premarket": {"date": "2026-09-09",
        "snapshot_id": snapshot["id"], "block_new_position": False}}
    gated = w.apply_execution_gate({"rows": [{"buy_confirmation_status": "confirmed"}]},
                                 workflow, NOW.replace(day=9, hour=10))
    assert gated["workflow"]["execution_allowed"] is False


def test_unexplained_empty_result_is_still_a_failure(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    monkeypatch.setattr("app.services.stockformer.multifactor_select", lambda *a, **k: {
        "as_of": str(DAY), "rows": [], "total": 0})
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    assert w.tick(repo, now=NOW, trading_day=True)["status"] == "prediction_failed"
    assert not (w._root(tmp_path) / "latest.json").exists()


def test_failed_notification_retries_without_claiming_sent(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    attempts = []
    monkeypatch.setattr(configured, "send_markdown", lambda *a: attempts.append(1) or {"status": "failed"})
    snapshot = {"id": "test", "as_of": str(DAY), "rows": [], "readiness": w.readiness(tmp_path, DAY)}
    assert w._notify(tmp_path, snapshot, NOW, configured)["status"] == "failed"
    w._notify(tmp_path, snapshot, NOW + timedelta(minutes=1), configured)
    assert len(attempts) == 1
    w._notify(tmp_path, snapshot, NOW + timedelta(minutes=31), configured)
    assert len(attempts) == 2


def test_notification_blocks_stale_or_changed_data(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    calls = []
    monkeypatch.setattr(configured, "send_markdown", lambda *a: calls.append(a) or {"status": "sent"})
    snapshot = {"id": "test", "as_of": str(DAY), "rows": [], "readiness": w.readiness(tmp_path, DAY)}
    assert w._notify(tmp_path, snapshot, NOW + timedelta(days=1), configured)["status"] == "blocked"
    write(tmp_path, "moneyflow", DAY, ["000001.SZ"], buy_lg_amount=5., sell_lg_amount=1., buy_elg_amount=0., sell_elg_amount=0.)
    assert w._notify(tmp_path, snapshot, NOW, configured)["status"] == "blocked"
    assert calls == []


def test_changed_inputs_during_prediction_are_not_published(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    def predict(*args, **kwargs):
        write(tmp_path, "moneyflow", DAY, ["000001.SZ"], buy_lg_amount=5., sell_lg_amount=1., buy_elg_amount=0., sell_elg_amount=0.)
        return {"as_of": str(DAY), "rows": [{"symbol": "000001.SZ"}]}
    monkeypatch.setattr("app.services.stockformer.multifactor_select", predict)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    assert w.tick(repo, now=NOW, trading_day=True)["status"] == "waiting_data"
    assert not (w._root(tmp_path) / "latest.json").exists()


def test_no_review_or_outside_session_never_confirms_entry():
    payload = {"rows": [{"decision": "候选", "buy_confirmation_status": "confirmed"}]}
    w.apply_execution_gate(payload, {}, NOW)
    assert payload["rows"][0]["buy_confirmation_status"] == "blocked"
    assert payload["rows"][0]["position_multiplier"] == 0


def test_review_does_not_override_existing_intraday_rejection():
    now = NOW.replace(day=9, hour=10)
    workflow = {"snapshot": {"id": "one", "as_of": str(DAY)},
                "premarket": {"date": "2026-09-09", "snapshot_id": "one", "block_new_position": False}}
    result = w.apply_execution_gate({"rows": [{"buy_confirmation_status": "waiting"}]}, workflow, now)
    assert result["workflow"]["execution_allowed"]
    assert result["rows"][0]["buy_confirmation_status"] == "waiting"


def test_new_input_version_keeps_old_snapshot(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    monkeypatch.setattr("app.services.stockformer.multifactor_select", lambda *a, **k: {
        "as_of": str(DAY), "rows": [{"symbol": "000001.SZ"}]} )
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    first = w.tick(repo, now=NOW, trading_day=True, bot_service=configured)
    write(tmp_path, "moneyflow", DAY, ["000001.SZ"], buy_lg_amount=5., sell_lg_amount=1., buy_elg_amount=0., sell_elg_amount=0.)
    second = w.tick(repo, now=NOW + timedelta(minutes=31), trading_day=True, bot_service=configured)
    assert first["snapshot_id"] != second["snapshot_id"]
    assert len(list((w._root(tmp_path) / "snapshots").glob("*.json"))) == 2


def test_stale_global_data_and_missing_news_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.global_risk.build_global_risk", lambda *a, **k: {
        "data_success": True, "cache_fallback": True, "updated_at": "2026-09-08T08:00:00"})
    result = w._premarket(tmp_path, {"id": "x", "as_of": "2026-09-07", "generated_at": "2026-09-07T19:30:00+08:00"}, NOW.replace(hour=8), None)
    assert result["block_new_position"]
    assert result["status"] == "observe"


def test_wecom_non_json_200_is_not_delivery_success(monkeypatch):
    from app.services import webhook_adapter
    class Response:
        status_code = 200
        def json(self):
            raise ValueError("not JSON")
    monkeypatch.setattr("httpx.post", lambda *a, **k: Response())
    assert webhook_adapter._post_wecom("https://example.invalid", {}) is False


def test_premarket_excludes_future_news_and_rejects_risk_events(monkeypatch, tmp_path):
    seed(tmp_path)
    now = NOW.replace(day=9, hour=8)
    monkeypatch.setattr("app.services.global_risk.build_global_risk", lambda *a, **k: {
        "data_success": True, "updated_at": now.isoformat(), "recommendation_adjustment": {"block_new_position": False}})
    write(tmp_path, "news", now.date(), ["000001.SZ"], published_at="2026-09-09T07:00:00+08:00", market="cn", title="行业公告")
    snapshot = {"id": "one", "as_of": str(DAY), "generated_at": NOW.isoformat()}
    review = w._premarket(tmp_path, snapshot, now, None)
    assert not review["block_new_position"]
    write(tmp_path, "news", now.date(), ["000001.SZ"], published_at="2026-09-09T09:00:00+08:00", market="cn", title="行业公告")
    assert w._premarket(tmp_path, snapshot, now, None)["events"] == []
    write(tmp_path, "news", now.date(), ["000001.SZ"], published_at="2026-09-09T07:00:00+08:00", market="cn", title="公司立案调查")
    assert w._premarket(tmp_path, snapshot, now, None)["block_new_position"]


def test_workflow_api_read_and_config(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.short_term_ai import router
    prefs = {}
    monkeypatch.setattr("app.services.preferences.load", lambda: prefs)
    monkeypatch.setattr("app.services.preferences.save", lambda p: prefs.update(p))
    monkeypatch.setattr("app.services.preferences.get_wecom_webhook_url", lambda: "")
    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.include_router(router)
    client = TestClient(app)
    assert client.get("/api/short-term-ai/workflow").json()["config"]["enabled"]
    result = client.post("/api/short-term-ai/workflow/config", json={"enabled": False, "notify": False})
    assert result.status_code == 200
    assert not result.json()["config"]["enabled"]
    assert not result.json()["config"]["notify"]


def test_missing_data_is_repaired_before_prediction(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    write(tmp_path, "moneyflow", DAY, ["000001.SZ"], buy_lg_amount=float("nan"),
          sell_lg_amount=0., buy_elg_amount=0., sell_elg_amount=0.)
    calls = []
    def repair(ready):
        calls.append("repair")
        assert not ready["ready"]
        seed(tmp_path)
        return {"status": "checked", "datasets": []}
    def predict(*args, **kwargs):
        calls.append("predict")
        return {"as_of": str(DAY), "rows": [{"symbol": "000001.SZ"}]}
    monkeypatch.setattr("app.services.stockformer.multifactor_select", predict)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    result = w.tick(repo, now=NOW, trading_day=True, repair_data=repair, bot_service=configured)
    assert calls == ["repair", "predict"]
    assert result["notification"]["status"] == "sent"


def test_failed_repair_waits_until_persisted_retry(configured, tmp_path):
    calls = []
    retry_at = (NOW + timedelta(minutes=15)).isoformat()
    def repair(ready):
        calls.append(ready)
        return {"status": "retry", "next_retry_at": retry_at, "datasets": []}
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    result = w.tick(repo, now=NOW, trading_day=True, repair_data=repair)
    assert result["status"] == "waiting_data"
    assert result["next_check_at"] == retry_at
    w.tick(repo, now=NOW + timedelta(minutes=5), trading_day=True, repair_data=repair)
    assert len(calls) == 1
    assert not (w._root(tmp_path) / "latest.json").exists()


def test_retry_only_fetches_missing_funds_and_keeps_backoff(monkeypatch, tmp_path):
    from app.services import close_data_sync
    seed(tmp_path)
    write(tmp_path, "moneyflow", DAY, ["000001.SZ"], buy_lg_amount=float("nan"),
          sell_lg_amount=0., buy_elg_amount=0., sell_elg_amount=0.)
    calls = []
    def fetch(repo, caps, dataset, day, missing, emit):
        calls.append((dataset, day, missing))
        raise TimeoutError("contains secret URL, never expose")
    monkeypatch.setattr(close_data_sync, "_fetch_missing", fetch)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path),
                           refresh_cache=lambda: None, refresh_index_views=lambda: None)
    monkeypatch.setattr("app.api.data.invalidate_storage_cache", lambda: None)
    first = w.retry_missing_data(repo, None, DAY, now=NOW)
    assert calls == [("moneyflow", DAY, {"000001"})]
    assert first["datasets"][0]["error"] == "TimeoutError"
    assert "secret" not in str(first)
    w.retry_missing_data(repo, None, DAY, now=NOW + timedelta(minutes=1))
    assert len(calls) == 1
    second = w.retry_missing_data(repo, None, DAY, now=NOW + timedelta(minutes=6))
    assert len(calls) == 2
    assert second["datasets"][0]["attempts"] == 2
    assert second["next_retry_at"] == (NOW + timedelta(minutes=21)).isoformat()


def test_retry_rechecks_previously_succeeded_sync(monkeypatch, tmp_path):
    from app.services import close_data_sync
    seed(tmp_path)
    write(tmp_path, "moneyflow", DAY, ["000001.SZ"], buy_lg_amount=float("nan"),
          sell_lg_amount=0., buy_elg_amount=0., sell_elg_amount=0.)
    close_data_sync._save(tmp_path, {str(DAY): {"moneyflow": {"status": "succeeded", "attempts": 10}}})
    calls = []
    def fetch(*args):
        calls.append(1)
        seed(tmp_path)
        return 1
    monkeypatch.setattr(close_data_sync, "_fetch_missing", fetch)
    monkeypatch.setattr("app.api.data.invalidate_storage_cache", lambda: None)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path),
                           refresh_cache=lambda: None, refresh_index_views=lambda: None)
    result = w.retry_missing_data(repo, None, DAY, now=NOW)
    assert calls == [1]
    assert result["datasets"][0]["status"] == "succeeded"
    assert w.readiness(tmp_path, DAY)["ready"]


def test_daily_retry_requests_only_missing_symbols_for_target_day(monkeypatch, tmp_path):
    from app.services import kline_sync
    seed(tmp_path)
    prices = dict(open=10., high=11., low=9., close=10., volume=100., amount=1000.)
    write(tmp_path, "kline_daily", DAY - timedelta(days=1), ["000001.SZ", "600000.SH"], **prices)
    calls = []
    def fetch(symbols, repo, caps, **kwargs):
        calls.append((symbols, kwargs))
        return 0
    monkeypatch.setattr(kline_sync, "sync_and_persist_daily_batch", fetch)
    monkeypatch.setattr("app.api.data.invalidate_storage_cache", lambda: None)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path),
                           refresh_cache=lambda: None, refresh_index_views=lambda: None)
    w.retry_missing_data(repo, None, DAY, now=NOW)
    assert calls[0][0] == ["600000"]
    assert calls[0][1]["start_date"].date() == DAY
    assert calls[0][1]["end_date"].date() == DAY
    assert not calls[0][1].get("replace_existing", False)


def test_date_pipeline_does_not_rewrite_history_or_include_future(monkeypatch, tmp_path):
    from app.enriched_generation import get_enriched_generation
    from app.indicators import pipeline
    seed(tmp_path)
    prices = dict(open=10., high=11., low=9., close=10., volume=100., amount=1000.)
    write(tmp_path, "kline_daily", DAY + timedelta(days=1), ["000001.SZ"], **prices)
    write(tmp_path, "kline_daily_enriched", DAY - timedelta(days=1), ["000001.SZ"], **prices)
    old = tmp_path / "kline_daily_enriched" / f"date={DAY - timedelta(days=1)}" / "part.parquet"
    before = old.read_bytes()
    seen = []
    def compute(raw, **kwargs):
        seen.extend(raw["date"].to_list())
        return raw
    monkeypatch.setattr(pipeline, "compute_enriched", compute)
    assert pipeline.run_date_pipeline(tmp_path, DAY) == 1
    assert max(seen) == DAY
    assert old.read_bytes() == before
    assert get_enriched_generation(tmp_path, "stock")
    assert not (tmp_path / "kline_daily_enriched" / f"date={DAY + timedelta(days=1)}").exists()


def test_date_pipeline_real_computation(tmp_path):
    from app.indicators.pipeline import run_date_pipeline
    seed(tmp_path)
    assert run_date_pipeline(tmp_path, DAY) == 1
    frame = w._frame(tmp_path, "kline_daily_enriched", DAY)
    assert frame["close"].to_list() == [10.]
    assert frame["raw_close"].to_list() == [10.]


def test_intraday_snapshot_is_not_a_completed_daily_record(tmp_path):
    seed(tmp_path)
    prices = dict(open=10., high=11., low=9., close=10., volume=100., amount=1000.)
    write(tmp_path, "kline_daily", DAY, ["000001.SZ"], **prices,
          quote_ts=int(NOW.replace(hour=11).timestamp() * 1000))
    result = w.readiness(tmp_path, DAY)
    assert not result["ready"]
    assert result["missing"]["daily"] == ["000001"]
    write(tmp_path, "kline_daily", DAY, ["000001.SZ"], **prices,
          quote_ts=int(NOW.replace(hour=15).timestamp() * 1000))
    assert w.readiness(tmp_path, DAY)["ready"]


def test_old_enriched_must_be_rebuilt_after_raw_update(tmp_path):
    seed(tmp_path)
    write(tmp_path, "kline_daily_enriched", DAY, ["000001.SZ"],
          open=10., high=11., low=9., close=10., volume=100., amount=1000., raw_close=9.9)
    result = w.readiness(tmp_path, DAY)
    assert not result["ready"]
    assert result["missing"]["enriched"] == ["000001"]


def test_unknown_bot_delivery_is_not_automatically_resent(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    calls = []
    monkeypatch.setattr(configured, "send_markdown", lambda *args: calls.append(args) or {"status": "unknown"})
    snapshot = {"id": "test", "as_of": str(DAY), "rows": [], "readiness": w.readiness(tmp_path, DAY)}
    assert w._notify(tmp_path, snapshot, NOW, configured)["status"] == "unknown"
    assert w._notify(tmp_path, snapshot, NOW + timedelta(minutes=31), configured)["status"] == "unknown"
    assert len(calls) == 1


def test_missing_recipient_or_disconnected_bot_never_uses_webhook(configured, monkeypatch, tmp_path):
    seed(tmp_path)
    monkeypatch.setattr("app.services.webhook_adapter.send_wecom_markdown", lambda *args: pytest.fail("no webhook fallback"))
    snapshot = {"id": "test", "as_of": str(DAY), "rows": [], "readiness": w.readiness(tmp_path, DAY)}
    assert w._notify(tmp_path, snapshot, NOW)["status"] == "not_connected"
    cfg = w.config()
    cfg["chat_id"] = ""
    monkeypatch.setattr(w, "config", lambda: cfg)
    assert w._notify(tmp_path, snapshot, NOW, configured)["status"] == "missing_recipient"


def test_recipient_api_accepts_only_explicitly_selected_known_single_chat(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.short_term_ai import router
    prefs = {}
    monkeypatch.setattr("app.services.preferences.load", lambda: prefs)
    monkeypatch.setattr("app.services.preferences.save", prefs.update)
    monkeypatch.setattr("app.services.preferences.get_wecom_bot_id", lambda: "bot")
    app = FastAPI()
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.state.wecom_bot_service = SimpleNamespace(status=lambda: {"connected": True, "chats": [
        {"chatid": "user1", "kind": "single"}, {"chatid": "group1", "kind": "group"}]})
    app.include_router(router)
    client = TestClient(app)
    for target in ("unknown", "group1"):
        assert client.post("/api/short-term-ai/workflow/config", json={"chat_id": target}).status_code == 400
    assert client.post("/api/short-term-ai/workflow/config", json={"chat_id": "user1"}).status_code == 200
    assert prefs["prediction_workflow_chat_id"] == "user1"
    assert prefs["prediction_workflow_chat_bot_id"] == "bot"
