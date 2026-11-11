from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from app.services import stockformer_daily_shadow as shadow
from app.strategy import paper

NOW = datetime(2026, 10, 20, 19, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
REAL_SETTLE = shadow._settle


@pytest.fixture
def setup(tmp_path, monkeypatch):
    meta = {"model_id": "research", "signal_weights": {}, "training_universe": ["000001"]}
    monkeypatch.setattr(shadow, "_frozen_assets", lambda *a: {"model": "immutable"})
    monkeypatch.setattr(shadow.sf, "_model_meta", lambda *a: meta)
    monkeypatch.setattr(shadow, "cn_now", lambda: NOW)
    monkeypatch.setattr(shadow, "_input_version", lambda *a: "inputs-v1")
    monkeypatch.setattr(shadow, "readiness", lambda *a: {"ready": True, "reason": "ready"})
    monkeypatch.setattr(shadow, "_settle", lambda *a: {"filled": 0, "trade_count": 0})
    monkeypatch.setattr(paper, "read_daily_bar", lambda *a: {"open": 10., "close": 10.})
    monkeypatch.setattr(paper, "_now_iso", lambda: NOW.isoformat())
    frame = pl.DataFrame({
        "symbol": ["000001"], "date": [NOW.date()], "close": [10.],
        "pred_ret_raw_5d": [.02], "pred_ret_5d": [.01], "net_margin_score": [.009],
        "_prediction_pass": [True], "_net_margin_pass": [True],
        "_forecast_pass": [True], "_entry_rules_pass": [True], "_market_gate_pass": [True],
    })
    monkeypatch.setattr(shadow, "_compute", lambda *a: ({"signal_status": "no_signal", "signal_weights": {}}, frame, frame))
    shadow.configure(tmp_path, "research", start_date=NOW.date())
    return tmp_path, monkeypatch, frame


def test_zero_signal_is_immutable_daily_evidence_without_orders(setup):
    root, monkeypatch, _ = setup
    first = shadow.tick(root, now=NOW, trading_day=True)
    assert first["status"] == "recorded"
    assert first["observation"]["no_trade_reason"] == "model_no_signal"
    assert first["observation"]["selected_count"] == 0
    assert first["observation"]["prediction_pass"] == 1
    assert first["generated_at"] == NOW.isoformat()
    assert paper.load_orders(root, first["account_id"]) == []
    monkeypatch.setattr(shadow, "_compute", lambda *a: pytest.fail("duplicate inference"))
    assert shadow.tick(root, now=NOW + timedelta(minutes=10), trading_day=True) == first
    assert shadow.status(root)["recorded_days"] == 1


def test_intraday_and_holiday_do_not_predict(setup):
    root, monkeypatch, _ = setup
    monkeypatch.setattr(shadow, "_compute", lambda *a: pytest.fail("premature inference"))
    assert shadow.tick(root, now=NOW.replace(hour=10), trading_day=True)["status"] == "waiting_close"
    assert shadow.tick(root, now=NOW, trading_day=False)["status"] == "non_trading_day"
    assert shadow.tick(root, now=NOW, trading_day=None)["status"] == "calendar_unknown"


def test_missing_data_is_persisted_and_retried_without_empty_success(setup):
    root, monkeypatch, _ = setup
    monkeypatch.setattr(shadow, "readiness", lambda *a: {"ready": False, "reason": "missing minutes"})
    result = shadow.tick(root, now=NOW, trading_day=True)
    assert result["status"] == "waiting_data"
    assert not shadow.due(root, NOW + timedelta(minutes=1))
    assert shadow.due(root, NOW + timedelta(minutes=16))
    assert shadow.status(root)["recorded_days"] == 0


def test_changed_inputs_or_model_do_not_publish_or_trade(setup):
    root, monkeypatch, _ = setup
    versions = iter(["before", "after"])
    monkeypatch.setattr(shadow, "_input_version", lambda *a: next(versions))
    assert shadow.tick(root, now=NOW, trading_day=True)["status"] == "inputs_changed"
    assert shadow.status(root)["recorded_days"] == 0
    monkeypatch.setattr(shadow, "_frozen_assets", lambda *a: {"model": "changed"})
    assert shadow.tick(root, now=NOW, trading_day=True)["status"] == "blocked_protocol_changed"


def test_prediction_finishing_next_day_cannot_submit_at_past_open(setup):
    root, monkeypatch, frame = setup
    monkeypatch.setattr(shadow, "cn_now", lambda: NOW + timedelta(hours=15))
    monkeypatch.setattr(shadow, "_compute", lambda *a: ({"signal_status": "active", "signal_weights": {"5d": 1}}, frame, frame))
    result = shadow.tick(root, now=NOW, trading_day=True)
    assert result["status"] == "late_record"
    assert result["observation"]["selected_count"] == 0
    assert result["eligible_for_forward"] is False
    assert not paper.load_orders(root, result["account_id"])


def test_order_recovery_deduplicates_by_snapshot_source(setup):
    root, monkeypatch, frame = setup
    monkeypatch.setattr(shadow, "_compute", lambda *a: ({"signal_status": "active", "signal_weights": {"5d": 1}}, frame, frame))
    result = shadow.tick(root, now=NOW, trading_day=True)
    account = result["account_id"]
    orders = paper.load_orders(root, account)
    assert len(orders) == 1
    assert orders[0]["order_type"] == "next_open"
    assert orders[0]["qty"] % 100 == 0
    assert orders[0]["created_at"][:10] == str(NOW.date())
    assert shadow.tick(root, now=NOW + timedelta(minutes=5), trading_day=True) == result
    assert len(paper.load_orders(root, account)) == 1


def test_configuration_cannot_replace_existing_frozen_protocol(setup):
    root, _, _ = setup
    with pytest.raises(ValueError, match="already configured"):
        shadow.configure(root, "another", start_date=date(2026, 10, 21))


def test_daily_shadow_api_is_read_only(setup):
    from types import SimpleNamespace

    from app.api.stockformer import daily_shadow_status

    root, _, _ = setup
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repo=SimpleNamespace(store=SimpleNamespace(data_dir=root)))))
    assert daily_shadow_status(request)["recorded_days"] == 0
    assert not (shadow._root(root) / "state.json").exists()


def test_published_snapshot_recovers_orders_once_after_crash(setup):
    root, monkeypatch, frame = setup
    monkeypatch.setattr(shadow, "_compute", lambda *a: ({"signal_status": "active", "signal_weights": {"5d": 1}}, frame, frame))
    original = shadow._orders
    monkeypatch.setattr(shadow, "_orders", lambda *a: (_ for _ in ()).throw(OSError("interrupted")))
    assert shadow.tick(root, now=NOW, trading_day=True)["status"] == "error"
    monkeypatch.setattr(shadow, "_orders", original)
    monkeypatch.setattr(shadow, "_compute", lambda *a: pytest.fail("must preserve published prediction"))
    result = shadow.tick(root, now=NOW, trading_day=True)
    assert result["status"] == "recorded"
    assert len(paper.load_orders(root, result["account_id"])) == 1


def test_failed_settlement_is_never_replayed_automatically(setup):
    root, monkeypatch, _ = setup
    # Exercise the real once-per-session wrapper with an ambiguous crash.
    calls = []
    monkeypatch.setattr(paper, "settle_day", lambda *a, **k: calls.append("settle") or (_ for _ in ()).throw(OSError("disk full")))
    cfg = shadow._read(shadow._root(root) / "config.json")
    with pytest.raises(OSError):
        REAL_SETTLE(root, cfg, NOW.date())
    with pytest.raises(ValueError, match="结算中断"):
        REAL_SETTLE(root, cfg, NOW.date())
    assert calls == ["settle"]


def test_scheduler_registers_independent_tracked_research_job(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.jobs import daily_pipeline
    from app.services import preferences

    class Scheduler:
        def __init__(self, **kwargs):
            self.jobs = {}

        def add_job(self, fn, **kwargs):
            self.jobs[kwargs["id"]] = (fn, kwargs)

        def start(self):
            pass

    calls = []
    monkeypatch.setattr(daily_pipeline, "AsyncIOScheduler", Scheduler)
    monkeypatch.setattr(preferences, "load", lambda: {})
    monkeypatch.setattr("app.market_time.cn_now", lambda: NOW)
    monkeypatch.setattr("app.services.trading_day.is_trading_day", lambda: True)
    monkeypatch.setattr(shadow, "due", lambda *a: True)
    monkeypatch.setattr(shadow, "tick", lambda *a, **k: calls.append(k["trading_day"]) or {"status": "recorded"})
    monkeypatch.setattr(daily_pipeline, "_run_tracked", lambda task, name: calls.append(name) or task())
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    scheduler = daily_pipeline.start_scheduler(repo, None)
    fn, options = scheduler.jobs["stockformer_daily_shadow"]
    assert options["max_instances"] == 1
    fn()
    assert calls == ["stockformer_daily_shadow", True]


def test_fixed_exit_counts_actual_sessions_and_queues_only_once(setup):
    root, monkeypatch, _ = setup
    cfg = shadow._read(shadow._root(root) / "config.json")
    shadow._account(root, cfg)
    days = [date(2026, 10, n) for n in (14, 15, 16, 19, 20)]
    for day in days:
        (root / "kline_daily" / f"date={day}").mkdir(parents=True)
    monkeypatch.setattr(paper, "load_positions", lambda *a: {"000001": {"qty": 100, "lots": [{"date": "2026-10-14", "qty": 100}]}})
    first = shadow._queue_exits(root, cfg, NOW.date())
    assert first[0]["side"] == "sell"
    assert first[0]["order_id"]
    assert shadow._queue_exits(root, cfg, NOW.date()) == []
    assert paper.load_orders(root, cfg["account_id"])[0]["order_type"] == "next_open"


def test_missing_features_do_not_block_settlement_and_due_exits(setup):
    root, monkeypatch, _ = setup
    calls = []
    monkeypatch.setattr(shadow, "readiness", lambda *a: {"ready": False, "reason": "资金未齐", "checks": {"日K": True}})
    monkeypatch.setattr(shadow, "_settle", lambda *a: calls.append("settle") or {})
    monkeypatch.setattr(shadow, "_queue_exits", lambda *a: calls.append("exits") or [])
    result = shadow.tick(root, now=NOW, trading_day=True)
    assert result["status"] == "waiting_data"
    assert calls == ["settle", "exits"]
