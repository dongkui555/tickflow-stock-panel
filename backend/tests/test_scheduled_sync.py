from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.services.scheduled_sync import AfterCloseSync


NOW = datetime(2026, 9, 4, 16, tzinfo=ZoneInfo("Asia/Shanghai"))
SCHEDULE = {"hour": 15, "minute": 30}


def test_busy_does_not_consume_attempt_and_retries_after_restart(tmp_path):
    queue = AfterCloseSync(tmp_path)
    row = queue.tick(NOW, SCHEDULE, True, lambda started: False)
    assert row["status"] == "pending"
    assert row["attempts"] == 0

    def success(started):
        started()
        return True

    row = AfterCloseSync(tmp_path).tick(NOW, SCHEDULE, True, success)
    assert row["status"] == "succeeded"
    assert row["attempts"] == 1
    AfterCloseSync(tmp_path).tick(
        NOW, SCHEDULE, True, lambda _: (_ for _ in ()).throw(AssertionError())
    )


def test_failed_job_backoff_and_bounded_retry(tmp_path):
    queue = AfterCloseSync(tmp_path, clock=lambda: NOW)

    def fail(started):
        started()
        return False

    assert queue.tick(NOW, SCHEDULE, True, fail)["status"] == "retry"
    assert queue.tick(NOW + timedelta(minutes=5), SCHEDULE, True, fail)["attempts"] == 1
    assert queue.tick(NOW + timedelta(minutes=15), SCHEDULE, True, fail)["attempts"] == 2
    assert queue.tick(NOW + timedelta(minutes=30), SCHEDULE, True, fail)["status"] == "failed"


def test_before_close_holiday_or_unknown_calendar_does_not_enqueue(tmp_path):
    queue = AfterCloseSync(tmp_path)

    def unexpected(_):
        raise AssertionError("must not run")

    assert queue.tick(NOW.replace(hour=14), SCHEDULE, True, unexpected)["status"] == "not_due"
    assert queue.tick(NOW, SCHEDULE, False, unexpected)["status"] == "not_due"
    assert queue.tick(NOW, SCHEDULE, None, unexpected)["status"] == "not_due"


def test_recover_interrupted_running_job(tmp_path):
    queue = AfterCloseSync(tmp_path)
    queue._save({"2026-09-04": {"status": "running", "attempts": 1}})

    def success(started):
        started()
        return True

    row = queue.tick(NOW, SCHEDULE, True, success)
    assert row["status"] == "succeeded"
    assert row["attempts"] == 2


def test_after_close_forces_all_market_without_expanding_intraday(monkeypatch, tmp_path):
    from datetime import date
    from types import SimpleNamespace
    from app.jobs import daily_pipeline

    seen = []
    repo = SimpleNamespace(
        store=SimpleNamespace(data_dir=tmp_path),
        latest_daily_date=date.today,
        rebuild_views=lambda: None,
        clear_cache=lambda: None,
        refresh_cache=lambda **kw: None,
    )

    def run(repo, caps, **kwargs):
        seen.append(kwargs["minute_scope_override"])
        return {}

    monkeypatch.setattr(daily_pipeline, "run_now", run)
    monkeypatch.setattr(daily_pipeline, "_get_app_state", lambda: None)
    monkeypatch.setattr("app.api.data.invalidate_storage_cache", lambda: None)
    daily_pipeline._run_scheduled_incremental_sync(repo, None, "after_close")
    daily_pipeline._run_scheduled_incremental_sync(repo, None, "premarket")
    daily_pipeline._run_scheduled_incremental_sync(repo, None, "midday")
    result = daily_pipeline._run_scheduled_incremental_sync(repo, None, "evening")
    assert seen == ["all", None, None]
    assert result["scheduled_slot"] == "evening"
    assert result["scheduled_mode"] == "late_data_finalize"


def test_evening_schedule_defaults_preserve_existing_slots(monkeypatch):
    from app.services import preferences
    saved = []
    monkeypatch.setattr(preferences, "load", lambda: {
        "pipeline_schedule": {"hour": 16, "minute": 10},
        "full_sync_schedule": {"premarket": {"hour": 8, "minute": 10}},
    })
    monkeypatch.setattr(preferences, "save", saved.append)
    schedule = preferences.get_full_sync_schedule()
    assert schedule["evening"] == {"hour": 19, "minute": 0}
    assert schedule["after_close"] == {"hour": 16, "minute": 10}
    preferences.set_pipeline_schedule(15, 45)
    assert saved[-1]["full_sync_schedule"]["evening"] == {"hour": 19, "minute": 0}


def test_evening_queue_independent_and_busy_recovers_after_restart(tmp_path):
    now = NOW.replace(hour=19)
    evening = {"hour": 19, "minute": 0}
    AfterCloseSync(tmp_path)._save({now.date().isoformat(): {"status": "succeeded", "attempts": 1}})
    queue = AfterCloseSync(tmp_path, state_file="evening_sync.json", clock=lambda: now)
    assert queue.tick(now - timedelta(seconds=1), evening, True, lambda _: False)["status"] == "not_due"
    assert queue.tick(now, evening, None, lambda _: False)["status"] == "not_due"
    row = queue.tick(now, evening, True, lambda _: False)
    assert row["status"] == "pending"
    assert row["attempts"] == 0

    def success(started):
        started()
        return True

    restarted = AfterCloseSync(tmp_path, state_file="evening_sync.json", clock=lambda: now)
    assert restarted.tick(now + timedelta(minutes=5), evening, True, success)["status"] == "succeeded"
    row = restarted.tick(now + timedelta(minutes=10), evening, True,
                         lambda _: (_ for _ in ()).throw(AssertionError("duplicate execution")))
    assert row["attempts"] == 1


def test_evening_cron_registered_with_retry_and_calls_incremental(monkeypatch, tmp_path):
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

    monkeypatch.setattr(daily_pipeline, "AsyncIOScheduler", Scheduler)
    monkeypatch.setattr(preferences, "load", lambda: {})
    monkeypatch.setattr("app.market_time.cn_now", lambda: NOW.replace(hour=19))
    monkeypatch.setattr("app.services.trading_day.is_trading_day", lambda: True)
    calls = []

    def run(repo, caps, slot, **kwargs):
        calls.append(slot)
        return {"scheduled_slot": slot}

    def tracked(fn, label, on_started=None):
        if on_started:
            on_started()
        fn()
        return True

    monkeypatch.setattr(daily_pipeline, "_run_scheduled_incremental_sync", run)
    monkeypatch.setattr(daily_pipeline, "_run_tracked", tracked)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    scheduler = daily_pipeline.start_scheduler(repo, None)
    fn, config = scheduler.jobs["evening_pipeline"]
    assert "daily_pipeline" in scheduler.jobs
    next_time = config["trigger"].get_next_fire_time(None, NOW)
    assert next_time.hour == 19 and next_time.minute == 0
    assert str(next_time.tzinfo) == "Asia/Shanghai"
    assert scheduler.jobs["evening_pipeline_retry"][1]["trigger"].interval == timedelta(minutes=5)
    fn()
    fn()
    assert calls == ["evening"]
