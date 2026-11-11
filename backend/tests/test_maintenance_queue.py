from datetime import date, datetime, timedelta
from types import ModuleType, SimpleNamespace
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from app.services import maintenance_queue as queue

NOW = datetime(2026, 9, 8, 20, tzinfo=ZoneInfo("Asia/Shanghai"))


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(queue, "cn_now", lambda: NOW)
    monkeypatch.setattr(queue, "_last_queue", "history")


def runner(fn, label, on_started=None):
    on_started()
    try:
        fn(on_progress=lambda *a, **kw: None)
        return True
    except Exception:
        return False


def test_busy_slot_does_not_consume_work_and_survives_restart(tmp_path):
    queue.enqueue(tmp_path, "minute", ["000001.SZ"], days=2)
    assert not queue.tick(tmp_path, lambda *a, **kw: False, lambda *a: None, now=NOW)
    assert queue.status(tmp_path)[0]["attempts"] == 0
    queue.recover(tmp_path)
    assert queue.tick(tmp_path, runner, lambda *a: {}, now=NOW)
    assert queue.status(tmp_path)[0]["status"] == "succeeded"


def test_bounded_batches_fairness_and_separate_progress(tmp_path):
    queue.enqueue(tmp_path, "minute", [f"{i:06}.SZ" for i in range(120)])
    queue.enqueue(tmp_path, "adjustment", ["A", "B"])
    seen = []

    def execute(task, progress):
        seen.append((task["kind"], len(task["items"])))
        progress("work", 50, "half")
        live = next(r for r in queue.status(tmp_path) if r["id"] == task["id"])
        assert live["status"] == "running" and live["batch_progress"] == 50
        return {}

    queue.tick(tmp_path, runner, execute, now=NOW)
    queue.tick(tmp_path, runner, execute, now=NOW)
    assert seen == [("minute", 50), ("adjustment", 2)]
    minute = next(r for r in queue.status(tmp_path) if r["queue"] == "minute")
    assert minute["completed"] == 50 and minute["total"] == 120


def test_partial_completion_moves_bad_stocks_to_tail_and_backoff(tmp_path):
    symbols = [f"{i:06}.SZ" for i in range(60)]
    queue.enqueue(tmp_path, "minute", symbols)
    assert not queue.tick(tmp_path, runner, lambda t, p: {
        "completed_items": t["items"][1:], "error": "source_pending"}, now=NOW)
    row = queue.status(tmp_path)[0]
    assert row["completed"] == 49 and row["status"] == "retry"
    assert not queue.tick(tmp_path, runner, lambda *a: pytest.fail("backoff"), now=NOW)
    # Repeated automatic enqueue does not add already checked stocks again.
    queue.enqueue(tmp_path, "minute", symbols)
    assert queue.status(tmp_path)[0]["total"] == 60
    seen = []
    queue.tick(tmp_path, runner, lambda t, p: seen.extend(t["items"]), now=NOW + timedelta(minutes=2))
    assert seen == [*symbols[50:], symbols[0]]


def test_minute_rotation_backs_off_after_one_pass_even_with_more_than_batch_size(tmp_path):
    queue.enqueue(tmp_path, "minute", [f"{i:06}" for i in range(52)])
    def execute(task, progress):
        return {"completed_items": [], "error": "source incomplete"}
    for offset in (0, 2, 4):
        queue.tick(tmp_path, runner, execute, now=NOW + timedelta(minutes=offset))
    row = queue.status(tmp_path)[0]
    assert row["completed"] == 0 and row["attempts"] == 3
    assert datetime.fromisoformat(row["retry_at"]) >= NOW + timedelta(minutes=9)
    assert not queue.tick(tmp_path, runner, execute, now=NOW + timedelta(minutes=6))


def test_minute_progress_resets_stale_retry_delay(tmp_path):
    queue.enqueue(tmp_path, "minute", [f"{i:06}" for i in range(52)])
    rows = queue._read(tmp_path)
    rows[0]["attempts"] = 92
    queue._write(tmp_path, rows)
    queue.tick(tmp_path, runner, lambda t, p: {
        "completed_items": t["items"][:40], "error": "remaining conflicts"}, now=NOW)
    row = queue.status(tmp_path)[0]
    assert row["completed"] == 40 and row["attempts"] == 1
    assert datetime.fromisoformat(row["retry_at"]) == NOW + timedelta(minutes=1)


def test_running_enqueue_is_not_lost_and_regime_waits_for_derivation(tmp_path):
    queue.enqueue(tmp_path, "adjustment", ["A"])
    queue.enqueue(tmp_path, "regime", ["2026-09-07"])

    def execute(t, progress):
        assert t["kind"] == "adjustment"
        queue.enqueue(tmp_path, "adjustment", ["B"])
        raise TimeoutError("secret must not be exposed")

    queue.tick(tmp_path, runner, execute, now=NOW)
    rows = queue.status(tmp_path)
    assert any(r["kind"] == "adjustment" and r["status"] == "pending" for r in rows)
    assert next(r for r in rows if r["status"] == "retry")["error"] == "TimeoutError"
    seen = []
    queue.tick(tmp_path, runner, lambda t, p: seen.append(t["kind"]), now=NOW)
    assert seen == ["adjustment"]
    assert not queue.tick(tmp_path, runner, lambda *a: pytest.fail("regime must wait"), now=NOW)


def test_history_not_run_in_market_hours(tmp_path):
    queue.enqueue(tmp_path, "adjustment", ["A"])
    assert not queue.tick(tmp_path, runner, lambda *a: pytest.fail("not now"),
                          now=NOW, allow_history=False)


def test_full_market_minute_waits_until_bulk_window(tmp_path):
    queue.enqueue(tmp_path, "minute", ["000001.SZ"], as_of="2026-09-08T15:00", days=1, scope="all")
    assert not queue.tick(
        tmp_path, runner, lambda *a: pytest.fail("bulk minute must wait"),
        now=NOW, allow_bulk_minute=False,
    )
    assert queue.tick(tmp_path, runner, lambda *a: {}, now=NOW, allow_bulk_minute=True)


def test_latest_minute_day_runs_before_old_gap(tmp_path):
    queue.enqueue(tmp_path, "minute", ["000001.SZ"], as_of="2026-09-08T15:00", days=1, scope="all")
    queue.enqueue(tmp_path, "minute", ["000002.SZ"], as_of="2026-09-10T15:00", days=1, scope="all")
    seen = []
    queue.tick(tmp_path, runner, lambda task, progress: seen.append(task["options"]["as_of"]), now=NOW)
    assert seen == ["2026-09-10T15:00"]


def test_history_gap_rescan_runs_oldest_date_first(tmp_path):
    queue.enqueue(tmp_path, "minute", ["000001.SZ"],
                  as_of="2026-09-08T15:00", days=0,
                  target_date="2026-09-08", scope="history:2026-09-08",
                  mode="history_gap_rescan")
    queue.enqueue(tmp_path, "minute", ["000002.SZ"],
                  as_of="2026-09-10T15:00", days=0,
                  target_date="2026-09-10", scope="history:2026-09-10",
                  mode="history_gap_rescan")
    seen = []
    queue.tick(tmp_path, runner,
               lambda task, progress: seen.append(task["options"]["target_date"]),
               now=NOW)
    assert seen == ["2026-09-08"]


def test_full_market_minute_releases_slot_between_successful_batches(tmp_path, monkeypatch):
    monkeypatch.setenv("MAINTENANCE_MINUTE_BATCH_SIZE", "10")
    monkeypatch.setenv("MAINTENANCE_BULK_MINUTE_COOLDOWN_SECONDS", "15")
    queue.enqueue(tmp_path, "minute", [f"{i:06}.SZ" for i in range(20)],
                  as_of="2026-09-10T15:00", days=1, scope="all")
    assert queue.tick(tmp_path, runner, lambda *a: {}, now=NOW)
    row = queue.status(tmp_path)[0]
    assert row["completed"] == 10 and row["status"] == "pending"
    assert datetime.fromisoformat(row["retry_at"]) == NOW + timedelta(seconds=15)
    assert not queue.tick(tmp_path, runner, lambda *a: pytest.fail("cooldown"), now=NOW)
    assert queue.tick(tmp_path, runner, lambda *a: {}, now=NOW + timedelta(seconds=15))


def test_minute_date_gate_finishes_one_day_before_unlocking_next(tmp_path, monkeypatch):
    monkeypatch.setenv("MAINTENANCE_MINUTE_BATCH_SIZE", "10")
    pipeline_jobs = ModuleType("app.services.pipeline_jobs")
    pipeline_jobs.job_store = SimpleNamespace(active_id=lambda: "test-job")
    monkeypatch.setitem(__import__("sys").modules, "app.services.pipeline_jobs", pipeline_jobs)
    symbols = ["000001.SZ", "000002.SZ"]
    for day in (date(2026, 9, 7), date(2026, 9, 8)):
        path = tmp_path / "kline_daily" / f"date={day}" / "part.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"date": [day], "symbol": [symbols[0]], "close": [10.0]}).write_parquet(path)

    queue.enqueue(tmp_path, "minute", symbols, as_of="2026-09-08T15:00", days=1, scope="all")
    seen = []

    def execute(task, progress):
        seen.append((task["options"]["target_date"], task["options"]["days"], task["items"]))
        return {"completed_items": task["items"]}

    assert queue.tick(tmp_path, runner, execute, now=NOW)
    first = queue.status(tmp_path)[0]
    assert seen == [("2026-09-07", 0, symbols)]
    assert first["gate_date"] == "2026-09-08"
    assert first["gate_position"] == 2 and first["gate_total_dates"] == 2
    assert first["gate_remaining"] == len(symbols)
    assert first["completed"] == 0 and first["remaining"] == len(symbols)

    assert queue.tick(tmp_path, runner, execute, now=NOW + timedelta(seconds=15))
    final = queue.status(tmp_path)[0]
    assert seen[-1] == ("2026-09-08", 0, symbols)
    assert final["status"] == "succeeded"
    assert final["completed"] == len(symbols) and final["remaining"] == 0


def test_date_gate_does_not_duplicate_batch_when_enqueue_repeats(tmp_path, monkeypatch):
    monkeypatch.setenv("MAINTENANCE_MINUTE_BATCH_SIZE", "10")
    symbols = [f"{i:06}.SZ" for i in range(20)]
    options = {"as_of": "2026-09-08T15:00", "days": 0, "scope": "all"}
    path = tmp_path / "kline_daily/date=2026-09-08/part.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": symbols}).write_parquet(path)
    queue.enqueue(tmp_path, "minute", symbols, **options)
    seen = []

    def execute(task, progress):
        seen.append(task["items"])
        return {"completed_items": [], "error": "source incomplete"}

    queue.tick(tmp_path, runner, execute, now=NOW)
    queue.enqueue(tmp_path, "minute", symbols, **options)
    queue.tick(tmp_path, runner, execute, now=NOW + timedelta(minutes=2))
    assert seen == [symbols[:10], symbols[:10]]


def test_date_gate_zero_progress_backoff_and_bounded_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("MAINTENANCE_MINUTE_BATCH_SIZE", "10")
    monkeypatch.setattr(queue, "_MAX_NO_PROGRESS_ATTEMPTS", 2)
    symbols = [f"{i:06}.SZ" for i in range(20)]
    options = {"as_of": "2026-09-08T15:00", "days": 0, "scope": "all"}
    path = tmp_path / "kline_daily/date=2026-09-08/part.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": symbols}).write_parquet(path)
    queue.enqueue(tmp_path, "minute", symbols, **options)
    execute = lambda *a: {"completed_items": [], "error": "source incomplete"}
    queue.tick(tmp_path, runner, execute, now=NOW)
    row = queue.status(tmp_path)[0]
    due = datetime.fromisoformat(row["retry_at"])
    queue.tick(tmp_path, runner, execute, now=due + timedelta(seconds=1))
    row = queue.status(tmp_path)[0]
    assert row["status"] == "paused_review"
    assert row["remaining"] == 20 and row["completed"] == 0
    assert row["retry_at"] is None


def test_identity_failure_pauses_without_losing_items(tmp_path):
    queue.enqueue(tmp_path, "minute", ["000001.SZ"])

    def execute(task, progress):
        return {
            "completed_items": [],
            "error": "identity_pending",
            "identity_pending": [{"reason": "security_not_in_dimension"}],
        }

    assert not queue.tick(tmp_path, runner, execute, now=NOW)
    row = queue.status(tmp_path)[0]
    assert row["status"] == "paused_review"
    assert row["remaining"] == 1 and row["completed"] == 0
    assert row["retry_at"] is None


def test_cancelled_job_keeps_its_owned_slot_until_worker_returns(tmp_path):
    from app.services.pipeline_jobs import JobStore, release_run_slot, try_acquire_run_slot
    jobs = JobStore(store_dir=tmp_path / "jobs")
    job_id, _ = jobs.create()
    assert try_acquire_run_slot(job_id)
    try:
        jobs.start(job_id)
        jobs.terminate(job_id, "test cancellation")
        assert not try_acquire_run_slot("queued-next")
    finally:
        release_run_slot(job_id)
    assert try_acquire_run_slot("queued-next")
    release_run_slot("queued-next")


def test_recent_daily_correction_updates_without_factor_change(tmp_path, monkeypatch):
    from app.indicators import pipeline
    from app.services import daily_maintenance as daily
    days = [date(2026, 9, d) for d in (1, 2, 3, 4, 7, 8)]
    for d in days:
        for table in ("kline_daily", "kline_daily_enriched"):
            path = tmp_path / table / f"date={d}" / "part.parquet"
            path.parent.mkdir(parents=True)
            pl.DataFrame({"date": [d], "symbol": ["A"], "close": [10.0]}).write_parquet(path)
    before = daily.partition_versions(tmp_path, "kline_daily")
    before[days[-1]] = []
    called = []
    monkeypatch.setattr(pipeline, "run_date_pipeline", lambda root, day: called.append(day) or 1)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    count, targets = daily.update_recent_enriched(repo, before, [], days[-1], lambda *a, **k: None)
    assert count == 1 and targets == [days[-1]] and called == [days[-1]]
    called.clear()
    before[days[0]] = []
    daily.update_recent_enriched(repo, before, [], days[-1], lambda *a, **k: None)
    assert called == days[-5:]
    assert queue.status(tmp_path)[0]["kind"] == "enriched_dates"
    assert queue.status(tmp_path)[0]["total"] == 1


@pytest.mark.parametrize("as_of", ["2026-09-08T15:00", "2026-09-08T11:30"])
def test_minute_batch_validates_suffixed_symbols_and_day_prices(tmp_path, monkeypatch, as_of):
    from app import market_time
    from app.jobs import daily_pipeline
    from app.services import daily_maintenance, minute_repair
    day = date(2026, 9, 8)
    stamps = [datetime(2026, 9, 8, 9, 31) + timedelta(minutes=i) for i in range(120)]
    stamps += [datetime(2026, 9, 8, 13, 1) + timedelta(minutes=i) for i in range(120)]
    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
    daily = pl.DataFrame({"symbol": codes, "date": [day] * 3,
                         **{c: [10.0] * 3 for c in ("open", "high", "low", "close")},
                         "volume": [24000.0] * 3, "amount": [240000.0] * 3})
    parts = []
    for i, symbol in enumerate(codes):
        times = stamps[:2] if i == 1 else stamps
        parts.append(pl.DataFrame({"symbol": [symbol] * len(times), "datetime": times,
            **{c: [20.0 if i == 2 else 10.0] * len(times) for c in ("open", "high", "low", "close")},
            "volume": [1.0] * len(times), "amount": [1000.0] * len(times)}))
    for table, frame in (("kline_daily", daily), ("kline_minute", pl.concat(parts))):
        path = tmp_path / table / f"date={day}" / "part.parquet"
        path.parent.mkdir(parents=True)
        frame.write_parquet(path)
    calls = []
    monkeypatch.setattr(market_time, "cn_now", lambda: NOW)
    monkeypatch.setattr(minute_repair, "repair_minute_gaps", lambda *a, **kw: calls.append(kw) or {})
    monkeypatch.setattr(daily_maintenance, "_cross_source_verify_failures", lambda *a, **kw: [])
    monkeypatch.setattr(daily_pipeline, "_invalidate", lambda *a: None)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    result = daily_maintenance.execute(repo, None, {
        "id": "minute-test", "kind": "minute", "label": "minute test", "items": codes,
        "options": {"as_of": as_of, "days": 1}}, lambda *a, **kw: None)
    assert result["completed_items"] == [codes[0]]
    assert result["error"]
    assert result["validated_until"] == "2026-09-08T15:00:00"
    assert calls[0]["start_date"] == calls[0]["end_date"] == day
    assert calls[0]["task_id"] == "minute-test"
    assert calls[0]["symbols"] == codes[1:]
    assert {r["symbol"]: r["reason"] for r in result["validation_failures"]} == {
        "000002": "source_incomplete", "000003": "minute_turnover_unit_conflict"}


def test_midday_validation_checks_interior_minutes_not_just_latest(tmp_path):
    from test_minute_failover import bars

    from app.services.daily_maintenance import minute_validation_failures
    day = date(2026, 9, 7)
    frame = bars(day=day).head(120).filter(pl.col("datetime") != datetime(2026, 9, 7, 10))
    path = tmp_path / f"kline_minute/date={day}/part.parquet"
    path.parent.mkdir(parents=True)
    frame.write_parquet(path)
    result = minute_validation_failures(SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        ["000701"], datetime(2026, 9, 7), datetime(2026, 9, 7, 11, 30))
    assert len(result) == 1 and result[0]["missing_times"] == ["10:00"]


def test_queue_status_api_is_read_only_and_manual_regime_is_queued(tmp_path):
    from app.api.pipeline import maintenance_queue_status
    from app.api.regime import regime_recompute
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repo=repo)))
    assert maintenance_queue_status(request) == {"tasks": [], "minute_gap_queue": {}}
    assert not queue._path(tmp_path).exists()
    path = tmp_path / "kline_daily_enriched" / "date=2026-09-07" / "part.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({"date": [date(2026, 9, 7)]}).write_parquet(path)
    response = regime_recompute(request)
    assert response["queued"] and response["computed"] == 0
    status = maintenance_queue_status(request)["tasks"]
    assert status[0]["id"] == response["task_id"] and status[0]["status"] == "pending"
    assert "items" not in status[0]


@pytest.mark.parametrize("listing", ["20260909", "2026-09-09"])
def test_minute_validation_classifies_pre_listing_and_zero_trade_days(tmp_path, listing):
    from app.services.daily_maintenance import minute_validation_report

    day = date(2026, 9, 8)
    instruments = tmp_path / "instruments/instruments.parquet"
    instruments.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": ["000001.SZ"], "listing_date": [listing]}).write_parquet(instruments)
    daily = tmp_path / f"kline_daily/date={day}/part.parquet"
    daily.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": ["000002.SZ"], "date": [day],
                  **{c: [10.] for c in ("open", "high", "low", "close")},
                  "volume": [0.], "amount": [0.]}).write_parquet(daily)
    result, statuses = minute_validation_report(SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        ["000001", "000002", "000003"], datetime(2026, 9, 8), datetime(2026, 9, 8, 15))
    assert [(r["symbol"], r["reason"]) for r in result] == [
        ("000001", "not_listed_yet"), ("000003", "daily_reference_missing")]
    assert {r["symbol"]: r["status"] for r in statuses} == {
        "000001": "pre_listing", "000002": "no_trade", "000003": "unknown"}


def test_minute_validation_uses_resolvable_bare_daily_identity_when_lifecycle_is_missing(tmp_path):
    from app.services.daily_maintenance import minute_validation_failures

    day = date(2026, 9, 8)
    instruments = tmp_path / "instruments/instruments.parquet"
    instruments.parent.mkdir(parents=True)
    # Keep the lifecycle index non-empty but omit the requested stock. The
    # bare daily row is accepted only because 000001 has a deterministic
    # market mapping; it is not a Shenzhen default.
    pl.DataFrame({"symbol": ["000002.SZ"], "listing_date": ["20200101"]}).write_parquet(instruments)
    daily = tmp_path / f"kline_daily/date={day}/part.parquet"
    daily.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": ["000001"], "date": [day],
                  **{c: [10.] for c in ("open", "high", "low", "close")},
                  "volume": [100.], "amount": [1000.]}).write_parquet(daily)

    result = minute_validation_failures(
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        ["000001"], datetime(2026, 9, 8), datetime(2026, 9, 8, 15),
    )

    assert result and result[0]["reason"] != "identity_history_unknown"


def test_minute_validation_classifies_missing_daily_reference_after_identity_check(tmp_path):
    from app.services.daily_maintenance import minute_validation_failures

    day = date(2026, 9, 21)
    instruments = tmp_path / "instruments/instruments.parquet"
    instruments.parent.mkdir(parents=True)
    pl.DataFrame({"symbol": ["000002.SZ"], "listing_date": ["20200101"]}).write_parquet(instruments)

    result = minute_validation_failures(
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        ["000016"], datetime(2026, 9, 21), datetime(2026, 9, 21, 15),
    )

    assert result and result[0]["reason"] == "daily_reference_missing"


def test_recover_pauses_broad_index_history_and_keeps_core_work(tmp_path):
    from app.services.index_const import CORE_INDEX_SYMBOLS

    core = CORE_INDEX_SYMBOLS[0]
    queue.enqueue(tmp_path, "index_daily", [core, "000300.SH", "932000.SH"],
                  start_date="2026-09-01", end_date="2026-09-09")
    queue.recover(tmp_path)
    row = queue._read(tmp_path)[0]
    assert row["items"] == [core]
    assert row["review_items"] == ["000300.SH", "932000.SH"]
    assert row["status"] == "pending"
