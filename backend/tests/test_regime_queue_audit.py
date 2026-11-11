import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from app.services import maintenance_queue as queue
from app.services import regime_builder
from app.services import regime_queue_audit as audit


def seed(root):
    days = [f"2026-08-{d:02}" for d in (24, 25, 26)]
    for day in days:
        path = root / f"kline_daily_enriched/date={day}/part.parquet"
        path.parent.mkdir(parents=True)
        pl.DataFrame({"date": [date.fromisoformat(day)], "symbol": ["000001.SZ"],
                      "close": [10.], "amount": [100.]}).write_parquet(path)
    p = regime_builder.regime_path(root)
    p.parent.mkdir(parents=True)
    pl.DataFrame({"date": [date.fromisoformat(d) for d in days[:2]],
                  "score": [50., 60.], "up_count": [1, 1], "down_count": [0, 0]}).write_parquet(p)
    state = {"legacy_mtime_ns": 0, "dates": {d: regime_builder._source_stamp(root, d) for d in days}}
    state["dates"][days[1]] = {"old": [1, 2]}
    regime_builder._save_source_state(root, state)
    return days


def legacy(root, days):
    task_id = queue.enqueue(root, "regime", days)
    rows = queue._read(root)
    task = next(t for t in rows if t["id"] == task_id)
    task.update(items=days, review_items=[], status="pending", completed=110, total=110 + len(days))
    queue._write(root, rows)
    return task_id


def test_classification_does_not_assume_mtime_change_means_error(tmp_path):
    days = seed(tmp_path)
    findings = audit.classify(tmp_path, days)
    assert [f["decision"] for f in findings] == ["skip", "review", "keep"]
    assert [f["reason"] for f in findings] == ["same_recorded_sources", "source_version_changed_unproven", "result_missing"]


def test_startup_backs_up_and_reduces_without_altering_history(tmp_path):
    days = seed(tmp_path)
    ident = legacy(tmp_path, days)
    protected = [regime_builder.regime_path(tmp_path), tmp_path / "regime_history/sources.json"]
    snapshot = tmp_path / "user_data/prediction_workflow/snapshots/published.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text('{"immutable":true}')
    protected += [snapshot, *tmp_path.glob("kline_daily_enriched/*/*.parquet")]
    before = {p: p.read_bytes() for p in protected}
    original = queue._path(tmp_path).read_bytes()
    queue.recover(tmp_path)
    task = next(t for t in queue._read(tmp_path) if t["id"] == ident)
    assert task["items"] == [days[2]] and task["review_items"] == [days[1]]
    assert task["skipped_items"] == [days[0]] and task["completed"] == 110
    assert task["options"] == {"reason": "missing_result"}
    report = tmp_path / task["audit_report"]
    assert report.with_name("queue-before.json").read_bytes() == original
    assert json.loads(report.read_text())["tasks"][0]["counts"] == {"skip": 1, "review": 1, "keep": 1}
    assert all(p.read_bytes() == content for p, content in before.items())
    queue.recover(tmp_path)
    assert len(list(tmp_path.glob("user_data/maintenance_audits/*/report.json"))) == 1
    status = queue.status(tmp_path)[0]
    assert (status["remaining"], status["review_count"], status["skipped"]) == (1, 1, 1)
    assert "review_items" not in status


def test_unverified_history_never_runs_and_deduplicates_across_reasons(tmp_path):
    days = seed(tmp_path)
    legacy(tmp_path, [days[1]])
    queue.recover(tmp_path)
    for _ in range(3):
        assert queue.enqueue(tmp_path, "regime", [days[1]], reason="source_change_unverified") is None
    assert len(queue._read(tmp_path)) == 1
    assert not queue.tick(tmp_path, lambda *a, **kw: pytest.fail("must not run"), lambda *a: None)
    assert queue.status(tmp_path)[0]["status"] == "paused_review"


def test_new_verified_change_bypasses_hold_and_resolves_it_after_completion(tmp_path):
    from test_maintenance_queue import runner
    days = seed(tmp_path)
    legacy(tmp_path, [days[1]])
    queue.recover(tmp_path)
    ident = queue.enqueue(tmp_path, "regime", [days[1]], reason="input_changed")
    assert queue.tick(tmp_path, runner, lambda *a: {})
    statuses = {t["id"]: t for t in queue.status(tmp_path)}
    assert statuses[ident]["status"] == "succeeded"
    held = next(t for t in statuses.values() if t["id"] != ident)
    assert held["status"] == "reviewed" and held["resolved"] == 1 and held["completed"] == 110


@pytest.mark.parametrize("options", [{"reason": "input_changed"}, {"reason": "manual_recompute", "force": True}])
def test_verified_and_manual_work_not_cancelled_by_audit(tmp_path, options):
    days = seed(tmp_path)
    queue.enqueue(tmp_path, "regime", days, **options)
    original = queue._read(tmp_path)
    queue.recover(tmp_path)
    assert queue._read(tmp_path) == original


@pytest.mark.parametrize("failure", ["missing_input", "bad_schema", "zero_price", "empty_input", "corrupt_result"])
def test_invalid_data_not_treated_as_a_verified_rebuild_request(tmp_path, failure):
    days = seed(tmp_path)
    path = tmp_path / f"kline_daily_enriched/date={days[2]}/part.parquet"
    if failure == "missing_input":
        path.unlink()
    elif failure == "bad_schema":
        pl.DataFrame({"date": [date.fromisoformat(days[2])]}).write_parquet(path)
    elif failure == "zero_price":
        pl.read_parquet(path).with_columns(pl.lit(0.).alias("close")).write_parquet(path)
    elif failure == "empty_input":
        pl.read_parquet(path).head(0).write_parquet(path)
    else:
        regime_builder.regime_path(tmp_path).write_bytes(b"bad")
    legacy(tmp_path, [days[2]])
    queue.recover(tmp_path)
    assert queue.status(tmp_path)[0]["status"] == "paused_review"
    assert queue.status(tmp_path)[0]["remaining"] == 0


def test_failed_backup_never_mutates_queue(tmp_path, monkeypatch):
    days = seed(tmp_path)
    legacy(tmp_path, days)
    original = queue._path(tmp_path).read_bytes()
    monkeypatch.setattr(Path, "write_bytes", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        queue.recover(tmp_path)
    assert queue._path(tmp_path).read_bytes() == original


def test_busy_worker_cannot_be_audited(tmp_path):
    queue._worker_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="busy"):
            queue.audit_regime(tmp_path)
    finally:
        queue._worker_lock.release()


def test_manual_recheck_can_confirm_previously_held_sources(tmp_path):
    days = seed(tmp_path)
    legacy(tmp_path, [days[1]])
    queue.recover(tmp_path)
    state = regime_builder._source_state(tmp_path)
    state["dates"][days[1]] = regime_builder._source_stamp(tmp_path, days[1])
    regime_builder._save_source_state(tmp_path, state)
    queue.audit_regime(tmp_path)
    assert queue.status(tmp_path)[0]["status"] == "reviewed"
    assert queue.status(tmp_path)[0]["skipped"] == 1


def test_daily_detected_timestamp_changes_do_not_requeue_for_execution(tmp_path):
    from app.services.daily_maintenance import queue_regime_history
    days = seed(tmp_path)
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    queue_regime_history(repo, [], date(2026, 9, 9))
    pending = [t for t in queue._read(tmp_path) if t["status"] == "pending"]
    held = [t for t in queue._read(tmp_path) if t["status"] == "paused_review"]
    assert len(pending) == 1 and pending[0]["items"] == [days[2]]
    assert len(held) == 1 and held[0]["review_items"] == [days[1]]


def test_direct_execution_without_change_evidence_is_rejected(tmp_path):
    from app.services.daily_maintenance import execute
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    with pytest.raises(RuntimeError, match="evidence"):
        execute(repo, None, {"kind": "regime", "label": "regime", "items": ["2026-08-24"], "options": {}}, lambda *a, **kw: None)


def test_queue_audit_endpoint_respects_global_data_writer(tmp_path):
    from fastapi import HTTPException

    from app.api.pipeline import audit_regime_queue
    from app.services.pipeline_jobs import release_run_slot, try_acquire_run_slot
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repo=SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)))))
    assert try_acquire_run_slot("test-writer")
    try:
        with pytest.raises(HTTPException) as error:
            audit_regime_queue(request)
        assert error.value.status_code == 409
    finally:
        release_run_slot("test-writer")


def test_already_repaired_missing_result_is_not_counted_as_recomputed(tmp_path):
    from test_maintenance_queue import runner
    queue.enqueue(tmp_path, "regime", ["2026-08-24"], reason="missing_result")
    assert queue.tick(tmp_path, runner, lambda *a: {"rows": 0, "skipped_items": ["2026-08-24"]})
    task = queue.status(tmp_path)[0]
    assert task["completed"] == 0 and task["skipped"] == 1 and task["remaining"] == 0


def test_duplicate_legacy_dates_keep_only_one_review_owner(tmp_path):
    days = seed(tmp_path)
    ident = legacy(tmp_path, [days[1]])
    duplicate = queue.enqueue(tmp_path, "regime", [days[1]], reason="missing_or_changed")
    queue.recover(tmp_path)
    tasks = {t["id"]: t for t in queue._read(tmp_path)}
    assert tasks[ident]["review_items"] == [days[1]]
    assert tasks[duplicate]["status"] == "reviewed" and tasks[duplicate]["review_items"] == []
    assert tasks[duplicate]["linked_review_items"] == {days[1]: ident}
