from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from app.plugins.stockdb import manager


def slot(root, name):
    path = root / "slots" / name
    (path / "data").mkdir(parents=True)
    (path / "stockdb.conf").write_text("readonly", encoding="utf-8")
    (path / "sync_url.txt").write_text("local", encoding="utf-8")
    (path / "data" / "CURRENT").write_text("MANIFEST", encoding="utf-8")
    return path


def test_reader_start_failure_waits_and_recovers_without_claiming_data_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, 'ROOT', tmp_path)
    monkeypatch.setattr(manager, 'STATE', tmp_path / 'manager-status.json')
    monkeypatch.setattr(manager, 'STOP', False)
    monkeypatch.setattr(manager, 'RETRY_SECONDS', 10)
    manager._write_state(active_slot='a', data_ready=True, last_success_key='old')
    process, log = Mock(), Mock()
    start = Mock(side_effect=[RuntimeError('upstream unavailable'), (process, log)])
    monkeypatch.setattr(manager, '_start_server', start)
    sleeps = []

    def pause(seconds):
        state = manager._read_state()
        assert state['status'] == 'reader_unavailable'
        assert state['error'] == 'reader_start_failed'
        assert state['error_type'] == 'RuntimeError'
        assert not state['data_ready'] and state['active_slot'] == 'a'
        assert state['last_success_key'] == 'old'
        sleeps.append(seconds)

    monkeypatch.setattr(manager.time, 'sleep', pause)
    assert manager._start_server_with_retry(tmp_path / 'slots/a') == (process, log)
    assert sleeps == [5, 5] and start.call_count == 2
    assert not manager._read_state()['data_ready']


def test_reader_retry_can_stop_during_backoff(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, 'ROOT', tmp_path)
    monkeypatch.setattr(manager, 'STATE', tmp_path / 'manager-status.json')
    monkeypatch.setattr(manager, 'STOP', False)
    monkeypatch.setattr(manager, 'RETRY_SECONDS', 600)
    start = Mock(side_effect=OSError('unavailable executable'))
    monkeypatch.setattr(manager, '_start_server', start)
    monkeypatch.setattr(manager.time, 'sleep', lambda _: setattr(manager, 'STOP', True))
    assert manager._start_server_with_retry(tmp_path / 'slots/a') == (None, None)
    assert start.call_count == 1
    assert not manager._read_state()['data_ready']


def test_reader_launch_oserror_closes_log(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, 'ROOT', tmp_path)
    handle = Mock()
    monkeypatch.setattr('builtins.open', Mock(return_value=handle))
    monkeypatch.setattr(manager.subprocess, 'Popen', Mock(side_effect=PermissionError('unavailable')))
    with pytest.raises(PermissionError):
        manager._start_server(tmp_path)
    handle.close.assert_called_once()


def test_failed_candidate_and_temporary_rollback_failure_preserve_active_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, 'ROOT', tmp_path)
    monkeypatch.setattr(manager, 'SLOTS', tmp_path / 'slots')
    monkeypatch.setattr(manager, 'STATE', tmp_path / 'manager-status.json')
    monkeypatch.setattr(manager, 'STOP', False)
    monkeypatch.setattr(manager, 'RETRY_SECONDS', 5)
    slot(tmp_path, 'a')
    slot(tmp_path, 'b')
    manager._write_state(active_slot='a', data_ready=True)
    monkeypatch.setattr(manager.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(manager, '_update_log_errors', lambda *a: (False, False))
    monkeypatch.setattr(manager, '_stop_server', Mock())
    old_process, old_log = SimpleNamespace(pid=1), Mock()
    start = Mock(side_effect=[RuntimeError('candidate unavailable'),
                             RuntimeError('temporary upstream failure'), (old_process, old_log)])
    monkeypatch.setattr(manager, '_start_server', start)
    monkeypatch.setattr(manager.time, 'sleep', Mock())
    assert manager._update('a', old_process, old_log, '2026-11-02 19:20') == (
        'a', old_process, old_log, False)
    assert [call.args[0].name for call in start.call_args_list] == ['b', 'a', 'a']
    state = manager._read_state()
    assert state['active_slot'] == 'a' and not state['data_ready']
    assert state['error'] == 'candidate_reader_start_failed'


def test_latest_due_uses_last_after_close_slot(monkeypatch):
    monkeypatch.setattr(manager, "UPDATE_TIMES", ("15:20", "18:40"))
    tz = ZoneInfo("Asia/Shanghai")
    assert manager._latest_due(datetime(2026, 9, 10, 15, 19, tzinfo=tz)) is None
    assert manager._latest_due(datetime(2026, 9, 10, 16, 0, tzinfo=tz)) == "2026-09-10 15:20"
    assert manager._latest_due(datetime(2026, 9, 10, 19, 0, tzinfo=tz)) == "2026-09-10 18:40"
    assert manager._latest_due(datetime(2026, 9, 12, 19, 0, tzinfo=tz)) is None


def test_snapshot_update_is_manual_by_default(tmp_path, monkeypatch):
    trigger = tmp_path / "update.trigger"
    monkeypatch.setattr(manager, "UPDATE_TRIGGER", trigger)
    monkeypatch.setattr(manager, "AUTO_UPDATE_ENABLED", False)
    now = datetime(2026, 9, 10, 19, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert manager._requested_update(now) is None

    trigger.write_text("repair-20260910", encoding="utf-8")
    assert manager._requested_update(now) == "manual repair-20260910"
    assert not trigger.exists()


def test_update_log_rejects_partial_manifest_success(tmp_path):
    path = tmp_path / "updater.log"
    path.write_text("old failure ignored\n", encoding="utf-8")
    offset = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write("[file] 4/10 失败 broken.ldb\nmanifest failed, http status=451\n")

    assert manager._update_log_failed(path, offset) is True
    assert manager._update_log_failed(path, path.stat().st_size) is False


def test_update_log_treats_single_file_failure_as_recoverable(tmp_path):
    path = tmp_path / "updater.log"
    path.write_text("run started\n", encoding="utf-8")
    offset = path.stat().st_size
    with path.open("a", encoding="utf-8") as handle:
        handle.write("[file] 4/10 失败 003453.log\n")

    assert manager._update_log_errors(path, offset) == (False, True)


def test_candidate_readiness_counts_only_regular_session_minutes(monkeypatch):
    rows = [
        ["minute", {"date": int(f"20260910{minute}00")}]
        for minute in (
            [f"{hour:02d}{minute:02d}" for hour in (9, 10) for minute in range(60)]
        )
    ]
    assert manager._regular_minute_count(rows, "20260910") == 89
    assert manager._target_trade_date("2026-09-10 19:20") == "20260910"
    assert manager._target_trade_date("manual repair") is None


def test_failed_inactive_update_keeps_serving_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(manager, "STATE", tmp_path / "status.json")
    slot(tmp_path, "a")
    slot(tmp_path, "b")
    process, server_log = object(), Mock()
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=2))
    stop = Mock()
    monkeypatch.setattr(manager, "_stop_server", stop)
    active, returned_process, returned_log, ok = manager._update(
        "a", process, server_log, "2026-09-10 15:20")
    assert (active, returned_process, returned_log, ok) == ("a", process, server_log, False)
    stop.assert_not_called()


@pytest.mark.parametrize("failure", [PermissionError("not executable"), FileNotFoundError("missing")])
def test_updater_launch_failure_keeps_reader_running_and_records_retry(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(manager, "STATE", tmp_path / "status.json")
    slot(tmp_path, "a")
    slot(tmp_path, "b")
    manager._write_state(active_slot="a", data_ready=True)
    process, server_log = object(), Mock()
    stop = Mock()
    monkeypatch.setattr(manager, "_stop_server", stop)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(manager.subprocess, "run", fail)
    result = manager._update("a", process, server_log, "2026-09-10 15:20")
    assert result == ("a", process, server_log, False)
    stop.assert_not_called()
    state = manager._read_state()
    assert state["error"] == "updater_unavailable"
    assert state["error_type"] == type(failure).__name__
    assert state["data_ready"] is True
    assert state["retry_after_seconds"] == manager.RETRY_SECONDS


def test_failed_refresh_keeps_published_partial_snapshot_serving(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(manager, "STATE", tmp_path / "status.json")
    slot(tmp_path, "a")
    slot(tmp_path, "b")
    manager._write_state(
        active_slot="a", partial_update=True, data_ready=False,
        last_published_key="2026-09-11 19:20",
    )
    process, server_log = object(), Mock()
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=2))
    stop = Mock()
    monkeypatch.setattr(manager, "_stop_server", stop)

    active, returned_process, returned_log, ok = manager._update(
        "a", process, server_log, "2026-09-11 19:20")

    assert (active, returned_process, returned_log, ok) == ("a", process, server_log, False)
    state = manager._read_state()
    assert state["status"] == "serving_partial"
    assert state["partial_update"] is True


def test_successful_inactive_update_switches_after_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(manager, "STATE", tmp_path / "status.json")
    slot(tmp_path, "a")
    slot(tmp_path, "b")
    process, server_log = object(), Mock()
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    stop = Mock()
    monkeypatch.setattr(manager, "_stop_server", stop)
    new_process, new_log = SimpleNamespace(pid=42), Mock()
    monkeypatch.setattr(manager, "_start_server", lambda path: (new_process, new_log))
    monkeypatch.setattr(
        manager,
        "_verify_candidate_data",
        lambda key: {"ready": True, "publishable": True},
    )
    active, returned_process, returned_log, ok = manager._update(
        "a", process, server_log, "2026-09-10 18:40")
    assert (active, returned_process, returned_log, ok) == ("b", new_process, new_log, True)
    stop.assert_called_once_with(process, server_log)


def test_candidate_without_scheduled_day_data_restores_old_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(manager, "STATE", tmp_path / "status.json")
    slot(tmp_path, "a")
    slot(tmp_path, "b")
    process, server_log = SimpleNamespace(pid=1), Mock()
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    stop = Mock()
    monkeypatch.setattr(manager, "_stop_server", stop)
    candidate = (SimpleNamespace(pid=2), Mock())
    restored = (SimpleNamespace(pid=3), Mock())
    starts = iter((candidate, restored))
    monkeypatch.setattr(manager, "_start_server", lambda path: next(starts))
    monkeypatch.setattr(
        manager,
        "_verify_candidate_data",
        lambda key: {"ready": False, "coverage": 0.0},
    )

    active, returned_process, returned_log, ok = manager._update(
        "a", process, server_log, "2026-09-10 19:20")

    assert (active, returned_process.pid, ok) == ("a", 3, False)
    state = manager._read_state()
    assert state["status"] == "data_not_ready"
    assert state["data_ready"] is False
    assert state["invalid_slot"] == "b"


def test_partial_candidate_is_published_and_keeps_retry_state(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "SLOTS", tmp_path / "slots")
    monkeypatch.setattr(manager, "STATE", tmp_path / "status.json")
    slot(tmp_path, "a")
    slot(tmp_path, "b")
    process, server_log = SimpleNamespace(pid=1), Mock()
    monkeypatch.setattr(manager.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    stop = Mock()
    monkeypatch.setattr(manager, "_stop_server", stop)
    new_process, new_log = SimpleNamespace(pid=42), Mock()
    monkeypatch.setattr(manager, "_start_server", lambda path: (new_process, new_log))
    monkeypatch.setattr(manager, "_verify_candidate_data", lambda key: {
        "ready": False, "publishable": True, "partial": True,
        "coverage": 0.93, "ready_symbols": 5174, "expected_symbols": 5550,
    })

    active, returned_process, returned_log, ok = manager._update(
        "a", process, server_log, "2026-09-11 19:20")

    assert (active, returned_process, returned_log, ok) == ("b", new_process, new_log, True)
    state = manager._read_state()
    assert state["status"] == "serving_partial"
    assert state["partial_update"] is True
    assert state["data_ready"] is False
    assert state["data_readiness"]["ready_symbols"] == 5174
    stop.assert_called_once_with(process, server_log)


def test_partial_snapshot_retries_same_schedule_key_over_weekend(tmp_path, monkeypatch):
    state_path = tmp_path / "status.json"
    state_path.write_text(
        '{"partial_update": true, "last_attempt_key": "2026-09-11 19:20"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(manager, "STATE", state_path)
    monkeypatch.setattr(manager, "AUTO_UPDATE_ENABLED", True)
    monkeypatch.setattr(manager, "UPDATE_TRIGGER", tmp_path / "missing.trigger")

    now = datetime(2026, 9, 12, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    assert manager._requested_update(now) == "2026-09-11 19:20"


def test_partial_snapshot_does_not_starve_newer_scheduled_trade_day(tmp_path, monkeypatch):
    state_path = tmp_path / "status.json"
    state_path.write_text(
        '{"partial_update": true, "last_attempt_key": "manual 2026-09-12T05:38:59Z"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(manager, "STATE", state_path)
    monkeypatch.setattr(manager, "AUTO_UPDATE_ENABLED", True)
    monkeypatch.setattr(manager, "UPDATE_TIMES", ("15:20", "18:40"))
    monkeypatch.setattr(manager, "UPDATE_TRIGGER", tmp_path / "missing.trigger")
    now = datetime(2026, 10, 15, 19, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert manager._requested_update(now) == "2026-10-15 18:40"


def test_readiness_universe_excludes_92_and_no_trade(tmp_path, monkeypatch):
    import polars as pl

    monkeypatch.setattr(manager, "READINESS_DATA_DIR", tmp_path)
    path = tmp_path / "kline_daily/date=2026-09-11/part.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame({
        "symbol": ["000001.SZ", "920001.BJ", "000002.SZ", "000003.SZ"],
        "volume": [100.0, 100.0, 0.0, 100.0],
        "amount": [1000.0, 1000.0, 0.0, 0.0],
    }).write_parquet(path)

    symbols, error = manager._read_trade_symbols("20260911")

    assert error is None
    assert symbols == ("000001", "000003")
