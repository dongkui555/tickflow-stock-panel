"""Run the local StockDB reader and refresh an inactive snapshot on schedule."""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from zoneinfo import ZoneInfo

ROOT = Path(os.environ.get("STOCKDB_WORK_DIR", "/work"))
SLOTS = ROOT / "slots"
STATE = ROOT / "manager-status.json"
SERVER = os.environ.get("STOCKDB_SERVER_BINARY", "/opt/stockdb/stockdb")
UPDATER = os.environ.get("STOCKDB_UPDATER_BINARY", "/opt/stockdb/stockdb-updater")
UPDATE_TIMES = tuple(sorted(os.environ.get("STOCKDB_UPDATE_TIMES", "15:20,18:40").split(",")))
RETRY_SECONDS = max(60, int(os.environ.get("STOCKDB_UPDATE_RETRY_SECONDS", "600")))
UPDATER_TIMEOUT_SECONDS = max(
    120, int(os.environ.get("STOCKDB_UPDATER_TIMEOUT_SECONDS", "900"))
)
AUTO_UPDATE_ENABLED = os.environ.get("STOCKDB_AUTO_UPDATE_ENABLED", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
UPDATE_TRIGGER = Path(os.environ.get("STOCKDB_UPDATE_TRIGGER", str(ROOT / "update.trigger")))
READINESS_MODE = os.environ.get("STOCKDB_UPDATE_READINESS_MODE", "full").strip().lower()
READINESS_DATA_DIR = Path(os.environ.get("STOCKDB_UPDATE_DATA_DIR", "/app-data"))
READINESS_MIN_BARS = max(240, int(os.environ.get("STOCKDB_UPDATE_READINESS_MIN_BARS", "240")))
READINESS_TIMEOUT_SECONDS = max(
    1.0, float(os.environ.get("STOCKDB_UPDATE_READINESS_TIMEOUT_SECONDS", "5"))
)
READINESS_MAX_WORKERS = max(1, int(os.environ.get("STOCKDB_UPDATE_READINESS_MAX_WORKERS", "16")))
READINESS_SAMPLE_SYMBOLS = tuple(
    symbol.strip().split(".", 1)[0]
    for symbol in os.environ.get(
        "STOCKDB_UPDATE_READINESS_SYMBOLS",
        "000001,000333,000651,000858,300750,600000,600519,601318,688981",
    ).split(",")
    if symbol.strip()
)
TZ = ZoneInfo("Asia/Shanghai")
STOP = False


def _write_state(**changes):
    try:
        current = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    except (OSError, ValueError):
        current = {}
    current.update(changes, updated_at=datetime.now(TZ).isoformat())
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, STATE)
    return current


def _read_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    except (OSError, ValueError):
        return {}


def _ready(slot: Path) -> bool:
    return (slot / "stockdb.conf").is_file() and (slot / "sync_url.txt").is_file() \
        and (slot / "data" / "CURRENT").is_file()


def _bootstrap_inactive(active: Path, inactive: Path):
    if _ready(inactive):
        return
    if inactive.exists():
        shutil.rmtree(inactive)
    shutil.copytree(active, inactive)


def _update_log_failed(path: Path, offset: int) -> bool:
    """The updater can return zero after a partial manifest failure; inspect this run."""
    manifest_failed, file_failed = _update_log_errors(path, offset)
    return manifest_failed or file_failed


def _update_log_errors(path: Path, offset: int) -> tuple[bool, bool]:
    """Separate fatal manifest errors from recoverable per-file misses."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            text = handle.read()
    except OSError:
        return True, True
    manifest_failed = bool(re.search(r"manifest failed|请求失败:", text, re.IGNORECASE))
    file_failed = bool(re.search(r"\[file\].*(?:failed|失败)", text, re.IGNORECASE))
    return manifest_failed, file_failed


def _target_trade_date(schedule_key: str) -> str | None:
    match = re.match(r"^(?:manual\s+)?(\d{4})-?(\d{2})-?(\d{2})", schedule_key.strip())
    return "".join(match.groups()) if match else None


def _regular_minute_count(rows, trade_date: str) -> int:
    seen = set()
    for row in rows if isinstance(rows, list) else ():
        payload = row[1] if isinstance(row, list) and len(row) > 1 else row
        if not isinstance(payload, dict):
            continue
        stamp = str(payload.get("date", ""))
        if not stamp.startswith(trade_date) or len(stamp) < 12:
            continue
        minute = stamp[8:12]
        if "0931" <= minute <= "1130" or "1301" <= minute <= "1500":
            seen.add(minute)
    return len(seen)


def _read_trade_symbols(trade_date: str) -> tuple[tuple[str, ...], str | None]:
    """Read the day's expected universe from the app's persisted daily K snapshot."""
    path = READINESS_DATA_DIR / "kline_daily" / f"date={trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}" / "part.parquet"
    if not path.is_file():
        return (), f"daily_universe_missing:{path}"
    try:
        import polars as pl

        frame = pl.read_parquet(path)
        if "symbol" not in frame.columns or "volume" not in frame.columns:
            return (), "daily_universe_missing_trade_columns"
        # StockDB is not the source for 北交所 92xxxx.  A zero-volume and
        # zero-amount daily bar is also a no-trade day, not a missing minute
        # session.  Keep malformed one-sided values in the gate so they fail
        # visibly instead of being mistaken for a suspension.
        traded = pl.col("volume").fill_null(0) > 0
        if "amount" in frame.columns:
            traded = traded | (pl.col("amount").fill_null(0) > 0)
        frame = frame.filter(traded).with_columns(
            pl.col("symbol").cast(pl.String).str.extract(r"(\d{6})", 1).alias("_code")
        ).filter(~pl.col("_code").str.starts_with("92"))
        symbols = tuple(sorted(set(frame.get_column("_code").drop_nulls().to_list())))
    except Exception as exc:
        return (), f"daily_universe_read_failed:{type(exc).__name__}"
    return symbols, None


def _check_symbol_minutes(symbol: str, trade_date: str) -> tuple[str, int, str | None]:
    query = quote(f"分钟k:{symbol}:{trade_date}*", safe="")
    try:
        with urlopen(
            f"http://127.0.0.1:7899/?cmd=get&t={query}&json=1",
            timeout=READINESS_TIMEOUT_SECONDS,
        ) as response:
            rows = json.loads(response.read().decode("utf-8"))
        bars = _regular_minute_count(rows, trade_date)
        return symbol, bars, None if bars >= READINESS_MIN_BARS else "minute_bars_below_required"
    except Exception as exc:
        return symbol, 0, f"request_failed:{type(exc).__name__}"


def _verify_candidate_data(schedule_key: str) -> dict:
    """Require the scheduled trade day's minute bars before publishing a snapshot.

    A clean updater manifest proves file transfer completeness. This independent
    check catches a source that publishes a complete snapshot before today's
    minute records have actually landed.
    """
    trade_date = _target_trade_date(schedule_key)
    if not trade_date:
        return {
            "ready": True,
            "scope": "manifest_only",
            "reason": "manual_trigger_without_trade_date",
        }
    if READINESS_MODE == "sample":
        expected = READINESS_SAMPLE_SYMBOLS
        scope = "scheduled_day_sample"
        universe_error = None
    else:
        expected, universe_error = _read_trade_symbols(trade_date)
        scope = "full_market_traded_universe"
    if universe_error:
        return {
            "ready": False,
            "scope": scope,
            "trade_date": trade_date,
            "required_bars": READINESS_MIN_BARS,
            "checked_symbols": 0,
            "ready_symbols": 0,
            "coverage": 0.0,
            "failures": {},
            "reason": universe_error,
        }
    results = []
    with ThreadPoolExecutor(max_workers=READINESS_MAX_WORKERS) as pool:
        for result in pool.map(lambda symbol: _check_symbol_minutes(symbol, trade_date), expected):
            results.append(result)
    failures = {symbol: reason for symbol, _bars, reason in results if reason}
    ready_symbols = [symbol for symbol, _bars, reason in results if reason is None]
    ready = bool(expected) and len(ready_symbols) == len(expected)
    return {
        "ready": ready,
        # A partial snapshot is safe to publish when it contains at least one
        # independently verified traded symbol. Missing symbols stay on the
        # app's Promax/TDX/offline failover path and are retried separately.
        "publishable": bool(ready_symbols),
        "partial": bool(ready_symbols) and not ready,
        "scope": scope,
        "trade_date": trade_date,
        "required_bars": READINESS_MIN_BARS,
        "expected_symbols": len(expected),
        "checked_symbols": len(results),
        "ready_symbols": len(ready_symbols),
        "coverage": round(len(ready_symbols) / len(results), 4) if results else 0.0,
        "failure_count": len(failures),
        "failure_sample": dict(list(failures.items())[:50]),
    }


def _server_ready(timeout=20) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 7899), timeout=1):
                return True
        except OSError:
            time.sleep(.25)
    return False


def _start_server(slot: Path):
    log = open(ROOT / "server.log", "ab", buffering=0)  # noqa: SIM115
    try:
        process = subprocess.Popen([SERVER, "stockdb.conf"], cwd=slot, stdout=log, stderr=subprocess.STDOUT)
    except OSError:
        log.close()
        raise
    if not _server_ready() or process.poll() is not None:
        _stop_server(process, log)
        raise RuntimeError(f"StockDB reader failed to start from slot {slot.name}")
    return process, log


def _stop_server(process, log):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if log:
        log.close()


def _start_server_with_retry(slot: Path):
    """Keep a transient vendor startup failure recoverable without claiming readiness."""
    while not STOP:
        try:
            return _start_server(slot)
        except (OSError, RuntimeError) as exc:
            _write_state(status="reader_unavailable", error="reader_start_failed",
                         error_type=type(exc).__name__, active_slot=slot.name,
                         data_ready=False, pid=None, retry_after_seconds=RETRY_SECONDS,
                         reader_retry_at=(datetime.now(TZ) + timedelta(seconds=RETRY_SECONDS)).isoformat())
            for elapsed in range(0, RETRY_SECONDS, 5):
                if STOP:
                    return None, None
                time.sleep(min(5, RETRY_SECONDS - elapsed))
    return None, None


def _update(active_name: str, process, server_log, schedule_key: str):
    inactive_name = "b" if active_name == "a" else "a"
    active, inactive = SLOTS / active_name, SLOTS / inactive_name
    previous_state = _read_state()
    was_partial = bool(previous_state.get("partial_update"))
    _write_state(status="copying", scheduled_for=schedule_key, active_slot=active_name)
    update_log_path = ROOT / f"updater-{inactive_name}.log"
    if update_log_path.exists() and update_log_path.stat().st_size > 20 * 1024 * 1024:
        update_log_path.replace(update_log_path.with_suffix(".previous.log"))
    started = datetime.now(TZ)
    log_offset = update_log_path.stat().st_size if update_log_path.exists() else 0
    _write_state(status="updating", last_attempt_key=schedule_key, last_attempt_at=started.isoformat())
    _bootstrap_inactive(active, inactive)
    timed_out = False
    launch_error_type = None
    try:
        with update_log_path.open("ab", buffering=0) as update_log:
            result = subprocess.run(
                [UPDATER, "--threads", "1", "--retries", "2"], cwd=inactive,
                stdout=update_log, stderr=subprocess.STDOUT,
                timeout=UPDATER_TIMEOUT_SECONDS, check=False,
            )
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        exit_code = -1
        timed_out = True
    except OSError as exc:
        # Keep the current reader available when a mounted updater is missing
        # or lacks an executable bit; the manager will retry the failed slot.
        exit_code = -1
        launch_error_type = type(exc).__name__
    manifest_failed, file_failed = _update_log_errors(update_log_path, log_offset)
    if exit_code != 0 or not _ready(inactive) or manifest_failed:
        _write_state(
            status="serving_partial" if was_partial else "update_failed",
            active_slot=active_name, partial_update=was_partial,
            data_ready=False if was_partial else previous_state.get("data_ready", False),
            exit_code=exit_code,
            error=("updater_unavailable" if launch_error_type else
                   "updater_timeout" if timed_out else "updater_returned_failure"),
            error_type=launch_error_type,
            invalid_slot=inactive_name, update_log_failed=manifest_failed,
            updater_file_failures=file_failed,
            retry_after_seconds=RETRY_SECONDS,
            updater_timeout_seconds=UPDATER_TIMEOUT_SECONDS,
        )
        return active_name, process, server_log, False

    _write_state(status="switching", candidate_slot=inactive_name, error_type=None)
    _stop_server(process, server_log)
    try:
        new_process, new_log = _start_server(inactive)
    except (OSError, RuntimeError) as exc:
        old_process, old_log = _start_server_with_retry(active)
        if old_process is None:
            return active_name, None, None, False
        _write_state(
            status="serving_partial" if was_partial else "switch_failed",
            error="candidate_reader_start_failed", error_type=type(exc).__name__,
            active_slot=active_name, partial_update=was_partial, data_ready=False,
            retry_after_seconds=RETRY_SECONDS,
        )
        return active_name, old_process, old_log, False
    readiness = _verify_candidate_data(schedule_key)
    if not readiness.get("publishable"):
        _stop_server(new_process, new_log)
        old_process, old_log = _start_server_with_retry(active)
        if old_process is None:
            return active_name, None, None, False
        _write_state(
            status="serving_partial" if was_partial else "data_not_ready",
            error="candidate_snapshot_missing_scheduled_day_minutes",
            active_slot=active_name,
            candidate_slot=None,
            invalid_slot=inactive_name,
            data_ready=False,
            partial_update=was_partial,
            data_readiness=readiness,
            retry_after_seconds=RETRY_SECONDS,
            pid=old_process.pid,
        )
        return active_name, old_process, old_log, False
    if not readiness["ready"]:
        # Do not discard a usable candidate merely because a minority of
        # symbols is late. The candidate is now the serving snapshot; the
        # manager loop keeps retrying the same scheduled date until it becomes
        # complete or the source remains unavailable and is classified by the
        # app's per-symbol coverage policy.
        _write_state(
            status="serving_partial", active_slot=inactive_name,
            last_published_key=schedule_key,
            last_published_at=datetime.now(TZ).isoformat(),
            exit_code=0, error="candidate_snapshot_partial_published",
            candidate_slot=None, invalid_slot=None, update_log_failed=False,
            updater_file_failures=file_failed,
            data_ready=False, data_readiness=readiness,
            partial_update=True, retry_after_seconds=RETRY_SECONDS,
            pid=new_process.pid,
        )
        return inactive_name, new_process, new_log, True
    _write_state(status="serving", active_slot=inactive_name, last_success_key=schedule_key,
                 last_success_at=datetime.now(TZ).isoformat(), exit_code=0, error=None,
                 candidate_slot=None, invalid_slot=None, update_log_failed=False,
                 updater_file_failures=file_failed,
                 data_ready=True, data_readiness=readiness, partial_update=False,
                 last_published_key=schedule_key,
                 last_published_at=datetime.now(TZ).isoformat(), pid=new_process.pid)
    return inactive_name, new_process, new_log, True


def _latest_due(now: datetime) -> str | None:
    if now.weekday() >= 5:
        return None
    due = [f"{now.date()} {value}" for value in UPDATE_TIMES
           if now.strftime("%H:%M") >= value]
    return max(due) if due else None


def _consume_manual_trigger(now: datetime) -> str | None:
    """Claim a one-shot full snapshot refresh request from the writable work directory."""
    if not UPDATE_TRIGGER.exists():
        return None
    try:
        requested = UPDATE_TRIGGER.read_text(encoding="utf-8").strip()
        UPDATE_TRIGGER.unlink()
    except OSError:
        return None
    label = requested[:80] if requested else now.isoformat(timespec="seconds")
    return f"manual {label}"


def _requested_update(now: datetime) -> str | None:
    manual = _consume_manual_trigger(now)
    if manual:
        return manual
    state = _read_state()
    due = _latest_due(now) if AUTO_UPDATE_ENABLED else None
    if AUTO_UPDATE_ENABLED and state.get("partial_update") and state.get("last_attempt_key"):
        # Partial publication is itself useful. Continue updating that same
        # trade date on the manager retry interval, including over weekends,
        # rather than waiting for the next market day and losing the tail.
        previous = str(state["last_attempt_key"])
        previous_date = _target_trade_date(previous)
        if due and (previous_date is None or _target_trade_date(due) > previous_date):
            return due
        return previous
    return due


def _signal(_signum, _frame):
    global STOP
    STOP = True


def main():
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _signal)
    state = _read_state()
    active_name = state.get("active_slot", "a")
    active = SLOTS / active_name
    if not _ready(active) or active_name == state.get("invalid_slot"):
        alternate = SLOTS / ("b" if active_name == "a" else "a")
        if not _ready(alternate):
            raise RuntimeError("No verified StockDB snapshot found in /work/slots/a or /work/slots/b")
        active, active_name = alternate, alternate.name
    process, server_log = _start_server_with_retry(active)
    if process is None:
        _write_state(status="stopped", active_slot=active_name, pid=None)
        return
    partial_update = bool(state.get("partial_update"))
    _write_state(
        status="serving_partial" if partial_update else "serving",
        active_slot=active_name,
        pid=process.pid,
        candidate_slot=None,
        error=None,
        error_type=None,
        reader_retry_at=None,
        auto_update_enabled=AUTO_UPDATE_ENABLED,
        update_trigger=str(UPDATE_TRIGGER),
        partial_update=partial_update,
    )
    try:
        while not STOP:
            if process.poll() is not None:
                _stop_server(process, server_log)
                process, server_log = _start_server_with_retry(SLOTS / active_name)
                if process is None:
                    break
                current_state = _read_state()
                _write_state(
                    status="serving_partial" if current_state.get("partial_update") else "serving",
                    active_slot=active_name, pid=process.pid, recovered_reader=True,
                    error=None, error_type=None, reader_retry_at=None,
                )
            now = datetime.now(TZ)
            due = _requested_update(now)
            state = _read_state()
            attempted = state.get("last_attempt_key") == due
            last_attempt = datetime.fromisoformat(state["last_attempt_at"]) if state.get("last_attempt_at") else None
            retry_due = (last_attempt is None or (now - last_attempt).total_seconds() >= RETRY_SECONDS)
            if due and (state.get("partial_update") or state.get("last_success_key") != due) \
                    and (not attempted or retry_due):
                active_name, process, server_log, _ = _update(
                    active_name, process, server_log, due)
                if process is None:
                    break
            time.sleep(5)
    finally:
        _stop_server(process, server_log)
        _write_state(status="stopped", active_slot=active_name)


if __name__ == "__main__":
    main()
