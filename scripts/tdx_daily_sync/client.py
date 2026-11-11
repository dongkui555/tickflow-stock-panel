"""Windows scheduled entry point for the TDX post-close import."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import subprocess
import time as wait
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from uuid import uuid4

from source import day_records, fetch_financial, sha


CN = timezone(timedelta(hours=8))
OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
           "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]
RETRY = {"waiting_daily", "calendar_unavailable", "busy", "waiting_pipeline", "waiting_minute_source",
         "financial_pending"}
ACTIVE = {"launching", "validating", "cross_checking", "stopping_writers", "publishing", "auditing",
          "restoring_services"}


def run(args: list[str], timeout: int = 90) -> str:
    return subprocess.check_output(args, text=True, encoding="utf-8", timeout=timeout).strip()


def remote(host: str, base: PurePosixPath, mode: str, day: date, *extra: str) -> str:
    entry = str(base / "scripts/tdx_daily_sync/server.py")
    return run(["ssh", *OPTIONS, host, "python3", entry, mode, str(day), *extra])


def log(root: Path, status: str, **extra) -> None:
    record = {"checked_at": datetime.now(CN).isoformat(), "status": status, **extra}
    line = json.dumps(record, ensure_ascii=False)
    with (root / "runs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def upload(host: str, base: PurePosixPath, day: date, path: Path, requested: dict, local: Path,
           dry_run: bool, logs: Path) -> str:
    run_id = uuid4().hex[:16]
    capture = path / "minutes.jsonl.gz"
    counts = {}
    with gzip.open(capture, "wt", encoding="utf-8", compresslevel=3) as stream:
        for symbol in requested["symbols"]:
            data, status = day_records(local, symbol, day)
            counts[status] = counts.get(status, 0) + 1
            stream.write(json.dumps({"symbol": symbol, "status": status,
                                     "records": base64.b64encode(data).decode("ascii")}) + "\n")
    try:
        finance = fetch_financial(day, local, path)
        finance_error = None
    except (OSError, ValueError, OverflowError) as exc:
        finance = None
        finance_error = type(exc).__name__
    manifest = {"date": str(day), "minutes_sha256": sha(capture), "minute_capture_counts": counts,
                "daily_sha256_at_capture": requested["daily_sha256"], "financial": finance,
                "financial_source_error": finance_error}
    (path / "upload.json").write_text(json.dumps(manifest), encoding="utf-8")
    remote(host, base, "init", day, run_id)
    target = base / "runtime/data/import_reports/tdx_daily_sync" / f"date={day}" / run_id
    files = [str(capture)]
    if finance:
        files.append(str(path / finance["name"]))
    run(["scp", *OPTIONS, *files, f"{host}:{target}/"], timeout=180)
    run(["scp", *OPTIONS, str(path / "upload.json"), f"{host}:{target}/"], timeout=30)
    log(logs, "uploaded", run_id=run_id, counts=counts, financial_period=finance and finance["period"],
        financial_source_error=finance_error, evidence=str(target))
    remote(host, base, "start", day, run_id, *(["--dry-run"] if dry_run else []))
    return run_id


def monitor(host: str, base: PurePosixPath, day: date, run_id: str, logs: Path) -> dict:
    previous = None
    stop_at = wait.monotonic() + 3600
    while wait.monotonic() < stop_at:
        try:
            state = json.loads(remote(host, base, "status", day, run_id))
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            log(logs, "status_temporarily_unavailable", run_id=run_id, error=type(exc).__name__)
            wait.sleep(30)
            continue
        if state["status"] != previous:
            staged = state.get("staging", {})
            financial = staged.get("financial", {})
            log(logs, state["status"], run_id=run_id, minute=staged.get("minute"),
                financial_status=financial.get("status"),
                financial_additions={k: v["rows"] for k, v in financial.get("tables", {}).items()},
                verification=state.get("verification"), error=state.get("error_type"))
            previous = state["status"]
        if previous not in ACTIVE:
            return state
        wait.sleep(15)
    raise TimeoutError("Check the detached Ubuntu supervisor for this run")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdx", type=Path, required=True, help="TDX vipdoc directory")
    parser.add_argument("--host", default="tickpanel-ubuntu")
    parser.add_argument("--server-root", default="/data/tick-stock-panel-v035")
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--day", type=date.fromisoformat)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = PurePosixPath(args.server_root)
    args.logs.mkdir(parents=True, exist_ok=True)
    current = datetime.now(CN)
    day = args.day or current.date()
    if not args.dry_run and (day != current.date() or not time(17) <= current.time() < time(18, 35)):
        log(args.logs, "outside_schedule_window", day=str(day))
        return 0
    if not args.tdx.is_dir():
        log(args.logs, "tdx_cache_unavailable", day=str(day))
        return 1
    while True:
        try:
            plan = json.loads(remote(args.host, base, "plan", day, *(["--dry-run"] if args.dry_run else [])))
            if plan["status"] == "closed":
                log(args.logs, "closed", day=str(day))
                return 0
            if plan["status"] == "ready":
                with TemporaryDirectory(prefix="tdx-sync-") as temporary:
                    run_id = upload(args.host, base, day, Path(temporary), plan, args.tdx, args.dry_run, args.logs)
                state = monitor(args.host, base, day, run_id, args.logs)
                status = state["status"]
            else:
                status = plan["status"]
                log(args.logs, status, day=str(day))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            log(args.logs, "transport_or_input_error", day=str(day), error=type(exc).__name__)
            return 1
        if status in {"already_current", "completed", "validated_read_only"}:
            return 0
        if status not in RETRY or args.dry_run or datetime.now(CN).time() >= time(18, 30):
            return 1
        wait.sleep(300)


if __name__ == "__main__":
    raise SystemExit(main())
