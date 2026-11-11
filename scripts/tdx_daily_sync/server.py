"""Ubuntu-side coordinator. Market data is writable only during maintenance."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import re
import subprocess
import sys
from datetime import date, datetime, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
DATA = BASE / "runtime/data"
RUNS = DATA / "import_reports/tdx_daily_sync"
CN = ZoneInfo("Asia/Shanghai")
ROLES = ("app", "frozen-pair-shadow", "decision-pair-shadow", "repaired-pair-shadow", "rank-pair-shadow")
APP = "tickpanel-v035-app-1"
PYTHON = "/app/.venv/bin/python"


def now() -> datetime:
    return datetime.now(CN)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def docker(*args: str, timeout: int = 30) -> str:
    return subprocess.check_output(["docker", *args], timeout=timeout, text=True).strip()


def info(role: str) -> dict:
    return json.loads(docker("inspect", f"tickpanel-v035-{role}-1"))[0]


def api(path: str, **query):
    suffix = "?" + urlencode(query) if query else ""
    with urlopen(f"http://127.0.0.1:3018/{path}{suffix}", timeout=15) as response:
        return json.load(response)


def idle() -> bool:
    jobs = api("api/pipeline/jobs", limit=20)
    return not jobs.get("active_id") and not any(j["status"] in {"running", "pending"} for j in jobs["jobs"]) and not api("api/financials/status")["syncing"]


def market_hashes(day: date) -> dict:
    values = {}
    for name in ("kline_daily", "kline_minute"):
        path = DATA / name / f"date={day}" / "part.parquet"
        if path.exists():
            with path.open("rb") as stream:
                values[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            values[name] = None
    return values


def job_container(image: str, command: list[str], *, writable: bool = False, name: str | None = None):
    cmd = ["docker", "run", "--rm", "--network", "none", "--read-only", "--memory", "4g",
           "--memory-swap", "4g", "--cpus", "2", "--tmpfs", "/tmp:size=512m,mode=1777",
           "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "POLARS_MAX_THREADS=2", "-e", "OMP_NUM_THREADS=2",
           "-e", "NUMBA_CACHE_DIR=/tmp/numba", "-v", f"{DATA}:/app/data" + ("" if writable else ":ro"),
           "-v", f"{CODE}:/tdx-sync:ro"]
    if name:
        cmd += ["--name", name]
    return cmd + [image, PYTHON, *command]


def run_container(image: str, command: list[str], out: Path, log_name: str, *, writable: bool = False, name: str | None = None):
    cmd = job_container(image, command, writable=writable, name=name)
    if not writable:
        index = cmd.index(image)
        cmd[index:index] = ["-v", f"{out}:/run-receipts"]
    with (out / log_name).open("x", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=2400 if writable else 240, check=True)


def restore(out: Path, stopped: list[str], images: dict) -> None:
    if stopped:
        core = [role for role in stopped if role != "rank-pair-shadow"]
        if core:
            subprocess.run(["python3", str(BASE / "scripts/restore_panel_services.py"),
                            "--maintenance-record", str(out / "maintenance.json"), "--roles", *core],
                           check=True, timeout=420)
        if "rank-pair-shadow" in stopped:
            docker("start", "tickpanel-v035-rank-pair-shadow-1", timeout=45)
    assert api("health")["status"] == "ok"
    for role in stopped:
        live = info(role)
        assert live["State"]["Running"] and not live["State"]["OOMKilled"] and live["Image"] == images[role]


def audit(image: str, day: date, out: Path, tag: str):
    run_container(image, ["/tdx-sync/audit_market_minute_coverage.py", "--data-dir", "/app/data",
                          "--start", str(day), "--end", str(day), "--markets", "SH", "SZ", "BJ",
                          "--output-dir", f"/run-receipts/{tag}-coverage"], out, f"{tag}-coverage.log")


def cross_source(day: date, out: Path, failures: list[dict]) -> None:
    decisions = []
    if failures:
        code = out / "minute-failures.json"
        assert len(failures) <= 32
        # The already configured app provides the authorized data source
        # connections. The subprocess only reads the supplied evidence.
        copied = f"/tmp/tdx-sync-{out.name}"
        docker("cp", str(CODE), f"{APP}:{copied}")
        try:
            result = subprocess.run(["docker", "exec", "-i", "-e", "PYTHONDONTWRITEBYTECODE=1", APP,
                                     PYTHON, f"{copied}/worker.py", "reconcile", str(day)],
                                    input=code.read_bytes(), capture_output=True, timeout=420, check=True)
            decisions = json.loads(result.stdout)
            assert {x["symbol"] for x in decisions} == {x["symbol"] for x in failures}
        finally:
            docker("exec", APP, "rm", "-r", "--", copied)
    save(out / "peer-evidence.json", decisions)


def verify(day: date, out: Path, staged: dict, coverage_dir: str) -> dict:
    receipt = read(out / "published.json")
    coverage = read(out / coverage_dir / "report.json")["summary"]
    with (out / coverage_dir / "missing_or_partial.csv").open(encoding="utf-8", newline="") as stream:
        remaining = {f"{r['code']}.{r['market']}" for r in csv.DictReader(stream)}
    known = {r["symbol"] for r in read(out / "minute-failures.json")}
    if not remaining <= known or sum(r["expected_active_stock_days"] for r in coverage) != staged["minute"]["expected"]:
        raise ValueError("Coverage audit found unverified post-publication gaps or a changed universe")
    samples = []
    entries = read(out / "staged/report.json")
    by_market = {}
    for item in entries:
        by_market.setdefault(item["symbol"].split(".")[1], item["symbol"])
    for symbol in by_market.values():
        bars = api("api/kline/minute", symbol=symbol, date=str(day), live="false")
        assert bars["source"] == "local" and len(bars["rows"]) == 240
        times = [row["datetime"][11:16] for row in bars["rows"]]
        assert times == sorted(set(times)) and "11:30" in times and "13:00" not in times
        samples.append(symbol)
    finance = api("api/financials/status")["tables"]
    if staged["financial"]["status"] == "staged":
        expected = staged["financial"]["package"]["period"]
        for table, rows in receipt["financial_rows_added"].items():
            if rows:
                assert finance[table]["latest_date"] >= expected
                endpoint = table.replace("_", "-")
                sample = staged["financial"]["tables"][table]["sample_symbol"]
                reported = api(f"api/financials/{endpoint}", symbol=sample)["data"]
                assert sum(r["period_end"] == expected and r["symbol"] == sample for r in reported) == 1
    return {"published": receipt, "market_coverage": coverage, "minute_api_samples": samples,
            "coverage_directory": coverage_dir, "remaining_stock_days": sorted(remaining)}


def execute(day: date, out: Path, dry_run: bool) -> None:
    def state(status: str, **extra):
        save(out / "state.json", {"status": status, "date": str(day), "checked_at": now().isoformat(), **extra})

    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / ".maintenance.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            state("busy")
            return
        if not dry_run and (day != now().date() or not time(17) <= now().time() < time(18, 35)):
            state("outside_publish_window")
            return
        if not dry_run and not idle():
            state("waiting_pipeline")
            return
        image = info("app")["Image"]
        before = market_hashes(day)
        if read(out / "upload.json")["daily_sha256_at_capture"] != before["kline_daily"]:
            state("waiting_pipeline", reason="daily_reference_updated_after_capture")
            return
        state("validating")
        audit(image, day, out, "before")
        try:
            run_container(image, ["/tdx-sync/worker.py", "stage", str(day), "--out", "/run-receipts"],
                          out, "staging.log")
        except subprocess.CalledProcessError:
            if before == market_hashes(day):
                raise
            state("waiting_pipeline", reason="partitions_updated_during_validation")
            return
        staged = read(out / "staging.json")
        failures = read(out / "minute-failures.json")
        if before != market_hashes(day):
            state("waiting_pipeline", staging=staged)
            return
        if dry_run:
            state("validated_read_only", staging=staged)
            return
        if len(failures) > 32:
            state("waiting_minute_source", staging=staged)
            return
        if failures and not staged["minute"]["accepted"] and all(row["reason"] in {
            "source_empty", "source_incomplete", "source_pending", "source_unavailable"
        } for row in failures):
            state("waiting_minute_source", staging=staged)
            return
        finance = staged["financial"]
        if not (staged["minute"]["accepted"] or failures or any(t["rows"] for t in finance["tables"].values())):
            state("already_current" if finance["status"] == "staged" else "financial_pending", staging=staged)
            return
        state("cross_checking", staging=staged)
        cross_source(day, out, failures)
        if not idle() or before != market_hashes(day) or now().time() >= time(18, 35):
            state("waiting_pipeline", staging=staged)
            return
        previous = {role: info(role) for role in ROLES}
        assert all(v["State"]["Running"] and not v["State"]["OOMKilled"] for v in previous.values())
        images = {role: value["Image"] for role, value in previous.items()}
        assert images["app"] == image
        save(out / "maintenance.json", {"images": images, "all_application_writer_roles": ROLES,
                                         "started_at": now().isoformat()})
        stopped = []
        name = f"tickpanel-tdx-sync-{out.name}"
        state("stopping_writers", staging=staged)
        try:
            for role in ROLES:
                stopped.append(role)
                docker("stop", "--timeout", "60", f"tickpanel-v035-{role}-1", timeout=90)
                assert not info(role)["State"]["Running"]
            state("publishing", staging=staged)
            target = Path("/app/data") / out.relative_to(DATA)
            try:
                run_container(image, ["/tdx-sync/worker.py", "publish", str(day), "--out", str(target)],
                              out, "publish.log", writable=True, name=name)
            finally:
                result = subprocess.run(["docker", "inspect", name], capture_output=True, timeout=15)
                if result.returncode == 0 and json.loads(result.stdout)[0]["State"]["Running"]:
                    docker("stop", "--timeout", "60", name, timeout=90)
        finally:
            state("restoring_services", staging=staged)
            restore(out, stopped, images)
        state("auditing", staging=staged)
        for tag in ("after", "after-retry"):
            final_hashes = market_hashes(day)
            audit(image, day, out, tag)
            if market_hashes(day) == final_hashes:
                break
        else:
            raise ValueError("Daily or minute partition changed during both readback audits")
        verification = verify(day, out, staged, f"{tag}-coverage")
        if market_hashes(day) != final_hashes:
            raise ValueError("Market partition changed during panel readback")
        verification["market_partition_sha256"] = final_hashes
        state("completed" if finance["status"] == "staged" else "minute_completed_financial_pending",
              staging=staged, verification=verification)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "init", "start", "run", "status"))
    parser.add_argument("day", type=date.fromisoformat)
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    day = args.day
    if args.mode == "plan":
        image = info("app")["Image"]
        result = json.loads(subprocess.check_output(
            job_container(image, ["/tdx-sync/worker.py", "plan", str(day)]), text=True, timeout=60
        ))
        if result["status"] == "ready" and not args.dry_run and not idle():
            result = {"status": "waiting_pipeline", "date": str(day)}
        print(json.dumps(result))
        return
    if not re.fullmatch(r"[a-f0-9]{16}", args.run_id or ""):
        raise ValueError("Invalid scheduled run id")
    out = RUNS / f"date={day}" / args.run_id
    if args.mode == "init":
        out.mkdir(parents=True, exist_ok=False)
    elif args.mode == "status":
        path = out / "state.json"
        state = read(path) if path.is_file() else {"status": "launching"}
        print(json.dumps(state))
    elif args.mode == "start":
        assert (out / "upload.json").is_file() and not (out / "supervisor.log").exists()
        command = [sys.executable, str(Path(__file__).resolve()), "run", str(day), args.run_id]
        if args.dry_run:
            command.append("--dry-run")
        with (out / "supervisor.log").open("x", encoding="utf-8") as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        save(out / "supervisor.json", {"pid": proc.pid})
    else:
        try:
            execute(day, out, args.dry_run)
        except Exception as exc:
            previous = read(out / "state.json") if (out / "state.json").is_file() else {}
            save(out / "state.json", {"status": "failed", "date": str(day), "checked_at": now().isoformat(),
                                     "previous_stage": previous.get("status"), "error_type": type(exc).__name__})
            raise


if __name__ == "__main__":
    main()
