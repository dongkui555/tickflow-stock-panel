"""Durable minute/recalculation work, executed in bounded units by the pipeline slot."""

from __future__ import annotations

import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from app.market_time import cn_now

_lock = threading.RLock()
_worker_lock = threading.Lock()
_last_queue = "history"
_MAX_NO_PROGRESS_ATTEMPTS = 8
_LABELS = {
    "minute": "分钟增量补差", "adjustment": "复权历史重算",
    "enriched_dates": "历史指标补算", "regime": "历史市场环境重算",
    "index_daily": "指数日K增量补缺",
}


def _path(data_dir):
    return Path(data_dir) / "user_data" / "maintenance_queue.json"


def _read(data_dir):
    path = _path(data_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _write(data_dir, rows):
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def _minute_gate_dates(data_dir, row):
    """Return closed trading dates in one durable minute job, oldest first."""
    if row.get("kind") != "minute":
        return []
    options = row.get("options") or {}
    try:
        end = datetime.fromisoformat(options.get("as_of", "")).date()
        lookback = max(0, int(options.get("days", 0) or 0))
    except (TypeError, ValueError):
        return []
    start = end - timedelta(days=lookback)
    daily = Path(data_dir) / "kline_daily"
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
        if any((daily / f"date={start + timedelta(days=offset)}").glob("*.parquet"))
    ]


def _reset_minute_gate(row):
    for key in ("minute_gate_date", "minute_gate_pending", "minute_gate_dates",
                "minute_gate_history", "minute_gate_legacy_recheck"):
        row.pop(key, None)


def _prepare_minute_gate(data_dir, row, size):
    """Prepare one date-scoped minute batch and its durable gate state.

    The queue remains item-based for compatibility, but the worker receives
    only one closed trading date at a time.  ``minute_gate_pending`` tracks
    the unresolved symbols for the current date; ``row['items']`` is kept for
    the full window and is cleared only after the last date passes.
    """
    dates = sorted(set(row.get("minute_gate_dates") or []) |
                   set(_minute_gate_dates(data_dir, row)))
    if not dates:
        return None

    if row.get("minute_gate_date") not in dates:
        # A legacy multi-day row may already have removed symbols after a
        # combined-window check. Recheck those symbols on the later date so
        # the new barrier cannot inherit an unsafe completion count.
        old_pending = list(row.get("items", []))
        known = sorted(set(row.get("known_items", row.get("items", []))))
        if len(dates) > 1 and int(row.get("completed", 0) or 0) > 0:
            row["items"] = known
            row["total"] = len(known)
            row["completed"] = 0
            row["minute_gate_pending"] = old_pending
            row["minute_gate_legacy_recheck"] = True
        else:
            row["minute_gate_pending"] = list(row.get("items", []))
        row["minute_gate_dates"] = dates
        row["minute_gate_date"] = dates[0]
        row["minute_gate_history"] = list(row.get("minute_gate_history", []))
    else:
        row["minute_gate_dates"] = dates
        pending = set(row.get("minute_gate_pending", row.get("items", [])))
        row["minute_gate_pending"] = [s for s in row.get("items", []) if s in pending]

    # A successful prior batch can leave the current date empty between ticks.
    # Advance the cursor without making a source request.
    while not row["minute_gate_pending"]:
        index = dates.index(row["minute_gate_date"])
        if index >= len(dates) - 1:
            return None
        row["minute_gate_history"].append({
            "date": row["minute_gate_date"],
            "status": "succeeded",
        })
        row["minute_gate_date"] = dates[index + 1]
        row["minute_gate_pending"] = list(row.get("items", []))

    target_date = row["minute_gate_date"]
    options = dict(row.get("options") or {})
    options.update({
        "as_of": f"{target_date}T15:00",
        "days": 0,
        "target_date": target_date,
        "gate_position": dates.index(target_date) + 1,
        "gate_total_dates": len(dates),
    })
    batch = row["minute_gate_pending"][:size]
    snapshot = dict(row, items=batch)
    snapshot["options"] = options
    snapshot["minute_gate_dates"] = dates
    return snapshot


def _coalesce_minute_rows(rows):
    """Merge duplicate pending minute jobs for the same sync scope.

    ``as_of`` changes on every scheduled run, so it cannot be part of the
    identity of a durable gap-repair job. Running jobs remain immutable; only
    pending/retry rows are merged here.
    """
    groups = {}
    for row in rows:
        if row.get("kind") != "minute" or row.get("status") not in {"pending", "retry"}:
            continue
        scope = (row.get("options") or {}).get("scope", "")
        groups.setdefault(scope, []).append(row)
    remove = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        master = max(group, key=lambda row: (int(row.get("completed", 0)), row.get("updated_at", "")))
        known = set(master.get("known_items", master.get("items", [])))
        done = known - set(master.get("items", []))
        for row in group:
            known.update(row.get("known_items", row.get("items", [])))
        remaining = set()
        for row in group:
            remaining.update(row.get("items", []))
        remaining -= done
        master["known_items"] = sorted(known)
        master["items"] = sorted(remaining)
        master["total"] = len(known)
        master["completed"] = len(known - remaining)
        master["message"] = f"已合并重复分钟任务; 已确认 {master['completed']}/{master['total']} 只"
        # Keep the newest requested cutoff while retaining the largest lookback.
        options = dict(master.get("options") or {})
        for row in group:
            candidate = row.get("options") or {}
            if candidate.get("as_of", "") > options.get("as_of", ""):
                options["as_of"] = candidate["as_of"]
            options["days"] = max(int(options.get("days", 0) or 0), int(candidate.get("days", 0) or 0))
        master["options"] = options
        # A wider/newer window must restart its date cursor. Completed symbols
        # stay completed only when the whole durable window had finished.
        _reset_minute_gate(master)
        master["retry_at"] = min((row.get("retry_at") or cn_now().isoformat()) for row in group)
        master["updated_at"] = cn_now().isoformat()
        remove.update(id(row) for row in group if row is not master)
    return [row for row in rows if id(row) not in remove]


def enqueue(data_dir, kind, items, **options):
    if kind not in _LABELS:
        raise ValueError("Unknown maintenance task")
    items = sorted(set(str(item) for item in items))
    if not items:
        return None
    with _lock:
        rows = _read(data_dir)
        rows = _coalesce_minute_rows(rows)
        from app.services.regime_queue_audit import requires_review
        review = kind == "regime" and requires_review(options)
        if review:
            # Repeated mtime detection must not resurrect the same unverified history.
            held = {d for r in rows if r["kind"] == "regime" for d in r.get("review_items", [])}
            items = [d for d in items if d not in held]
            if not items:
                return None
        # A running unit owns an immutable snapshot. Concurrent new work gets its own entry.
        for row in rows:
            if (row["kind"] == kind and row["options"] == options
                    and row["status"] in {"pending", "retry"} and not review):
                known = row.get("known_items", row["items"]) if kind in {"minute", "index_daily"} else row["items"]
                extra = sorted(set(items) - set(known))
                row["items"] += extra
                row["known_items"] = sorted(set(known) | set(items))
                row["total"] += len(extra)
                if kind == "minute" and row.get("minute_gate_pending") is not None:
                    row["minute_gate_pending"] = sorted(set(row["minute_gate_pending"]) | set(extra))
                _write(data_dir, rows)
                return row["id"]
        row = {
            "id": uuid.uuid4().hex[:10], "kind": kind, "label": _LABELS[kind],
            "queue": "minute" if kind == "minute" else "history",
            "status": "paused_review" if review else "pending",
            "items": [] if review else items, "options": options,
            "review_items": items if review else [],
            "known_items": items,
            "total": len(items), "completed": 0, "batch_progress": 0,
            "attempts": 0, "job_id": None,
            "message": "缺少输入变化证据, 待核验且不自动重算" if review else "等待执行槽", "error": None,
            "created_at": cn_now().isoformat(), "updated_at": cn_now().isoformat(),
        }
        rows.append(row)
        # Never evict unfinished work.
        terminal = [r for r in rows if r["status"] == "succeeded"][-30:]
        rows = [r for r in rows if r["status"] != "succeeded"] + terminal
        _write(data_dir, rows)
        return row["id"]


def status(data_dir):
    with _lock:
        result = []
        for row in _read(data_dir):
            public = {k: v for k, v in row.items() if k not in {
                "items", "known_items", "review_items", "skipped_items", "unknown_items", "resolved_items",
                "linked_review_items", "minute_gate_pending", "minute_gate_dates",
                "authorization_pending", "identity_pending",
            }}
            pending = row.get("minute_gate_pending")
            dates = row.get("minute_gate_dates") or _minute_gate_dates(data_dir, row)
            if row.get("kind") == "minute" and dates:
                gate_date = row.get("minute_gate_date") or dates[0]
                gate_remaining = len(pending) if isinstance(pending, list) else len(row["items"])
                public.update(
                    gate_date=gate_date,
                    gate_position=dates.index(gate_date) + 1 if gate_date in dates else 1,
                    gate_total_dates=len(dates),
                    gate_completed=int(row.get("minute_gate_completed", 0) or 0),
                    gate_total_items=len(row.get("known_items", row.get("items", []))),
                    gate_remaining=gate_remaining,
                    window_remaining=len(row["items"]),
                )
            result.append(dict(
                public, remaining=(public.get("gate_remaining", len(row["items"]))),
                review_count=len(row.get("review_items", [])),
                skipped=len(row.get("skipped_items", [])),
                unknown=len(row.get("unknown_items", [])),
                resolved=len(row.get("resolved_items", [])),
                authorization_count=len(row.get("authorization_pending", [])),
                identity_pending_count=len(row.get("identity_pending", [])),
            ))
        return result


def resume_authorized(data_dir, task_id=None):
    """Release queue rows after their exact conflict requests are approved."""
    from app.services import minute_repair_authorization as authorization

    # Audit can contain a historical grant event even when a request was later
    # resolved/reopened. Use the durable decision state as the authority.
    approved = authorization.approved_ids(data_dir)
    resumed = 0
    with _lock:
        rows = _read(data_dir)
        for row in rows:
            if task_id is not None and row["id"] != str(task_id) and row.get("authorization_task_id") != str(task_id):
                continue
            pending = row.get("authorization_pending", [])
            if not pending:
                continue
            remaining = [entry for entry in pending if entry.get("id") not in approved]
            if len(remaining) == len(pending):
                continue
            row["authorization_pending"] = remaining
            if not remaining and row.get("status") == "paused_authorization":
                row.update(status="pending", attempts=0, retry_at=cn_now().isoformat(),
                           error=None, message="冲突替换授权已生效，按原任务范围继续校验")
                resumed += 1
            row["updated_at"] = cn_now().isoformat()
        if resumed:
            _write(data_dir, rows)
    return resumed


def _audit_locked(data_dir, rows, *, recheck=False):
    from app.services.regime_queue_audit import reconcile
    updated = deepcopy(rows)
    reports = reconcile(data_dir, updated, recheck=recheck)
    if not reports:
        return rows, None
    stamp = cn_now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = Path(data_dir) / "user_data" / "maintenance_audits" / stamp
    folder.mkdir(parents=True)
    # Persist exact pre-mutation bytes and the findings before committing the new queue.
    (folder / "queue-before.json").write_bytes(_path(data_dir).read_bytes())
    report = {"created_at": cn_now().isoformat(), "tasks": reports,
              "scope": "queue_only; no prices, indicators, source stamps or forecasts written"}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for task in updated:
        if any(r["task_id"] == task["id"] for r in reports):
            task.update(audit_report=str((folder / "report.json").relative_to(data_dir)),
                        updated_at=cn_now().isoformat())
    return updated, report


def audit_regime(data_dir):
    """Caller must own the pipeline write slot (or run before scheduler startup)."""
    if not _worker_lock.acquire(blocking=False):
        raise RuntimeError("maintenance_worker_busy")
    try:
        with _lock:
            rows, report = _audit_locked(data_dir, _read(data_dir), recheck=True)
            if report:
                _write(data_dir, rows)
            return report or {"tasks": []}
    finally:
        _worker_lock.release()


def checkpoint(data_dir, task_id, completed, *, retry_after=0):
    """A foreground index batch checkpoints work before handing it to the worker."""
    with _lock:
        rows = _read(data_dir)
        row = next(r for r in rows if r['id'] == task_id)
        if row['status'] == 'running':
            return
        done = set(completed) & set(row['items'])
        row['items'] = [s for s in row['items'] if s not in done]
        row['completed'] += len(done)
        row.update(status=(row['status'] if row['status'] == 'retry' else 'pending') if row['items'] else 'succeeded',
                   message=f"已完成 {row['completed']}/{row['total']} 只, 未完成 {len(row['items'])} 只",
                   updated_at=cn_now().isoformat())
        if row['items'] and retry_after:
            due = (cn_now() + timedelta(seconds=retry_after)).isoformat()
            row.update(status='retry', retry_at=max(row.get('retry_at', ''), due))
        if not row['items']:
            row.update(batch_progress=100, error=None)
        _write(data_dir, rows)


def recover(data_dir):
    """Only called at application scheduler startup, never by read-only API requests."""
    with _lock:
        rows = _read(data_dir)
        rows = _coalesce_minute_rows(rows)
        for row in rows:
            if row["status"] == "running":
                row.update(status="pending", batch_progress=0, message="重启后增量续跑")
            if row.get("kind") == "index_daily" and row.get("status") in {"pending", "retry"}:
                from app.services.index_const import CORE_INDEX_SYMBOLS
                core = set(CORE_INDEX_SYMBOLS)
                daily = [item for item in row.get("items", []) if item in core]
                broad = [item for item in row.get("items", []) if item not in core]
                if broad:
                    row["review_items"] = sorted(set(row.get("review_items", [])) | set(broad))
                    row["items"] = daily
                    row.update(
                        status="pending" if daily else "paused_review",
                        message=(f"每日核心指数待补 {len(daily)} 只；"
                                 f"全量历史指数 {len(broad)} 只已移出日常队列"),
                        updated_at=cn_now().isoformat(),
                    )
        rows, _ = _audit_locked(data_dir, rows)
        if rows:
            _write(data_dir, rows)


def tick(data_dir, runner, execute, *, now=None, allow_history=True, allow_bulk_minute=True):
    """runner(fn, label, on_started) uses the existing tracked, exclusive execution slot."""
    global _last_queue
    now = now or cn_now()
    if not _worker_lock.acquire(blocking=False):
        return False
    try:
        with _lock:
            rows = _read(data_dir)
            # Legacy/unknown work cannot run even if inserted after startup.
            from app.services.regime_queue_audit import requires_review
            for row in rows:
                if (row["kind"] == "regime" and row["status"] in {"pending", "retry"}
                        and requires_review(row.get("options", {}))):
                    row["review_items"] = sorted(set(row["items"]) | set(row.get("review_items", [])))
                    row.update(items=[], status="paused_review", message="缺少输入变化证据, 已暂停历史重算")
                    _write(data_dir, rows)
            derived_pending = any(r["kind"] in {"adjustment", "enriched_dates"}
                                  and r["status"] != "succeeded" for r in rows)
            due = [r for r in rows if r["status"] in {"pending", "retry"}
                   and r.get("retry_at", "") <= now.isoformat()
                   and (allow_history or r["queue"] == "minute")
                   and (allow_bulk_minute or not (
                       r["kind"] == "minute" and r.get("options", {}).get("scope") == "all"
                   ))
                   and not (derived_pending and r["kind"] == "regime")]

            def priority(row):
                # The latest market day must become usable before older minute
                # holes. Historical minute repair remains durable, but cannot
                # delay today's chart and model inputs.
                freshness = 0.0
                if row["kind"] == "minute":
                    try:
                        as_of = datetime.fromisoformat(row.get("options", {}).get("as_of", "")).timestamp()
                        # A historical rescan is intentionally date-ordered:
                        # finish the oldest unresolved day before opening a
                        # newer one.  Intraday/daily freshness keeps its
                        # existing latest-day priority.
                        if (row.get("options", {}).get("mode") == "history_gap_rescan"):
                            freshness = as_of
                        else:
                            freshness = -as_of
                    except (TypeError, ValueError):
                        pass
                return (row["queue"] == _last_queue, row["kind"] == "regime", freshness, row["created_at"])

            due.sort(key=priority)
            if not due:
                return False
            row = due[0]
            task_id = row["id"]
            # Local StockDB can return a few hundred symbols per maintenance
            # unit. A larger outer batch reduces repeated full-day parquet
            # validation while the provider still applies its own request
            # limits internally.
            minute_batch = max(10, min(1000, int(os.getenv("MAINTENANCE_MINUTE_BATCH_SIZE", "50"))))
            size = {"minute": minute_batch, "adjustment": 20, "enriched_dates": 1,
                    "regime": 10, "index_daily": 20}[row["kind"]]
            gate_snapshot = _prepare_minute_gate(data_dir, row, size) if row["kind"] == "minute" else None
            if gate_snapshot is not None:
                batch = gate_snapshot["items"]
                snapshot = gate_snapshot
                # Persist initialization/cursor advancement before the worker
                # starts so a restart resumes the same date, not the window.
                _write(data_dir, rows)
            else:
                batch = row["items"][:size]
                snapshot = dict(row, items=batch)

        def update(**fields):
            with _lock:
                current = _read(data_dir)
                item = next(r for r in current if r["id"] == task_id)
                item.update(fields, updated_at=cn_now().isoformat())
                _write(data_dir, current)

        started = False
        error = None
        result = None

        def mark_started():
            nonlocal started
            from app.services.pipeline_jobs import job_store
            started = True
            update(status="running", job_id=job_store.active_id(), batch_progress=0,
                   message=f"本批 {len(batch)} 项", error=None)

        def task(on_progress=None):
            def progress(stage, pct, msg, **kw):
                update(batch_progress=pct, message=msg)
                if on_progress:
                    on_progress(stage, pct, msg, **kw)
            nonlocal error, result
            try:
                result = execute(snapshot, progress)
                return result
            except Exception as exc:
                # Identity failures are deterministic data-quality findings,
                # not transient provider failures. Keep the batch intact for
                # review instead of sending it through an endless retry loop.
                from app.services.security_identity import IdentityError
                if isinstance(exc, IdentityError):
                    error = str(exc) or type(exc).__name__
                    result = {
                        "completed_items": [],
                        "error": "identity_pending",
                        "identity_pending": [{"reason": str(exc) or type(exc).__name__}],
                    }
                    return result
                error = getattr(exc, "reason", None) or type(exc).__name__
                raise

        success = runner(task, row["label"], on_started=mark_started)
        if not started:
            return False
        _last_queue = row["queue"]
        with _lock:
            current = _read(data_dir)
            item = next(r for r in current if r["id"] == task_id)
            completed = batch if success else []
            if isinstance(result, dict) and "completed_items" in result:
                completed = [v for v in result["completed_items"] if v in batch]
                success = len(completed) == len(batch)
                error = result.get("error")
                if "validation_failures" in result:
                    item["validation_failures"] = result["validation_failures"]
                if "source_errors" in result:
                    item["source_errors"] = result["source_errors"]
                if "repair" in result and isinstance(result["repair"], dict):
                    repair = result["repair"]
                    item["repair_summary"] = {
                        key: repair.get(key, 0)
                        for key in (
                            "inserted_rows", "updated_rows", "duplicate_rows", "rejected_rows",
                            "written_rows", "repaired_stock_days",
                        )
                        if key in repair
                    }
                authorization_required = result.get("authorization_required", [])
                if authorization_required:
                    existing = {entry.get("id"): entry for entry in item.get("authorization_pending", [])}
                    existing.update({entry.get("id"): entry for entry in authorization_required if entry.get("id")})
                    item["authorization_pending"] = list(existing.values())
                    item["authorization_task_id"] = (
                        authorization_required[0].get("task_id")
                        or item.get("authorization_task_id")
                        or item["id"]
                    )
                if result.get("identity_pending"):
                    item["identity_pending"] = result["identity_pending"]
            identity_blocked = isinstance(result, dict) and bool(result.get("identity_pending"))
            skipped = set(result.get("skipped_items", [])) & set(completed) if isinstance(result, dict) else set()
            if skipped:
                item["skipped_items"] = sorted(set(item.get("skipped_items", [])) | skipped)
            unknown = set(result.get("unknown_items", [])) & set(completed) if isinstance(result, dict) else set()
            if unknown:
                item["unknown_items"] = sorted(set(item.get("unknown_items", [])) | unknown)
            gate_date = (snapshot.get("options") or {}).get("target_date") if item["kind"] == "minute" else None
            gate_dates = snapshot.get("minute_gate_dates", []) if gate_date else []
            if gate_date and gate_dates:
                # Date gate: a symbol completed for the current day is not
                # removed from the durable task until every gated day passes.
                stage_pending = list(item.get("minute_gate_pending", item["items"]))
                stage_pending = [v for v in stage_pending if v not in set(completed)]
                source_retry = isinstance(result, dict) and result.get("source_retry", False)
                batch_remaining = [v for v in batch if v in stage_pending]
                untouched = [v for v in stage_pending if v not in set(batch_remaining)]
                item["minute_gate_pending"] = batch_remaining + untouched if source_retry else untouched + batch_remaining
                item["minute_gate_completed"] = len(item["items"]) - len(item["minute_gate_pending"])
                stage_complete = not item["minute_gate_pending"]
                gate_index = gate_dates.index(gate_date)
                if identity_blocked:
                    item.update(
                        status="paused_review", attempts=0, retry_at=None,
                        error="identity_pending",
                        message=(f"身份校验未通过，已暂停；保留 {len(item['minute_gate_pending'])} 只待核验"),
                    )
                    success = False
                elif item.get("authorization_pending"):
                    item.update(status="paused_authorization", attempts=0, retry_at=None,
                                error=None, message=(
                                    f"日期闸门 {gate_date} 已完成可自动入库项；"
                                    f"新增冲突替换 {len(item['authorization_pending'])} 项待授权"
                                ))
                    success = False
                elif stage_complete and gate_index < len(gate_dates) - 1:
                    item.setdefault("minute_gate_history", []).append({
                        "date": gate_date,
                        "status": "succeeded",
                        "completed": len(item["items"]),
                    })
                    next_date = gate_dates[gate_index + 1]
                    item["minute_gate_date"] = next_date
                    item["minute_gate_pending"] = list(item["items"])
                    item["minute_gate_completed"] = 0
                    item.update(status="pending", attempts=0, batch_progress=100, error=None,
                                message=f"日期闸门 {gate_date} 已完成，进入 {next_date}；当前任务仍有 {len(item['items'])} 只待处理")
                    if item.get("options", {}).get("scope") == "all":
                        cooldown = max(5, int(os.getenv("MAINTENANCE_BULK_MINUTE_COOLDOWN_SECONDS", "15")))
                        item["retry_at"] = (max(now, cn_now()) + timedelta(seconds=cooldown)).isoformat()
                    success = True
                elif stage_complete:
                    # Last date: now the item-level completion has real meaning.
                    final_items = set(item["items"])
                    item["items"] = []
                    item["minute_gate_completed"] = len(final_items)
                    item["completed"] += len(final_items) - len(set(item.get("skipped_items", [])) & final_items)
                    item.update(status="paused_review" if item.get("review_items") else "succeeded",
                                attempts=0, batch_progress=100, error=None,
                                message=f"日期闸门 {gate_date} 完成；窗口内 {item['completed']}/{item['total']} 只通过")
                    success = True
                elif success:
                    item.update(status="pending", attempts=0, batch_progress=100, error=None,
                                message=f"日期闸门 {gate_date} 已完成本批，当前日期剩余 {len(item['minute_gate_pending'])} 只")
                    if item.get("options", {}).get("scope") == "all":
                        cooldown = max(5, int(os.getenv("MAINTENANCE_BULK_MINUTE_COOLDOWN_SECONDS", "15")))
                        item["retry_at"] = (max(now, cn_now()) + timedelta(seconds=cooldown)).isoformat()
                else:
                    # Keep the same date at the head; only unresolved symbols
                    # rotate, so a bad source cannot unlock a later date.
                    attempts = 1 if completed else item["attempts"] + 1
                    delay = 60 if untouched else 60 * min(180, 5 * 2 ** min(attempts - 1, 6))
                    if isinstance(result, dict):
                        delay = max(delay, result.get("retry_after_seconds", 0))
                    if attempts >= _MAX_NO_PROGRESS_ATTEMPTS and not completed:
                        item.update(
                            status="paused_review", attempts=0, retry_at=None,
                            error="no_progress_limit",
                            message=(f"连续 {attempts} 次无进展，已暂停；"
                                     f"保留 {len(item['minute_gate_pending'])} 只待核验，不再自动循环"),
                        )
                    else:
                        item.update(status="retry", attempts=attempts, error=error or "任务未完成",
                                    retry_at=(max(now, cn_now()) + timedelta(seconds=delay)).isoformat(),
                                    message=f"日期闸门 {gate_date} 部分完成; 当日剩余 {len(item['minute_gate_pending'])} 只")
                        if source_retry:
                            item["message"] = f"日期闸门 {gate_date} 等待主备源; {delay} 秒后重试当日剩余"
                    success = False
            else:
                item["items"] = [v for v in item["items"] if v not in set(completed)]
                item["completed"] += len(completed) - len(skipped)
                if success:
                    item.update(status="pending" if item["items"] else "paused_review" if item.get("review_items") else "succeeded",
                                attempts=0, batch_progress=100, error=None)
                    if (item["items"] and item["kind"] == "minute"
                            and item.get("options", {}).get("scope") == "all"):
                        # Release the shared heavy-job slot between full-market
                        # batches so user-triggered analysis/sync work can run.
                        cooldown = max(5, int(os.getenv("MAINTENANCE_BULK_MINUTE_COOLDOWN_SECONDS", "15")))
                        item["retry_at"] = (max(now, cn_now()) + timedelta(seconds=cooldown)).isoformat()
                        item["message"] = (
                            f"已确认 {item['completed']}/{item['total']} 只; "
                            f"{cooldown} 秒后继续后台补差"
                        )
            if identity_blocked and not (gate_date and gate_dates):
                item.update(
                    status="paused_review", attempts=0, retry_at=None,
                    error="identity_pending",
                    message=(f"身份校验未通过，已暂停；保留 {len(item['items'])} 只待核验"),
                )
            if (not success and not (gate_date and gate_dates)
                    and not item.get("authorization_pending") and not identity_blocked):
                attempts = 1 if completed and item["kind"] == "minute" else item["attempts"] + 1
                # One unavailable stock must not starve the rest of the market.
                remaining = [v for v in batch if v not in set(completed)]
                untouched = [v for v in item["items"] if v not in set(remaining)]
                source_retry = isinstance(result, dict) and result.get('source_retry', False)
                # Pending gateway requests must be revisited after cooldown, not
                # rotated into thousands of new preparation requests.
                item["items"] = remaining + untouched if source_retry else untouched + remaining
                delay = 60 if untouched else 60 * min(180, 5 * 2 ** min(attempts - 1, 6))
                if item['kind'] in {'index_daily', 'minute'}:
                    # Rotate through one full pass promptly, then back off even when
                    # every remaining item keeps returning unusable data.
                    first_pass = max(1, (item['total'] + size - 1) // size)
                    if attempts > first_pass:
                        delay = max(delay, 60 * min(180, 5 * 2 ** min(attempts - first_pass - 1, 6)))
                if isinstance(result, dict):
                    delay = max(delay, result.get('retry_after_seconds', 0))
                if attempts >= _MAX_NO_PROGRESS_ATTEMPTS and not completed:
                    item.update(
                        status="paused_review", attempts=0, retry_at=None,
                        error="no_progress_limit",
                        message=(f"连续 {attempts} 次无进展，已暂停；"
                                 f"保留 {len(item['items'])} 只待核验，不再自动循环"),
                    )
                else:
                    item.update(status="retry", attempts=attempts, error=error or "任务未完成",
                                retry_at=(max(now, cn_now()) + timedelta(seconds=delay)).isoformat(),
                                message=f"部分完成 {item['completed']}/{item['total']}, 未完成 {len(item['items'])}; 已落盘数据保留, 等待增量重试")
                    if source_retry:
                        item['message'] = f"主备分钟源冷却中; {delay} 秒后续试本批; 已确认 {item['completed']}/{item['total']} 只"
            if item.get("authorization_pending") and not (gate_date and gate_dates):
                item.update(status="paused_authorization", attempts=0, retry_at=None, error=None,
                            message=f"新增冲突替换 {len(item['authorization_pending'])} 项待授权；其余已通过校验项已自动处理")
            item["updated_at"] = cn_now().isoformat()
            if item["kind"] == "regime" and completed:
                for held in current:
                    resolved = set(held.get("review_items", [])) & set(completed)
                    if held["kind"] != "regime" or not resolved:
                        continue
                    held["review_items"] = [d for d in held["review_items"] if d not in resolved]
                    held["resolved_items"] = sorted(set(held.get("resolved_items", [])) | resolved)
                    if not held["items"] and not held["review_items"]:
                        held["status"] = "reviewed"
                    held["message"] = f"另有明确修复任务已处理 {len(held['resolved_items'])} 日; 待核验 {len(held['review_items'])} 日"
            _write(data_dir, current)
        return success
    finally:
        _worker_lock.release()
