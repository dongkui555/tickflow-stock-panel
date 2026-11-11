"""Durable, exact stock-day grants. Authorization never substitutes validation."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from app.market_time import cn_now

OPERATION = "replace_minute"


def canonical(symbol):
    from app.services.security_identity import exchange_for_code

    value = str(symbol).strip().upper()
    if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", value):
        raise ValueError("Authorization requires a market-qualified stock identity")
    code, exchange = value.split(".")
    if exchange_for_code(code, "stock") != exchange:
        raise ValueError("Authorization stock market mismatch")
    return value


def scope_id(start, end, symbols=None, stock_days=None, task_id=None):
    if task_id:
        return str(task_id)
    payload = [str(start), str(end), sorted(symbols) if symbols is not None else None,
               sorted((s, str(d)) for s, d in stock_days) if stock_days is not None else None]
    return "minute-" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:24]


@contextmanager
def _db(root):
    path = Path(root) / "user_data" / "minute_repair_authorizations.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, source TEXT NOT NULL,
                operation TEXT NOT NULL, symbol TEXT NOT NULL, trade_date TEXT NOT NULL,
                state TEXT NOT NULL, created_at TEXT NOT NULL, approved_at TEXT,
                UNIQUE(task_id, source, operation, symbol, trade_date)
            );
            CREATE TABLE IF NOT EXISTS audit (
                seq INTEGER PRIMARY KEY, decision_id TEXT NOT NULL, event TEXT NOT NULL,
                actor TEXT NOT NULL, at TEXT NOT NULL, backup TEXT
            );
            CREATE TABLE IF NOT EXISTS scopes (
                task_id TEXT NOT NULL, source TEXT NOT NULL, operation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(task_id, source, operation)
            );
        """)
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _key(task, source, symbol, day, operation=OPERATION):
    if not task or not re.fullmatch(r"[a-z0-9_-]+", source):
        raise ValueError("Missing task or invalid source")
    return (str(task), source, operation, canonical(symbol), date.fromisoformat(str(day)).isoformat())


def _decision(conn, key, *, approve, actor):
    row = conn.execute("SELECT * FROM decisions WHERE task_id=? AND source=? AND operation=? "
                       "AND symbol=? AND trade_date=?", key).fetchone()
    stamp = cn_now().isoformat()
    if row is None:
        ident = uuid.uuid4().hex
        conn.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
                     (ident, *key, "approved" if approve else "pending", stamp, stamp if approve else None))
    else:
        ident = row["id"]
        if row["state"] == "approved":
            return ident
        if not approve and row["state"] == "pending":
            return ident
        if approve:
            conn.execute("UPDATE decisions SET state='approved', approved_at=? WHERE id=?", (stamp, ident))
        else:
            # A previously validated import may have resolved the request, but
            # a later source can legitimately re-create the same exact conflict.
            conn.execute("UPDATE decisions SET state='pending', approved_at=NULL WHERE id=?", (ident,))
    conn.execute("INSERT INTO audit(decision_id,event,actor,at) VALUES (?,?,?,?)",
                 (ident, "granted" if approve else "requested", actor, stamp))
    return ident


def grant(root, task, source, stock_days, *, actor="explicit_request", operation=OPERATION):
    keys = [_key(task, source, s, d, operation) for s, d in stock_days]
    if not keys:
        raise ValueError("Empty authorization scope")
    with _db(root) as conn:
        ids = [_decision(conn, k, approve=True, actor=actor) for k in keys]
    return ids[0]


def allowed(root, task, source, symbol, day, *, operation=OPERATION):
    key = _key(task, source, symbol, day, operation)
    with _db(root) as conn:
        row = conn.execute("SELECT id FROM decisions WHERE task_id=? AND source=? AND operation=? "
                           "AND symbol=? AND trade_date=? AND approved_at IS NOT NULL", key).fetchone()
    return row["id"] if row else None


def grants(root, task):
    with _db(root) as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM decisions WHERE task_id=? AND approved_at IS NOT NULL", (task,))]


def approved_ids(root):
    with _db(root) as conn:
        return {row["id"] for row in conn.execute(
            "SELECT id FROM decisions WHERE approved_at IS NOT NULL AND state IN ('approved','resolved')"
        ).fetchall()}


def open_scope(root, task, source, *, operation=OPERATION):
    """Register a task/source scope and return whether this is its first source.

    The first-source check is scoped by task and operation.  A source change
    inside an existing minute-repair task is deliberately not treated as a new
    authorization, so a new source still requires an explicit decision.
    """
    if not task or not re.fullmatch(r"[a-z0-9_-]+", source):
        raise ValueError("Missing task or invalid source")
    with _db(root) as conn:
        existing = conn.execute("SELECT 1 FROM scopes WHERE task_id=? AND operation=?",
                                (task, operation)).fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO scopes(task_id,source,operation,created_at) VALUES (?,?,?,?)",
            (task, source, operation, cn_now().isoformat()),
        )
    return existing is None


def request_replacement(root, task, source, symbol, day):
    with _db(root) as conn:
        return _decision(conn, _key(task, source, symbol, day), approve=False, actor="validator")


def approve(root, request_ids, *, task=None):
    ids = list(dict.fromkeys(request_ids))
    if not ids:
        raise ValueError("No conflict requests selected")
    with _db(root) as conn:
        rows = [conn.execute("SELECT * FROM decisions WHERE id=?", (i,)).fetchone() for i in ids]
        if any(r is None for r in rows):
            raise ValueError("Unknown conflict request; refresh before approving")
        if task is not None and any(r["task_id"] != str(task) for r in rows):
            raise ValueError("Conflict request is outside the selected task scope")
        for row in rows:
            _decision(conn, tuple(row[k] for k in ("task_id", "source", "operation", "symbol", "trade_date")),
                      approve=True, actor="user_confirmation")
        return [dict(conn.execute("SELECT * FROM decisions WHERE id=?", (i,)).fetchone()) for i in ids]


def used(root, grant_id, symbol, day, *, backup):
    with _db(root) as conn:
        row = conn.execute("SELECT * FROM decisions WHERE id=? AND approved_at IS NOT NULL",
                           (grant_id,)).fetchone()
        if row is None or row["symbol"] != canonical(symbol) or row["trade_date"] != str(day):
            raise ValueError("Replacement audit does not match authorization")
        conn.execute("INSERT INTO audit(decision_id,event,actor,at,backup) VALUES (?,?,?,?,?)",
                     (grant_id, "used", "minute_repair", cn_now().isoformat(), backup))


def resolve(root, task, source, symbol, day):
    """A non-conflicting import may make an old approval request unnecessary."""
    with _db(root) as conn:
        rows = conn.execute("SELECT id FROM decisions WHERE task_id=? AND source=? AND symbol=? AND trade_date=? "
                            "AND state='pending'", (task, source, canonical(symbol), str(day))).fetchall()
        for row in rows:
            conn.execute("UPDATE decisions SET state='resolved' WHERE id=?", (row["id"],))
            conn.execute("INSERT INTO audit(decision_id,event,actor,at) VALUES (?,?,?,?)",
                         (row["id"], "resolved", "validated_import", cn_now().isoformat()))


def snapshot(root):
    with _db(root) as conn:
        pending = [dict(r) for r in conn.execute("SELECT * FROM decisions WHERE state='pending' ORDER BY created_at")]
        audit = [dict(r) for r in conn.execute(
            "SELECT d.task_id,d.source,d.operation,d.symbol,d.trade_date,a.* "
            "FROM audit a JOIN decisions d ON d.id=a.decision_id ORDER BY a.seq DESC LIMIT 100")]
    return {"pending": pending, "audit": audit}
