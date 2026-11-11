"""Persistent per-stock-day minute repair queue.

The queue is deliberately separate from ``maintenance_queue.json``.  The
latter tracks worker jobs, while this SQLite table tracks the actual unit of
work: one stock on one closed trading date.  This keeps retries idempotent and
prevents a source timeout from causing a new scan of the whole history.
"""
from __future__ import annotations

import sqlite3
import time
import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import polars as pl

from app.market_time import cn_now


DB_RELATIVE_PATH = Path("user_data/minute_gap_queue.sqlite")
QUEUE_SCHEMA_VERSION = 2
SOURCE_CHAIN = "qmt,promax,tdx,stockdb,offline"
TERMINAL = {
    "market_closed", "suspended", "suspension_data_conflict",
    "no_trade", "pre_listing", "delisted", "code_changed",
    "identity_unknown", "identity_conflict", "source_unsupported", "unknown",
    "regular_session", "regular_session_close_auction",
}

# Cross-source verification produces states that must survive the next daily
# queue refresh.  They are deliberately separate from the minute retry states:
# a daily-baseline candidate needs a daily-data repair workflow, while a
# quarantine needs review or a new independent source, not another hot retry.
CROSS_SOURCE_PRESERVED = {
    "daily_baseline_repair_candidate", "quarantine", "source_no_record",
    "existing_values_conflict", "regular_session",
    "regular_session_close_auction",
}


def _path(data_dir) -> Path:
    return Path(data_dir) / DB_RELATIVE_PATH


@contextmanager
def _connect(data_dir):
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        conn = None
        try:
            conn = sqlite3.connect(str(path), timeout=20)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=20000")
            _ensure_schema(conn, data_dir)
            break
        except sqlite3.OperationalError as exc:
            if conn is not None:
                conn.close()
            code = getattr(exc, "sqlite_errorcode", 0) & 0xff
            transient = code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_CANTOPEN}
            transient = transient or str(exc).lower() in {
                "database is locked", "database table is locked", "unable to open database file",
            }
            if not transient or attempt == 3:
                raise
            time.sleep(0.1 * (3 ** attempt))
        except Exception:
            if conn is not None:
                conn.close()
            raise
    # Retry setup only; replaying a transaction could duplicate queue changes.
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection, data_dir=None) -> None:
    # Status polling must not mutate the journal or schema of a live queue.
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='minute_gap_queue'"
    ).fetchone()
    if table:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(minute_gap_queue)")}
        if {"asset_type", "exchange", "code"}.issubset(columns):
            return
        _migrate_legacy_schema(conn, data_dir)
        return
    if not table:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS minute_gap_queue (
            asset_type TEXT NOT NULL,
            exchange TEXT NOT NULL,
            code TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            status TEXT NOT NULL,
            expected_minutes INTEGER NOT NULL DEFAULT 240,
            actual_minutes INTEGER NOT NULL DEFAULT 0,
            missing_minutes INTEGER NOT NULL DEFAULT 240,
            reason TEXT NOT NULL DEFAULT 'minute_gap',
            source TEXT NOT NULL DEFAULT '',
            source_chain TEXT NOT NULL DEFAULT 'qmt,promax,tdx,stockdb,offline',
            attempts INTEGER NOT NULL DEFAULT 0,
            next_retry_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(asset_type, exchange, code, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_minute_gap_due
            ON minute_gap_queue(status, trade_date, next_retry_at);
        CREATE INDEX IF NOT EXISTS idx_minute_gap_date
            ON minute_gap_queue(trade_date, status);
        CREATE TABLE IF NOT EXISTS minute_gap_identity_quarantine (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            reason TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        PRAGMA user_version = 2;
        """
    )


def _queue_lookup(data_dir) -> dict[str, str]:
    from app.services.security_identity import build_instrument_lookup

    root = Path(data_dir)
    frames = []
    for dataset in ("instruments", "instruments_history", "instruments_lifecycle"):
        directory = root / dataset
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.parquet")):
            try:
                frame = pl.read_parquet(path)
            except Exception:
                continue
            if "code" in frame.columns or "symbol" in frame.columns:
                frames.append(frame)
    return build_instrument_lookup(frames)


def _parse_queue_identity(data_dir, value, asset_type: str = "stock", lookup: dict[str, str] | None = None):
    from app.services.security_identity import IdentityError, parse_identity

    try:
        identity = parse_identity(
            value,
            asset_type=asset_type,
            lookup=lookup if lookup is not None else _queue_lookup(data_dir),
            require_explicit=True,
        )
        return identity, None
    except IdentityError as exc:
        return None, str(exc)


def _migrate_legacy_schema(conn: sqlite3.Connection, data_dir) -> None:
    """Upgrade v1 code/date rows without guessing an exchange."""
    now = cn_now().replace(tzinfo=None).isoformat(timespec="seconds")
    conn.executescript(
        """
        ALTER TABLE minute_gap_queue RENAME TO minute_gap_queue_v1;
        CREATE TABLE minute_gap_queue (
            asset_type TEXT NOT NULL,
            exchange TEXT NOT NULL,
            code TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            status TEXT NOT NULL,
            expected_minutes INTEGER NOT NULL DEFAULT 240,
            actual_minutes INTEGER NOT NULL DEFAULT 0,
            missing_minutes INTEGER NOT NULL DEFAULT 240,
            reason TEXT NOT NULL DEFAULT 'minute_gap',
            source TEXT NOT NULL DEFAULT '',
            source_chain TEXT NOT NULL DEFAULT 'qmt,promax,tdx,stockdb,offline',
            attempts INTEGER NOT NULL DEFAULT 0,
            next_retry_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(asset_type, exchange, code, trade_date)
        );
        CREATE TABLE IF NOT EXISTS minute_gap_identity_quarantine (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            reason TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_minute_gap_due
            ON minute_gap_queue(status, trade_date, next_retry_at);
        CREATE INDEX IF NOT EXISTS idx_minute_gap_date
            ON minute_gap_queue(trade_date, status);
        """
    )
    lookup = _queue_lookup(data_dir) if data_dir is not None else {}
    for row in conn.execute("SELECT * FROM minute_gap_queue_v1").fetchall():
        identity, reason = _parse_queue_identity(
            data_dir, row["symbol"] if isinstance(row, sqlite3.Row) else row[0], lookup=lookup
        )
        raw_symbol = row["symbol"] if isinstance(row, sqlite3.Row) else row[0]
        trade_date = row["trade_date"] if isinstance(row, sqlite3.Row) else row[1]
        if identity is None:
            conn.execute(
                "INSERT INTO minute_gap_identity_quarantine(original_symbol, trade_date, reason, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(raw_symbol), str(trade_date), reason or "identity_unknown", json.dumps(dict(row), ensure_ascii=False), now),
            )
            continue
        values = dict(row)
        conn.execute(
            """INSERT OR IGNORE INTO minute_gap_queue(
                asset_type, exchange, code, symbol, trade_date, status,
                expected_minutes, actual_minutes, missing_minutes, reason, source,
                source_chain, attempts, next_retry_at, last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                identity.asset_type, identity.exchange, identity.code, identity.symbol,
                str(values["trade_date"]), values["status"], values["expected_minutes"],
                values["actual_minutes"], values["missing_minutes"], values["reason"],
                values["source"], values["source_chain"], values["attempts"],
                values["next_retry_at"], values["last_error"], values["created_at"],
                values["updated_at"],
            ),
        )
    conn.execute("DROP TABLE minute_gap_queue_v1")
    conn.execute("PRAGMA user_version = 2")


def _code(value) -> str:
    from app.services.minute_status import code_of

    return code_of(value)


def _listing_dates(data_dir) -> dict[str, date]:
    """Read listing dates when available; absence must not invent one."""
    path = Path(data_dir) / "instruments" / "instruments.parquet"
    if not path.exists():
        return {}
    try:
        frame = pl.read_parquet(path)
        if not {"symbol", "listing_date"}.issubset(frame.columns):
            return {}
        from app.services.minute_repair import keyed

        parsed = keyed(frame).with_columns(
            pl.coalesce(
                pl.col("listing_date").cast(pl.String).str.to_date("%Y%m%d", strict=False),
                pl.col("listing_date").cast(pl.String).str.to_date("%Y-%m-%d", strict=False),
            ).alias("_listed")
        )
        result = {}
        for row in parsed.filter(pl.col("_listed").is_not_null()).select(["_code", "_listed"]).iter_rows():
            result.setdefault(row[0], row[1])
        return result
    except Exception:
        return {}


def _daily_state(data_dir, day: date):
    """Return raw daily rows, validated references and no-trade codes."""
    path = Path(data_dir) / "kline_daily" / f"date={day}" / "part.parquet"
    if not path.exists():
        return None, {}, set()
    try:
        daily = pl.read_parquet(path)
        from app.services.minute_repair import daily_reference, keyed

        tagged = keyed(daily)
        no_trade = set(tagged.filter(
            (pl.col("volume") == 0) & (pl.col("amount") == 0)
        )["_code"].drop_nulls())
        return daily, daily_reference(daily, data_dir=data_dir), no_trade
    except Exception:
        return None, {}, set()


def _stored(data_dir, day: date):
    path = Path(data_dir) / "kline_minute" / f"date={day}" / "part.parquet"
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_parquet(path)
    except Exception:
        return pl.DataFrame()


def _counts(frame: pl.DataFrame, day: date) -> dict[str, int]:
    if frame.is_empty() or not {"symbol", "datetime"}.issubset(frame.columns):
        return {}
    from app.services.minute_repair import keyed, regular

    try:
        clean = keyed(regular(frame, day))
        return dict(clean.group_by("_code").agg(
            pl.col("datetime").n_unique().alias("n")
        ).iter_rows())
    except Exception:
        return {}


def _upsert(conn, record: dict, now: str) -> None:
    conn.execute(
        """INSERT INTO minute_gap_queue
           (asset_type, exchange, code, symbol, trade_date, status, expected_minutes, actual_minutes,
            missing_minutes, reason, source, source_chain, attempts,
            next_retry_at, last_error, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(asset_type, exchange, code, trade_date) DO UPDATE SET
             symbol=excluded.symbol,
             status=excluded.status, expected_minutes=excluded.expected_minutes,
             actual_minutes=excluded.actual_minutes, missing_minutes=excluded.missing_minutes,
             reason=excluded.reason, source=COALESCE(NULLIF(excluded.source, ''), minute_gap_queue.source),
             source_chain=excluded.source_chain, next_retry_at=excluded.next_retry_at,
             last_error=excluded.last_error, updated_at=excluded.updated_at""",
        (
            record["asset_type"], record["exchange"], record["code"], record["symbol"],
            record["trade_date"], record["status"],
            int(record.get("expected_minutes", 240)), int(record.get("actual_minutes", 0)),
            int(record.get("missing_minutes", 0)), record.get("reason", "minute_gap"),
            record.get("source", ""), record.get("source_chain", SOURCE_CHAIN),
            int(record.get("attempts", 0)), record.get("next_retry_at"),
            record.get("last_error"), record.get("created_at", now), now,
        ),
    )


def ensure_date(data_dir, symbols: Iterable[str], day: date | str) -> dict:
    """Classify one date before any provider request and persist only real work.

    Complete stock-days are removed from the queue.  Terminal observations are
    retained as an audit trail but are never returned by ``pending_batch``.
    """
    day = date.fromisoformat(str(day)[:10]) if not isinstance(day, date) else day
    lookup = _queue_lookup(data_dir)
    identities = []
    unknown = []
    for value in symbols:
        identity, reason = _parse_queue_identity(data_dir, value, lookup=lookup)
        if identity is None:
            unknown.append((str(value), reason or "identity_unknown"))
        elif identity.security_key not in {item.security_key for item in identities}:
            identities.append(identity)
    if not identities and not unknown:
        return {"date": day.isoformat(), "queued": 0, "terminal": 0, "complete": 0}
    daily, refs, no_trade = _daily_state(data_dir, day)
    from app.services.market_status import historical_suspension_status, market_day_status

    market_status = market_day_status(data_dir, day)
    listing = _listing_dates(data_dir)
    from app.services.security_lifecycle import load_lifecycle_index, validate_security_lifecycle
    lifecycle_index = load_lifecycle_index(data_dir)
    stored = _stored(data_dir, day)
    counts = _counts(stored, day)
    from app.services.minute_repair import checked_additions, keyed, qmt_session_is_valid, regular

    current = cn_now().replace(tzinfo=None).isoformat(timespec="seconds")
    queued = terminal = complete = len(unknown)
    with _connect(data_dir) as conn:
        now = current
        for raw_symbol, reason in unknown:
            conn.execute(
                "INSERT INTO minute_gap_identity_quarantine(original_symbol, trade_date, reason, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (raw_symbol, day.isoformat(), reason, "ensure_date", now),
            )
        for identity in identities:
            code = identity.code
            base = {
                "asset_type": identity.asset_type, "exchange": identity.exchange,
                "code": identity.code, "symbol": identity.symbol,
                "trade_date": day.isoformat(), "source_chain": SOURCE_CHAIN,
            }
            if market_status["status"] == "market_closed":
                base.update(
                    status="market_closed", actual_minutes=0, missing_minutes=0,
                    expected_minutes=0, reason=market_status["reason"],
                )
                _upsert(conn, base, current)
                terminal += 1
                continue
            # Lifecycle validation runs before any provider request.  An empty
            # lifecycle index means this is an old/minimal test or import
            # directory; preserve the historical listing-only behavior until a
            # dimension is available instead of inventing a status.
            if lifecycle_index:
                lifecycle = validate_security_lifecycle(
                    code, day, data_dir, index=lifecycle_index,
                )
                if not lifecycle.ok:
                    state = {
                        "not_listed_yet": ("pre_listing", "上市前无行情"),
                        "delisted_on_date": ("delisted", "退市后无行情"),
                        "code_changed": ("code_changed", f"代码已变更为 {lifecycle.successor_symbol or '新证券身份'}"),
                        "market_identity_conflict": ("identity_conflict", "证券市场身份冲突"),
                        "identity_history_unknown": ("identity_unknown", "历史证券身份未核验"),
                        "lifecycle_date_missing": ("identity_unknown", "生命周期有效日期缺失"),
                    }.get(lifecycle.reason)
                    if state:
                        base.update(
                            status=state[0], actual_minutes=0, missing_minutes=0,
                            expected_minutes=0, reason=state[1],
                            last_error=f"lifecycle_reason={lifecycle.reason}",
                        )
                        _upsert(conn, base, current)
                        terminal += 1
                        continue
            if code in listing and day < listing[code]:
                base.update(status="pre_listing", actual_minutes=0, missing_minutes=0,
                            expected_minutes=0, reason="上市前无行情")
                _upsert(conn, base, current)
                terminal += 1
                continue
            if daily is None:
                base.update(status="unknown", reason="daily_partition_missing",
                            last_error="日K分区不存在, 暂不请求分钟源")
                _upsert(conn, base, current)
                terminal += 1
                continue
            daily_codes = set(refs) | no_trade
            suspension = historical_suspension_status(data_dir, code, day)
            if suspension["status"] == "suspended":
                if code in no_trade or code not in daily_codes:
                    base.update(
                        status="suspended", actual_minutes=0, missing_minutes=0,
                        expected_minutes=0, reason=suspension["reason"],
                        last_error=f"suspension_interval={suspension.get('start_date')}"
                    )
                    _upsert(conn, base, current)
                    terminal += 1
                    continue
                base.update(
                    status="suspension_data_conflict", actual_minutes=0,
                    missing_minutes=0, expected_minutes=0,
                    reason="历史停牌记录与日K成交冲突",
                    last_error=suspension["reason"],
                )
                _upsert(conn, base, current)
                terminal += 1
                continue
            if code in no_trade:
                base.update(status="no_trade", actual_minutes=0, missing_minutes=0,
                            expected_minutes=0, reason="日K成交量和成交额均为0")
                _upsert(conn, base, current)
                terminal += 1
                continue
            reference = refs.get(code)
            if reference is None:
                base.update(status="unknown", reason="daily_reference_missing",
                            last_error="日K无可用量价参考, 不把分钟数据当作完整")
                _upsert(conn, base, current)
                terminal += 1
                continue
            part = regular(keyed(stored).filter(pl.col("_code") == code).drop("_code"), day) if not stored.is_empty() else pl.DataFrame()
            actual = int(part["datetime"].n_unique()) if not part.is_empty() else 0
            existing = conn.execute(
                "SELECT status, attempts, next_retry_at, created_at, source, last_error, reason "
                "FROM minute_gap_queue WHERE asset_type=? AND exchange=? AND code=? AND trade_date=?",
                (identity.asset_type, identity.exchange, identity.code, day.isoformat()),
            ).fetchone()
            if existing and existing["status"] == "complete" and existing["source"] == "qmt":
                # A QMT import may be complete with 239 effective bars and
                # must be revalidated with the QMT contract.  Rechecking it
                # as a generic local source incorrectly turns a verified row
                # back into a pending gap because QMT volume is lot-based.
                if qmt_session_is_valid(part, day, reference):
                    base.update(
                        status="complete", actual_minutes=actual,
                        missing_minutes=0, expected_minutes=240,
                        reason="minute_complete",
                        attempts=existing["attempts"],
                        created_at=existing["created_at"],
                        source="qmt", last_error=None,
                    )
                    _upsert(conn, base, current)
                    complete += 1
                    continue
            if existing and existing["status"] == "source_unsupported" and actual < 240:
                base.update(status="source_unsupported", actual_minutes=actual,
                            missing_minutes=max(0, 240 - actual), reason="所有在线源均不覆盖",
                            attempts=existing["attempts"], next_retry_at=None,
                            created_at=existing["created_at"], source=existing["source"],
                            last_error=existing["last_error"])
                _upsert(conn, base, current)
                terminal += 1
                continue
            if existing and existing["status"] in CROSS_SOURCE_PRESERVED and (
                actual < 240 or existing["status"] in {
                    "regular_session", "regular_session_close_auction",
                }
            ):
                # Do not let ensure_date turn a prior third-party verdict back
                # into a generic pending row.  The verdict is the audit trail;
                # a separate repair/review action must explicitly reopen it.
                base.update(
                    status=existing["status"], actual_minutes=actual,
                    missing_minutes=max(0, 240 - actual),
                    expected_minutes=240, reason=existing["reason"] or existing["status"],
                    attempts=existing["attempts"], next_retry_at=None,
                    created_at=existing["created_at"], source=existing["source"],
                    last_error=existing["last_error"],
                )
                _upsert(conn, base, current)
                terminal += 1
                continue
            _, reason = checked_additions(
                part, part.clear(), reference, day, source="local"
            )
            if reason in {"accepted", "already_complete"}:
                base.update(status="complete", actual_minutes=actual,
                            missing_minutes=0, expected_minutes=240,
                            reason="minute_complete")
                if existing:
                    base.update(attempts=existing["attempts"],
                                created_at=existing["created_at"],
                                source=existing["source"],
                                last_error=None)
                _upsert(conn, base, current)
                complete += 1
                continue
            base.update(status="pending", actual_minutes=actual,
                        missing_minutes=max(0, 240 - actual), reason=reason or "minute_gap")
            if existing:
                base.update(attempts=existing["attempts"], next_retry_at=existing["next_retry_at"],
                            created_at=existing["created_at"], source=existing["source"],
                            last_error=existing["last_error"])
            _upsert(conn, base, current)
            queued += 1
        conn.commit()
    return {"date": day.isoformat(), "queued": queued, "terminal": terminal, "complete": complete}


def pending_batch(data_dir, day: date | str | None = None, limit: int = 25) -> list[str]:
    """Return due symbols for the earliest pending date, never a later date."""
    now = cn_now().replace(tzinfo=None).isoformat(timespec="seconds")
    with _connect(data_dir) as conn:
        if day is None:
            row = conn.execute(
                "SELECT min(trade_date) AS day FROM minute_gap_queue "
                "WHERE status IN ('pending','retry','qmt_turnover_conflict') "
                "AND (next_retry_at IS NULL OR next_retry_at <= ?)",
                (now,),
            ).fetchone()
            day = row["day"] if row else None
        else:
            day = date.fromisoformat(str(day)[:10]).isoformat() if not isinstance(day, date) else day.isoformat()
        if not day:
            return []
        rows = conn.execute(
            "SELECT symbol FROM minute_gap_queue WHERE trade_date=? "
            "AND status IN ('pending','retry','qmt_turnover_conflict') "
            "AND (next_retry_at IS NULL OR next_retry_at <= ?) "
            "ORDER BY symbol LIMIT ?", (day, now, max(1, int(limit))),
        ).fetchall()
        return [row["symbol"] for row in rows]


def earliest_pending_date(data_dir) -> str | None:
    now = cn_now().replace(tzinfo=None).isoformat(timespec="seconds")
    with _connect(data_dir) as conn:
        row = conn.execute(
            "SELECT min(trade_date) AS day FROM minute_gap_queue "
            "WHERE status IN ('pending','retry','qmt_turnover_conflict') "
            "AND (next_retry_at IS NULL OR next_retry_at <= ?)",
            (now,),
        ).fetchone()
        return row["day"] if row else None


def record_statuses(data_dir, statuses: Iterable[dict], source_errors: Iterable[dict] | None = None) -> dict:
    """Persist post-fetch validation and schedule only unresolved real gaps."""
    errors = list(source_errors or [])
    now = cn_now().replace(tzinfo=None)
    changed = {"complete": 0, "retry": 0, "terminal": 0}
    lookup = _queue_lookup(data_dir)
    with _connect(data_dir) as conn:
        for raw in statuses:
            identity, identity_reason = _parse_queue_identity(
                data_dir, raw.get("symbol"), str(raw.get("asset_type") or "stock"), lookup
            )
            day = str(raw.get("date") or "")[:10]
            if identity is None or len(day) != 10:
                if identity is None and day:
                    conn.execute(
                        "INSERT INTO minute_gap_identity_quarantine(original_symbol, trade_date, reason, payload, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (str(raw.get("symbol") or ""), day, identity_reason or "identity_unknown", json.dumps(raw, ensure_ascii=False), now.isoformat(timespec="seconds")),
                    )
                continue
            code = identity.code
            status = str(raw.get("status") or "unknown")
            reason = str(raw.get("reason") or raw.get("quality_status") or "minute_gap")
            actual = int(raw.get("actual_minutes") or 0)
            expected = int(raw.get("expected_minutes") or 240)
            source = ""
            matching = [item for item in errors if str(item.get("symbol") or "").strip().upper() == identity.symbol and str(item.get("date") or "")[:10] == day]
            if not matching:
                matching = [item for item in errors if _code(item.get("symbol")) == code and str(item.get("date") or "")[:10] == day]
            if matching:
                source = ",".join(dict.fromkeys(str(item.get("source") or "") for item in matching if item.get("source")))
            old = conn.execute(
                "SELECT attempts, created_at, source, source_chain FROM minute_gap_queue "
                "WHERE asset_type=? AND exchange=? AND code=? AND trade_date=?",
                (identity.asset_type, identity.exchange, identity.code, day),
            ).fetchone()
            if status == "complete":
                _upsert(conn, {
                    "asset_type": identity.asset_type, "exchange": identity.exchange,
                    "code": identity.code, "symbol": identity.symbol,
                    "trade_date": day, "status": "complete",
                    "expected_minutes": expected, "actual_minutes": actual,
                    "missing_minutes": 0, "reason": "minute_complete",
                    "source": source, "attempts": int(old["attempts"] if old else 0),
                    "created_at": old["created_at"] if old else now.isoformat(timespec="seconds"),
                    "last_error": None,
                }, now.isoformat(timespec="seconds"))
                changed["complete"] += 1
                continue
            terminal_reasons = {
                "daily_partition_missing", "daily_reference_missing",
                "invalid_symbol", "invalid_session_timestamps",
                "invalid_minute_fields", "daily_price_or_turnover_conflict",
            }
            is_terminal = status in {
                "market_closed", "suspended", "suspension_data_conflict",
                "no_trade", "pre_listing", "delisted", "code_changed",
                "identity_unknown", "identity_conflict", "source_unsupported",
                "regular_session", "regular_session_close_auction",
            } or (
                status == "unknown" and reason in terminal_reasons
            )
            if is_terminal:
                next_retry = None
                changed["terminal"] += 1
                queue_status = status
            else:
                attempts = int(old["attempts"] if old else 0) + 1
                # Network/source responses back off; structural validation is
                # retried through the next source but never in a hot loop.
                delay = min(3600, 60 * (2 ** min(attempts - 1, 5)))
                next_retry = (now + timedelta(seconds=delay)).isoformat(timespec="seconds")
                changed["retry"] += 1
                queue_status = "retry"
            conn.execute(
                """INSERT INTO minute_gap_queue
                   (asset_type, exchange, code, symbol, trade_date, status, expected_minutes, actual_minutes,
                    missing_minutes, reason, source, source_chain, attempts,
                    next_retry_at, last_error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(asset_type, exchange, code, trade_date) DO UPDATE SET
                     symbol=excluded.symbol,
                     status=excluded.status, expected_minutes=excluded.expected_minutes,
                     actual_minutes=excluded.actual_minutes, missing_minutes=excluded.missing_minutes,
                     reason=excluded.reason, source=COALESCE(NULLIF(excluded.source, ''), minute_gap_queue.source),
                     source_chain=excluded.source_chain, attempts=excluded.attempts,
                     next_retry_at=excluded.next_retry_at, last_error=excluded.last_error,
                     updated_at=excluded.updated_at""",
                (identity.asset_type, identity.exchange, identity.code, identity.symbol, day, queue_status,
                 expected, actual, max(0, expected - actual), reason,
                 source, SOURCE_CHAIN, int(old["attempts"] if old else 0) + (0 if is_terminal else 1),
                 next_retry, reason, old["created_at"] if old else now.isoformat(timespec="seconds"),
                 now.isoformat(timespec="seconds")),
            )
        conn.commit()
    return changed


def record_cross_source_results(data_dir, results: Iterable[dict]) -> dict:
    """Persist third-party verification decisions without losing their meaning.

    The verifier is intentionally allowed to write only queue state.  It never
    replaces a daily or minute partition.  A source outage becomes a delayed
    ``retry`` row; a baseline candidate or quarantine is retained as its own
    non-claimable status so the regular minute worker cannot silently consume
    it as if it were an ordinary gap.
    """
    now = cn_now().replace(tzinfo=None)
    timestamp = now.isoformat(timespec="seconds")
    changed = {
        "retry": 0,
        "minute_import_candidate": 0,
        "daily_baseline_repair_candidate": 0,
        "quarantine": 0,
        "source_no_record": 0,
        "existing_values_conflict": 0,
        "qmt_turnover_conflict": 0,
    }
    lookup = _queue_lookup(data_dir)
    with _connect(data_dir) as conn:
        for raw in results:
            identity, identity_reason = _parse_queue_identity(
                data_dir, raw.get("symbol"), str(raw.get("asset_type") or "stock"), lookup
            )
            day = str(raw.get("date") or "")[:10]
            if identity is None or len(day) != 10:
                if identity is None and len(day) == 10:
                    conn.execute(
                        "INSERT INTO minute_gap_identity_quarantine(original_symbol, trade_date, reason, payload, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (str(raw.get("symbol") or ""), day, identity_reason or "identity_unknown", json.dumps(raw, ensure_ascii=False), timestamp),
                    )
                continue
            code = identity.code
            status = str(raw.get("status") or "quarantine")
            reason = str(raw.get("reason") or status)
            expected = int(raw.get("expected_minutes") or 240)
            actual = int(raw.get("actual_minutes") or raw.get("minute_rows") or 0)
            matching_sources = [
                str(value).strip() for value in raw.get("sources_checked", [])
                if str(value).strip()
            ]
            source = ",".join(dict.fromkeys(matching_sources))
            old = conn.execute(
                "SELECT attempts, created_at, source, source_chain FROM minute_gap_queue "
                "WHERE asset_type=? AND exchange=? AND code=? AND trade_date=?",
                (identity.asset_type, identity.exchange, identity.code, day),
            ).fetchone()
            attempts = int(old["attempts"] if old else 0)
            created_at = old["created_at"] if old else timestamp
            source_chain = str(raw.get("source_chain") or SOURCE_CHAIN)

            if status == "source_unavailable" or bool(raw.get("retryable")):
                delay = max(60, int(raw.get("retry_after_seconds") or 300))
                queue_status = "retry"
                next_retry = (now + timedelta(seconds=delay)).isoformat(timespec="seconds")
                queue_reason = reason
                attempts += 1
                changed["retry"] += 1
            elif status == "minute_import_candidate":
                # The next minute worker may attempt the validated source chain,
                # but this row remains distinguishable from an unverified gap.
                queue_status = "retry"
                next_retry = timestamp
                queue_reason = "third_party_verified"
                changed["minute_import_candidate"] += 1
            elif status in {
                "qmt_turnover_conflict",
                "daily_baseline_repair_candidate", "quarantine",
                "source_no_record", "existing_values_conflict",
            }:
                queue_status = status
                next_retry = None
                queue_reason = reason
                changed[status] += 1
            else:
                queue_status = "quarantine"
                next_retry = None
                queue_reason = f"unknown_cross_source_status:{status}"
                changed["quarantine"] += 1

            _upsert(conn, {
                "asset_type": identity.asset_type,
                "exchange": identity.exchange,
                "code": identity.code,
                "symbol": identity.symbol,
                "trade_date": day,
                "status": queue_status,
                "expected_minutes": expected,
                "actual_minutes": actual,
                "missing_minutes": max(0, expected - actual),
                "reason": queue_reason,
                "source": source,
                "source_chain": source_chain,
                "attempts": attempts,
                "next_retry_at": next_retry,
                "last_error": reason if queue_status in {"retry", "quarantine", "source_no_record"} else None,
                "created_at": created_at,
            }, timestamp)
        conn.commit()
    return changed


def summary(data_dir) -> dict:
    with _connect(data_dir) as conn:
        rows = conn.execute(
            "SELECT status, reason, actual_minutes, expected_minutes, trade_date "
            "FROM minute_gap_queue ORDER BY trade_date"
        ).fetchall()
        if not rows:
            return {}

        by_status: dict[str, dict[str, object]] = {}
        by_category: dict[str, dict[str, object]] = {}

        def add(bucket: dict[str, dict[str, object]], key: str, day: str) -> None:
            item = bucket.setdefault(key, {"count": 0, "first_date": day, "last_date": day})
            item["count"] = int(item["count"]) + 1
            item["first_date"] = min(str(item["first_date"]), day)
            item["last_date"] = max(str(item["last_date"]), day)

        def category(status: str, reason: str, actual: int, expected: int) -> str:
            normalized = f"{status} {reason}".lower()
            if status in {
                "complete", "regular_session", "regular_session_close_auction",
                "market_closed", "suspended", "no_trade", "pre_listing", "delisted",
                "code_changed", "source_unsupported",
            }:
                return "complete"
            if reason == "third_party_verified":
                return "verified_candidate"
            if status in {
                "qmt_turnover_conflict", "existing_values_conflict",
                "daily_baseline_repair_candidate",
            } or "conflict" in normalized or "turnover" in normalized:
                return "conflict"
            if status in {"quarantine", "source_no_record", "identity_unknown", "identity_conflict", "unknown"}:
                return "review"
            if reason in {"source_empty", "source_no_record"}:
                return "source_empty"
            if reason == "not_requested":
                return "not_requested"
            if reason in {
                "source_unavailable", "third_party_source_unavailable", "backup_unavailable",
                "source_pending", "rate_limited", "request_timeout",
            }:
                return "source_unavailable"
            if reason == "source_incomplete" or (actual > 0 and actual < expected):
                return "source_incomplete"
            if status == "retry":
                return "source_unavailable"
            return "pending_gap"

        for row in rows:
            status = str(row["status"] or "unknown")
            reason = str(row["reason"] or "minute_gap")
            day = str(row["trade_date"] or "")[:10]
            add(by_status, status, day)
            add(by_category, category(
                status, reason, int(row["actual_minutes"] or 0), int(row["expected_minutes"] or 240)
            ), day)

        for key in (
            "pending_gap", "source_incomplete", "source_empty", "source_unavailable",
            "conflict", "verified_candidate", "not_requested", "review", "complete",
        ):
            by_category.setdefault(key, {"count": 0, "first_date": None, "last_date": None})
        by_status["categories"] = by_category
        return by_status
