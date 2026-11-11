"""Expose the QMT research kernel as a small local HTTP market-data service.

QMT injects its data functions into the research Jupyter kernel.  Calling the
vendor's Python package from another process is not equivalent and, on many
installations, fails authentication.  This bridge keeps one QMT kernel alive
and executes narrowly-scoped read-only market-data requests in that kernel.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
import websocket


RESULT_MARKER = "QMT_BRIDGE_RESULT="
DEFAULT_CONFIG = Path(r"D:\国信iQuant策略交易平台\config\jupyter_notebook_config.py")
DEFAULT_PUSH_DB = Path(__file__).resolve().parents[2] / "data" / "user_data" / "qmt_push.sqlite"


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _quote_datetime(row: dict[str, Any]) -> datetime | None:
    for key in ("datetime", "stime", "time"):
        value = row.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        try:
            if len(digits) >= 14 and not (len(digits) == 16 and digits.startswith(("16", "17", "18", "19", "20"))):
                return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
            number = float(value)
            if number > 10_000_000_000:
                number /= 1000.0
            if number > 1_000_000_000:
                return datetime.fromtimestamp(number)
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except (TypeError, ValueError, OSError):
            continue
    return None


def _trade_date(row: dict[str, Any]) -> str | None:
    """Parse a daily-bar date without accepting an invalid/future date."""
    for key in ("date", "trade_date", "datetime", "stime", "time"):
        value = row.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        try:
            if len(digits) >= 8:
                parsed = datetime.strptime(digits[:8], "%Y%m%d").date()
            else:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00")).date()
            if parsed > datetime.now().date():
                return None
            return parsed.isoformat()
        except (TypeError, ValueError, OSError):
            continue
    return None


def _valid_symbol(value: Any) -> str | None:
    symbol = str(value or "").strip().upper()
    return symbol if re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", symbol) else None


def _minute_label(value: datetime) -> datetime:
    base = value.replace(second=0, microsecond=0)
    # QMT minute bars use the interval end, e.g. 09:31 for (09:30, 09:31].
    if value.second or value.microsecond or value.strftime("%H%M") in {"0930", "1300"}:
        base += timedelta(minutes=1)
    return base


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class QmtPushStore:
    """Durable cache populated by an authenticated in-client ContextInfo model."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS latest_quote (
                    symbol TEXT PRIMARY KEY, quote_time TEXT NOT NULL,
                    received_at REAL NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS minute_bar (
                    symbol TEXT NOT NULL, bar_time TEXT NOT NULL,
                    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
                    close REAL NOT NULL, volume REAL NOT NULL, amount REAL NOT NULL,
                    PRIMARY KEY(symbol, bar_time)
                );
                CREATE TABLE IF NOT EXISTS daily_bar (
                    symbol TEXT NOT NULL, trade_date TEXT NOT NULL,
                    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
                    close REAL NOT NULL, volume REAL NOT NULL, amount REAL NOT NULL,
                    PRIMARY KEY(symbol, trade_date)
                );
                CREATE TABLE IF NOT EXISTS auction_quote (
                    symbol TEXT NOT NULL, trade_date TEXT NOT NULL,
                    quote_time TEXT NOT NULL, received_at REAL NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(symbol, trade_date)
                );
                CREATE TABLE IF NOT EXISTS heartbeat (
                    source TEXT PRIMARY KEY, received_at REAL NOT NULL,
                    quote_time TEXT, details TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS minute_sync (
                    source TEXT NOT NULL, symbol TEXT NOT NULL,
                    trade_date TEXT NOT NULL, received_at REAL NOT NULL,
                    rows INTEGER NOT NULL, rejected INTEGER NOT NULL,
                    inserted INTEGER NOT NULL DEFAULT 0,
                    updated INTEGER NOT NULL DEFAULT 0,
                    duplicate INTEGER NOT NULL DEFAULT 0,
                    complete INTEGER NOT NULL DEFAULT 0,
                    missing INTEGER NOT NULL DEFAULT 240,
                    PRIMARY KEY(source, symbol, trade_date)
                );
                CREATE TABLE IF NOT EXISTS daily_sync (
                    source TEXT NOT NULL, symbol TEXT NOT NULL,
                    trade_date TEXT NOT NULL, received_at REAL NOT NULL,
                    PRIMARY KEY(source, symbol, trade_date)
                );
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                """
            )
            # Existing bridge databases predate the detailed ingest counters.
            # Migrate in place so upgrading the bridge never discards its cache.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(minute_sync)")}
            for name, kind in (
                ("inserted", "INTEGER NOT NULL DEFAULT 0"),
                ("updated", "INTEGER NOT NULL DEFAULT 0"),
                ("duplicate", "INTEGER NOT NULL DEFAULT 0"),
                ("complete", "INTEGER NOT NULL DEFAULT 0"),
                ("missing", "INTEGER NOT NULL DEFAULT 240"),
            ):
                if name not in columns:
                    conn.execute(f"ALTER TABLE minute_sync ADD COLUMN {name} {kind}")
            # A historical QMT cache can be multi-gigabyte.  Health probes
            # must not derive the latest bar by scanning minute_bar.
            existing_meta = conn.execute(
                "SELECT value FROM cache_meta WHERE key='minute_latest_time'"
            ).fetchone()
            if existing_meta is None:
                latest = conn.execute(
                    "SELECT bar_time FROM minute_bar ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                if latest and latest[0]:
                    conn.execute(
                        "INSERT INTO cache_meta(key,value) VALUES('minute_latest_time',?)",
                        (str(latest[0]),),
                    )

    @staticmethod
    def _update_latest_minute(conn: sqlite3.Connection, value: str | None) -> None:
        if not value:
            return
        current = conn.execute(
            "SELECT value FROM cache_meta WHERE key='minute_latest_time'"
        ).fetchone()
        if current is None or str(value) > str(current[0]):
            conn.execute(
                "INSERT INTO cache_meta(key,value) VALUES('minute_latest_time',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(value),),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10, factory=_ClosingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _continuous(value: datetime) -> bool:
        hm = value.hour * 100 + value.minute
        return 930 <= hm <= 1130 or 1300 <= hm <= 1500

    def ingest_quotes(self, rows: list[dict[str, Any]], source: str = "iquant_context") -> dict[str, Any]:
        accepted = minutes = auctions = 0
        now_ts = time.time()
        latest_time: datetime | None = None
        with self._lock, self._connect() as conn:
            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                symbol = _valid_symbol(raw.get("symbol") or raw.get("stockCode") or raw.get("code"))
                quote_time = _quote_datetime(raw)
                price = _as_float(raw.get("lastPrice") or raw.get("close"))
                if not symbol or quote_time is None or price is None or price <= 0:
                    continue
                latest_time = max(latest_time, quote_time) if latest_time else quote_time
                payload = dict(raw)
                payload.update(symbol=symbol, datetime=quote_time.strftime("%Y-%m-%d %H:%M:%S"), source="qmt_context")
                cumulative_volume = _as_float(raw.get("volume")) or 0.0
                cumulative_amount = _as_float(raw.get("amount")) or 0.0
                previous = conn.execute(
                    "SELECT quote_time, payload FROM latest_quote WHERE symbol=?", (symbol,)
                ).fetchone()
                prev_volume = prev_amount = 0.0
                same_day = False
                if previous:
                    try:
                        previous_payload = json.loads(previous["payload"])
                        previous_time = datetime.fromisoformat(previous["quote_time"])
                        same_day = previous_time.date() == quote_time.date()
                        prev_volume = _as_float(previous_payload.get("volume")) or 0.0
                        prev_amount = _as_float(previous_payload.get("amount")) or 0.0
                    except (ValueError, TypeError, json.JSONDecodeError):
                        pass
                volume_delta = max(0.0, cumulative_volume - prev_volume) if same_day else 0.0
                amount_delta = max(0.0, cumulative_amount - prev_amount) if same_day else 0.0
                encoded = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
                conn.execute(
                    "INSERT INTO latest_quote VALUES(?,?,?,?) "
                    "ON CONFLICT(symbol) DO UPDATE SET quote_time=excluded.quote_time, "
                    "received_at=excluded.received_at, payload=excluded.payload",
                    (symbol, quote_time.isoformat(sep=" "), now_ts, encoded),
                )
                hm = quote_time.hour * 100 + quote_time.minute
                if 915 <= hm <= 925:
                    conn.execute(
                        "INSERT INTO auction_quote VALUES(?,?,?,?,?) "
                        "ON CONFLICT(symbol,trade_date) DO UPDATE SET quote_time=excluded.quote_time, "
                        "received_at=excluded.received_at, payload=excluded.payload",
                        (symbol, quote_time.date().isoformat(), quote_time.isoformat(sep=" "), now_ts, encoded),
                    )
                    auctions += 1
                if self._continuous(quote_time):
                    bar_time = _minute_label(quote_time).isoformat(sep=" ")
                    current = conn.execute(
                        "SELECT open,high,low,close,volume,amount FROM minute_bar WHERE symbol=? AND bar_time=?",
                        (symbol, bar_time),
                    ).fetchone()
                    if current:
                        conn.execute(
                            "UPDATE minute_bar SET high=?,low=?,close=?,volume=?,amount=? WHERE symbol=? AND bar_time=?",
                            (max(current["high"], price), min(current["low"], price), price,
                             current["volume"] + volume_delta, current["amount"] + amount_delta,
                             symbol, bar_time),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO minute_bar VALUES(?,?,?,?,?,?,?,?)",
                            (symbol, bar_time, price, price, price, price, volume_delta, amount_delta),
                        )
                    self._update_latest_minute(conn, bar_time)
                    minutes += 1
                day_open = _as_float(raw.get("open")) or price
                day_high = _as_float(raw.get("high")) or price
                day_low = _as_float(raw.get("low")) or price
                conn.execute(
                    "INSERT INTO daily_bar VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(symbol,trade_date) DO UPDATE SET open=excluded.open,high=excluded.high,"
                    "low=excluded.low,close=excluded.close,volume=excluded.volume,amount=excluded.amount",
                    (symbol, quote_time.date().isoformat(), day_open, day_high, day_low, price,
                     cumulative_volume, cumulative_amount),
                )
                accepted += 1
            details = json.dumps({"accepted": accepted, "received": len(rows)}, separators=(",", ":"))
            conn.execute(
                "INSERT INTO heartbeat VALUES(?,?,?,?) ON CONFLICT(source) DO UPDATE SET "
                "received_at=excluded.received_at,quote_time=excluded.quote_time,details=excluded.details",
                (source, now_ts, latest_time.isoformat(sep=" ") if latest_time else None, details),
            )
            conn.commit()
        return {"received": len(rows), "accepted": accepted, "minute_updates": minutes,
                "auction_updates": auctions, "latest_time": latest_time.isoformat() if latest_time else None}

    def ingest_minute_bars(self, rows: list[dict[str, Any]], source: str = "iquant_history") -> dict[str, Any]:
        """Upsert validated 1-minute bars pushed from the authenticated QMT runtime."""
        accepted = rejected = inserted = updated = duplicate = 0
        by_symbol_day: dict[tuple[str, str], dict[str, int]] = {}
        rejected_by_symbol_day: dict[tuple[str, str], int] = {}
        now_ts = time.time()
        with self._lock, self._connect() as conn:
            for raw in rows:
                if not isinstance(raw, dict):
                    rejected += 1
                    continue
                symbol = _valid_symbol(raw.get("symbol") or raw.get("stockCode") or raw.get("code"))
                bar_time = _quote_datetime(raw)
                values = {
                    key: _as_float(raw.get(key))
                    for key in ("open", "high", "low", "close")
                }
                volume = _as_float(raw.get("volume"))
                amount = _as_float(raw.get("amount"))
                valid_ohlc = all(value is not None and value > 0 for value in values.values())
                valid_turnover = volume is not None and volume >= 0 and amount is not None and amount >= 0
                if (not symbol or bar_time is None or not self._continuous(bar_time)
                        or not valid_ohlc or not valid_turnover):
                    rejected += 1
                    if symbol and bar_time:
                        key = (symbol, bar_time.date().isoformat())
                        rejected_by_symbol_day[key] = rejected_by_symbol_day.get(key, 0) + 1
                    continue
                high = values["high"]
                low = values["low"]
                if high < max(values["open"], values["close"]) or low > min(values["open"], values["close"]):
                    rejected += 1
                    key = (symbol, bar_time.date().isoformat())
                    rejected_by_symbol_day[key] = rejected_by_symbol_day.get(key, 0) + 1
                    continue
                bar_time = _minute_label(bar_time)
                trade_date = bar_time.date().isoformat()
                payload = (symbol, bar_time.isoformat(sep=" "), values["open"], values["high"],
                           values["low"], values["close"], volume, amount)
                self._update_latest_minute(conn, payload[1])
                current = conn.execute(
                    "SELECT open,high,low,close,volume,amount FROM minute_bar "
                    "WHERE symbol=? AND bar_time=?", (symbol, payload[1])
                ).fetchone()
                if current is None:
                    conn.execute("INSERT INTO minute_bar VALUES(?,?,?,?,?,?,?,?)", payload)
                    inserted += 1
                else:
                    same = all(abs(float(current[index]) - float(payload[index + 2])) <= 1e-12
                               for index in range(6))
                    if same:
                        duplicate += 1
                    else:
                        conn.execute(
                            "UPDATE minute_bar SET open=?,high=?,low=?,close=?,volume=?,amount=? "
                            "WHERE symbol=? AND bar_time=?",
                            (*payload[2:], symbol, payload[1]),
                        )
                        updated += 1
                accepted += 1
                key = (symbol, trade_date)
                stats = by_symbol_day.setdefault(key, {"rows": 0, "inserted": 0, "updated": 0, "duplicate": 0})
                stats["rows"] += 1
                if current is None:
                    stats["inserted"] += 1
                elif same:
                    stats["duplicate"] += 1
                else:
                    stats["updated"] += 1
            for key, stats in by_symbol_day.items():
                symbol, trade_date = key
                stored_rows = conn.execute(
                    "SELECT count(*) FROM minute_bar WHERE symbol=? AND bar_time LIKE ?",
                    (symbol, f"{trade_date}%"),
                ).fetchone()[0]
                complete = int(stored_rows >= 240)
                missing = max(0, 240 - stored_rows)
                conn.execute(
                    "INSERT INTO minute_sync "
                    "(source,symbol,trade_date,received_at,rows,rejected,inserted,updated,duplicate,complete,missing) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(source,symbol,trade_date) DO UPDATE SET "
                    "received_at=excluded.received_at,rows=excluded.rows,rejected=excluded.rejected,"
                    "inserted=excluded.inserted,updated=excluded.updated,duplicate=excluded.duplicate,"
                    "complete=excluded.complete,missing=excluded.missing",
                    (source, symbol, trade_date, now_ts, stats["rows"],
                     rejected_by_symbol_day.get(key, 0), stats["inserted"], stats["updated"],
                     stats["duplicate"], complete, missing),
                )
            details = json.dumps({
                "accepted": accepted, "rejected": rejected,
                "inserted": inserted, "updated": updated, "duplicate": duplicate,
                "symbols": len({key[0] for key in by_symbol_day}),
                "days": len({key[1] for key in by_symbol_day}),
            }, separators=(",", ":"))
            conn.execute(
                "INSERT INTO heartbeat VALUES(?,?,?,?) ON CONFLICT(source) DO UPDATE SET "
                "received_at=excluded.received_at,quote_time=excluded.quote_time,details=excluded.details",
                (source, now_ts, None, details),
            )
            conn.commit()
        return {
            "received": len(rows), "accepted": accepted, "rejected": rejected,
            "inserted": inserted, "updated": updated, "duplicate": duplicate,
            "symbols": len({key[0] for key in by_symbol_day}),
            "days": len({key[1] for key in by_symbol_day}),
        }

    def ingest_daily_bars(self, rows: list[dict[str, Any]], source: str = "iquant_index_daily") -> dict[str, Any]:
        """Upsert validated daily bars pushed from the authenticated QMT runtime."""
        accepted = rejected = 0
        by_symbol_day: dict[tuple[str, str], int] = {}
        now_ts = time.time()
        latest_date: str | None = None
        with self._lock, self._connect() as conn:
            for raw in rows:
                if not isinstance(raw, dict):
                    rejected += 1
                    continue
                symbol = _valid_symbol(raw.get("symbol") or raw.get("stockCode") or raw.get("code"))
                trade_date = _trade_date(raw)
                values = {key: _as_float(raw.get(key)) for key in ("open", "high", "low", "close")}
                volume = _as_float(raw.get("volume"))
                amount = _as_float(raw.get("amount"))
                valid_ohlc = all(value is not None and value > 0 for value in values.values())
                valid_turnover = volume is not None and volume >= 0 and amount is not None and amount >= 0
                if not symbol or not trade_date or not valid_ohlc or not valid_turnover:
                    rejected += 1
                    continue
                if values["high"] < max(values["open"], values["close"]) or values["low"] > min(values["open"], values["close"]):
                    rejected += 1
                    continue
                conn.execute(
                    "INSERT INTO daily_bar VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(symbol,trade_date) DO UPDATE SET open=excluded.open,high=excluded.high,"
                    "low=excluded.low,close=excluded.close,volume=excluded.volume,amount=excluded.amount",
                    (symbol, trade_date, values["open"], values["high"], values["low"], values["close"], volume, amount),
                )
                conn.execute(
                    "INSERT INTO daily_sync VALUES(?,?,?,?) "
                    "ON CONFLICT(source,symbol,trade_date) DO UPDATE SET received_at=excluded.received_at",
                    (source, symbol, trade_date, now_ts),
                )
                accepted += 1
                by_symbol_day[(symbol, trade_date)] = by_symbol_day.get((symbol, trade_date), 0) + 1
                latest_date = max(latest_date, trade_date) if latest_date else trade_date
            details = json.dumps({
                "accepted": accepted, "rejected": rejected,
                "symbols": len({key[0] for key in by_symbol_day}),
                "days": len({key[1] for key in by_symbol_day}),
            }, separators=(",", ":"))
            conn.execute(
                "INSERT INTO heartbeat VALUES(?,?,?,?) ON CONFLICT(source) DO UPDATE SET "
                "received_at=excluded.received_at,quote_time=excluded.quote_time,details=excluded.details",
                (source, now_ts, latest_date, details),
            )
            conn.commit()
        return {
            "received": len(rows), "accepted": accepted, "rejected": rejected,
            "symbols": len({key[0] for key in by_symbol_day}),
            "days": len({key[1] for key in by_symbol_day}),
            "latest_date": latest_date,
        }

    def capabilities(self) -> dict[str, Any]:
        now_ts = time.time()
        with self._connect() as conn:
            heartbeat = conn.execute(
                "SELECT * FROM heartbeat WHERE source='iquant_context' LIMIT 1"
            ).fetchone()
            history_heartbeat = conn.execute(
                "SELECT * FROM heartbeat WHERE source='iquant_history' LIMIT 1"
            ).fetchone()
            index_heartbeat = conn.execute(
                "SELECT * FROM heartbeat WHERE source='iquant_index_daily' LIMIT 1"
            ).fetchone()
            age = max(0.0, now_ts - float(heartbeat["received_at"])) if heartbeat else None
            latest_meta = conn.execute(
                "SELECT value FROM cache_meta WHERE key='minute_latest_time'"
            ).fetchone()
            latest_time = str(latest_meta[0]) if latest_meta and latest_meta[0] else None
            latest_day = latest_time[:10] if latest_time else datetime.now().date().isoformat()
            symbols = conn.execute("SELECT COUNT(*) FROM latest_quote WHERE received_at>=?", (now_ts - 900,)).fetchone()[0]
            quote_rows = conn.execute(
                "SELECT payload FROM latest_quote WHERE received_at>=? ORDER BY received_at DESC LIMIT 50",
                (now_ts - 900,),
            ).fetchall()
            depth5 = False
            for row in quote_rows:
                try:
                    payload = json.loads(row["payload"])
                    depth5 = (
                        len(payload.get("bidPrice") or []) >= 5
                        and len(payload.get("askPrice") or []) >= 5
                        and len(payload.get("bidVol") or []) >= 5
                        and len(payload.get("askVol") or []) >= 5
                    )
                except (TypeError, json.JSONDecodeError):
                    depth5 = False
                if depth5:
                    break
            # Never scan the multi-gigabyte minute_bar table from a health
            # endpoint. The timestamp is maintained transactionally during
            # both QMT ingestion paths; the journal supplies a cheap day count.
            minute_count_row = conn.execute(
                "SELECT COALESCE(SUM(rows),0) FROM minute_sync WHERE trade_date=?",
                (latest_day,),
            ).fetchone()
            minute_count = int(minute_count_row[0] or 0) if minute_count_row else 0
            minute = minute_count or (1 if latest_time else 0)
            minute_latest = (latest_time, minute_count)
            auction = conn.execute("SELECT COUNT(*) FROM auction_quote WHERE trade_date=(SELECT MAX(trade_date) FROM auction_quote)").fetchone()[0]
            core = conn.execute(
                "SELECT COUNT(*) FROM daily_bar WHERE trade_date=(SELECT MAX(trade_date) FROM daily_bar) "
                "AND symbol IN ('000001.SH','399001.SZ','399006.SZ','000680.SH')",
            ).fetchone()[0]
            history_age = None
            history_details: dict[str, Any] = {}
            if history_heartbeat:
                history_age = round(max(0.0, now_ts - float(history_heartbeat["received_at"])), 1)
                try:
                    history_details = json.loads(history_heartbeat["details"])
                except (TypeError, json.JSONDecodeError):
                    history_details = {}
            index_details: dict[str, Any] = {}
            index_age = None
            if index_heartbeat:
                index_age = round(max(0.0, now_ts - float(index_heartbeat["received_at"])), 1)
                try:
                    index_details = json.loads(index_heartbeat["details"])
                except (TypeError, json.JSONDecodeError):
                    index_details = {}
            return {"push_active": bool(heartbeat and age is not None and age <= 900 and symbols >= 2),
                    "push_age_seconds": round(age, 1) if age is not None else None,
                    "push_symbols": int(symbols), "push_minute": bool(minute),
                    "push_minute_date": latest_day,
                    "push_minute_latest": minute_latest[0],
                    "push_minute_rows": int(minute_latest[1] or 0),
                    "push_minute_sync_age_seconds": history_age,
                    "push_minute_sync": history_details,
                    "push_depth5": depth5,
                    "push_auction": bool(auction), "push_auction_symbols": int(auction),
                    "push_daily_index": core >= 2, "push_daily_index_symbols": int(core),
                    "push_daily_index_sync_age_seconds": index_age,
                    "push_daily_index_sync": index_details,
                    "push_quote_time": heartbeat["quote_time"] if heartbeat else None}

    def quotes(self, symbols: list[str], max_age_seconds: float | None = None) -> list[dict[str, Any]]:
        if not symbols:
            return []
        placeholders = ",".join("?" for _ in symbols)
        with self._connect() as conn:
            clauses = [f"symbol IN ({placeholders})"]
            params: list[Any] = list(symbols)
            if max_age_seconds is not None:
                clauses.append("received_at>=?")
                params.append(time.time() - max(0.0, max_age_seconds))
            rows = conn.execute(
                f"SELECT payload,received_at FROM latest_quote WHERE {' AND '.join(clauses)}", params
            ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["receivedAt"] = row["received_at"]
            result.append(payload)
        return result

    def klines(self, symbol: str, freq: str, start: str, end: str, count: int) -> list[dict[str, Any]]:
        symbol = _valid_symbol(symbol)
        if not symbol:
            return []
        table, time_col = ("daily_bar", "trade_date") if freq == "1d" else ("minute_bar", "bar_time")
        clauses, params = ["symbol=?"], [symbol]
        if start:
            clauses.append(f"{time_col}>=?")
            params.append(self._range_time(start, daily=freq == "1d"))
        if end:
            clauses.append(f"{time_col}<=?")
            params.append(self._range_time(end, daily=freq == "1d", end=True))
        limit = "" if count < 0 else f" LIMIT {max(1, min(count, 20000))}"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE {' AND '.join(clauses)} ORDER BY {time_col} DESC{limit}", params
            ).fetchall()
        result = []
        for row in reversed(rows):
            item = dict(row)
            item["symbol"] = symbol
            item["datetime"] = item.pop(time_col)
            result.append(item)
        return result

    @staticmethod
    def _range_time(value: str, *, daily: bool, end: bool = False) -> str:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if len(digits) >= 8:
            day = datetime.strptime(digits[:8], "%Y%m%d")
            if not daily and len(digits) >= 14:
                day = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
            elif not daily and end:
                day = day.replace(hour=23, minute=59, second=59)
            return day.date().isoformat() if daily else day.isoformat(sep=" ")
        return str(value)

    def auctions(self, trade_date: str, symbols: list[str]) -> list[dict[str, Any]]:
        day = self._range_time(trade_date, daily=True)
        clauses, params = ["trade_date=?"], [day]
        if symbols:
            clauses.append(f"symbol IN ({','.join('?' for _ in symbols)})")
            params.extend(symbols)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT symbol,payload FROM auction_quote WHERE {' AND '.join(clauses)}", params
            ).fetchall()
        result = []
        for row in rows:
            quote = json.loads(row["payload"])
            price = _as_float(quote.get("lastPrice") or quote.get("open"))
            if not price or price <= 0:
                continue
            result.append({"symbol": row["symbol"], "date": day, "market": "open",
                           "open": price, "close": price, "high": price, "low": price,
                           "volume": _as_float(quote.get("pvolume")) or 0.0,
                           "amount": _as_float(quote.get("amount")) or 0.0,
                           "source": "qmt_context"})
        return result


def _load_notebook_token(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"c\.NotebookApp\.token\s*=\s*(.+)", text)
    if not match:
        raise RuntimeError(f"QMT Jupyter token not found in {config_path}")
    return str(ast.literal_eval(match.group(1).strip()))


def _json_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class QmtKernel:
    """Serialized access to one QMT research kernel."""

    def __init__(self, base_url: str, config_path: Path, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.ws_base_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.config_path = config_path
        self.token = _load_notebook_token(config_path)
        self.timeout = timeout
        self.kernel_id: str | None = None
        self._lock = threading.RLock()

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self.token}"}

    def start(self) -> None:
        with self._lock:
            # QMT rewrites its authentication token when the client restarts.
            self.token = _load_notebook_token(self.config_path)
            if self.kernel_id and self._kernel_alive():
                return
            response = requests.post(
                f"{self.base_url}/api/kernels",
                headers=self.headers,
                json={},
                timeout=self.timeout,
            )
            response.raise_for_status()
            self.kernel_id = str(response.json()["id"])

    def stop(self) -> None:
        with self._lock:
            kernel_id, self.kernel_id = self.kernel_id, None
            if kernel_id:
                try:
                    requests.delete(
                        f"{self.base_url}/api/kernels/{kernel_id}",
                        headers=self.headers,
                        timeout=5,
                    )
                except requests.RequestException:
                    pass

    def _kernel_alive(self) -> bool:
        if not self.kernel_id:
            return False
        try:
            response = requests.get(
                f"{self.base_url}/api/kernels/{self.kernel_id}",
                headers=self.headers,
                timeout=5,
            )
            return response.ok
        except requests.RequestException:
            return False

    def execute_json(self, code: str) -> Any:
        with self._lock:
            for attempt in range(2):
                try:
                    self.start()
                    return self._execute_json_once(code)
                except Exception:
                    self.stop()
                    if attempt:
                        raise
            raise RuntimeError("QMT kernel execution failed")

    def _execute_json_once(self, code: str) -> Any:
        if not self.kernel_id:
            raise RuntimeError("QMT kernel is not running")
        session_id = uuid.uuid4().hex
        message_id = uuid.uuid4().hex
        ws = websocket.create_connection(
            f"{self.ws_base_url}/api/kernels/{self.kernel_id}/channels?token={self.token}",
            timeout=self.timeout,
        )
        try:
            ws.send(json.dumps({
                "header": {
                    "msg_id": message_id,
                    "username": "tickflow_qmt_bridge",
                    "session": session_id,
                    "msg_type": "execute_request",
                    "version": "5.3",
                },
                "parent_header": {},
                "metadata": {},
                "content": {
                    "code": code,
                    "silent": False,
                    "store_history": False,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": True,
                },
                "channel": "shell",
                "buffers": [],
            }))
            output: list[str] = []
            errors: list[str] = []
            deadline = time.time() + self.timeout
            while time.time() < deadline:
                message = json.loads(ws.recv())
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                msg_type = message.get("msg_type")
                content = message.get("content", {})
                if msg_type == "stream":
                    output.append(str(content.get("text", "")))
                elif msg_type == "error":
                    errors.append("\n".join(content.get("traceback", [])))
                elif msg_type == "status" and content.get("execution_state") == "idle":
                    break
            if errors:
                raise RuntimeError(errors[-1])
            joined = "\n".join(output)
            matches = re.findall(rf"{re.escape(RESULT_MARKER)}(.+)", joined)
            if not matches:
                raise RuntimeError(f"QMT response marker missing: {joined[-500:]}")
            return json.loads(matches[-1])
        finally:
            ws.close()


_KERNEL_HELPERS = r'''
import json as _qjson
import math as _qmath
from datetime import datetime as _qdatetime

def _qclean(value):
    if value is None:
        return None
    try:
        if _qmath.isnan(value) or _qmath.isinf(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _qclean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_qclean(v) for v in value]
    return str(value)

def _qtime(value):
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 14:
        dt = _qdatetime.strptime(digits[:14], "%Y%m%d%H%M%S")
    elif len(digits) >= 12:
        dt = _qdatetime.strptime(digits[:12], "%Y%m%d%H%M")
    else:
        return None
    return dt

def _qrange(value, start=False):
    """Normalize API date parameters for QMT historical K-line queries.

    QMT accepts a full datetime for minute history.  Passing a bare YYYYMMDD
    silently returns an empty frame, which made downloaded BJ .DAT files look
    unavailable to the bridge.  Keep explicit timestamps unchanged and map a
    date-only range to the complete A-share session.
    """
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 14:
        return digits[:14]
    if len(digits) >= 8:
        return digits[:8] + ("093000" if start else "150000")
    return text

def _qin_session(dt, minute=False):
    if dt is None:
        return False
    hm = dt.hour * 100 + dt.minute
    if minute:
        return 931 <= hm <= 1130 or 1301 <= hm <= 1500
    return 915 <= hm <= 1130 or 1300 <= hm <= 1500

def _qemit(value):
    print("QMT_BRIDGE_RESULT=" + _qjson.dumps(_qclean(value), ensure_ascii=False, default=str))
'''


def _kline_code(symbol: str, period: str, start: str, end: str, count: int) -> str:
    intraday = period != "1d"
    return _KERNEL_HELPERS + f'''
_start = _qrange({_json_literal(start)}, start=True) if {str(intraday)} else {_json_literal(start)}
_end = _qrange({_json_literal(end)}, start=False) if {str(intraday)} else {_json_literal(end)}
_df = get_market_data(
    fields=["open", "high", "low", "close", "volume", "amount"],
    stock_code=[{_json_literal(symbol)}],
    start_time=_start,
    end_time=_end,
    period={_json_literal(period)},
    count={count},
)
_rows = []
if hasattr(_df, "iterrows"):
    for _idx, _row in _df.iterrows():
        _dt = _qtime(_idx)
        if _dt is None or ({str(intraday)} and not _qin_session(_dt, minute=True)):
            continue
        _close = _qclean(_row.get("close"))
        if _close is None or float(_close) <= 0:
            continue
        _rows.append({{
            "symbol": {_json_literal(symbol)},
            "datetime": _dt.strftime("%Y-%m-%d %H:%M:%S"),
            "open": _qclean(_row.get("open")),
            "high": _qclean(_row.get("high")),
            "low": _qclean(_row.get("low")),
            "close": _close,
            "volume": _qclean(_row.get("volume")),
            "amount": _qclean(_row.get("amount")),
        }})
_qemit(_rows)
'''


def _ticks_code(symbol: str, start: str, end: str, count: int) -> str:
    return _KERNEL_HELPERS + f'''
_df = get_market_data(
    fields=["quoter"], stock_code=[{_json_literal(symbol)}],
    start_time={_json_literal(start)}, end_time={_json_literal(end)},
    period="tick", count={count},
)
_rows = []
for _idx, _row in _df.iterrows():
    _dt = _qtime(_idx)
    _quote = _row.get("quoter") or {{}}
    _last = _quote.get("lastPrice") or 0
    _volume = _quote.get("pvolume") or _quote.get("volume") or 0
    _amount = _quote.get("amount") or 0
    # QMT emits heartbeat records after close.  They are not market ticks.
    if not _qin_session(_dt) or float(_last) <= 0 or (float(_volume) <= 0 and float(_amount) <= 0):
        continue
    _rows.append({{"symbol": {_json_literal(symbol)}, "datetime": _dt.strftime("%Y-%m-%d %H:%M:%S"), **_qclean(_quote)}})
_qemit(_rows)
'''


def _quote_code(symbols: list[str]) -> str:
    return _KERNEL_HELPERS + f'''
_out = []
_symbols = {_json_literal(symbols)}
_now = _qdatetime.now()
_quote_dt = _now.replace(second=0, microsecond=0)
_hm = _now.hour * 100 + _now.minute
if _now.weekday() >= 5 or _hm < 915:
    _quote_dt = (_now.replace(hour=15, minute=0, second=0, microsecond=0))
    while _quote_dt.weekday() >= 5 or _quote_dt.date() >= _now.date():
        from datetime import timedelta as _qtd
        _quote_dt -= _qtd(days=1)
elif 1130 < _hm < 1300:
    _quote_dt = _now.replace(hour=11, minute=30, second=0, microsecond=0)
elif _hm > 1500:
    _quote_dt = _now.replace(hour=15, minute=0, second=0, microsecond=0)

# Query one symbol at a time.  Some unauthenticated QMT formula sessions return
# the first symbol's quote for every member of a batch, which is unsafe.
for _symbol in _symbols:
    _raw = get_market_data(fields=["quoter"], stock_code=[_symbol], period="tick", count=-1)
    _quote = {{}}
    if isinstance(_raw, dict) and "lastPrice" in _raw:
        _quote = _raw
    elif hasattr(_raw, "iterrows"):
        for _idx, _row in _raw.iterrows():
            _quote = _row.get("quoter") or {{}}
            break
    _last = float(_quote.get("lastPrice") or 0)
    _volume = float(_quote.get("pvolume") or _quote.get("volume") or 0)
    _amount = float(_quote.get("amount") or 0)
    if _last <= 0 or (_volume <= 0 and _amount <= 0):
        continue
    _out.append({{
        "symbol": _symbol,
        "datetime": _quote_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "cumulativeVolume": _volume,
        "cumulativeAmount": _amount,
        **_qclean(_quote),
    }})
_qemit(_out)
'''


def _l2_code(symbol: str, kind: str, start: str, end: str, count: int) -> str:
    period = {"quote": "l2quote", "order": "l2order", "transaction": "l2transaction"}[kind]
    return _KERNEL_HELPERS + f'''
_df = get_market_data(
    fields=["quoter"], stock_code=[{_json_literal(symbol)}],
    start_time={_json_literal(start)}, end_time={_json_literal(end)},
    period={_json_literal(period)}, count={count},
)
_rows = []
for _idx, _row in _df.iterrows():
    _dt = _qtime(_idx)
    if _qin_session(_dt):
        _rows.append({{"symbol": {_json_literal(symbol)}, "datetime": _dt.strftime("%Y-%m-%d %H:%M:%S"), "data": _qclean(_row.get("quoter"))}})
_qemit(_rows)
'''


class QmtApi:
    def __init__(self, kernel: QmtKernel, push_store: QmtPushStore, *, qmt_root: Path | None = None) -> None:
        self.kernel = kernel
        self.push_store = push_store
        self.qmt_root = qmt_root.resolve(strict=True) if qmt_root is not None else None
        self._capabilities: dict[str, Any] | None = None
        self._capabilities_at = 0.0

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "source": "qmt",
            "kernel_id": self.kernel.kernel_id,
            "capabilities": self.capabilities(),
        }

    def capabilities(self, refresh: bool = False) -> dict[str, Any]:
        pushed = self.push_store.capabilities()
        # The authenticated ContextInfo stream is authoritative for live
        # quotes, but it does not prove that the historical QMT cache is
        # queryable.  An explicit refresh is therefore allowed to probe the
        # research kernel even while the live stream is healthy; normal health
        # checks remain cheap because they omit refresh=1.
        should_probe_kernel = refresh
        if should_probe_kernel:
            code = _KERNEL_HELPERS + r'''
_caps = {
    "realtime": False, "depth5": False, "minute": False, "daily_index": False, "tick": False,
    "symbol_routing": False,
    "l2_quote": False, "l2_order": False, "l2_transaction": False,
    "get_full_tick_api": callable(globals().get("get_full_tick")),
    "subscribe_quote_api": callable(globals().get("subscribe_quote")),
    "subscribe_whole_quote_api": callable(globals().get("subscribe_whole_quote")),
    "get_market_data_ex_api": callable(globals().get("get_market_data_ex")),
    "set_universe_api": callable(globals().get("set_universe")),
    "get_universe_api": callable(globals().get("get_universe")),
}
if _caps["get_universe_api"]:
    try:
        _caps["universe_sample"] = list(get_universe())[:20]
    except Exception as _universe_exc:
        _caps["universe_error"] = type(_universe_exc).__name__ + ": " + str(_universe_exc)
_caps["context_objects"] = [
    str(_name) for _name, _value in globals().items()
    if any(hasattr(_value, _method) for _method in ("get_full_tick", "subscribe_quote", "subscribe_whole_quote"))
][:20]
try:
    from xtquant import xtdata as _qxtdata
    _caps["xtdata_import"] = True
    try:
        _caps["xtdata_subscribe_test"] = _qclean(_qxtdata.subscribe_quote("000002.SZ", period="tick", count=0))
    except Exception as _sub_exc:
        _caps["xtdata_subscribe_error"] = type(_sub_exc).__name__ + ": " + str(_sub_exc)
    _full = _qxtdata.get_full_tick(["000001.SZ", "000002.SZ"])
    _caps["xtdata_full_tick_symbols"] = sorted(list((_full or {}).keys()))
except Exception as _xt_exc:
    _caps["xtdata_import"] = False
    _caps["xtdata_error"] = type(_xt_exc).__name__ + ": " + str(_xt_exc)
try:
    _tick = get_market_data(fields=["quoter"], stock_code=["000001.SZ"], period="tick", count=500)
    for _idx, _row in _tick.iterrows():
        _dt = _qtime(_idx)
        _q = _row.get("quoter") or {}
        if _qin_session(_dt) and float(_q.get("lastPrice") or 0) > 0:
            _caps["realtime"] = True
            _caps["tick"] = True
            _caps["depth5"] = len(_q.get("bidPrice") or []) >= 5 and len(_q.get("askPrice") or []) >= 5
    _daily_indices = []
    for _index_symbol in ("000001.SH", "399001.SZ"):
        try:
            _daily = get_market_data(fields=["close"], stock_code=[_index_symbol], period="1d", count=1)
            if len(_daily) > 0:
                _daily_indices.append(_index_symbol)
        except Exception:
            pass
    _caps["daily_index_symbols"] = _daily_indices
    _caps["daily_index"] = len(_daily_indices) == 2
    for _kind, _period in (("l2_quote", "l2quote"), ("l2_order", "l2order"), ("l2_transaction", "l2transaction")):
        try:
            _caps[_kind] = len(get_market_data(fields=["quoter"], stock_code=["000001.SZ"], period=_period, count=1)) > 0
        except Exception:
            _caps[_kind] = False
except Exception:
    pass

# Historical files remain queryable while MiniQMT/live quotes are unavailable.
# Inspect an actual in-session bar independently of the tick probe.
try:
    from datetime import timedelta as _qtd
    # count=1 can return an after-close heartbeat, even with explicit bounds.
    # A bounded recent history query reaches the same cache as real imports.
    _minute = get_market_data(
        fields=["close"], stock_code=["000001.SZ"], period="1m", count=-1,
        start_time=(_qdatetime.now() - _qtd(days=14)).strftime("%Y%m%d000000"),
        end_time=_qdatetime.now().strftime("%Y%m%d150000"),
    )
    if hasattr(_minute, "iterrows"):
        for _idx, _row in _minute.iterrows():
            _close = _qclean(_row.get("close"))
            if _qin_session(_qtime(_idx), minute=True) and _close is not None and float(_close) > 0:
                _caps["minute"] = True
except Exception as _minute_exc:
    _caps["minute_error"] = type(_minute_exc).__name__

# A connection is not usable unless two different liquid stocks can be routed
# independently.  This catches a QMT GUI session that is logged in while the
# MiniQMT/Python quote session is not authenticated.
_probe = {}
for _symbol in ("000001.SZ", "600000.SH"):
    try:
        _raw = get_market_data(fields=["quoter"], stock_code=[_symbol], period="tick", count=-1)
        _quote = {}
        if isinstance(_raw, dict) and "lastPrice" in _raw:
            _quote = _raw
        elif hasattr(_raw, "iterrows"):
            for _idx, _row in _raw.iterrows():
                _quote = _row.get("quoter") or {}
                break
        _last = float(_quote.get("lastPrice") or 0)
        if _last > 0:
            _probe[_symbol] = [
                _qclean(_quote.get("lastPrice")), _qclean(_quote.get("lastClose")),
                _qclean(_quote.get("open")), _qclean(_quote.get("high")),
                _qclean(_quote.get("low")), _qclean(_quote.get("amount")),
            ]
    except Exception:
        pass
_caps["routing_probe_symbols"] = sorted(_probe)
_caps["symbol_routing"] = len(_probe) == 2 and _probe["000001.SZ"] != _probe["600000.SH"]
if not _caps["symbol_routing"]:
    _caps["realtime"] = False
    _caps["depth5"] = False
    _caps["tick"] = False
    _caps["routing_error"] = "MiniQMT API unavailable or multi-symbol routing validation failed"
_qemit(_caps)
'''
            try:
                self._capabilities = dict(self.kernel.execute_json(code))
            except Exception as exc:
                self._capabilities = {
                    "realtime": False, "depth5": False, "minute": False,
                    "daily_index": False, "tick": False, "symbol_routing": False,
                    "l2_quote": False, "l2_order": False, "l2_transaction": False,
                    "kernel_error": f"{type(exc).__name__}: {exc}",
                }
            self._capabilities_at = time.monotonic()
        capabilities = dict(self._capabilities or {
            "realtime": False, "depth5": False, "minute": False,
            "daily_index": False, "tick": False, "symbol_routing": False,
            "l2_quote": False, "l2_order": False, "l2_transaction": False,
        })
        capabilities.update(pushed)
        if pushed.get("push_active"):
            capabilities["realtime"] = True
            capabilities["depth5"] = bool(pushed.get("push_depth5"))
            capabilities["symbol_routing"] = True
            capabilities["routing_probe_symbols"] = ["context_push"]
        if pushed.get("push_minute"):
            capabilities["minute"] = True
        if pushed.get("push_daily_index"):
            capabilities["daily_index"] = True
            capabilities["daily_index_symbols"] = ["context_push"]
        return capabilities

    def quote(self, symbols: list[str]) -> list[dict[str, Any]]:
        pushed = self.push_store.quotes(symbols, max_age_seconds=900)
        pushed_by_symbol = {row.get("symbol"): row for row in pushed}
        missing = [symbol for symbol in symbols if symbol not in pushed_by_symbol]
        if missing and (self._capabilities or {}).get("symbol_routing"):
            try:
                for row in self.kernel.execute_json(_quote_code(missing)):
                    pushed_by_symbol[row.get("symbol")] = row
            except Exception:
                pass
        return [pushed_by_symbol[symbol] for symbol in symbols if symbol in pushed_by_symbol]

    def kline(self, symbol: str, freq: str, start: str, end: str, count: int) -> list[dict[str, Any]]:
        rows = self._kline(symbol, freq, start, end, count)
        if not rows or self.qmt_root is None or freq != "1m" or not (start or end):
            return rows
        # Keep this outside the history/push fallback handler: a precision
        # conflict must reject the response, never return truncated volumes.
        return self._precise_minute_volume(symbol, rows)

    def _precise_minute_volume(self, symbol: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not _valid_symbol(symbol):
            raise ValueError("QMT precision identity conflict")
        script_dir = str(Path(__file__).resolve().parents[2] / "scripts")
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from qmt_dat import qmt_minute_volume_lots, read_qmt_minute_records

        tz = timezone(timedelta(hours=8))
        stamps = []
        for row in rows:
            if row.get("symbol") != symbol:
                raise ValueError("QMT precision identity conflict")
            when = datetime.fromisoformat(str(row["datetime"]))
            if when.tzinfo is not None or when.second or when.microsecond:
                raise ValueError("QMT precision timestamp conflict")
            stamps.append(int(when.replace(tzinfo=tz).timestamp()))
        if len(set(stamps)) != len(stamps):
            raise ValueError("QMT precision duplicate timestamp conflict")
        code, market = symbol.split(".")
        path = self.qmt_root / market / "60" / f"{code}.DAT"
        if not path.is_file():
            raise ValueError("QMT minute precision unavailable: DAT file missing")
        records = read_qmt_minute_records(path, min(stamps), max(stamps))
        volumes = qmt_minute_volume_lots(records)
        exact = {int(record["timestamp"]): (record, float(volume))
                 for record, volume in zip(records, volumes, strict=True)}
        if len(exact) != len(records):
            raise ValueError("QMT precision duplicate DAT timestamp conflict")
        result = []
        for row, stamp in zip(rows, stamps, strict=True):
            if stamp not in exact:
                raise ValueError("QMT minute precision unavailable: DAT record missing")
            record, volume = exact[stamp]
            for field in ("open", "high", "low", "close"):
                value = _as_float(row.get(field))
                if value is None or abs(value - float(record[field]) / 1000.0) > 0.0005:
                    raise ValueError(f"QMT precision {field} conflict")
            amount = _as_float(row.get("amount"))
            actual_volume = _as_float(row.get("volume"))
            if amount is None or abs(amount - float(record["amount"])) > 0.001:
                raise ValueError("QMT precision amount conflict")
            if actual_volume is None or not any(
                abs(actual_volume - expected) < 1e-6 for expected in (float(record["volume"]), volume)
            ):
                raise ValueError("QMT precision volume conflict")
            result.append({**row, "volume": volume, "volume_source": "qmt_dat_exact"})
        return result

    def _kline(self, symbol: str, freq: str, start: str, end: str, count: int) -> list[dict[str, Any]]:
        period = {"60m": "1h", "1h": "1h"}.get(freq, freq)
        if period not in {"1m", "5m", "15m", "30m", "1h", "1d"}:
            raise ValueError(f"unsupported QMT K-line frequency: {freq}")
        pushed: list[dict[str, Any]] = []
        if period in {"1m", "1d"}:
            pushed = self.push_store.klines(symbol, period, start, end, count)
            # A push cache is a live/collector cache and may contain only a
            # prefix of a historical session.  For explicit historical
            # ranges, keep it as fallback instead of treating any non-empty
            # result as authoritative.
            if pushed and not (start or end):
                return pushed
        # The push stream can be healthy while historical bars are only in
        # QMT's local cache.  Always consult the merged capabilities after a
        # push-cache miss; using self._capabilities directly kept this branch
        # disabled whenever the first health check came from push data.
        kernel_caps = self.capabilities(refresh=bool(start or end))
        capability = "daily_index" if period == "1d" else "minute"
        # Historical QMT queries are single-symbol operations.  They remain
        # useful even when the two-symbol live-routing probe is degraded (for
        # example while a QMT session is reconnecting), so an explicit range
        # must still be attempted whenever the requested dataset is relevant.
        should_try_kernel = bool(start or end) or (
            kernel_caps.get("symbol_routing") and kernel_caps.get(capability)
        )
        if should_try_kernel:
            try:
                historical = list(self.kernel.execute_json(_kline_code(symbol, period, start, end, count)))
                if historical:
                    if not pushed:
                        return historical
                    expected_by_period = {"1m": 240, "5m": 48, "15m": 16, "30m": 8, "1h": 4}
                    # A complete local QMT session is authoritative.  Do
                    # not append live-cache heartbeat/partial bars and turn
                    # a valid 240-bar day into an invalid 241-bar day.
                    expected = expected_by_period.get(period)
                    if expected and len(historical) >= expected:
                        return historical
                    # A closed historical day must never be padded from the
                    # live push cache.  Push rows belong to the current
                    # session and can otherwise make an old, incomplete QMT
                    # file look complete while mixing dates/row counts.
                    end_digits = "".join(ch for ch in str(end or "") if ch.isdigit())
                    if period != "1d" and len(end_digits) >= 8:
                        try:
                            if end_digits[:8] < datetime.now().strftime("%Y%m%d"):
                                return historical
                        except Exception:
                            pass
                    # QMT's local history is authoritative for overlapping
                    # timestamps; the push cache only fills gaps QMT did not
                    # return (for example an in-progress current session).
                    merged = {str(row.get("datetime")): row for row in historical}
                    # Never let a live push row outside an explicit historical
                    # range leak into the response.  Without this guard a
                    # short historical result (for example a file ending at
                    # 14:57) was padded with today's 15:01 cache row and
                    # appeared to contain 241/242 bars.
                    start_bound = self._range_time(start, daily=False) if start else ""
                    end_bound = self._range_time(end, daily=False, end=True) if end else ""
                    for row in pushed:
                        key = str(row.get("datetime"))
                        if start_bound and key < start_bound:
                            continue
                        if end_bound and key > end_bound:
                            continue
                        if period != "1d":
                            try:
                                hm = int(key[11:13]) * 100 + int(key[14:16])
                                if not (931 <= hm <= 1130 or 1301 <= hm <= 1500):
                                    continue
                            except (TypeError, ValueError):
                                continue
                        merged.setdefault(key, row)
                    result = [merged[key] for key in sorted(merged)]
                    return result if count < 0 else result[-max(1, min(count, 20000)):]
            except Exception:
                pass
        return pushed

    def ticks(self, symbol: str, start: str, end: str, count: int) -> list[dict[str, Any]]:
        kernel_caps = self._capabilities or {}
        if kernel_caps.get("symbol_routing") and kernel_caps.get("tick"):
            try:
                return list(self.kernel.execute_json(_ticks_code(symbol, start, end, count)))
            except Exception:
                pass
        # A latest quote is a snapshot, not a transaction tick. It is exposed
        # only as a one-row live fallback and is never advertised as tick data.
        quotes = self.push_store.quotes([symbol], max_age_seconds=900)
        return quotes[-1:] if quotes else []

    def auction(self, trade_date: str, symbols: list[str]) -> list[dict[str, Any]]:
        return self.push_store.auctions(trade_date, symbols)

    def l2(self, symbol: str, kind: str, start: str, end: str, count: int) -> list[dict[str, Any]]:
        capability = f"l2_{kind}"
        if not self.capabilities().get(capability):
            raise PermissionError("QMT L2 API is unavailable; confirm Guosen Shanghai/Shenzhen L2 entitlement")
        return list(self.kernel.execute_json(_l2_code(symbol, kind, start, end, count)))


class Handler(BaseHTTPRequestHandler):
    api: QmtApi
    api_key: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.address_string()} {fmt % args}")

    def _reply(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.api_key and self.headers.get("X-API-Key", "") != self.api_key:
                self._reply(HTTPStatus.UNAUTHORIZED, {"code": 401, "message": "invalid API key"})
                return
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            one = lambda key, default="": params.get(key, [default])[0]
            if parsed.path == "/api/health":
                data = self.api.health()
            elif parsed.path == "/api/capabilities":
                data = self.api.capabilities(refresh=one("refresh") == "1")
            elif parsed.path == "/api/quote":
                symbols = [s.strip().upper() for s in one("symbols", one("code")).split(",") if s.strip()]
                if not symbols:
                    raise ValueError("code or symbols is required")
                data = self.api.quote(symbols[:50])
            elif parsed.path == "/api/kline":
                data = self.api.kline(
                    one("code").upper(), one("freq", "1m").lower(), one("start"), one("end"), int(one("count", "-1")),
                )
            elif parsed.path == "/api/tick":
                data = self.api.ticks(one("code").upper(), one("start"), one("end"), int(one("count", "5000")))
            elif parsed.path == "/api/auction":
                symbols = [s.strip().upper() for s in one("symbols").split(",") if s.strip()]
                data = self.api.auction(one("date"), symbols)
            elif parsed.path.startswith("/api/l2/"):
                kind = parsed.path.rsplit("/", 1)[-1]
                if kind not in {"quote", "order", "transaction"}:
                    raise ValueError("invalid L2 kind")
                data = self.api.l2(one("code").upper(), kind, one("start"), one("end"), int(one("count", "5000")))
            else:
                self._reply(HTTPStatus.NOT_FOUND, {"code": 404, "message": "not found"})
                return
            self._reply(HTTPStatus.OK, {"code": 0, "data": data})
        except PermissionError as exc:
            self._reply(HTTPStatus.CONFLICT, {"code": 409, "message": str(exc)})
        except ValueError as exc:
            self._reply(HTTPStatus.BAD_REQUEST, {"code": 400, "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._reply(HTTPStatus.SERVICE_UNAVAILABLE, {"code": 503, "message": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.api_key and self.headers.get("X-API-Key", "") != self.api_key:
                self._reply(HTTPStatus.UNAUTHORIZED, {"code": 401, "message": "invalid API key"})
                return
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/push/quotes", "/api/push/minutes", "/api/push/daily"}:
                self._reply(HTTPStatus.NOT_FOUND, {"code": 404, "message": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            max_length = 64 * 1024 * 1024 if parsed.path == "/api/push/minutes" else 16 * 1024 * 1024
            if length <= 0 or length > max_length:
                raise ValueError(f"request body must be between 1 byte and {max_length // (1024 * 1024)} MiB")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                raise ValueError("rows must be a list")
            source = str(payload.get("source") or (
                "iquant_history" if parsed.path.endswith("/minutes")
                else "iquant_index_daily" if parsed.path.endswith("/daily")
                else "iquant_context"
            ))
            if parsed.path.endswith("/minutes"):
                data = self.api.push_store.ingest_minute_bars(rows, source)
            elif parsed.path.endswith("/daily"):
                data = self.api.push_store.ingest_daily_bars(rows, source)
            else:
                data = self.api.push_store.ingest_quotes(rows, source)
            self.api._capabilities_at = 0.0
            self._reply(HTTPStatus.OK, {"code": 0, "data": data})
        except (ValueError, json.JSONDecodeError) as exc:
            self._reply(HTTPStatus.BAD_REQUEST, {"code": 400, "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._reply(HTTPStatus.SERVICE_UNAVAILABLE, {"code": 503, "message": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="QMT local HTTP market-data bridge")
    parser.add_argument("--host", default=os.getenv("QMT_BRIDGE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QMT_BRIDGE_PORT", "8188")))
    parser.add_argument("--jupyter-url", default=os.getenv("QMT_JUPYTER_URL", "http://127.0.0.1:8888"))
    parser.add_argument("--config", type=Path, default=Path(os.getenv("QMT_JUPYTER_CONFIG", str(DEFAULT_CONFIG))))
    parser.add_argument("--push-db", type=Path, default=Path(os.getenv("QMT_PUSH_DB", str(DEFAULT_PUSH_DB))))
    parser.add_argument("--qmt-data-root", type=Path, default=os.getenv("QMT_DATA_ROOT") or None,
                        help="Read-only DAT root for exact fractional-lot volumes on bounded 1m requests")
    args = parser.parse_args()

    kernel = QmtKernel(args.jupyter_url, args.config, timeout=float(os.getenv("QMT_BRIDGE_TIMEOUT_SECONDS", "45")))
    Handler.api = QmtApi(kernel, QmtPushStore(args.push_db), qmt_root=args.qmt_data_root)
    Handler.api_key = os.getenv("QMT_BRIDGE_API_KEY", "").strip()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"QMT bridge listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        kernel.stop()


if __name__ == "__main__":
    main()
