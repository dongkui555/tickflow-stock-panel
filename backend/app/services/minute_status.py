"""Durable per-stock-day minute coverage state.

The state is an audit record, not a substitute for minute data.  In particular,
unknown rows stay unknown until a later validation proves the session complete.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


STATUS_FILE = Path("user_data/minute_coverage_status.json")
TERMINAL_STATUSES = {
    "no_trade", "pre_listing", "source_unsupported", "unknown",
    "regular_session", "regular_session_close_auction",
}


def code_of(value) -> str:
    text = str(value or "").upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else ""


def _day_of(value) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")[:10]


def key(symbol, day) -> str:
    return f"{code_of(symbol)}|{_day_of(day)}"


def _path(data_dir) -> Path:
    return Path(data_dir) / STATUS_FILE


def load(data_dir) -> dict:
    path = _path(data_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError, TypeError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    items = value.get("items")
    value["items"] = items if isinstance(items, dict) else {}
    return value


def save(data_dir, state: dict) -> None:
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["version"] = 1
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, default=str)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def upsert(data_dir, records: Iterable[dict]) -> dict:
    state = load(data_dir)
    items = state["items"]
    for raw in records:
        if not isinstance(raw, dict):
            continue
        symbol = code_of(raw.get("symbol"))
        day = _day_of(raw.get("date"))
        if not symbol or len(day) != 10:
            continue
        status = str(raw.get("status") or "unknown")
        item = dict(items.get(f"{symbol}|{day}") or {})
        item.update(raw, symbol=symbol, date=day, status=status)
        item["updated_at"] = datetime.now().isoformat(timespec="seconds")
        items[f"{symbol}|{day}"] = item
    save(data_dir, state)
    return state


def get(data_dir, symbol, day) -> dict | None:
    return load(data_dir)["items"].get(key(symbol, day))


def for_date(data_dir, symbols: Iterable[str], day) -> dict[str, dict]:
    items = load(data_dir)["items"]
    day_text = _day_of(day)
    return {
        code_of(symbol): items[f"{code_of(symbol)}|{day_text}"]
        for symbol in symbols
        if code_of(symbol) and f"{code_of(symbol)}|{day_text}" in items
    }


def counts(data_dir, day=None) -> dict[str, int]:
    values = load(data_dir)["items"].values()
    day_text = _day_of(day) if day is not None else None
    result: dict[str, int] = {}
    for item in values:
        if day_text is not None and item.get("date") != day_text:
            continue
        status = str(item.get("status") or "unknown")
        result[status] = result.get(status, 0) + 1
    return result


def is_minute_eligible(item: dict | None) -> bool:
    return bool(item and item.get("status") == "complete"
                and int(item.get("actual_minutes") or 0) >= int(item.get("expected_minutes") or 240))
