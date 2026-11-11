"""Durable after-close retries sharing the pipeline's existing execution slot."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class AfterCloseSync:
    def __init__(self, data_dir: Path, clock=None, *, state_file="after_close_sync.json"):
        self.path = data_dir / "user_data" / state_file
        self.lock = threading.Lock()
        self.recovered = False
        self.clock = clock

    def due(self, now: datetime, schedule: dict, trading_day: bool | None) -> bool:
        """Read-only priority check before allowing a background repair batch."""
        due = now.replace(hour=schedule["hour"], minute=schedule["minute"], second=0, microsecond=0)
        if trading_day is not True or now.weekday() >= 5 or now < due:
            return False
        state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if str(now.date()) not in state:
            return True
        cutoff = str(now.date() - timedelta(days=3))
        return any(day >= cutoff and item.get("status") in {"pending", "running", "retry"}
                   and item.get("retry_at", "") <= now.isoformat() for day, item in state.items())

    def _save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def tick(self, now: datetime, schedule: dict, trading_day: bool | None, runner) -> dict:
        if not self.lock.acquire(blocking=False):
            return {"status": "running"}
        try:
            state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            if not self.recovered:
                for item in state.values():
                    if item["status"] == "running":
                        item["status"] = "pending"
                self.recovered = True
            day = now.date().isoformat()
            due = now.replace(
                hour=schedule["hour"], minute=schedule["minute"], second=0, microsecond=0
            )
            # Unknown calendar is not treated as a confirmed trading session.
            if now >= due and now.weekday() < 5 and trading_day is True and day not in state:
                state[day] = {
                    "status": "pending",
                    "attempts": 0,
                    "scope": "all",
                    "mode": "incremental",
                }
            if now < due or now.weekday() >= 5 or trading_day is not True:
                self._save(state)
                return state.get(day, {"status": "not_due"})
            for target, item in sorted(state.items()):
                if item["status"] not in {"pending", "retry"}:
                    continue
                if target < (now.date() - timedelta(days=3)).isoformat():
                    item["status"] = "expired"
                    continue
                if item.get("retry_at", "") > now.isoformat():
                    continue
                started = False

                def mark_started():
                    nonlocal started
                    started = True
                    item.update(
                        status="running", attempts=item["attempts"] + 1, started_at=now.isoformat()
                    )
                    self._save(state)

                self._save(state)
                try:
                    success = runner(mark_started)
                except Exception:
                    logger.exception("after-close sync runner failed")
                    success = False
                finished = max(now, self.clock() if self.clock else datetime.now(now.tzinfo))
                if success:
                    item.update(status="succeeded", finished_at=finished.isoformat())
                elif not started:
                    item.update(status="pending", reason="Another data job owns the execution slot")
                else:
                    item.update(
                        status="retry" if item["attempts"] < 3 else "failed",
                        retry_at=(finished + timedelta(minutes=15)).isoformat(),
                    )
                self._save(state)
                return item
            self._save(state)
            return state.get(day, {"status": "not_due"})
        finally:
            self.lock.release()
