"""Insert-only sector forecasts and append-only validation revisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)


class ForecastStore:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir).resolve() / "user_data" / "sector_forecasts_v2.sqlite3"

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id TEXT PRIMARY KEY, as_of TEXT NOT NULL, origin TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS validations (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, forecast_id TEXT NOT NULL,
                horizon INTEGER NOT NULL CHECK(horizon IN (1,2)),
                digest TEXT NOT NULL, payload TEXT NOT NULL,
                UNIQUE(forecast_id, horizon, digest)
            );
            CREATE TRIGGER IF NOT EXISTS forecasts_no_update BEFORE UPDATE ON forecasts
                BEGIN SELECT RAISE(ABORT, 'Forecasts are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS forecasts_no_delete BEFORE DELETE ON forecasts
                BEGIN SELECT RAISE(ABORT, 'Forecasts are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS validations_no_update BEFORE UPDATE ON validations
                BEGIN SELECT RAISE(ABORT, 'Validations are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS validations_no_delete BEFORE DELETE ON validations
                BEGIN SELECT RAISE(ABORT, 'Validations are append-only'); END;
        """)
        return db

    def save(self, snapshot: dict) -> dict:
        db = self._connect()
        try:
            with db:
                db.execute(
                    "INSERT OR IGNORE INTO forecasts VALUES (?, ?, ?, ?)",
                    (snapshot["id"], snapshot["as_of"], snapshot["origin"], encode(snapshot)),
                )
                row = db.execute(
                    "SELECT payload FROM forecasts WHERE id=?", (snapshot["id"],)
                ).fetchone()
            return json.loads(row[0])
        finally:
            db.close()

    def list(self, origin: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
            rows = db.execute(
                "SELECT payload FROM forecasts WHERE (? IS NULL OR origin=?) ORDER BY as_of DESC, rowid DESC",
                (origin, origin),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def append_validation(self, forecast_id: str, horizon: int, payload: dict):
        # The evidence hash excludes wall-clock validation time for idempotent refreshes.
        evidence = {k: v for k, v in payload.items() if k != "validated_at"}
        digest = hashlib.sha256(encode(evidence).encode()).hexdigest()
        db = self._connect()
        try:
            with db:
                if not db.execute("SELECT 1 FROM forecasts WHERE id=?", (forecast_id,)).fetchone():
                    raise ValueError("Unknown forecast")
                db.execute(
                    "INSERT OR IGNORE INTO validations(forecast_id,horizon,digest,payload) VALUES(?,?,?,?)",
                    (forecast_id, horizon, digest, encode(payload)),
                )
        finally:
            db.close()

    def validations(self, forecast_id: str) -> list[dict]:
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
            rows = db.execute(
                "SELECT seq,horizon,payload FROM validations WHERE forecast_id=? ORDER BY seq",
                (forecast_id,),
            ).fetchall()
        return [{"revision": r[0], "horizon_days": r[1], **json.loads(r[2])} for r in rows]
