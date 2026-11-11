"""Maintenance lock and version gate for the canonical identity migration."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


LOCK_NAME = ".identity-migration.lock"
VERSION_NAME = "identity_schema.json"
CANONICAL_VERSION = "canonical-identity-v1"


class IdentityMigrationLockError(RuntimeError):
    """A writer or second migration attempted to run during migration."""


def _lock_path(root: Path) -> Path:
    return Path(root) / LOCK_NAME


def require_writes_allowed(root: Path) -> None:
    path = _lock_path(Path(root))
    if path.exists():
        raise IdentityMigrationLockError(f"identity migration in progress: {path}")


@contextmanager
def migration_lock(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = _lock_path(root)
    payload = {
        "pid": os.getpid(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    created = False
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        created = True
        yield path
    except FileExistsError as exc:
        raise IdentityMigrationLockError(f"identity migration already locked: {path}") from exc
    finally:
        if created and path.exists():
            path.unlink()


def publish_identity_version(root: Path, *, version: str = CANONICAL_VERSION) -> Path:
    root = Path(root)
    target = root / VERSION_NAME
    temporary = target.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps({"schema_version": version}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def read_identity_version(root: Path) -> str | None:
    path = Path(root) / VERSION_NAME
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("schema_version") or "") or None
    except (OSError, ValueError, TypeError):
        return None
