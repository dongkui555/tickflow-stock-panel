from pathlib import Path

import pytest

from app.services.identity_schema import (
    IdentityMigrationLockError,
    migration_lock,
    require_writes_allowed,
)


def test_writer_is_blocked_while_migration_lock_exists(tmp_path: Path):
    with migration_lock(tmp_path):
        with pytest.raises(IdentityMigrationLockError):
            require_writes_allowed(tmp_path)
    require_writes_allowed(tmp_path)


def test_second_migration_lock_is_rejected(tmp_path: Path):
    with migration_lock(tmp_path):
        with pytest.raises(IdentityMigrationLockError):
            with migration_lock(tmp_path):
                pass
