import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from app.services import minute_gap_queue as queue


class QueueStorageTests(TestCase):
    def test_existing_schema_can_be_checked_without_schema_writes(self):
        with sqlite3.connect(":memory:") as conn:
            queue._ensure_schema(conn)

            def read_only_schema(action, first, second, *_):
                if action in {sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX}:
                    return sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_PRAGMA and first == "journal_mode" and second:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            conn.set_authorizer(read_only_schema)
            queue._ensure_schema(conn)
            self.assertEqual(conn.execute("SELECT count(*) FROM minute_gap_queue").fetchone()[0], 0)

    def test_connection_is_closed_after_context_exit(self):
        with TemporaryDirectory() as folder:
            with queue._connect(Path(folder)) as conn:
                self.assertEqual(conn.execute("SELECT 1").fetchone()[0], 1)
            with self.assertRaises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")

    def test_transient_open_failure_is_retried_before_transaction(self):
        connect = sqlite3.connect
        attempts = []

        def flaky_connect(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise sqlite3.OperationalError("unable to open database file")
            return connect(*args, **kwargs)

        with TemporaryDirectory() as folder:
            with patch.object(queue.sqlite3, "connect", side_effect=flaky_connect):
                self.assertEqual(queue.summary(Path(folder)), {})

    def test_transaction_failure_rolls_back_and_is_not_replayed(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            queue.summary(root)
            with self.assertRaisesRegex(sqlite3.OperationalError, "transaction failed"):
                with queue._connect(root) as conn:
                    conn.execute("CREATE TABLE IF NOT EXISTS rollback_probe (value INTEGER)")
                    conn.execute("INSERT INTO rollback_probe VALUES (1)")
                    raise sqlite3.OperationalError("transaction failed")
            with queue._connect(root) as check:
                self.assertEqual(check.execute("SELECT count(*) FROM rollback_probe").fetchone()[0], 0)
