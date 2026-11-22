"""SQLite cache with durable per-evaluation completion records."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from .checkpoints import json_safe


class EvaluationCache:
    """Process-safe evaluation results and retry state.

    Connections are short lived and never serialized to workers. A result only
    becomes reusable in the same transaction that marks it complete. Pending
    records are diagnostic and can always be rescheduled after interruption.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS evaluations (
                key TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, error TEXT, updated REAL NOT NULL)""")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def get(self, key: str) -> dict[str, Any] | None:
        """Return only a committed complete result."""
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM evaluations WHERE key=? AND status='complete'", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Commit compact metrics for one completed evaluation."""
        payload = json.dumps(json_safe(value), sort_keys=True, allow_nan=False)
        with self._connect() as connection:
            connection.execute("""INSERT INTO evaluations(key,status,payload,updated)
                VALUES (?,'complete',?,?) ON CONFLICT(key) DO UPDATE SET
                status='complete',payload=excluded.payload,error=NULL,updated=excluded.updated""", (key, payload, time.time()))

    def pending(self, key: str) -> None:
        """Record an attempt without overwriting an already completed result."""
        with self._connect() as connection:
            connection.execute("""INSERT INTO evaluations(key,status,attempts,updated)
                VALUES (?,'pending',1,?) ON CONFLICT(key) DO UPDATE SET
                status='pending',attempts=attempts+1,updated=excluded.updated
                WHERE evaluations.status!='complete'""", (key, time.time()))

    def failed(self, key: str, error: str) -> None:
        """Retain failure details for interrupted or retryable work."""
        with self._connect() as connection:
            connection.execute("""UPDATE evaluations SET status='failed',error=?,updated=?
                WHERE key=? AND status!='complete'""", (str(error), time.time(), key))

    mark_failed = failed

    def completed(self) -> list[dict[str, Any]]:
        """Read all completed result payloads in stable key order."""
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM evaluations WHERE status='complete' ORDER BY key").fetchall()
        return [json.loads(row[0]) for row in rows]

    def incomplete(self) -> list[dict[str, Any]]:
        """Describe pending/failed evaluations for checkpoint recovery."""
        with self._connect() as connection:
            rows = connection.execute("SELECT key,status,attempts,error FROM evaluations WHERE status!='complete' ORDER BY key").fetchall()
        return [dict(zip(("key", "status", "attempts", "error"), row)) for row in rows]
