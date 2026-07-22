import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class SQLiteJobStore:
    """Durable self-host job store. Use a shared database backend for replicas."""

    def __init__(self, database_path: str) -> None:
        directory = os.path.dirname(os.path.abspath(database_path))
        os.makedirs(directory, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(self, job_id: str, tenant_id: str, payload: Dict[str, Any]) -> None:
        now = self._now()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, tenant_id, "queued", json.dumps(payload), "{}", now, now),
            )

    def update(self, job_id: str, status: str, result: Optional[Dict[str, Any]] = None) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE jobs SET status = ?, result_json = ?, updated_at = ? WHERE id = ?",
                (status, json.dumps(result or {}), self._now(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown job {job_id}")

    def get(self, job_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        query = "SELECT * FROM jobs WHERE id = ?"
        params: List[Any] = [job_id]
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        with self._lock:
            row = self._connection.execute(query, params).fetchone()
        if row is None:
            raise KeyError(job_id)
        return {
            "id": row["id"],
            "tenant_id": row["tenant_id"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            **json.loads(row["result_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def active_count(self, tenant_id: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
        params: List[Any] = []
        if tenant_id is not None:
            query += " AND tenant_id = ?"
            params.append(tenant_id)
        with self._lock:
            return int(self._connection.execute(query, params).fetchone()[0])

    def list(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT id FROM jobs"
        params: List[Any] = []
        if tenant_id is not None:
            query += " WHERE tenant_id = ?"
            params.append(tenant_id)
        query += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self.get(row["id"]) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()
