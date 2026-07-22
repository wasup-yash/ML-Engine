import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class SQLiteTenantStore:
    """Durable tenant quotas, key revocations, and usage metering for one deployment."""

    def __init__(self, database_path: str, default_quotas: Dict[str, Any]) -> None:
        directory = os.path.dirname(os.path.abspath(database_path))
        os.makedirs(directory, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.default_quotas = self._normalize_quotas(default_quotas)
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    id TEXT PRIMARY KEY,
                    quotas_json TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS key_revocations (
                    key_id TEXT PRIMARY KEY,
                    revoked_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_quotas(quotas: Dict[str, Any]) -> Dict[str, int]:
        return {
            "max_concurrent_jobs": max(1, int(quotas.get("max_concurrent_jobs", 1))),
            "requests_per_minute": max(1, int(quotas.get("requests_per_minute", 600))),
        }

    def ensure_tenant(self, tenant_id: str) -> None:
        now = self._now()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO tenants VALUES (?, ?, ?, ?, ?)",
                (tenant_id, json.dumps(self.default_quotas), 1, now, now),
            )

    def create_or_update(self, tenant_id: str, quotas: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.ensure_tenant(tenant_id)
        current = self.get(tenant_id)
        merged = self._normalize_quotas({**current["quotas"], **(quotas or {})})
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE tenants SET quotas_json = ?, active = 1, updated_at = ? WHERE id = ?",
                (json.dumps(merged), self._now(), tenant_id),
            )
        return self.get(tenant_id)

    def get(self, tenant_id: str) -> Dict[str, Any]:
        self.ensure_tenant(tenant_id)
        with self._lock:
            row = self._connection.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone()
        if row is None:
            raise KeyError(tenant_id)
        return {
            "id": row["id"],
            "quotas": json.loads(row["quotas_json"]),
            "active": bool(row["active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT id FROM tenants ORDER BY id").fetchall()
        return [self.get(row["id"]) for row in rows]

    def is_key_active(self, key_id: str, tenant_id: str) -> bool:
        tenant = self.get(tenant_id)
        with self._lock:
            revoked = self._connection.execute(
                "SELECT 1 FROM key_revocations WHERE key_id = ?", (key_id,)
            ).fetchone()
        return tenant["active"] and revoked is None

    def revoke_key(self, key_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO key_revocations VALUES (?, ?)", (key_id, self._now())
            )

    def record_usage(
        self, tenant_id: str, event_type: str, quantity: float = 1.0, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.ensure_tenant(tenant_id)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO usage_events (tenant_id, event_type, quantity, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (tenant_id, event_type, float(quantity), json.dumps(metadata or {}), self._now()),
            )

    def usage_summary(self, tenant_id: str) -> Dict[str, float]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT event_type, SUM(quantity) AS total FROM usage_events WHERE tenant_id = ? GROUP BY event_type",
                (tenant_id,),
            ).fetchall()
        return {row["event_type"]: float(row["total"]) for row in rows}

    def close(self) -> None:
        with self._lock:
            self._connection.close()
