import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Iterable


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    key_id: str
    tenant_id: str
    scopes: frozenset[str]


class APIKeyAuthenticator:
    """Authenticates SHA-256 hashed API keys supplied via an environment variable."""

    def __init__(self, config: Dict[str, Any], tenant_store: Any = None) -> None:
        self.required = bool(config.get("auth_required", True))
        self.hashes_env = str(config.get("api_key_hashes_env", "ML_ENGINE_API_KEY_HASHES"))
        self.tenant_store = tenant_store
        self._keys = self._load_keys()

        if self.required and not self._keys:
            raise ValueError(
                f"Authentication is required but {self.hashes_env} does not contain API key hashes"
            )

    def _load_keys(self) -> Dict[str, Principal]:
        raw = os.getenv(self.hashes_env, "{}")
        try:
            records = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.hashes_env} must contain a JSON object") from exc
        if not isinstance(records, dict):
            raise ValueError(f"{self.hashes_env} must contain a JSON object")

        keys: Dict[str, Principal] = {}
        for key_hash, metadata in records.items():
            if not isinstance(metadata, dict) or len(key_hash) != 64:
                raise ValueError("API key hash records must map SHA-256 hashes to metadata objects")
            scopes = metadata.get("scopes", [])
            if not isinstance(scopes, list) or not all(isinstance(scope, str) for scope in scopes):
                raise ValueError("API key scopes must be a list of strings")
            keys[key_hash.lower()] = Principal(
                key_id=str(metadata.get("key_id", key_hash[:12])),
                tenant_id=str(metadata.get("tenant_id", "default")),
                scopes=frozenset(scopes),
            )
        return keys

    def authenticate(self, supplied_key: str | None) -> Principal:
        if not self.required:
            return Principal("development", "development", frozenset({"predict", "admin"}))
        if not supplied_key:
            raise AuthenticationError("Missing X-API-Key header")

        candidate = hashlib.sha256(supplied_key.encode("utf-8")).hexdigest()
        for key_hash, principal in self._keys.items():
            if hmac.compare_digest(candidate, key_hash):
                if self.tenant_store is not None and not self.tenant_store.is_key_active(
                    principal.key_id, principal.tenant_id
                ):
                    raise AuthenticationError("API key has been revoked or tenant is inactive")
                return principal
        raise AuthenticationError("Invalid API key")

    @staticmethod
    def require_scope(principal: Principal, required_scope: str) -> None:
        if required_scope not in principal.scopes:
            raise AuthorizationError(f"API key does not have {required_scope!r} scope")


class FixedWindowRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = max(1, int(requests_per_minute))
        self._lock = Lock()
        self._windows: Dict[str, tuple[int, int]] = {}

    def check(self, key_id: str, requests_per_minute: int | None = None) -> bool:
        limit = max(1, int(requests_per_minute or self.requests_per_minute))
        now = int(time.time())
        window = now // 60
        with self._lock:
            previous_window, count = self._windows.get(key_id, (window, 0))
            if previous_window != window:
                count = 0
            count += 1
            self._windows[key_id] = (window, count)
            return count <= limit


def is_trusted_path(path: str, trusted_roots: Iterable[str]) -> bool:
    resolved_path = os.path.realpath(path)
    for root in trusted_roots:
        resolved_root = os.path.realpath(root)
        try:
            if os.path.commonpath([resolved_path, resolved_root]) == resolved_root:
                return True
        except ValueError:
            continue
    return False


def validate_safetensors_file(path: str) -> None:
    file_size = os.path.getsize(path)
    if file_size < 9:
        raise ValueError("Invalid safetensors file: file is too small")
    with open(path, "rb") as file:
        header_length = int.from_bytes(file.read(8), byteorder="little", signed=False)
        if header_length <= 0 or header_length > 100 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length")
        if 8 + header_length > file_size:
            raise ValueError("Invalid safetensors header exceeds file size")
        try:
            header = json.loads(file.read(header_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(descriptor, dict) or "dtype" not in descriptor or "data_offsets" not in descriptor:
            raise ValueError("Invalid safetensors tensor descriptor")
