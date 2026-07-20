import os
from abc import ABC, abstractmethod
from typing import Any, Dict


class ArtifactStorage(ABC):
    @abstractmethod
    def materialize(self, uri: str) -> str:
        """Return a local, readable artifact path for a model URI."""


class LocalArtifactStorage(ArtifactStorage):
    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)

    def materialize(self, uri: str) -> str:
        path = uri.removeprefix("file://")
        resolved = os.path.realpath(path if os.path.isabs(path) else os.path.join(self.root, path))
        if os.path.commonpath([resolved, self.root]) != self.root:
            raise PermissionError("Artifact path escapes configured storage root")
        if not os.path.exists(resolved):
            raise FileNotFoundError(resolved)
        return resolved


def create_artifact_storage(config: Dict[str, Any]) -> ArtifactStorage:
    backend = config.get("artifact_storage_backend", "local")
    if backend == "local":
        return LocalArtifactStorage(config.get("artifact_storage_root", "."))
    if backend in {"s3", "gcs"}:
        raise NotImplementedError(
            f"{backend} storage requires a deployment-specific adapter. "
            "Use the ArtifactStorage interface with your cloud SDK and credential provider."
        )
    raise ValueError(f"Unsupported artifact_storage_backend: {backend}")
