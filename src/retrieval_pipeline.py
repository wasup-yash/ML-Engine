import hashlib
import json
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
from PIL import Image

from src.logger import get_logger

logger = get_logger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        index_path: str,
        embedding_dim: int = 512,
        processor: Any = None,
        model: Any = None,
    ) -> None:
        self.index_path = index_path
        self.embedding_dim = int(embedding_dim)
        self.meta_path = f"{index_path}.meta.json"
        self.processor = processor
        self.model = model
        self.metadata: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

        configured_dim = getattr(getattr(model, "config", None), "projection_dim", None)
        if configured_dim:
            self.embedding_dim = int(configured_dim)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.embedding_dim = self.index.d
            logger.info(f"Loaded retrieval index from {index_path} (dim={self.embedding_dim})")
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created new retrieval index at {index_path} (dim={self.embedding_dim})")

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        if len(self.metadata) > self.index.ntotal:
            logger.warning(
                "Retrieval metadata has more entries than the FAISS index; truncating metadata"
            )
            self.metadata = self.metadata[: self.index.ntotal]

    def _fallback_embedding(self, content: bytes) -> np.ndarray:
        digest = hashlib.sha256(content).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)

    def _embed_text_model(self, text: str) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        if self.processor is None:
            return None
        if not hasattr(self.model, "get_text_features"):
            return None

        # The deterministic FAISS fallback must work without a torch installation.
        # Only import torch when a CLIP-like encoder was actually configured.
        import torch

        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        vector = features[0].detach().cpu().float().numpy()
        norm = np.linalg.norm(vector) + 1e-12
        return (vector / norm).astype(np.float32)

    def _embed_image_model(self, image: Image.Image) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        if self.processor is None:
            return None
        if not hasattr(self.model, "get_image_features"):
            return None

        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        vector = features[0].detach().cpu().float().numpy()
        norm = np.linalg.norm(vector) + 1e-12
        return (vector / norm).astype(np.float32)

    def embed(self, text_or_image: Union[str, Image.Image]) -> np.ndarray:
        if isinstance(text_or_image, str):
            vector = self._embed_text_model(text_or_image)
            if vector is not None:
                return vector
            return self._fallback_embedding(b"text\0" + text_or_image.encode("utf-8"))

        if isinstance(text_or_image, Image.Image):
            vector = self._embed_image_model(text_or_image)
            if vector is not None:
                return vector
            image = text_or_image.convert("RGB")
            header = f"{image.width}x{image.height}\0".encode("ascii")
            return self._fallback_embedding(b"image\0" + header + image.tobytes())

        raise TypeError("embed expects either a string or PIL.Image.Image")

    def add(self, text_or_image: Union[str, Image.Image], metadata: Optional[Dict[str, Any]] = None) -> int:
        vector: np.ndarray = self.embed(text_or_image).astype(np.float32).reshape(1, -1)
        self._validate_dimension(vector)
        with self._lock:
            self.index.add(vector)
            self.metadata.append(metadata or {})
            return len(self.metadata) - 1

    def retrieve(self, query: Union[str, Image.Image], top_k: int = 3) -> List[Dict[str, Any]]:
        vector: np.ndarray = self.embed(query).astype(np.float32).reshape(1, -1)
        self._validate_dimension(vector)
        with self._lock:
            if self.index.ntotal == 0:
                return []
            k = max(1, min(int(top_k), self.index.ntotal))
            distances, indices = self.index.search(vector, k)
            metadata = list(self.metadata)
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            meta = metadata[idx] if idx < len(metadata) else {}
            results.append(
                {
                    "index": int(idx),
                    "distance": float(dist),
                    "metadata": meta,
                }
            )
        return results

    def _validate_dimension(self, vector: np.ndarray) -> None:
        if vector.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension {vector.shape[1]} does not match FAISS index dimension {self.index.d}"
            )

    def save(self) -> Tuple[str, str]:
        directory = os.path.dirname(self.index_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with self._lock:
            index_fd, index_tmp = tempfile.mkstemp(prefix="faiss-", dir=directory or ".")
            meta_fd, meta_tmp = tempfile.mkstemp(prefix="faiss-meta-", dir=directory or ".")
            os.close(index_fd)
            try:
                faiss.write_index(self.index, index_tmp)
                with os.fdopen(meta_fd, "w", encoding="utf-8") as f:
                    json.dump(self.metadata, f, ensure_ascii=True, separators=(",", ":"))
                os.replace(index_tmp, self.index_path)
                os.replace(meta_tmp, self.meta_path)
            except Exception:
                for path in (index_tmp, meta_tmp):
                    if os.path.exists(path):
                        os.unlink(path)
                raise
        return self.index_path, self.meta_path
