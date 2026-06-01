import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
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

    def _fallback_embedding(self, key: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
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

        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
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

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        vector = features[0].detach().cpu().float().numpy()
        norm = np.linalg.norm(vector) + 1e-12
        return (vector / norm).astype(np.float32)

    def embed(self, text_or_image: Union[str, Image.Image]) -> np.ndarray:
        if isinstance(text_or_image, str):
            vector = self._embed_text_model(text_or_image)
            if vector is not None:
                return vector
            return self._fallback_embedding(f"text::{text_or_image}")

        if isinstance(text_or_image, Image.Image):
            vector = self._embed_image_model(text_or_image)
            if vector is not None:
                return vector
            return self._fallback_embedding(f"image::{text_or_image.size}")

        raise TypeError("embed expects either a string or PIL.Image.Image")

    def add(self, text_or_image: Union[str, Image.Image], metadata: Optional[Dict[str, Any]] = None) -> int:
        vector = self.embed(text_or_image).astype(np.float32).reshape(1, -1)
        self.index.add(vector)
        self.metadata.append(metadata or {})
        return len(self.metadata) - 1

    def retrieve(self, query: Union[str, Image.Image], top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        vector = self.embed(query).astype(np.float32).reshape(1, -1)
        k = max(1, min(int(top_k), self.index.ntotal))
        distances, indices = self.index.search(vector, k)
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append(
                {
                    "index": int(idx),
                    "distance": float(dist),
                    "metadata": meta,
                }
            )
        return results

    def save(self) -> Tuple[str, str]:
        directory = os.path.dirname(self.index_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=True, indent=2)
        return self.index_path, self.meta_path
