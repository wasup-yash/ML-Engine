import random
from threading import Lock
from typing import Any, Dict, Optional, Tuple


class ModelRegistry:
    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        traffic_split: Optional[Dict[str, float]] = None,
        shadow_model: Optional[str] = None,
    ) -> None:
        self._lock = Lock()
        self.models: Dict[str, Any] = models.copy() if models else {}
        self.traffic_split: Dict[str, float] = self._normalize_split(traffic_split or {})
        self.shadow_model = shadow_model

    def _normalize_split(self, split: Dict[str, float]) -> Dict[str, float]:
        filtered = {k: max(0.0, float(v)) for k, v in split.items()}
        total = sum(filtered.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in filtered.items()}

    def add_model(self, name: str, model: Any) -> None:
        with self._lock:
            self.models[name] = model

    def set_traffic_split(self, split: Dict[str, float]) -> None:
        with self._lock:
            self.traffic_split = self._normalize_split(split)

    def set_shadow_model(self, name: Optional[str]) -> None:
        with self._lock:
            self.shadow_model = name

    def route(self, request: Optional[Dict[str, Any]] = None) -> Tuple[str, Any]:
        del request
        with self._lock:
            if not self.models:
                raise ValueError("ModelRegistry has no models")

            if not self.traffic_split:
                first_name = next(iter(self.models.keys()))
                return first_name, self.models[first_name]

            sample = random.random()
            cumulative = 0.0
            chosen_name = None
            for name, weight in self.traffic_split.items():
                if name not in self.models:
                    continue
                cumulative += weight
                if sample <= cumulative:
                    chosen_name = name
                    break

            if chosen_name is None:
                chosen_name = next(iter(self.models.keys()))
            return chosen_name, self.models[chosen_name]

    def get_shadow(self) -> Optional[Tuple[str, Any]]:
        with self._lock:
            if not self.shadow_model:
                return None
            model = self.models.get(self.shadow_model)
            if model is None:
                return None
            return self.shadow_model, model
