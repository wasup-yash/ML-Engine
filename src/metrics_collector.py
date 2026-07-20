import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, Histogram, generate_latest

from src.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    def __init__(
        self,
        poll_interval_sec: float = 1.0,
        gpu_index: int = 0,
    ) -> None:
        self.registry = CollectorRegistry()
        self.inference_latency_seconds = Histogram(
            "inference_latency_seconds",
            "Inference latency in seconds",
            labelnames=("endpoint", "model"),
            registry=self.registry,
        )
        self.batch_size_observed = Histogram(
            "batch_size_observed",
            "Observed request batch sizes",
            labelnames=("endpoint",),
            registry=self.registry,
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        )
        self.gpu_memory_used_bytes = Gauge(
            "gpu_memory_used_bytes",
            "GPU memory used in bytes",
            labelnames=("gpu_index",),
            registry=self.registry,
        )

        self.poll_interval_sec = max(0.2, float(poll_interval_sec))
        self.gpu_index = int(gpu_index)
        self._stop_event = threading.Event()
        self._gpu_thread: Optional[threading.Thread] = None

    @contextmanager
    def track_inference(self, endpoint: str, model_name: str = "default"):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.inference_latency_seconds.labels(endpoint=endpoint, model=model_name).observe(elapsed)

    def observe_batch_size(self, endpoint: str, batch_size: int) -> None:
        self.batch_size_observed.labels(endpoint=endpoint).observe(max(1, int(batch_size)))

    def _gpu_poll_loop(self) -> None:
        try:
            import pynvml
        except Exception as e:
            logger.warning(f"pynvml unavailable, GPU memory metrics disabled: {e}")
            return

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as e:
            logger.warning(f"Failed to initialise NVML for GPU metrics: {e}")
            return

        while not self._stop_event.is_set():
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_memory_used_bytes.labels(gpu_index=str(self.gpu_index)).set(float(mem.used))
            except Exception:
                pass
            self._stop_event.wait(self.poll_interval_sec)

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def start_gpu_poller(self) -> None:
        if self._gpu_thread and self._gpu_thread.is_alive():
            return
        self._stop_event.clear()
        self._gpu_thread = threading.Thread(target=self._gpu_poll_loop, daemon=True, name="gpu-metrics-poller")
        self._gpu_thread.start()

    def stop_gpu_poller(self) -> None:
        self._stop_event.set()
        if self._gpu_thread and self._gpu_thread.is_alive():
            self._gpu_thread.join(timeout=2.0)

    def gpu_memory_info(self) -> Optional[Dict[str, int]]:
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {"total": int(memory.total), "used": int(memory.used), "free": int(memory.free)}
        except Exception:
            return None

    def render_prometheus(self) -> Dict[str, Any]:
        payload = generate_latest(self.registry)
        return {"content": payload, "content_type": CONTENT_TYPE_LATEST}
