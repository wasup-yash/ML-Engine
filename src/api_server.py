import base64
import inspect
import importlib.util
import json
import os
import threading
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from src.adapter_loader import load_adapter, merge_lora
from src.config_manager import get_config
from src.dynamic_batcher import AsyncBatcher
from src.finetune_engine import run_lora_finetune
from src.inference_engine import batch_inference
from src.logger import get_logger
from src.metrics_collector import MetricsCollector
from src.model_loader import load_model
from src.model_registry import ModelRegistry
from src.multimodal_loader import load_multimodal_model, run_multimodal_inference
from src.retrieval_pipeline import RetrievalPipeline
from src.video_pipeline import infer_sequence

logger = get_logger(__name__)


class PredictionInput(BaseModel):
    data: Union[List[List[float]], List[float], Dict[str, Any]] = Field(
        ...,
        description="Input data for prediction. Can be a 2D array, 1D array, or dictionary of features",
    )


class PredictionResponse(BaseModel):
    prediction: Union[List[Any], Any] = Field(..., description="Model prediction(s)")


class FineTuneRequest(BaseModel):
    dataset: List[Dict[str, Any]] = Field(..., description="Small labelled dataset")
    config: Dict[str, Any] = Field(default_factory=dict, description="Optional fine-tune overrides")


class VideoPredictionRequest(BaseModel):
    frames: List[str] = Field(..., description="Base64-encoded frames in temporal order")


class APIServer:
    def __init__(
        self,
        model: Any,
        host: str = "127.0.0.1",
        port: int = 5000,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.host = host
        self.port = port
        self.config = config or get_config()
        self.model_lock = threading.Lock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.multimodal_bundle = None
        self.multimodal_lock = threading.Lock()
        self.retrieval_pipeline: Optional[RetrievalPipeline] = None
        self.pre_hooks: List[Any] = []
        self.post_hooks: List[Any] = []

        self.metrics = MetricsCollector(
            poll_interval_sec=float(self.config.get("metrics_gpu_poll_interval_sec", 1.0))
        )
        self.metrics.start_gpu_poller()

        self.model_registry = self._build_model_registry(model)
        self.infer_batcher = AsyncBatcher(
            infer_fn=self._infer_batch,
            max_batch_size=int(self.config.get("batch_max_size", 32)),
            max_wait_ms=int(self.config.get("batch_max_wait_ms", 20)),
        )

        self.app = FastAPI(
            title="ML Model Serving API",
            description="API for serving machine learning model predictions",
            version="0.5.0",
        )

        self.plugins: List[Any] = []
        self._register_routes()
        self._load_plugins()

    def _build_model_registry(self, default_model: Any) -> ModelRegistry:
        registry = ModelRegistry(
            models={"default": default_model},
            traffic_split=self.config.get("traffic_split", {}),
            shadow_model=self.config.get("shadow_model"),
        )

        variants = self.config.get("model_variants", {}) or {}
        if not isinstance(variants, dict):
            logger.warning("model_variants must be a dictionary; ignoring")
            return registry

        for slot_name, slot_config in variants.items():
            if not isinstance(slot_config, dict):
                logger.warning(f"Skipping model variant {slot_name}: expected dict config")
                continue
            try:
                loaded = load_model(
                    model_path=slot_config.get("model_path", self.config["model_path"]),
                    format=slot_config.get("model_format", self.config.get("model_format")),
                    quantization=slot_config.get("quantization", self.config.get("quantization")),
                    adapter_path=slot_config.get("adapter_path", self.config.get("adapter_path")),
                    merge_adapter=bool(slot_config.get("merge_adapter", False)),
                )
                registry.add_model(slot_name, loaded)
                logger.info(f"Loaded model variant slot={slot_name}")
            except Exception as e:
                logger.error(f"Failed to load model variant slot={slot_name}: {e}")

        return registry

    def _load_plugins(self) -> None:
        plugin_dir = self.config.get("plugin_dir", "plugin")
        if not os.path.exists(plugin_dir):
            return
        for file_name in os.listdir(plugin_dir):
            if file_name.endswith(".py"):
                self._load_plugin(os.path.join(plugin_dir, file_name))

    def _load_plugin(self, plugin_path: str) -> None:
        module_name = os.path.basename(plugin_path)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        if hasattr(module, "register"):
            module.register(self)
            self.plugins.append(module)
            logger.info(f"Loaded plugin: {module_name}")

    def add_middleware(self, middleware_class: Any) -> None:
        self.app.add_middleware(middleware_class)

    def register_hook(self, hook_type: str, hook_func: Any) -> None:
        if hook_type == "pre":
            self.pre_hooks.append(hook_func)
            return
        if hook_type == "post":
            self.post_hooks.append(hook_func)
            return
        if hook_type in {"startup", "shutdown"}:
            self.app.add_event_handler(hook_type, hook_func)
            return
        logger.warning(f"Unsupported hook_type={hook_type}; hook ignored")

    def _route_model(self, request_context: Optional[Dict[str, Any]] = None) -> tuple[str, Any]:
        try:
            return self.model_registry.route(request_context)
        except Exception:
            return "default", self.model

    def _predict_with_model(self, model: Any, data: Union[List[List[float]], List[float], Dict[str, Any]]) -> Any:
        if isinstance(data, dict):
            return model.predict(data)

        data_array = np.array(data)
        if data_array.ndim == 1:
            data_array = data_array.reshape(1, -1)
        return model.predict(data_array)

    def _sample_count(self, data: Union[List[List[float]], List[float], Dict[str, Any]]) -> int:
        if isinstance(data, dict):
            return 1
        arr = np.array(data)
        if arr.ndim <= 1:
            return 1
        return int(arr.shape[0])

    def _normalize_for_json(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, list):
            return [self._normalize_for_json(v) for v in value]
        if isinstance(value, tuple):
            return [self._normalize_for_json(v) for v in value]
        if isinstance(value, dict):
            return {k: self._normalize_for_json(v) for k, v in value.items()}
        return value

    def _infer_batch(self, batch_inputs: List[Any]) -> List[Any]:
        model_name, model = self._route_model({"endpoint": "/infer"})
        self.metrics.observe_batch_size("/infer", len(batch_inputs))
        with self.metrics.track_inference("/infer", model_name):
            with self.model_lock:
                outputs = batch_inference(model, batch_inputs, batch_size=len(batch_inputs))
        return self._normalize_for_json(outputs)

    def _get_multimodal_bundle(self):
        if self.multimodal_bundle is None:
            with self.multimodal_lock:
                if self.multimodal_bundle is None:
                    self.multimodal_bundle = load_multimodal_model(self.config)
        return self.multimodal_bundle

    def _get_retrieval_pipeline(self) -> Optional[RetrievalPipeline]:
        index_path = self.config.get("retrieval_index_path")
        if not index_path:
            return None

        if self.retrieval_pipeline is None:
            bundle = self._get_multimodal_bundle()
            self.retrieval_pipeline = RetrievalPipeline(
                index_path=index_path,
                embedding_dim=int(self.config.get("retrieval_embedding_dim", 512)),
                processor=getattr(bundle, "processor", None),
                model=getattr(bundle, "model", None),
            )
        return self.retrieval_pipeline

    def _run_finetune_job(
        self, job_id: str, dataset: List[Dict[str, Any]], config_overrides: Dict[str, Any]
    ) -> None:
        try:
            self.jobs[job_id]["status"] = "running"
            merged_config = self.config.copy()
            merged_config.update(config_overrides or {})

            with self.model_lock:
                result = run_lora_finetune(self.model, dataset, merged_config)
                self.model = result["model"]
                self.model_registry.add_model("default", self.model)

            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["adapter_path"] = result["adapter_path"]
        except Exception as e:
            logger.exception("Fine-tune job failed")
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)

    def _decode_frame(self, frame_b64: str) -> np.ndarray:
        payload = frame_b64
        if "," in frame_b64 and frame_b64.strip().startswith("data:"):
            payload = frame_b64.split(",", 1)[1]
        frame_bytes = base64.b64decode(payload)
        image = Image.open(BytesIO(frame_bytes)).convert("RGB")
        return np.array(image)

    def _retrieved_context(self, retrieved_items: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for item in retrieved_items:
            meta = item.get("metadata", {})
            if isinstance(meta, dict):
                text = meta.get("text") or meta.get("payload")
                if text:
                    lines.append(str(text))
                else:
                    lines.append(json.dumps(meta))
            else:
                lines.append(str(meta))
        return "\n".join(lines)

    def _register_routes(self) -> None:
        @self.app.middleware("http")
        async def _hook_middleware(request, call_next):
            for hook in self.pre_hooks:
                try:
                    maybe = hook(request)
                    if inspect.isawaitable(maybe):
                        await maybe
                except Exception:
                    logger.exception("Pre-hook failed")

            response = await call_next(request)

            for hook in self.post_hooks:
                try:
                    maybe = hook(response)
                    if inspect.isawaitable(maybe):
                        await maybe
                except Exception:
                    logger.exception("Post-hook failed")
            return response

        @self.app.get("/")
        def root() -> Dict[str, str]:
            return {"message": "ML Model Serving API is running"}

        @self.app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok"}

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(input_data: PredictionInput) -> Dict[str, Any]:
            try:
                data = input_data.data
                model_name, model = self._route_model({"endpoint": "/predict"})
                self.metrics.observe_batch_size("/predict", self._sample_count(data))

                with self.metrics.track_inference("/predict", model_name):
                    with self.model_lock:
                        result = self._predict_with_model(model, data)
                result = self._normalize_for_json(result)
                return {"prediction": result}
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        @self.app.post("/infer")
        async def infer(inputs: Union[List[float], List[List[float]]]) -> Dict[str, Any]:
            try:
                if len(inputs) == 0:
                    return {"results": []}

                if isinstance(inputs[0], list):
                    # Request provided a local batch; route each sample through the shared dynamic batcher.
                    results = [await self.infer_batcher.submit(sample) for sample in inputs]  # type: ignore[index]
                else:
                    result = await self.infer_batcher.submit(inputs)
                    results = [result]
                return {"results": self._normalize_for_json(results)}
            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        @self.app.post("/predict_multimodal")
        async def predict_multimodal(
            payload: str = Form(..., description="JSON string with at least {'text': '...'}"),
            image: Optional[UploadFile] = File(None),
        ) -> Dict[str, Any]:
            try:
                request_payload = json.loads(payload or "{}")
                if not isinstance(request_payload, dict):
                    raise ValueError("payload must decode to a JSON object")
                text = str(request_payload.get("text", "")).strip()
                if not text:
                    raise ValueError("payload.text is required")

                pil_image = None
                if image is not None:
                    pil_image = Image.open(BytesIO(await image.read())).convert("RGB")

                bundle = self._get_multimodal_bundle()
                retrieval = self._get_retrieval_pipeline()
                retrieved_items: List[Dict[str, Any]] = []
                retrieved_context = ""

                if retrieval is not None:
                    query = pil_image if pil_image is not None else text
                    retrieved_items = retrieval.retrieve(
                        query=query,
                        top_k=int(self.config.get("retrieval_top_k", 3)),
                    )
                    retrieved_context = self._retrieved_context(retrieved_items)

                with self.metrics.track_inference("/predict_multimodal", str(bundle.model_type)):
                    result = run_multimodal_inference(
                        bundle=bundle,
                        text=text,
                        image=pil_image,
                        retrieved_context=retrieved_context,
                        max_new_tokens=int(request_payload.get("max_new_tokens", 128)),
                    )

                if retrieval is not None:
                    retrieval.add(text, metadata={"text": text, "type": "text"})
                    if pil_image is not None:
                        retrieval.add(pil_image, metadata={"type": "image", "text": text})
                    retrieval.save()

                return {"prediction": result, "retrieved_context": retrieved_items}
            except Exception as e:
                logger.error(f"Multimodal prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Multimodal prediction error: {str(e)}")

        @self.app.post("/predict_video")
        async def predict_video(request: VideoPredictionRequest) -> StreamingResponse:
            try:
                frames = [self._decode_frame(frame_b64) for frame_b64 in request.frames]
                model_name, model = self._route_model({"endpoint": "/predict_video"})
                self.metrics.observe_batch_size("/predict_video", len(frames))

                with self.metrics.track_inference("/predict_video", model_name):
                    with self.model_lock:
                        results = infer_sequence(model, frames, self.config)

                async def stream():
                    for row in results:
                        yield (json.dumps(row) + "\n").encode("utf-8")

                media_type = self.config.get("video_stream_media_type", "application/jsonl")
                return StreamingResponse(stream(), media_type=media_type)
            except Exception as e:
                logger.error(f"Video prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Video prediction error: {str(e)}")

        @self.app.get("/metrics")
        async def metrics() -> Response:
            payload = self.metrics.render_prometheus()
            return Response(content=payload["content"], media_type=payload["content_type"])

        @self.app.post("/benchmark")
        async def benchmark(input_data: PredictionInput) -> Dict[str, Any]:
            try:
                warmup_calls = int(self.config.get("benchmark_warmup_calls", 20))
                timed_calls = int(self.config.get("benchmark_timed_calls", 100))
                data = input_data.data
                model_name, model = self._route_model({"endpoint": "/benchmark"})
                self.metrics.observe_batch_size("/benchmark", self._sample_count(data))

                with self.model_lock:
                    for _ in range(warmup_calls):
                        self._predict_with_model(model, data)

                    latencies: List[float] = []
                    for _ in range(timed_calls):
                        t0 = time.perf_counter()
                        with self.metrics.track_inference("/benchmark", model_name):
                            self._predict_with_model(model, data)
                        latencies.append(time.perf_counter() - t0)

                if not latencies:
                    raise ValueError("No latency samples recorded")

                p50 = float(np.percentile(latencies, 50))
                p95 = float(np.percentile(latencies, 95))
                p99 = float(np.percentile(latencies, 99))
                throughput = float(timed_calls / max(sum(latencies), 1e-9))

                return {
                    "model": model_name,
                    "warmup_calls": warmup_calls,
                    "timed_calls": timed_calls,
                    "latency_seconds": {"p50": p50, "p95": p95, "p99": p99},
                    "throughput_rps": throughput,
                }
            except Exception as e:
                logger.error(f"Benchmark error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")

        @self.app.post("/upload_model")
        async def upload_model(file: UploadFile) -> Dict[str, str]:
            try:
                with open(self.config["model_path"], "wb") as f:
                    f.write(await file.read())

                with self.model_lock:
                    self.model = load_model(
                        self.config["model_path"],
                        self.config.get("model_format"),
                        quantization=self.config.get("quantization"),
                        adapter_path=self.config.get("adapter_path"),
                        merge_adapter=bool(self.config.get("merge_adapter", False)),
                    )
                    self.model_registry.add_model("default", self.model)
                logger.info("Model reloaded successfully")
                return {"message": "Model reloaded successfully"}
            except Exception as e:
                logger.error(f"Model upload error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Model upload error: {str(e)}")

        @self.app.post("/finetune")
        async def finetune(request: FineTuneRequest) -> Dict[str, str]:
            job_id = str(uuid.uuid4())
            self.jobs[job_id] = {"status": "queued"}

            worker = threading.Thread(
                target=self._run_finetune_job,
                args=(job_id, request.dataset, request.config),
                daemon=True,
            )
            worker.start()
            return {"job_id": job_id, "status": "queued"}

        @self.app.get("/finetune/{job_id}")
        async def finetune_status(job_id: str) -> Dict[str, Any]:
            if job_id not in self.jobs:
                raise HTTPException(status_code=404, detail="Job ID not found")
            return self.jobs[job_id]

        @self.app.post("/load_adapter")
        async def load_adapter_endpoint(
            adapter_path: Optional[str] = Form(None),
            merge: bool = Form(False),
            file: Optional[UploadFile] = File(None),
        ) -> Dict[str, Any]:
            try:
                resolved_path = adapter_path
                if file is not None:
                    save_root = self.config.get("adapter_upload_dir", "./uploaded_adapters")
                    os.makedirs(save_root, exist_ok=True)
                    save_path = os.path.join(save_root, file.filename)
                    with open(save_path, "wb") as f:
                        f.write(await file.read())
                    resolved_path = save_path

                if not resolved_path:
                    raise ValueError("Provide either adapter_path or file upload")

                with self.model_lock:
                    self.model = load_adapter(self.model, resolved_path)
                    if merge:
                        self.model = merge_lora(self.model)
                    self.model_registry.add_model("default", self.model)

                self.config["adapter_path"] = resolved_path
                self.config["merge_adapter"] = merge
                return {"message": "Adapter loaded successfully", "adapter_path": resolved_path}
            except Exception as e:
                logger.error(f"Adapter load error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Adapter load error: {str(e)}")

    def run(self) -> None:
        logger.info(f"Starting API server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)
