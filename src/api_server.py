import importlib.util
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.adapter_loader import load_adapter, merge_lora
from src.config_manager import get_config
from src.finetune_engine import run_lora_finetune
from src.inference_engine import batch_inference
from src.logger import get_logger
from src.model_loader import load_model

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
        self.app = FastAPI(
            title="ML Model Serving API",
            description="API for serving machine learning model predictions",
            version="0.2.0",
        )

        self.plugins: List[Any] = []
        self._register_routes()
        self._load_plugins()

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

            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["adapter_path"] = result["adapter_path"]
        except Exception as e:
            logger.exception("Fine-tune job failed")
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)

    def _register_routes(self) -> None:
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
                with self.model_lock:
                    if isinstance(data, dict):
                        result = self.model.predict(data)
                    else:
                        data_array = np.array(data)
                        if data_array.ndim == 1:
                            data_array = data_array.reshape(1, -1)
                        result = self.model.predict(data_array)

                if isinstance(result, np.ndarray):
                    result = result.tolist()
                return {"prediction": result}
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        @self.app.post("/infer")
        async def infer(inputs: List[float]) -> Dict[str, Any]:
            try:
                batch_size = int(self.config.get("batch_size", 32))
                with self.model_lock:
                    results = batch_inference(self.model, inputs, batch_size)
                return {"results": results}
            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

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

                self.config["adapter_path"] = resolved_path
                self.config["merge_adapter"] = merge
                return {"message": "Adapter loaded successfully", "adapter_path": resolved_path}
            except Exception as e:
                logger.error(f"Adapter load error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Adapter load error: {str(e)}")

    def run(self) -> None:
        logger.info(f"Starting API server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)

    def register_hook(self, hook_type: str, func):
        if not hasattr(self, '_hooks'):
            self._hooks = {"pre": [], "post": []}
        self._hooks[hook_type].append(func)