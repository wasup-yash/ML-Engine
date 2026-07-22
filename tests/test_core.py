import asyncio
import base64
import importlib
import json
import os
import hashlib
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api_server import APIServer
from src.config_manager import DEFAULT_CONFIG, validate_config
from src.dynamic_batcher import AsyncBatcher
from src.model_loader import load_model, validate_model
from src.model_registry import ModelRegistry
from src.retrieval_pipeline import RetrievalPipeline
from src.security import validate_safetensors_file
from src.state_store import SQLiteJobStore
from src.storage import LocalArtifactStorage


class RecordingModel:
    def __init__(self):
        self.batches = []

    def predict(self, inputs):
        values = np.asarray(inputs)
        self.batches.append(values.copy())
        if values.ndim >= 3:
            return np.asarray([float(values.mean())])
        return values.sum(axis=1)


class CoreTests(unittest.TestCase):
    def test_yaml_model_format_validation_matches_cli_choices(self):
        with self.assertRaisesRegex(ValueError, "Unsupported model_format"):
            validate_config({"model_format": "not-a-real-format"})
        validate_config({"model_format": "joblib"})

    def test_model_zoo_has_no_import_time_download(self):
        module = importlib.import_module("src.model_zoo")
        self.assertTrue(hasattr(module, "ModelZoo"))

    def test_phase_zero_artifacts_are_not_present(self):
        root = os.path.dirname(os.path.dirname(__file__))
        self.assertFalse(os.path.exists(os.path.join(root, "model.joblib")))
        self.assertFalse(os.path.exists(os.path.join(root, "tmp_test_write.txt")))

    def test_dockerfile_starts_server_and_has_healthcheck(self):
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "DOCKERFILE"), encoding="utf-8") as file:
            dockerfile = file.read()
        self.assertIn('CMD ["python", "run_engine.py"', dockerfile)
        self.assertIn("HEALTHCHECK", dockerfile)
        self.assertIn("USER mlengine", dockerfile)

    def test_model_validation_does_not_run_arbitrary_prediction(self):
        class Model:
            def predict(self, _):
                raise AssertionError("validation must not run prediction")

        validate_model(Model())

    def test_async_batcher_coalesces_concurrent_submissions(self):
        async def scenario():
            observed = []

            def infer(items):
                observed.append(list(items))
                return [item * 2 for item in items]

            batcher = AsyncBatcher(infer, max_batch_size=8, max_wait_ms=50)
            try:
                results = await asyncio.gather(batcher.submit(1), batcher.submit(2))
                self.assertEqual(results, [2, 4])
                self.assertEqual(observed, [[1, 2]])
            finally:
                await batcher.close()

        asyncio.run(scenario())

    def test_registry_renormalizes_split_for_available_models(self):
        model = object()
        registry = ModelRegistry(
            models={"v1": model},
            traffic_split={"missing": 0.9, "v1": 0.1},
        )
        with patch("src.model_registry.random.random", return_value=0.95):
            name, routed = registry.route()
        self.assertEqual(name, "v1")
        self.assertIs(routed, model)

    def test_retrieval_fallback_embeddings_are_stable_and_content_based(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/index.faiss"
            first = RetrievalPipeline(path, embedding_dim=16)
            second = RetrievalPipeline(path, embedding_dim=16)

            np.testing.assert_array_equal(first.embed("same"), second.embed("same"))

            image_a = Image.new("RGB", (4, 4), "black")
            image_b = Image.new("RGB", (4, 4), "white")
            self.assertFalse(np.array_equal(first.embed(image_a), first.embed(image_b)))

    def test_api_batches_local_infer_request_and_streams_video(self):
        model = RecordingModel()
        config = DEFAULT_CONFIG.copy()
        config.update(
            {
                "auth_required": False,
                "plugin_dir": "missing-test-plugins",
                "batch_max_wait_ms": 5,
                "benchmark_warmup_calls": 1,
                "benchmark_timed_calls": 2,
            }
        )
        server = APIServer(model, config=config)

        image = Image.new("RGB", (2, 2), "white")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        with TestClient(server.app) as client:
            response = client.post("/infer", json=[[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"results": [3.0, 7.0]})
            self.assertEqual(model.batches[-1].shape[0], 2)

            response = client.post("/predict_video", json={"frames": [encoded, encoded]})
            self.assertEqual(response.status_code, 200)
            rows = [json.loads(line) for line in response.text.strip().splitlines()]
            self.assertEqual([row["frame_index"] for row in rows], [0, 1])

    def test_empty_model_returns_service_unavailable(self):
        config = DEFAULT_CONFIG.copy()
        config.update({"auth_required": False, "plugin_dir": "missing-test-plugins"})
        with TestClient(APIServer(None, config=config).app) as client:
            self.assertEqual(client.post("/predict", json={"data": [1.0]}).status_code, 503)

    def test_authentication_scopes_and_health_exemption(self):
        api_key = "predict-secret"
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        config = DEFAULT_CONFIG.copy()
        config.update(
            {
                "plugin_dir": "missing-test-plugins",
                "auth_required": True,
                "api_key_hashes_env": "TEST_API_KEY_HASHES",
            }
        )
        old_value = os.environ.get("TEST_API_KEY_HASHES")
        os.environ["TEST_API_KEY_HASHES"] = json.dumps(
            {key_hash: {"key_id": "predictor", "tenant_id": "tenant-a", "scopes": ["predict"]}}
        )
        try:
            with TestClient(APIServer(RecordingModel(), config=config).app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.post("/predict", json={"data": [1.0, 2.0]}).status_code, 401)
                self.assertEqual(
                    client.post(
                        "/predict",
                        json={"data": [1.0, 2.0]},
                        headers={"X-API-Key": api_key},
                    ).status_code,
                    200,
                )
                self.assertEqual(
                    client.post(
                        "/finetune",
                        json={"dataset": [{"text": "a", "label": "b"}]},
                        headers={"X-API-Key": api_key},
                    ).status_code,
                    403,
                )
        finally:
            if old_value is None:
                os.environ.pop("TEST_API_KEY_HASHES", None)
            else:
                os.environ["TEST_API_KEY_HASHES"] = old_value

    def test_safetensors_header_validation(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as file:
            path = file.name
            header = json.dumps(
                {"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
            ).encode("utf-8")
            file.write(len(header).to_bytes(8, "little"))
            file.write(header)
            file.write(b"\x00\x00\x00\x00")
        try:
            validate_safetensors_file(path)
        finally:
            os.unlink(path)

    def test_unsafe_deserialization_requires_explicit_trusted_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.joblib")
            import joblib

            joblib.dump(RecordingModel(), path)
            with self.assertRaises(PermissionError):
                load_model(path, "joblib")
            model = load_model(
                path,
                "joblib",
                allow_unsafe_deserialization=True,
                trusted_model_paths=[directory],
            )
            self.assertIsInstance(model, RecordingModel)

    def test_rate_limit_rejects_second_authenticated_request_with_security_headers(self):
        api_key = "rate-limit-secret"
        key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        config = DEFAULT_CONFIG.copy()
        config.update(
            {
                "plugin_dir": "missing-test-plugins",
                "auth_required": True,
                "api_key_hashes_env": "TEST_RATE_LIMIT_KEYS",
                "rate_limit_requests_per_minute": 1,
            }
        )
        old_value = os.environ.get("TEST_RATE_LIMIT_KEYS")
        os.environ["TEST_RATE_LIMIT_KEYS"] = json.dumps(
            {key_hash: {"key_id": "limited", "tenant_id": "tenant-a", "scopes": ["predict"]}}
        )
        try:
            with TestClient(APIServer(RecordingModel(), config=config).app) as client:
                headers = {"X-API-Key": api_key, "X-Request-ID": "rate-limit-test"}
                self.assertEqual(client.post("/predict", json={"data": [1.0]}, headers=headers).status_code, 200)
                response = client.post("/predict", json={"data": [1.0]}, headers=headers)
                self.assertEqual(response.status_code, 429)
                self.assertEqual(response.headers["X-Request-ID"], "rate-limit-test")
                self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        finally:
            if old_value is None:
                os.environ.pop("TEST_RATE_LIMIT_KEYS", None)
            else:
                os.environ["TEST_RATE_LIMIT_KEYS"] = old_value

    def test_sqlite_job_state_survives_store_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "jobs.sqlite3")
            first = SQLiteJobStore(path)
            first.create("job-1", "tenant-a", {"dataset_size": 2})
            first.update("job-1", "running")
            first.close()

            second = SQLiteJobStore(path)
            job = second.get("job-1", "tenant-a")
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["payload"], {"dataset_size": 2})
            second.close()

    def test_local_storage_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = os.path.join(directory, "artifact.onnx")
            with open(artifact, "wb") as file:
                file.write(b"test")
            storage = LocalArtifactStorage(directory)
            self.assertEqual(storage.materialize("artifact.onnx"), artifact)
            with self.assertRaises(PermissionError):
                storage.materialize("../outside.onnx")


if __name__ == "__main__":
    unittest.main()
