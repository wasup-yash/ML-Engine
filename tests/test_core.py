import asyncio
import base64
import json
import tempfile
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api_server import APIServer
from src.config_manager import DEFAULT_CONFIG
from src.dynamic_batcher import AsyncBatcher
from src.model_loader import validate_model
from src.model_registry import ModelRegistry
from src.retrieval_pipeline import RetrievalPipeline


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


if __name__ == "__main__":
    unittest.main()
