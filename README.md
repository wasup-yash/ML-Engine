# ML Serving Engine

ML Serving Engine is a self-hostable inference service for classical, ONNX, PyTorch/Hugging Face, multimodal, video, LoRA-adapted, and Diffusers workloads. It provides one secured and observable API surface instead of a bespoke FastAPI wrapper per model team.

## What is shipped

- Explicit model-format loading with INT8/INT4 configuration, LoRA adapter load/merge, and Diffusers directory detection.
- `/predict`, dynamically batched `/infer`, multipart `/predict_multimodal`, streaming `/predict_video`, async `/finetune`, `/benchmark`, `/metrics`, and model-version routing/shadowing.
- Scoped API-key authentication, per-key rate limiting, CORS allowlisting, structured JSON logs, request IDs, OpenTelemetry hooks, audit events, and Prometheus metrics.
- Durable SQLite fine-tune job state, local artifact-storage boundary, GPU-memory admission guardrails, and non-root container execution.

See [deployment guidance](docs/DEPLOYMENT.md), the [security review](docs/SECURITY.md), [on-call runbook](docs/RUNBOOK.md), and [B2B overview](docs/WHY_BUY.md).
Measured results and reproduction instructions are in [the benchmark baseline](docs/BENCHMARKS.md).

## Quickstart

Requires Python 3.10-3.13. Install the pinned runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

For a local readiness check without a model or credentials:

```bash
ML_ENGINE_AUTH_REQUIRED=false python run_engine.py --config config/default_config.yaml --allow_empty_model
curl http://127.0.0.1:5000/health
```

The expected response includes `status`, `model_loaded`, and `job_queue_depth`. A no-model instance returns 503 from prediction routes by design.

For production, leave authentication enabled, inject a hashed API-key map through `ML_ENGINE_API_KEY_HASHES`, and use an explicit model format. Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <raw-api-key>' \
  -d '{"data":[5.1,3.5,1.4,0.2]}'
```

## Safety Defaults

`joblib`, `pickle`, `.pt`, and auto-detected serialized models can execute attacker-controlled code while loading. The service refuses them until both `allow_unsafe_deserialization: true` and an operator-owned `trusted_model_paths` entry are configured. Customer uploads support explicit ONNX only; adapters must pass safetensors-header validation.

Do not place API keys or cloud credentials in YAML. Environment variables override configuration as `ML_ENGINE_<CONFIG_KEY>`; use your secret manager to inject them. `trusted_remote_code_models` is an exact-path allowlist, never a global flag.

## Operations

`GET /health` is unauthenticated for container orchestration. Every other endpoint requires `predict` or `admin` scope by default. `GET /metrics` exposes Prometheus text; `/benchmark` runs configured warmup/timed loops and reports p50/p95/p99 plus throughput for the selected model.

Dynamic batching is local to each process. For GPU models, use one worker per GPU and measure before increasing concurrency. SQLite job state and the in-memory rate limiter are single-instance defaults; use external shared implementations before replica-based deployment.

## Development

```bash
python -m pip install -r requirements-dev.txt
ruff check src tests run_engine.py
mypy src run_engine.py
coverage run -m unittest discover -s tests -v
coverage report
```

The current suite validates security controls, durable job state, dynamic batching, retrieval fallback behavior, API authentication/rate limiting, and video streaming. The repository uses semantic versioning; changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Container

```bash
docker build -f DOCKERFILE -t ml-serving-engine:0.5.0 .
docker run --rm -p 5000:5000 -e ML_ENGINE_API_KEY_HASHES="$ML_ENGINE_API_KEY_HASHES" ml-serving-engine:0.5.0
```

The container is intentionally not configured with a development key. It will fail closed if authentication is enabled and the key-hash environment variable is missing.
