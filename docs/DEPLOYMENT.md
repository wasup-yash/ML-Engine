# Deployment Guide

## Configuration

The server reads YAML, CLI values, then `ML_ENGINE_<CONFIG_KEY>` environment overrides. Do not enable `allow_unsafe_deserialization` unless `trusted_model_paths` contains only operator-owned artifacts.

Create one API-key hash without storing the raw key in configuration:

```bash
python -c "import hashlib; print(hashlib.sha256(b'replace-this-key').hexdigest())"
export ML_ENGINE_API_KEY_HASHES='{"<hash>":{"key_id":"customer-a","tenant_id":"customer-a","scopes":["predict","admin"]}}'
```

Run an empty-model readiness instance for platform validation:

```bash
ML_ENGINE_AUTH_REQUIRED=false python run_engine.py --config config/default_config.yaml --allow_empty_model
curl http://127.0.0.1:5000/health
```

Production instances should leave authentication enabled and load an explicit, trusted model format. `joblib`, `pickle`, and `.pt` artifacts require both the unsafe switch and trusted root.

## Container

```bash
docker build -f DOCKERFILE -t ml-serving-engine:0.5.0 .
docker run --rm -p 5000:5000 -e ML_ENGINE_API_KEY_HASHES="$ML_ENGINE_API_KEY_HASHES" -v /srv/ml-engine/models:/models:ro -e ML_ENGINE_MODEL_PATH=/models/model.onnx -e ML_ENGINE_MODEL_FORMAT=onnx ml-serving-engine:0.5.0
```

The image uses a non-root account and an HTTP `/health` health check. Put it behind an HTTPS reverse proxy/load balancer; do not expose port 5000 directly to the public internet.

## Scaling

Use multiple Uvicorn/Gunicorn workers only for CPU-safe model formats and only after measuring memory duplication. GPU models normally use one worker per GPU. SQLite and the in-process rate limiter are for a single instance; replicas need a shared job/rate-limit store. Dynamic batching is process-local, so requests must be consistently routed to the same replica when batch efficiency matters.
