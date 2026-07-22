# Security Review

## Controls implemented

- Every route except `/health` requires an API key by default. API key records are SHA-256 hashes from the environment variable named by `api_key_hashes_env`; keys have `predict` or `admin` scopes and a tenant identifier.
- Requests have an in-process fixed-window rate limit per key, correlation IDs, JSON logs, CORS allowlisting, and `nosniff`, frame-deny, referrer, and no-store response headers.
- Unsafe `joblib`, `pickle`, `torch`, and `auto` deserialization is denied unless the operator enables it and the artifact resolves under `trusted_model_paths`. Treat unsafe deserialization as a CVE-class remote-code-execution risk.
- Untrusted model upload accepts explicit ONNX only. Adapter uploads require flat `.safetensors` names, bounded streaming, and safetensors-header validation.
- `trust_remote_code` is denied globally. It can be enabled only for an exact entry in `trusted_remote_code_models`, producing an audit log entry.

## Residual risk and mitigation

- Pickle-family loading remains available for self-host compatibility. Keep it disabled for customer uploads, restrict the trusted root to read-only operator-owned storage, and verify artifact provenance/signatures before enabling it.
- SQLite and fixed-window limiting are single-instance defaults. Use a shared Redis/Postgres-backed implementation and gateway rate limiting for replicas.
- The bundled S3/GCS storage interfaces are intentional seams, not implementations. Use workload identity and a deployment-specific adapter; never put cloud credentials in YAML.
- TLS terminates at the ingress/reverse proxy. Enforce HTTPS, HSTS, request-size limits, and network policy there.

## Secrets

Do not put API keys or cloud credentials in YAML. Set `ML_ENGINE_API_KEY_HASHES` from your secret manager. The reference deployment integration is AWS Secrets Manager injected as an ECS/Kubernetes environment variable. Rotate the secret, restart instances, and revoke the retired hash at the gateway/store.
