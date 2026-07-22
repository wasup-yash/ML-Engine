# On-Call Runbook

## Alert signals

- Rising `inference_latency_seconds` p95/p99 or `batch_size_observed` falling to 1 under load.
- `gpu_memory_used_bytes` near device capacity or 409 responses from GPU guardrails.
- `/health` reports no model loaded or a growing `job_queue_depth`.

## First three steps

1. Query `/health`, `/metrics`, and the last structured logs for the request/tenant ID.
2. Check GPU memory/processes and confirm the intended model/version is loaded.
3. Compare `/benchmark` with the last known measurement using the same payload; drain or roll back a variant if latency regresses.

## Common actions

- A 401/403 means missing key or scope. Validate the hash mapping injected from the secret manager, never log the raw key.
- A 409 during load/fine-tune means the configured projection plus reserve exceeds free GPU memory. Stop/defer the job or adjust a measured projection.
- Fine-tune state persists in the configured SQLite file. Back it up before host maintenance; use a shared durable backend before horizontal scaling.
