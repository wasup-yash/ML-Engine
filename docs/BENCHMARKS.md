# Benchmark Baseline

## Method

Run `POST /benchmark` with 20 warmup calls and 100 timed calls. The endpoint measures model execution with `time.perf_counter()` while holding the model lock; it does not include HTTP/network serialization or queueing delay.

## Recorded baseline

On 2026-07-21, the local Windows Python 3.10 environment measured an in-process NumPy sum model with one four-float sample:

| Format/workload | p50 | p95 | p99 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| In-process NumPy `predict` baseline | 3.00 us | 3.10 us | 3.40 us | 332,779 RPS |

This is a harness sanity check, not a customer capacity claim. No ONNX, torch, Hugging Face, Diffusers, GPU, or Docker benchmark is published because no representative artifact/GPU/Docker daemon was available in this environment.

## Reproduce per model

Use an authenticated request with a representative payload and record the exact model version, hardware, batch configuration, warmup/timed counts, and concurrency:

```bash
curl -X POST http://localhost:5000/benchmark \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <raw-api-key>' \
  -d '{"data":[[1.0,2.0,3.0,4.0]]}'
```

For dynamic batching, additionally run a concurrent load test against `/infer` and compare observed `batch_size_observed` plus end-to-end client latency. Tune `batch_max_wait_ms` and `batch_max_size` only from those measurements.
