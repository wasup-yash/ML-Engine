import argparse
import os
from typing import Any, Dict

import yaml

from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = {
    "model_path": "./model.joblib",
    "model_format": None,
    "host": "127.0.0.1",
    "port": 5000,
    "log_level": "INFO",
    "batch_size": 32,
    "quantization": None,
    "adapter_path": None,
    "merge_adapter": False,
    "batch_max_wait_ms": 20,
    "batch_max_size": 32,
    "multimodal_model_path": None,
    "model_card_path": None,
    "retrieval_index_path": None,
    "retrieval_top_k": 3,
    "retrieval_embedding_dim": 512,
    "video_stream_media_type": "application/jsonl",
    "benchmark_warmup_calls": 20,
    "benchmark_timed_calls": 100,
    "metrics_gpu_poll_interval_sec": 1.0,
    "model_variants": {},
    "traffic_split": {},
    "shadow_model": None,
}


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    logger.info("Configuration loaded successfully")
    return config


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="ML Model Serving Engine")

    parser.add_argument("--config", type=str, help="Path to configuration file (YAML)")
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument(
        "--model_format",
        type=str,
        choices=[
            "joblib",
            "pickle",
            "onnx",
            "torch",
            "tensorflow",
            "diffusers",
            "llava",
            "openvla",
            "clip",
            "vision2seq",
            "multimodal",
            "auto",
        ],
        help="Format of the model file (if not auto-detected)",
    )
    parser.add_argument("--host", type=str, help="Host address for the API server")
    parser.add_argument("--port", type=int, help="Port for the API server")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--batch_size", type=int, help="Legacy static batch size for /infer")
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["int8", "int4"],
        help="Quantization mode to apply at model load time",
    )
    parser.add_argument("--adapter_path", type=str, help="Path to LoRA adapter directory/file")
    parser.add_argument(
        "--merge_adapter",
        action="store_true",
        help="Merge loaded LoRA adapter into the base model and unload adapter modules",
    )
    parser.add_argument(
        "--batch_max_wait_ms",
        type=int,
        help="Dynamic batching queue wait budget in milliseconds",
    )
    parser.add_argument(
        "--batch_max_size",
        type=int,
        help="Dynamic batching max batch size",
    )
    parser.add_argument("--multimodal_model_path", type=str, help="Path or HF id for multimodal model")
    parser.add_argument("--model_card_path", type=str, help="Path to model card file for type detection")
    parser.add_argument("--retrieval_index_path", type=str, help="Path to FAISS retrieval index")
    parser.add_argument("--retrieval_top_k", type=int, help="Top-k retrieval results to inject as context")
    parser.add_argument("--retrieval_embedding_dim", type=int, help="Embedding dimension for new FAISS index")
    parser.add_argument("--video_stream_media_type", type=str, help="Media type for /predict_video stream")
    parser.add_argument("--benchmark_warmup_calls", type=int, help="Warmup call count for /benchmark")
    parser.add_argument("--benchmark_timed_calls", type=int, help="Timed call count for /benchmark")
    parser.add_argument(
        "--metrics_gpu_poll_interval_sec",
        type=float,
        help="GPU metric polling interval in seconds",
    )
    parser.add_argument(
        "--model_variants",
        type=str,
        help='JSON object mapping model slot to {model_path, model_format, quantization, adapter_path, merge_adapter}',
    )
    parser.add_argument(
        "--traffic_split",
        type=str,
        help='JSON object mapping model slot to traffic weight, e.g. {"v1":0.9,"v2":0.1}',
    )
    parser.add_argument("--shadow_model", type=str, help="Shadow model slot name")

    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    # Optional JSON-encoded dict overrides for CLI usage.
    for dict_key in ("model_variants", "traffic_split"):
        raw = args_dict.get(dict_key)
        if isinstance(raw, str):
            import json

            args_dict[dict_key] = json.loads(raw)
    return args_dict


def get_config() -> Dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    args = parse_args()

    if "config" in args:
        config_path = args["config"]
        file_config = load_config_from_yaml(config_path)
        config.update(file_config)
        args = {k: v for k, v in args.items() if k != "config"}

    config.update(args)
    logger.info(f"Final configuration: {config}")
    return config
