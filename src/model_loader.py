import os
import pickle
from typing import Any, Optional

import joblib
import numpy as np
import onnxruntime
import tensorflow as tf
import torch

from src.logger import get_logger

logger = get_logger(__name__)


class ModelValidationError(Exception):
    pass


def _is_diffusers_directory(model_path: str) -> bool:
    return os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model_index.json"))


def validate_model(model: Any, input_sample: Optional[Any] = None) -> None:
    validation_errors = []

    if isinstance(model, torch.nn.Module):
        return

    if isinstance(model, onnxruntime.InferenceSession):
        return

    if callable(model):
        return

    if not hasattr(model, "predict"):
        validation_errors.append("Model missing required 'predict' method")

    if input_sample is not None and hasattr(model, "predict"):
        try:
            output = model.predict(input_sample)
            if not isinstance(output, (list, np.ndarray)):
                validation_errors.append("Model output must be list or numpy array")
        except Exception as e:
            validation_errors.append(f"Input validation failed: {str(e)}")

    if validation_errors:
        raise ModelValidationError("\n".join(validation_errors))


def _detect_format(model_path: str) -> str:
    if _is_diffusers_directory(model_path):
        return "diffusers"

    if os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "saved_model.pb")):
            return "tensorflow"
        return "auto"

    _, ext = os.path.splitext(model_path)
    ext = ext.lower()
    if ext in [".joblib", ".jlib"]:
        return "joblib"
    if ext in [".pkl", ".pickle"]:
        return "pickle"
    if ext == ".onnx":
        return "onnx"
    if ext in [".pt", ".pth"]:
        return "torch"
    if ext in [".pb", ".savedmodel"]:
        return "tensorflow"
    logger.warning(f"Could not auto-detect model format from extension: {ext}")
    return "auto"


def _load_base_model(model_path: str, format: str) -> Any:
    if format == "auto":
        if _is_diffusers_directory(model_path):
            format = "diffusers"
        else:
            logger.info("Attempting to load model with multiple formats")
            try:
                model = joblib.load(model_path)
                logger.info("Model loaded successfully with joblib")
                return model
            except Exception as je:
                logger.warning(f"Joblib loading failed: {str(je)}, trying pickle")
                try:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    logger.info("Model loaded successfully with pickle")
                    return model
                except Exception as pe:
                    logger.warning(f"Pickle loading failed: {str(pe)}, trying torch")
                    return torch.load(model_path, map_location="cpu")

    if format == "diffusers":
        from diffusers import DiffusionPipeline

        return DiffusionPipeline.from_pretrained(model_path)

    if format == "joblib":
        return joblib.load(model_path)
    if format == "pickle":
        with open(model_path, "rb") as f:
            return pickle.load(f)
    if format == "onnx":
        return onnxruntime.InferenceSession(model_path)
    if format == "torch":
        return torch.load(model_path, map_location="cpu")
    if format == "tensorflow":
        return tf.saved_model.load(model_path)
    raise ValueError(f"Unsupported model format: {format}")


def _apply_quantization(model: Any, quantization: Optional[str]) -> Any:
    if not quantization:
        return model

    quantization = quantization.lower()
    if quantization not in {"int8", "int4"}:
        raise ValueError("quantization must be one of: int8, int4, or null")

    if not isinstance(model, torch.nn.Module):
        logger.warning(
            "Quantization requested but loaded model is not a torch.nn.Module; skipping quantization"
        )
        return model

    if quantization == "int8":
        logger.info("Applying int8 dynamic quantization via torch.ao.quantization")
        try:
            return torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            logger.warning(f"Failed to apply int8 quantization; using non-quantized model: {e}")
            return model

    # int4 support is framework/model-specific; keep plumbing in place and attempt bitsandbytes-based prep
    logger.info("int4 quantization requested")
    try:
        import bitsandbytes as bnb  # noqa: F401

        logger.warning(
            "bitsandbytes is installed, but generic in-place int4 wrapping is model-architecture specific. "
            "Returning unmodified model. Use adapter-aware HF loading path for full int4 support."
        )
    except Exception:
        logger.warning("bitsandbytes not available; cannot apply int4 quantization")
    return model


def load_model(
    model_path: str,
    format: Optional[str] = None,
    quantization: Optional[str] = None,
    adapter_path: Optional[str] = None,
    merge_adapter: bool = False,
) -> Any:
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_format = (format or _detect_format(model_path)).lower()
    logger.info(f"Loading model from {model_path} using {model_format} format")

    try:
        model = _load_base_model(model_path, model_format)
        model = _apply_quantization(model, quantization)

        if adapter_path:
            from src.adapter_loader import load_adapter, merge_lora

            logger.info(f"Loading adapter from {adapter_path}")
            model = load_adapter(model, adapter_path)
            if merge_adapter:
                logger.info("Merging adapter weights into base model")
                model = merge_lora(model)

        validate_model(model, input_sample=np.random.rand(1, 4))
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
