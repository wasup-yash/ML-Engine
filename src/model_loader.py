import os
import pickle
from typing import Any, Optional

import joblib
import numpy as np

from src.logger import get_logger
from src.security import is_trusted_path

logger = get_logger(__name__)


class ModelValidationError(Exception):
    pass


UNSAFE_DESERIALIZATION_FORMATS = frozenset({"joblib", "pickle", "torch", "auto"})


def _is_diffusers_directory(model_path: str) -> bool:
    return os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model_index.json"))


def _is_huggingface_directory(model_path: str) -> bool:
    return os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json"))


def validate_model(model: Any, input_sample: Optional[Any] = None) -> None:
    validation_errors = []

    if callable(model) or hasattr(model, "run"):
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
        if _is_huggingface_directory(model_path):
            return "huggingface"
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
    logger.warning(f"Could not auto-detect model format from extension: {ext}")
    return "auto"


def _load_huggingface_model(model_path: str, quantization: Optional[str]) -> Any:
    from transformers import AutoModel, AutoModelForCausalLM

    kwargs = {"local_files_only": True}
    if quantization:
        from transformers import BitsAndBytesConfig

        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quantization == "int8",
            load_in_4bit=quantization == "int4",
        )

    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except (ValueError, TypeError):
        return AutoModel.from_pretrained(model_path, **kwargs)


def _load_base_model(
    model_path: str, format: str, quantization: Optional[str] = None
) -> Any:
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
                    import torch

                    return torch.load(model_path, map_location="cpu")

    if format == "diffusers":
        from diffusers import DiffusionPipeline

        return DiffusionPipeline.from_pretrained(model_path)

    if format == "huggingface":
        return _load_huggingface_model(model_path, quantization)

    if format == "joblib":
        return joblib.load(model_path)
    if format == "pickle":
        with open(model_path, "rb") as f:
            return pickle.load(f)
    if format == "onnx":
        import onnxruntime

        available = onnxruntime.get_available_providers()
        preferred = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available
        ]
        return onnxruntime.InferenceSession(model_path, providers=preferred or available)
    if format == "torch":
        import torch

        return torch.load(model_path, map_location="cpu")
    raise ValueError(f"Unsupported model format: {format}")


def _apply_quantization(model: Any, quantization: Optional[str]) -> Any:
    if not quantization:
        return model

    quantization = quantization.lower()
    if quantization not in {"int8", "int4"}:
        raise ValueError("quantization must be one of: int8, int4, or null")

    import torch

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

    raise ValueError(
        "Generic torch INT4 conversion is unsupported. Load a Hugging Face directory with "
        "model_format=huggingface so bitsandbytes quantization is applied during from_pretrained()."
    )


def load_model(
    model_path: str,
    format: Optional[str] = None,
    quantization: Optional[str] = None,
    adapter_path: Optional[str] = None,
    merge_adapter: bool = False,
    trusted_model_paths: Optional[list[str]] = None,
    allow_unsafe_deserialization: bool = False,
    storage: Any = None,
) -> Any:
    if storage is not None:
        model_path = storage.materialize(model_path)
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_format = (format or _detect_format(model_path)).lower()
    if model_format in UNSAFE_DESERIALIZATION_FORMATS:
        if not allow_unsafe_deserialization:
            raise PermissionError(
                f"Refusing unsafe {model_format} deserialization. Configure an explicit trusted "
                "model path and set allow_unsafe_deserialization only for artifacts you control."
            )
        if not is_trusted_path(model_path, trusted_model_paths or []):
            raise PermissionError(
                f"Model path is not in trusted_model_paths: {model_path}"
            )
    logger.info(f"Loading model from {model_path} using {model_format} format")

    try:
        model = _load_base_model(model_path, model_format, quantization)
        if model_format != "huggingface":
            model = _apply_quantization(model, quantization)

        if adapter_path:
            from src.adapter_loader import load_adapter, merge_lora

            logger.info(f"Loading adapter from {adapter_path}")
            model = load_adapter(model, adapter_path)
            if merge_adapter:
                logger.info("Merging adapter weights into base model")
                model = merge_lora(model)

        validate_model(model)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
