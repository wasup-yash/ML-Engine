import os
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


def _normalize_adapter_path(adapter_path: str) -> str:
    resolved = os.path.abspath(adapter_path)
    if os.path.isdir(resolved):
        return resolved

    if os.path.isfile(resolved) and resolved.endswith(".safetensors"):
        parent = os.path.dirname(resolved)
        config_file = os.path.join(parent, "adapter_config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                "adapter_config.json not found next to .safetensors file. "
                "Provide a PEFT adapter directory or include adapter_config.json."
            )
        return parent

    raise FileNotFoundError(f"Adapter path does not exist or is unsupported: {adapter_path}")


def load_adapter(base_model: Any, adapter_path: str) -> Any:
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("peft is required for adapter loading") from e

    normalized_path = _normalize_adapter_path(adapter_path)
    logger.info(f"Loading PEFT adapter from {normalized_path}")
    return PeftModel.from_pretrained(base_model, normalized_path, local_files_only=True)


def merge_lora(model: Any) -> Any:
    if not hasattr(model, "merge_and_unload"):
        logger.warning("Model does not expose merge_and_unload(); skipping LoRA merge")
        return model
    return model.merge_and_unload()
