import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoModelForVision2Seq, AutoProcessor, CLIPModel

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MultimodalBundle:
    model_type: str
    model_path: str
    processor: Any
    model: Any


SUPPORTED_MODEL_TYPES = {
    "llava",
    "openvla",
    "clip",
    "vision2seq",
    "multimodal",
}


def _read_model_card(model_path: str, card_path: Optional[str]) -> str:
    candidates = []
    if card_path:
        candidates.append(card_path)
    if os.path.isdir(model_path):
        candidates.extend(
            [
                os.path.join(model_path, "README.md"),
                os.path.join(model_path, "modelcard.json"),
                os.path.join(model_path, "config.json"),
            ]
        )

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().lower()
        except Exception:
            continue
    return ""


def detect_model_type(config: Dict[str, Any]) -> str:
    model_format = (config.get("model_format") or "").lower()
    if model_format in SUPPORTED_MODEL_TYPES:
        return model_format

    model_path = config.get("multimodal_model_path") or config.get("model_path", "")
    model_card_text = _read_model_card(model_path, config.get("model_card_path"))

    if "openvla" in model_card_text:
        return "openvla"
    if "llava" in model_card_text:
        return "llava"
    if "clip" in model_card_text:
        return "clip"
    if "vision2seq" in model_card_text:
        return "vision2seq"
    if "multimodal" in model_card_text:
        return "multimodal"

    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                model_type = str(payload.get("model_type", "")).lower()
                if "llava" in model_type:
                    return "llava"
                if "clip" in model_type:
                    return "clip"
                if "vla" in model_type:
                    return "openvla"
            except Exception:
                pass

    return "multimodal"


def load_multimodal_model(config: Dict[str, Any]) -> MultimodalBundle:
    model_path = config.get("multimodal_model_path") or config.get("model_path")
    if not model_path:
        raise ValueError("multimodal_model_path or model_path must be set")

    model_type = detect_model_type(config)
    logger.info(f"Loading multimodal model {model_path} as type={model_type}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if model_type == "clip":
        model = CLIPModel.from_pretrained(model_path, trust_remote_code=True)
    elif model_type in {"llava", "openvla", "vision2seq", "multimodal"}:
        try:
            model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return MultimodalBundle(model_type=model_type, model_path=model_path, processor=processor, model=model)


def _build_prompt(text: str, retrieved_context: str = "") -> str:
    if not retrieved_context:
        return text
    return f"Context:\n{retrieved_context}\n\nUser Query:\n{text}"


def run_multimodal_inference(
    bundle: MultimodalBundle,
    text: str,
    image: Optional[Image.Image] = None,
    retrieved_context: str = "",
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    prompt = _build_prompt(text, retrieved_context)

    if bundle.model_type == "clip":
        if image is None:
            raise ValueError("CLIP inference requires an image input")
        inputs = bundle.processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bundle.model(**inputs)
            similarity = outputs.logits_per_image.squeeze().float().cpu().item()
        return {"type": "clip_similarity", "score": similarity}

    processor_kwargs = {"text": prompt, "return_tensors": "pt"}
    if image is not None:
        processor_kwargs["images"] = image
    inputs = bundle.processor(**processor_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            inputs[key] = value.to(device)

    with torch.no_grad():
        if hasattr(bundle.model, "generate"):
            output_tokens = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens)
            decoded = bundle.processor.batch_decode(output_tokens, skip_special_tokens=True)
            prediction = decoded[0] if decoded else ""
        else:
            raw = bundle.model(**inputs)
            prediction = str(raw)

    return {"type": bundle.model_type, "prediction": prediction}
