from typing import Any, Dict, Iterable, Iterator, List

import numpy as np
import onnxruntime


def _infer_single_frame(model: Any, frame: np.ndarray) -> Any:
    if hasattr(model, "predict"):
        frame_batch = np.expand_dims(frame, axis=0)
        result = model.predict(frame_batch)
        return result[0] if isinstance(result, (list, np.ndarray)) else result

    if isinstance(model, onnxruntime.InferenceSession):
        input_name = model.get_inputs()[0].name
        frame_batch = np.expand_dims(frame, axis=0).astype(np.float32)
        result = model.run(None, {input_name: frame_batch})[0]
        return result[0] if isinstance(result, np.ndarray) and result.ndim > 0 else result

    if callable(model):
        return model(frame)

    raise ValueError("Unsupported model type for video inference")


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if hasattr(value, "detach"):
        return _normalize(value.detach().cpu().numpy())
    return value


def iter_sequence(
    model: Any, frames: Iterable[np.ndarray], config: Dict[str, Any]
) -> Iterator[Dict[str, Any]]:
    del config
    for idx, frame in enumerate(frames):
        prediction = _infer_single_frame(model, frame)
        yield {"frame_index": idx, "prediction": _normalize(prediction)}


def infer_sequence(model: Any, frames: List[np.ndarray], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(iter_sequence(model, frames, config))
