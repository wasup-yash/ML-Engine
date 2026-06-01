from typing import Any, Dict, List

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


def infer_sequence(model: Any, frames: List[np.ndarray], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        prediction = _infer_single_frame(model, frame)
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        predictions.append({"frame_index": idx, "prediction": prediction})
    return predictions
