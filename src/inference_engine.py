import numpy as np
from typing import Any, List


def _is_onnx_session(model: Any) -> bool:
    return model.__class__.__module__.startswith("onnxruntime") and hasattr(model, "run")


def _to_list(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def batch_inference(model: Any, inputs: List[Any], batch_size: int) -> List[Any]:
    """
    Perform batched inference on the inputs.

    Args:
        model: Loaded model object.
        inputs: List of input data.
        batch_size: Number of inputs per batch.

    Returns:
        List of inference results.
    """
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        if hasattr(model, "predict"):
            results.extend(_to_list(model.predict(batch)))
        elif _is_onnx_session(model):
            input_name = model.get_inputs()[0].name
            results.extend(_to_list(model.run(None, {input_name: np.asarray(batch)})[0]))
        elif callable(model):
            try:
                import torch

                tensor = torch.as_tensor(np.asarray(batch))
                with torch.inference_mode():
                    output = model(tensor)
                if hasattr(output, "detach"):
                    output = output.detach().cpu().numpy()
            except (ImportError, TypeError):
                output = model(batch)
            results.extend(_to_list(output))
        else:
            raise ValueError("Unsupported model type for batch inference")
    return results
