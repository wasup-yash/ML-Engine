import numpy as np
from typing import Any, List
import onnxruntime

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
            # For models with a `predict` method (e.g., scikit-learn, XGBoost)
            results.extend(model.predict(batch))
        elif isinstance(model, onnxruntime.InferenceSession):
            # For ONNX models
            input_name = model.get_inputs()[0].name
            results.extend(model.run(None, {input_name: np.array(batch)})[0])
        else:
            raise ValueError("Unsupported model type for batch inference")
    return results