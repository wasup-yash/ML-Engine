import os
import onnxruntime
import yaml
import argparse
import numpy as np
from typing import Dict, Any
from src.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = {
    "model_path": "./model.joblib",
    "model_format": None,  
    "host": "127.0.0.1",
    "port": 5000,
    "log_level": "INFO",
    "batch_size": 32,
}

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise
import numpy as np
from typing import Any, List

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
            results.extend(model.predict(batch))
        elif isinstance(model, onnxruntime.InferenceSession):
            input_name = model.get_inputs()[0].name
            results.extend(model.run(None, {input_name: np.array(batch)})[0])
        else:
            raise ValueError("Unsupported model type for batch inference")
    return results

def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments.
    
    Returns:
        Dictionary of command line arguments
    """
    parser = argparse.ArgumentParser(description="ML Model Serving Engine")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (YAML)"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="Path to the model file"
    )
    parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch size for inference"
   )
    parser.add_argument(
        "--model_format",
        type=str,
        choices=["joblib", "pickle"],
        help="Format of the model file (if not auto-detected)"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        help="Host address for the API server"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        help="Port for the API server"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    
    return args_dict

def get_config() -> Dict[str, Any]:
    """
    Get the final configuration by merging defaults, config file, and command line args.
    
    Priority order (highest to lowest):
    1. Command line arguments
    2. Configuration file
    3. Default configuration
    
    Returns:
        Merged configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    args = parse_args()
    if "config" in args:
        config_path = args["config"]
        file_config = load_config_from_yaml(config_path)
        config.update(file_config)
        args_copy = args.copy()
        args_copy.pop("config")
        config.update(args_copy)
    else:
        config.update(args)
    logger.info(f"Final configuration: {config}")
    return config