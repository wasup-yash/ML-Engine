import os
import joblib
import pickle
import onnxruntime
import torch
import tensorflow as tf
from typing import Any, Dict, Optional
from src.logger import get_logger

logger = get_logger(__name__)

def load_model(model_path: str, format: Optional[str] = None) -> Any:
    """
    Load a machine learning model from disk.
    
    Args:
        model_path: Path to the model file
        format: Format of the model file ('joblib', 'pickle', or None to auto-detect)
        
    Returns:
        The loaded model object
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the format is unsupported or auto-detection fails
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}") 
    if format is None:
        _, ext = os.path.splitext(model_path)
        if ext.lower() in ['.joblib', '.jlib']:
            format = 'joblib'
        elif ext.lower() in ['.pkl', '.pickle']:
            format = 'pickle'
        elif ext.lower() == ".onnx":
            format = "onnx"
        elif ext.lower() in [".pt", ".pth"]:
            format = "torch"
        elif ext.lower() in [".pb", ".savedmodel"]:
            format = "tensorflow"
        else:
            logger.warning(f"Could not auto-detect model format from extension: {ext}")
           
            format = 'auto'    
    logger.info(f"Loading model from {model_path} using {format} format")
    
    try:
        if format.lower() == 'auto':
         
            logger.info(f"Attempting to load model with multiple formats")
            try:
              
                logger.info(f"Trying joblib format")
                model = joblib.load(model_path)
                logger.info(f"Model loaded successfully with joblib")
                return model
            except Exception as je:
                logger.warning(f"Joblib loading failed: {str(je)}, trying pickle")
                try:
                   
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    logger.info(f"Model loaded successfully with pickle")
                    return model
                except Exception as pe:
                    logger.error(f"Both joblib and pickle failed to load the model")
                    raise ValueError(f"Could not load model with either joblib or pickle. Errors: joblib - {str(je)}; pickle - {str(pe)}")
                    
        elif format.lower() == 'joblib':
            logger.info(f"Loading model from: {model_path} using joblib format")
            try:
                model = joblib.load(model_path)
            except KeyError as ke:
                logger.error(f"Joblib loading error - possible version incompatibility: {str(ke)}")
                raise ValueError(f"Model file may be corrupted or incompatible with this version of joblib/Python. Error: {str(ke)}")
            except Exception as je:
                logger.error(f"Error loading model with joblib: {str(je)}")
                raise ValueError(f"Failed to load model with joblib: {str(je)}")
        elif format.lower() == 'pickle':
            logger.info(f"Loading model from: {model_path} using pickle format")
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except KeyError as ke:
                logger.error(f"Pickle loading error - possible version incompatibility: {str(ke)}")
                raise ValueError(f"Model file may be corrupted or incompatible with this version of pickle/Python. Error: {str(ke)}")
            except Exception as pe:
                logger.error(f"Error loading model with pickle: {str(pe)}")
                raise ValueError(f"Failed to load model with pickle: {str(pe)}")
        elif format.lower() == 'onnx':
            logger.info(f"Loading model from: {model_path} using onnx format")
            try:
                model = onnxruntime.InferenceSession(model_path)
            except KeyError as ke:
                logger.error(f"ONNX loading error - possible version incompatibility: {str(ke)}")
                raise ValueError(f"Model file may be corrupted or incompatible with this version of onnxruntime/Python. Error: {str(ke)}")
            except Exception as oe:
                logger.error(f"Error loading model with onnx: {str(oe)}")
                raise ValueError(f"Failed to load model with onnx: {str(oe)}")
        elif format.lower() == 'torch':
            logger.info(f"Loading model from: {model_path} using joblib format")
            try:
                model = torch.load(model_path)
            except KeyError as ke:
                logger.error(f"torch loading error - possible version incompatibility: {str(ke)}")
                raise ValueError(f"Model file may be corrupted or incompatible with this version of torch/Python. Error: {str(ke)}")
            except Exception as je:
                logger.error(f"Error loading model with torch: {str(je)}")
                raise ValueError(f"Failed to load model with torch: {str(je)}")
        elif format.lower() == 'tensorflow':
            logger.info(f"Loading model from: {model_path} using tensorflow format")
            try:
                model = tf.saved_model.load(model_path)
            except KeyError as ke:
                logger.error(f"tensorflow loading error - possible version incompatibility: {str(ke)}")
                raise ValueError(f"Model file may be corrupted or incompatible with this version of tensorflow/Python. Error: {str(ke)}")
            except Exception as je:
                logger.error(f"Error loading model with tensorflow: {str(je)}")
                raise ValueError(f"Failed to load model with tensorflow: {str(je)}")
        else:
            logger.error(f"Unsupported model format: {format}")
            raise ValueError(f"Unsupported model format: {format}")
            
        logger.info(f"Model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise