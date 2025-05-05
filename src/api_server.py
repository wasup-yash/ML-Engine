# import numpy as np
# from typing import Any, Dict, List, Union
# from fastapi import FastAPI, HTTPException,UploadFile
# from src.config_manager import get_config
# from src.model_loader import load_model
# from pydantic import BaseModel, Field
# import uvicorn
# from src.logger import get_logger

# logger = get_logger(__name__)

# class PredictionInput(BaseModel):
#     """Schema for prediction input data."""
#     data: Union[List[List[float]], List[float], Dict[str, Any]] = Field(
#         ..., 
#         description="Input data for prediction. Can be a 2D array, 1D array, or dictionary of features"
#     )

# class PredictionResponse(BaseModel):
#     """Schema for prediction response."""
#     prediction: Union[List[Any], Any] = Field(
#         ..., 
#         description="Model prediction(s)"
#     )

# class APIServer:
#     """API server for serving ML model predictions."""
    
#     def __init__(self, model: Any, host: str = "127.0.0.1", port: int = 5000):
#         """
#         Initialize the API server.
        
#         Args:
#             model: The loaded ML model object
#             host: Host address to bind the server
#             port: Port to bind the server
#         """
#         self.model = model
#         self.host = host
#         self.port = port
#         self.app = FastAPI(
#             title="ML Model Serving API",
#             description="API for serving machine learning model predictions",
#             version="0.1.0",
#         )
        
        
#         self._register_routes()
        
#     def _register_routes(self):
#         """Register API routes."""
        
#         @self.app.get("/")
#         def root():
#             """Root endpoint."""
#             return {"message": "ML Model Serving API is running"}
        
#         @self.app.get("/health")
#         def health():
#             """Health check endpoint."""
#             return {"status": "ok"}
        
#         @self.app.post("/predict", response_model=PredictionResponse)
#         async def predict(input_data: PredictionInput):
#             """
#             Make predictions using the loaded model.
            
#             Args:
#                 input_data: Input data for prediction
                
#             Returns:
#                 Model predictions
#             """
#             try:
                
#                 data = input_data.data
                
                
#                 if isinstance(data, dict):
                    
#                     logger.info(f"Making prediction with feature dictionary")
#                     result = self.model.predict(data)
#                 else:
                  
#                     data_array = np.array(data)
                   
#                     if data_array.ndim == 1:
#                         data_array = data_array.reshape(1, -1)
                    
#                     logger.info(f"Making prediction with array of shape {data_array.shape}")
#                     result = self.model.predict(data_array)
                
#                 if isinstance(result, np.ndarray):
#                     result = result.tolist()
                
#                 logger.info(f"Prediction successful")
#                 return {"prediction": result}
            
#             except Exception as e:
#                 logger.error(f"Prediction error: {str(e)}")
#                 raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
#     def run(self):
#         """Run the API server."""
#         logger.info(f"Starting API server on {self.host}:{self.port}")
#         uvicorn.run(self.app, host=self.host, port=self.port)

import numpy as np
from typing import Any, Dict, List, Union
from fastapi import FastAPI, HTTPException, UploadFile
from src.config_manager import get_config
from src.model_loader import load_model
from src.inference_engine import batch_inference
from pydantic import BaseModel, Field
import uvicorn
from src.logger import get_logger

logger = get_logger(__name__)
config = get_config()
app = FastAPI()
class PredictionInput(BaseModel):
    """Schema for prediction input data."""
    data: Union[List[List[float]], List[float], Dict[str, Any]] = Field(
        ..., 
        description="Input data for prediction. Can be a 2D array, 1D array, or dictionary of features"
    )

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction: Union[List[Any], Any] = Field(
        ..., 
        description="Model prediction(s)"
    )

class APIServer:
    """API server for serving ML model predictions."""
    
    def __init__(self, model: Any, host: str = "127.0.0.1", port: int = 5000):
        """
        Initialize the API server.
        
        Args:
            model: The loaded ML model object
            host: Host address to bind the server
            port: Port to bind the server
        """
        self.model = model
        self.host = host
        self.port = port
        self.config = get_config()
        self.app = FastAPI(
            title="ML Model Serving API",
            description="API for serving machine learning model predictions",
            version="0.1.0",
        )
        
        self._register_routes()
        
    def _register_routes(self):
        """Register API routes."""
        
        @self.app.get("/")
        def root():
            """Root endpoint."""
            return {"message": "ML Model Serving API is running"}
        
        @self.app.get("/health")
        def health():
            """Health check endpoint."""
            return {"status": "ok"}
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(input_data: PredictionInput):
            """
            Make predictions using the loaded model.
            
            Args:
                input_data: Input data for prediction
                
            Returns:
                Model predictions
            """
            try:
                data = input_data.data
                
                if isinstance(data, dict):
                    logger.info(f"Making prediction with feature dictionary")
                    result = self.model.predict(data)
                else:
                    data_array = np.array(data)
                    if data_array.ndim == 1:
                        data_array = data_array.reshape(1, -1)
                    
                    logger.info(f"Making prediction with array of shape {data_array.shape}")
                    result = self.model.predict(data_array)
                
                if isinstance(result, np.ndarray):
                    result = result.tolist()
                
                logger.info(f"Prediction successful")
                return {"prediction": result}
            
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
        @self.app.post("/infer")
        async def infer(inputs: List[float]):
            """
            Perform batched inference on the provided inputs.
            """
            try:
                batch_size = self.config["batch_size"]
                results = batch_inference(self.model, inputs, batch_size)
                return {"results": results}
            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
        @self.app.post("/upload_model")
        async def upload_model(file: UploadFile):
            """
            Upload a new model file and reload it.
            """
            try:
                with open(self.config["model_path"], "wb") as f:
                    f.write(await file.read())
                self.model = load_model(self.config["model_path"], self.config["model_format"])
                logger.info("Model reloaded successfully")
                return {"message": "Model reloaded successfully"}
            except Exception as e:
                logger.error(f"Model upload error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Model upload error: {str(e)}")
    
    def run(self):
        """Run the API server."""
        logger.info(f"Starting API server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)

from src.watcher import watch_model_file

def reload_model():
    global model
    model = load_model(config["model_path"], config["model_format"])
    print("Model reloaded due to file change")

# Start the watcher in a separate thread
import threading
threading.Thread(target=watch_model_file, args=(config["model_path"], reload_model), daemon=True).start()