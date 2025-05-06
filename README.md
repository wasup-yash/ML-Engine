# ML Model Serving Engine

A lightweight engine for serving machine learning models via a REST API.

## Features

- Load ML models saved in standard formats (joblib, pickle, onnx, pytorch, Tensorflow Savedmodel)
- Expose models through a FastAPI-based REST API
- Configure via command-line arguments or YAML configuration files
- Built-in logging
- Async Inference (Async API endpoints via FastAPI)
- Batch Intference

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-serving-engine.git
   cd ml-serving-engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Engine

1. With default configuration (looks for `model.joblib` in current directory):
   ```bash
   python run_engine.py
   ```

2. With a configuration file:
   ```bash
   python run_engine.py --config config/default_config.yaml
   ```

3. With command-line arguments:
   ```bash
   python run_engine.py --model_path ./models/my_model.joblib --host 0.0.0.0 --port 8000
   ```

### Using the API

Once the engine is running, you can make predictions via the API:

```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [5.1, 3.5, 1.4, 0.2]}'
```

Response:
```json
{
  "prediction": [0]
}
```

You can also visit the auto-generated API documentation at: `http://localhost:5000/docs`

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `model_path` | Path to the model file | `./model.joblib` |
| `model_format` | Format of the model file (auto-detect if not specified) | `null` |
| `host` | Host address to bind the server | `127.0.0.1` |
| `port` | Port to bind the server | `5000` |
| `log_level` | Logging level | `INFO` |

## API Endpoints

- `GET /` - Root endpoint (health check)
- `GET /health` - Health check endpoint
- `POST /predict` - Make predictions using the loaded model

## Example YAML Configuration

```yaml
model_path: "./models/classifier.joblib"
model_format: "joblib"

host: "0.0.0.0"
port: 8000

log_level: "INFO"
```

## Creating a Sample Model

Here's a simple example of creating and saving a model for use with this engine:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.joblib")
```

## License

This project is licensed under the MIT License.