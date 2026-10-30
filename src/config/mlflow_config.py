import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

MLFLOW_CACHE_DIR = Path("tmp/mlflow_cache")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "logistic_regression.onnx"
MODEL_ALIAS = "champion"
MODEL_ARTIFACT_PATH = "model"