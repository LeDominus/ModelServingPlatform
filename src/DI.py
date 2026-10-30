from fastapi import Request
from mlflow import MlflowClient


def load_mlflow_client(request: Request) -> MlflowClient:
    return request.app.state.mlflow_client
