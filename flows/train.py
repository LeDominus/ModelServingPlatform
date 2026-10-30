from pathlib import Path

from mlflow.tracking import MlflowClient
from prefect import flow, task
from src.config.mlflow_config import MLFLOW_TRACKING_URI
from src.config.prefect_train_config import DATA_PATH
from src.training.train_model import train_pipeline

@task
def train_model(data_path: Path, mlflow_client: MlflowClient) -> None:
    """Задача для обучения модели и вывода лога"""
    print("[train_model] Начало процесса обучения...")
    train_pipeline(data_path, mlflow_client)
    print("[train_model] Переобучение модели успешно завершено")

@flow
def train(data_path: Path = DATA_PATH, mlflow_client: MlflowClient | None = None) -> None:
    """Точка входа"""
    client = mlflow_client or MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    train_model(data_path, client)

if __name__ == "__main__":
    train()
