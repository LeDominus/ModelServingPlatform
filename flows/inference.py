from prefect import task, flow
from pathlib import Path
from src.config.prefect_inference_config import DATA_PATH
from src.inference.inference_model import inference_pipeline

@task
def load_model():
    """Загрузка модели из Mlflow"""

"""Пока оставляю прогноз через ONNX runtime, далее подключу Triton Inference Server"""

@task
def run_inference(model, data_path: Path | None = DATA_PATH):
    """Прогноз загруженной моделью"""
    print("[inference] Начинаем прогноз...")
    inference_pipeline(model, data_path)
    print("[inference] Прогноз завершен успешно")

@flow
def inference():
    """Точка входа для прогноза"""
    model_future = load_model.submit()
    run_inference.submit(DATA_PATH, wait_for=[model_future])
    