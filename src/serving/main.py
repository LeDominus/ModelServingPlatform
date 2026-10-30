import uvicorn
from mlflow import MlflowClient
from fastapi import Depends, FastAPI, status
from src.DI import load_mlflow_client
from flows.train import train
from flows.inference import inference
from src.config.mlflow_config import MLFLOW_TRACKING_URI
from src.config.prefect_train_config import DATA_PATH
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Запуск приложения...")
    try:
        app.state.mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        print("Mlflow успешно инициализирован")
    except Exception as e:
        raise RuntimeError(f"Ошибка при инициализации mlflow: {e}") from e

    print("Приложение запущено")
    yield
    
app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/prefect/train", status_code=status.HTTP_201_CREATED)
def start_prefect_train(mlflow_client: MlflowClient = Depends(load_mlflow_client)):
    train(DATA_PATH, mlflow_client)

@app.get("/prefect/inference", status_code=status.HTTP_201_CREATED)
def start_prefect_inference(): 
    inference()

if __name__ == "__main__":
    uvicorn.run(app, port=7777)