import matplotlib
matplotlib.use("Agg")
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import __max_supported_opset__, to_onnx
from src.config.mlflow_config import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_ARTIFACT_PATH,
)
from src.config.prefect_train_config import (
    TARGET,
    RANDOM_STATE,
    PARAM_GRID,
    REGISTERED_MODEL_NAME,
    TEST_SIZE,
    EXPERIMENT_NAME,
)


def _load_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df[TARGET] = (df[TARGET] == "Yes").astype(np.int64)
    return df


def _preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.get_dummies(df.drop(columns=[TARGET]), dtype=np.float32)
    y = df[TARGET].to_numpy()
    return X, y


def _search_hyperparams(X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        PARAM_GRID,
        scoring="roc_auc",
        cv=7,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return {
        key.rsplit("__", 1)[-1]: value for key, value in search.best_params_.items()
    }


def _fit_model(X_train: pd.DataFrame, y_train: np.ndarray, params: dict) -> Pipeline:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params),
    )
    model.fit(X_train, y_train)
    return model


def _evaluate(
    model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    return metrics, proba, pred


def _convert_to_onnx(model: Pipeline, X_train: pd.DataFrame):
    sample_input = X_train.iloc[:1].to_numpy().astype(np.float32)
    return to_onnx(model, sample_input, target_opset=__max_supported_opset__)


def _log_figure(filename: str, build_plot) -> None:
    fig, ax = plt.subplots()
    build_plot(ax)
    mlflow.log_figure(fig, filename)
    plt.close(fig)


def _save_graphics_to_mlflow(
    y_test: np.ndarray, proba: np.ndarray, pred: np.ndarray
) -> None:
    _log_figure(
        "confusion_matrix.png",
        lambda ax: ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax),
    )
    _log_figure(
        "roc_curve.png",
        lambda ax: RocCurveDisplay.from_predictions(y_test, proba, ax=ax),
    )


def _save_model_to_mlflow(onx, X_sample: np.ndarray) -> None:
    mlflow.onnx.log_model(
        onx,
        artifact_path=MODEL_ARTIFACT_PATH,
        signature=infer_signature(X_sample),
        registered_model_name=REGISTERED_MODEL_NAME,
    )


def _promote_model(client: MlflowClient) -> None:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, latest.version)


def train_pipeline(data_path: Path, client: MlflowClient) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        features, target = _preprocess(_load_data(data_path))
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )
    
        with mlflow.start_run():
            print("[train_pipeline]: Перебор гиперпараметров...")
            params = _search_hyperparams(X_train, y_train)
            print("[train_pipeline]: Гиперпараметры установлены...")
            
            print("[train_pipeline]: Обучение модели...")
            model = _fit_model(X_train, y_train, params)
            print("[train_pipeline]: Модель обучена...")
            
            print("[train_pipeline]: Считаем метрики и конвертируем в onnx...")
            metrics, proba, pred = _evaluate(model, X_test, y_test)
            onx = _convert_to_onnx(model, X_train)
    
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            _save_graphics_to_mlflow(y_test, proba, pred)
            _save_model_to_mlflow(onx, X_train.iloc[:1].to_numpy().astype(np.float32))
            _promote_model(client)
            print("[train_pipeline]: Тренировочный pipeline завершен")
    except Exception as e:
        raise RuntimeError(f"Ошибка при выполении train_pipeline: {e}") from e

