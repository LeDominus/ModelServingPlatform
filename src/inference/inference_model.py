import pandas as pd
from pathlib import Path


def inference_pipeline(model, data_path: Path):
    """Pipeline для прогноза"""
    data = pd.read_csv(data_path)
    try:
        return model.fit_predict(data)
    except Exception as e:
        raise ValueError(f"Ошибка при прогнозе: {e}") from e