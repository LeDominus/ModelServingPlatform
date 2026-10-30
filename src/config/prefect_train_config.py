from pathlib import Path

TARGET = "Churn"
DATA_PATH = Path("data/raw/churn_data.csv")
EXPERIMENT_NAME = "churn_training"
REGISTERED_MODEL_NAME = "logistic_regression"
TEST_SIZE = 0.2
RANDOM_STATE = 42
PARAM_GRID = {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}