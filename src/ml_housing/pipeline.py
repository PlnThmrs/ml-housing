import json
from pathlib import Path

import joblib

from src.ml_housing.config import MODEL_FILENAME, RANDOM_STATE, TEST_SIZE
from src.ml_housing.data import load_housing_data
from src.ml_housing.evaluate import evaluate_model
from src.ml_housing.features import split_features_target, split_train_test
from src.ml_housing.train import train_model


def run_pipeline(artifacts_dir: str = "artifacts") -> dict:
    """Exécute le pipeline ML complet en local."""
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    df = load_housing_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    joblib.dump(model, artifacts_path / MODEL_FILENAME)
    with open(artifacts_path / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics
