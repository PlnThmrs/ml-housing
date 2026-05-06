from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# Cache variables
_model = None
_preprocessing = None


def get_latest_model():
    """Trouve et charge le modèle avec le numéro de version le plus élevé."""
    global _model
    if _model is not None:
        return _model

    artifacts_path = Path("artifacts/models")
    models = list(artifacts_path.glob("model_v*.joblib"))
    if not models:
        # Fallback au modèle sans version si existant
        default_model = artifacts_path / "model.joblib"
        if default_model.exists():
            _model = joblib.load(default_model)
            return _model
        raise FileNotFoundError("Aucun modèle trouvé dans le dossier artifacts.")

    # Trie par numéro de version et prend le dernier
    latest_model_path = sorted(models, key=lambda x: int(x.stem.split("_v")[-1]))[-1]
    _model = joblib.load(latest_model_path)
    return _model


def get_latest_preprocessing():
    """Trouve et charge le preprocessing avec le numéro de version le plus élevé."""
    global _preprocessing
    if _preprocessing is not None:
        return _preprocessing

    artifacts_path = Path("artifacts/preprocessing")
    preprocessings = list(artifacts_path.glob("preprocessing_v*.joblib"))
    if not preprocessings:
        # Fallback au preprocessing sans version si existant
        default_preprocessing = artifacts_path / "preprocessing.joblib"
        if default_preprocessing.exists():
            _preprocessing = joblib.load(default_preprocessing)
            return _preprocessing
        raise FileNotFoundError("Aucun preprocessing trouvé dans le dossier artifacts.")

    # Trie par numéro de version et prend le dernier
    latest_preprocessing_path = sorted(
        preprocessings, key=lambda x: int(x.stem.split("_v")[-1])
    )[-1]
    _preprocessing = joblib.load(latest_preprocessing_path)
    return _preprocessing


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    model = get_latest_model()
    preprocessing = get_latest_preprocessing()
    df = pd.DataFrame([data])
    df_preprocessed = preprocessing.transform(df)
    prediction = model.predict(df_preprocessed)[0]
    return {"prediction": float(prediction)}
