import joblib
from fastapi import FastAPI

from backend.storage.s3_client import (
    download_model_from_s3,
    download_preprocessing_from_s3,
)
from src.prediction.predict import predict

app = FastAPI()

# Cache variables
model = None
preprocessing = None


@app.on_event("startup")
def load_artifacts_on_startup():
    """Télécharge modèle et prétraitement depuis MinIO puis les charge en mémoire."""

    global model, preprocessing

    model_path = download_model_from_s3()
    model = joblib.load(model_path)
    print(f"Modèle chargé en mémoire depuis : {model_path}")

    preprocessing_path = download_preprocessing_from_s3()
    preprocessing = joblib.load(preprocessing_path)
    print(f"Prétraitement chargé en mémoire depuis : {preprocessing_path}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def make_prediction(data: dict):
    return predict(data, model, preprocessing)
