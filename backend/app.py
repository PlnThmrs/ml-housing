from fastapi import FastAPI

from src.prediction.model_loader import get_latest_model
from src.prediction.predict import predict
from src.prediction.preprocessing_loader import get_latest_preprocessing

app = FastAPI()

# Cache variables
_model = None
_preprocessing = None

model = get_latest_model(_model)
preprocessing = get_latest_preprocessing(_preprocessing)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def make_prediction(data: dict):
    return predict(data, model, preprocessing)
