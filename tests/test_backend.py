import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ajouter le chemin du backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app import app


@pytest.fixture
def client():
    """Fixture pour créer un client de test FastAPI."""
    return TestClient(app)


def test_health_endpoint(client):
    """Teste l'endpoint /health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_with_valid_data(client):
    """Teste l'endpoint /predict avec des données valides."""
    payload = {
        "MedInc": 3.5,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] > 0  # Prix immobilier toujours positif


def test_predict_endpoint_returns_float_prediction(client):
    """Teste que /predict retourne bien une prédiction au format float."""
    payload = {
        "MedInc": 5.0,
        "HouseAge": 15.0,
        "AveRooms": 6.0,
        "AveBedrms": 2.0,
        "Population": 2000.0,
        "AveOccup": 2.5,
        "Latitude": 35.0,
        "Longitude": -119.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    prediction = response.json()["prediction"]
    assert isinstance(prediction, float)


def test_predict_with_different_inputs(client):
    """Teste /predict avec différentes entrées."""
    test_cases = [
        {
            "MedInc": 2.0,
            "HouseAge": 30.0,
            "AveRooms": 4.0,
            "AveBedrms": 1.0,
            "Population": 500.0,
            "AveOccup": 2.0,
            "Latitude": 32.0,
            "Longitude": -117.0,
        },
        {
            "MedInc": 8.0,
            "HouseAge": 5.0,
            "AveRooms": 7.0,
            "AveBedrms": 3.0,
            "Population": 3000.0,
            "AveOccup": 4.0,
            "Latitude": 37.0,
            "Longitude": -122.0,
        },
    ]

    for payload in test_cases:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert isinstance(response.json()["prediction"], float)
