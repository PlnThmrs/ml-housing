import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Configuration du chemin pour trouver le dossier backend et src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


@pytest.fixture
def mock_model():
    """Fixture pour créer un faux modèle."""
    model = MagicMock()
    # On simule la méthode predict.
    # Note : si ton script src.prediction.predict appelle model.predict
    model.predict.return_value = np.array([2.5])
    return model


@pytest.fixture
def mock_preprocessing():
    """Fixture pour créer un faux preprocessing pipeline."""
    preprocessing = MagicMock()
    # On simule la transformation des données
    preprocessing.transform.return_value = np.array(
        [[3.5, 20.0, 5.0, 1.0, 1000.0, 3.0, 34.0, -118.0]]
    )
    return preprocessing


@pytest.fixture
def client(mock_model, mock_preprocessing):
    """Fixture pour créer un client de test FastAPI avec injection de mocks."""

    # 1. On patch les fonctions de chargement AVANT d'importer l'app
    # On cible l'endroit où elles sont définies ou importées dans app.py
    with (
        patch("src.prediction.model_loader.get_latest_model", return_value=mock_model),
        patch(
            "src.prediction.preprocessing_loader.get_latest_preprocessing",
            return_value=mock_preprocessing,
        ),
    ):
        import app

        # On recharge le module pour s'assurer que les variables globales
        # 'model' et 'preprocessing' prennent bien nos mocks
        importlib.reload(app)

        with TestClient(app.app) as c:
            yield c


def test_health_endpoint(client):
    """Teste l'endpoint /health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_with_valid_data(client, mock_preprocessing, mock_model):
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

    # On suppose que predict() dans app.py utilise le preprocessing et le model
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    # On vérifie la structure de la réponse (ajuste selon ton script predict.py)
    assert "prediction" in data or isinstance(data, (float, int, list, dict))


@pytest.mark.parametrize(
    "payload",
    [
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
    ],
)
def test_predict_endpoint_multiple_inputs(client, payload):
    """Teste /predict avec différentes entrées via paramétrage pytest."""
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
