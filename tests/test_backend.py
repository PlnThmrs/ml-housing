import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ajouter le chemin du backend
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


@pytest.fixture
def mock_model():
    """Fixture pour créer un faux modèle."""
    model = MagicMock()
    # Simuler les prédictions (le format dépend de ce qu'attend ton app, ici un float)
    model.predict.return_value = np.array([2.5])
    return model


@pytest.fixture
def mock_preprocessing():
    """Fixture pour créer un faux preprocessing pipeline."""
    preprocessing = MagicMock()
    # Simuler la transformation des données
    preprocessing.transform.return_value = np.array(
        [[3.5, 20.0, 5.0, 1.0, 1000.0, 3.0, 34.0, -118.0]]
    )
    return preprocessing


@pytest.fixture
def client(mock_model, mock_preprocessing):
    """Fixture pour créer un client de test FastAPI avec modèles moqués."""
    # Utilisation de patch en tant que context manager pour injecter les mocks
    with (
        patch("app.get_latest_model", return_value=mock_model),
        patch("app.get_latest_preprocessing", return_value=mock_preprocessing),
    ):
        # Réimporter l'app pour s'assurer que les dépendances sont injectées
        import app as app_module

        # Forcer la réinitialisation si l'app charge les modèles au démarrage
        app_module.model = mock_model
        app_module.preprocessing = mock_preprocessing

        with TestClient(app_module.app) as c:
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

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] > 0

    # Vérifier que les mocks ont été sollicités
    mock_preprocessing.transform.assert_called()
    mock_model.predict.assert_called()


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
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)
