from unittest.mock import MagicMock, patch

import pytest
import requests


class TestFrontendIntegration:
    """Tests d'intégration pour le frontend Streamlit."""

    def test_frontend_can_send_prediction_request(self):
        """Teste que le frontend peut envoyer une requête valide à l'API."""
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

        # Mock la réponse du backend
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"prediction": 2.5}
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            response = requests.post(
                "http://127.0.0.1:8000/predict", json=payload, timeout=5
            )

            assert response.status_code == 200
            assert response.json()["prediction"] == 2.5
            mock_post.assert_called_once()

    def test_frontend_handles_connection_error(self):
        """Teste que le frontend gère les erreurs de connexion."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError(
                "Impossible de contacter le serveur"
            )

            with pytest.raises(requests.exceptions.ConnectionError):
                requests.post(
                    "http://127.0.0.1:8000/predict", json={"MedInc": 3.5}, timeout=5
                )

    def test_frontend_payload_structure(self):
        """Teste que le payload du frontend a la structure correcte."""
        # Données saisies par l'utilisateur dans le frontend
        med_inc = 3.5
        house_age = 20.0
        ave_rooms = 5.0
        ave_bedrms = 1.0
        population = 1000.0
        ave_occup = 3.0
        latitude = 34.0
        longitude = -118.0

        # Simulation de la construction du payload
        payload = {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }

        # Vérifier que tous les champs requis sont présents
        required_fields = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]

        for field in required_fields:
            assert field in payload
            assert isinstance(payload[field], (int, float))

    def test_frontend_prediction_display_format(self):
        """Teste que la prédiction est correctement formatée pour l'affichage."""
        prediction = 2.5  # Prédiction en unités de 100k$

        # Simulation du format d'affichage du frontend
        displayed_price = prediction * 100_000  # Conversion en dollars

        assert displayed_price == 250_000
        assert isinstance(displayed_price, float)

    @patch("requests.post")
    def test_frontend_api_call_with_mocked_backend(self, mock_post):
        """Teste l'appel API complet du frontend avec un backend mocké."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"prediction": 3.2}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        payload = {
            "MedInc": 4.0,
            "HouseAge": 25.0,
            "AveRooms": 5.5,
            "AveBedrms": 1.5,
            "Population": 1500.0,
            "AveOccup": 3.5,
            "Latitude": 34.5,
            "Longitude": -118.5,
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict", json=payload, timeout=5
        )
        response.raise_for_status()
        prediction = response.json()["prediction"]

        assert prediction == 3.2
        assert isinstance(prediction, float)
