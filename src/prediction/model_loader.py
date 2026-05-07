from pathlib import Path

import joblib


def get_latest_model(_model=None):
    """Trouve et charge le modèle avec le numéro de version le plus élevé."""

    if _model is not None:
        return _model

    latest_model_path = "artifacts/models/model_latest.joblib"
    if Path(latest_model_path).exists():
        _model = joblib.load(latest_model_path)
        return _model
    raise FileNotFoundError("Aucun modèle trouvé dans le dossier artifacts.")
