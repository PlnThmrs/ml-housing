from pathlib import Path

import joblib


def get_latest_preprocessing(_preprocessing=None):
    """Trouve et charge le preprocessing avec le numéro de version le plus élevé."""

    if _preprocessing is not None:
        return _preprocessing

    latest_preprocessing_path = "artifacts/preprocessing/preprocessing_latest.joblib"
    if Path(latest_preprocessing_path).exists():
        _preprocessing = joblib.load(latest_preprocessing_path)
        return _preprocessing
    raise FileNotFoundError("Aucun preprocessing trouvé dans le dossier artifacts.")
