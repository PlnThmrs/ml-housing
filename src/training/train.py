from sklearn.ensemble import RandomForestRegressor

from ..prediction.config import RANDOM_STATE


def train_model(X_train, y_train) -> RandomForestRegressor:
    """Entraîne un modèle de régression."""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
