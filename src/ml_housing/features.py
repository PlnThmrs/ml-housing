import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE

TARGET_COLUMN = "MedHouseVal"


def split_features_target(df: pd.DataFrame):
    """Sépare les variables explicatives X et la cible y."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_train_test(
    X,
    y,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Découpe les données en train/test."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
