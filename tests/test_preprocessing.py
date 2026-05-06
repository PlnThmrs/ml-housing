import numpy as np
import pandas as pd
from src.ml_housing.preprocessing import get_preprocessing_pipeline


def test_get_preprocessing_pipeline_returns_pipeline():
    pipeline = get_preprocessing_pipeline()

    assert pipeline is not None
    assert pipeline.named_steps["imputer"] is not None
    assert pipeline.named_steps["scaler"] is not None


def test_preprocessing_pipeline_imputes_and_scales():
    pipeline = get_preprocessing_pipeline()

    df = pd.DataFrame(
        {
            "feature_1": [1.0, np.nan, 3.0, 4.0],
            "feature_2": [10.0, 20.0, np.nan, 40.0],
        }
    )

    transformed = pipeline.fit_transform(df)

    assert transformed.shape == (4, 2)
    assert not np.isnan(transformed).any()
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(transformed.std(axis=0, ddof=0), 1.0, atol=1e-7)
