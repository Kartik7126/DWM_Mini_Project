"""
Unit tests for data preprocessing, imputation, scaling, and OOD detection.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessing import CardiacPreprocessor
from src.config import EXCLUDED_COLUMNS


def test_preprocessor_excludes_target():
    df, _ = load_dataset()
    preprocessor = CardiacPreprocessor()
    X = preprocessor.fit_transform(df)

    assert preprocessor.is_fitted is True
    assert "target" not in preprocessor.selected_features
    assert "record_id" not in preprocessor.selected_features
    assert X.shape[0] == len(df)
    assert not np.isnan(X).any(), "Preprocessed matrix should have no NaN values."


def test_preprocessor_scaling_properties():
    df, _ = load_dataset()
    preprocessor = CardiacPreprocessor()
    X = preprocessor.fit_transform(df)

    # Numerical features should have approximately zero mean in scaled space
    num_indices = list(range(len(preprocessor.numerical_cols)))
    means = np.mean(X[:, num_indices], axis=0)
    for m in means:
        assert abs(m) < 1e-1, f"Expected scaled mean near 0, got {m}"


def test_out_of_distribution_detection():
    df, _ = load_dataset()
    preprocessor = CardiacPreprocessor()
    preprocessor.fit(df)

    # Normal record
    normal_record = {
        "age": 55.0,
        "trestbps": 130.0,
        "chol": 240.0,
        "thalach": 150.0,
        "oldpeak": 1.0,
        "ca": 0.0
    }
    warnings_normal = preprocessor.check_out_of_distribution(normal_record)
    assert len(warnings_normal) == 0

    # Extreme record (age 115, cholesterol 750)
    extreme_record = {
        "age": 115.0,
        "trestbps": 280.0,
        "chol": 750.0,
        "thalach": 150.0,
        "oldpeak": 1.0,
        "ca": 0.0
    }
    warnings_extreme = preprocessor.check_out_of_distribution(extreme_record)
    assert len(warnings_extreme) >= 2
    warned_features = [w["feature"] for w in warnings_extreme]
    assert "age" in warned_features
    assert "chol" in warned_features
