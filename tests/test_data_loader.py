"""
Unit tests for data loading, validation, and demo data generation.
"""

import pytest
import pandas as pd
from src.data_loader import load_dataset, generate_synthetic_demo_dataset
from src.config import ALL_CLUSTERING_FEATURES, EXCLUDED_COLUMNS


def test_load_default_dataset():
    df, report = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 303
    assert "record_id" in df.columns
    assert df["record_id"].iloc[0] == "REC-001"
    assert report["is_valid"] is True
    assert report["raw_rows"] == 303
    assert report["completeness_pct"] > 95.0
    assert "target" in df.columns  # Target present in raw data, but excluded from clustering


def test_dataset_clustering_features_present():
    df, report = load_dataset()
    for feat in ALL_CLUSTERING_FEATURES:
        assert feat in report["present_clustering_features"]
    assert "target" not in report["present_clustering_features"]


def test_generate_synthetic_demo_dataset():
    df_syn, report_syn = generate_synthetic_demo_dataset(n_samples=50, seed=123)
    assert len(df_syn) == 50
    assert report_syn["is_synthetic"] is True
    assert report_syn["is_valid"] is True
    assert "record_id" in df_syn.columns
    assert df_syn["age"].min() >= 29
    assert df_syn["age"].max() <= 77
