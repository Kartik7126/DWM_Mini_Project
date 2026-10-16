"""
Unit tests for cluster profiling, distinguishing features, and comparisons.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessing import CardiacPreprocessor
from src.clustering import CardiacClusteringModel
from src.cluster_analysis import (
    generate_cluster_profiles,
    compare_patient_to_cluster,
    compare_two_clusters
)


def test_cluster_profiles_and_interpretation():
    df, _ = load_dataset()
    preprocessor = CardiacPreprocessor()
    X = preprocessor.fit_transform(df)

    model = CardiacClusteringModel(random_state=42)
    model.fit(X, k=3)

    profiles = generate_cluster_profiles(df, model.cluster_labels)

    assert len(profiles) == 3
    for c_id in [0, 1, 2]:
        p = profiles[c_id]
        assert "Cluster " in p["title"]
        assert p["record_count"] > 0
        assert "numerical_stats" in p
        assert "categorical_stats" in p
        assert "narrative" in p
        assert "age" in p["numerical_stats"]


def test_patient_and_two_cluster_comparisons():
    df, _ = load_dataset()
    preprocessor = CardiacPreprocessor()
    X = preprocessor.fit_transform(df)

    model = CardiacClusteringModel(random_state=42)
    model.fit(X, k=3)

    profiles = generate_cluster_profiles(df, model.cluster_labels)

    # Patient comparison
    patient_dict = df.iloc[0].to_dict()
    assigned_c = model.cluster_labels[0]
    p_comp = compare_patient_to_cluster(patient_dict, profiles[assigned_c])

    assert p_comp["cluster_id"] == assigned_c
    assert len(p_comp["comparison_items"]) > 0

    # Two clusters comparison
    two_comp_df = compare_two_clusters(profiles[0], profiles[1])
    assert isinstance(two_comp_df, pd.DataFrame)
    assert len(two_comp_df) > 0
    assert "Feature" in two_comp_df.columns
    assert "Higher Cohort" in two_comp_df.columns
