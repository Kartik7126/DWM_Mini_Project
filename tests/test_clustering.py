"""
Unit tests for K-Means clustering, evaluation, and new record prediction.
"""

import pytest
import numpy as np
from src.data_loader import load_dataset
from src.preprocessing import CardiacPreprocessor
from src.clustering import CardiacClusteringModel


def test_kmeans_evaluation_and_fitting():
    df, _ = load_dataset()
    preprocessor = CardiacPreprocessor()
    X = preprocessor.fit_transform(df)

    model = CardiacClusteringModel(random_state=42)
    eval_results = model.evaluate_candidate_k(X, k_min=2, k_max=5)

    assert 2 in eval_results
    assert 3 in eval_results
    assert 4 in eval_results
    assert 5 in eval_results

    # Check inertia decreases as K increases
    assert eval_results[2]["inertia"] > eval_results[3]["inertia"]
    assert eval_results[3]["inertia"] > eval_results[4]["inertia"]

    # Recommended K should be valid
    assert model.recommended_k in [2, 3, 4, 5]
    assert len(model.recommendation_reason) > 10

    # Fit model with K=3
    model.fit(X, k=3)
    assert model.is_fitted is True
    assert len(model.cluster_labels) == len(df)
    assert len(np.unique(model.cluster_labels)) == 3
    assert model.overall_silhouette is not None
    assert model.pca_coords.shape == (len(df), 2)
    assert model.pca_centroids.shape == (3, 2)


def test_predict_new_record():
    df, _ = load_dataset()
    preprocessor = CardiacPreprocessor()
    X = preprocessor.fit_transform(df)

    model = CardiacClusteringModel(random_state=42)
    model.fit(X, k=3)

    # Transform a single row
    sample_df = df.iloc[[0]].copy()
    x_scaled = preprocessor.transform(sample_df)

    cluster_id, similarity_dict, min_dist, meta = model.predict_new_record(x_scaled)

    assert cluster_id in [0, 1, 2]
    assert cluster_id == model.cluster_labels[0]
    assert len(similarity_dict) == 3
    # Sum of percentages should be ~ 100%
    assert abs(sum(similarity_dict.values()) - 100.0) < 1.0
    assert min_dist >= 0.0
    assert len(meta["pca_coord"]) == 2
