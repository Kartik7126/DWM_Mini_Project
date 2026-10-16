"""
Core unsupervised machine learning module: K-Means clustering,
Elbow method, Silhouette score evaluation, and patient assignment.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

from src.config import DEFAULT_RANDOM_STATE, N_INIT, DEFAULT_K_MIN, DEFAULT_K_MAX


class CardiacClusteringModel:
    """
    Manages K-Means clustering, evaluation metrics, dimensionality reduction,
    and single-patient similarity profiling.
    """

    def __init__(self, random_state: int = DEFAULT_RANDOM_STATE, n_init: int = N_INIT):
        self.random_state = random_state
        self.n_init = n_init
        self.k: int = 3
        self.kmeans: Optional[KMeans] = None
        self.pca: Optional[PCA] = None
        self.pca_coords: Optional[np.ndarray] = None
        self.pca_centroids: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.overall_silhouette: Optional[float] = None
        self.sample_silhouettes: Optional[np.ndarray] = None
        self.evaluation_results: Dict[int, Dict[str, Any]] = {}
        self.recommended_k: int = 3
        self.recommendation_reason: str = ""
        self.scaled_X: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def evaluate_candidate_k(
        self, X: np.ndarray, k_min: int = DEFAULT_K_MIN, k_max: int = DEFAULT_K_MAX
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate candidate values of K using Inertia (Elbow method) and Silhouette Scores.
        """
        n_samples = X.shape[0]
        max_possible_k = min(k_max, n_samples - 1)
        k_min = max(2, min(k_min, max_possible_k))

        results = {}
        for k_val in range(k_min, max_possible_k + 1):
            km = KMeans(
                n_clusters=k_val,
                random_state=self.random_state,
                n_init=self.n_init
            )
            labels = km.fit_predict(X)
            inertia = float(km.inertia_)

            # Check cluster sizes to avoid degenerate clusters
            unique, counts = np.unique(labels, return_counts=True)
            min_cluster_size = int(np.min(counts))
            cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

            # Silhouette score is only valid if more than 1 cluster and less than n_samples
            if 1 < len(np.unique(labels)) < n_samples:
                sil_score = float(silhouette_score(X, labels))
            else:
                sil_score = -1.0

            results[k_val] = {
                "k": k_val,
                "inertia": round(inertia, 2),
                "silhouette_score": round(sil_score, 4),
                "min_cluster_size": min_cluster_size,
                "cluster_sizes": cluster_sizes
            }

        self.evaluation_results = results
        self.recommended_k, self.recommendation_reason = self._determine_recommended_k(results)
        return results

    def _determine_recommended_k(self, results: Dict[int, Dict[str, Any]]) -> Tuple[int, str]:
        """
        Select a defensible K balancing silhouette score, elbow rate of change,
        and cluster size balance (avoiding singleton or tiny clusters).
        """
        if not results:
            return 3, "Default heuristic applied (K=3)."

        k_candidates = sorted(results.keys())
        if len(k_candidates) == 1:
            return k_candidates[0], "Only one candidate K evaluated."

        # Compute percentage inertia drop from previous K
        inertia_drops = {}
        for i in range(1, len(k_candidates)):
            prev_k = k_candidates[i - 1]
            curr_k = k_candidates[i]
            prev_inertia = results[prev_k]["inertia"]
            curr_inertia = results[curr_k]["inertia"]
            if prev_inertia > 0:
                pct_drop = (prev_inertia - curr_inertia) / prev_inertia
            else:
                pct_drop = 0.0
            inertia_drops[curr_k] = pct_drop

        # Score candidates considering silhouette, cluster balance, and elbow drop
        scores = {}
        for k_val, meta in results.items():
            sil = meta["silhouette_score"]
            min_size = meta["min_cluster_size"]

            # Penalize degenerate clusters (fewer than 15 samples)
            size_penalty = 0.0
            if min_size < 15:
                size_penalty = 0.20
            elif min_size < 25:
                size_penalty = 0.08

            # Reward elbow drop if available
            drop_bonus = inertia_drops.get(k_val, 0.15) * 0.25

            score = sil + drop_bonus - size_penalty
            scores[k_val] = score

        best_k = max(scores, key=scores.get)
        best_meta = results[best_k]

        reason = (
            f"K={best_k} achieves a defensible balance with a Silhouette Score of "
            f"{best_meta['silhouette_score']:.3f}, balanced cluster sizes "
            f"(smallest cluster has {best_meta['min_cluster_size']} patients), "
            f"and substantial reduction in intra-cluster variance (SSE/Inertia = {best_meta['inertia']:.1f})."
        )

        return best_k, reason

    def fit(self, X: np.ndarray, k: Optional[int] = None) -> "CardiacClusteringModel":
        """
        Fit K-Means model with selected K and perform PCA dimensionality reduction.
        """
        if k is not None:
            self.k = int(k)
        elif self.recommended_k:
            self.k = self.recommended_k
        else:
            self.k = 3

        self.scaled_X = X

        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            n_init=self.n_init
        )
        self.cluster_labels = self.kmeans.fit_predict(X)

        # Calculate final silhouette metrics
        if len(np.unique(self.cluster_labels)) > 1 and X.shape[0] > self.k:
            self.overall_silhouette = float(silhouette_score(X, self.cluster_labels))
            self.sample_silhouettes = silhouette_samples(X, self.cluster_labels)
        else:
            self.overall_silhouette = 0.0
            self.sample_silhouettes = np.zeros(X.shape[0])

        # Fit 2D PCA for visualization
        self.pca = PCA(n_components=2, random_state=self.random_state)
        self.pca_coords = self.pca.fit_transform(X)
        self.pca_centroids = self.pca.transform(self.kmeans.cluster_centers_)

        self.is_fitted = True
        return self

    def predict_new_record(
        self, x_scaled: np.ndarray
    ) -> Tuple[int, Dict[int, float], float, Dict[str, Any]]:
        """
        Assign a single scaled patient record to the nearest cluster centroid
        and calculate relative similarity scores across all clusters.
        """
        if not self.is_fitted or self.kmeans is None:
            raise RuntimeError("Model must be fitted before predicting new records.")

        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)

        centroids = self.kmeans.cluster_centers_

        # Calculate Euclidean distances to all cluster centroids
        distances = np.linalg.norm(centroids - x_scaled, axis=1)
        assigned_cluster = int(np.argmin(distances))
        min_distance = float(distances[assigned_cluster])

        # Calculate relative similarity percentage using inverse distance weighting
        eps = 1e-5
        inv_distances = 1.0 / (distances + eps)
        similarity_percentages = (inv_distances / np.sum(inv_distances)) * 100.0
        similarity_dict = {
            int(cluster_idx): round(float(pct), 1)
            for cluster_idx, pct in enumerate(similarity_percentages)
        }

        # Project new record into 2D PCA space
        pca_coord = self.pca.transform(x_scaled)[0]

        meta = {
            "assigned_cluster": assigned_cluster,
            "distance_to_centroid": round(min_distance, 3),
            "distances_all": {int(c): round(float(d), 3) for c, d in enumerate(distances)},
            "similarity_percentages": similarity_dict,
            "pca_coord": [round(float(pca_coord[0]), 3), round(float(pca_coord[1]), 3)]
        }

        return assigned_cluster, similarity_dict, min_distance, meta
