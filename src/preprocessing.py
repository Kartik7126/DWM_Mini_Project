"""
Data preprocessing, feature engineering, imputation, and standardization pipeline.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_CLUSTERING_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_DEFINITIONS
)


class CardiacPreprocessor:
    """
    Stateful preprocessor for cardiac patient data.
    Ensures zero data leakage by fitting transformers strictly on training data
    and applying identical transformations to new patient records.
    """

    def __init__(self, feature_subset: Optional[List[str]] = None):
        self.feature_subset = feature_subset
        self.selected_features: List[str] = []
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_out: List[str] = []
        self.training_stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def _determine_columns(self, df: pd.DataFrame) -> None:
        """Identify available numerical and categorical columns from the dataframe."""
        available_cols = [
            c for c in df.columns 
            if c not in EXCLUDED_COLUMNS and c != "record_id"
        ]

        if self.feature_subset:
            candidate_cols = [c for c in self.feature_subset if c in available_cols]
        else:
            candidate_cols = [c for c in ALL_CLUSTERING_FEATURES if c in available_cols]

        self.selected_features = candidate_cols
        self.numerical_cols = [c for c in candidate_cols if c in NUMERICAL_FEATURES]
        self.categorical_cols = [c for c in candidate_cols if c in CATEGORICAL_FEATURES]

        # Record training distribution statistics for Out-of-Distribution (OOD) checks
        self.training_stats = {}
        for col in self.numerical_cols:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(series) > 0:
                q25, q75 = np.percentile(series, [25, 75])
                iqr = q75 - q25
                self.training_stats[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=1)) if len(series) > 1 else 1.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "q25": float(q25),
                    "q75": float(q75),
                    "iqr": float(iqr),
                    "lower_bound": float(series.min() - 0.2 * (series.max() - series.min())),
                    "upper_bound": float(series.max() + 0.2 * (series.max() - series.min()))
                }

    def fit(self, df: pd.DataFrame) -> "CardiacPreprocessor":
        """Fit imputation and scaling transformers on the input dataframe."""
        self._determine_columns(df)

        transformers = []

        if self.numerical_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", num_pipe, self.numerical_cols))

        if self.categorical_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"))
            ])
            transformers.append(("cat", cat_pipe, self.categorical_cols))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        # Fit the pipeline
        self.pipeline.fit(df[self.selected_features])

        # Extract output feature names
        feature_names = []
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        if self.categorical_cols:
            cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["onehot"]
            try:
                encoded_names = cat_encoder.get_feature_names_out(self.categorical_cols)
                feature_names.extend(list(encoded_names))
            except Exception:
                feature_names.extend(self.categorical_cols)

        self.feature_names_out = feature_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe into scaled feature matrix using fitted parameters."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("CardiacPreprocessor must be fitted before calling transform().")

        # Ensure all required columns are present in input dataframe
        df_input = df.copy()
        for col in self.selected_features:
            if col not in df_input.columns:
                # Fill missing column with median or default
                if col in self.training_stats:
                    df_input[col] = self.training_stats[col]["mean"]
                else:
                    df_input[col] = FEATURE_DEFINITIONS.get(col, {}).get("default", 0.0)

        transformed = self.pipeline.transform(df_input[self.selected_features])
        return transformed

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and return transformed feature matrix."""
        self.fit(df)
        return self.transform(df)

    def check_out_of_distribution(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate if a patient record contains values significantly outside the training distribution.
        Returns a list of warning flags for any anomalous feature values.
        """
        warnings = []
        for col, stats in self.training_stats.items():
            if col in record and record[col] is not None:
                val = float(record[col])
                mean = stats["mean"]
                std = stats["std"] if stats["std"] > 1e-6 else 1.0
                z_score = abs(val - mean) / std

                display_name = FEATURE_DEFINITIONS.get(col, {}).get("display_name", col)
                unit = FEATURE_DEFINITIONS.get(col, {}).get("unit", "")

                if val < stats["min"] or val > stats["max"]:
                    severity = "high" if z_score > 3.0 else "moderate"
                    warnings.append({
                        "feature": col,
                        "display_name": display_name,
                        "value": val,
                        "unit": unit,
                        "z_score": round(z_score, 2),
                        "training_range": f"{stats['min']:.1f} – {stats['max']:.1f} {unit}".strip(),
                        "severity": severity,
                        "message": (
                            f"{display_name} ({val} {unit}) is outside the observed cohort range "
                            f"({stats['min']:.1f} to {stats['max']:.1f} {unit}, z-score: {z_score:.1f})."
                        )
                    })
        return warnings
