"""
Data loading, validation, and data quality inspection module.
"""

import os
from typing import Tuple, Dict, Any, Optional, Union
import io
import pandas as pd
import numpy as np

from src.config import (
    ALL_CLUSTERING_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_DEFINITIONS,
    DATASET_METADATA
)


def get_default_data_path() -> str:
    """Return the absolute path to the default UCI heart disease CSV."""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(current_dir, "data", "heart_disease_uci.csv")


def load_dataset(
    source: Optional[Union[str, io.BytesIO, io.StringIO]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load dataset from file path or uploaded file buffer.
    
    Returns:
        df: pd.DataFrame with anonymous record_id column
        report: Dict containing metadata, validation flags, and statistics
    """
    source_name = "Default UCI Cleveland Dataset"
    
    if source is None:
        source = get_default_data_path()
        if not os.path.exists(source):
            raise FileNotFoundError(f"Default dataset not found at {source}")
        source_name = os.path.basename(source)
        df_raw = pd.read_csv(source)
    elif isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found at {source}")
        source_name = os.path.basename(source)
        df_raw = pd.read_csv(source)
    else:
        # Uploaded file object
        source_name = getattr(source, "name", "Uploaded CSV File")
        df_raw = pd.read_csv(source)

    # Clean whitespace in column names and convert to lowercase
    df_raw.columns = [str(c).strip().lower() for c in df_raw.columns]

    # Validate and inspect
    df, validation_report = validate_and_structure_dataset(df_raw, source_name)
    return df, validation_report


def validate_and_structure_dataset(
    df: pd.DataFrame, source_name: str = "Dataset"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate columns, audit missing data and duplicates, and assign anonymous record IDs.
    """
    raw_rows, raw_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())

    # Identify existing clustering features
    present_clustering_features = [
        col for col in ALL_CLUSTERING_FEATURES if col in df.columns
    ]
    missing_required_features = [
        col for col in ALL_CLUSTERING_FEATURES if col not in df.columns
    ]
    
    # Identify target or excluded columns
    present_excluded_columns = [
        col for col in df.columns if col in EXCLUDED_COLUMNS
    ]

    # Check minimum validity: needs at least age and a few clinical features
    is_valid = len(present_clustering_features) >= 4 and ("age" in df.columns or "trestbps" in df.columns)

    # Missing value audit
    missing_counts = df[present_clustering_features].isnull().sum().to_dict()
    missing_pct = {
        col: float((count / raw_rows) * 100) if raw_rows > 0 else 0.0
        for col, count in missing_counts.items()
    }
    total_cells = raw_rows * len(present_clustering_features)
    total_missing_cells = sum(missing_counts.values())
    completeness_pct = (
        100.0 - (total_missing_cells / total_cells * 100.0)
        if total_cells > 0 else 100.0
    )

    # Create a clean working copy
    df_clean = df.copy()

    # Assign clear, anonymous Record IDs if not present
    if "record_id" not in df_clean.columns:
        df_clean.insert(0, "record_id", [f"REC-{i+1:03d}" for i in range(len(df_clean))])

    # Basic summary statistics for numeric features
    present_numeric = [c for c in NUMERICAL_FEATURES if c in df_clean.columns]
    desc_stats = {}
    if present_numeric:
        desc_df = df_clean[present_numeric].describe().T
        desc_stats = desc_df[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].to_dict(orient="index")

    report = {
        "source_name": source_name,
        "is_valid": is_valid,
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "duplicate_rows": duplicate_rows,
        "present_clustering_features": present_clustering_features,
        "missing_required_features": missing_required_features,
        "present_excluded_columns": present_excluded_columns,
        "missing_counts": missing_counts,
        "missing_pct": missing_pct,
        "total_missing_cells": total_missing_cells,
        "completeness_pct": round(completeness_pct, 2),
        "desc_stats": desc_stats,
        "data_types": {col: str(df_clean[col].dtype) for col in df_clean.columns},
        "is_synthetic": False
    }

    return df_clean, report


def generate_synthetic_demo_dataset(n_samples: int = 150, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate synthetic cardiac patient data clearly labeled as synthetic for demo purposes.
    Maintains realistic clinical boundaries matching the Cleveland dataset format.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(55, 9, n_samples).clip(29, 77).round()
    sex = rng.choice([0.0, 1.0], size=n_samples, p=[0.32, 0.68])
    cp = rng.choice([1.0, 2.0, 3.0, 4.0], size=n_samples, p=[0.08, 0.17, 0.28, 0.47])
    trestbps = rng.normal(131, 17, n_samples).clip(94, 200).round()
    chol = rng.normal(246, 51, n_samples).clip(126, 450).round()
    fbs = rng.choice([0.0, 1.0], size=n_samples, p=[0.85, 0.15])
    restecg = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=[0.50, 0.05, 0.45])
    thalach = rng.normal(149, 23, n_samples).clip(71, 202).round()
    exang = rng.choice([0.0, 1.0], size=n_samples, p=[0.67, 0.33])
    oldpeak = rng.exponential(1.0, n_samples).clip(0.0, 6.0).round(1)
    slope = rng.choice([1.0, 2.0, 3.0], size=n_samples, p=[0.47, 0.46, 0.07])
    ca = rng.choice([0.0, 1.0, 2.0, 3.0], size=n_samples, p=[0.59, 0.22, 0.13, 0.06])
    thal = rng.choice([3.0, 6.0, 7.0], size=n_samples, p=[0.55, 0.06, 0.39])
    target = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.54, 0.18, 0.12, 0.16])

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target
    })

    df, report = validate_and_structure_dataset(df, "Synthetic Demo Records (Clearly Labeled Demo)")
    report["is_synthetic"] = True
    return df, report
