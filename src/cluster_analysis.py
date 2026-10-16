"""
Cluster profiling, clinical feature interpretation, and comparative analytics module.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    FEATURE_DEFINITIONS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    THEME_COLORS
)


def generate_cluster_profiles(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_cols: Optional[List[str]] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Generate comprehensive statistical profiles and neutral, feature-based clinical
    interpretations for each discovered K-Means cluster.
    """
    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels
    total_records = len(df_clustered)

    if feature_cols is None:
        feature_cols = [c for c in FEATURE_DEFINITIONS.keys() if c in df_clustered.columns]

    numeric_cols = [c for c in feature_cols if c in NUMERICAL_FEATURES]
    categorical_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]

    # Global population statistics
    global_stats = {}
    for col in numeric_cols:
        series = pd.to_numeric(df_clustered[col], errors='coerce').dropna()
        global_stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 1.0,
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max())
        }

    unique_clusters = sorted(np.unique(cluster_labels))
    profiles = {}

    for c_id in unique_clusters:
        sub_df = df_clustered[df_clustered["cluster"] == c_id]
        c_count = len(sub_df)
        c_pct = (c_count / total_records) * 100.0 if total_records > 0 else 0.0

        # Numeric feature statistics
        num_stats = {}
        distinguishing_features = []

        for col in numeric_cols:
            series = pd.to_numeric(sub_df[col], errors='coerce').dropna()
            if len(series) > 0:
                c_mean = float(series.mean())
                c_med = float(series.median())
                c_std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
                g_mean = global_stats[col]["mean"]
                g_std = global_stats[col]["std"] if global_stats[col]["std"] > 1e-6 else 1.0

                z_diff = (c_mean - g_mean) / g_std
                pct_diff = ((c_mean - g_mean) / g_mean * 100.0) if abs(g_mean) > 1e-6 else 0.0

                disp_name = FEATURE_DEFINITIONS.get(col, {}).get("display_name", col)
                unit = FEATURE_DEFINITIONS.get(col, {}).get("unit", "")

                num_stats[col] = {
                    "mean": round(c_mean, 2),
                    "median": round(c_med, 2),
                    "std": round(c_std, 2),
                    "pop_mean": round(g_mean, 2),
                    "pop_median": round(global_stats[col]["median"], 2),
                    "pct_diff": round(pct_diff, 1),
                    "z_diff": round(z_diff, 2),
                    "display_name": disp_name,
                    "unit": unit
                }

                # Highlighting significant distinguishing features (|z_diff| >= 0.35)
                if abs(z_diff) >= 0.35:
                    direction = "higher" if z_diff > 0 else "lower"
                    distinguishing_features.append({
                        "feature": col,
                        "display_name": disp_name,
                        "direction": direction,
                        "z_diff": round(z_diff, 2),
                        "c_mean": round(c_mean, 1),
                        "pop_mean": round(g_mean, 1),
                        "unit": unit,
                        "summary": f"{disp_name} is {abs(pct_diff):.1f}% {direction} than cohort average ({c_mean:.1f} vs {g_mean:.1f} {unit})"
                    })

        # Sort distinguishing features by absolute deviation
        distinguishing_features.sort(key=lambda x: abs(x["z_diff"]), reverse=True)

        # Categorical feature distributions
        cat_stats = {}
        for col in categorical_cols:
            counts = sub_df[col].value_counts(dropna=False).to_dict()
            options = FEATURE_DEFINITIONS.get(col, {}).get("options", {})
            cat_dist = {}
            for val, cnt in counts.items():
                label = options.get(val, f"Category {val}")
                pct = round((cnt / c_count) * 100.0, 1)
                cat_dist[label] = {"count": int(cnt), "percentage": pct}
            cat_stats[col] = cat_dist

        # Generate neutral, human-readable feature-based cluster title & summary
        title, narrative = _generate_neutral_cluster_narrative(
            c_id, c_count, c_pct, num_stats, distinguishing_features, cat_stats
        )

        profiles[int(c_id)] = {
            "cluster_id": int(c_id),
            "record_count": c_count,
            "percentage": round(c_pct, 1),
            "title": title,
            "narrative": narrative,
            "distinguishing_features": distinguishing_features,
            "numerical_stats": num_stats,
            "categorical_stats": cat_stats
        }

    return profiles


def _generate_neutral_cluster_narrative(
    cluster_id: int,
    count: int,
    pct: float,
    num_stats: Dict[str, Any],
    dist_features: List[Dict[str, Any]],
    cat_stats: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Generate neutral, strictly descriptive labels based on empirical feature patterns.
    Avoids clinical diagnostics like 'healthy' or 'high-risk'.
    """
    # Key descriptor candidates
    descriptors = []
    for df in dist_features[:3]:
        feat = df["feature"]
        dir_word = "higher" if df["direction"] == "higher" else "comparatively lower"
        if feat == "age":
            descriptors.append("older age profile" if df["direction"] == "higher" else "younger age profile")
        elif feat == "thalach":
            descriptors.append(f"{dir_word} peak exercise heart rate")
        elif feat == "trestbps":
            descriptors.append(f"{dir_word} resting blood pressure")
        elif feat == "chol":
            descriptors.append(f"{dir_word} serum cholesterol")
        elif feat == "oldpeak":
            descriptors.append(f"{dir_word} ST depression")
        elif feat == "ca":
            descriptors.append(f"{dir_word} major vessels visualized")
        else:
            descriptors.append(f"{dir_word} {df['display_name'].lower()}")

    if descriptors:
        title = f"Cluster {cluster_id}: Patient Cohort with " + " and ".join(descriptors[:2])
    else:
        title = f"Cluster {cluster_id}: Patient Cohort with Average Clinical Indicators"

    # Narrative paragraph
    bullet_points = [f"- **Size**: {count} patients ({pct:.1f}% of loaded cohort)"]
    if "age" in num_stats:
        bullet_points.append(f"- **Age Profile**: Mean age {num_stats['age']['mean']} years (median: {num_stats['age']['median']} yrs)")
    
    if dist_features:
        lead_insights = ", ".join([df["summary"] for df in dist_features[:3]])
        bullet_points.append(f"- **Key Characteristics**: {lead_insights}")
    else:
        bullet_points.append("- **Key Characteristics**: Features in this group closely align with overall cohort medians.")

    narrative = "\n".join(bullet_points)
    return title, narrative


def compare_patient_to_cluster(
    patient_record: Dict[str, Any],
    cluster_profile: Dict[str, Any],
    global_stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Any]:
    """
    Compare an individual patient's metrics against their assigned cluster's mean and median.
    """
    num_stats = cluster_profile["numerical_stats"]
    comparison_items = []

    for feat, stats in num_stats.items():
        if feat in patient_record and patient_record[feat] is not None:
            p_val = float(patient_record[feat])
            c_mean = stats["mean"]
            c_median = stats["median"]
            unit = stats["unit"]
            disp_name = stats["display_name"]

            diff_mean = p_val - c_mean
            pct_diff = (diff_mean / c_mean * 100.0) if abs(c_mean) > 1e-6 else 0.0

            status = "Above Average" if diff_mean > 0 else ("Below Average" if diff_mean < 0 else "Equal to Average")

            comparison_items.append({
                "feature": feat,
                "display_name": disp_name,
                "patient_value": p_val,
                "cluster_mean": c_mean,
                "cluster_median": c_median,
                "difference": round(diff_mean, 2),
                "pct_difference": round(pct_diff, 1),
                "unit": unit,
                "status": status
            })

    return {
        "cluster_id": cluster_profile["cluster_id"],
        "cluster_title": cluster_profile["title"],
        "comparison_items": comparison_items
    }


def compare_two_clusters(
    profile_a: Dict[str, Any], profile_b: Dict[str, Any]
) -> pd.DataFrame:
    """
    Produce a side-by-side comparison dataframe for two selected clusters.
    """
    stats_a = profile_a["numerical_stats"]
    stats_b = profile_b["numerical_stats"]

    rows = []
    for feat in stats_a.keys():
        if feat in stats_b:
            disp_name = stats_a[feat]["display_name"]
            unit = stats_a[feat]["unit"]
            mean_a = stats_a[feat]["mean"]
            mean_b = stats_b[feat]["mean"]
            med_a = stats_a[feat]["median"]
            med_b = stats_b[feat]["median"]
            diff = mean_a - mean_b
            higher = f"Cluster {profile_a['cluster_id']}" if diff > 0 else (f"Cluster {profile_b['cluster_id']}" if diff < 0 else "Equal")

            rows.append({
                "Feature": disp_name,
                "Unit": unit,
                f"Cluster {profile_a['cluster_id']} Mean": mean_a,
                f"Cluster {profile_a['cluster_id']} Median": med_a,
                f"Cluster {profile_b['cluster_id']} Mean": mean_b,
                f"Cluster {profile_b['cluster_id']} Median": med_b,
                "Absolute Diff": round(abs(diff), 2),
                "Higher Cohort": higher
            })

    return pd.DataFrame(rows)
