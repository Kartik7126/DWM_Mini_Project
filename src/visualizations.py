"""
Plotly visualization components adhering to clinical analytics styling standards.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import THEME_COLORS, FEATURE_DEFINITIONS, NUMERICAL_FEATURES


def _apply_clinical_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply consistent healthcare theme styling to a Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 15, "color": THEME_COLORS["primary_dark"], "family": "Inter, Arial, sans-serif"},
            "x": 0.02,
            "xanchor": "left"
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=height,
        margin={"l": 45, "r": 30, "t": 45 if title else 25, "b": 45},
        font={"family": "Inter, Arial, sans-serif", "color": THEME_COLORS["text"], "size": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": THEME_COLORS["border"],
            "borderwidth": 1
        },
        hoverlabel={"bgcolor": "#FFFFFF", "font_size": 12, "font_family": "Inter, Arial, sans-serif"}
    )
    fig.update_xaxes(
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
        tickfont={"size": 11}
    )
    fig.update_yaxes(
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        zerolinecolor="#CBD5E1",
        tickfont={"size": 11}
    )
    return fig


def plot_cluster_distribution(df_clustered: pd.DataFrame) -> go.Figure:
    """Plot patient count and percentage across discovered clusters."""
    counts = df_clustered["cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["cluster", "count"]
    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(1)
    counts["cluster_name"] = counts["cluster"].apply(lambda c: f"Cluster {c}")

    colors = [THEME_COLORS["cluster_palette"][i % len(THEME_COLORS["cluster_palette"])] for i in counts["cluster"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts["cluster_name"],
        y=counts["count"],
        text=[f"{cnt} ({p}%)" for cnt, p in zip(counts["count"], counts["pct"])],
        textposition="outside",
        marker={"color": colors, "line": {"width": 1, "color": "#FFFFFF"}},
        hovertemplate="<b>%{x}</b><br>Patients: %{y}<br>Proportion: %{text}<extra></extra>"
    ))

    fig.update_yaxes(range=[0, max(counts["count"]) * 1.25], title="Number of Patients")
    fig.update_xaxes(title="Cluster Profile")
    return _apply_clinical_layout(fig, "Patient Distribution Across Clusters", height=380)


def plot_cluster_proportions_pie(df_clustered: pd.DataFrame) -> go.Figure:
    """Donut chart showing proportional representation of clusters."""
    counts = df_clustered["cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["cluster", "count"]
    labels = [f"Cluster {c}" for c in counts["cluster"]]
    colors = [THEME_COLORS["cluster_palette"][i % len(THEME_COLORS["cluster_palette"])] for i in counts["cluster"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts["count"],
        hole=0.45,
        marker={"colors": colors, "line": {"color": "#FFFFFF", "width": 2}},
        textinfo="percent+label",
        hoverinfo="label+value+percent"
    )])
    return _apply_clinical_layout(fig, "Cohort Cluster Proportions", height=380)


def plot_age_distribution_by_cluster(df_clustered: pd.DataFrame) -> go.Figure:
    """Box plot showing age distributions across clusters."""
    if "age" not in df_clustered.columns:
        return go.Figure()

    fig = go.Figure()
    clusters = sorted(df_clustered["cluster"].unique())

    for c in clusters:
        sub = df_clustered[df_clustered["cluster"] == c]
        color = THEME_COLORS["cluster_palette"][c % len(THEME_COLORS["cluster_palette"])]
        fig.add_trace(go.Box(
            y=sub["age"],
            name=f"Cluster {c}",
            marker_color=color,
            boxpoints="outliers",
            jitter=0.2,
            pointpos=-1.8,
            hovertemplate="<b>Cluster " + str(c) + "</b><br>Age: %{y} years<extra></extra>"
        ))

    fig.update_yaxes(title="Age (years)")
    fig.update_xaxes(title="Cluster")
    return _apply_clinical_layout(fig, "Age Distribution by Cluster", height=380)


def plot_pca_scatter(
    pca_coords: np.ndarray,
    cluster_labels: np.ndarray,
    pca_centroids: Optional[np.ndarray] = None,
    explained_variance: Optional[List[float]] = None,
    highlight_point: Optional[List[float]] = None,
    highlight_label: str = "Selected Patient"
) -> go.Figure:
    """2D PCA projection scatter plot colored by cluster membership."""
    fig = go.Figure()
    unique_clusters = sorted(np.unique(cluster_labels))

    for c in unique_clusters:
        idx = np.where(cluster_labels == c)[0]
        color = THEME_COLORS["cluster_palette"][c % len(THEME_COLORS["cluster_palette"])]
        fig.add_trace(go.Scatter(
            x=pca_coords[idx, 0],
            y=pca_coords[idx, 1],
            mode="markers",
            name=f"Cluster {c}",
            marker={"color": color, "size": 7, "opacity": 0.75, "line": {"width": 0.5, "color": "#FFFFFF"}},
            hovertemplate=f"<b>Cluster {c}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
        ))

    # Add centroids
    if pca_centroids is not None:
        fig.add_trace(go.Scatter(
            x=pca_centroids[:, 0],
            y=pca_centroids[:, 1],
            mode="markers+text",
            name="Centroids",
            text=[f"C{i}" for i in range(len(pca_centroids))],
            textposition="top center",
            marker={"symbol": "diamond-tall", "size": 13, "color": THEME_COLORS["primary_dark"], "line": {"width": 1.5, "color": "#FFFFFF"}},
            hovertemplate="<b>Centroid %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
        ))

    # Highlight point (e.g. analyzed patient)
    if highlight_point is not None:
        fig.add_trace(go.Scatter(
            x=[highlight_point[0]],
            y=[highlight_point[1]],
            mode="markers+text",
            name=highlight_label,
            text=[highlight_label],
            textposition="bottom center",
            marker={"symbol": "star", "size": 16, "color": THEME_COLORS["danger"], "line": {"width": 1.5, "color": "#FFFFFF"}},
            hovertemplate=f"<b>{highlight_label}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
        ))

    x_title = "Principal Component 1"
    y_title = "Principal Component 2"
    if explained_variance is not None and len(explained_variance) >= 2:
        x_title += f" ({explained_variance[0]*100:.1f}% var)"
        y_title += f" ({explained_variance[1]*100:.1f}% var)"

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    return _apply_clinical_layout(fig, "2D PCA Projection of Patient Clusters", height=430)


def plot_clinical_feature_comparison_bars(
    df_clustered: pd.DataFrame, features: Optional[List[str]] = None
) -> go.Figure:
    """Grouped bar chart comparing key clinical feature averages across clusters."""
    if features is None:
        features = ["trestbps", "chol", "thalach", "oldpeak"]
    features = [f for f in features if f in df_clustered.columns]

    clusters = sorted(df_clustered["cluster"].unique())
    fig = go.Figure()

    for c in clusters:
        sub = df_clustered[df_clustered["cluster"] == c]
        means = [sub[f].mean() for f in features]
        labels = [FEATURE_DEFINITIONS.get(f, {}).get("display_name", f) for f in features]
        color = THEME_COLORS["cluster_palette"][c % len(THEME_COLORS["cluster_palette"])]

        fig.add_trace(go.Bar(
            name=f"Cluster {c}",
            x=labels,
            y=means,
            marker_color=color,
            hovertemplate="<b>Cluster " + str(c) + "</b><br>%{x}: %{y:.1f}<extra></extra>"
        ))

    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Clinical Feature")
    fig.update_yaxes(title="Mean Value (Native Units)")
    return _apply_clinical_layout(fig, "Clinical Feature Comparison Across Clusters", height=400)


def plot_elbow_curve(
    evaluation_results: Dict[int, Dict[str, Any]], selected_k: int
) -> go.Figure:
    """Plot K vs Inertia (Elbow Method) with selected K marker."""
    k_vals = sorted(evaluation_results.keys())
    inertias = [evaluation_results[k]["inertia"] for k in k_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=inertias,
        mode="lines+markers",
        name="Inertia (SSE)",
        line={"color": THEME_COLORS["secondary"], "width": 3},
        marker={"size": 8, "color": THEME_COLORS["secondary"]},
        hovertemplate="<b>K = %{x}</b><br>Inertia: %{y:.1f}<extra></extra>"
    ))

    # Highlight selected K
    if selected_k in evaluation_results:
        fig.add_trace(go.Scatter(
            x=[selected_k],
            y=[evaluation_results[selected_k]["inertia"]],
            mode="markers",
            name=f"Selected K ({selected_k})",
            marker={"size": 14, "symbol": "circle", "color": THEME_COLORS["danger"], "line": {"width": 2, "color": "#FFFFFF"}},
            hovertemplate=f"<b>Selected K = {selected_k}</b><br>Inertia: %{{y:.1f}}<extra></extra>"
        ))

    fig.update_xaxes(title="Number of Clusters (K)", dtick=1)
    fig.update_yaxes(title="Within-Cluster Sum of Squares (Inertia)")
    return _apply_clinical_layout(fig, "Elbow Method: Inertia vs K", height=380)


def plot_silhouette_curve(
    evaluation_results: Dict[int, Dict[str, Any]], selected_k: int
) -> go.Figure:
    """Plot K vs Silhouette Score."""
    k_vals = sorted(evaluation_results.keys())
    sil_scores = [evaluation_results[k]["silhouette_score"] for k in k_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=sil_scores,
        mode="lines+markers",
        name="Silhouette Score",
        line={"color": THEME_COLORS["accent"], "width": 3},
        marker={"size": 8, "color": THEME_COLORS["accent"]},
        hovertemplate="<b>K = %{x}</b><br>Silhouette Score: %{y:.4f}<extra></extra>"
    ))

    # Highlight selected K
    if selected_k in evaluation_results:
        fig.add_trace(go.Scatter(
            x=[selected_k],
            y=[evaluation_results[selected_k]["silhouette_score"]],
            mode="markers",
            name=f"Selected K ({selected_k})",
            marker={"size": 14, "symbol": "circle", "color": THEME_COLORS["danger"], "line": {"width": 2, "color": "#FFFFFF"}},
            hovertemplate=f"<b>Selected K = {selected_k}</b><br>Score: %{{y:.4f}}<extra></extra>"
        ))

    fig.update_xaxes(title="Number of Clusters (K)", dtick=1)
    fig.update_yaxes(title="Mean Silhouette Coefficient")
    return _apply_clinical_layout(fig, "Silhouette Score by Number of Clusters (K)", height=380)


def plot_cluster_feature_heatmap(profiles: Dict[int, Dict[str, Any]]) -> go.Figure:
    """Heatmap of standardized feature deviations (Z-diff) across clusters."""
    clusters = sorted(profiles.keys())
    first_profile = profiles[clusters[0]]
    features = list(first_profile["numerical_stats"].keys())

    z_matrix = []
    y_labels = [f"Cluster {c}" for c in clusters]
    x_labels = [FEATURE_DEFINITIONS.get(f, {}).get("display_name", f) for f in features]

    for c in clusters:
        row = [profiles[c]["numerical_stats"][f]["z_diff"] for f in features]
        z_matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=x_labels,
        y=y_labels,
        colorscale="RdBu_r",
        zmid=0.0,
        text=np.round(z_matrix, 2),
        texttemplate="%{text}",
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f} std dev<extra></extra>"
    ))

    fig.update_xaxes(tickangle=-25)
    return _apply_clinical_layout(fig, "Standardized Feature Deviation Heatmap (Cohort Z-Scores)", height=360)


def plot_patient_vs_cluster_radar(
    patient_record: Dict[str, Any],
    cluster_profile: Dict[str, Any],
    training_stats: Dict[str, Dict[str, float]]
) -> go.Figure:
    """
    Radar chart comparing a patient's normalized metrics (0 to 1) against
    the cluster mean and the global cohort average.
    """
    num_stats = cluster_profile["numerical_stats"]
    features = [f for f in num_stats.keys() if f in training_stats and f in patient_record]

    if not features:
        return go.Figure()

    labels = [FEATURE_DEFINITIONS.get(f, {}).get("display_name", f) for f in features]

    p_norm, c_norm, pop_norm = [], [], []

    for f in features:
        min_v = training_stats[f]["min"]
        max_v = training_stats[f]["max"]
        span = (max_v - min_v) if max_v > min_v else 1.0

        p_val = float(patient_record[f])
        c_val = float(num_stats[f]["mean"])
        pop_val = float(num_stats[f]["pop_mean"])

        p_norm.append(np.clip((p_val - min_v) / span, 0.0, 1.0))
        c_norm.append(np.clip((c_val - min_v) / span, 0.0, 1.0))
        pop_norm.append(np.clip((pop_val - min_v) / span, 0.0, 1.0))

    # Close loop
    labels_closed = labels + [labels[0]]
    p_norm_closed = p_norm + [p_norm[0]]
    c_norm_closed = c_norm + [c_norm[0]]
    pop_norm_closed = pop_norm + [pop_norm[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=p_norm_closed,
        theta=labels_closed,
        fill="toself",
        name="Analyzed Patient",
        line_color=THEME_COLORS["danger"],
        fillcolor="rgba(225, 29, 72, 0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=c_norm_closed,
        theta=labels_closed,
        fill="toself",
        name=f"Cluster {cluster_profile['cluster_id']} Average",
        line_color=THEME_COLORS["primary"],
        fillcolor="rgba(15, 118, 110, 0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=pop_norm_closed,
        theta=labels_closed,
        name="Cohort Benchmark (Population Mean)",
        line_color=THEME_COLORS["text_muted"],
        line={"dash": "dot"}
    ))

    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 1], "tickfont": {"size": 9}},
            "bgcolor": "#F8FAFC"
        }
    )
    return _apply_clinical_layout(fig, "Multivariate Normalized Profile (Radar View)", height=430)


def plot_patient_delta_bars(comparison_items: List[Dict[str, Any]]) -> go.Figure:
    """Horizontal diverging bar chart showing patient percentage deviation from cluster mean."""
    labels = [item["display_name"] for item in comparison_items]
    pcts = [item["pct_difference"] for item in comparison_items]
    colors = [
        THEME_COLORS["danger"] if p > 0 else THEME_COLORS["secondary"]
        for p in pcts
    ]

    fig = go.Figure(go.Bar(
        y=labels,
        x=pcts,
        orientation="h",
        marker_color=colors,
        text=[f"{p:+.1f}%" for p in pcts],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Difference from Cluster Average: %{x:+.1f}%<extra></extra>"
    ))

    fig.update_xaxes(title="Percentage Deviation from Cluster Mean (%)")
    fig.update_yaxes(autorange="reversed")
    return _apply_clinical_layout(fig, "Patient Variance from Cluster Mean", height=380)


def plot_similarity_to_all_clusters(similarity_percentages: Dict[int, float]) -> go.Figure:
    """Bar chart showing relative similarity percentages across all clusters."""
    c_labels = [f"Cluster {c}" for c in similarity_percentages.keys()]
    pcts = list(similarity_percentages.values())
    colors = [THEME_COLORS["cluster_palette"][c % len(THEME_COLORS["cluster_palette"])] for c in similarity_percentages.keys()]

    fig = go.Figure(go.Bar(
        x=pcts,
        y=c_labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Similarity Score: %{x:.1f}%<extra></extra>"
    ))

    fig.update_xaxes(range=[0, max(pcts) * 1.25], title="Relative Similarity Profile (%)")
    fig.update_yaxes(autorange="reversed")
    return _apply_clinical_layout(fig, "Similarity Distribution Across Cohort Clusters", height=280)


def plot_simple_patient_comparison(
    patient_record: Dict[str, Any],
    cluster_profile: Dict[str, Any]
) -> go.Figure:
    """
    Simple, intuitive horizontal comparison bar chart showing how a patient's
    key vitals compare directly against the cluster average.
    100% represents the cluster average baseline.
    """
    num_stats = cluster_profile.get("numerical_stats", {})
    features = [f for f in ["trestbps", "chol", "thalach", "oldpeak", "age"] if f in num_stats and f in patient_record]

    labels = []
    ratios = []
    bar_texts = []
    colors = []

    for f in features:
        disp_name = FEATURE_DEFINITIONS.get(f, {}).get("display_name", f)
        unit = FEATURE_DEFINITIONS.get(f, {}).get("unit", "")
        p_val = float(patient_record[f])
        c_mean = float(num_stats[f]["mean"])

        ratio = (p_val / c_mean * 100.0) if c_mean > 1e-4 else 100.0
        ratios.append(round(ratio, 1))
        labels.append(disp_name)

        diff = p_val - c_mean
        diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
        bar_texts.append(f"Patient: {p_val:.1f} {unit}  (Cluster Avg: {c_mean:.1f} {unit} | {diff_str})")

        if ratio > 105.0:
            colors.append("#E11D48")  # Coral/Red (Higher than cluster average)
        elif ratio < 95.0:
            colors.append("#0284C7")  # Medical Blue (Lower than cluster average)
        else:
            colors.append("#0D9488")  # Teal (Matched to cluster average)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,
        x=ratios,
        orientation="h",
        marker_color=colors,
        text=bar_texts,
        textposition="inside",
        insidetextanchor="start",
        textfont={"size": 12, "color": "#FFFFFF"},
        hovertemplate="<b>%{y}</b><br>%{text}<br>Level vs Cluster Avg: %{x:.1f}%<extra></extra>"
    ))

    # Reference line at 100% (Cluster Average Baseline)
    fig.add_vline(
        x=100.0,
        line_dash="dash",
        line_color="#0F172A",
        line_width=2,
        annotation_text="100% Cluster Average Baseline",
        annotation_position="top right"
    )

    max_x = max(max(ratios) * 1.25, 130.0) if ratios else 130.0
    fig.update_xaxes(
        title="Patient Level Relative to Cluster Average (%)  [100% = Cluster Mean]",
        range=[0, max_x]
    )
    fig.update_yaxes(autorange="reversed")

    c_id = cluster_profile.get("cluster_id", "")
    return _apply_clinical_layout(fig, f"Patient Vitals vs Cluster {c_id} Average", height=320)


def plot_hospital_bed_allocation(resource_data: Dict[str, Any]) -> go.Figure:
    """Stacked bar chart of hospital bed requirements by cluster."""
    clusters = sorted(resource_data["clusters"].keys())
    c_labels = [f"Cluster {c}" for c in clusters]

    ccu_beds = [resource_data["clusters"][c]["bed_allocation"]["ccu_beds"] for c in clusters]
    telemetry_beds = [resource_data["clusters"][c]["bed_allocation"]["telemetry_beds"] for c in clusters]
    ward_beds = [resource_data["clusters"][c]["bed_allocation"]["ward_beds"] for c in clusters]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="CCU / ICU Beds",
        x=c_labels,
        y=ccu_beds,
        marker_color="#E11D48",  # Rose-600
        text=[f"{v} beds" if v > 0 else "" for v in ccu_beds],
        textposition="inside",
        hovertemplate="<b>%{x}</b><br>CCU/ICU Beds: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name="Telemetry Step-Down",
        x=c_labels,
        y=telemetry_beds,
        marker_color="#D97706",  # Amber-600
        text=[f"{v} beds" if v > 0 else "" for v in telemetry_beds],
        textposition="inside",
        hovertemplate="<b>%{x}</b><br>Telemetry Beds: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name="General Cardiology Ward",
        x=c_labels,
        y=ward_beds,
        marker_color="#0D9488",  # Teal-600
        text=[f"{v} beds" if v > 0 else "" for v in ward_beds],
        textposition="inside",
        hovertemplate="<b>%{x}</b><br>Ward Beds: %{y}<extra></extra>"
    ))

    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Patient Phenotype Cluster")
    fig.update_yaxes(title="Inpatient Beds Required")
    return _apply_clinical_layout(fig, "Hospital Inpatient Bed Allocation by Cluster", height=380)


def plot_diagnostic_procedural_load(resource_data: Dict[str, Any]) -> go.Figure:
    """Grouped bar chart displaying diagnostic & procedural facility load."""
    clusters = sorted(resource_data["clusters"].keys())
    c_labels = [f"Cluster {c}" for c in clusters]

    cath_slots = [resource_data["clusters"][c]["diagnostic_slots"]["cath_lab_slots"] for c in clusters]
    echo_slots = [resource_data["clusters"][c]["diagnostic_slots"]["echo_slots"] for c in clusters]
    spect_slots = [resource_data["clusters"][c]["diagnostic_slots"]["spect_ct_slots"] for c in clusters]
    holter_slots = [resource_data["clusters"][c]["diagnostic_slots"]["holter_slots"] for c in clusters]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Cath Lab Angiography", x=c_labels, y=cath_slots, marker_color="#DC2626"))
    fig.add_trace(go.Bar(name="Echocardiography (TTE)", x=c_labels, y=echo_slots, marker_color="#0284C7"))
    fig.add_trace(go.Bar(name="Nuclear SPECT / CT", x=c_labels, y=spect_slots, marker_color="#8B5CF6"))
    fig.add_trace(go.Bar(name="24-hr Holter ECG", x=c_labels, y=holter_slots, marker_color="#10B981"))

    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Patient Phenotype Cluster")
    fig.update_yaxes(title="Estimated Diagnostic Procedure Slots")
    return _apply_clinical_layout(fig, "Diagnostic & Interventional Facility Demand", height=380)


def plot_cluster_risk_radar(profiles: Dict[int, Dict[str, Any]]) -> go.Figure:
    """Comparative polar radar chart displaying core clinical risk factor axes."""
    categories = [
        "Ischemic Stress (ST/Angina)",
        "Hypertension (Stage 2)",
        "Coronary Vessels (1+)",
        "Perfusion Defect",
        "Metabolic/Cholesterol"
    ]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    clusters = sorted(profiles.keys())

    for c in clusters:
        risk = profiles[c].get("risk_profile", {}).get("risk_indicators", {})
        if not risk:
            continue

        r_values = [
            risk.get("ischemic_burden_pct", 0.0),
            risk.get("htn_stage2_pct", 0.0),
            risk.get("vessels_1plus_pct", 0.0),
            risk.get("perfusion_defect_pct", 0.0),
            risk.get("metabolic_pct", 0.0)
        ]
        r_values_closed = r_values + [r_values[0]]
        color = THEME_COLORS["cluster_palette"][c % len(THEME_COLORS["cluster_palette"])]

        fig.add_trace(go.Scatterpolar(
            r=r_values_closed,
            theta=categories_closed,
            name=f"Cluster {c}",
            line_color=color,
            fill="toself",
            fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.12)"
        ))

    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 9}},
            "bgcolor": "#F8FAFC"
        }
    )
    return _apply_clinical_layout(fig, "Cardiovascular Risk Factor Profiles Across Clusters", height=420)


def plot_triage_acuity_donut(resource_data: Dict[str, Any]) -> go.Figure:
    """Donut chart illustrating cohort volume across Clinical Acuity Tiers."""
    tier_counts = {"Critical (CCU / Immediate)": 0, "Intermediate (Telemetry)": 0, "Ambulatory / Prevention": 0}

    for c, c_info in resource_data["clusters"].items():
        tier_key = c_info["risk_profile"].get("acuity_tier_key", "TIER_2")
        count = c_info["patient_count"]
        if tier_key == "TIER_1":
            tier_counts["Critical (CCU / Immediate)"] += count
        elif tier_key == "TIER_2":
            tier_counts["Intermediate (Telemetry)"] += count
        else:
            tier_counts["Ambulatory / Prevention"] += count

    labels = list(tier_counts.keys())
    values = list(tier_counts.values())
    colors = ["#E11D48", "#D97706", "#0D9488"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker={"colors": colors, "line": {"color": "#FFFFFF", "width": 2}},
        textinfo="percent+label",
        hoverinfo="label+value+percent"
    )])
    return _apply_clinical_layout(fig, "Cohort Clinical Acuity Breakdown", height=380)
