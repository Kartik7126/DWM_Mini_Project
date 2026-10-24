"""
Cardiac Phenotype Profiling & Hospital Resource Allocation Platform
Interactive Clinical Analytics, Risk Stratification, and Operational Capacity Planning
"""

import sys
import os
import io
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATASET_METADATA,
    FEATURE_DEFINITIONS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_CLUSTERING_FEATURES,
    THEME_COLORS,
    MEDICAL_DISCLAIMER,
    DEFAULT_K,
    DEFAULT_K_MIN,
    DEFAULT_K_MAX,
    DEFAULT_RANDOM_STATE
)
from src.data_loader import (
    load_dataset,
    generate_synthetic_demo_dataset,
    get_default_data_path
)
from src.preprocessing import CardiacPreprocessor
from src.clustering import CardiacClusteringModel
from src.cluster_analysis import (
    generate_cluster_profiles,
    compare_patient_to_cluster,
    compare_two_clusters
)
from src.resource_allocation import (
    analyze_cluster_risk_factors,
    compute_hospital_resource_allocation,
    get_hospital_care_pathway,
    get_patient_hospital_recommendation,
    ACUITY_TIERS
)
from src.visualizations import (
    plot_cluster_distribution,
    plot_cluster_proportions_pie,
    plot_age_distribution_by_cluster,
    plot_pca_scatter,
    plot_clinical_feature_comparison_bars,
    plot_elbow_curve,
    plot_silhouette_curve,
    plot_cluster_feature_heatmap,
    plot_patient_vs_cluster_radar,
    plot_patient_delta_bars,
    plot_similarity_to_all_clusters,
    plot_simple_patient_comparison,
    plot_hospital_bed_allocation,
    plot_diagnostic_procedural_load,
    plot_cluster_risk_radar,
    plot_triage_acuity_donut
)

# Set page config
st.set_page_config(
    page_title="Cardiac Phenotyping & Hospital Resource Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clinical CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background-color: #F8FAFC;
    }
    
    /* Header Container */
    .clinical-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #0D9488 100%);
        padding: 22px 30px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px -2px rgba(15, 23, 42, 0.15);
    }
    
    .clinical-header h1 {
        color: #FFFFFF !important;
        font-size: 24px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.4px;
    }
    
    .clinical-header p {
        color: #BAE6FD;
        font-size: 13.5px;
        margin: 0;
    }
    
    /* KPI Card Component */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px -1px rgba(0, 0, 0, 0.08);
    }
    
    .kpi-title {
        color: #64748B;
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .kpi-value {
        color: #0F172A;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    
    .kpi-subtitle {
        color: #0D9488;
        font-size: 12px;
        font-weight: 500;
    }
    
    /* Clinical Advisory Banner */
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 10px 16px;
        border-radius: 6px;
        color: #92400E;
        font-size: 12.5px;
        line-height: 1.5;
        margin-bottom: 18px;
    }
    
    /* Content Card */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* Pathway Protocol Step */
    .protocol-step {
        background: #F8FAFC;
        border-left: 3px solid #0284C7;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 13px;
        color: #1E293B;
    }
    
    /* Button custom overrides */
    .stButton>button {
        background-color: #0F766E;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 7px 14px;
    }
    .stButton>button:hover {
        background-color: #115E59;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
def init_session_state():
    if "df" not in st.session_state:
        df, report = load_dataset()
        st.session_state["df"] = df
        st.session_state["report"] = report

    if "k_selected" not in st.session_state:
        st.session_state["k_selected"] = DEFAULT_K

    if "preprocessor" not in st.session_state or "model" not in st.session_state:
        _train_pipeline()


def _train_pipeline(k_override: int = None):
    """Fit preprocessing and clustering pipeline on the active dataset."""
    df = st.session_state["df"]
    preprocessor = CardiacPreprocessor()
    X = preprocessor.fit_transform(df)
    st.session_state["preprocessor"] = preprocessor
    st.session_state["X_scaled"] = X

    model = CardiacClusteringModel(random_state=DEFAULT_RANDOM_STATE)
    # Evaluate candidates K=2 to 8
    eval_results = model.evaluate_candidate_k(X, k_min=DEFAULT_K_MIN, k_max=DEFAULT_K_MAX)
    st.session_state["eval_results"] = eval_results
    st.session_state["recommended_k"] = model.recommended_k
    st.session_state["recommendation_reason"] = model.recommendation_reason

    # Choose K
    k = k_override if k_override is not None else st.session_state.get("k_selected", model.recommended_k)
    st.session_state["k_selected"] = k

    # Fit model
    model.fit(X, k=k)
    st.session_state["model"] = model

    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered["cluster"] = model.cluster_labels
    st.session_state["df_clustered"] = df_clustered

    # Generate cluster profiles & risk stratifications
    profiles = generate_cluster_profiles(df_clustered, model.cluster_labels)
    st.session_state["profiles"] = profiles

    # Resource allocation data
    st.session_state["resource_allocation"] = compute_hospital_resource_allocation(df_clustered)


init_session_state()

# ----------------- SIDEBAR NAVIGATION -----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 26px;">🏥</span>
        <div>
            <h2 style="font-size: 17px; margin: 0; color: #0F172A; font-weight: 700;">Clinical Analytics</h2>
            <p style="font-size: 11px; margin: 0; color: #64748B;">Cardiology Resource & Phenotyping</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Clinical Workflows")
    
    page = st.radio(
        "Select Analytical View:",
        [
            "1. Clinical Cohort Overview",
            "2. Hospital Resource Allocation & Care Pathways",
            "3. Patient Registry & Triage Explorer",
            "4. Patient Clinical Profile & Intake",
            "5. System Diagnostics & Data Management"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### Phenotype Granularity")
    
    eval_results = st.session_state.get("eval_results", {})
    available_ks = sorted(eval_results.keys()) if eval_results else [2, 3, 4, 5, 6, 7, 8]
    
    current_k = st.session_state.get("k_selected", 3)
    new_k = st.select_slider(
        "Number of Clusters (K):",
        options=available_ks,
        value=current_k,
        help="Adjust the number of patient phenotypes to dynamically reallocate hospital resources."
    )

    if new_k != current_k:
        st.session_state["k_selected"] = new_k
        _train_pipeline(k_override=new_k)
        st.rerun()

    rec_k = st.session_state.get("recommended_k", 3)
    if new_k == rec_k:
        st.caption(f"⭐ **Current K ({new_k}) matches optimal algorithmic clustering.**")
    else:
        st.caption(f"💡 Recommended K: **{rec_k}**")
        if st.button(f"Reset to Recommended (K={rec_k})", use_container_width=True):
            st.session_state["k_selected"] = rec_k
            _train_pipeline(k_override=rec_k)
            st.rerun()

    st.markdown("---")
    st.markdown("### Cohort Provenance")
    st.caption(f"**Source:** {st.session_state['report']['source_name']}")
    st.caption(f"**Census:** {len(st.session_state['df'])} active patient records")
    if st.session_state["report"].get("is_synthetic", False):
        st.warning("⚠️ Using synthetic demo cohort")


# ----------------- GLOBAL HEADER & CLINICAL DISCLAIMER -----------------
st.markdown("""
<div class="clinical-header">
    <h1>Cardiac Phenotype Profiling & Hospital Resource Platform</h1>
    <p>Clinical Decision Support | Unsupervised Patient Stratification, Risk Factor Deconvolution & Capacity Planning</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="disclaimer-banner">
    <strong>⚖️ Clinical Decision Support Advisory:</strong> {MEDICAL_DISCLAIMER}
</div>
""", unsafe_allow_html=True)


# =====================================================================
# PAGE 1: CLINICAL COHORT OVERVIEW
# =====================================================================
if page == "1. Clinical Cohort Overview":
    st.subheader("Clinical Cohort Overview")
    st.caption("Executive cardiology overview, phenotype distributions, and multivariate physiological characteristics.")

    df_clustered = st.session_state["df_clustered"]
    model = st.session_state["model"]
    report = st.session_state["report"]
    k_selected = st.session_state["k_selected"]
    profiles = st.session_state["profiles"]
    resource_data = st.session_state.get("resource_allocation") or compute_hospital_resource_allocation(df_clustered)

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Cohort Census</div>
            <div class="kpi-value">{len(df_clustered)}</div>
            <div class="kpi-subtitle">Total patient records</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Discovered Phenotypes</div>
            <div class="kpi-value">{k_selected}</div>
            <div class="kpi-subtitle">Distinct clinical clusters</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        tot_beds = resource_data["total_resources"]["total_inpatient_beds"]
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Beds Required</div>
            <div class="kpi-value">{tot_beds}</div>
            <div class="kpi-subtitle">Inpatient bed capacity</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        cath_slots = resource_data["total_resources"]["cath_lab_slots"]
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Cath Lab Demand</div>
            <div class="kpi-value">{cath_slots}</div>
            <div class="kpi-subtitle">Angiography slots required</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic Clinical Summary
    largest_cluster_id = max(profiles, key=lambda c: profiles[c]["record_count"])
    largest_p = profiles[largest_cluster_id]
    smallest_cluster_id = min(profiles, key=lambda c: profiles[c]["record_count"])
    smallest_p = profiles[smallest_cluster_id]

    st.markdown(f"""
    <div class="content-box" style="border-left: 4px solid #0F766E;">
        <h4 style="margin: 0 0 6px 0; color: #0F172A; font-size: 15px;">📊 Clinical Cohort Profile Summary</h4>
        <p style="margin: 0; font-size: 13.5px; color: #334155; line-height: 1.6;">
            The unsupervised clustering algorithm segments the <strong>{len(df_clustered)}</strong> cardiology patients into <strong>{k_selected}</strong> distinct clinical phenotypes. 
            The predominant subgroup is <strong>Cluster {largest_cluster_id}</strong> representing <strong>{largest_p['record_count']} patients ({largest_p['percentage']}%)</strong>, 
            while <strong>Cluster {smallest_cluster_id}</strong> encompasses <strong>{smallest_p['record_count']} patients ({smallest_p['percentage']}%)</strong>. 
            Principal physiological differentiators include ischemic stress markers (<code>oldpeak</code> ST depression, exercise-induced angina), 
            resting systolic hypertension (<code>trestbps</code>), and peak chronotropic tolerance (<code>thalach</code>).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: Distribution & Donut
    col_d1, col_d2 = st.columns([1.2, 1.0])
    with col_d1:
        fig_dist = plot_cluster_distribution(df_clustered)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_d2:
        fig_pie = plot_cluster_proportions_pie(df_clustered)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Row 2: Age distribution & Clinical indicators
    col_d3, col_d4 = st.columns([1.0, 1.2])
    with col_d3:
        fig_age = plot_age_distribution_by_cluster(df_clustered)
        st.plotly_chart(fig_age, use_container_width=True)
    with col_d4:
        fig_comp = plot_clinical_feature_comparison_bars(df_clustered)
        st.plotly_chart(fig_comp, use_container_width=True)


# =====================================================================
# PAGE 2: HOSPITAL RESOURCE ALLOCATION & CARE PATHWAYS
# =====================================================================
elif page == "2. Hospital Resource Allocation & Care Pathways":
    st.subheader("Hospital Resource Allocation & Care Pathways")
    st.caption("Operational capacity planning, bed and staffing requirements, and targeted clinical intervention pathways.")

    df_clustered = st.session_state["df_clustered"]
    profiles = st.session_state["profiles"]
    clusters = sorted(profiles.keys())

    # Interactive Capacity Scaler / Admissions Simulator
    st.markdown("#### ⚙️ Hospital Inpatient Admissions & Capacity Simulator")
    sim_col1, sim_col2 = st.columns([1.5, 1.0])
    with sim_col1:
        use_simulation = st.toggle("Simulate Custom Inpatient Intake Volume", value=False)
        if use_simulation:
            sim_volume = st.slider("Projected Monthly Patient Inflow (Admissions):", min_value=50, max_value=1000, value=len(df_clustered), step=25)
        else:
            sim_volume = None

    with sim_col2:
        curr_vol = sim_volume if use_simulation else len(df_clustered)
        st.info(f"💡 Calculations are actively scaled to an inpatient census of **{curr_vol} patients** distributed across the {len(clusters)} phenotype clusters.")

    # Recompute resource allocation for active volume
    resource_data = compute_hospital_resource_allocation(df_clustered, simulated_intake_volume=sim_volume)
    tot = resource_data["total_resources"]

    # Hospital Capacity KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #E11D48;">
            <div class="kpi-title">CCU / ICU Beds</div>
            <div class="kpi-value" style="color: #E11D48;">{tot['ccu_beds']}</div>
            <div class="kpi-subtitle">Critical care monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #D97706;">
            <div class="kpi-title">Telemetry Beds</div>
            <div class="kpi-value" style="color: #D97706;">{tot['telemetry_beds']}</div>
            <div class="kpi-subtitle">Step-down continuous ECG</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #0D9488;">
            <div class="kpi-title">General Ward Beds</div>
            <div class="kpi-value" style="color: #0D9488;">{tot['ward_beds']}</div>
            <div class="kpi-subtitle">Inpatient cardiology ward</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #DC2626;">
            <div class="kpi-title">Cath Lab Angiography</div>
            <div class="kpi-value" style="color: #DC2626;">{tot['cath_lab_slots']}</div>
            <div class="kpi-subtitle">Catheterization slots</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #0284C7;">
            <div class="kpi-title">Nursing Shifts / Day</div>
            <div class="kpi-value" style="color: #0284C7;">{tot['total_nurse_shifts_per_day']}</div>
            <div class="kpi-subtitle">Specialized nurse shifts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Operational Visualizations
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_beds = plot_hospital_bed_allocation(resource_data)
        st.plotly_chart(fig_beds, use_container_width=True)
    with col_v2:
        fig_procs = plot_diagnostic_procedural_load(resource_data)
        st.plotly_chart(fig_procs, use_container_width=True)

    # Clinical Acuity Breakdown
    st.markdown("---")
    st.markdown("#### Clinical Acuity Stratification Across Discovered Clusters")
    col_act1, col_act2 = st.columns([1.1, 1.3])
    with col_act1:
        fig_triage = plot_triage_acuity_donut(resource_data)
        st.plotly_chart(fig_triage, use_container_width=True)
    with col_act2:
        st.markdown("##### Acuity Stratification Methodology & Triage Tiers")
        st.markdown("""
        - 🔴 **High Acuity (CCU / Immediate Workup):** Patients exhibiting marked exercise ST depression (≥1.5mm), exercise angina, multi-vessel fluoroscopy findings, or reversible perfusion defects.
        - 🟡 **Moderate Acuity (Step-Down Telemetry):** Patients with Stage 2 systolic hypertension (≥140 mm Hg), borderline ischemic changes, or isolated vascular disease requiring telemetry stabilization.
        - 🟢 **Standard Acuity (Ward / Ambulatory):** Stable hemodynamic profile, preserved exercise heart rate, and mild cardiometabolic markers managed via medical therapy and secondary prevention.
        """)


# =====================================================================
# PAGE 3: PATIENT REGISTRY & TRIAGE EXPLORER
# =====================================================================
elif page == "3. Patient Registry & Triage Explorer":
    st.subheader("Patient Registry & Triage Explorer")
    st.caption("Search, filter, and inspect individual patient records with clinical acuity badges and resource assignments.")

    df_clustered = st.session_state["df_clustered"]
    profiles = st.session_state["profiles"]
    clusters = sorted(df_clustered["cluster"].unique())

    # Map each cluster's acuity to patients for quick badges
    acuity_map = {c: profiles[c].get("risk_profile", {}).get("acuity_tier_name", f"Cluster {c}") for c in clusters}
    bed_map = {c: profiles[c].get("risk_profile", {}).get("bed_type", "Standard Ward") for c in clusters}
    
    df_explorer = df_clustered.copy()
    df_explorer["Clinical Acuity"] = df_explorer["cluster"].map(acuity_map)
    df_explorer["Allocated Bed"] = df_explorer["cluster"].map(bed_map)

    # Filter Controls
    with st.expander("🔍 Search and Clinical Filter Controls", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            search_query = st.text_input("Search Patient Record ID (e.g. REC-015):", "").strip()

        with f_col2:
            cluster_filter = st.multiselect(
                "Filter by Cluster:",
                options=clusters,
                default=clusters,
                format_func=lambda c: f"Cluster {c}"
            )

        with f_col3:
            if "trestbps" in df_explorer.columns:
                sbp_filter = st.selectbox(
                    "Resting Blood Pressure Filter:",
                    ["All Patients", "Hypertensive (SBP >= 140 mm Hg)", "Severe Hypertensive (SBP >= 160 mm Hg)", "Normotensive (< 140 mm Hg)"]
                )
            else:
                sbp_filter = "All Patients"

        with f_col4:
            if "age" in df_explorer.columns:
                min_age = int(df_explorer["age"].min())
                max_age = int(df_explorer["age"].max())
                age_range = st.slider("Age Filter (years):", min_age, max_age, (min_age, max_age))
            else:
                age_range = None

    # Apply Filters
    filtered_df = df_explorer.copy()

    if search_query:
        filtered_df = filtered_df[filtered_df["record_id"].astype(str).str.contains(search_query, case=False, na=False)]

    if cluster_filter:
        filtered_df = filtered_df[filtered_df["cluster"].isin(cluster_filter)]

    if sbp_filter == "Hypertensive (SBP >= 140 mm Hg)":
        filtered_df = filtered_df[filtered_df["trestbps"] >= 140.0]
    elif sbp_filter == "Severe Hypertensive (SBP >= 160 mm Hg)":
        filtered_df = filtered_df[filtered_df["trestbps"] >= 160.0]
    elif sbp_filter == "Normotensive (< 140 mm Hg)":
        filtered_df = filtered_df[filtered_df["trestbps"] < 140.0]

    if age_range and "age" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])
        ]

    st.markdown(f"**Displaying {len(filtered_df)} of {len(df_clustered)} patient records**")

    # Table columns order
    display_cols = ["record_id", "cluster", "Clinical Acuity", "Allocated Bed"] + [
        c for c in filtered_df.columns if c not in ["record_id", "cluster", "Clinical Acuity", "Allocated Bed"]
    ]
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
        height=380
    )

    # Actions & Deep Dive
    act1, act2 = st.columns([1.4, 1.0])

    with act1:
        st.markdown("#### Patient Clinical Deep-Dive")
        selected_record_id = st.selectbox(
            "Select Patient Record to inspect clinical vitals & triage profile:",
            options=filtered_df["record_id"].tolist() if len(filtered_df) > 0 else []
        )

        if selected_record_id:
            record_row = filtered_df[filtered_df["record_id"] == selected_record_id].iloc[0]
            c_val = int(record_row["cluster"])
            tier_name = acuity_map.get(c_val, "Standard")
            bed_str = bed_map.get(c_val, "Ward Bed")

            st.markdown(f"""
            <div class="content-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h3 style="margin: 0; color: #0F172A;">Patient Record: {record_row['record_id']}</h3>
                    <span style="background: #0284C7; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 13px;">
                        Phenotype Cluster {c_val}
                    </span>
                </div>
                <div style="background: #F1F5F9; padding: 10px 14px; border-radius: 6px; margin-bottom: 12px;">
                    <strong>Assigned Acuity:</strong> {tier_name} | <strong>Recommended Unit:</strong> {bed_str}
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px;">
                    <div><strong>Age:</strong> {record_row.get('age', 'N/A')} yrs</div>
                    <div><strong>Sex:</strong> {'Male' if record_row.get('sex') == 1 else 'Female'}</div>
                    <div><strong>Resting BP:</strong> {record_row.get('trestbps', 'N/A')} mm Hg</div>
                    <div><strong>Cholesterol:</strong> {record_row.get('chol', 'N/A')} mg/dl</div>
                    <div><strong>Max Heart Rate:</strong> {record_row.get('thalach', 'N/A')} bpm</div>
                    <div><strong>ST Depression:</strong> {record_row.get('oldpeak', 'N/A')} mm</div>
                    <div><strong>Fluoroscopy Vessels:</strong> {record_row.get('ca', 'N/A')}</div>
                    <div><strong>Chest Pain Type:</strong> {record_row.get('cp', 'N/A')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with act2:
        st.markdown("#### Registry Export")
        st.write("Export patient cohort lists with cluster assignments and clinical acuity tiers for hospital EHR integration.")

        filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Cohort (CSV)",
            data=filtered_csv,
            file_name="cardiac_filtered_patients.csv",
            mime="text/csv",
            use_container_width=True
        )

        full_csv = df_explorer.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Full Clustered Cohort (CSV)",
            data=full_csv,
            file_name="cardiac_clustered_cohort.csv",
            mime="text/csv",
            use_container_width=True
        )


# =====================================================================
# PAGE 4: PATIENT CLINICAL PROFILE & INTAKE
# =====================================================================
elif page == "4. Patient Clinical Profile & Intake":
    st.subheader("Patient Clinical Profile & Intake")
    st.caption("Individual patient similarity profiling, nearest-centroid phenotype assignment, and personalized hospital care pathway recommendations.")

    df_clustered = st.session_state["df_clustered"]
    preprocessor = st.session_state["preprocessor"]
    model = st.session_state["model"]
    profiles = st.session_state["profiles"]

    mode = st.radio(
        "Select Clinical Assessment Mode:",
        ["Mode A: Profile Existing Cohort Patient", "Mode B: Intake & Triage New Patient"],
        horizontal=True
    )

    if mode == "Mode A: Profile Existing Cohort Patient":
        st.markdown("#### Mode A: Profile an Existing Cohort Record")
        patient_options = df_clustered["record_id"].tolist()
        selected_rec = st.selectbox("Select Patient Record:", patient_options)

        if selected_rec:
            record_row = df_clustered[df_clustered["record_id"] == selected_rec].iloc[0]
            patient_dict = record_row.to_dict()
            assigned_c = int(record_row["cluster"])
            cluster_profile = profiles[assigned_c]
            risk_profile = cluster_profile.get("risk_profile", {})

            # Vector transform and similarity calculation
            x_scaled = preprocessor.transform(pd.DataFrame([patient_dict]))
            _, similarity_dict, min_dist, meta = model.predict_new_record(x_scaled)

            # Patient-specific hospital recommendation
            patient_rec = get_patient_hospital_recommendation(patient_dict, assigned_c, risk_profile)

            # Summary Card
            st.markdown(f"""
            <div class="content-box" style="border-left: 5px solid {patient_rec['badge_color']};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 style="margin: 0; color: #0F172A;">Patient Profile: {selected_rec}</h3>
                    <span style="background: {patient_rec['badge_color']}; color: #FFFFFF; padding: 5px 14px; border-radius: 20px; font-weight: 700; font-size: 13.5px;">
                        Phenotype Cluster {assigned_c} — {patient_rec['cluster_acuity']}
                    </span>
                </div>
                <p style="margin: 0 0 10px 0; color: #1E293B; font-weight: 600; font-size: 14.5px;">
                    {cluster_profile['title']}
                </p>
                <div style="background: {patient_rec['bg_color']}; padding: 10px 14px; border-radius: 6px; font-size: 13.5px; color: #0F172A; line-height: 1.5;">
                    <strong>Recommended Inpatient Unit:</strong> {patient_rec['recommended_bed']} &nbsp;|&nbsp; 
                    <strong>Nursing Ratio:</strong> {patient_rec['nursing_ratio']} &nbsp;|&nbsp; 
                    <strong>Triage Response:</strong> {patient_rec['triage_urgency']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Escalation Red-Flags if any
            if patient_rec["escalation_flags"]:
                st.warning("⚠️ **Individual Clinical Red-Flag Indicators:**")
                for flag in patient_rec["escalation_flags"]:
                    st.caption(f"• **{flag}**")

            # Comparative Chart
            st.markdown("##### Phenotype Cluster Similarity Distribution")
            fig_sim = plot_similarity_to_all_clusters(similarity_dict)
            st.plotly_chart(fig_sim, use_container_width=True)

            # Simple Clinical Vitals Comparison vs Cluster Average
            st.markdown("---")
            st.markdown("#### Patient Vitals vs Assigned Cluster Average")
            comp_result = compare_patient_to_cluster(patient_dict, cluster_profile)
            num_stats = cluster_profile.get("numerical_stats", {})

            # 4 Simple, Understandable Metric Cards
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                p_bp = float(patient_dict.get("trestbps", 130))
                c_bp = float(num_stats.get("trestbps", {}).get("mean", 130))
                diff_bp = p_bp - c_bp
                st.metric(
                    label="Resting Blood Pressure",
                    value=f"{p_bp:.0f} mm Hg",
                    delta=f"{diff_bp:+.1f} mm Hg vs Cluster Avg",
                    delta_color="inverse"
                )
            with m_col2:
                p_chol = float(patient_dict.get("chol", 240))
                c_chol = float(num_stats.get("chol", {}).get("mean", 240))
                diff_chol = p_chol - c_chol
                st.metric(
                    label="Serum Cholesterol",
                    value=f"{p_chol:.0f} mg/dl",
                    delta=f"{diff_chol:+.1f} mg/dl vs Cluster Avg",
                    delta_color="inverse"
                )
            with m_col3:
                p_hr = float(patient_dict.get("thalach", 150))
                c_hr = float(num_stats.get("thalach", {}).get("mean", 150))
                diff_hr = p_hr - c_hr
                st.metric(
                    label="Peak Heart Rate",
                    value=f"{p_hr:.0f} bpm",
                    delta=f"{diff_hr:+.1f} bpm vs Cluster Avg",
                    delta_color="normal"
                )
            with m_col4:
                p_st = float(patient_dict.get("oldpeak", 1.0))
                c_st = float(num_stats.get("oldpeak", {}).get("mean", 1.0))
                diff_st = p_st - c_st
                st.metric(
                    label="ST Depression",
                    value=f"{p_st:.1f} mm",
                    delta=f"{diff_st:+.1f} mm vs Cluster Avg",
                    delta_color="inverse"
                )

            # Simpler comparison graph
            fig_simple_comp = plot_simple_patient_comparison(patient_dict, cluster_profile)
            st.plotly_chart(fig_simple_comp, use_container_width=True)

            # Simple Breakdown Table expander
            with st.expander("📋 View Complete Numerical Parameter Breakdown vs Cluster Average", expanded=False):
                comp_df = pd.DataFrame(comp_result["comparison_items"])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Actionable Clinical Protocol for this patient
            with st.expander("📋 View Actionable Clinical Hospital Care Protocol for this Patient", expanded=True):
                h1, h2 = st.columns(2)
                with h1:
                    st.markdown("**Immediate Inpatient Protocol:**")
                    for act in patient_rec["immediate_actions"]:
                        st.markdown(f"- ⚡ {act}")
                    st.markdown("**Diagnostic Plan:**")
                    for diag in patient_rec["diagnostic_plan"]:
                        st.markdown(f"- 🔬 {diag}")
                with h2:
                    st.markdown("**Pharmacotherapy Strategy:**")
                    for drug in patient_rec["pharmacotherapy"]:
                        st.markdown(f"- 💊 {drug}")
                    st.markdown("**Post-Discharge & Cardiac Rehab:**")
                    for dis in patient_rec["discharge_plan"]:
                        st.markdown(f"- 🏃 {dis}")

    else:
        # Mode B: Intake & Triage New Patient
        st.markdown("#### Mode B: Intake & Triage a New Hypothetical Patient")
        st.info("💡 Enter physiological and diagnostic metrics below. The system standardizes values using the fitted pipeline, assigns the patient to the nearest phenotype centroid, and outputs an immediate bed and care pathway recommendation.")

        with st.form("new_patient_intake_form"):
            st.markdown("##### 1. Demographics & Hemodynamics")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                age_val = st.number_input("Age (years):", min_value=18, max_value=100, value=58)
            with f2:
                sex_val = st.selectbox("Biological Sex:", options=[1.0, 0.0], format_func=lambda s: "Male (1)" if s == 1.0 else "Female (0)")
            with f3:
                trestbps_val = st.number_input("Resting Blood Pressure (mm Hg):", min_value=80, max_value=240, value=142)
            with f4:
                chol_val = st.number_input("Serum Cholesterol (mg/dl):", min_value=100, max_value=600, value=250)

            st.markdown("##### 2. Exercise Stress Testing & Ischemia Indicators")
            f5, f6, f7, f8 = st.columns(4)
            with f5:
                cp_val = st.selectbox(
                    "Chest Pain Presentation:",
                    options=[1.0, 2.0, 3.0, 4.0],
                    format_func=lambda c: FEATURE_DEFINITIONS["cp"]["options"][c]
                )
            with f6:
                thalach_val = st.number_input("Peak Exercise Heart Rate (bpm):", min_value=60, max_value=220, value=140)
            with f7:
                exang_val = st.selectbox("Exercise-Induced Angina:", options=[0.0, 1.0], format_func=lambda e: "No (0)" if e == 0.0 else "Yes (1)")
            with f8:
                oldpeak_val = st.number_input("ST Depression (oldpeak mm):", min_value=0.0, max_value=7.0, value=1.4, step=0.1)

            st.markdown("##### 3. Diagnostic & Electrocardiographic Findings")
            f9, f10, f11, f12 = st.columns(4)
            with f9:
                fbs_val = st.selectbox("Fasting Blood Sugar > 120 mg/dl:", options=[0.0, 1.0], format_func=lambda f: "False (<=120 mg/dl)" if f == 0.0 else "True (>120 mg/dl)")
            with f10:
                restecg_val = st.selectbox(
                    "Resting ECG Result:",
                    options=[0.0, 1.0, 2.0],
                    format_func=lambda r: FEATURE_DEFINITIONS["restecg"]["options"][r]
                )
            with f11:
                slope_val = st.selectbox(
                    "Peak ST Segment Slope:",
                    options=[1.0, 2.0, 3.0],
                    format_func=lambda s: FEATURE_DEFINITIONS["slope"]["options"][s]
                )
            with f12:
                ca_val = st.selectbox("Fluoroscopy Major Vessels (0-3):", options=[0.0, 1.0, 2.0, 3.0], index=1)

            thal_val = st.selectbox(
                "Thallium Scintigraphy Finding:",
                options=[3.0, 6.0, 7.0],
                format_func=lambda t: FEATURE_DEFINITIONS["thal"]["options"][t]
            )

            submit_patient = st.form_submit_button("⚡ Evaluate Phenotype & Recommend Hospital Care Plan", use_container_width=True)

        if submit_patient:
            new_record = {
                "age": float(age_val),
                "sex": float(sex_val),
                "cp": float(cp_val),
                "trestbps": float(trestbps_val),
                "chol": float(chol_val),
                "fbs": float(fbs_val),
                "restecg": float(restecg_val),
                "thalach": float(thalach_val),
                "exang": float(exang_val),
                "oldpeak": float(oldpeak_val),
                "slope": float(slope_val),
                "ca": float(ca_val),
                "thal": float(thal_val)
            }

            # 1. Out of Distribution Check
            ood_warnings = preprocessor.check_out_of_distribution(new_record)
            if ood_warnings:
                st.warning("⚠️ **Physiological Range Alert:** One or more entered features deviate significantly from the baseline cohort distribution. Phenotype proximity may exhibit reduced fidelity.")
                for w in ood_warnings:
                    st.caption(f"• **{w['display_name']}**: {w['message']}")

            # 2. Preprocess & Predict
            new_df = pd.DataFrame([new_record])
            x_scaled = preprocessor.transform(new_df)
            assigned_c, sim_dict, min_dist, meta = model.predict_new_record(x_scaled)
            cluster_profile = profiles[assigned_c]
            risk_profile = cluster_profile.get("risk_profile", {})

            # Patient-specific hospital recommendation
            patient_rec = get_patient_hospital_recommendation(new_record, assigned_c, risk_profile)

            # Results Display
            st.markdown(f"""
            <div class="content-box" style="border-left: 5px solid {patient_rec['badge_color']};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 style="margin: 0; color: #0F172A;">Intake Assessment: Phenotype Cluster {assigned_c}</h3>
                    <span style="background: {patient_rec['badge_color']}; color: #FFFFFF; padding: 5px 14px; border-radius: 20px; font-weight: 700; font-size: 13.5px;">
                        {patient_rec['cluster_acuity']}
                    </span>
                </div>
                <p style="margin: 0 0 10px 0; color: #1E293B; font-weight: 600; font-size: 14.5px;">
                    {cluster_profile['title']}
                </p>
                <div style="background: {patient_rec['bg_color']}; padding: 10px 14px; border-radius: 6px; font-size: 13.5px; color: #0F172A; line-height: 1.5;">
                    <strong>Recommended Bed Unit:</strong> {patient_rec['recommended_bed']} &nbsp;|&nbsp; 
                    <strong>Nursing Ratio:</strong> {patient_rec['nursing_ratio']} &nbsp;|&nbsp; 
                    <strong>Triage Response:</strong> {patient_rec['triage_urgency']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if patient_rec["escalation_flags"]:
                st.warning("⚠️ **Patient-Specific Escalation Triggers:**")
                for flag in patient_rec["escalation_flags"]:
                    st.caption(f"• {flag}")

            st.markdown("##### Cluster Proximity Spectrum")
            fig_sim = plot_similarity_to_all_clusters(sim_dict)
            st.plotly_chart(fig_sim, use_container_width=True)

            # Simple Clinical Vitals Comparison vs Cluster Average
            st.markdown("---")
            st.markdown("#### Patient Vitals vs Assigned Cluster Average")
            comp_result = compare_patient_to_cluster(new_record, cluster_profile)
            num_stats = cluster_profile.get("numerical_stats", {})

            # 4 Simple, Understandable Metric Cards
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                p_bp = float(new_record.get("trestbps", 130))
                c_bp = float(num_stats.get("trestbps", {}).get("mean", 130))
                diff_bp = p_bp - c_bp
                st.metric(
                    label="Resting Blood Pressure",
                    value=f"{p_bp:.0f} mm Hg",
                    delta=f"{diff_bp:+.1f} mm Hg vs Cluster Avg",
                    delta_color="inverse"
                )
            with m_col2:
                p_chol = float(new_record.get("chol", 240))
                c_chol = float(num_stats.get("chol", {}).get("mean", 240))
                diff_chol = p_chol - c_chol
                st.metric(
                    label="Serum Cholesterol",
                    value=f"{p_chol:.0f} mg/dl",
                    delta=f"{diff_chol:+.1f} mg/dl vs Cluster Avg",
                    delta_color="inverse"
                )
            with m_col3:
                p_hr = float(new_record.get("thalach", 150))
                c_hr = float(num_stats.get("thalach", {}).get("mean", 150))
                diff_hr = p_hr - c_hr
                st.metric(
                    label="Peak Heart Rate",
                    value=f"{p_hr:.0f} bpm",
                    delta=f"{diff_hr:+.1f} bpm vs Cluster Avg",
                    delta_color="normal"
                )
            with m_col4:
                p_st = float(new_record.get("oldpeak", 1.0))
                c_st = float(num_stats.get("oldpeak", {}).get("mean", 1.0))
                diff_st = p_st - c_st
                st.metric(
                    label="ST Depression",
                    value=f"{p_st:.1f} mm",
                    delta=f"{diff_st:+.1f} mm vs Cluster Avg",
                    delta_color="inverse"
                )

            # Simpler comparison graph
            fig_simple_comp = plot_simple_patient_comparison(new_record, cluster_profile)
            st.plotly_chart(fig_simple_comp, use_container_width=True)

            # Simple Breakdown Table expander
            with st.expander("📋 View Complete Numerical Parameter Breakdown vs Cluster Average", expanded=False):
                comp_df = pd.DataFrame(comp_result["comparison_items"])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Actionable Clinical Protocol
            st.markdown("---")
            st.markdown("#### 🏥 Recommended Hospital Clinical Action Plan for this Admitted Patient")
            h1, h2 = st.columns(2)
            with h1:
                st.markdown("**1. Immediate Inpatient Triage & Bed Placement:**")
                for act in patient_rec["immediate_actions"]:
                    st.markdown(f"- ⚡ {act}")
                st.markdown("**2. Diagnostic Angiography & Imaging:**")
                for diag in patient_rec["diagnostic_plan"]:
                    st.markdown(f"- 🔬 {diag}")
            with h2:
                st.markdown("**3. Guideline-Directed Pharmacotherapy:**")
                for drug in patient_rec["pharmacotherapy"]:
                    st.markdown(f"- 💊 {drug}")
                st.markdown("**4. Discharge Follow-Up & Cardiac Rehab:**")
                for dis in patient_rec["discharge_plan"]:
                    st.markdown(f"- 🏃 {dis}")


# =====================================================================
# PAGE 5: SYSTEM DIAGNOSTICS & DATA MANAGEMENT
# =====================================================================
elif page == "5. System Diagnostics & Data Management":
    st.subheader("System Diagnostics & Data Management")
    st.caption("Algorithm performance diagnostics, clustering metrics (Elbow & Silhouette), dataset quality inspection, and CSV file upload.")

    eval_results = st.session_state["eval_results"]
    k_selected = st.session_state["k_selected"]
    rec_k = st.session_state["recommended_k"]
    rec_reason = st.session_state["recommendation_reason"]
    model = st.session_state["model"]
    report = st.session_state["report"]
    df = st.session_state["df"]
    df_clustered = st.session_state["df_clustered"]

    # Model Evaluation Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Active Clusters (K)", k_selected)
    with m2:
        st.metric("Optimal Recommended K", rec_k)
    with m3:
        st.metric("Silhouette Coefficient", f"{model.overall_silhouette:.4f}" if model.overall_silhouette else "N/A")
    with m4:
        inertia_val = eval_results.get(k_selected, {}).get("inertia", "N/A")
        st.metric("Sum of Squared Errors (SSE)", f"{inertia_val}")

    st.markdown(f"""
    <div class="content-box" style="border-left: 4px solid #0284C7;">
        <h4 style="margin: 0 0 6px 0; color: #0F172A; font-size: 14.5px;">📐 Algorithmic Selection Rationale</h4>
        <p style="margin: 0; font-size: 13.5px; color: #334155; line-height: 1.5;">{rec_reason}</p>
    </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_elbow = plot_elbow_curve(eval_results, k_selected)
        st.plotly_chart(fig_elbow, use_container_width=True)
    with col_e2:
        fig_sil = plot_silhouette_curve(eval_results, k_selected)
        st.plotly_chart(fig_sil, use_container_width=True)

    # Diagnostic Table
    st.markdown("#### Candidate K Evaluation Summary")
    eval_table = []
    for k_val, meta in sorted(eval_results.items()):
        eval_table.append({
            "Clusters (K)": k_val,
            "Inertia (SSE)": meta["inertia"],
            "Silhouette Score": meta["silhouette_score"],
            "Smallest Cluster Size": meta["min_cluster_size"],
            "Cluster Size Distribution": str(meta["cluster_sizes"]),
            "Status": "⭐ Active" if k_val == k_selected else ("💡 Recommended" if k_val == rec_k else "")
        })
    st.dataframe(pd.DataFrame(eval_table), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Dataset Management & Quality Audit")

    # Data Quality Metrics
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Total Records", report["raw_rows"])
    with q2:
        st.metric("Total Attributes", report["raw_cols"])
    with q3:
        st.metric("Data Completeness", f"{report['completeness_pct']}%")
    with q4:
        st.metric("Duplicate Observations", report["duplicate_rows"])

    # CSV Upload Workflow
    st.markdown("##### Upload Custom Patient Dataset (CSV)")
    st.write("Upload a custom cardiovascular cohort. The system verifies column schema compatibility, performs median/mode imputation, and recalibrates the clustering and hospital resource models.")

    uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
    if uploaded_file is not None:
        try:
            new_df, new_report = load_dataset(uploaded_file)
            if new_report["is_valid"]:
                st.success(f"✅ Successfully validated **{new_report['source_name']}** ({new_report['raw_rows']} rows, {len(new_report['present_clustering_features'])} compatible attributes).")
                if st.button("Apply and Retrain Pipeline with Uploaded Dataset", use_container_width=True):
                    st.session_state["df"] = new_df
                    st.session_state["report"] = new_report
                    _train_pipeline()
                    st.rerun()
            else:
                st.error("❌ Uploaded CSV lacks required clinical features (must contain at least 'age' or 'trestbps' and 4 compatible cardiac attributes).")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")

    # Dataset Reset / Synthetic Generation
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("🔄 Reset to Default Benchmark Dataset", use_container_width=True):
            df, report = load_dataset()
            st.session_state["df"] = df
            st.session_state["report"] = report
            _train_pipeline()
            st.rerun()

    with col_d2:
        if st.button("🧪 Generate Synthetic Demo Cohort", use_container_width=True):
            df_syn, report_syn = generate_synthetic_demo_dataset()
            st.session_state["df"] = df_syn
            st.session_state["report"] = report_syn
            _train_pipeline()
            st.rerun()

    # Feature Dictionary
    st.markdown("---")
    st.markdown("##### Clinical Feature Schema & Imputation Audit")
    dict_rows = []
    for col, meta in FEATURE_DEFINITIONS.items():
        present = "✅ Present" if col in report["present_clustering_features"] else "❌ Missing"
        miss_count = report["missing_counts"].get(col, 0)
        miss_pct = report["missing_pct"].get(col, 0.0)
        
        dict_rows.append({
            "Feature": col,
            "Display Name": meta["display_name"],
            "Type": meta["type"].capitalize(),
            "Unit / Categories": meta.get("unit", ""),
            "Clinical Description": meta["description"],
            "Status": present,
            "Missing Values": f"{miss_count} ({miss_pct:.1f}%)"
        })

    st.dataframe(pd.DataFrame(dict_rows), use_container_width=True, hide_index=True)

    # Clustered dataset download
    full_csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Full Clustered Dataset (CSV)",
        data=full_csv,
        file_name="cardiac_clustered_cohort.csv",
        mime="text/csv",
        use_container_width=True
    )
