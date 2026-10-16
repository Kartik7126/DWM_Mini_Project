"""
CardioInsight — Cardiac Patient Profiling & Hospital Analytics Using K-Means Clustering
Main Streamlit Application
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
    plot_similarity_to_all_clusters
)

# Set page config
st.set_page_config(
    page_title="CardioInsight | Cardiac Patient Profiling & Analytics",
    page_icon="🫀",
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
        background: linear-gradient(135deg, #0F2942 0%, #1E3A8A 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .clinical-header h1 {
        color: #FFFFFF !important;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    
    .clinical-header p {
        color: #93C5FD;
        font-size: 14px;
        margin: 0;
    }
    
    /* KPI Card Component */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    
    .kpi-title {
        color: #64748B;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        color: #0F2942;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .kpi-subtitle {
        color: #0D9488;
        font-size: 12px;
        font-weight: 500;
    }
    
    /* Clinical Disclaimer Box */
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 12px 16px;
        border-radius: 6px;
        color: #92400E;
        font-size: 12.5px;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    
    /* Content Card */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    .cluster-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    
    /* Button custom overrides */
    .stButton>button {
        background-color: #0F766E;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
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
        # Load default dataset
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

    # Generate cluster profiles
    profiles = generate_cluster_profiles(df_clustered, model.cluster_labels)
    st.session_state["profiles"] = profiles


init_session_state()

# ----------------- SIDEBAR NAVIGATION -----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 28px;">🫀</span>
        <div>
            <h2 style="font-size: 18px; margin: 0; color: #0F2942; font-weight: 700;">CardioInsight</h2>
            <p style="font-size: 11px; margin: 0; color: #64748B;">Patient Profiling & Analytics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Navigation")
    
    page = st.radio(
        "Select Analytical View:",
        [
            "1. Overview Dashboard",
            "2. Patient Explorer",
            "3. Patient Profile Analysis",
            "4. Cluster Insights",
            "5. Model Evaluation",
            "6. Dataset Management"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### Model Controls")
    
    # Active K slider
    eval_results = st.session_state.get("eval_results", {})
    available_ks = sorted(eval_results.keys()) if eval_results else [2, 3, 4, 5, 6, 7, 8]
    
    current_k = st.session_state.get("k_selected", 3)
    new_k = st.select_slider(
        "Active Clusters (K):",
        options=available_ks,
        value=current_k,
        help="Adjust the number of clusters to dynamically retrain K-Means."
    )

    if new_k != current_k:
        st.session_state["k_selected"] = new_k
        _train_pipeline(k_override=new_k)
        st.rerun()

    rec_k = st.session_state.get("recommended_k", 3)
    if new_k == rec_k:
        st.caption(f"⭐ **Current K ({new_k}) matches algorithm recommendation.**")
    else:
        st.caption(f"💡 Recommended K: **{rec_k}**")
        if st.button(f"Reset to Recommended (K={rec_k})", use_container_width=True):
            st.session_state["k_selected"] = rec_k
            _train_pipeline(k_override=rec_k)
            st.rerun()

    st.markdown("---")
    st.markdown("### Dataset Attribution")
    st.caption(f"**Cohort:** {st.session_state['report']['source_name']}")
    st.caption(f"**Records:** {len(st.session_state['df'])} observations")
    if st.session_state["report"].get("is_synthetic", False):
        st.warning("⚠️ Using synthetic demo cohort")
    st.caption("UCI Machine Learning Repository (Cleveland Clinic Foundation)")


# ----------------- GLOBAL HEADER & MEDICAL DISCLAIMER -----------------
st.markdown("""
<div class="clinical-header">
    <h1>CardioInsight — Cardiac Patient Profiling & Hospital Analytics</h1>
    <p>Data Warehousing & Mining (DWM) Academic Framework | Unsupervised Clinical Phenotyping via K-Means</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="disclaimer-banner">
    <strong>⚖️ Academic Research Notice & Clinical Safety Disclaimer:</strong> {MEDICAL_DISCLAIMER}
</div>
""", unsafe_allow_html=True)


# =====================================================================
# PAGE 1: OVERVIEW DASHBOARD
# =====================================================================
if page == "1. Overview Dashboard":
    st.subheader("Overview Dashboard")
    st.caption("Executive clinical cohort summary, clustering distribution, and multivariate visual analytics.")

    df = st.session_state["df"]
    df_clustered = st.session_state["df_clustered"]
    model = st.session_state["model"]
    report = st.session_state["report"]
    k_selected = st.session_state["k_selected"]
    profiles = st.session_state["profiles"]

    # KPI Summary Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Patients</div>
            <div class="kpi-value">{len(df_clustered)}</div>
            <div class="kpi-subtitle">Eligible records</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Discovered Phenotypes</div>
            <div class="kpi-value">{k_selected}</div>
            <div class="kpi-subtitle">Distinct clusters</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Selected K</div>
            <div class="kpi-value">{k_selected}</div>
            <div class="kpi-subtitle">Recommended: K={st.session_state.get('recommended_k', 3)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        sil_score = model.overall_silhouette
        sil_display = f"{sil_score:.3f}" if sil_score is not None else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Silhouette Score</div>
            <div class="kpi-value">{sil_display}</div>
            <div class="kpi-subtitle">Cluster cohesion index</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        completeness = report.get("completeness_pct", 100.0)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Data Completeness</div>
            <div class="kpi-value">{completeness}%</div>
            <div class="kpi-subtitle">{report.get('total_missing_cells', 0)} missing values imputed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Automated Executive Summary
    largest_cluster_id = max(profiles, key=lambda c: profiles[c]["record_count"])
    largest_p = profiles[largest_cluster_id]
    smallest_cluster_id = min(profiles, key=lambda c: profiles[c]["record_count"])
    smallest_p = profiles[smallest_cluster_id]

    st.markdown(f"""
    <div class="content-box" style="border-left: 4px solid #0F766E;">
        <h4 style="margin: 0 0 8px 0; color: #0F2942; font-size: 15px;">📊 Automated Cohort Profiling Insight</h4>
        <p style="margin: 0; font-size: 13.5px; color: #334155; line-height: 1.6;">
            The active K-Means model partitions the <strong>{len(df_clustered)}</strong> cardiology observations into <strong>{k_selected}</strong> distinct patient clusters. 
            The largest subgroup is <strong>Cluster {largest_cluster_id}</strong> comprising <strong>{largest_p['record_count']} patients ({largest_p['percentage']}%)</strong>, 
            while <strong>Cluster {smallest_cluster_id}</strong> represents the most focused subset with <strong>{smallest_p['record_count']} patients ({smallest_p['percentage']}%)</strong>. 
            Key multivariate differentiators observed across these cohorts include resting blood pressure (<code>trestbps</code>), serum cholesterol (<code>chol</code>), 
            and maximum exercise heart rate (<code>thalach</code>).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Charts Row 1: Cluster Distribution & Donut
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        fig_dist = plot_cluster_distribution(df_clustered)
        st.plotly_chart(fig_dist, use_container_width=True)
    with c2:
        fig_pie = plot_cluster_proportions_pie(df_clustered)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Charts Row 2: Age Distribution & Clinical Feature Comparison
    c3, c4 = st.columns([1.0, 1.2])
    with c3:
        fig_age = plot_age_distribution_by_cluster(df_clustered)
        st.plotly_chart(fig_age, use_container_width=True)
    with c4:
        fig_comp = plot_clinical_feature_comparison_bars(df_clustered)
        st.plotly_chart(fig_comp, use_container_width=True)

    # Charts Row 3: 2D PCA Visualization
    st.markdown("#### Dimensionality Reduction: 2D Principal Component Space")
    st.caption("PCA maps the multi-attribute clinical space into two principal orthogonal dimensions to visualize cluster separation and centroid locations.")
    
    explained_var = model.pca.explained_variance_ratio_ if model.pca else [0.0, 0.0]
    fig_pca = plot_pca_scatter(
        model.pca_coords,
        model.cluster_labels,
        pca_centroids=model.pca_centroids,
        explained_variance=explained_var
    )
    st.plotly_chart(fig_pca, use_container_width=True)


# =====================================================================
# PAGE 2: PATIENT EXPLORER
# =====================================================================
elif page == "2. Patient Explorer":
    st.subheader("Patient Record Explorer")
    st.caption("Search, filter, and inspect individual patient records with anonymized indices and cluster labels.")

    df_clustered = st.session_state["df_clustered"]
    clusters = sorted(df_clustered["cluster"].unique())

    # Filter Controls
    with st.expander("🔍 Search and Filter Options", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            search_query = st.text_input("Search Record ID (e.g. REC-015):", "").strip()

        with f_col2:
            cluster_filter = st.multiselect(
                "Filter by Cluster:",
                options=clusters,
                default=clusters,
                format_func=lambda c: f"Cluster {c}"
            )

        with f_col3:
            if "sex" in df_clustered.columns:
                sex_options = ["All", "Male (1)", "Female (0)"]
                sex_choice = st.selectbox("Biological Sex:", sex_options)
            else:
                sex_choice = "All"

        with f_col4:
            if "age" in df_clustered.columns:
                min_age = int(df_clustered["age"].min())
                max_age = int(df_clustered["age"].max())
                age_range = st.slider("Age Range (years):", min_age, max_age, (min_age, max_age))
            else:
                age_range = None

    # Apply Filters
    filtered_df = df_clustered.copy()

    if search_query:
        filtered_df = filtered_df[filtered_df["record_id"].astype(str).str.contains(search_query, case=False, na=False)]

    if cluster_filter:
        filtered_df = filtered_df[filtered_df["cluster"].isin(cluster_filter)]

    if sex_choice == "Male (1)":
        filtered_df = filtered_df[filtered_df["sex"] == 1.0]
    elif sex_choice == "Female (0)":
        filtered_df = filtered_df[filtered_df["sex"] == 0.0]

    if age_range and "age" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])
        ]

    st.markdown(f"**Showing {len(filtered_df)} of {len(df_clustered)} records**")

    # Display Data Table
    # Reorder columns to show record_id and cluster first
    cols_order = ["record_id", "cluster"] + [c for c in filtered_df.columns if c not in ["record_id", "cluster"]]
    st.dataframe(
        filtered_df[cols_order],
        use_container_width=True,
        hide_index=True,
        height=380
    )

    # Actions: Patient Detail Card and CSV Export
    action_col1, action_col2 = st.columns([1.5, 1.0])

    with action_col1:
        st.markdown("#### Patient Feature Deep-Dive")
        selected_record_id = st.selectbox(
            "Select a patient to inspect individual clinical parameters:",
            options=filtered_df["record_id"].tolist() if len(filtered_df) > 0 else []
        )

        if selected_record_id:
            record_row = filtered_df[filtered_df["record_id"] == selected_record_id].iloc[0]
            st.markdown(f"""
            <div class="content-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <h3 style="margin: 0; color: #0F2942;">Patient Record: {record_row['record_id']}</h3>
                    <span style="background: #0284C7; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600;">
                        Assigned Cluster {record_row['cluster']}
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 10px;">
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

    with action_col2:
        st.markdown("#### Data Export")
        st.write("Download filtered cohort results or the full clustered database with assigned phenotype labels.")

        filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Records (CSV)",
            data=filtered_csv,
            file_name="cardioinsight_filtered_patients.csv",
            mime="text/csv",
            use_container_width=True
        )

        full_csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Full Clustered Dataset (CSV)",
            data=full_csv,
            file_name="cardioinsight_full_clustered_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )


# =====================================================================
# PAGE 3: PATIENT PROFILE ANALYSIS
# =====================================================================
elif page == "3. Patient Profile Analysis":
    st.subheader("Patient Profile Analysis")
    st.caption("Multivariate similarity profiling against cluster centroids with out-of-distribution anomaly safeguards.")

    df_clustered = st.session_state["df_clustered"]
    preprocessor = st.session_state["preprocessor"]
    model = st.session_state["model"]
    profiles = st.session_state["profiles"]

    mode = st.radio(
        "Select Analytical Mode:",
        ["Mode A: Profile Existing Cohort Record", "Mode B: Profile New Hypothetical Patient"],
        horizontal=True
    )

    if mode == "Mode A: Profile Existing Cohort Record":
        st.markdown("#### Mode A: Profile an Existing Patient Record")
        patient_options = df_clustered["record_id"].tolist()
        selected_rec = st.selectbox("Select Patient Record:", patient_options)

        if selected_rec:
            record_row = df_clustered[df_clustered["record_id"] == selected_rec].iloc[0]
            patient_dict = record_row.to_dict()
            assigned_c = int(record_row["cluster"])
            cluster_profile = profiles[assigned_c]

            # Vector transform and similarity calculation
            x_scaled = preprocessor.transform(pd.DataFrame([patient_dict]))
            _, similarity_dict, min_dist, meta = model.predict_new_record(x_scaled)

            # Profile Summary Banner
            st.markdown(f"""
            <div class="content-box" style="border-left: 5px solid #0D9488;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 style="margin: 0; color: #0F2942;">Patient Profile: {selected_rec}</h3>
                    <span style="background: #0D9488; color: #FFFFFF; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                        Assigned Cluster {assigned_c}
                    </span>
                </div>
                <p style="margin: 0 0 10px 0; color: #1E293B; font-weight: 600; font-size: 15px;">
                    {cluster_profile['title']}
                </p>
                <div style="font-size: 13.5px; color: #334155; line-height: 1.5;">
                    Distance to Cluster Centroid: <strong>{min_dist:.2f}</strong> standard units in normalized feature space.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Comparative Charts
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("##### Relative Similarity to All Cohort Clusters")
                fig_sim = plot_similarity_to_all_clusters(similarity_dict)
                st.plotly_chart(fig_sim, use_container_width=True)

            with col_right:
                st.markdown("##### 2D PCA Space Placement")
                fig_pca_pt = plot_pca_scatter(
                    model.pca_coords,
                    model.cluster_labels,
                    pca_centroids=model.pca_centroids,
                    highlight_point=meta["pca_coord"],
                    highlight_label=f"Patient {selected_rec}"
                )
                st.plotly_chart(fig_pca_pt, use_container_width=True)

            # Multivariate Radar and Delta Bar Comparison
            st.markdown("---")
            st.markdown("#### Patient Feature Comparison vs Assigned Cluster Average")
            comp_result = compare_patient_to_cluster(patient_dict, cluster_profile)
            
            c_radar, c_delta = st.columns(2)
            with c_radar:
                fig_radar = plot_patient_vs_cluster_radar(patient_dict, cluster_profile, preprocessor.training_stats)
                st.plotly_chart(fig_radar, use_container_width=True)

            with c_delta:
                fig_delta = plot_patient_delta_bars(comp_result["comparison_items"])
                st.plotly_chart(fig_delta, use_container_width=True)

            # Tabular Comparison
            with st.expander("📋 Detailed Numerical Feature Breakdown vs Cluster Mean & Median", expanded=False):
                comp_df = pd.DataFrame(comp_result["comparison_items"])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

    else:
        # Mode B: New Hypothetical Patient
        st.markdown("#### Mode B: Profile a New Hypothetical Patient")
        st.info("💡 Enter physiological and diagnostic metrics below. The input will be validated, standardized using the training pipeline, and assigned to the nearest cluster centroid.")

        with st.form("hypothetical_patient_form"):
            st.markdown("##### 1. Demographic & Primary Vitals")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                age_val = st.number_input("Age (years):", min_value=18, max_value=100, value=56)
            with f2:
                sex_val = st.selectbox("Biological Sex:", options=[1.0, 0.0], format_func=lambda s: "Male (1)" if s == 1.0 else "Female (0)")
            with f3:
                trestbps_val = st.number_input("Resting Blood Pressure (mm Hg):", min_value=80, max_value=240, value=135)
            with f4:
                chol_val = st.number_input("Serum Cholesterol (mg/dl):", min_value=100, max_value=600, value=245)

            st.markdown("##### 2. Cardiac & Exercise Stress Test Parameters")
            f5, f6, f7, f8 = st.columns(4)
            with f5:
                cp_val = st.selectbox(
                    "Chest Pain Type:",
                    options=[1.0, 2.0, 3.0, 4.0],
                    format_func=lambda c: FEATURE_DEFINITIONS["cp"]["options"][c]
                )
            with f6:
                thalach_val = st.number_input("Max Heart Rate Achieved (bpm):", min_value=60, max_value=220, value=150)
            with f7:
                exang_val = st.selectbox("Exercise-Induced Angina:", options=[0.0, 1.0], format_func=lambda e: "No (0)" if e == 0.0 else "Yes (1)")
            with f8:
                oldpeak_val = st.number_input("ST Depression (oldpeak mm):", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

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
                    "Peak Exercise ST Slope:",
                    options=[1.0, 2.0, 3.0],
                    format_func=lambda s: FEATURE_DEFINITIONS["slope"]["options"][s]
                )
            with f12:
                ca_val = st.selectbox("Fluoroscopy Major Vessels (0-3):", options=[0.0, 1.0, 2.0, 3.0])

            thal_val = st.selectbox(
                "Thallium Scintigraphy Result:",
                options=[3.0, 6.0, 7.0],
                format_func=lambda t: FEATURE_DEFINITIONS["thal"]["options"][t]
            )

            submit_patient = st.form_submit_button("⚡ Evaluate Similarity Profile & Assign Cluster", use_container_width=True)

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
                st.warning("⚠️ **Out-of-Distribution Warning:** One or more entered features deviate significantly from the training cohort distribution. Nearest-centroid cluster assignment may exhibit reduced fidelity.")
                for w in ood_warnings:
                    st.caption(f"• **{w['display_name']}**: {w['message']}")

            # 2. Preprocess & Predict
            new_df = pd.DataFrame([new_record])
            x_scaled = preprocessor.transform(new_df)
            assigned_c, sim_dict, min_dist, meta = model.predict_new_record(x_scaled)
            cluster_profile = profiles[assigned_c]

            # Results Display
            st.markdown(f"""
            <div class="content-box" style="border-left: 5px solid #0F766E;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 style="margin: 0; color: #0F2942;">Hypothetical Patient Similarity Profile</h3>
                    <span style="background: #0F766E; color: #FFFFFF; padding: 6px 14px; border-radius: 20px; font-weight: 700;">
                        Assigned to Cluster {assigned_c}
                    </span>
                </div>
                <p style="margin: 0 0 10px 0; color: #0F2942; font-weight: 600; font-size: 15px;">
                    {cluster_profile['title']}
                </p>
                <p style="margin: 0; font-size: 13.5px; color: #334155; line-height: 1.5;">
                    The patient's normalized vector exhibits closest mathematical proximity to <strong>Cluster {assigned_c}</strong> centroid (Euclidean distance: {min_dist:.2f}). 
                    This profile indicates empirical similarity to the observed characteristics of Cluster {assigned_c} within the analyzed population.
                </p>
            </div>
            """, unsafe_allow_html=True)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("##### Cluster Similarity Breakdown")
                fig_sim = plot_similarity_to_all_clusters(sim_dict)
                st.plotly_chart(fig_sim, use_container_width=True)

            with col_p2:
                st.markdown("##### 2D PCA Space Projection")
                fig_pca_pt = plot_pca_scatter(
                    model.pca_coords,
                    model.cluster_labels,
                    pca_centroids=model.pca_centroids,
                    highlight_point=meta["pca_coord"],
                    highlight_label="New Patient"
                )
                st.plotly_chart(fig_pca_pt, use_container_width=True)

            # Radar & Delta Comparison
            comp_result = compare_patient_to_cluster(new_record, cluster_profile)
            c_rad, c_del = st.columns(2)
            with c_rad:
                fig_radar = plot_patient_vs_cluster_radar(new_record, cluster_profile, preprocessor.training_stats)
                st.plotly_chart(fig_radar, use_container_width=True)

            with c_del:
                fig_delta = plot_patient_delta_bars(comp_result["comparison_items"])
                st.plotly_chart(fig_delta, use_container_width=True)


# =====================================================================
# PAGE 4: CLUSTER INSIGHTS
# =====================================================================
elif page == "4. Cluster Insights":
    st.subheader("Cluster Insights & Phenotypic Profiling")
    st.caption("Human-interpretable statistical profiles, key distinguishing features, and comparative cohort analytics.")

    profiles = st.session_state["profiles"]
    df_clustered = st.session_state["df_clustered"]
    clusters = sorted(profiles.keys())

    # Heatmap of Standardized Features
    st.markdown("#### Standardized Cluster Comparison Heatmap")
    st.caption("Values represent standardized deviations (Z-scores) from the overall cohort average. Blue represents lower than cohort mean; red indicates higher than cohort mean.")
    fig_heat = plot_cluster_feature_heatmap(profiles)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Discovered Cluster Phenotype Profiles")

    # Detailed Cluster Profile Cards
    for c_id in clusters:
        p = profiles[c_id]
        with st.expander(f"📌 {p['title']} ({p['record_count']} patients, {p['percentage']}%)", expanded=(c_id == 0)):
            col_info, col_dist = st.columns([1.2, 1.0])

            with col_info:
                st.markdown(f"**Cohort Summary:**")
                st.markdown(p["narrative"])

                # Numerical statistics table
                st.markdown("**Numerical Feature Means vs Population:**")
                num_data = []
                for feat, s in p["numerical_stats"].items():
                    num_data.append({
                        "Feature": s["display_name"],
                        "Cluster Mean": f"{s['mean']} {s['unit']}",
                        "Cluster Median": f"{s['median']} {s['unit']}",
                        "Population Mean": f"{s['pop_mean']} {s['unit']}",
                        "Deviation (Z)": f"{s['z_diff']:+.2f} σ"
                    })
                st.dataframe(pd.DataFrame(num_data), use_container_width=True, hide_index=True)

            with col_dist:
                st.markdown("**Top Distinguishing Features:**")
                if p["distinguishing_features"]:
                    for df_item in p["distinguishing_features"]:
                        badge_color = "#E11D48" if df_item["direction"] == "higher" else "#0284C7"
                        st.markdown(f"""
                        <div style="background: #F1F5F9; border-left: 3px solid {badge_color}; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; font-size: 13px;">
                            <strong>{df_item['display_name']}</strong> ({df_item['direction'].upper()}): 
                            {df_item['summary']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("This cluster's values closely mirror the global population averages.")

                # Categorical distributions preview
                st.markdown("**Key Categorical Indicators:**")
                if "sex" in p["categorical_stats"]:
                    st.write("**Sex Breakdown:**", p["categorical_stats"]["sex"])
                if "cp" in p["categorical_stats"]:
                    st.write("**Chest Pain Types:**", p["categorical_stats"]["cp"])

    # Side-by-Side Comparison of Two Clusters
    st.markdown("---")
    st.markdown("#### Side-by-Side Cluster Comparison")
    st.caption("Select any two clusters to inspect feature differences and comparative gradients.")

    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        cluster_a = st.selectbox("Select Primary Cluster:", clusters, index=0)
    with comp_c2:
        cluster_b = st.selectbox("Select Comparison Cluster:", clusters, index=min(1, len(clusters)-1))

    if cluster_a != cluster_b:
        two_comp_df = compare_two_clusters(profiles[cluster_a], profiles[cluster_b])
        st.dataframe(two_comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Select two different clusters above to view comparative analysis.")


# =====================================================================
# PAGE 5: MODEL EVALUATION
# =====================================================================
elif page == "5. Model Evaluation":
    st.subheader("Model Evaluation & Clustering Metrics")
    st.caption("Academic Data Warehousing & Mining (DWM) diagnostics: Elbow analysis, Silhouette coefficients, and PCA variance.")

    eval_results = st.session_state["eval_results"]
    k_selected = st.session_state["k_selected"]
    rec_k = st.session_state["recommended_k"]
    rec_reason = st.session_state["recommendation_reason"]
    model = st.session_state["model"]

    # Evaluation Metric Highlights
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Selected K", k_selected)
    with m2:
        st.metric("Recommended K", rec_k)
    with m3:
        st.metric("Final Silhouette Score", f"{model.overall_silhouette:.4f}" if model.overall_silhouette else "N/A")
    with m4:
        inertia_val = eval_results.get(k_selected, {}).get("inertia", "N/A")
        st.metric("Final SSE (Inertia)", f"{inertia_val}")

    st.markdown(f"""
    <div class="content-box" style="border-left: 4px solid #0284C7;">
        <h4 style="margin: 0 0 6px 0; color: #0F2942; font-size: 15px;">📐 Selection Rationale</h4>
        <p style="margin: 0; font-size: 13.5px; color: #334155; line-height: 1.5;">{rec_reason}</p>
    </div>
    """, unsafe_allow_html=True)

    # Elbow and Silhouette Curves
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_elbow = plot_elbow_curve(eval_results, k_selected)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_e2:
        fig_sil = plot_silhouette_curve(eval_results, k_selected)
        st.plotly_chart(fig_sil, use_container_width=True)

    # Tabular Metrics Across K
    st.markdown("#### Candidate K Diagnostic Table")
    eval_table = []
    for k_val, meta in sorted(eval_results.items()):
        eval_table.append({
            "K (Clusters)": k_val,
            "Inertia (SSE)": meta["inertia"],
            "Silhouette Score": meta["silhouette_score"],
            "Smallest Cluster Size": meta["min_cluster_size"],
            "Cluster Size Distribution": str(meta["cluster_sizes"]),
            "Status": "⭐ Selected" if k_val == k_selected else ("💡 Recommended" if k_val == rec_k else "")
        })
    st.dataframe(pd.DataFrame(eval_table), use_container_width=True, hide_index=True)

    # PCA Variance Explanation
    st.markdown("#### Dimensionality Reduction Analytics (Principal Component Analysis)")
    if model.pca is not None:
        var_ratios = model.pca.explained_variance_ratio_
        pca_var_text = f"Principal Component 1 explains **{var_ratios[0]*100:.2f}%** of multivariate variance, and Principal Component 2 explains **{var_ratios[1]*100:.2f}%** (cumulative: **{(var_ratios[0]+var_ratios[1])*100:.2f}%**)."
    else:
        pca_var_text = "PCA model variance pending calculation."

    st.markdown(f"""
    <div class="content-box">
        <h5 style="margin: 0 0 6px 0; color: #0F2942;">Variance Explained by 2D PCA Space</h5>
        <p style="margin: 0 0 8px 0; font-size: 13.5px; color: #334155;">{pca_var_text}</p>
        <p style="margin: 0; font-size: 12.5px; color: #64748B;">
            <em>Note on DWM Methodology:</em> PCA is utilized here strictly as an unsupervised linear projection technique for 2D visualization of the high-dimensional clinical feature space. It does not reflect clinical risk or treatment outcome categories.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PAGE 6: DATASET MANAGEMENT
# =====================================================================
elif page == "6. Dataset Management":
    st.subheader("Dataset Management & Data Warehousing Inspection")
    st.caption("Data source provenance, column definitions, schema verification, data cleaning audit, and CSV file upload.")

    report = st.session_state["report"]
    df = st.session_state["df"]
    df_clustered = st.session_state["df_clustered"]

    # Dataset Provenance Card
    st.markdown(f"""
    <div class="content-box">
        <h4 style="margin: 0 0 6px 0; color: #0F2942;">Dataset Provenance & Attribution</h4>
        <p style="margin: 0 0 8px 0; font-size: 13.5px; color: #334155;"><strong>Source:</strong> {DATASET_METADATA['name']} ({DATASET_METADATA['year']})</p>
        <p style="margin: 0 0 8px 0; font-size: 13px; color: #475569;"><strong>Principal Investigators:</strong> {DATASET_METADATA['creators']}</p>
        <p style="margin: 0 0 8px 0; font-size: 13px; color: #475569;"><strong>Citation:</strong> <em>{DATASET_METADATA['citation']}</em></p>
        <p style="margin: 0; font-size: 12.5px; color: #64748B;"><strong>Licensing:</strong> {DATASET_METADATA['license']} | <a href="{DATASET_METADATA['source_url']}" target="_blank">UCI Archive Page</a></p>
    </div>
    """, unsafe_allow_html=True)

    # Data Quality Metrics
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Total Observations", report["raw_rows"])
    with q2:
        st.metric("Total Attributes", report["raw_cols"])
    with q3:
        st.metric("Completeness Ratio", f"{report['completeness_pct']}%")
    with q4:
        st.metric("Duplicate Rows", report["duplicate_rows"])

    st.markdown("---")

    # CSV Upload Workflow
    st.markdown("#### Upload Custom Cardiac Dataset (CSV)")
    st.write("Upload a custom cardiovascular dataset. The application will validate schema compatibility, impute missing values, and retrain the K-Means pipeline.")

    uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
    if uploaded_file is not None:
        try:
            new_df, new_report = load_dataset(uploaded_file)
            if new_report["is_valid"]:
                st.success(f"✅ Successfully validated **{new_report['source_name']}** ({new_report['raw_rows']} rows, {len(new_report['present_clustering_features'])} compatible features).")
                if st.button("Apply and Retrain with Uploaded Dataset", use_container_width=True):
                    st.session_state["df"] = new_df
                    st.session_state["report"] = new_report
                    _train_pipeline()
                    st.rerun()
            else:
                st.error("❌ Uploaded CSV lacks required clinical features (must contain at least 'age' or 'trestbps' and 4 compatible cardiac attributes).")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")

    # Reset or Load Synthetic Demo Data
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        if st.button("🔄 Reset to Default UCI Cleveland Dataset", use_container_width=True):
            df, report = load_dataset()
            st.session_state["df"] = df
            st.session_state["report"] = report
            _train_pipeline()
            st.rerun()

    with col_demo2:
        if st.button("🧪 Generate Synthetic Demo Cohort", use_container_width=True):
            df_syn, report_syn = generate_synthetic_demo_dataset()
            st.session_state["df"] = df_syn
            st.session_state["report"] = report_syn
            _train_pipeline()
            st.rerun()

    # Column Schema and Feature Dictionary
    st.markdown("---")
    st.markdown("#### Clinical Feature Dictionary & Missing Value Audit")
    
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
            "Status in Dataset": present,
            "Missing Values": f"{miss_count} ({miss_pct:.1f}%)"
        })

    st.dataframe(pd.DataFrame(dict_rows), use_container_width=True, hide_index=True)

    # Descriptive Statistics Preview
    st.markdown("#### Descriptive Statistics for Numerical Features")
    if report["desc_stats"]:
        st.dataframe(pd.DataFrame(report["desc_stats"]).T, use_container_width=True)

    # Raw Data Preview
    st.markdown("#### Raw Dataset Sample (First 10 Records)")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # Download button for clustered data
    full_csv = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Clustered Dataset (CSV)",
        data=full_csv,
        file_name="cardioinsight_clustered_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )
